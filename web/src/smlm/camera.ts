// Camera calibration from the SMLM movie itself: a port of torch_psf.camera.photon_transfer (offset and gain from
// background pixels). Step by step as the Python code:
//   blocks of `block` consecutive frames, the starts f0 + block i for i in spread_order(nstart)[:nblocks] (sorted);
//   per block: detect_spots (ADU frames, `threshold`, no border / isolation; the GPU Detector with the whole block as
//   one chunk, so the noise is that of the block) -> occupancy dilated by a (2R + 1)^2 max pool -> a sample is kept if
//   no spot is within R px in this frame or the previous one -> 4 rounds of 4-sigma clipping (mean / sd over the kept
//   samples) -> mean = the 4th-round mean; variance = var(x[t+1] - x[t]) / 2 over the pairs with both samples kept
//   (unbiased, n - 1) -> pixels with more than max(T // 4, 10) pairs -> pixels below the 80th percentile of the mean
//   -> nbins bins between the 1 % and 99 % quantiles (numpy linear interpolation) -> median mean and median variance
//   of the bins with >= 3 pixels -> np.polyfit degree 1: var = k (mean - offset) (offset_adu not given: offset =
//   -c / k, intercept 0) or var = k (mean - offset_adu) + c (given: intercept c clipped at 0);
//   over the blocks: medians of slope / offset / intercept; gain = excess_noise / k, read noise = sqrt(c0) gain.
// Numerics as the Python code (torch float32 per-pixel sums; here on the GPU, exact for integer ADU, i.e. as float64
// sums rounded to float32, see BlockStats; numpy 1.x for the rest, on the CPU: the quantiles in float64 from float32
// values, compared with the float32 means after rounding to float32 (numpy 1.x value-based casting), medians of
// float32 values in float32, polyfit in float64).
// The one deviation: a block is held in memory as float32 frames and on the GPU (the detector's input and filtered
// copies), so when block x H x W exceeds MAX_BLOCK_VALUES (32 Mi values = 128 MiB, the default WebGPU storage
// binding limit) or the device limits, the block length is reduced (block_used; 128 frames for 512 x 512). The result
// is then that of torch_psf photon_transfer(..., block=block_used). The 128 x 128 demo keeps block = 400.
import { Detector } from '../gpu/detect';
import occSrc from '../gpu/kernels/camera_occ.wgsl?raw';
import passSrc from '../gpu/kernels/camera_pass.wgsl?raw';
import { StopToken, Eta, Progress } from '../core/task';
import { openSource, MovieSource } from './movie';
import { spreadOrder } from './estimate';
import type { CameraOpts } from '../ui/protocol';

export const MAX_BLOCK_VALUES = 32 * 2 ** 20;

export interface PhotonTransferBlock {
  frames: [number, number]; slope: number; offset_adu: number; intercept_adu2: number; median_mean_adu: number;
  kept_fraction: number; n_px: number;
  n_spots: number;                  // detections in the block (web only, for diagnostics)
}

/** Camera fields (torch_psf Camera) + the report of photon_transfer + what the web version adds (block_used ...). */
export interface CameraCalibration {
  gain_e_per_adu: number; offset_adu: number; readnoise_e: number; excess_noise: number; type: 'scmos' | 'emccd';
  slope_adu: number; poisson_equivalent_gain: number; per_block: PhotonTransferBlock[]; slope_spread: number;
  block_used: number; block_requested: number; frames: [number, number]; partial: boolean; seconds: number;
}

const f32 = Math.fround;

/** numpy.quantile / percentile, method 'linear' (numpy 1.x _quantile / _lerp) of sorted float32 values: the
 *  difference of the neighbours in float32, the interpolation in float64. */
export function quantileSorted(s: Float32Array, q: number): number {
  const n = s.length, v = (n - 1) * q;
  if (v >= n - 1) return s[n - 1];
  if (v < 0) return s[0];
  const i = Math.floor(v), t = v - i, a = s[i], b = s[i + 1];
  const d = f32(b - a);
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

/** numpy.median of float32 values (the float32 mean of the two middle values for an even count). */
export function medianF32(a: Float32Array): number { return sortedMedianF32(Float32Array.from(a).sort()); }
function sortedMedianF32(s: Float32Array): number {
  const n = s.length;
  return n % 2 ? s[(n - 1) >> 1] : f32(f32(s[n / 2 - 1] + s[n / 2]) / 2);
}

/** numpy.median of float64 values. */
function median64(a: number[]): number {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** numpy.linspace(a, b, num) (i * step + start, the last = stop). */
function linspace(a: number, b: number, num: number): number[] {
  const step = (b - a) / (num - 1);
  return Array.from({ length: num }, (_, i) => (i === num - 1 ? b : i * step + a));
}

/** numpy.polyfit(x, y, 1) -> [k, c] (least squares in float64; the centred normal equations: the same solution as
 *  numpy's scaled lstsq to float64 rounding). */
function polyfit1(x: ArrayLike<number>, y: ArrayLike<number>): [number, number] {
  const n = x.length;
  let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxx = 0, sxy = 0; for (let i = 0; i < n; i++) { const dx = x[i] - mx; sxx += dx * dx; sxy += dx * (y[i] - my); }
  const k = sxy / sxx;
  return [k, my - k * mx];
}

const SU = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

/** The per-pixel statistics of one block on the GPU (the steps of photon_transfer between detect_spots and the pixel
 *  selection): occupancy from the detections dilated by the (2R + 1)^2 window (gpu/kernels/camera_occ.wgsl), then 5
 *  passes over the frames, one invocation per pixel (gpu/kernels/camera_pass.wgsl): count / sum / sum of squares of
 *  the kept samples of clip rounds 0..3, then of the kept frame differences. Between the passes the CPU forms mu and
 *  4 sd of the round from the sums (float64, rounded to float32 where torch stores float32: WGSL division and sqrt
 *  are not correctly rounded) and uploads them for the clip test of the later passes (float32 on the GPU, as torch).
 *  The sums are exact for integer ADU (compensated float32 sums of exact terms, centred on the first frame): the same
 *  numbers as float64 sums. Returns the 4th-round mean, the difference variance, the pair count per pixel and the
 *  kept fraction of the pairs. */
export class BlockStats {
  private dev: GPUDevice; private H: number; private W: number; private HW: number;
  private pOcc: GPUComputePipeline; private pSum: GPUComputePipeline;
  private uOcc: GPUBuffer; private uSum: GPUBuffer; private occ: GPUBuffer; private clip: GPUBuffer; private acc: GPUBuffer; private rb: GPUBuffer;
  private det: GPUBuffer | null = null; private detCap = 0;

  constructor(dev: GPUDevice, H: number, W: number, Tmax: number) {
    this.dev = dev; this.H = H; this.W = W; this.HW = H * W;
    const HW = this.HW;
    this.pOcc = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: occSrc }), entryPoint: 'occupancy' } });
    this.pSum = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: passSrc }), entryPoint: 'sums' } });
    this.uOcc = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | CD });
    this.uSum = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | CD });
    this.occ = dev.createBuffer({ size: Math.ceil(Tmax / 32) * HW * 4, usage: SU | CD });
    this.clip = dev.createBuffer({ size: 8 * HW * 4, usage: SU | CD });
    this.acc = dev.createBuffer({ size: 5 * HW * 4, usage: SU | CS });
    this.rb = dev.createBuffer({ size: 5 * HW * 4, usage: GPUBufferUsage.MAP_READ | CD });
  }

  destroy() { for (const b of [this.uOcc, this.uSum, this.occ, this.clip, this.acc, this.rb, this.det]) b?.destroy(); }

  private groups(n: number, wg: number): [number, number] { const g = Math.max(1, Math.ceil(n / wg)), gx = Math.min(g, 65535); return [gx, Math.ceil(g / gx)]; }

  /** x: the block [T, H, W] f32 on the GPU (e.g. Detector.input after detect()) and its first frame on the CPU. */
  async run(x: GPUBuffer, x0: Float32Array, T: number, d: { frame: Int32Array; y: Int32Array; x: Int32Array }, R: number) {
    const { dev, H, W, HW } = this;
    const nd = d.frame.length;
    if (!this.det || nd > this.detCap) {
      this.det?.destroy();
      this.detCap = Math.max(nd, 1024);
      this.det = dev.createBuffer({ size: this.detCap * 12, usage: SU | CD });
    }
    const dd = new Uint32Array(3 * nd);
    for (let i = 0; i < nd; i++) { dd[3 * i] = d.frame[i]; dd[3 * i + 1] = d.y[i]; dd[3 * i + 2] = d.x[i]; }
    if (nd) dev.queue.writeBuffer(this.det, 0, dd);
    dev.queue.writeBuffer(this.uOcc, 0, new Uint32Array([HW, H, W, R, nd, 0, 0, 0]));
    let enc = dev.createCommandEncoder();
    enc.clearBuffer(this.occ, 0, Math.ceil(T / 32) * HW * 4);
    if (nd) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pOcc);
      pass.setBindGroup(0, dev.createBindGroup({ layout: this.pOcc.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uOcc } }, { binding: 1, resource: { buffer: this.det, size: nd * 12 } },
        { binding: 2, resource: { buffer: this.occ } }] }));
      pass.dispatchWorkgroups(...this.groups(nd, 64));
      pass.end();
    }
    dev.queue.submit([enc.finish()]);
    const bg = dev.createBindGroup({ layout: this.pSum.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uSum } }, { binding: 1, resource: { buffer: x, size: T * HW * 4 } },
      { binding: 2, resource: { buffer: this.occ } }, { binding: 3, resource: { buffer: this.clip } }, { binding: 4, resource: { buffer: this.acc } }] });
    const mu = new Float32Array(HW), lim = new Float32Array(HW), v = new Float32Array(HW), n = new Int32Array(HW);
    let kept = 0;
    for (let round = 0; round < 5; round++) {
      dev.queue.writeBuffer(this.uSum, 0, new Uint32Array([T, HW, round, 0]));
      enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pSum);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(...this.groups(HW, 256));
      pass.end();
      enc.copyBufferToBuffer(this.acc, 0, this.rb, 0, 5 * HW * 4);
      dev.queue.submit([enc.finish()]);
      await this.rb.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(this.rb.getMappedRange());
      if (round < 4) {
        // mu = sum(x keep) / n, sd = sqrt(sum((x - mu)^2 keep) / n) (n clamped at 1), from the sums of y = x - x0:
        // sum x = n x0 + S1, sum (x - mu)^2 = S2 - 2 c S1 + n c^2 with c = mu - x0
        for (let p = 0; p < HW; p++) {
          const cnt = a[p], c1 = Math.max(cnt, 1), S1 = a[HW + p] + a[2 * HW + p], S2 = a[3 * HW + p] + a[4 * HW + p];
          const u = f32(f32(cnt * x0[p] + S1) / c1), c = u - x0[p];
          const ss = Math.max(S2 - 2 * c * S1 + cnt * c * c, 0);
          mu[p] = u; lim[p] = f32(4 * f32(Math.sqrt(f32(f32(ss) / c1))));
        }
        this.rb.unmap();
        dev.queue.writeBuffer(this.clip, round * HW * 4, mu);
        dev.queue.writeBuffer(this.clip, (4 + round) * HW * 4, lim);
      } else {
        // var = sum (d - dm)^2 / max(n - 1, 1) / 2 over the kept pairs, dm = sum d / max(n, 1)
        for (let p = 0; p < HW; p++) {
          const cnt = a[p], S1 = a[HW + p] + a[2 * HW + p], S2 = a[3 * HW + p] + a[4 * HW + p];
          const dm = f32(f32(S1) / Math.max(cnt, 1));
          const ss = Math.max(S2 - 2 * dm * S1 + cnt * dm * dm, 0);
          v[p] = f32(f32(f32(ss) / Math.max(cnt - 1, 1)) / 2);
          n[p] = cnt; kept += cnt;
        }
        this.rb.unmap();
      }
    }
    return { mu, var: v, n, kept_fraction: f32(kept / Math.max(1, (T - 1) * HW)) };
  }
}

/** The line fit of one block from the per-pixel mean / variance of the selected pixels (photon_transfer after `ok`).
 *  null when fewer than 4 nbins pixels (the block is skipped). */
export function fitBlock(mb: Float32Array, vb: Float32Array, nbins: number, offsetAdu: number | null) {
  if (mb.length < 4 * nbins) return null;
  const sorted = Float32Array.from(mb).sort();
  const p80 = f32(quantileSorted(sorted, 0.8));               // np.percentile(mb, 80), compared in float32
  let nlow = 0; while (nlow < sorted.length && sorted[nlow] < p80) nlow++;
  const low = sorted.subarray(0, nlow);                       // mb[mb < p80], sorted
  const e = linspace(0.01, 0.99, nbins + 1).map((q) => f32(quantileSorted(low, q)));
  // bin i: e[i] <= mb < e[i + 1] (the edges are non-decreasing, so a pixel is in at most one bin: the last i with
  // e[i] <= mb, if mb < e[i + 1])
  const bm: number[][] = Array.from({ length: nbins }, () => []), bv: number[][] = Array.from({ length: nbins }, () => []);
  for (let p = 0; p < mb.length; p++) {
    const m = mb[p];
    if (!(m >= e[0]) || !(m < e[nbins])) continue;
    let lo = 0, hi = nbins;                                   // e[lo] <= m < e[hi]
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (e[mid] <= m) lo = mid; else hi = mid; }
    bm[lo].push(m); bv[lo].push(vb[p]);
  }
  const xm: number[] = [], xv: number[] = [];
  for (let i = 0; i < nbins; i++) if (bm[i].length >= 3) { xm.push(medianF32(Float32Array.from(bm[i]))); xv.push(medianF32(Float32Array.from(bv[i]))); }
  const median_mean = sortedMedianF32(sorted);
  if (offsetAdu === null) {
    const [k, c] = polyfit1(xm, xv);
    return { k, off: -c / k, c0: 0, nbins_used: xm.length, median_mean };
  }
  const o32 = f32(offsetAdu);
  const [k, c] = polyfit1(xm.map((m) => f32(m - o32)), xv);
  return { k, off: offsetAdu, c0: Math.max(c, 0), nbins_used: xm.length, median_mean };
}

/** torch_psf.camera.photon_transfer on a movie (ADU). Options as the Python keywords (nblocks 8, block 400,
 *  mask_radius 4, threshold 4, offset_adu None, excess_noise 1, nbins 24, frames = the whole movie). "Enough" stops
 *  after the current block and uses the blocks so far (partial). */
export async function photonTransfer(device: GPUDevice, source: File[] | MovieSource, o: CameraOpts = {}, tok: StopToken = new StopToken(),
  progress: (p: Progress) => void = () => {}): Promise<CameraCalibration> {
  const t0 = performance.now();
  const mv = await openSource(source, progress);
  const nblocks = o.nblocks ?? 8, R = Math.trunc(o.maskRadius ?? 4), threshold = o.threshold ?? 4.0, nbins = o.nbins ?? 24;
  const F2 = o.excessNoise ?? 1.0, offsetAdu = o.offsetAdu ?? null;
  const [f0, f1] = o.frames ?? [0, mv.n];
  const HW = mv.H * mv.W;
  const requested = Math.min(o.block ?? 400, f1 - f0);
  const lim = Math.min(MAX_BLOCK_VALUES, Math.floor(device.limits.maxStorageBufferBindingSize / 4), Math.floor(device.limits.maxBufferSize / 4));
  const block = Math.max(1, Math.min(requested, Math.floor(lim / HW)));
  const nstart = Math.max(Math.floor((f1 - f0) / block), 1);
  const starts = spreadOrder(nstart).slice(0, nblocks).map((i) => f0 + block * i).sort((a, b) => a - b);
  // peaks cannot be closer than nms // 2 + 1 = 5 px (Chebyshev) without exact ties: room for every possible peak
  const det = new Detector(device, mv.H, mv.W, block, { threshold, border: 0, isolation: null, maxPeaks: Math.max(4096, block * Math.ceil(mv.H / 5) * Math.ceil(mv.W / 5)) });
  const stats = new BlockStats(device, mv.H, mv.W, block);
  const per: PhotonTransferBlock[] = [];
  const eta = new Eta();
  let partial = false;
  try {
    for (let b = 0; b < starts.length; b++) {
      tok.check();
      if (tok.stopEarly && per.length) { partial = true; break; }
      const b0 = starts[b];
      const x = await mv.read(b0, b0 + block);
      const T = x.length / HW;
      const d = await det.detect(x, T, 0);
      const st = await stats.run(det.input, x.subarray(0, HW), T, d, R);
      const need = Math.max(Math.floor(T / 4), 10);
      let nok = 0; for (let p = 0; p < HW; p++) if (st.n[p] > need) nok++;
      const mb = new Float32Array(nok), vb = new Float32Array(nok);
      for (let p = 0, j = 0; p < HW; p++) if (st.n[p] > need) { mb[j] = st.mu[p]; vb[j] = st.var[p]; j++; }
      const fb = fitBlock(mb, vb, nbins, offsetAdu);
      if (fb) per.push({ frames: [b0, b0 + block], slope: fb.k, offset_adu: fb.off, intercept_adu2: fb.c0, median_mean_adu: fb.median_mean,
        kept_fraction: st.kept_fraction, n_px: nok, n_spots: d.frame.length });
      progress({ stage: 'camera', done: b + 1, total: starts.length, info: { blocks: per.length, ...(fb ? { slope: fb.k, offset_adu: fb.off } : {}) }, ...eta.update(b + 1, starts.length) });
    }
  } finally { det.destroy(); stats.destroy(); }
  if (!per.length) throw new Error('photon_transfer: not enough background pixels');
  const k = median64(per.map((p) => p.slope));
  const off = median64(per.map((p) => p.offset_adu));
  const c0 = median64(per.map((p) => p.intercept_adu2));
  const g = F2 / k;
  let spread = 0;
  if (per.length > 1) {
    const s = per.map((p) => p.slope), m = s.reduce((a, v) => a + v, 0) / s.length;
    spread = Math.sqrt(s.reduce((a, v) => a + (v - m) ** 2, 0) / s.length) / k;
  }
  return { gain_e_per_adu: g, offset_adu: off, readnoise_e: Math.sqrt(c0) * g, excess_noise: F2, type: F2 > 1 ? 'emccd' : 'scmos',
    slope_adu: k, poisson_equivalent_gain: 1 / k, per_block: per, slope_spread: spread,
    block_used: block, block_requested: requested, frames: [f0, f1], partial, seconds: (performance.now() - t0) / 1000 };
}
