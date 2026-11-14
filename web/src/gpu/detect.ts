// Spot detection on the GPU: a port of torch_psf.detect.detect_spots and torch_psf.spots.extract_rois.
// Same math and defaults: band-pass = blur(sigma0) - blur(sigma1) (Gaussian kernels truncated at 4 sigma, zero
// padding, y then x), noise = 1.4826 x median |f - median f| over the whole chunk (noise 'mad'; torch.median = the
// lower median), peaks = f == max_pool(f, nms) and f > threshold x noise, border and isolation rules, score = f / noise.
// Detections are returned in torch.nonzero order (frame, y, x).
import src from './kernels/detect.wgsl?raw';

const SU = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

export interface DetectOptions {
  threshold?: number;          // x noise (default 6)
  sigma?: [number, number];    // band-pass (default 2, 6 px)
  nms?: number;                // max-pool window (default 9)
  border?: number;             // px (extract_spots default: (P - 1) / 2 + 2)
  isolation?: number | null;   // px (default off)
}

export interface Detections {
  frame: Int32Array; y: Int32Array; x: Int32Array; score: Float32Array;
  noise: number;               // the noise of the chunk (filtered-image units)
  median: number;              // median of the filtered chunk
}

/** torch_psf.detect.gaussian_kernel (float32). */
export function gaussianKernel(sigma: number, truncate = 4.0): Float32Array {
  const r = Math.trunc(truncate * sigma);
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.fround(Math.exp(Math.fround(-(i * i) / Math.fround(2 * sigma * sigma)))); k[i + r] = v; s = Math.fround(s + v); }
  for (let i = 0; i < k.length; i++) k[i] = Math.fround(k[i] / s);
  return k;
}

function keyToFloat(key: number): number {
  const b = (key & 0x80000000) !== 0 ? (key & 0x7fffffff) >>> 0 : (~key) >>> 0;
  const dv = new DataView(new ArrayBuffer(4));
  dv.setUint32(0, b, true);
  return dv.getFloat32(0, true);
}

export class Detector {
  private dev: GPUDevice;
  private layout: GPUBindGroupLayout;
  private pipes: Record<string, GPUComputePipeline> = {};
  private H: number; private W: number; private Tmax: number; private Tb: number;
  private raw: GPUBuffer; private F: GPUBuffer; private A: GPUBuffer; private B: GPUBuffer; private C: GPUBuffer;
  private map: GPUBuffer; private k0: GPUBuffer; private k1: GPUBuffer; private hist: GPUBuffer; private peaks: GPUBuffer;
  private dummy: GPUBuffer; private dummyU: GPUBuffer;
  private kl0: number; private kl1: number;
  readonly maxPeaks: number;
  private opt: Required<DetectOptions>;

  /** Frames of H x W, chunks of up to Tmax frames (the noise is estimated per chunk, as in torch_psf). */
  constructor(dev: GPUDevice, H: number, W: number, Tmax: number, opt: DetectOptions = {}) {
    this.dev = dev; this.H = H; this.W = W; this.Tmax = Tmax;
    this.opt = { threshold: opt.threshold ?? 6, sigma: opt.sigma ?? [2, 6], nms: opt.nms ?? 9, border: opt.border ?? 0, isolation: opt.isolation ?? null } as Required<DetectOptions>;
    const fb = H * W * 4;
    this.Tb = Math.max(1, Math.min(Tmax, Math.floor((64 * 2 ** 20) / fb)));
    this.maxPeaks = Math.max(4096, Tmax * 64);
    const mk = (size: number, usage = SU | CD | CS) => dev.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    this.raw = mk(Tmax * fb); this.F = mk(Tmax * fb);
    this.A = mk(this.Tb * fb); this.B = mk(this.Tb * fb); this.C = mk(this.Tb * fb); this.map = mk(this.Tb * fb);
    const g0 = gaussianKernel(this.opt.sigma[0]), g1 = gaussianKernel(this.opt.sigma[1]);
    this.kl0 = g0.length; this.kl1 = g1.length;
    this.k0 = mk(g0.byteLength); this.k1 = mk(g1.byteLength);
    dev.queue.writeBuffer(this.k0, 0, g0); dev.queue.writeBuffer(this.k1, 0, g1);
    this.hist = mk(65536 * 4); this.peaks = mk(this.maxPeaks * 16);
    this.dummy = mk(16); this.dummyU = mk(16);
    const ro = { type: 'read-only-storage' as const }, rw = { type: 'storage' as const };
    this.layout = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: ro },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: rw },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: ro },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: rw },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: rw },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: ro },
    ] });
    const module = dev.createShaderModule({ code: src });
    const pl = dev.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    for (const e of ['blur_y', 'blur_x', 'dog', 'hist_hi', 'hist_lo', 'peak_map', 'peaks_final'])
      this.pipes[e] = dev.createComputePipeline({ layout: pl, compute: { module, entryPoint: e } });
  }

  destroy() {
    for (const b of this.ubs) b.destroy();
    for (const b of [this.raw, this.F, this.A, this.B, this.C, this.map, this.k0, this.k1, this.hist, this.peaks, this.dummy, this.dummyU]) b.destroy();
  }

  private pass(enc: GPUCommandEncoder, entry: string, T: number, u: Partial<Record<string, number>>,
    b: { src?: GPUBuffer; dst?: GPUBuffer; kern?: GPUBuffer; src2?: GPUBuffer; hist?: boolean; peaks?: boolean }) {
    const dv = new DataView(new ArrayBuffer(80));
    const fields = ['T', 'H', 'W', 'klen', 'mode', 'prefix', 'kth', 'nms', 'thr', 'med', 'border', 'iso', 'frame0', 'maxpk', 'soff', 'doff', 's2off'];
    const vals: Record<string, number> = { T, H: this.H, W: this.W, nms: this.opt.nms, border: this.opt.border, iso: this.opt.isolation ?? 0, maxpk: this.maxPeaks, ...u };
    fields.forEach((f, i) => {
      const v = vals[f] ?? 0;
      if (f === 'thr' || f === 'med') dv.setFloat32(4 * i, v, true); else dv.setUint32(4 * i, v >>> 0, true);
    });
    const ub = this.dev.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | CD });
    this.dev.queue.writeBuffer(ub, 0, dv.buffer);
    this.ubs.push(ub);
    const bg = this.dev.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: ub } },
      { binding: 1, resource: { buffer: b.src ?? this.dummy } },
      { binding: 2, resource: { buffer: b.dst ?? this.dummyU } },
      { binding: 3, resource: { buffer: b.kern ?? this.k0 } },
      { binding: 4, resource: { buffer: b.hist ? this.hist : this.dummyHist() } },
      { binding: 5, resource: { buffer: b.peaks ? this.peaks : this.dummyPk() } },
      { binding: 6, resource: { buffer: b.src2 ?? this.dummy } },
    ] });
    const n = Math.ceil((T * this.H * this.W) / 256);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipes[entry]);
    pass.setBindGroup(0, bg);
    const gx = Math.min(n, 65535);
    pass.dispatchWorkgroups(gx, Math.ceil(n / gx));
    pass.end();
  }
  private ubs: GPUBuffer[] = [];
  private _dh: GPUBuffer | null = null; private _dp: GPUBuffer | null = null;
  private dummyHist() { return (this._dh ??= this.dev.createBuffer({ size: 16, usage: SU })); }
  private dummyPk() { return (this._dp ??= this.dev.createBuffer({ size: 16, usage: SU })); }

  private async readU32(buf: GPUBuffer, n: number): Promise<Uint32Array> {
    const rb = this.dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return out;
  }

  /** Exact k-th smallest (0-based) of f (mode 0) or |f - med| (mode 1) over the T frames of the chunk. */
  private async select(T: number, kth: number, mode: number, med: number): Promise<number> {
    const nb = 65536 * 4;
    let enc = this.dev.createCommandEncoder();
    enc.clearBuffer(this.hist, 0, nb);
    this.pass(enc, 'hist_hi', T, { mode, med, kth }, { src: this.F, hist: true });
    this.dev.queue.submit([enc.finish()]);
    let h = await this.readU32(this.hist, 65536);
    let c = 0, hi = 0;
    for (; hi < 65536; hi++) { if (c + h[hi] > kth) break; c += h[hi]; }
    const rest = kth - c;
    enc = this.dev.createCommandEncoder();
    enc.clearBuffer(this.hist, 0, nb);
    this.pass(enc, 'hist_lo', T, { mode, med, prefix: hi }, { src: this.F, hist: true });
    this.dev.queue.submit([enc.finish()]);
    h = await this.readU32(this.hist, 65536);
    let c2 = 0, lo = 0;
    for (; lo < 65536; lo++) { if (c2 + h[lo] > rest) break; c2 += h[lo]; }
    return keyToFloat(((hi << 16) | lo) >>> 0);
  }

  /** Detect spots on a chunk of T frames (photons [T, H, W]); frame0 = number of the first frame (for the output). */
  async detect(frames: Float32Array, T: number, frame0: number): Promise<Detections> {
    const { H, W } = this, fpx = H * W;
    if (T > this.Tmax) throw new Error('chunk larger than the detector capacity');
    this.dev.queue.writeBuffer(this.raw, 0, frames.buffer, frames.byteOffset, T * fpx * 4);
    // band-pass, in sub-batches of Tb frames, into F
    const enc = this.dev.createCommandEncoder();
    for (let s = 0; s < T; s += this.Tb) {
      const Tb = Math.min(this.Tb, T - s), off = s * fpx;
      this.pass(enc, 'blur_y', Tb, { klen: this.kl0, soff: off }, { src: this.raw, dst: this.A, kern: this.k0 });
      this.pass(enc, 'blur_x', Tb, { klen: this.kl0 }, { src: this.A, dst: this.B, kern: this.k0 });
      this.pass(enc, 'blur_y', Tb, { klen: this.kl1, soff: off }, { src: this.raw, dst: this.A, kern: this.k1 });
      this.pass(enc, 'blur_x', Tb, { klen: this.kl1 }, { src: this.A, dst: this.C, kern: this.k1 });
      this.pass(enc, 'dog', Tb, { doff: off }, { src: this.B, src2: this.C, dst: this.F });
    }
    this.dev.queue.submit([enc.finish()]);
    // noise 'mad': 1.4826 x median |f - median f| over the chunk (lower median, as torch.median)
    const n = T * fpx, kth = (n - 1) >> 1;
    const med = await this.select(T, kth, 0, 0);
    const mad = await this.select(T, kth, 1, med);
    const sig = Math.fround(Math.fround(1.4826) * mad);
    const thr = Math.fround(Math.fround(this.opt.threshold) * sig);
    // peaks
    const enc2 = this.dev.createCommandEncoder();
    enc2.clearBuffer(this.hist, 0, 16);
    for (let s = 0; s < T; s += this.Tb) {
      const Tb = Math.min(this.Tb, T - s), off = s * fpx;
      this.pass(enc2, 'peak_map', Tb, { thr, soff: off }, { src: this.F, dst: this.map });
      this.pass(enc2, 'peaks_final', Tb, { soff: off, frame0: frame0 + s }, { src: this.F, src2: this.map, hist: true, peaks: true });
    }
    this.dev.queue.submit([enc2.finish()]);
    const cnt = (await this.readU32(this.hist, 1))[0];
    if (cnt > this.maxPeaks) throw new Error(`more than ${this.maxPeaks} peaks in one chunk; raise the threshold`);
    const pk = cnt ? await this.readU32(this.peaks, cnt * 4) : new Uint32Array(0);
    const idx = Array.from({ length: cnt }, (_, i) => i);
    idx.sort((a, b) => (pk[4 * a] - pk[4 * b]) || (pk[4 * a + 1] - pk[4 * b + 1]) || (pk[4 * a + 2] - pk[4 * b + 2]));
    const fv = new Float32Array(pk.buffer);
    const out: Detections = { frame: new Int32Array(cnt), y: new Int32Array(cnt), x: new Int32Array(cnt), score: new Float32Array(cnt), noise: sig, median: med };
    idx.forEach((k, i) => {
      out.frame[i] = pk[4 * k]; out.y[i] = pk[4 * k + 1]; out.x[i] = pk[4 * k + 2];
      out.score[i] = Math.fround(fv[4 * k + 3] / sig);
    });
    for (const b of this.ubs) b.destroy();
    this.ubs = [];
    return out;
  }
}

/** torch_psf.spots.extract_rois: ROIs [n, P, P] of frames [T, H, W] centred on (t - t0, y, x). */
export function extractRois(frames: Float32Array, H: number, W: number, t0: number, d: { frame: Int32Array; y: Int32Array; x: Int32Array }, P: number, first = 0, count = d.frame.length - first): Float32Array {
  const h = (P - 1) >> 1;
  const out = new Float32Array(count * P * P);
  for (let n = 0; n < count; n++) {
    const k = first + n, t = d.frame[k] - t0;
    for (let i = 0; i < P; i++) {
      const row = (t * H + d.y[k] - h + i) * W + d.x[k] - h;
      out.set(frames.subarray(row, row + P), (n * P + i) * P);
    }
  }
  return out;
}
