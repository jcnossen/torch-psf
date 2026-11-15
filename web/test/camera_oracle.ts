// CPU reference for the photon-transfer per-pixel statistics (test oracle for smlm/camera.ts BlockStats, which runs
// them on the GPU): the same steps in float64 sums, rounded to float32 where torch stores float32.
const f32 = Math.fround;

/** Per-pixel background statistics of one block x [T, H, W] (ADU) with the detections (frame, y, x): the steps of
 *  photon_transfer between detect_spots and the pixel selection. Returns the 4th-round mean, the difference variance,
 *  the pair count per pixel and the number of kept pairs. */
export function cpuBlockStats(x: Float32Array, T: number, H: number, W: number, det: { frame: Int32Array; y: Int32Array; x: Int32Array }, R: number) {
  const HW = H * W, N = T * HW;
  // occupancy, dilated: Fn.max_pool2d(occ, 2R + 1, 1, R) > 0 = any detection within the (2R + 1)^2 square
  const occ = new Uint8Array(N);
  for (let i = 0; i < det.frame.length; i++) {
    const o = det.frame[i] * HW, y0 = Math.max(0, det.y[i] - R), y1 = Math.min(H - 1, det.y[i] + R);
    const x0 = Math.max(0, det.x[i] - R), x1 = Math.min(W - 1, det.x[i] + R);
    for (let yy = y0; yy <= y1; yy++) occ.fill(1, o + yy * W + x0, o + yy * W + x1 + 1);
  }
  // keep = ~occ; keep[1:] &= ~occ[:-1]. Every pixel is independent from here on: tiles of TILE pixels, the passes
  // over the frames per tile. Per pixel the count, sum and sum of squares of the kept samples in float64, so
  // sum((x - mu)^2) = S2 - 2 mu S1 + n mu^2 with the float32 mu (exact for integer ADU, else to ~1e-14: well below the
  // float32 rounding of torch's sums), and a clip pass subtracts the samples it drops (the sums of the next round
  // without another pass); the last clip pass accumulates the kept frame differences: 5 passes over the block.
  // Results rounded to float32 where torch stores a float32 tensor; the clip test |x - mu| < 4 sd in float32 as torch
  // (the float32 difference is formed only near the limit: elsewhere the float64 test gives the same answer).
  const keep = new Uint8Array(N);
  const mu = new Float32Array(HW), n = new Int32Array(HW), v = new Float32Array(HW);
  const TILE = 4096;
  const cnt = new Int32Array(TILE), s1 = new Float64Array(TILE), s2 = new Float64Array(TILE), lim = new Float32Array(TILE), lim0 = new Float64Array(TILE);
  const m = new Float64Array(TILE);
  let kept = 0;
  for (let p0 = 0; p0 < HW; p0 += TILE) {
    const L = Math.min(TILE, HW - p0);
    cnt.fill(0); s1.fill(0); s2.fill(0);
    passInit(x, occ, keep, HW, T, p0, L, cnt, s1, s2);
    for (let round = 0; round < 4; round++) {
      // mu = sum(x keep) / n; sd = sqrt(sum((x - mu)^2 keep) / n) (n clamped at 1)
      for (let q = 0; q < L; q++) {
        const c = Math.max(cnt[q], 1), u = f32(f32(s1[q]) / c);
        m[q] = mu[p0 + q] = u;
        const ss = Math.max(s2[q] - 2 * u * s1[q] + cnt[q] * u * u, 0);
        lim[q] = f32(4 * f32(Math.sqrt(f32(f32(ss) / c)))); lim0[q] = lim[q] * (1 - 1e-6);
      }
      // keep &= |x - mu| < 4 sd (the sums updated; after the last round: the sums of the kept frame differences)
      if (round === 3) { cnt.fill(0); s1.fill(0); s2.fill(0); }
      passClip(x, keep, HW, T, p0, L, m, lim, lim0, cnt, s1, s2, round === 3);
    }
    // var = sum (d - dm)^2 / max(n - 1, 1) / 2 over the kept pairs, dm = sum d / max(n, 1)
    for (let q = 0; q < L; q++) {
      const dm = f32(f32(s1[q]) / Math.max(cnt[q], 1));
      const ss = Math.max(s2[q] - 2 * dm * s1[q] + cnt[q] * dm * dm, 0);
      v[p0 + q] = f32(f32(f32(ss) / Math.max(cnt[q] - 1, 1)) / 2);
      n[p0 + q] = cnt[q]; kept += cnt[q];
    }
  }
  return { mu, var: v, n, kept_fraction: f32(kept / Math.max(1, (T - 1) * HW)) };
}

function passInit(x: Float32Array, occ: Uint8Array, keep: Uint8Array, HW: number, T: number, p0: number, L: number, cnt: Int32Array, s1: Float64Array, s2: Float64Array) {
  for (let t = 0; t < T; t++) {
    const o = t * HW + p0;
    for (let q = 0; q < L; q++) {
      const i = o + q, k = occ[i] | (t > 0 ? occ[i - HW] : 0) ? 0 : 1, xv = x[i];
      keep[i] = k; cnt[q] += k; s1[q] += k * xv; s2[q] += k * (xv * xv);
    }
  }
}
function passClip(x: Float32Array, keep: Uint8Array, HW: number, T: number, p0: number, L: number, m: Float64Array, lim: Float32Array,
  lim0: Float64Array, cnt: Int32Array, s1: Float64Array, s2: Float64Array, last: boolean) {
  for (let t = 0; t < T; t++) {
    const o = t * HW + p0;
    for (let q = 0; q < L; q++) {
      const i = o + q;
      if (keep[i]) {
        const xv = x[i], ad = Math.abs(xv - m[q]);
        if (!(ad < lim0[q]) && !(f32(ad) < lim[q])) {
          keep[i] = 0;
          if (!last) { cnt[q]--; s1[q] -= xv; s2[q] -= xv * xv; }
        }
      }
    }
    if (last && t > 0) for (let q = 0; q < L; q++) { const i = o + q, k = keep[i] & keep[i - HW], d = f32(x[i] - x[i - HW]); cnt[q] += k; s1[q] += k * d; s2[q] += k * (d * d); }
  }
}

