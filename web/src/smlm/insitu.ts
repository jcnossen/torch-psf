// In-situ PSF estimation from SMLM spots by the knot route: a port of torch_psf.estimation.fit_zernike_knots (route 'K')
// and torch_psf.pipeline.estimate_psf_insitu. Per iteration: knots of the CubicSplinePSF from the vectorial model at
// the current Zernikes (with the Zernike derivatives of every knot image, for the vector-Jacobian product), emitter
// parameters by Levenberg-Marquardt on the spline (zstarts and 30 iterations the first time, 3 afterwards), the NLL
// gradient on the knots, its VJP to the Zernikes, an Adam step (lr 2e-3, betas 0.9 / 0.999, eps 1e-12).
import { VecEngine } from '../gpu/ops';
import { SplineLocalizer, SplineGrid, LocResult } from '../gpu/spline_lm';
import { VectorialConfig } from '../psf/vectorial';
import { StopToken } from '../core/task';

export interface KnotGridFull extends SplineGrid { c: number; Pl: number; }

/** Knots [nz, nv, nu] of the vectorial model at zcoef and their derivatives with respect to every Zernike
 *  [nZ, nz, nv, nu] (torch_psf CubicSplinePSF.build_planes: knot z = emitter height, focus fixed). */
export async function knotsAndJacobian(dev: GPUDevice, cfg: VectorialConfig, zcoef: ArrayLike<number>, g: KnotGridFull, focus_um: number,
  check?: () => void): Promise<{ knots: Float32Array; dK: Float32Array }> {
  const S = g.S, Pl = g.Pl, nK = g.nz * g.nv * g.nu, nZ = cfg.zernike_noll.length;
  const eng = new VecEngine(dev, { ...cfg, roisize: Pl }, ['zern']);
  try {
    eng.setZernike(zcoef);
    const ps: { x: number; y: number; ze: number; f: number; I: number; bg: number }[] = [];
    for (let k = 0; k < g.nz; k++) for (let b = 0; b < S; b++) for (let a = 0; a < S; a++)
      ps.push({ x: a / S, y: b / S, ze: g.z0 + g.hz * k, f: focus_um, I: 1, bg: 0 });
    const knots = new Float32Array(nK), dK = new Float32Array(nZ * nK);
    const RR = Pl * Pl, D = eng.D, C = eng.jacobianChunk;
    for (let s = 0; s < ps.length; s += C) {
      check?.();
      const part = ps.slice(s, s + C), B = part.length;
      const J = await eng.psfAndJacobian(part);            // [D, B, Pl, Pl]
      for (let q = 0; q < B; q++) {
        const n = s + q, k = Math.floor(n / (S * S)), b = Math.floor(n / S) % S, a = n % S;
        for (let i = 0; i < Pl; i++) for (let j = 0; j < Pl; j++) {
          const kn = (k * g.nv + i * S + (S - 1 - b)) * g.nu + j * S + (S - 1 - a);
          knots[kn] = J[q * RR + i * Pl + j];
          for (let z = 0; z < nZ; z++) dK[z * nK + kn] = J[((1 + z) * B + q) * RR + i * Pl + j];
        }
      }
    }
    if (D !== 1 + nZ) throw new Error('unexpected derivative directions');
    return { knots, dK };
  } finally { eng.destroy(); }
}

export interface InsituIter { it: number; nll: number; seconds: number; zcoef: number[]; }

/** torch_psf.estimation.fit_zernike_knots (route 'K') on ROIs [N, P, P] (photons). */
export async function fitZernikeKnots(dev: GPUDevice, rois: Float32Array, N: number, cfg: VectorialConfig, zc0: number[], g: KnotGridFull,
  focus_um: number, o: { iters?: number; lr?: number; locIters?: number; tok?: StopToken; log?: (h: InsituIter) => void; var?: number } = {}) {
  const iters = o.iters ?? 80, lr = o.lr ?? 2e-3, nZ = zc0.length, nK = g.nz * g.nv * g.nu;
  let zc = Float64Array.from(zc0);
  const m = new Float64Array(nZ), v = new Float64Array(nZ);
  const hist: InsituIter[] = [];
  let loc: SplineLocalizer | null = null;
  let th: Float32Array | null = null, K0: Float32Array | null = null, lastKnots: Float32Array | null = null;
  const t0 = performance.now();
  try {
    for (let it = 0; it < iters; it++) {
      o.tok?.check();
      const { knots, dK } = await knotsAndJacobian(dev, cfg, Array.from(zc, Math.fround), g, focus_um, () => o.tok?.check());
      if (!K0) K0 = knots;
      lastKnots = knots;
      if (!loc) loc = new SplineLocalizer(dev, g, knots); else loc.setKnots(knots);
      const r: LocResult = th ? await loc.run(rois, N, th, { iters: o.locIters ?? 3, var: o.var }) : await loc.localize(rois, N, { var: o.var });
      const thNow: Float32Array = r.theta;
      th = thNow;
      const { gK, nll } = await loc.knotGradient(rois, N, thNow, o.var ?? 0);
      const gz = new Float64Array(nZ);
      for (let z = 0; z < nZ; z++) { let s = 0; const off = z * nK; for (let k = 0; k < nK; k++) s += gK[k] * dK[off + k]; gz[z] = s; }
      const h: InsituIter = { it, nll, seconds: (performance.now() - t0) / 1000, zcoef: Array.from(zc) };
      hist.push(h); o.log?.(h);
      for (let z = 0; z < nZ; z++) {
        m[z] = 0.9 * m[z] + 0.1 * gz[z];
        v[z] = 0.999 * v[z] + 0.001 * gz[z] * gz[z];
        const step = (m[z] / (1 - 0.9 ** (it + 1))) / (Math.sqrt(v[z] / (1 - 0.999 ** (it + 1))) + 1e-12);
        zc[z] = Math.fround(zc[z] - lr * step);
      }
      if (o.tok?.stopEarly) break;
    }
  } finally { loc?.destroy(); }
  return { zcoef: Array.from(zc), theta: th!, K0: K0!, knots: lastKnots!, history: hist, seconds: (performance.now() - t0) / 1000 };
}

/** Value-only spline render of one emitter (CPU; torch_psf CubicSplinePSF.render), for z_scale. */
export function renderSpline(K: Float32Array, g: KnotGridFull, x: number, y: number, z: number): Float32Array {
  const P = g.roisize, S = g.S, h = (P - 1) >> 1;
  const oS0 = (-h + (g.c * S + S - 1) / S) * S;
  const cr = (t: number) => { const t2 = t * t, t3 = t2 * t; return [-0.5 * t + t2 - 0.5 * t3, 1 - 2.5 * t2 + 1.5 * t3, 0.5 * t + 2 * t2 - 1.5 * t3, -0.5 * t2 + 0.5 * t3]; };
  const tz = Math.min(Math.max((z - g.z0) / g.hz, 1), g.nz - 2 - 1e-4);
  const ty = oS0 - Math.min(Math.max(y, -g.xmax), g.xmax) * S, tx = oS0 - Math.min(Math.max(x, -g.xmax), g.xmax) * S;
  const iz = Math.floor(tz), iy = Math.floor(ty), ix = Math.floor(tx);
  const wz = cr(tz - iz), wy = cr(ty - iy), wx = cr(tx - ix);
  const out = new Float32Array(P * P);
  for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) {
    let v = 0;
    for (let c = 0; c < 4; c++) for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++)
      v += wz[c] * wy[a] * wx[b] * K[((iz - 1 + c) * g.nv + (iy - 1 + i * S + a)) * g.nu + (ix - 1 + j * S + b)];
    out[i * P + j] = v;
  }
  return out;
}

/** torch_psf.estimate.z_scale(psf_new, psf_ref): bright noise-free emitters at 33 heights rendered with the new knots,
 *  localized with the reference knots; slope of z_ref against z_new over the inner 80 % and the offset at the centre. */
export async function zScale(dev: GPUDevice, Knew: Float32Array, Kref: Float32Array, g: KnotGridFull, n = 33): Promise<{ scale: number; offset: number }> {
  const P2 = g.roisize ** 2, rois = new Float32Array(n * P2), z: number[] = [];
  for (let k = 0; k < n; k++) {
    const zz = Math.fround(g.zlo + (g.zhi - g.zlo) * k / (n - 1)); z.push(zz);
    const r = renderSpline(Knew, g, 0, 0, zz);
    for (let i = 0; i < P2; i++) rois[k * P2 + i] = 1e5 * r[i] + 1;
  }
  const loc = new SplineLocalizer(dev, g, Kref);
  try {
    const r = await loc.localize(rois, n);
    const lo = Math.floor(0.1 * n), hi = n - Math.floor(0.1 * n);
    let mz = 0; for (let k = lo; k < hi; k++) mz += z[k]; mz /= hi - lo;
    let sxy = 0, sxx = 0, sy = 0;
    for (let k = lo; k < hi; k++) { const dx = z[k] - mz, yy = r.theta[k * 5 + 2]; sxy += dx * yy; sxx += dx * dx; sy += yy; }
    return { scale: sxy / sxx, offset: sy / (hi - lo) - mz };
  } finally { loc.destroy(); }
}

/** torch_psf.estimation.split_checkerboard: parity of the tile x tile px tile of each position. */
export function checkerboard(x: number, y: number, tile = 16): number { return (Math.floor(x / tile) + Math.floor(y / tile)) % 2; }
