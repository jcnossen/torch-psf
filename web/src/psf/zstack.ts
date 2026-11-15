// The model rendered through z for the psf.zip zstack.tif (torch_psf.psfzip.render_zstack): planes evenly over the
// z range (numpy.linspace), lateral sampling 1 / k camera px by k x k sub-pixel emitter shifts. Sample (m, n) holds
// the pixel-integrated PSF of a camera pixel centred at ((n - (kP - 1) / 2) / k, (m - (kP - 1) / 2) / k) px from the
// emitter. Vectorial models on the GPU (VecEngine), splines on the CPU (CubicSplinePSF.render, Catmull-Rom).
import { VecEngine } from '../gpu/ops';
import type { Optics } from './vectorial';

/** numpy.linspace(lo, hi, n) (endpoint included), bit for bit. */
export function linspace(lo: number, hi: number, n: number): number[] {
  if (n === 1) return [lo];
  const step = (hi - lo) / (n - 1);
  const z = Array.from({ length: n }, (_, i) => i * step + lo);
  z[n - 1] = hi;
  return z;
}

/** A vectorial model as stored in psf.json (the web app's subset: no pupil maps, blur or dipole terms). */
export interface VectorialModel {
  optics: Optics; noll: number[]; zcoef: ArrayLike<number>; focus_um: number;
  npupil: number; oversample: number; roisize: number; z_range_um: [number, number] | null;
  apod?: [number, number, number]; saf?: [number, number, number, number];
}

/** torch_psf CubicSplinePSF grid ('spline' dict of psf.json) and knots [nz, n, n]. */
export interface SplineModel {
  roisize: number; z_range: [number, number]; oversample: number; dz_um: number; zpad: number; xy_limit: number;
  c: number; z0_um: number; nz: number; u0_px: number; pixel_um: number | null;
  knots: Float32Array;
}

/** The CubicSplinePSF grid for these settings (the layout the class builds; xy_limit reduced as the class does). */
export function splineDict(roisize: number, z_range: [number, number], S: number, dz_um: number, zpad: number, xy_limit: number, pixel_um: number | null): Omit<SplineModel, 'knots'> {
  const h = (roisize - 1) / 2;
  const c = Math.ceil(h + xy_limit) + 2, nu = (2 * c + 1) * S;
  const nz = Math.round((z_range[1] - z_range[0]) / dz_um) + 1 + 2 * zpad;
  const u0 = -(c * S + S - 1) / S, oS0 = (-h - u0) * S;
  const lim = Math.min((oS0 - 1) / S, (nu - 2 - oS0 - (roisize - 1) * S) / S - 1e-4);
  return { roisize, z_range: [z_range[0], z_range[1]], oversample: S, dz_um, zpad, xy_limit: Math.min(xy_limit, Math.max(lim, 0)),
    c, z0_um: z_range[0] - zpad * dz_um, nz, u0_px: u0, pixel_um };
}

const G = [[0, 1, 0, 0], [-0.5, 0, 0.5, 0], [1, -2.5, 2, -0.5], [-0.5, 1.5, -1.5, 0.5]];   // Catmull-Rom, tau = 0.5
function crWeights(t: number): number[] {
  const p = [1, t, t * t, t * t * t];
  return [0, 1, 2, 3].map((j) => p[0] * G[0][j] + p[1] * G[1][j] + p[2] * G[2][j] + p[3] * G[3][j]);
}

/** CubicSplinePSF.render (value only) of one emitter (x, y px from the ROI centre, z um) -> [P, P]. */
export function splineRender(s: SplineModel, x: number, y: number, z: number): Float32Array {
  const P = s.roisize, S = s.oversample, n = (2 * s.c + 1) * S, nn = n * n;
  const h = (P - 1) / 2, oS0 = (-h - s.u0_px) * S, xm = s.xy_limit;
  const cl = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
  const t = [cl((z - s.z0_um) / s.dz_um, 1, s.nz - 2 - 1e-4), oS0 - cl(y, -xm, xm) * S, oS0 - cl(x, -xm, xm) * S];
  const ix = t.map(Math.floor), w = t.map((v, i) => crWeights(v - ix[i]));
  const out = new Float32Array(P * P);
  const K = s.knots;
  for (let p = 0; p < P; p++) for (let q = 0; q < P; q++) {
    let acc = 0;
    for (let a = 0; a < 4; a++) {
      const kz = (ix[0] - 1 + a) * nn;
      for (let b = 0; b < 4; b++) {
        const kv = kz + (ix[1] - 1 + p * S + b) * n + ix[2] - 1 + q * S;
        const wab = w[0][a] * w[1][b];
        for (let c = 0; c < 4; c++) acc += wab * w[2][c] * K[kv + c];
      }
    }
    out[p * P + q] = acc;
  }
  return out;
}

export interface ZStack { stack: Float32Array; Z: number; C: number; H: number; W: number; z_um: number[]; sample_px: number }

/** Interleave renders [Z, k (y shift), k (x shift), P, P] to [Z, 1, kP, kP] (render_zstack's flip + permute). */
function interleave(img: Float32Array, Z: number, k: number, P: number): Float32Array {
  const W = k * P, out = new Float32Array(Z * W * W);
  for (let z = 0; z < Z; z++) for (let iy = 0; iy < k; iy++) for (let ix = 0; ix < k; ix++) {
    const src = ((z * k + iy) * k + ix) * P * P;
    for (let p = 0; p < P; p++) for (let q = 0; q < P; q++)
      out[(z * W + p * k + (k - 1 - iy)) * W + q * k + (k - 1 - ix)] = img[src + p * P + q];
  }
  return out;
}

function shifts(k: number): number[] { return Array.from({ length: k }, (_, i) => (i - (k - 1) / 2) / k); }

function checkK(k: number) { if (!Number.isInteger(k) || k < 1) throw new Error(`z-stack sampling: 1 / k px with an integer k >= 1, not k = ${k}`); }

/** render_zstack of a vectorial model: planes over z_range_um (emitter height at focus_um). */
export async function vectorialZstack(dev: GPUDevice, m: VectorialModel, planes = 25, k = 1): Promise<ZStack> {
  checkK(k);
  const zr = m.z_range_um ?? [0, 1];
  const zs = linspace(zr[0], zr[1], planes), a = shifts(k), P = m.roisize;
  const eng = new VecEngine(dev, { ...m.optics, roisize: P, npupil: m.npupil, oversample: m.oversample, zernike_noll: m.noll, apod: m.apod, saf: m.saf }, [], { maxChunk: 128 });
  try {
    eng.setZernike(m.zcoef);
    const ps = [];
    for (const z of zs) for (let iy = 0; iy < k; iy++) for (let ix = 0; ix < k; ix++) ps.push({ x: a[ix], y: a[iy], ze: z, f: m.focus_um, I: 1, bg: 0 });
    eng.setImages(ps);
    await eng.run(ps.length, 'psf');
    const img = await eng.readPsf(ps.length);
    return { stack: interleave(img, zs.length, k, P), Z: zs.length, C: 1, H: k * P, W: k * P, z_um: zs, sample_px: 1 / k };
  } finally { eng.destroy(); }
}

/** render_zstack of a spline model: planes over its z_range. */
export function splineZstack(s: SplineModel, planes = 25, k = 1): ZStack {
  checkK(k);
  const zs = linspace(s.z_range[0], s.z_range[1], planes), a = shifts(k), P = s.roisize, P2 = P * P;
  const img = new Float32Array(zs.length * k * k * P2);
  let o = 0;
  for (const z of zs) for (let iy = 0; iy < k; iy++) for (let ix = 0; ix < k; ix++) { img.set(splineRender(s, a[ix], a[iy], z), o); o += P2; }
  return { stack: interleave(img, zs.length, k, P), Z: zs.length, C: 1, H: k * P, W: k * P, z_um: zs, sample_px: 1 / k };
}
