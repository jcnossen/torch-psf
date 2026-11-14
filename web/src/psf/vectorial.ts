// Vectorial PSF: parameters and the pupil-plane precomputation (CPU, float64), the same physics and names as
// torch_psf.VectorialPSF (= SIMCODE simcode/psf/vectorial.py). The image computation runs on the GPU (gpu/ops.ts).
//
// Physics: freely rotating dipole at height z_e above the coverslip in a medium of index n_med, Fresnel s/p
// transmission into the immersion (n_imm) including supercritical-angle fluorescence, aplanatic apodization,
// Zernike phase (Noll, waves rms), defocus phase k0 (n_med cos_med z_e - n_imm cos_imm f) with f the focal-plane
// position relative to the coverslip (um), matrix Fourier transform to an oversampled pixel grid, pixel integration.
// Normalization: the PSF integrates to 1 over the image plane for an aberration-free emitter at z_e = 0.

export interface Optics {
  pixel_um: number;
  NA: number;
  wavelength_um: number;
  n_imm: number;
  n_med: number;
}

export interface VectorialConfig extends Optics {
  roisize: number;
  npupil: number;
  oversample: number;
  zernike_noll: number[];
  apod?: [number, number, number];
  saf?: [number, number, number, number];
}

export const DEFAULT_NOLL = Array.from({ length: 17 }, (_, i) => i + 5);

/** Noll index -> (n, m); m > 0 cosine, m < 0 sine term. */
export function nollToNm(j: number): [number, number] {
  let n = 0, j1 = j - 1;
  while (j1 > n) { n += 1; j1 -= n; }
  const m = (j % 2 === 0 ? 1 : -1) * ((n % 2) + 2 * Math.trunc((j1 + ((n + 1) % 2)) / 2));
  return [n, m];
}

function fact(n: number): number { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; }

/** Noll-normalized Zernike polynomial (rms 1 over the unit disk). */
export function zernike(j: number, rho: number, phi: number): number {
  const [n, m] = nollToNm(j);
  const am = Math.abs(m);
  let R = 0;
  for (let s = 0; s <= (n - am) / 2; s++) {
    const c = (s % 2 ? -1 : 1) * fact(n - s) / (fact(s) * fact((n + am) / 2 - s) * fact((n - am) / 2 - s));
    R += c * rho ** (n - 2 * s);
  }
  if (m === 0) return Math.sqrt(n + 1) * R;
  return Math.sqrt(2 * (n + 1)) * R * (m > 0 ? Math.cos(am * phi) : Math.sin(am * phi));
}

// --- minimal complex arithmetic (float64)
type C = [number, number];
const cadd = (a: C, b: C): C => [a[0] + b[0], a[1] + b[1]];
const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cdiv = (a: C, b: C): C => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
const cscale = (a: C, s: number): C => [a[0] * s, a[1] * s];
function csqrt(a: C): C {
  const r = Math.hypot(a[0], a[1]);
  const re = Math.sqrt(Math.max(0, (r + a[0]) / 2));
  let im = Math.sqrt(Math.max(0, (r - a[0]) / 2));
  if (a[1] < 0) im = -im;
  return [re, im];
}
const cexpi = (p: number): C => [Math.cos(p), Math.sin(p)];

export interface PupilData {
  N: number;
  ns: number;
  /** [6, N*N] complex interleaved: Ex_x, Ey_x, Ex_y, Ey_y, Ex_z, Ey_z (x, y, z dipole) times apodization */
  P: Float32Array;
  /** [N*N] complex k0 n_med cos(theta_med) (imag > 0: evanescent) */
  kzMed: Float32Array;
  /** [N*N] k0 n_imm cos(theta_imm) */
  kzImm: Float64Array;
  /** [N] pupil kx (= ky) samples, rad / um */
  k1: Float64Array;
  /** [nzern, N*N] Zernike basis times mask */
  Z: Float64Array;
  mask: Uint8Array;
  /** [ns, N] complex: exp(i u kx), u the oversampled image coordinates (um) */
  F: Float32Array;
  /** final scale: psf = C * mean_os(sum_c |E_c|^2 / 3) */
  C: number;
}

export function pupilData(cfg: VectorialConfig): PupilData {
  const N = cfg.npupil, NN = N * N;
  const NA = cfg.NA, n1 = cfg.n_med, n2 = cfg.n_imm;
  const k0 = 2 * Math.PI / cfg.wavelength_um;
  const apodc = cfg.apod ?? [0, 0, 0];
  const saf = cfg.saf ?? [0, 0, 0, 0];
  const g = new Float64Array(N);
  for (let i = 0; i < N; i++) g[i] = (i - (N - 1) / 2) / (N / 2);
  const P = new Float32Array(6 * NN * 2);
  const kzMed = new Float32Array(NN * 2);
  const kzImm = new Float64Array(NN);
  const mask = new Uint8Array(NN);
  const nzern = cfg.zernike_noll.length;
  const Z = new Float64Array(nzern * NN);
  let energy = 0;
  for (let iy = 0; iy < N; iy++) {
    for (let ix = 0; ix < N; ix++) {
      const k = iy * N + ix;
      const px = g[ix], py = g[iy];
      const rho = Math.sqrt(px * px + py * py);
      const phi = Math.atan2(py, px);
      const m = rho <= 1 ? 1 : 0;
      mask[k] = m;
      for (let z = 0; z < nzern; z++) Z[z * NN + k] = m ? zernike(cfg.zernike_noll[z], Math.min(rho, 1), phi) : 0;
      const sImm = NA * rho / n2;
      const cImm = Math.sqrt(Math.max(1 - sImm * sImm, 1e-6));
      const sMed = NA * rho / n1;
      let cMed = csqrt([1 - sMed * sMed, 0]);
      if (cMed[1] < 0) cMed = [-cMed[0], -cMed[1]];
      kzImm[k] = k0 * n2 * cImm;
      kzMed[2 * k] = k0 * n1 * cMed[0];
      kzMed[2 * k + 1] = k0 * n1 * cMed[1];
      if (!m) continue;
      const ts = cdiv(cscale(cMed, 2 * n1), cadd(cscale(cMed, n1), [n2 * cImm, 0]));
      const tp = cdiv(cscale(cMed, 2 * n1), cadd(cscale(cMed, n2), [n1 * cImm, 0]));
      const cphi = Math.cos(phi), sphi = Math.sin(phi);
      let apod: C = cdiv([(1 / Math.sqrt(cImm)) * (n2 / n1) * cImm, 0], cMed);
      const r2 = rho * rho;
      apod = cscale(apod, Math.exp(apodc[0] * r2 + apodc[1] * r2 * r2 + apodc[2] * r2 * r2 * r2));
      const ann = rho * NA > n1 ? 1 : 0;
      apod = cscale(apod, 1 + ann * (saf[2] * px + saf[3] * py));
      apod = cmul(apod, cexpi(2 * Math.PI * ann * (saf[0] * px + saf[1] * py)));
      const tpc = cmul(tp, cMed);
      const f: C[] = [
        cadd(cscale(tpc, cphi * cphi), cscale(ts, sphi * sphi)),
        cadd(cscale(tpc, cphi * sphi), cscale(ts, -sphi * cphi)),
        cadd(cscale(tpc, sphi * cphi), cscale(ts, -cphi * sphi)),
        cadd(cscale(tpc, sphi * sphi), cscale(ts, cphi * cphi)),
        cscale(tp, -sMed * cphi),
        cscale(tp, -sMed * sphi),
      ];
      for (let c = 0; c < 6; c++) {
        const v = cmul(f[c], apod);
        P[2 * (c * NN + k)] = v[0];
        P[2 * (c * NN + k) + 1] = v[1];
        energy += v[0] * v[0] + v[1] * v[1];
      }
    }
  }
  const k1 = new Float64Array(N);
  for (let i = 0; i < N; i++) k1[i] = k0 * NA * g[i];
  const os = cfg.oversample, R = cfg.roisize, ns = R * os;
  const F = new Float32Array(ns * N * 2);
  for (let v = 0; v < ns; v++) {
    const u = (v - (ns - 1) / 2) / os * cfg.pixel_um;
    for (let i = 0; i < N; i++) {
      F[2 * (v * N + i)] = Math.cos(u * k1[i]);
      F[2 * (v * N + i) + 1] = Math.sin(u * k1[i]);
    }
  }
  const dk = (k1[1] - k1[0]) ** 2;
  // psf = mean_os(sum|E|^2/3) * (dk/4pi^2)^2 * px^2 / (sum|P|^2/3 * dk/4pi^2)
  const C = (dk / (4 * Math.PI ** 2)) * cfg.pixel_um ** 2 * 3 / energy;
  return { N, ns, P, kzMed, kzImm, k1, Z, mask, F, C };
}

/** Aberration phase (rad) 2 pi sum_j c_j Z_j on the pupil grid. */
export function aberrationPhase(pd: PupilData, zcoef: ArrayLike<number>): Float64Array {
  const NN = pd.N * pd.N;
  const W = new Float64Array(NN);
  for (let z = 0; z < zcoef.length; z++) {
    const c = zcoef[z] * 2 * Math.PI;
    if (c === 0) continue;
    for (let k = 0; k < NN; k++) W[k] += c * pd.Z[z * NN + k];
  }
  return W;
}

/** Derivative directions of the GPU Jacobian. 'zern' expands to one direction per Zernike coefficient. */
export type Direction = 'x' | 'y' | 'focus' | 'ze' | 'zern';

/** The pupil multipliers G_theta (complex, [D, N*N]) such that dPb/dtheta = G_theta Pb; G_0 = 1 (the value). */
export function directionMultipliers(cfg: VectorialConfig, pd: PupilData, dirs: Direction[]): { G: Float32Array; names: string[] } {
  const N = pd.N, NN = N * N;
  const names = ['value'];
  const rows: ((k: number) => C)[] = [() => [1, 0]];
  for (const d of dirs) {
    if (d === 'x') { names.push('x'); rows.push((k) => [0, -pd.k1[k % N] * cfg.pixel_um]); }
    else if (d === 'y') { names.push('y'); rows.push((k) => [0, -pd.k1[Math.floor(k / N)] * cfg.pixel_um]); }
    else if (d === 'focus') { names.push('focus'); rows.push((k) => [0, -pd.kzImm[k]]); }
    else if (d === 'ze') { names.push('ze'); rows.push((k) => [-pd.kzMed[2 * k + 1], pd.kzMed[2 * k]]); }
    else {
      cfg.zernike_noll.forEach((j, z) => { names.push('noll' + j); rows.push((k) => [0, 2 * Math.PI * pd.Z[z * NN + k]]); });
    }
  }
  const G = new Float32Array(rows.length * NN * 2);
  rows.forEach((f, t) => {
    for (let k = 0; k < NN; k++) { const v = f(k); G[2 * (t * NN + k)] = v[0]; G[2 * (t * NN + k) + 1] = v[1]; }
  });
  return { G, names };
}
