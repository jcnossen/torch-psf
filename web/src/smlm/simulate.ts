// The demo SMLM movie, simulated in the browser at start-up (nothing is stored in the repository). The same model as
// torch_psf.simulate.simulate_reference_movie / simulate_movie: binding sites that blink with a two-state Markov
// chain (mean on-time mean_on frames; the on-probability gives emitters_per_frame emitters on average), log-normal
// photons, constant background, Poisson noise, then the camera: adu = round(photons / gain + offset + N(0, readnoise
// / gain)) clipped to 0..65535. The PSF is the astigmatic preset of torch_psf.simulate.PRESETS (Zernikes 5: 0.02,
// 6: 0.10, 11: 0.03 waves, focal plane 0.4 um, emitters 0 .. 0.8 um), rendered through its spline. Sites: on curved
// filaments (default, as torch_psf.simulate.filament_sites; the in-situ estimate handles them since the overlap
// rejection of InSituEstimator) or uniform in the frame and in z (sites: 'uniform', as simulate_reference_movie).
// The random numbers are not torch's: same model and statistics, not the same bits.
import { buildKnots } from '../psf/spline';
import { DEFAULT_NOLL, Optics } from '../psf/vectorial';
import { splineGrid } from '../gpu/spline_lm';
import { renderSpline, KnotGridFull } from './insitu';

export const DEMO_ZERNIKE: Record<number, number> = { 5: 0.02, 6: 0.10, 11: 0.03 };
export const DEMO_OPTICS: Optics = { pixel_um: 0.108, NA: 1.49, wavelength_um: 0.68, n_imm: 1.518, n_med: 1.33 };
export const DEMO = { T: 600, H: 128, W: 128, nsites: 900, emitters_per_frame: 10, mean_on: 3, photons: 3000, photons_logsd: 0.3, bg: 20,
  gain: 0.5, offset: 100, readnoise: 1.5, focus_um: 0.4, zlo: 0.0, zhi: 0.8, roisize: 19, filaments: 7, sites: 'filaments' as 'uniform' | 'filaments', seed: 1 };

/** mulberry32 */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function normal(u: () => number) { return Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u()); }
/** Poisson: Knuth below 30, transformed rejection (PTRS, Hormann 1993) above. */
function poisson(lam: number, u: () => number): number {
  if (lam < 30) { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= u(); } while (p > L); return k - 1; }
  const slam = Math.sqrt(lam), loglam = Math.log(lam), b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
  const invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
  for (;;) {
    const U = u() - 0.5, V = u(), us = 0.5 - Math.abs(U);
    const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
    if (us >= 0.07 && V <= vr) return k;
    if (k < 0 || (us < 0.013 && V > us)) continue;
    if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - lgam(k + 1)) return k;
  }
}
function lgam(x: number): number {   // Stirling with corrections (x >= 1)
  if (x < 7) { let p = 1; while (x < 7) { p *= x; x++; } return lgam(x) - Math.log(p); }
  const x2 = 1 / (x * x);
  return (x - 0.5) * Math.log(x) - x + 0.9189385332046727 + (1 / 12 - x2 * (1 / 360 - x2 / 1260)) / x;
}

export interface DemoMovie {
  adu: Uint16Array; T: number; H: number; W: number;
  truth: Float32Array;        // [n, 5] = frame, x px, y px, z um, photons
  zernike: Record<number, number>; optics: Optics; params: typeof DEMO;
}

export async function simulateDemo(dev: GPUDevice, p = DEMO): Promise<DemoMovie> {
  const u = rng(p.seed);
  // filament sites: smooth random curves (quadratic Bezier), z varying slowly along each
  const sx: number[] = [], sy: number[] = [], sz: number[] = [];
  const m = 6;
  if (p.sites === 'uniform') {
    for (let i = 0; i < p.nsites; i++) { sx.push(4 + (p.W - 8) * u()); sy.push(4 + (p.H - 8) * u()); sz.push(p.zlo + (p.zhi - p.zlo) * u()); }
  }
  for (let f = 0; f < (p.sites === 'filaments' ? p.filaments : 0); f++) {
    const P0 = [m + u() * (p.W - 2 * m), m + u() * (p.H - 2 * m)], P2 = [m + u() * (p.W - 2 * m), m + u() * (p.H - 2 * m)];
    const P1 = [m + u() * (p.W - 2 * m), m + u() * (p.H - 2 * m)];
    const z0 = p.zlo + 0.1 + 0.6 * u(), z1 = p.zlo + 0.1 + 0.6 * u();
    const n = Math.round(p.nsites / p.filaments);
    for (let i = 0; i < n; i++) {
      const t = u();
      const x = (1 - t) ** 2 * P0[0] + 2 * t * (1 - t) * P1[0] + t * t * P2[0] + 0.25 * normal(u);
      const y = (1 - t) ** 2 * P0[1] + 2 * t * (1 - t) * P1[1] + t * t * P2[1] + 0.25 * normal(u);
      if (x < m || y < m || x > p.W - m || y > p.H - m) continue;
      sx.push(x); sy.push(y); sz.push(z0 + (z1 - z0) * t + 0.02 * normal(u));
    }
  }
  const ns = sx.length;
  const frac = p.emitters_per_frame / ns, pOn = frac / (1 - frac) / p.mean_on;
  // the PSF through its spline (torch_psf CubicSplinePSF grid), roisize 19 as the preset
  const zc = DEFAULT_NOLL.map((j) => DEMO_ZERNIKE[j] ?? 0);
  const g = splineGrid(p.roisize, p.zlo, p.zhi, 2, 0.04, 2, 3.0) as KnotGridFull;
  const k = await buildKnots(dev, { ...DEMO_OPTICS, roisize: 0, npupil: 64, oversample: 3, zernike_noll: DEFAULT_NOLL }, zc,
    { roisize: p.roisize, focus_um: p.focus_um, zmin: p.zlo, zmax: p.zhi, S: 2, hz: 0.04, zpad: 2, xmax: 3.0 });
  const ph = new Float32Array(p.T * p.H * p.W).fill(p.bg);
  const truth: number[] = [];
  const state = new Uint8Array(ns);
  const R = p.roisize, hr = (R - 1) >> 1;
  for (let t = 0; t < p.T; t++) {
    for (let s = 0; s < ns; s++) {
      const v = u();
      state[s] = state[s] ? (v > 1 / p.mean_on ? 1 : 0) : (v < pOn ? 1 : 0);
      if (!state[s]) continue;
      const I = Math.exp(Math.log(p.photons) + p.photons_logsd * normal(u));
      const cx = Math.round(sx[s]), cy = Math.round(sy[s]);
      const img = renderSpline(k.knots, g, sx[s] - cx, sy[s] - cy, sz[s]);
      for (let i = 0; i < R; i++) {
        const y = cy - hr + i; if (y < 0 || y >= p.H) continue;
        for (let j = 0; j < R; j++) {
          const x = cx - hr + j; if (x < 0 || x >= p.W) continue;
          ph[(t * p.H + y) * p.W + x] += I * img[i * R + j];
        }
      }
      truth.push(t, sx[s], sy[s], sz[s], I);
    }
  }
  const adu = new Uint16Array(ph.length);
  for (let i = 0; i < ph.length; i++) {
    const n = poisson(ph[i], u);
    adu[i] = Math.max(0, Math.min(65535, Math.round(n / p.gain + p.offset + (p.readnoise / p.gain) * normal(u))));
  }
  return { adu, T: p.T, H: p.H, W: p.W, truth: Float32Array.from(truth), zernike: { ...DEMO_ZERNIKE }, optics: { ...DEMO_OPTICS }, params: p };
}

/** A movie held in memory (ADU), with the reader interface of io/tiff.TiffMovie. */
export class ArrayMovie {
  constructor(public adu: Uint16Array, public n: number, public H: number, public W: number) {}
  async read(f0: number, f1: number): Promise<Float32Array> {
    const fp = this.H * this.W;
    return Float32Array.from(this.adu.subarray(f0 * fp, f1 * fp));
  }
}
