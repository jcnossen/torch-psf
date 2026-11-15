// Synthetic, seeded input for the PDF fit report test: every section of FitReportData filled.
import type { FitReportData, ReportImage } from '../src/report/fitreport';
import type { QualityRange, QualityRoi } from '../src/report/quality';
import { qualityBands } from '../src/report/quality';
import { zernike } from '../src/psf/vectorial';

function rng(seed: number) {   // mulberry32
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export function syntheticReportData(): FitReportData {
  const r = rng(7);
  const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  const poisson = (m: number) => Math.max(0, Math.round(m + Math.sqrt(m) * gauss()));
  const noll = Array.from({ length: 17 }, (_, i) => i + 5);
  const truth = noll.map((j) => ({ 5: 0.02, 6: 0.1, 11: 0.03, 12: -0.015 } as Record<number, number>)[j] ?? 0);
  const values = truth.map((t) => t + 0.003 * gauss());
  const errors = noll.map(() => 0.002 + 0.002 * r());
  const start = noll.map((j) => (j === 6 ? 0.08 : 0));

  const nIt = 60, loss1 = new Float64Array(nIt), loss2 = new Float64Array(45), x2 = new Float64Array(45);
  for (let i = 0; i < nIt; i++) loss1[i] = 2.31 + 0.4 * Math.exp(-i / 9) + 0.002 * gauss();
  for (let i = 0; i < 45; i++) { x2[i] = nIt + i; loss2[i] = 2.27 + 0.03 * Math.exp(-i / 6) + 0.001 * gauss(); }

  const nAcc = 1758, nRej = 92;
  const dev = (n: number, m: number, s: number) => Float64Array.from({ length: n }, () => m + s * Math.abs(gauss()));
  const photons = Float64Array.from({ length: nAcc }, () => 2000 * Math.exp(0.5 * gauss()));
  const photRej = Float64Array.from({ length: nRej }, () => 4000 * Math.exp(0.4 * gauss()));
  const bg = Float64Array.from({ length: nAcc }, () => 20 + 3 * gauss());
  const z = Float64Array.from({ length: nAcc }, () => -0.6 + 1.2 * r());

  // pupil: unit disk; amplitude an aplanatic-like 1/sqrt(cos theta) rise, phase the Zernike sum (waves)
  const N = 64, amp = new Float32Array(N * N), ph = new Float32Array(N * N), NA = 1.49, nimm = 1.518;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = (j - (N - 1) / 2) / (N / 2), y = (i - (N - 1) / 2) / (N / 2), rho = Math.hypot(x, y), k = i * N + j;
    if (rho > 1) { amp[k] = NaN; ph[k] = NaN; continue; }
    const s = (rho * NA) / nimm;
    amp[k] = 1 / Math.sqrt(Math.sqrt(1 - s * s));
    let w = 0; noll.forEach((jj, q) => { w += values[q] * zernike(jj, rho, Math.atan2(y, x)); });
    ph[k] = w;
  }
  const pext: [number, number, number, number] = [-NA, NA, NA, -NA];

  // PSF: astigmatic Gaussian, sigma_x / sigma_y change with z
  const R = 33, zs = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6];
  const spot = (zz: number, w: number, h: number, x0 = (w - 1) / 2, y0 = (h - 1) / 2, I = 1, b = 0) => {
    const sx = 2.2 * Math.sqrt(1 + ((zz - 0.25) / 0.4) ** 2), sy = 2.2 * Math.sqrt(1 + ((zz + 0.25) / 0.4) ** 2);
    const a = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) a[y * w + x] = b + (I / (2 * Math.PI * sx * sy)) * Math.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2));
    return a;
  };
  const xy: ReportImage[] = zs.map((zz) => ({ data: spot(zz, R, R), width: R, height: R, label: `z = ${zz.toFixed(1)} µm` }));
  const NZ = 61, xz = new Float32Array(NZ * R);
  for (let k = 0; k < NZ; k++) { const zz = 0.8 - (1.6 * k) / (NZ - 1), s = spot(zz, R, R); xz.set(s.subarray(((R - 1) / 2) * R, ((R + 1) / 2) * R), k * R); }

  const P = 15;
  const examples = [0, 1, 2, 3].map((q) => {
    const zz = -0.5 + 0.33 * q, x0 = 7 + 0.6 * gauss(), y0 = 7 + 0.6 * gauss(), I = 1500 + 800 * q;
    const model = spot(zz, P, P, x0, y0, I, 18);
    return { label: `spot ${q + 1}: frame ${100 * q + 17}, z ${zz.toFixed(2)} µm`, model, data: Float32Array.from(model, poisson), width: P, height: P };
  });

  // quality range: deviance-like q (1 + noise, a tail of overlaps), rejection threshold 1.25; the rejected ROIs have
  // a dim second spot in the corner that the single-emitter model leaves in the difference
  const qd = Float32Array.from({ length: nAcc + nRej }, (_, i) => (i < nAcc ? 0.9 + 0.12 * Math.abs(gauss()) : 1.2 + 2 * Math.abs(gauss())));
  const thr = 1.25, bd = qualityBands(qd, (i) => qd[i] < thr);
  const qroi = (i: number): QualityRoi => {
    const zz = -0.5 + r(), x0 = 7 + 0.5 * gauss(), y0 = 7 + 0.5 * gauss(), I = 1500 + 2000 * r();
    const model = spot(zz, P, P, x0, y0, I, 18), bad = Math.max(0, qd[i] - 1);
    const truthImg = bad > 0 ? Float32Array.from(spot(zz, P, P, x0 + 5, y0 - 5, 600 * Math.min(bad, 3), 0), (v, k) => v + model[k]) : model;
    return { q: qd[i], rank: bd.rank[i], data: Float32Array.from(truthImg, poisson), model, width: P, height: P, photons: I, bg: 18, z: zz, label: `training spot ${i}, frame ${(i * 7) % 600}` };
  };
  const quality: QualityRange = {
    q: qd, cutoff: thr, cutoffKind: 'rejection', qLabel: 'Poisson deviance per px (first pass)', unit: 'training spots',
    notes: ['q = Poisson deviance per px of a training spot after its single-emitter fit (synthetic values).', `Rejection: q >= ${thr}; ${nRej} of ${nAcc + nRej} spots rejected.`],
    best: bd.best.map(qroi), lastAccepted: bd.lastAccepted.map(qroi), firstRejected: bd.firstRejected.map(qroi),
  };

  const G = 24, field = (f: (u: number, v: number) => number) => {
    const a = new Float32Array(G * G);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) a[i * G + j] = f(j / (G - 1) - 0.5, i / (G - 1) - 0.5);
    return a;
  };
  const fext: [number, number, number, number] = [0, 51.2, 0, 51.2];
  const zf = [
    { noll: 5, f: (u: number, v: number) => 0.02 + 0.03 * u * v },
    { noll: 6, f: (u: number, v: number) => 0.1 + 0.02 * (u * u - v * v) },
    { noll: 7, f: (u: number) => 0.015 * u },
    { noll: 8, f: (_: number, v: number) => -0.012 * v },
    { noll: 11, f: (u: number, v: number) => 0.03 - 0.01 * (u * u + v * v) },
  ].map(({ noll: j, f }) => ({ noll: j, map: { data: field(f), width: G, height: G, extent: fext, xlabel: 'x (µm)', ylabel: 'y (µm)' } }));

  return {
    title: 'PSF fit report',
    subtitle: 'Synthetic test data: in-situ estimate from demo_movie.tif (600 frames, 128 x 128 px). Every section of the report filled.',
    created: '2026-11-15T12:00:00.000Z',
    settings: {
      'pixel size (µm)': 0.1, NA, 'wavelength (µm)': 0.68, 'n immersion / medium': [nimm, 1.33], 'camera gain (e/ADU)': 0.45, 'camera offset (ADU)': 100,
      'ROI size (px)': P, 'z range (µm)': [-0.6, 0.6], 'focal plane (µm)': 0, 'Zernike modes (Noll)': '5-21', iterations: 80, 'reject overlapping spots': true,
      'start PSF': 'generic astigmatic (Noll 6 = 0.08 λ rms)',
    },
    results: {
      'spots (fit / held out)': '1758 / 496 of 2346', 'held-out NLL / px, fitted - start': -0.0123, 'held-out median χ² / px': 1.02,
      'z scale vs start': 1.003, 'iterations': '60 + 45 (second pass)', 'time': '41.2 s', 'rms vs truth (waves)': 0.0009,
      'warnings': 'none',
    },
    rois: { accepted: nAcc, rejected: nRej, heldout: 496, note: 'Rejected: Poisson deviance per pixel above median + 3 x scale (overlapping emitters).' },
    zernike: { noll, values, errors, start, truth },
    loss: { series: [{ y: loss1, label: 'pass 1' }, { x: x2, y: loss2, label: 'pass 2' }], ylabel: 'NLL / px' },
    histograms: [
      { title: 'Deviance per pixel', xlabel: 'deviance / px', accepted: dev(nAcc, 0.9, 0.12), rejected: dev(nRej, 1.4, 0.5), threshold: 1.25 },
      { title: 'Photons', xlabel: 'photons', accepted: photons, rejected: photRej },
      { title: 'Background', xlabel: 'photons / px', accepted: bg },
      { title: 'z (localized)', xlabel: 'z (µm)', accepted: z, bins: 30 },
    ],
    pupil: { amplitude: { data: amp, width: N, height: N, extent: pext, xlabel: 'kx / k0', ylabel: 'ky / k0' }, phase: { data: ph, width: N, height: N, extent: pext, xlabel: 'kx / k0', ylabel: 'ky / k0' } },
    psf: { xy, xz: { data: xz, width: R, height: NZ, label: 'xz section (z up)', extent: [-1.65, 1.65, 0.8, -0.8], xlabel: 'x (µm)', ylabel: 'z (µm)', pixelAspect: (1.6 / (NZ - 1)) / 0.1 }, log: 3 },
    examples,
    quality,
    zernikeField: zf,
    notes: ['Synthetic data generated by test/report_data.ts (seeded); the numbers are not from a fit.', 'Greek and symbols: σ, λ, χ², ±, µm, ≤, ≈, Δz.'],
  };
}
