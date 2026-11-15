// FitReportData from the web app's results (in-situ estimate, bead fit). Loaded on demand by the UI (dynamic import),
// so the report code is not part of the main bundle.
import { pupilData, aberrationPhase, Optics } from '../psf/vectorial';
import type { InsituReport } from '../smlm/estimate';
import type { FitSummary, InsituForm, VolumeSpec } from '../ui/protocol';
import { NCOL } from '../smlm/movie';
import { FitReportData, ReportImage, SettingValue, makeFitReport } from './fitreport';

export { makeFitReport };

/** Effective pupil amplitude (rms over the dipole field components, with the apodization) and the aberration phase
 *  (waves) on an n x n grid of the pupil; NaN outside the aperture. */
export function pupilMaps(optics: Optics, noll: number[], zcoef: number[], n = 64): { amplitude: ReportImage; phase: ReportImage } {
  const pd = pupilData({ ...optics, roisize: 9, npupil: n, oversample: 1, zernike_noll: noll });
  const W = aberrationPhase(pd, zcoef), NN = n * n;
  const amp = new Float32Array(NN), ph = new Float32Array(NN);
  let amax = 0;
  for (let k = 0; k < NN; k++) {
    if (!pd.mask[k]) { amp[k] = NaN; ph[k] = NaN; continue; }
    let s = 0;
    for (let c = 0; c < 6; c++) { const re = pd.P[2 * (c * NN + k)], im = pd.P[2 * (c * NN + k) + 1]; s += re * re + im * im; }
    amp[k] = Math.sqrt(s / 3); amax = Math.max(amax, amp[k]);
    ph[k] = W[k] / (2 * Math.PI);
  }
  for (let k = 0; k < NN; k++) amp[k] /= amax || 1;
  const e = optics.NA;
  const ext: [number, number, number, number] = [-e, e, -e, e];
  return {
    amplitude: { data: amp, width: n, height: n, origin: 'lower', extent: ext, xlabel: 'NA x', ylabel: 'NA y' },
    phase: { data: ph, width: n, height: n, origin: 'lower', extent: ext, xlabel: 'NA x', ylabel: 'NA y' },
  };
}

/** xy slices at up to 7 planes and the xz section through the centre of a PSF volume [nz, ny, nx]. */
export function volumeSlices(vol: Float32Array, spec: VolumeSpec, zname: string): FitReportData['psf'] {
  const nx = spec.nxy, nz = spec.nz, dz = (spec.zmax - spec.zmin) / Math.max(1, nz - 1);
  const k = Math.min(7, nz), xy: ReportImage[] = [];
  for (let i = 0; i < k; i++) {
    const iz = Math.round((i * (nz - 1)) / Math.max(1, k - 1));
    xy.push({ data: vol.subarray(iz * nx * nx, (iz + 1) * nx * nx), width: nx, height: nx, label: `${zname} ${(spec.zmin + iz * dz).toFixed(2)} µm` });
  }
  const xz = new Float32Array(nz * nx), row = nx >> 1;
  for (let z = 0; z < nz; z++) xz.set(vol.subarray((z * nx + row) * nx, (z * nx + row + 1) * nx), z * nx);
  const hw = (nx * spec.dxy_um) / 2;
  return { xy, xz: { data: xz, width: nx, height: nz, origin: 'lower', label: 'xz section', extent: [-hw, hw, spec.zmin, spec.zmax], xlabel: 'x (µm)', ylabel: `${zname} (µm)`, pixelAspect: dz / spec.dxy_um }, log: 3 };
}

function opticsSettings(o: Optics): Record<string, SettingValue> {
  return { 'pixel size (µm)': o.pixel_um, NA: o.NA, 'wavelength (µm)': o.wavelength_um, 'n immersion / medium': [o.n_imm, o.n_med] };
}

/** Split a concatenated history (iteration counter restarting per pass) into one series per pass, each as the NLL
 *  above its own minimum relative to it (the passes fit different spot sets, so their totals differ). */
function passes(h: { it: number; nll: number }[]) {
  const out: { x: number[]; y: number[]; label: string }[] = [];
  h.forEach((e, i) => {
    if (!out.length || e.it <= h[i - 1].it) out.push({ x: [], y: [], label: `pass ${out.length + 1}` });
    out[out.length - 1].x.push(i); out[out.length - 1].y.push(e.nll);
  });
  for (const p of out) { const m = Math.min(...p.y), a = Math.abs(m) || 1; p.y = p.y.map((v) => (v - m) / a); }
  return out;
}

export interface InsituContext {
  form: InsituForm; results: [string, string][]; source: string; truth?: number[] | null;
  locs?: Float32Array | null; volume?: { data: Float32Array; spec: VolumeSpec } | null;
}
export function insituReportData(r: InsituReport, c: InsituContext): FitReportData {
  const f = c.form, px = f.optics.pixel_um;
  const col = (k: number, s = 1) => { const L = c.locs!, n = L.length / NCOL, a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = L[i * NCOL + k] * s; return a; };
  const hasLocs = !!c.locs && c.locs.length > 0;
  return {
    title: 'PSF fit report: in-situ estimate',
    subtitle: `Movie: ${c.source}. torch-psf web, in-situ vectorial PSF fit (knot route)${r.partial ? '; stopped early (partial result)' : ''}.`,
    settings: {
      ...opticsSettings(f.optics), 'camera gain (e/ADU) / offset (ADU)': [f.gain, f.offset], 'read noise (e)': f.readnoise_e,
      'focal plane (µm)': f.focus_um, 'emitter z range (µm)': [f.zlo, f.zhi], 'ROI size (px)': f.roisize, 'detection threshold': f.threshold,
      'spots for the fit / held out': [f.nspots, f.nheldout], iterations: f.iters, 'overlap rejection (deviance k)': f.rejectOverlap ? f.devianceK ?? 3 : 'off',
      'spline refinement': !!f.refine, 'start Zernikes (Noll: waves)': f.noll.map((j, i) => [j, f.zcoef[i]]).filter(([, v]) => v !== 0).map(([j, v]) => `${j}: ${v}`).join(', ') || 'none',
    },
    results: Object.fromEntries(c.results),
    rois: { accepted: r.n_train - r.n_rejected, rejected: r.n_rejected, heldout: r.n_test, note: r.deviance ? `Fit spots rejected after the first pass: deviance per pixel above ${r.deviance.threshold.toFixed(3)} (overlapping emitters); the fit was then repeated without them. Held out: not fitted, used for the held-out NLL.` : undefined },
    zernike: { noll: r.noll, values: r.zcoef, start: r.zcoef_start, truth: c.truth ?? null },
    loss: r.history.length ? { series: passes(r.history), ylabel: '(NLL - pass minimum) / |minimum|', xlabel: 'iteration (all passes)', logy: true } : undefined,
    histograms: hasLocs ? [
      { title: 'Photons', xlabel: 'photons', accepted: col(4) },
      { title: 'Background', xlabel: 'photons / px', accepted: col(5) },
      { title: 'z', xlabel: 'z (µm)', accepted: col(3) },
      { title: 'CRLB x', xlabel: 'nm', accepted: col(6, 1000 * px) },
      { title: 'CRLB z', xlabel: 'nm', accepted: col(8, 1000) },
    ] : [],
    pupil: pupilMaps(f.optics, r.noll, r.zcoef),
    psf: c.volume ? volumeSlices(c.volume.data, c.volume.spec, 'z') : undefined,
    notes: [
      ...(hasLocs ? ['Histograms: all localizations of the movie with the fitted PSF.'] : []),
      ...r.warnings,
    ],
  };
}

export interface BeadContext { form: { optics: Optics; npupil: number }; results: [string, string][]; settings: Record<string, SettingValue> }
export function beadReportData(r: FitSummary, c: BeadContext): FitReportData {
  const R = r.R, ex = r.beads.map((b, k) => ({ b, k })).sort((a, b) => +b.b.heldout - +a.b.heldout).slice(0, 4);
  return {
    title: 'PSF fit report: bead z-stack',
    subtitle: `Stacks: ${r.stacks.map((s) => s.name).join(', ')}. torch-psf web bead fit${r.partial ? ' (partial result)' : ''}.`,
    settings: { ...opticsSettings(c.form.optics), ...c.settings },
    results: Object.fromEntries(c.results),
    rois: { accepted: r.beads.filter((b) => !b.heldout).length, heldout: r.beads.filter((b) => b.heldout).length, note: 'Beads used in the fit and held out (refitted with the PSF frozen).' },
    zernike: { noll: r.noll, values: r.zcoef },
    pupil: pupilMaps(c.form.optics, r.noll, r.zcoef, c.form.npupil),
    psf: volumeSlices(r.volume, r.volumeSpec, 'defocus'),
    examples: ex.map(({ b, k }) => {
      const j = b.nz >> 1, s = j * R * R;
      return { label: `bead ${k + 1}${b.heldout ? ' (held out)' : ''}, plane ${j} of ${b.nz}`, data: r.data[k].subarray(s, s + R * R), model: r.model[k].subarray(s, s + R * R), width: R, height: R };
    }),
  };
}
