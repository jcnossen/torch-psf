// SMLM movie -> localizations (worker side): torch_psf.pipeline.localize_movie with a ConvSplinePSF built from a
// vectorial PSF (torch_psf ConvSplinePSF.from_vectorial grid defaults). Per chunk of frames: ADU -> photons
// (camera.to_photons: (adu - offset) x gain), detect_spots (+ extract_spots border rule (P - 1) / 2 + 2), extract_rois,
// localize (zstarts 5, 30 iterations), crlb, chi2, at_limit -> rows of the localization table (torch_psf.table).
import { TiffMovie } from '../io/tiff';
import { Detector } from '../gpu/detect';
import { extractRois } from '../gpu/detect';
import { SplineLocalizer, splineGrid } from '../gpu/spline_lm';
import { buildKnots } from '../psf/spline';
import { StopToken, Eta, Progress } from '../core/task';
import type { MovieForm } from '../ui/protocol';

/** Columns of a localization row (torch_psf.table.LOC_DTYPE). */
export const LOC_COLUMNS = ['frame', 'x', 'y', 'z', 'photons', 'bg', 'crlb_x', 'crlb_y', 'crlb_z', 'crlb_photons', 'crlb_bg',
  'loglik', 'chi2', 'at_limit'] as const;
export const NCOL = LOC_COLUMNS.length;

export interface MovieSummary {
  n_locs: number; frames_read: number; frames_total: number; partial: boolean; seconds: number;
  H: number; W: number; spots_per_s: number; at_limit_fraction: number;
}

export async function localizeMovie(device: GPUDevice, files: File[], form: MovieForm, tok: StopToken,
  progress: (p: Progress) => void, emit: (rows: Float32Array) => void): Promise<MovieSummary> {
  const mv = await TiffMovie.open(files, (msg) => progress({ stage: 'header', done: 0, total: 1, info: { msg } }));
  const P = form.roisize, h = (P - 1) >> 1;
  const nFrames = form.maxFrames > 0 ? Math.min(mv.n, form.maxFrames) : mv.n;
  // the spline PSF (torch_psf ConvSplinePSF.from_vectorial: oversample 2, dz 0.04 um, zpad 2, xy_limit 3)
  const grid = splineGrid(P, form.zlo, form.zhi, 2, 0.04, 2, 3.0);
  progress({ stage: 'spline', done: 0, total: 1 });
  const k = await buildKnots(device, { ...form.optics, roisize: 0, npupil: form.npupil, oversample: form.oversample, zernike_noll: form.noll },
    form.zcoef, { roisize: P, focus_um: form.focus_um, zmin: form.zlo, zmax: form.zhi, S: grid.S, hz: grid.hz, zpad: 2, xmax: grid.xmax }, () => tok.check());
  const loc = new SplineLocalizer(device, grid, k.knots);
  const chunk = Math.max(1, Math.min(form.chunk, nFrames));
  const det = new Detector(device, mv.H, mv.W, chunk, { threshold: form.threshold, border: h + 2, isolation: form.isolation > 0 ? form.isolation : null });
  const eta = new Eta();
  const t0 = performance.now();
  let n = 0, nLimit = 0, done = 0, partial = false;
  const varE = form.readnoise_e > 0 ? Math.fround(form.readnoise_e * form.readnoise_e) : 0;
  try {
    for (let f0 = 0; f0 < nFrames; f0 += chunk) {
      tok.check();
      if (tok.stopEarly) { partial = true; break; }
      const f1 = Math.min(nFrames, f0 + chunk), T = f1 - f0;
      const fr = await mv.read(f0, f1);
      const gain = Math.fround(form.gain), off = Math.fround(form.offset);
      for (let i = 0; i < fr.length; i++) fr[i] = Math.fround(Math.fround(fr[i] - off) * gain);
      const d = await det.detect(fr, T, f0);
      const N = d.frame.length;
      if (N) {
        const rois = extractRois(fr, mv.H, mv.W, f0, d, P);
        const r = await loc.localize(rois, N, { var: varE });
        const rows = new Float32Array(N * NCOL);
        for (let i = 0; i < N; i++) {
          const o = i * NCOL, t = r.theta, c = r.crlb;
          const lim = Math.max(Math.abs(t[i * 5]), Math.abs(t[i * 5 + 1])) > grid.xmax - 0.06 ? 1 : 0;
          nLimit += lim;
          rows[o] = d.frame[i];
          rows[o + 1] = d.x[i] + t[i * 5]; rows[o + 2] = d.y[i] + t[i * 5 + 1];   // corner + (P - 1) / 2 = detection pixel
          rows[o + 3] = t[i * 5 + 2]; rows[o + 4] = t[i * 5 + 3]; rows[o + 5] = t[i * 5 + 4];
          for (let q = 0; q < 5; q++) rows[o + 6 + q] = c[i * 5 + q];
          rows[o + 11] = -r.nll[i]; rows[o + 12] = r.chi2[i]; rows[o + 13] = lim;
        }
        emit(rows);
        n += N;
      }
      done = f1;
      progress({ stage: 'localize', done, total: nFrames, info: { localizations: n }, ...eta.update(done, nFrames) });
    }
  } finally { det.destroy(); loc.destroy(); }
  const seconds = (performance.now() - t0) / 1000;
  return { n_locs: n, frames_read: done, frames_total: mv.n, partial, seconds, H: mv.H, W: mv.W, spots_per_s: n / Math.max(seconds, 1e-9), at_limit_fraction: n ? nLimit / n : 0 };
}

/** CSV of the localization rows (header = torch_psf.table.LOC_DTYPE names; x, y in px, z in um). */
export function locsToCsv(rows: Float32Array): string {
  const n = rows.length / NCOL;
  const out: string[] = [LOC_COLUMNS.join(',')];
  for (let i = 0; i < n; i++) {
    const o = i * NCOL;
    const v: string[] = [String(rows[o])];
    for (let q = 1; q < NCOL - 1; q++) v.push(Number(rows[o + q]).toPrecision(7));
    v.push(rows[o + NCOL - 1] ? 'True' : 'False');
    out.push(v.join(','));
  }
  return out.join('\n') + '\n';
}
