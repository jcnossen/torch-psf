// SMLM movie -> localizations (worker side): torch_psf.pipeline.localize_movie with a CubicSplinePSF built from a
// vectorial PSF (torch_psf CubicSplinePSF.from_vectorial grid defaults). Per chunk of frames: ADU -> photons
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

/** A movie reader: TiffMovie (streamed from files) or an in-memory movie (the demo). read() returns ADU frames. */
export interface MovieSource { n: number; H: number; W: number; read(f0: number, f1: number): Promise<Float32Array>; }

export async function openSource(src: File[] | MovieSource, progress?: (p: Progress) => void): Promise<MovieSource> {
  if (!Array.isArray(src)) return src;
  return TiffMovie.open(src, (msg) => progress?.({ stage: 'header', done: 0, total: 1, info: { msg } }));
}

export async function localizeMovie(device: GPUDevice, source: File[] | MovieSource, form: MovieForm, tok: StopToken,
  progress: (p: Progress) => void, emit: (rows: Float32Array) => void): Promise<MovieSummary> {
  const mv = await openSource(source, progress);
  const P = form.roisize, h = (P - 1) >> 1;
  const nFrames = form.maxFrames > 0 ? Math.min(mv.n, form.maxFrames) : mv.n;
  // the spline PSF (torch_psf CubicSplinePSF.from_vectorial: oversample 2, dz 0.04 um, zpad 2, xy_limit 3)
  const zpad = form.zpad ?? 2;
  const grid = splineGrid(P, form.zlo, form.zhi, 2, 0.04, zpad, 3.0);
  progress({ stage: 'spline', done: 0, total: 1 });
  const knots = form.knots ?? (await buildKnots(device, { ...form.optics, roisize: 0, npupil: form.npupil, oversample: form.oversample, zernike_noll: form.noll },
    form.zcoef, { roisize: P, focus_um: form.focus_um, zmin: form.zlo, zmax: form.zhi, S: grid.S, hz: grid.hz, zpad, xmax: grid.xmax }, () => tok.check())).knots;
  const loc = new SplineLocalizer(device, grid, knots);
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

/** ThunderSTORM csv as torch_psf Dataset.save_csv (extra=True): "id","frame","x [nm]","y [nm]","z [nm]",
 *  "intensity [photon]","offset [photon]","uncertainty [nm]","uncertainty_x [nm]","uncertainty_y [nm]",
 *  "uncertainty_z [nm]", then crlb_photons, crlb_bg, loglik, chi2, at_limit. x [nm] = (x px + 0.5) pixel_nm, frames from
 *  1, z [nm] = the PSF model's z (emitter height above the coverslip) in nm. */
export function locsToCsv(rows: Float32Array, pixel_nm: number): string {
  const n = rows.length / NCOL;
  const f = (v: number, d: number) => v.toFixed(d);
  const g = (v: number, p: number) => (Number.isFinite(v) ? String(Number(v.toPrecision(p))) : String(v));
  const out: string[] = ['"id","frame","x [nm]","y [nm]","z [nm]","intensity [photon]","offset [photon]","uncertainty [nm]",' +
    '"uncertainty_x [nm]","uncertainty_y [nm]","uncertainty_z [nm]","crlb_photons","crlb_bg","loglik","chi2","at_limit"'];
  for (let i = 0; i < n; i++) {
    const o = i * NCOL, r = (k: number) => rows[o + k];
    const ux = r(6) * pixel_nm, uy = r(7) * pixel_nm;
    out.push([i, r(0) + 1, f((r(1) + 0.5) * pixel_nm, 4), f((r(2) + 0.5) * pixel_nm, 4), f(r(3) * 1000, 4), g(r(4), 8), g(r(5), 8),
      f(Math.sqrt(0.5 * (ux * ux + uy * uy)), 5), f(ux, 5), f(uy, 5), f(r(8) * 1000, 5), g(r(9), 9), g(r(10), 9), g(r(11), 9), g(r(12), 9),
      r(13) ? 1 : 0].join(','));
  }
  return out.join('\n') + '\n';
}
