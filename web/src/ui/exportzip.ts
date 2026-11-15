// psf.zip requests of the web app's two fits (psf/makezip.ts builds the zip in the worker). Bead fit: the vectorial
// model at the export focus over the export z range (emitter heights), at the fit ROI; spline channel = the model
// tabulated on the export spline grid. In-situ fit: the model at the acquisition focus over the emitter z range;
// spline channel = the refined free knots when they were kept, else the model tabulated on the fit's knot grid.
import type { FitForm, FitSummary, InsituForm } from './protocol';
import type { InsituReport } from '../smlm/estimate';
import type { PsfZipRequest } from '../psf/makezip';

export interface ZipOptions { planes: number; k: number; spline: boolean }

export function beadZipRequest(r: FitSummary, f: FitForm, e: { focus_um: number; splineRoi: number; zmin: number; zmax: number },
  o: ZipOptions, provenance: Record<string, unknown>, pdf: Uint8Array, sections: string[]): PsfZipRequest {
  return {
    vectorial: { optics: f.optics, noll: r.noll, zcoef: r.zcoef, focus_um: e.focus_um, npupil: f.npupil, oversample: f.oversample,
      roisize: f.roisize, z_range_um: [e.zmin, e.zmax] },
    spline: o.spline ? { roisize: e.splineRoi, z_range: [e.zmin, e.zmax], S: 2, hz: 0.04, zpad: 3, xmax: 3 } : null,
    camera: { gain_e_per_adu: f.gain, offset_adu: f.offset },
    provenance,
    report: {
      estimator: 'torch-psf web bead fit',
      results: { n_beads: r.beads.length, n_heldout: r.beads.filter((b) => b.heldout).length, n_planes: r.frames_used, frames_total: r.frames_total,
        iterations: r.iterations, heldout_iterations: r.heldout_iterations, converged: r.converged, partial: r.partial,
        nll_train_px: r.nll_train_px, nll_heldout_px: r.nll_heldout_px, dz_scale: r.dz_scale, drift: r.drift },
      settings: { ...f },
      sections,
    },
    zstack: { planes: o.planes, k: o.k }, pdf, title: 'vectorial PSF (bead fit)' + (o.spline ? ', spline channel' : ''),
  };
}

export function insituZipRequest(r: InsituReport, f: InsituForm, o: ZipOptions, provenance: Record<string, unknown>, pdf: Uint8Array, sections: string[]): PsfZipRequest {
  const knots = o.spline && r.refine?.kept ? r.refine.knots : null;
  const { knots: _k, ...refine } = r.refine ?? { knots: null };
  const settings: Record<string, unknown> = { ...f };
  delete settings.knots;
  return {
    vectorial: { optics: f.optics, noll: r.noll, zcoef: r.zcoef, focus_um: f.focus_um, npupil: f.npupil, oversample: f.oversample,
      roisize: f.roisize, z_range_um: [f.zlo, f.zhi] },
    spline: o.spline ? { roisize: f.roisize, z_range: [f.zlo, f.zhi], S: 2, hz: 0.04, zpad: r.grid.zpad, xmax: 3, knots } : null,
    camera: { gain_e_per_adu: f.gain, offset_adu: f.offset, readnoise_e: f.readnoise_e, excess_noise: f.excessNoise ?? 1 },
    provenance: { ...provenance, ...(o.spline ? { spline_knots: knots ? 'refined free knots (kept)' : 'tabulated from the vectorial fit' } : {}) },
    report: {
      estimator: 'torch-psf web in-situ fit',
      results: { n_spots: r.n_spots, n_train: r.n_train, n_test: r.n_test, n_rejected: r.n_rejected, frames_read: r.frames_read, frames_total: r.frames_total,
        iterations: r.iterations, partial: r.partial, seconds: r.seconds, heldout: r.heldout, heldout_chi2_median: r.heldout_chi2_median,
        deviance: r.deviance, z_scale: r.z_scale, z_offset_um: r.z_offset_um, at_limit: r.at_limit, zernike_start: r.zcoef_start,
        refine: r.refine ? refine : null, warnings: r.warnings },
      settings, sections,
    },
    zstack: { planes: o.planes, k: o.k }, pdf, title: 'vectorial PSF (in-situ fit)' + (o.spline ? ', spline channel' : ''),
  };
}
