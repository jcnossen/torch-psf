// psf.json report.sections of a web fit report (docs/PSF_FORMAT.md section names).
import type { FitReportData } from './fitreport';

/** The psf.json report sections (docs/PSF_FORMAT.md names) that a web fit report with these data contains. */
export function reportSections(d: FitReportData): string[] {
  const s = ['summary'];
  if (d.rois) s.push('spots');
  if (d.quality) s.push('quality');
  if (d.histograms?.length) s.push('fit_stats');
  if (d.loss) s.push('convergence');
  if (d.zernike) s.push('aberrations');
  if (d.pupil) s.push('pupil');
  if (d.psf) s.push('psf');
  if (d.examples?.length) s.push('examples');
  if (d.zernikeField?.length) s.push('field');
  return s;
}
