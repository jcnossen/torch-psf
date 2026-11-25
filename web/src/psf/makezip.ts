// psf.zip of a fitted vectorial PSF (bead z-stack or in-situ fit): the spline channel (knots given, or tabulated from
// the vectorial model like CubicSplinePSF.knots_from_vectorial), the z-stack render, preview.png and the zip
// (io/psfzip.ts). Runs where a GPUDevice is (the compute worker; the tests). The PDF comes from the caller
// (report/fromapp.ts, written without a creation date).
import { buildKnots } from './spline';
import { type ZStack, SplineModel, VectorialModel, splineDict, splineZstack, vectorialZstack } from './zstack';
import { previewPng } from '../report/preview';
import { CameraInfo, writePsfZip } from '../io/psfzip';

export interface PsfZipRequest {
  vectorial: VectorialModel;
  field?: import('../dense/movie-field').MovieField;
  /** write channel 0 as a spline (source = the vectorial model): its grid, and the knots (else tabulated) */
  spline?: { roisize: number; z_range: [number, number]; S: number; hz: number; zpad: number; xmax: number; knots?: Float32Array | null } | null;
  camera?: CameraInfo | null;
  provenance?: Record<string, unknown>;
  report?: { estimator?: string; results?: Record<string, unknown>; settings?: Record<string, unknown>; sections?: string[] };
  zstack?: { planes?: number; k?: number };     // default 25 planes, 1 camera px sampling (k = 1)
  pdf: Uint8Array;
  title?: string;
}

/** The spline model of a request: the given knots on its grid, or the vectorial model tabulated on it. */
export async function splineOf(dev: GPUDevice, v: VectorialModel, s: NonNullable<PsfZipRequest['spline']>): Promise<SplineModel> {
  const g = splineDict(s.roisize, s.z_range, s.S, s.hz, s.zpad, s.xmax, v.optics.pixel_um);
  let knots = s.knots ?? null;
  if (!knots) {
    const k = await buildKnots(dev, { ...v.optics, roisize: v.roisize, npupil: v.npupil, oversample: v.oversample, zernike_noll: v.noll, apod: v.apod, saf: v.saf },
      v.zcoef, { roisize: s.roisize, focus_um: v.focus_um, zmin: s.z_range[0], zmax: s.z_range[1], S: s.S, hz: s.hz, zpad: s.zpad, xmax: s.xmax });
    if (k.nz !== g.nz || k.nu !== (2 * g.c + 1) * g.oversample) throw new Error('spline grid mismatch');
    knots = k.knots;
  }
  const n = (2 * g.c + 1) * g.oversample;
  if (knots.length !== g.nz * n * n) throw new Error(`spline knots: ${knots.length} values, the grid needs [${g.nz}, ${n}, ${n}]`);
  return { ...g, knots };
}

export async function buildPsfZip(dev: GPUDevice, r: Omit<PsfZipRequest,'pdf'> & {
  pdf: Uint8Array | ((volume: ZStack) => Promise<Uint8Array>);
}): Promise<{ bytes: Uint8Array; meta: Record<string, unknown>; pdf: Uint8Array }> {
  const planes = Math.round(r.zstack?.planes ?? 25), k = Math.round(r.zstack?.k ?? 1);
  if (!(planes >= 1 && planes <= 401)) throw new Error('z-stack planes: 1 to 401');
  if (!(k >= 1 && k <= 8)) throw new Error('z-stack sampling 1 / k px: k from 1 to 8');
  const spline = r.spline ? await splineOf(dev, r.vectorial, r.spline) : null;
  const zs = spline ? splineZstack(spline, planes, k) : await vectorialZstack(dev, r.vectorial, planes, k);
  const png = await previewPng(zs, ['ch0'], r.title ?? `${spline ? 'spline' : 'vectorial'} PSF`);
  const pdf = typeof r.pdf === 'function' ? await r.pdf(zs) : r.pdf;
  const archive = await writePsfZip({ vectorial: r.vectorial, field:r.field, spline, camera: r.camera, provenance: r.provenance, report: r.report, zstack: zs, png, pdf });
  return { ...archive, pdf };
}
