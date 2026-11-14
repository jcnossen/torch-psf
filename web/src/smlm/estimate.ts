// SMLM movie -> in-situ vectorial PSF: a port of torch_psf.estimators.InSituEstimator (method 'zernike', refine off;
// defaults: ROI 15 px, spline oversample 2, dz 0.04 um, zpad 5, xy_limit 3; spots for the fit / held out split by a
// checkerboard of 16 px tiles; fit_zernike_knots route K; overlap rejection: after the first pass each training spot is
// re-localized with the first-pass spline and dropped if its Poisson deviance per pixel exceeds 1 + k sqrt(2 / P^2)
// (k = 3), then a second pass with 3/4 of the iterations starts from the first-pass result; held-out check =
// compare_heldout of the start and the fitted spline; z_scale of the fitted knots against the start knots; fraction of
// fitted spots at the offset limit).
import { Detector, extractRois } from '../gpu/detect';
import { SplineLocalizer, splineGrid } from '../gpu/spline_lm';
import { StopToken, Eta, Progress } from '../core/task';
import { openSource, MovieSource } from './movie';
import { fitZernikeKnots, knotsAndJacobian, zScale, checkerboard, KnotGridFull, InsituIter } from './insitu';
import type { InsituForm } from '../ui/protocol';
import { fitFreeKnots } from './refine';

export interface InsituReport {
  zcoef: number[]; zcoef_start: number[]; noll: number[]; iterations: number; partial: boolean; seconds: number;
  n_spots: number; n_train: number; n_test: number; n_rejected: number; deviance_threshold: number | null;
  frames_read: number; frames_total: number;
  heldout: { start: number; fitted: number; diff: number } | null;   // mean NLL per pixel (constant dropped)
  z_scale: number; z_offset_um: number; at_limit: number; history: InsituIter[];
  /** refine=True: free knots regularised towards the vectorial spline (kept only if the held-out NLL improves). */
  refine: { kept: boolean; heldout_refined: number | null; z_scale_vs_vectorial: number; z_offset_um_vs_vectorial: number; outer: number; seconds: number;
    knots: Float32Array | null } | null;
  grid: { zpad: number };
}

export const FREE_KNOT_WARNING = 'free knots can improve the held-out likelihood and the repeat precision while warping the z axis ' +
  '(blinking data do not fix the z scale): check the z scale against a reference';

export async function estimateInsitu(dev: GPUDevice, source: File[] | MovieSource, form: InsituForm, tok: StopToken,
  progress: (p: Progress) => void): Promise<InsituReport> {
  const mv = await openSource(source, progress);
  const P = form.roisize, h = (P - 1) >> 1;
  const maxSpots = 2 * (form.nspots + form.nheldout);
  const chunk = Math.max(1, Math.min(form.chunk, mv.n));
  const nFrames = form.maxFrames > 0 ? Math.min(mv.n, form.maxFrames) : mv.n;
  // 1. spots (torch_psf.spots.extract_spots with max_spots: stop after the chunk that reaches it, keep the first ones)
  const det = new Detector(dev, mv.H, mv.W, chunk, { threshold: form.threshold, border: h + 2, isolation: form.isolation > 0 ? form.isolation : null });
  const roiParts: Float32Array[] = [], cx: number[] = [], cy: number[] = [];
  let n = 0, done = 0;
  const eta = new Eta();
  try {
    for (let f0 = 0; f0 < nFrames && n < maxSpots; f0 += chunk) {
      tok.check();
      if (tok.stopEarly) break;
      const f1 = Math.min(nFrames, f0 + chunk);
      const fr = await mv.read(f0, f1);
      const gain = Math.fround(form.gain), off = Math.fround(form.offset);
      for (let i = 0; i < fr.length; i++) fr[i] = Math.fround(Math.fround(fr[i] - off) * gain);
      const d = await det.detect(fr, f1 - f0, f0);
      roiParts.push(extractRois(fr, mv.H, mv.W, f0, d, P));
      for (let i = 0; i < d.x.length; i++) { cx.push(d.x[i]); cy.push(d.y[i]); }
      n += d.x.length; done = f1;
      progress({ stage: 'spots', done: Math.min(n, maxSpots), total: maxSpots, info: { frames: done }, ...eta.update(Math.min(n, maxSpots), maxSpots) });
    }
  } finally { det.destroy(); }
  if (tok.signal === 'enough') tok.signal = 'run';          // "enough" applied to the spot collection
  n = Math.min(n, maxSpots);
  const P2 = P * P, all = new Float32Array(n * P2);
  let o = 0; for (const r of roiParts) { const k = Math.min(r.length, all.length - o); all.set(r.subarray(0, k), o); o += k; }
  // 2. split: parity of the 16 px tile of the detection pixel (corner + (P - 1) / 2)
  const tr: number[] = [], te: number[] = [];
  for (let i = 0; i < n; i++) {
    if (checkerboard(cx[i], cy[i], form.tile) === 0) { if (tr.length < form.nspots) tr.push(i); }
    else if (te.length < form.nheldout) te.push(i);
  }
  if (tr.length < 50) throw new Error(`only ${tr.length} spots for the fit; lower the threshold or use more frames`);
  const take = (idx: number[]) => { const a = new Float32Array(idx.length * P2); idx.forEach((k, q) => a.set(all.subarray(k * P2, (k + 1) * P2), q * P2)); return a; };
  const dTr = take(tr), dTe = take(te);
  // 3. fit (route K) on the estimate_psf_insitu grid
  const g = splineGrid(P, form.zlo, form.zhi, 2, 0.04, 5, 3.0) as KnotGridFull;
  const cfg = { ...form.optics, roisize: 0, npupil: form.npupil, oversample: form.oversample, zernike_noll: form.noll };
  const reject = form.rejectOverlap ?? true, k = form.devianceK ?? 3.0;
  const it2 = Math.max(Math.floor(form.iters * 3 / 4), 1);
  const total = form.iters + (reject ? it2 : 0);
  const logFit = (offset: number, nRoi: number, stage: string) => (hh: InsituIter) => progress({ stage, done: offset + hh.it + 1, total,
    info: { nll_per_px: hh.nll / (nRoi * P2), zcoef: hh.zcoef.map((v) => v.toFixed(4)).join(' ') }, elapsed_s: hh.seconds });
  const tFit0 = performance.now();
  let fit = await fitZernikeKnots(dev, dTr, tr.length, cfg, form.zcoef, g, form.focus_um, { iters: form.iters, tok, log: logFit(0, tr.length, 'fit') });
  let partial = fit.history.length < form.iters;
  const K0 = fit.K0;
  let nRej = 0, thr: number | null = null, nFit = tr.length, history = fit.history, dFit = dTr;
  if (reject && !tok.stopEarly) {
    // spot_deviance with the first-pass PSF tabulated as a spline (localized afresh: zstarts, 30 iterations)
    const K1a = (await knotsAndJacobian(dev, cfg, fit.zcoef.map(Math.fround), g, form.focus_um)).knots;
    const loc = new SplineLocalizer(dev, g, K1a);
    let dv: Float32Array;
    try { dv = (await loc.localize(dTr, tr.length)).deviance; } finally { loc.destroy(); }
    thr = 1 + k * Math.sqrt(2 / P2);
    const keep: number[] = [];
    for (let i = 0; i < tr.length; i++) if (dv[i] < thr) keep.push(i);
    nRej = tr.length - keep.length; nFit = keep.length;
    const dKeep = new Float32Array(keep.length * P2);
    keep.forEach((i, q) => dKeep.set(dTr.subarray(i * P2, (i + 1) * P2), q * P2));
    dFit = dKeep;
    const fit2 = await fitZernikeKnots(dev, dKeep, keep.length, cfg, fit.zcoef, g, form.focus_um, { iters: it2, tok, log: logFit(form.iters, keep.length, 'fit (overlaps rejected)') });
    partial = partial || fit2.history.length < it2;
    history = [...fit.history, ...fit2.history];
    fit = { ...fit2, K0 };
  }
  // 4. report: at_limit, z_scale, held-out NLL of the start and the fitted spline
  let lim = 0;
  for (let i = 0; i < nFit; i++) if (Math.max(Math.abs(fit.theta[i * 5]), Math.abs(fit.theta[i * 5 + 1])) > g.xmax - 0.06) lim++;
  progress({ stage: 'report', done: 0, total: 2 });
  const K1 = (await knotsAndJacobian(dev, cfg, fit.zcoef.map(Math.fround), g, form.focus_um)).knots;
  const zs = await zScale(dev, K1, fit.K0, g);
  // optional refinement: free knots from the vectorial spline, on the (kept) training spots
  let refine: InsituReport['refine'] = null;
  if (form.refine && !tok.stopEarly) {
    const outer = form.refineOuter ?? 12;
    progress({ stage: 'refine', done: 0, total: outer });
    const rr = await fitFreeKnots(dev, g, K1, K1, dFit, nFit, { outer, tok,
      log: (h) => progress({ stage: 'refine', done: h.it + 1, total: outer, info: { loss: h.loss }, elapsed_s: h.seconds }) });
    const zr = await zScale(dev, rr.knots, K1, g);
    refine = { kept: false, heldout_refined: null, z_scale_vs_vectorial: zr.scale, z_offset_um_vs_vectorial: zr.offset, outer: rr.history.length, seconds: rr.seconds, knots: rr.knots };
  }
  let heldout: InsituReport['heldout'] = null;
  if (te.length) {
    const nllPx = async (K: Float32Array) => {
      const loc = new SplineLocalizer(dev, g, K);
      try { const r = await loc.localize(dTe, te.length); let s = 0; for (const v of r.nll) s += v; return s / te.length / P2; }
      finally { loc.destroy(); }
    };
    const a = await nllPx(fit.K0), b = await nllPx(K1);
    heldout = { start: a, fitted: b, diff: b - a };
    if (refine) { refine.heldout_refined = await nllPx(refine.knots!); refine.kept = refine.heldout_refined < b; }
  }
  if (refine && !refine.kept) refine.knots = null;
  return { zcoef: fit.zcoef, zcoef_start: [...form.zcoef], noll: form.noll, iterations: history.length, partial,
    seconds: (performance.now() - tFit0) / 1000,
    n_spots: n, n_train: tr.length, n_test: te.length, n_rejected: nRej, deviance_threshold: thr, frames_read: done, frames_total: mv.n, heldout,
    z_scale: zs.scale, z_offset_um: zs.offset, at_limit: lim / Math.max(1, nFit), history, refine, grid: { zpad: 5 } };
}
