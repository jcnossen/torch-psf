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
import { fitZernikeKnots, zScale, checkerboard, renderSpline, KnotGridFull, InsituIter } from './insitu';
import { vectorialKnots } from '../gpu/knot_engine';
import type { InsituForm } from '../ui/protocol';
import { fitFreeKnots } from './refine';
import { QualityRange, QualityRoi, qualityBands, median } from '../report/quality';

export interface InsituReport {
  zcoef: number[]; zcoef_start: number[]; noll: number[]; iterations: number; partial: boolean; seconds: number;
  n_spots: number; n_train: number; n_test: number; n_rejected: number; deviance_threshold: number | null;
  frames_read: number; frames_total: number;
  heldout: { start: number; fitted: number; diff: number } | null;   // mean NLL per pixel (constant dropped)
  heldout_chi2_median: number | null;   // held-out spots fitted with the returned PSF: ~1 if the camera model is right
  deviance: { rule: string; median: number; scale: number; threshold: number; rejected_fraction: number } | null;
  frames_requested: number; n_chunks: number; warnings: string[];
  z_scale: number; z_offset_um: number; at_limit: number; history: InsituIter[];
  /** refine=True: free knots regularised towards the vectorial spline (kept only if the held-out NLL improves). */
  refine: { kept: boolean; heldout_refined: number | null; z_scale_vs_vectorial: number; z_offset_um_vs_vectorial: number; outer: number; seconds: number;
    knots: Float32Array | null } | null;
  grid: { zpad: number };
  /** ROI quality range for the report (overlap rejection only): the first-pass deviance of every training spot, the
   *  threshold, and the ~9 ROIs shown with their first-pass fits (the other ROIs stay in the worker). */
  quality: QualityRange | null;
}

export const FREE_KNOT_WARNING = 'free knots can improve the held-out likelihood and the repeat precision while warping the z axis ' +
  '(blinking data do not fix the z scale): check the z scale against a reference';

export async function estimateInsitu(dev: GPUDevice, source: File[] | MovieSource, form: InsituForm, tok: StopToken,
  progress: (p: Progress) => void): Promise<InsituReport> {
  const mv = await openSource(source, progress);
  const P = form.roisize, h = (P - 1) >> 1;
  const maxSpots = 2 * (form.nspots + form.nheldout);
  const chunk = Math.max(1, Math.min(form.sampleChunk ?? 50, mv.n));
  const nFrames = form.maxFrames > 0 ? Math.min(mv.n, form.maxFrames) : mv.n;
  const warnings: string[] = [];
  // 1. spots: torch_psf.spots.extract_spots(sample='spread', chunk=50, max_spots): chunks in bit-reversed order over the
  //    frame range until max_spots, the first max_spots in read order kept, then sorted by frame (stable)
  let starts: number[] = []; for (let f = 0; f < nFrames; f += chunk) starts.push(f);
  if ((form.sample ?? 'spread') === 'spread') { const s0 = starts; starts = spreadOrder(s0.length).map((i) => s0[i]); }
  const det = new Detector(dev, mv.H, mv.W, chunk, { threshold: form.threshold, border: h + 2, isolation: form.isolation > 0 ? form.isolation : null });
  const roiParts: Float32Array[] = [], cx0: number[] = [], cy0: number[] = [], cf: number[] = [];
  let n = 0, done = 0, nChunks = 0;
  const eta = new Eta();
  const F2 = form.excessNoise ?? 1;
  // read-noise variance of the spots (photons^2): readnoise^2 / excess_noise^2, used in every likelihood (torch_psf 4e2bcac)
  const rvar = form.readnoise_e > 0 ? Math.fround(form.readnoise_e * form.readnoise_e / (F2 * F2)) : 0;
  try {
    for (const f0 of starts) {
      if (n >= maxSpots) break;
      tok.check();
      if (tok.stopEarly) break;
      const f1 = Math.min(nFrames, f0 + chunk);
      const fr = await mv.read(f0, f1);
      const gain = Math.fround(form.gain), off = Math.fround(form.offset);
      for (let i = 0; i < fr.length; i++) fr[i] = Math.fround(Math.fround(fr[i] - off) * gain);
      if (F2 !== 1) for (let i = 0; i < fr.length; i++) fr[i] = Math.fround(fr[i] / F2);
      const d = await det.detect(fr, f1 - f0, f0);
      roiParts.push(extractRois(fr, mv.H, mv.W, f0, d, P));
      for (let i = 0; i < d.x.length; i++) { cx0.push(d.x[i]); cy0.push(d.y[i]); cf.push(d.frame[i]); }
      n += d.x.length; done += f1 - f0; nChunks++;
      let vf = -1; for (let i = 0; i < d.frame.length; i++) if (d.frame[i] > vf) vf = d.frame[i];   // the chunk's last frame with spots
      const view = vf < 0 ? undefined : { frame: vf, x: [] as number[], y: [] as number[] };
      if (view) for (let i = 0; i < d.frame.length; i++) if (d.frame[i] === vf) { view.x.push(d.x[i]); view.y.push(d.y[i]); }
      progress({ stage: 'spots', done: Math.min(n, maxSpots), total: maxSpots, info: { frames: done }, view, ...eta.update(Math.min(n, maxSpots), maxSpots) });
    }
  } finally { det.destroy(); }
  if (tok.signal === 'enough') tok.signal = 'run';          // "enough" applied to the spot collection
  n = Math.min(n, maxSpots);
  const P2 = P * P, raw = new Float32Array(n * P2);
  let o = 0; for (const r of roiParts) { const k = Math.min(r.length, raw.length - o); raw.set(r.subarray(0, k), o); o += k; }
  const ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => (cf[a] - cf[b]) || (a - b));   // frame order, stable
  const all = new Float32Array(n * P2);
  ord.forEach((k, i) => all.set(raw.subarray(k * P2, (k + 1) * P2), i * P2));
  const cx = ord.map((k) => cx0[k]), cy = ord.map((k) => cy0[k]);
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
  let fit = await fitZernikeKnots(dev, dTr, tr.length, cfg, form.zcoef, g, form.focus_um, { iters: form.iters, tok, log: logFit(0, tr.length, 'fit'), var: rvar });
  let partial = fit.history.length < form.iters;
  const K0 = fit.K0;
  let nRej = 0, thr: number | null = null, nFit = tr.length, history = fit.history, dFit = dTr;
  let devRep: { rule: string; median: number; scale: number; threshold: number; rejected_fraction: number } | null = null;
  let quality: QualityRange | null = null;
  if (reject && !tok.stopEarly) {
    // spot_deviance with the first-pass PSF tabulated as a spline (localized afresh: zstarts, 30 iterations)
    const K1a = await vectorialKnots(dev, cfg, fit.zcoef.map(Math.fround), g, form.focus_um);
    const loc = new SplineLocalizer(dev, g, K1a);
    let dv: Float32Array, th1: Float32Array;
    try { const r1 = await loc.localize(dTr, tr.length, { var: rvar }); dv = r1.deviance; th1 = r1.theta; } finally { loc.destroy(); }
    // deviance_rule 'median' (default): median + k max(1.4826 MAD, median sqrt(2 / P^2)); 'poisson': 1 + k sqrt(2 / P^2)
    const med = median(dv);
    let scale: number;
    if ((form.devianceRule ?? 'median') === 'poisson') { scale = Math.sqrt(2 / P2); thr = 1 + k * scale; }
    else { scale = Math.max(1.4826 * median(Float32Array.from(dv, (v) => Math.abs(v - med))), med * Math.sqrt(2 / P2)); thr = med + k * scale; }
    devRep = { rule: form.devianceRule ?? 'median', median: med, scale, threshold: thr, rejected_fraction: 0 };
    const keep: number[] = [];
    for (let i = 0; i < tr.length; i++) if (dv[i] < thr) keep.push(i);
    nRej = tr.length - keep.length; nFit = keep.length;
    devRep.rejected_fraction = nRej / tr.length;
    if (devRep.rejected_fraction > 0.25) warnings.push(`${(100 * devRep.rejected_fraction).toFixed(0)} % of the training spots were rejected as overlaps`);
    if (devRep.median > 1.3) warnings.push(`median deviance per px ${devRep.median.toFixed(2)} (1 for Poisson statistics): check the camera gain / offset / excess noise`);
    quality = qualitySpots(dTr, dv, th1, thr, K1a, g, (i) => cf[ord[tr[i]]], devRep, k);
    const dKeep = new Float32Array(keep.length * P2);
    keep.forEach((i, q) => dKeep.set(dTr.subarray(i * P2, (i + 1) * P2), q * P2));
    dFit = dKeep;
    const fit2 = await fitZernikeKnots(dev, dKeep, keep.length, cfg, fit.zcoef, g, form.focus_um, { iters: it2, tok, log: logFit(form.iters, keep.length, 'fit (overlaps rejected)'), var: rvar });
    partial = partial || fit2.history.length < it2;
    history = [...fit.history, ...fit2.history];
    fit = { ...fit2, K0 };
  }
  // 4. report: at_limit, z_scale, held-out NLL of the start and the fitted spline
  let lim = 0;
  for (let i = 0; i < nFit; i++) if (Math.max(Math.abs(fit.theta[i * 5]), Math.abs(fit.theta[i * 5 + 1])) > g.xmax - 0.06) lim++;
  progress({ stage: 'report', done: 0, total: 2 });
  const K1 = await vectorialKnots(dev, cfg, fit.zcoef.map(Math.fround), g, form.focus_um);
  const zs = await zScale(dev, K1, fit.K0, g);
  // optional refinement: free knots from the vectorial spline, on the (kept) training spots
  let refine: InsituReport['refine'] = null;
  if (form.refine && !tok.stopEarly) {
    const outer = form.refineOuter ?? 12;
    progress({ stage: 'refine', done: 0, total: outer });
    const rr = await fitFreeKnots(dev, g, K1, K1, dFit, nFit, { outer, tok, var: rvar,
      log: (h) => progress({ stage: 'refine', done: h.it + 1, total: outer, info: { loss: h.loss }, elapsed_s: h.seconds }) });
    const zr = await zScale(dev, rr.knots, K1, g);
    refine = { kept: false, heldout_refined: null, z_scale_vs_vectorial: zr.scale, z_offset_um_vs_vectorial: zr.offset, outer: rr.history.length, seconds: rr.seconds, knots: rr.knots };
  }
  let heldout: InsituReport['heldout'] = null, heldoutChi2: number | null = null;
  if (te.length) {
    const chi2s: Float32Array[] = [];
    const nllPx = async (K: Float32Array) => {
      const loc = new SplineLocalizer(dev, g, K);
      try { const r = await loc.localize(dTe, te.length, { var: rvar }); chi2s.push(r.chi2); let s = 0; for (const v of r.nll) s += v; return s / te.length / P2; }
      finally { loc.destroy(); }
    };
    const a = await nllPx(fit.K0), b = await nllPx(K1);
    heldout = { start: a, fitted: b, diff: b - a };
    let chosen = chi2s[1];
    if (refine) { refine.heldout_refined = await nllPx(refine.knots!); refine.kept = refine.heldout_refined < b; if (refine.kept) chosen = chi2s[2]; }
    heldoutChi2 = median(chosen);                       // the held-out spots fitted with the returned PSF
    if (!(heldoutChi2 > 0.7 && heldoutChi2 < 1.4)) warnings.push(`held-out median chi2 per px ${heldoutChi2.toFixed(2)} (about 1 expected): check the camera gain / offset / excess noise`);
  }
  return { zcoef: fit.zcoef, zcoef_start: [...form.zcoef], noll: form.noll, iterations: history.length, partial,
    seconds: (performance.now() - tFit0) / 1000,
    n_spots: n, n_train: tr.length, n_test: te.length, n_rejected: nRej, deviance_threshold: thr, frames_read: done, frames_total: mv.n, heldout,
    z_scale: zs.scale, z_offset_um: zs.offset, at_limit: lim / Math.max(1, nFit), history, refine, grid: { zpad: 5 },
    heldout_chi2_median: heldoutChi2, deviance: devRep, frames_requested: nFrames, n_chunks: nChunks, quality,
    warnings: tr.length < form.nspots ? [...warnings, `only ${tr.length} training spots found (${form.nspots} requested)`] : warnings };
}

/** torch_psf.spots.spread_order: 0..n-1 in bit-reversed (van der Corput) order (every prefix spread over the range). */
export function spreadOrder(n: number): number[] {
  if (n <= 1) return Array.from({ length: n }, (_, i) => i);
  const bits = Math.ceil(Math.log2(n)), out: number[] = [];
  for (let i = 0; i < 1 << bits; i++) {
    let r = 0; for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    if (r < n) out.push(r);
  }
  return out;
}

/** The quality range of the training spots: q = first-pass deviance per pixel, cutoff = the rejection threshold; the
 *  shown ROIs (copies) with the model of their first-pass fit (photons x spline + bg). */
function qualitySpots(dTr: Float32Array, dv: Float32Array, theta: Float32Array, thr: number, K: Float32Array, g: KnotGridFull,
  frameOf: (i: number) => number, dr: { rule: string }, k: number): QualityRange {
  const P = g.roisize, P2 = P * P;
  const b = qualityBands(dv, (i) => dv[i] < thr);
  const roi = (i: number): QualityRoi => {
    const t = theta.subarray(i * 5, i * 5 + 5), psf = renderSpline(K, g, t[0], t[1], t[2]);
    const model = Float32Array.from(psf, (v) => t[3] * v + t[4]);
    return { q: dv[i], rank: b.rank[i], data: dTr.slice(i * P2, (i + 1) * P2), model, width: P, height: P,
      photons: t[3], bg: t[4], z: t[2], label: `training spot ${i}, frame ${frameOf(i)}` };
  };
  let nRej = 0; for (const v of dv) if (!(v < thr)) nRej++;
  const rule = dr.rule === 'poisson' ? `1 + ${k} sqrt(2 / P^2)` : `median + ${k} x max(1.4826 MAD, median sqrt(2 / P^2))`;
  return {
    q: Float32Array.from(dv), cutoff: thr, cutoffKind: 'rejection', qLabel: 'Poisson deviance per px (first pass)', unit: 'training spots',
    notes: [`q = Poisson deviance per px of a training spot after its single-emitter fit with the first-pass PSF (the values the estimator used; about 1 when the model explains the spot, overlapping emitters give more). Model images: that fit.`,
      `Rejection (overlap rule ${rule}): q >= ${thr.toFixed(3)}; ${nRej} of ${dv.length} spots rejected, the PSF was refitted without them.`],
    best: b.best.map(roi), lastAccepted: b.lastAccepted.map(roi), firstRejected: b.firstRejected.map(roi),
  };
}
