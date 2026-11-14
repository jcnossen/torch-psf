// SMLM movie -> in-situ vectorial PSF: a port of torch_psf.pipeline.estimate_psf_insitu (defaults: ROI 15 px, spline
// oversample 2, dz 0.04 um, zpad 5, xy_limit 3; spots for the fit / held out split by a checkerboard of 16 px tiles;
// fit_zernike_knots route K; held-out check = compare_heldout of the start and the fitted spline; z_scale of the fitted
// knots against the start knots; fraction of fitted spots at the offset limit).
import { Detector, extractRois } from '../gpu/detect';
import { SplineLocalizer, splineGrid } from '../gpu/spline_lm';
import { StopToken, Eta, Progress } from '../core/task';
import { openSource, MovieSource } from './movie';
import { fitZernikeKnots, knotsAndJacobian, zScale, checkerboard, KnotGridFull, InsituIter } from './insitu';
import type { MovieForm } from '../ui/protocol';

export interface InsituForm extends MovieForm { nspots: number; nheldout: number; iters: number; tile: number; }

export interface InsituReport {
  zcoef: number[]; zcoef_start: number[]; noll: number[]; iterations: number; partial: boolean; seconds: number;
  n_spots: number; n_train: number; n_test: number; frames_read: number; frames_total: number;
  heldout: { start: number; fitted: number; diff: number } | null;   // mean NLL per pixel (constant dropped)
  z_scale: number; z_offset_um: number; at_limit: number; history: InsituIter[];
}

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
  const fit = await fitZernikeKnots(dev, dTr, tr.length, cfg, form.zcoef, g, form.focus_um, {
    iters: form.iters, tok, log: (hh) => progress({ stage: 'fit', done: hh.it + 1, total: form.iters,
      info: { nll_per_px: hh.nll / (tr.length * P2), zcoef: hh.zcoef.map((v) => v.toFixed(4)).join(' ') }, elapsed_s: hh.seconds }) });
  const partial = fit.history.length < form.iters;
  // 4. report: at_limit, z_scale, held-out NLL of the start and the fitted spline
  let lim = 0;
  for (let i = 0; i < tr.length; i++) if (Math.max(Math.abs(fit.theta[i * 5]), Math.abs(fit.theta[i * 5 + 1])) > g.xmax - 0.06) lim++;
  progress({ stage: 'report', done: 0, total: 2 });
  const K1 = (await knotsAndJacobian(dev, cfg, fit.zcoef.map(Math.fround), g, form.focus_um)).knots;
  const zs = await zScale(dev, K1, fit.K0, g);
  let heldout: InsituReport['heldout'] = null;
  if (te.length) {
    const nllPx = async (K: Float32Array) => {
      const loc = new SplineLocalizer(dev, g, K);
      try { const r = await loc.localize(dTe, te.length); let s = 0; for (const v of r.nll) s += v; return s / te.length / P2; }
      finally { loc.destroy(); }
    };
    const a = await nllPx(fit.K0), b = await nllPx(K1);
    heldout = { start: a, fitted: b, diff: b - a };
  }
  return { zcoef: fit.zcoef, zcoef_start: [...form.zcoef], noll: form.noll, iterations: fit.history.length, partial, seconds: fit.seconds,
    n_spots: n, n_train: tr.length, n_test: te.length, frames_read: done, frames_total: mv.n, heldout,
    z_scale: zs.scale, z_offset_um: zs.offset, at_limit: lim / tr.length, history: fit.history };
}
