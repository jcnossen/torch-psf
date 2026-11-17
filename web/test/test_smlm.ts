// SMLM-movie mode against torch_psf (reference data: tests/make_web_smlm_reference.py -> test/.cache/smlm/).
import { test, check, fetchBuf, fetchJson } from './harness';
import { getDevice } from '../src/gpu/device';
import { TiffMovie } from '../src/io/tiff';
import { Detector } from '../src/gpu/detect';
import { SplineLocalizer, splineGrid } from '../src/gpu/spline_lm';
import { parseNpz } from '../src/io/npy';
import { buildKnots } from '../src/psf/spline';
import { localizeMovie, NCOL } from '../src/smlm/movie';
import { StopToken } from '../src/core/task';
import { knotsAndJacobian, fitZernikeKnots } from '../src/smlm/insitu';
import { estimateInsitu } from '../src/smlm/estimate';
import { fitFreeKnots } from '../src/smlm/refine';
import { readLocFile } from '../src/io/locfile';
import { photonTransfer, BlockStats } from '../src/smlm/camera';
import { cpuBlockStats } from './camera_oracle';
import { simulateDemo, ArrayMovie, DEMO } from '../src/smlm/simulate';

test('smlm: spot detection = torch_psf extract_spots / detect_spots', async () => {
  const meta = await fetchJson('smlm/movie.json');
  const ref = await fetchJson('smlm/detect.json');
  const { device } = await getDevice();
  const mv = await TiffMovie.open([new File([await fetchBuf('smlm/movie.tif')], 'movie.tif')]);
  const P = meta.roisize, h = (P - 1) >> 1, C = meta.chunk;
  const det = new Detector(device, mv.H, mv.W, C, { threshold: 6, border: h + 2 });
  const F: number[] = [], X: number[] = [], Y: number[] = [], S: number[] = [];
  const noise: string[] = [];
  let maxNoiseRel = 0, maxMedAbs = 0;
  try {
    for (let t0 = 0, ci = 0; t0 < mv.n; t0 += C, ci++) {
      const t1 = Math.min(mv.n, t0 + C);
      const fr = await mv.read(t0, t1);
      const d = await det.detect(fr, t1 - t0, t0);
      const rc = ref.chunks[ci];
      maxNoiseRel = Math.max(maxNoiseRel, Math.abs(d.noise - rc.noise) / rc.noise);
      maxMedAbs = Math.max(maxMedAbs, Math.abs(d.median - rc.median));
      noise.push(d.noise.toFixed(4));
      for (let i = 0; i < d.frame.length; i++) { F.push(d.frame[i]); X.push(d.x[i]); Y.push(d.y[i]); S.push(d.score[i]); }
    }
  } finally { det.destroy(); }
  const key = (f: number, x: number, y: number) => `${f},${x},${y}`;
  const mine = new Map(F.map((f, i) => [key(f, X[i], Y[i]), S[i]]));
  const theirs = new Map<string, number>(ref.frame.map((f: number, i: number) => [key(f, ref.corner[i][0] + h, ref.corner[i][1] + h), ref.score[i]]));
  const onlyMine = [...mine.keys()].filter((k) => !theirs.has(k));
  const onlyRef = [...theirs.keys()].filter((k) => !mine.has(k));
  let maxScore = 0;
  for (const [k, s] of mine) if (theirs.has(k)) maxScore = Math.max(maxScore, Math.abs(s - theirs.get(k)!) / theirs.get(k)!);
  const sameOrder = onlyMine.length === 0 && onlyRef.length === 0 && F.every((f, i) => key(f, X[i], Y[i]) === key(ref.frame[i], ref.corner[i][0] + h, ref.corner[i][1] + h));
  const msg = `${F.length} spots (torch_psf ${ref.frame.length}); only web ${onlyMine.length}, only torch_psf ${onlyRef.length}; ` +
    `noise rel. diff ${maxNoiseRel.toExponential(1)}, median abs diff ${maxMedAbs.toExponential(1)}, score rel. diff ${maxScore.toExponential(1)}; same order ${sameOrder}`;
  check(onlyMine.length + onlyRef.length <= Math.max(1, 0.002 * ref.frame.length), msg + ` [${onlyMine.slice(0, 4)}] [${onlyRef.slice(0, 4)}]`);
  check(maxNoiseRel < 1e-4 && maxScore < 1e-3, msg);
  return msg;
});

test('smlm: spline localization = torch_psf fit.localize / crlb (same knots)', async () => {
  const L = await fetchJson('smlm/localize.json');
  const meta = await fetchJson('smlm/movie.json');
  const z = await parseNpz(await fetchBuf('smlm/localize.npz'));
  const { device } = await getDevice();
  device.pushErrorScope('validation');
  const grid = splineGrid(meta.roisize, meta.z_range[0], meta.z_range[1], meta.grid.oversample, meta.grid.dz_um, meta.grid.zpad, meta.grid.xy_limit);
  check(grid.nz === L.nz && grid.nv === L.nv && Math.abs(grid.z0 - L.z0) < 1e-9, `grid ${grid.nz} ${grid.nv} ${grid.z0} vs ${L.nz} ${L.nv} ${L.z0}`);
  const loc = new SplineLocalizer(device, grid, new Float32Array(z.knots.data as Float32Array));
  const N = L.n, rois = new Float32Array(z.rois.data as Float32Array);
  const t0 = performance.now();
  const r = await loc.localize(rois, N);
  const ms = performance.now() - t0;
  loc.destroy();
  const validationError = await device.popErrorScope();
  check(!validationError, validationError?.message ?? '');
  const th = z.theta.data as Float32Array, nll = z.nll.data as Float32Array, cr = z.crlb.data as Float32Array, chi = z.tab_chi2.data as Float32Array;
  // per-spot agreement; the optimum is the same, the path differs only by float32 vs float64 solves
  const dx: number[] = [], dz: number[] = [], dn: number[] = [], dc: number[] = [], dchi: number[] = [];
  for (let n = 0; n < N; n++) {
    dx.push(Math.max(Math.abs(r.theta[n * 5] - th[n * 5]), Math.abs(r.theta[n * 5 + 1] - th[n * 5 + 1])));
    dz.push(Math.abs(r.theta[n * 5 + 2] - th[n * 5 + 2]));
    dn.push(r.nll[n] - nll[n]);
    dc.push(Math.abs(r.crlb[n * 5] - cr[n * 5]) / cr[n * 5]);
    dchi.push(Math.abs(r.chi2[n] - chi[n]) / chi[n]);
  }
  const q = (a: number[], p: number) => [...a].sort((x, y) => x - y)[Math.floor(p * (a.length - 1))];
  const msg = `${N} spots in ${ms.toFixed(0)} ms; |dxy| px median ${q(dx, 0.5).toExponential(1)} / 99% ${q(dx, 0.99).toExponential(1)}; ` +
    `|dz| um median ${q(dz, 0.5).toExponential(1)} / 99% ${q(dz, 0.99).toExponential(1)}; nll web - torch: median ${q(dn, 0.5).toExponential(1)}, ` +
    `min ${q(dn, 0).toExponential(1)}, max ${q(dn, 1).toExponential(1)}; crlb_x rel. 99% ${q(dc, 0.99).toExponential(1)}; chi2 rel. 99% ${q(dchi, 0.99).toExponential(1)}`;
  // float32 sums / solves on the GPU vs float64 solves in torch: tolerances well below the CRLB (~0.1 px, ~30 nm)
  check(q(dx, 0.99) < 0.01 && q(dz, 0.99) < 0.003 && q(dc, 0.99) < 0.01 && q(dchi, 0.99) < 0.01 && Math.abs(q(dn, 1)) < 0.05, msg);
  return msg;
});

test('smlm: knots from the vectorial model = torch_psf CubicSplinePSF.from_vectorial', async () => {
  const meta = await fetchJson('smlm/movie.json');
  const z = await parseNpz(await fetchBuf('smlm/localize.npz'));
  const { device } = await getDevice();
  const zc = meta.noll.map((j: number) => meta.zernike[String(j)] ?? 0);
  const k = await buildKnots(device, { ...meta.optics, roisize: 0, npupil: meta.npupil, oversample: meta.oversample, zernike_noll: meta.noll }, zc,
    { roisize: meta.roisize, focus_um: meta.focus_um, zmin: meta.z_range[0], zmax: meta.z_range[1], S: meta.grid.oversample, hz: meta.grid.dz_um, zpad: meta.grid.zpad, xmax: meta.grid.xy_limit });
  const ref = z.knots.data as Float32Array;
  check(k.knots.length === ref.length, `size ${k.knots.length} vs ${ref.length}`);
  let m = 0, d = 0; for (let i = 0; i < ref.length; i++) { m = Math.max(m, Math.abs(ref[i])); d = Math.max(d, Math.abs(k.knots[i] - ref[i])); }
  const msg = `${k.nz} x ${k.nv} x ${k.nu} knots; max |web - torch_psf| / max ${(d / m).toExponential(2)}`;
  check(d / m < 1e-4, msg);
  return msg;
});

test('smlm: whole movie pipeline (ADU -> detect -> knots -> localize) = torch_psf table', async () => {
  const meta = await fetchJson('smlm/movie.json');
  const z = await parseNpz(await fetchBuf('smlm/localize.npz'));
  const { device } = await getDevice();
  const files = [new File([await fetchBuf('smlm/movie.tif')], 'movie.tif')];
  const chunks: Float32Array[] = [];
  const s = await localizeMovie(device, files, {
    optics: meta.optics, gain: 1, offset: 0, readnoise_e: 0, focus_um: meta.focus_um, zlo: meta.z_range[0], zhi: meta.z_range[1],
    roisize: meta.roisize, threshold: 6, isolation: 0, chunk: meta.chunk, maxFrames: 0, npupil: meta.npupil, oversample: meta.oversample,
    noll: meta.noll, zcoef: meta.noll.map((j: number) => meta.zernike[String(j)] ?? 0),
  }, new StopToken(), () => {}, (r) => chunks.push(r));
  const rows = new Float32Array(s.n_locs * NCOL); let o = 0; for (const c of chunks) { rows.set(c, o); o += c.length; }
  const n = (z.tab_x.data as Float32Array).length;
  const col = (k: string) => z['tab_' + k].data as ArrayLike<number>;
  let dxy = 0, dzz = 0, frameBad = 0;
  const dx: number[] = [];
  for (let i = 0; i < n; i++) {
    if (rows[i * NCOL] !== Number(col('frame')[i])) frameBad++;
    const d = Math.max(Math.abs(rows[i * NCOL + 1] - col('x')[i]), Math.abs(rows[i * NCOL + 2] - col('y')[i]));
    dx.push(d); dxy = Math.max(dxy, d); dzz = Math.max(dzz, Math.abs(rows[i * NCOL + 3] - col('z')[i]));
  }
  dx.sort((a, b) => a - b);
  const msg = `${s.n_locs} localizations in ${s.seconds.toFixed(2)} s; first ${n} rows vs torch_psf: frames differ ${frameBad}, ` +
    `|dxy| median ${dx[n >> 1].toExponential(1)} px, 99% ${dx[Math.floor(0.99 * (n - 1))].toExponential(1)}, max ${dxy.toExponential(1)}; max |dz| ${dzz.toExponential(1)} um`;
  check(s.n_locs === 718 && frameBad === 0 && dx[Math.floor(0.99 * (n - 1))] < 0.01, msg);
  return msg;
});

test('insitu: knot gradient and Zernike VJP = torch_psf knot_gradient / vjp_vectorial', async () => {
  const meta = await fetchJson('smlm/insitu.json');
  const z = await parseNpz(await fetchBuf('smlm/insitu_grad.npz'));
  const { device } = await getDevice();
  const g = splineGrid(15, 0, 0.8, 2, 0.04, 5, 3.0);
  const Kref = z.knots.data as Float32Array, N = z.rois.shape[0];
  // knots and their Zernike derivatives from the web vectorial engine (start PSF: Noll 6 = 0.07)
  const cfg = { ...meta.optics, roisize: 0, npupil: 64, oversample: 3, zernike_noll: meta.noll };
  const zc0 = meta.noll.map((j: number) => meta.start_zernike[String(j)] ?? 0);
  const { knots, dK } = await knotsAndJacobian(device, cfg, zc0, g, meta.focus_um);
  let km = 0, kd = 0; for (let i = 0; i < Kref.length; i++) { km = Math.max(km, Math.abs(Kref[i])); kd = Math.max(kd, Math.abs(knots[i] - Kref[i])); }
  // knot gradient at torch_psf's theta on torch_psf's knots
  const loc = new SplineLocalizer(device, g, new Float32Array(Kref));
  const { gK, nll } = await loc.knotGradient(new Float32Array(z.rois.data as Float32Array), N, new Float32Array(z.theta.data as Float32Array));
  loc.destroy();
  const gr = z.gK.data as Float32Array;
  let gm = 0, gd = 0, dot = 0, n1 = 0, n2 = 0;
  for (let i = 0; i < gr.length; i++) { gm = Math.max(gm, Math.abs(gr[i])); gd = Math.max(gd, Math.abs(gK[i] - gr[i])); dot += gK[i] * gr[i]; n1 += gK[i] ** 2; n2 += gr[i] ** 2; }
  // VJP with torch_psf's gK and the web knot derivatives
  const nK = gr.length, gzr = z.gz.data as Float32Array;
  const gz = meta.noll.map((_: number, q: number) => { let s = 0; for (let k = 0; k < nK; k++) s += gr[k] * dK[q * nK + k]; return s; });
  let zm = 0, zd = 0; gzr.forEach((v, q) => { zm = Math.max(zm, Math.abs(v)); zd = Math.max(zd, Math.abs(gz[q] - v)); });
  const nllRef = Number((z.nll.data as Float64Array)[0]);
  const msg = `${N} ROIs; knots rel. ${(kd / km).toExponential(1)}; knot gradient max rel. ${(gd / gm).toExponential(1)}, cosine ${(dot / Math.sqrt(n1 * n2)).toFixed(7)}; ` +
    `NLL ${nll.toFixed(0)} vs ${nllRef.toFixed(0)} (rel. ${(Math.abs(nll - nllRef) / Math.abs(nllRef)).toExponential(1)}); Zernike gradient max rel. ${(zd / zm).toExponential(1)}`;
  check(kd / km < 1e-4 && gd / gm < 1e-3 && zd / zm < 1e-3, msg);
  return msg;
});

test('insitu: fit_zernike_knots on the torch_psf training ROIs (40 iterations) recovers the same Zernikes', async () => {
  const meta = await fetchJson('smlm/insitu.json');
  const z = await parseNpz(await fetchBuf('smlm/insitu_grad.npz'));
  const { device } = await getDevice();
  const g = splineGrid(15, 0, 0.8, 2, 0.04, 5, 3.0);
  const cfg = { ...meta.optics, roisize: 0, npupil: 64, oversample: 3, zernike_noll: meta.noll };
  const zc0 = meta.noll.map((j: number) => meta.start_zernike[String(j)] ?? 0);
  const N = z.rois.shape[0];
  const r = await fitZernikeKnots(device, new Float32Array(z.rois.data as Float32Array), N, cfg, zc0, g, meta.focus_um, { iters: meta.iters });
  const ref = await fetchJson('smlm/insitu_fit.json');
  const fin: number[] = ref.final_zernike, tru = meta.noll.map((j: number) => meta.truth_zernike[String(j)] ?? 0);
  const rms = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
  // the first Adam steps are sign(gradient) x lr, so the history must agree closely at the start
  const h1 = r.history[3].zcoef, p1: number[] = ref.history[3].zcoef;
  const msg = `${N} ROIs, ${r.history.length} iterations in ${r.seconds.toFixed(1)} s; zcoef rms vs torch_psf after 3 steps ${rms(h1, p1).toExponential(1)}, ` +
    `final ${rms(r.zcoef, fin).toExponential(1)} w; vs truth: web ${rms(r.zcoef, tru).toFixed(4)}, torch_psf ${rms(fin, tru).toFixed(4)} w; ` +
    `web Noll 5/6/11: ${[5, 6, 11].map((j) => r.zcoef[meta.noll.indexOf(j)].toFixed(4)).join(' / ')}`;
  check(rms(r.zcoef, fin) < 2e-3 && rms(h1, p1) < 1e-4, msg);
  return msg;
});

test('insitu: whole estimate (movie -> spots -> split -> fit -> overlap rejection -> 2nd pass -> held-out, z scale) = torch_psf InSituEstimator', async () => {
  const meta = await fetchJson('smlm/insitu.json');
  const { device } = await getDevice();
  const files = [new File([await fetchBuf('smlm/insitu_movie.tif')], 'insitu_movie.tif')];
  const r = await estimateInsitu(device, files, {
    optics: meta.optics, gain: meta.params.gain_e_per_adu, offset: meta.params.offset_adu, readnoise_e: meta.params.readnoise_e, focus_um: meta.focus_um,
    zlo: meta.z_range[0], zhi: meta.z_range[1], roisize: meta.roisize, threshold: 6, isolation: 0, chunk: 500, maxFrames: 0,
    npupil: 64, oversample: 3, noll: meta.noll, zcoef: meta.noll.map((j: number) => meta.start_zernike[String(j)] ?? 0),
    nspots: meta.nspots, nheldout: meta.nheldout, iters: meta.iters, tile: 16,
  }, new StopToken(), () => {});
  const rp = meta.report, fin: number[] = meta.final_zernike;
  const rms = Math.sqrt(r.zcoef.reduce((s, v, i) => s + (v - fin[i]) ** 2, 0) / fin.length);
  const hp = rp.heldout ? rp.heldout.vectorial.nll_per_px_vs_first : NaN;
  const msg = `spots ${r.n_spots} (torch_psf ${rp.n_spots}), train ${r.n_train} (${rp.n_train}), test ${r.n_test} (${rp.n_test}), rejected ${r.n_rejected} (${rp.n_rejected}); ` +
    `zcoef rms vs torch_psf ${rms.toExponential(1)} w; held-out NLL/px fitted - start ${r.heldout?.diff.toFixed(5)} (${hp.toFixed(5)}); ` +
    `z scale ${r.z_scale.toFixed(4)} (${rp.z_scale.toFixed(4)}); at limit ${r.at_limit.toFixed(4)} (${rp.at_limit.toFixed(4)}); ` +
    `deviance threshold ${r.deviance?.threshold.toFixed(4)} (${rp.deviance.threshold.toFixed(4)}); held-out chi2 median ${r.heldout_chi2_median?.toFixed(4)} (${rp.heldout_chi2_median.toFixed(4)}); ` +
    `frames read ${r.frames_read} (${rp.frames_read}); ${r.seconds.toFixed(1)} s`;
  check(r.n_spots === rp.n_spots && r.n_train === rp.n_train && r.frames_read === rp.frames_read && Math.abs(r.n_rejected - rp.n_rejected) <= 3 &&
    Math.abs(r.deviance!.threshold - rp.deviance.threshold) < 1e-3 && Math.abs(r.heldout_chi2_median! - rp.heldout_chi2_median) < 2e-3 && rms < 2e-3 && Math.abs((r.heldout?.diff ?? 0) - hp) < 2e-3 && Math.abs(r.z_scale - rp.z_scale) < 0.01, msg);
  return msg;
});

test('insitu: spot deviance per pixel = torch_psf estimation.spot_deviance (start spline)', async () => {
  const z = await parseNpz(await fetchBuf('smlm/insitu_grad.npz'));
  const dz = await parseNpz(await fetchBuf('smlm/insitu_deviance.npz'));
  const { device } = await getDevice();
  const g = splineGrid(15, 0, 0.8, 2, 0.04, 5, 3.0);
  const loc = new SplineLocalizer(device, g, new Float32Array(z.knots.data as Float32Array));
  const N = z.rois.shape[0];
  const r = await loc.localize(new Float32Array(z.rois.data as Float32Array), N);
  loc.destroy();
  const ref = dz.deviance.data as Float32Array, thr = 1 + 3 * Math.sqrt(2 / 225);
  let md = 0, flip = 0;
  for (let i = 0; i < N; i++) { md = Math.max(md, Math.abs(r.deviance[i] - ref[i])); if ((r.deviance[i] > thr) !== (ref[i] > thr)) flip++; }
  const msg = `${N} spots; max |deviance web - torch_psf| ${md.toExponential(1)}; spots on different sides of the 1.283 threshold: ${flip}`;
  check(md < 1e-3 && flip <= 2, msg);
  return msg;
});

test('insitu: free-knot refinement (3 outer L-BFGS iterations) = torch_psf fit_free_knots', async () => {
  const z = await parseNpz(await fetchBuf('smlm/insitu_grad.npz'));
  const zr = await parseNpz(await fetchBuf('smlm/insitu_refine.npz'));
  const meta = await fetchJson('smlm/insitu_refine.json');
  const { device } = await getDevice();
  const g = splineGrid(15, 0, 0.8, 2, 0.04, 5, 3.0) as any;
  const Kv = zr.K_vec.data as Float32Array, Kref = zr.knots.data as Float32Array;
  const r = await fitFreeKnots(device, g, Kv, Kv, new Float32Array(z.rois.data as Float32Array), z.rois.shape[0], { lamRef: meta.lam_ref, lamS: meta.lam_s, outer: meta.outer });
  let m = 0, dd = 0, dv = 0, ddv = 0;
  for (let i = 0; i < Kref.length; i++) { m = Math.max(m, Math.abs(Kref[i])); dd = Math.max(dd, Math.abs(r.knots[i] - Kref[i])); dv = Math.max(dv, Math.abs(Kref[i] - Kv[i])); }
  const lw = r.history.map((h) => h.loss), lp = meta.history.map((h: any) => h.loss);
  const msg = `${z.rois.shape[0]} ROIs, ${r.history.length} outer iterations in ${r.seconds.toFixed(1)} s; loss per outer web ${lw.map((v) => v.toFixed(0)).join(' / ')} vs torch_psf ${lp.map((v: number) => v.toFixed(0)).join(' / ')}; ` +
    `knots: max |web - torch_psf| ${(dd / m).toExponential(1)} of the peak, while the refinement moved them by ${(dv / m).toExponential(1)}`;
  check(dd < 0.1 * dv && Math.abs(lw[lw.length - 1] - lp[lp.length - 1]) < 50, msg);
  return msg;
});

test('insitu: whole estimate with refine=True = torch_psf InSituEstimator(refine=True)', async () => {
  const meta = await fetchJson('smlm/insitu.json');
  const ref = await fetchJson('smlm/insitu_refine_est.json');
  const { device } = await getDevice();
  const files = [new File([await fetchBuf('smlm/insitu_movie.tif')], 'insitu_movie.tif')];
  const r = await estimateInsitu(device, files, {
    optics: meta.optics, gain: meta.params.gain_e_per_adu, offset: meta.params.offset_adu, readnoise_e: meta.params.readnoise_e, focus_um: meta.focus_um,
    zlo: meta.z_range[0], zhi: meta.z_range[1], roisize: meta.roisize, threshold: 6, isolation: 0, chunk: 500, maxFrames: 0,
    npupil: 64, oversample: 3, noll: meta.noll, zcoef: meta.noll.map((j: number) => meta.start_zernike[String(j)] ?? 0),
    nspots: meta.nspots, nheldout: meta.nheldout, iters: meta.iters, tile: 16, refine: true, refineOuter: ref.refine_outer,
  }, new StopToken(), () => {});
  const f = r.refine!, h = ref.heldout;
  const dWeb = f.heldout_refined! - (r.heldout!.start + r.heldout!.diff), dPy = h.refined.nll_per_px - h.vectorial.nll_per_px;
  const msg = `kept web ${f.kept} / torch_psf ${ref.refine.kept}; held-out NLL/px refined - vectorial web ${dWeb.toExponential(2)} vs ${dPy.toExponential(2)}; ` +
    `z scale vs vectorial ${f.z_scale_vs_vectorial.toFixed(4)} vs ${ref.refine.z_scale_vs_vectorial.toFixed(4)}; ${f.outer} outer, ${f.seconds.toFixed(1)} s`;
  check(f.kept === ref.refine.kept && Math.abs(dWeb - dPy) < 1e-4 && Math.abs(f.z_scale_vs_vectorial - ref.refine.z_scale_vs_vectorial) < 0.01, msg);
  return msg;
});

test('viewer: localization files written by torch_psf (Dataset HDF5, Picasso HDF5 + yaml, ThunderSTORM csv)', async () => {
  const ref = await parseNpz(await fetchBuf('smlm/locs_ref.npz'));
  const pos = ref.pos.data as Float64Array, crlb = ref.crlb.data as Float32Array, fr = ref.frame.data as Int32Array, N = fr.length;
  const file = async (name: string) => new File([await fetchBuf('smlm/' + name)], name);
  const out: string[] = [];
  for (const [label, files, zTol] of [['dataset', ['locs_dataset.hdf5'], 1e-4], ['picasso', ['locs_picasso.hdf5', 'locs_picasso.yaml'], 1e-3], ['csv', ['locs_ts.csv'], 1e-3]] as [string, string[], number][]) {
    const L = await readLocFile(await Promise.all(files.map(file)), 108);
    check(L.n === N, `${label}: ${L.n} localizations, expected ${N}`);
    let dxy = 0, dz = 0, dc = 0, df = 0;
    for (let i = 0; i < N; i++) {
      const o = i * NCOL;
      dxy = Math.max(dxy, Math.abs(L.rows[o + 1] - pos[i * 3]), Math.abs(L.rows[o + 2] - pos[i * 3 + 1]));
      dz = Math.max(dz, Math.abs(L.rows[o + 3] - pos[i * 3 + 2] * 0.108));   // pos z in px of 108 nm -> um
      dc = Math.max(dc, Math.abs(L.rows[o + 6] - crlb[i * 3]));
      df = Math.max(df, Math.abs(L.rows[o] - fr[i]));
    }
    out.push(`${label} (${L.format}, ${L.W} x ${L.H} px, ${L.pixel_nm} nm): |dxy| ${dxy.toExponential(1)} px, |dz| ${dz.toExponential(1)} um, |dcrlb| ${dc.toExponential(1)} px, |dframe| ${df}`);
    check(dxy < 1e-3 && dz < zTol && dc < 1e-3 && df === 0 && L.pixel_nm === 108, out[out.length - 1]);
  }
  return out.join('; ');
});

test('camera: photon transfer (gain / offset / read noise from the movie) = torch_psf.camera.photon_transfer', async () => {
  const ref = await fetchJson('smlm/photon_transfer.json');
  const { device } = await getDevice();
  const buf = await fetchBuf('smlm/pt_movie.tif');
  const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);
  const out: string[] = [];
  let worst = 0, worstBlock = 0, spotsOff = 0, pxOff = 0;
  const worstField: Record<string, number> = {};
  for (const c of ref.cases) {
    const kw = c.kwargs;
    const r = await photonTransfer(device, [new File([buf], 'pt_movie.tif')], { nblocks: kw.nblocks, block: kw.block, maskRadius: kw.mask_radius,
      threshold: kw.threshold, offsetAdu: kw.offset_adu, excessNoise: kw.excess_noise, nbins: kw.nbins, frames: kw.frames });
    const P = c.report, C = c.camera;
    check(r.per_block.length === P.per_block.length && r.block_used === r.block_requested,
      `${JSON.stringify(kw)}: ${r.per_block.length} blocks (torch_psf ${P.per_block.length}), block ${r.block_used} / ${r.block_requested}`);
    const d = { gain: rel(r.gain_e_per_adu, C.gain_e_per_adu), offset: rel(r.offset_adu, C.offset_adu), slope: rel(r.slope_adu, P.slope_adu),
      readnoise: C.readnoise_e > 0 ? rel(r.readnoise_e, C.readnoise_e) : Math.abs(r.readnoise_e), spread: Math.abs(r.slope_spread - P.slope_spread) };
    worst = Math.max(worst, d.gain, d.offset, d.slope, d.readnoise);
    r.per_block.forEach((b, i) => {
      const q = P.per_block[i];
      check(b.frames[0] === q.frames[0] && b.frames[1] === q.frames[1], `block ${i} frames ${b.frames} vs ${q.frames}`);
      const db: Record<string, number> = { slope: rel(b.slope, q.slope), offset: rel(b.offset_adu, q.offset_adu), median_mean: rel(b.median_mean_adu, q.median_mean_adu),
        kept: rel(b.kept_fraction, q.kept_fraction), intercept: q.intercept_adu2 > 0 ? rel(b.intercept_adu2, q.intercept_adu2) : Math.abs(b.intercept_adu2) };
      for (const [f, v] of Object.entries(db)) { worstField[f] = Math.max(worstField[f] ?? 0, v); worstBlock = Math.max(worstBlock, v); }
      spotsOff += Math.abs(b.n_spots - c.n_spots[i]); pxOff += Math.abs(b.n_px - q.n_px);
    });
    out.push(`${JSON.stringify(kw)}: gain ${r.gain_e_per_adu.toFixed(6)} (torch_psf ${C.gain_e_per_adu.toFixed(6)}), offset ${r.offset_adu.toFixed(4)} ` +
      `(${C.offset_adu.toFixed(4)}), read noise ${r.readnoise_e.toFixed(5)} (${C.readnoise_e.toFixed(5)}) e-, rel. diff gain ${d.gain.toExponential(1)} ` +
      `offset ${d.offset.toExponential(1)} slope ${d.slope.toExponential(1)} readnoise ${d.readnoise.toExponential(1)}, ${r.seconds.toFixed(2)} s`);
  }
  const msg = out.join('; ') + `; per block: max rel. diff ${Object.entries(worstField).map(([f, v]) => `${f} ${v.toExponential(1)}`).join(', ')}, |n_spots diff| ${spotsOff}, |n_px diff| ${pxOff}`;
  // float32 per-pixel sums (torch on the GPU: other summation order) and float64 fits: ~1e-7 expected
  check(worst < 1e-5 && worstBlock < 1e-5 && spotsOff === 0 && pxOff === 0, msg);
  return msg;
});

test('camera: photon transfer on the demo movie (600 x 128 x 128), runtime', async () => {
  // info only: the demo background is uniform (20 photons), so the pixel means differ by noise and spots only and the
  // variance-vs-mean line has no lever arm: the gain is not determined by this movie (torch_psf gives the same kind of
  // number on such a movie). The comparison with torch_psf is the test above.
  const { device } = await getDevice();
  const d = await simulateDemo(device);
  const mv = new ArrayMovie(d.adu, d.T, d.H, d.W);
  const a = await photonTransfer(device, mv, {});
  const b = await photonTransfer(device, mv, { offsetAdu: DEMO.offset });
  const msg = `block ${a.block_used}, ${a.per_block.length} block(s), ${a.per_block[0].n_spots} spots; gain ${a.gain_e_per_adu.toFixed(4)} e-/ADU, offset ${a.offset_adu.toFixed(2)} ADU ` +
    `(${a.seconds.toFixed(2)} s); offset given: gain ${b.gain_e_per_adu.toFixed(4)}, read noise ${b.readnoise_e.toFixed(3)} e- (${b.seconds.toFixed(2)} s)`;
  check(a.block_used === 400 && a.per_block.length === 1 && Number.isFinite(a.gain_e_per_adu), msg);
  return msg;
});

test('camera: per-pixel statistics on the GPU (BlockStats) = the float64 CPU reference', async () => {
  // integer ADU (pt_movie, the demo): the GPU sums are exact, so mu / var / n must agree to the bit; non-integer data
  // (pt_movie x 0.37 + 0.1): compensated float32 sums, differences at the float32 rounding level
  const { device } = await getDevice();
  const pt = await TiffMovie.open([new File([await fetchBuf('smlm/pt_movie.tif')], 'pt_movie.tif')]);
  const demo = await simulateDemo(device);
  const dm = new ArrayMovie(demo.adu, demo.T, demo.H, demo.W);
  const out: string[] = [];
  let bad = 0;
  for (const [name, mv, f0, T, scale] of [['pt_movie', pt, 0, 400, 1], ['pt_movie', pt, 800, 400, 1], ['demo', dm, 0, 400, 1], ['pt_movie x 0.37 + 0.1', pt, 400, 400, 0.37]] as const) {
    const x = await mv.read(f0, f0 + T);
    if (scale !== 1) for (let i = 0; i < x.length; i++) x[i] = Math.fround(Math.fround(x[i] * scale) + 0.1);
    const det = new Detector(device, mv.H, mv.W, T, { threshold: 4, border: 0, isolation: null });
    const st = new BlockStats(device, mv.H, mv.W, T);
    try {
      const d = await det.detect(x, T, 0);
      const t0 = performance.now();
      const g = await st.run(det.input, x.subarray(0, mv.H * mv.W), T, d, 4);
      const ms = performance.now() - t0;
      const c = cpuBlockStats(x, T, mv.H, mv.W, d, 4);
      let dmu = 0, dv = 0, nmu = 0, nv = 0, nn = 0;
      for (let p = 0; p < mv.H * mv.W; p++) {
        if (g.mu[p] !== c.mu[p]) { nmu++; dmu = Math.max(dmu, Math.abs(g.mu[p] - c.mu[p]) / Math.abs(c.mu[p])); }
        if (g.var[p] !== c.var[p]) { nv++; dv = Math.max(dv, Math.abs(g.var[p] - c.var[p]) / Math.abs(c.var[p])); }
        if (g.n[p] !== c.n[p]) nn++;
      }
      if (scale === 1) bad += nmu + nv + nn + (g.kept_fraction !== c.kept_fraction ? 1 : 0);
      else if (nn > 2 || dmu > 1e-6 || dv > 1e-5) bad++;
      out.push(`${name} [${f0}, ${f0 + T}) ${mv.H}x${mv.W}: mu differ ${nmu} px (max rel ${dmu.toExponential(1)}), var ${nv} px (${dv.toExponential(1)}), n ${nn} px, kept ${g.kept_fraction} vs ${c.kept_fraction}; GPU ${ms.toFixed(0)} ms`);
    } finally { det.destroy(); st.destroy(); }
  }
  const msg = out.join('; ');
  check(bad === 0, msg);
  return msg;
});
