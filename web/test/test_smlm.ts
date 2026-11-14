// SMLM-movie mode against torch_psf (reference data: tests/make_web_smlm_reference.py -> test/.cache/smlm/).
import { test, check, fetchBuf, fetchJson } from './harness';
import { getDevice } from '../src/gpu/device';
import { TiffMovie } from '../src/io/tiff';
import { Detector } from '../src/gpu/detect';
import { SplineLocalizer, splineGrid } from '../src/gpu/spline_lm';
import { parseNpz } from '../src/io/npy';
import { buildKnots } from '../src/psf/spline';

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
  const grid = splineGrid(meta.roisize, meta.z_range[0], meta.z_range[1], meta.grid.oversample, meta.grid.dz_um, meta.grid.zpad, meta.grid.xy_limit);
  check(grid.nz === L.nz && grid.nv === L.nv && Math.abs(grid.z0 - L.z0) < 1e-9, `grid ${grid.nz} ${grid.nv} ${grid.z0} vs ${L.nz} ${L.nv} ${L.z0}`);
  const loc = new SplineLocalizer(device, grid, new Float32Array(z.knots.data as Float32Array));
  const N = L.n, rois = new Float32Array(z.rois.data as Float32Array);
  const t0 = performance.now();
  const r = await loc.localize(rois, N);
  const ms = performance.now() - t0;
  loc.destroy();
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

test('smlm: knots from the vectorial model = torch_psf ConvSplinePSF.from_vectorial', async () => {
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
