// Bead fit on the generated stacks (astigmatic and tetrapod-like presets) and the one-bead real crop vs the Python fit
// of the same model (fit_ref.json, made by tests/make_web_reference.py) and, for the generated stacks, the truth. Pass criteria (DESIGN.md): Zernike difference < 0.005 waves rms, focus < 5 nm,
// PSF image relative L2 < 1e-2 over the z range, held-out NLL per px within 0.1 %.
import { test, check, fetchBuf, fetchJson } from './harness';
import { getDevice } from '../src/gpu/device';
import { TiffMovie } from '../src/io/tiff';
import { findBeads, extractRoi, BeadFitter, Stack } from '../src/fit/beads';
import { VecEngine } from '../src/gpu/ops';
import { DEFAULT_NOLL } from '../src/psf/vectorial';

const rms = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / a.length); };

async function psfRelL2(dev: GPUDevice, o: any, R: number, za: ArrayLike<number>, zb: ArrayLike<number>, zr: [number, number]) {
  const eng = new VecEngine(dev, { ...o, roisize: R, zernike_noll: DEFAULT_NOLL }, []);
  const ps = Array.from({ length: 41 }, (_, i) => ({ x: 0, y: 0, ze: o.ze_um, f: zr[0] + (zr[1] - zr[0]) * i / 40 + o.ze_um, I: 1, bg: 0 }));
  const img = async (z: ArrayLike<number>) => { eng.setZernike(z); eng.setImages(ps); await eng.run(ps.length, 'psf'); return eng.readPsf(ps.length); };
  const a = await img(za), b = await img(zb);
  eng.destroy();
  let n = 0, d = 0; for (let i = 0; i < a.length; i++) { d += (a[i] - b[i]) ** 2; n += b[i] ** 2; }
  return Math.sqrt(d / n);
}

async function runCase(name: string, file: string) {
  const { device } = await getDevice();
  const ref = (await fetchJson('fit_ref.json'))[name];
  const meta = ref;
  const t0 = performance.now();
  const mv = await TiffMovie.open([new File([await fetchBuf(file)], file)]);
  const raw = await mv.read(0, mv.n);
  const tRead = performance.now() - t0;
  const ph = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) ph[i] = Math.max(0, Math.fround(Math.fround(raw[i] - meta.offset) * meta.gain));
  const stack: Stack = { name: file, nz: mv.n, H: mv.H, W: mv.W, data: ph };
  const R = ref.roisize;
  const beads = findBeads({ ...stack, data: raw }, R).map((b) => ({ ...b, stack: 0 }));
  const corners = beads.map((b) => [b.y0, b.x0]);
  check(JSON.stringify(corners) === JSON.stringify(ref.corners_yx), `${name}: beads ${JSON.stringify(corners)} != ${JSON.stringify(ref.corners_yx)}`);
  const o = ref.options;
  const fitter = new BeadFitter(device, [stack], beads, beads.map((b) => extractRoi(stack, b, R)), {
    optics: { pixel_um: o.pixel_um, NA: o.NA, wavelength_um: o.wavelength_um, n_imm: o.n_imm, n_med: o.n_med },
    roisize: R, npupil: o.npupil, oversample: o.oversample, sign: o.sign, dz_um: o.dz_um, ze_um: o.ze_um,
    fitDz: o.fit_dz, fitDrift: o.fit_drift, holdout: [ref.holdout], maxIter: 100,
  });
  const res = await fitter.fit();
  fitter.destroy();
  const z = Array.from(res.state.zcoef);
  const dzern = rms(z, ref.zernike_waves);
  const dfocus = Math.max(...ref.beads.map((b: any, k: number) => Math.abs(b.zoff - res.state.zoff[k]))) * 1000;
  const half = Math.abs(o.dz_um) * (mv.n - 1) / 2;   // the stack's z range
  const zr: [number, number] = [-half, half];
  const l2 = await psfRelL2(device, o, R, z, ref.zernike_waves, zr);
  const dho = ref.nll_heldout_px === undefined ? 0 : Math.abs(res.nll_heldout_px! / ref.nll_heldout_px - 1);
  const dtr = Math.abs(res.nll_train_px / ref.nll_train_px - 1);
  const msg = `${name}: ${res.iterations} it + ${res.heldout_iterations} held-out it, ${res.seconds.toFixed(2)} s (read ${tRead.toFixed(0)} ms); ` +
    `zern rms diff ${dzern.toFixed(4)} w, focus diff ${dfocus.toFixed(2)} nm, PSF rel L2 ${l2.toExponential(1)}, ` +
    `NLL/px train ${res.nll_train_px.toFixed(5)} (py ${ref.nll_train_px.toFixed(5)}), rel ${dtr.toExponential(1)}; ` + (ref.nll_heldout_px === undefined ? 'no held-out bead' : `held-out ${res.nll_heldout_px!.toFixed(5)} (py ${ref.nll_heldout_px.toFixed(5)}, rel ${dho.toExponential(1)})`) + `; dz scale ${Math.exp(res.state.logdz).toFixed(4)} (py ${ref.dz_scale.toFixed(4)})`;
  return { res, z, dzern, dfocus, l2, dho, dtr, msg };
}

for (const name of ['astig', 'tetrapod']) {
  test(`fit: generated ${name} stack recovers the Zernikes (vs truth and the Python fit)`, async () => {
    const truthJ = await fetchJson(`${name}.json`);
    const r = await runCase(name, `stack_${name}.tif`);
    const truth = DEFAULT_NOLL.map((j) => truthJ.zernike[String(j)] ?? 0);
    const dt = rms(r.z, truth);
    const msg = r.msg + `; vs truth ${dt.toFixed(4)} w rms`;
    check(r.dzern < 0.005 && r.dfocus < 5 && r.l2 < 1e-2 && r.dho < 1e-3 && dt < 0.005, msg);
    return msg;
  });
}

test('fit: real one-bead crop (PSF_calib zstack2-1) vs the Python fit of the same model', async () => {
  const r = await runCase('real', 'real_bead.tif');
  check(r.dzern < 0.005 && r.dfocus < 5 && r.l2 < 1e-2 && r.dtr < 1e-3, r.msg);
  return r.msg;
});
