import { test, check } from './harness';
import { getDevice } from '../src/gpu/device';
import { VecEngine } from '../src/gpu/ops';
import { splineGrid } from '../src/gpu/spline_lm';
import { fitPupilField, localizePupilField } from '../src/smlm/field-fit';
import { natPolynomial, fieldPupil, fieldBlock } from '../src/smlm/field-model';
import { buildPsfZip } from '../src/psf/makezip';
import { readPsfFile } from '../src/io/psfzip';
import { StopToken } from '../src/core/task';
import ref from './nat_reference.json';
import { estimateInsitu } from '../src/smlm/estimate';

test('NAT: coefficients, off-axis pixels and alternating fit vs PyTorch', async () => {
  const { device } = await getDevice();
  const cfg = { ...ref.optics, roisize: 7, npupil: 16, oversample: 2, zernike_noll: ref.noll };
  const truth = { method: 'nat' as const, parameters: ref.gamma, norm: { cx: 48, cy: 48, sx: 48, sy: 48 }, tile: 32 };
  ref.columns.forEach((expected, k) => {
    const actual = natPolynomial(ref.gamma.map((_, i) => Number(k === i)));
    expected.forEach((row, i) => row.forEach((v, j) => check(Math.abs(actual[i][j] - v) < 1e-12, 'gamma ' + k)));
  });
  const eng = new VecEngine(device, cfg, []);
  let pixels = 0;
  try {
    for (let i = 0; i < ref.xy.length; i += 4) {
      const [fx, fy] = ref.xy[i];
      eng.setZernike(fieldPupil(truth, ref.noll, ref.base, fx, fy));
      const pos = ref.pos.slice(i, i + 4).map(([x, y, ze]) => ({ x, y, ze, f: .1, I: 1, bg: 0 }));
      const out = await eng.psfAndJacobian(pos);
      pos.forEach((_, k) => {
        const image = ref.pixels[i + k], peak = Math.max(...image);
        image.forEach((v, j) => { pixels = Math.max(pixels, Math.abs(v - out[k * 49 + j]) / peak); });
      });
    }
  } finally { eng.destroy(); }
  check(pixels < 1e-4, 'relative pixel error ' + pixels);
  const fit = await fitPupilField(device, cfg, ref.base, splineGrid(7, 0, .2, 2, .04, 5, 3), .1, Float32Array.from(ref.data),
    ref.xy.map(p => p[0]), ref.xy.map(p => p[1]), 97, 97, 2, new StopToken(), () => { }, 3);
  const error = Math.max(...fit.parameters.map((v, i) => Math.abs(v - ref.expected[i])));
  check(error < 3e-4, 'max gamma difference (waves) ' + error + '; actual ' + fit.parameters + ' expected ' + ref.expected);
  const grid = splineGrid(7, 0, .2, 2, .04, 5, 3), data = Float32Array.from(ref.data);
  const x = ref.xy.map(p => p[0]), y = ref.xy.map(p => p[1]);
  const independent = await fitPupilField(device, cfg, ref.base, grid, .1, data, x, y, 97, 97, 2, new StopToken(), () => { }, 3, 'independent');
  const freeError = Math.max(...independent.parameters.map((v, i) => Math.abs(v - ref.independent[i])));
  check(freeError < 3e-4, 'independent field coefficient error ' + freeError);
  const cache = new Map();
  let nll = 0;
  try {
    const scored = await localizePupilField(device, truth, cfg, ref.base, grid, .1, data, x, y, 2, new StopToken(), cache);
    nll = Math.max(...scored.nll.map((v, i) => Math.abs(v - ref.nll[i]) / Math.abs(ref.nll[i])));
    check(nll < 1e-5, 'relative localization NLL error ' + nll);
  } finally { for (const loc of cache.values()) loc.destroy(); }
  const zip = await buildPsfZip(device, {
    field: fieldBlock(truth),
    vectorial: {
      optics: ref.optics, noll: ref.noll, zcoef: fieldPupil(truth, ref.noll, ref.base, 48, 48),
      focus_um: .1, npupil: 16, oversample: 2, roisize: 7, z_range_um: [0, .2]
    },
    pdf: new Uint8Array(), zstack: { planes: 3, k: 1 }
  });
  const loaded = await readPsfFile(zip.bytes);
  check(!!loaded.pupilField, 'missing field on archive reload');
  for (const [fx, fy] of ref.xy) {
    const z = fieldPupil(loaded.pupilField!, loaded.noll, loaded.zcoef, fx, fy);
    const expected = fieldPupil(truth, ref.noll, ref.base, fx, fy);
    check(z.every((v, i) => Math.abs(v - expected[i]) < 1e-10), 'reloaded field coefficients');
  }
  return 'peak-normalized pixel error ' + pixels.toExponential(2) + ', fitted coefficient error (waves) NAT ' +
    error.toExponential(2) + ', independent ' + freeError.toExponential(2) + ', relative NLL error ' + nll.toExponential(2);
});

test('NAT: in-situ movie selection and held-out scoring', async () => {
  const { device } = await getDevice();
  const width = 97, frames = 8, stack = new Float32Array(frames * width * width).fill(9);
  for (let f = 0; f < frames; f++) {
    let spot = 0;
    for (const x of [12, 32, 56, 80]) for (const y of [12, 32, 56, 80]) {
      const p = ((f + spot++) % 36) * 49;
      for (let i = 0; i < 49; i++) stack[(f * width + y - 3 + Math.floor(i / 7)) * width + x - 3 + i % 7] = ref.data[p + i];
    }
  }
  const source = { n: frames, W: width, H: width, read: async (a: number, b: number) => stack.slice(a * width * width, b * width * width) };
  const report = await estimateInsitu(device, source, {
    method: 'nat', optics: ref.optics, roisize: 7, noll: ref.noll.slice(0, 5), zcoef: ref.base.slice(0, 5),
    gain: 1, offset: 0, readnoise_e: 0, focus_um: .1, zlo: 0, zhi: .2, npupil: 16, oversample: 2,
    chunk: 4, maxFrames: 0, threshold: 3, isolation: 8, nspots: 100, nheldout: 100, iters: 1, tile: 16, rejectOverlap: false
  }, new StopToken(), () => { });
  const result = report.fieldFit!;
  check(report.noll.includes(4) && result.model.parameters.length === 13, 'defocus and thirteen NAT coefficients');
  check(report.n_test > 0 && Number.isFinite(result.heldout) && Number.isFinite(result.global), 'held-out scores');
  check(result.kept === (result.heldout! < result.global!), 'selected model must match held-out winner');
  return report.n_train + ' training and ' + report.n_test + ' held-out spots';
});
