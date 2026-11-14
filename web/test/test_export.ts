// Exports: spline knots vs SIMCODE's ConvSplinePSF z-stack, npz round trip, float32 TIFF round trip.
import { test, check, fetchBuf, fetchJson, maxAbs, maxAbsDiff } from './harness';
import { getDevice } from '../src/gpu/device';
import { parseNpz, writeNpz } from '../src/io/npy';
import { buildKnots } from '../src/psf/spline';
import { writeTiffStack } from '../src/io/tifwrite';
import { TiffMovie } from '../src/io/tiff';
import { DEFAULT_NOLL } from '../src/psf/vectorial';

test('export: spline knots vs SIMCODE ConvSplinePSF', async () => {
  const { device } = await getDevice();
  const m = await fetchJson('knots.json');
  const npz = await parseNpz(await fetchBuf('knots.npz'));
  const ref = npz.knots.data as Float32Array;
  const k = await buildKnots(device, { ...m.optics, zernike_noll: DEFAULT_NOLL }, npz.zcoef.data as Float32Array,
    { roisize: m.optics.roisize, focus_um: m.focus_um, zmax: m.zmax, S: m.S, hz: m.hz, zpad: m.zpad, xmax: m.xmax });
  check(JSON.stringify([k.nz, k.nv, k.nu]) === JSON.stringify(m.shape), `shape ${[k.nz, k.nv, k.nu]} vs ${m.shape}`);
  check(Math.abs(k.u0 - m.u0) < 1e-9 && Math.abs(k.z0 - m.z0) < 1e-9, `grid u0 ${k.u0} / ${m.u0}, z0 ${k.z0} / ${m.z0}`);
  const e = maxAbsDiff(k.knots, 0, ref, 0, ref.length) / maxAbs(ref);
  check(e < 1e-4, `knots rel err ${e}`);
  return `knots ${m.shape.join('x')}, max rel err ${e.toExponential(1)}`;
});

test('export: npz and float32 TIFF round trips', async () => {
  const a = new Float32Array(3 * 5 * 7).map((_, i) => Math.sin(i) * 100);
  const z = writeNpz({ knots: { data: a, shape: [3, 5, 7] }, hz: { data: new Float64Array([0.04]), shape: [] } });
  const r = await parseNpz(z.buffer);
  check(r.knots.shape.join() === '3,5,7' && maxAbsDiff(r.knots.data as Float32Array, 0, a, 0, a.length) === 0 && r.hz.data[0] === 0.04, 'npz round trip');
  const t = writeTiffStack(a, 3, 5, 7);
  const mv = await TiffMovie.open([new File([t], 'm.tif')]);
  const b = await mv.read(0, 3);
  check(mv.n === 3 && mv.H === 5 && mv.W === 7 && mv.dtype === 'float32' && maxAbsDiff(b, 0, a, 0, a.length) === 0, 'tif round trip');
  return 'ok';
});
