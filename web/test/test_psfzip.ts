// psf.zip (docs/PSF_FORMAT.md) between the web app and torch_psf:
//  - write: zips of a vectorial and a spline channel built in the browser (psf/makezip.ts), read back here, handed to
//    run_tests.py (window.__testArtifacts -> test/out/web_*.psf.zip) for the Python check tests/test_web_psfzip.py;
//  - read: psf.zip files written by torch_psf.psfzip.save_zip (tests/make_web_psfzip_reference.py -> /testdata/psfzip/):
//    the Zernikes, the referenced arrays only (camera extras of unsupported dtypes skipped), and the web render of the
//    model against the zstack.tif that Python rendered.
import { test, check, fetchBuf, fetchJson, maxAbs, maxAbsDiff } from './harness';
import { getDevice } from '../src/gpu/device';
import { buildPsfZip, PsfZipRequest } from '../src/psf/makezip';
import { psfFromMeta, readPsfFile, readPsfZip } from '../src/io/psfzip';
import { ZipReader, zipEntries } from '../src/io/zip';
import { renderedZstack, splineZstack, vectorialZstack, VectorialModel } from '../src/psf/zstack';
import { TiffMovie } from '../src/io/tiff';
import { makeFitReport } from '../src/report/fitreport';
import { DEFAULT_NOLL } from '../src/psf/vectorial';

function artifact(name: string, bytes: Uint8Array) {
  let b64 = '';
  for (let i = 0; i < bytes.length; i += 8192) b64 += String.fromCharCode(...bytes.subarray(i, i + 8192));
  ((window as any).__testArtifacts ??= {})[name] = btoa(b64);
}

async function readTif(b: Uint8Array): Promise<{ data: Float32Array; n: number; H: number; W: number }> {
  const mv = await TiffMovie.open([new File([b as BlobPart], 'zstack.tif')]);
  return { data: await mv.read(0, mv.n), n: mv.n, H: mv.H, W: mv.W };
}

const relErr = (a: ArrayLike<number>, b: ArrayLike<number>) => maxAbsDiff(a, 0, b, 0, b.length) / maxAbs(b);

// the web test model: the astigmatic preset with a few more Zernikes, small for speed
const ZC: Record<number, number> = { 5: 0.02, 6: 0.1, 8: -0.015, 11: 0.03 };
const MODEL: VectorialModel = {
  optics: { pixel_um: 0.10833, NA: 1.49, wavelength_um: 0.68, n_imm: 1.518, n_med: 1.33 },
  noll: DEFAULT_NOLL, zcoef: Float32Array.from(DEFAULT_NOLL, (j) => ZC[j] ?? 0), focus_um: 0.4,
  npupil: 32, oversample: 2, roisize: 13, z_range_um: [0.0, 0.8],
};

test('psfzip: write vectorial / spline zips, read back, reproducible', async () => {
  const { device } = await getDevice();
  const pdf = await makeFitReport({ title: 'psf.zip test', created: '', zernike: { noll: MODEL.noll, values: MODEL.zcoef } }, { creationDate: false });
  const base: PsfZipRequest = {
    vectorial: MODEL, camera: { gain_e_per_adu: 0.5, offset_adu: 100 }, provenance: { source: 'web/test/test_psfzip.ts' },
    report: { estimator: 'web test', results: { n_beads: 3 }, settings: { iters: 10 }, sections: ['summary', 'aberrations'] },
    zstack: { planes: 9, k: 2 }, pdf,
  };
  const cases: [string, PsfZipRequest][] = [
    ['web_vectorial.psf.zip', base],
    ['web_spline.psf.zip', { ...base, spline: { roisize: 11, z_range: [0.1, 0.7], S: 2, hz: 0.04, zpad: 2, xmax: 3 }, zstack: { planes: 7, k: 1 } }],
  ];
  const msgs: string[] = [];
  for (const [name, req] of cases) {
    const { bytes, meta } = await buildPsfZip(device, req);
    const again = await buildPsfZip(device, req);
    check(again.bytes.length === bytes.length && maxAbsDiff(again.bytes, 0, bytes, 0, bytes.length) === 0, `${name}: two writes differ (not reproducible)`);
    const ents = zipEntries(bytes);
    check(ents[0].name === 'psf.json', 'psf.json first');
    const names = ents.map((e) => e.name);
    for (const m of ['psf.json', 'zstack.tif', 'preview.png', 'fit_report.pdf']) check(names.includes(m), `${name}: member ${m}`);
    check(!names.some((n) => /resid/i.test(n)), 'no residuals member');
    check(ents.find((e) => e.name === 'preview.png')!.method === 0, 'preview.png stored');
    check(ents.filter((e) => e.name !== 'preview.png').every((e) => e.method === (typeof CompressionStream !== 'undefined' ? 8 : 0)), 'members deflated');
    const z = new ZipReader(bytes);
    const png = await z.read('preview.png');
    check(png[0] === 0x89 && png[1] === 0x50 && png[2] === 0x4e && png[3] === 0x47, 'preview.png is a PNG');
    // read back: the Zernikes exactly, the spline knots, the zstack.tif as written
    const r = await readPsfZip(bytes);
    check(r.noll.join() === MODEL.noll.join() && r.zcoef.every((v, i) => v === MODEL.zcoef[i]), `${name}: zernikes read back`);
    check(JSON.stringify(r.optics) === JSON.stringify(MODEL.optics) && r.focus_um === MODEL.focus_um, `${name}: optics / focus read back`);
    const zb = (meta as any).zstack;
    const t = await readTif(await z.read('zstack.tif'));
    check(t.n === zb.shape[0] * zb.shape[1] && t.H === zb.shape[2] && t.W === zb.shape[3], `${name}: zstack.tif shape ${[t.n, t.H, t.W]} vs ${zb.shape}`);
    if (req.spline) {
      check(r.channelType === 'spline' && !!r.spline, 'spline channel read back');
      const zs = splineZstack(r.spline!, zb.shape[0], 1);
      check(maxAbsDiff(zs.stack, 0, t.data, 0, t.data.length) === 0, 'spline zstack re-rendered from the read knots');
    } else {
      const zs = await vectorialZstack(device, { ...MODEL, z_range_um: r.z_range_um ?? null }, zb.shape[0], 2);
      check(maxAbsDiff(zs.stack, 0, t.data, 0, t.data.length) === 0, 'vectorial zstack re-rendered');
    }
    artifact(name, bytes);
    msgs.push(`${name} ${(bytes.length / 1024).toFixed(0)} kB, zstack ${zb.shape.join('x')}`);
  }
  artifact('web_psfzip_expected.json', new TextEncoder().encode(JSON.stringify({
    optics: MODEL.optics, zernike_noll: MODEL.noll, zernike_waves: Array.from(MODEL.zcoef), focus_um: MODEL.focus_um, npupil: MODEL.npupil,
    oversample: MODEL.oversample, roisize: MODEL.roisize, z_range_um: MODEL.z_range_um, camera: base.camera,
    spline: cases[1][1].spline, zstack: { vectorial: cases[0][1].zstack, spline: cases[1][1].zstack },
  })));
  return msgs.join('; ');
});

test('psfzip: read Python-written zips (zernikes, arrays, render vs Python zstack.tif)', async () => {
  const { device } = await getDevice();
  const ref = await fetchJson('psfzip/python_ref.json');
  const msgs: string[] = [];
  for (const name of ['python_vectorial.psf.zip', 'python_spline.psf.zip']) {
    const buf = new Uint8Array(await fetchBuf('psfzip/' + name));
    const r = await readPsfFile(new Blob([buf]));
    check(r.kind === 'psf.zip', `${name}: kind ${r.kind}`);
    check(r.noll.join() === ref.zernike_noll.join(), `${name}: noll`);
    const ez = Math.max(...r.zcoef.map((v, i) => Math.abs(v - ref.zernike_waves[i])));
    check(ez === 0, `${name}: zernikes differ by ${ez}`);
    for (const k of ['pixel_um', 'NA', 'wavelength_um', 'n_imm', 'n_med'] as const)
      check(Math.fround(r.optics![k]) === Math.fround(ref.optics[k]), `${name}: optics ${k} ${r.optics![k]} vs ${ref.optics[k]}`);
    check(r.focus_um === ref.focus_um && r.npupil === ref.npupil && r.oversample === ref.oversample && r.roisize === ref.roisize, `${name}: model settings`);
    const z = new ZipReader(buf), meta = JSON.parse(await z.text('psf.json')), zb = meta.zstack;
    const t = await readTif(await z.read('zstack.tif'));
    check(t.n === zb.shape[0] * zb.shape[1] && t.H === zb.shape[2], `${name}: Python zstack.tif ${[t.n, t.H, t.W]} vs ${zb.shape}`);
    const k = Math.round(1 / zb.sample_px);
    let e: number;
    if (r.channelType === 'spline') {
      check(!!r.spline && r.spline.nz === ref.spline.nz && r.spline.u0_px === ref.spline.u0_px, `${name}: spline grid`);
      e = relErr(splineZstack(r.spline!, zb.shape[0], k).stack, t.data);
      check(e < 1e-5, `${name}: web spline render vs Python zstack.tif, max rel err ${e}`);
    } else {
      const v: VectorialModel = { optics: r.optics!, noll: r.noll, zcoef: r.zcoef, focus_um: r.focus_um, npupil: r.npupil!, oversample: r.oversample!, roisize: r.roisize!, z_range_um: r.z_range_um ?? null };
      e = relErr((await vectorialZstack(device, v, zb.shape[0], k)).stack, t.data);
      check(e < 1e-4, `${name}: web vectorial render vs Python zstack.tif, max rel err ${e}`);
      // camera: maps parsed, extras not parsed by default; with cameraExtras the unsupported dtypes are skipped
      check(!!r.arrays['camera/offset_adu.npy'] && !Object.keys(r.arrays).some((m) => m.startsWith('camera/extra/')), `${name}: parsed members ${Object.keys(r.arrays)}`);
      const rx = await readPsfZip(buf, { cameraExtras: true });
      const skipped = rx.notes.filter((s) => / skipped /.test(s));
      check(!!rx.arrays['camera/extra/n_frames.npy'] && !!rx.arrays['camera/extra/variance.npy'] && skipped.length === 2 &&
        skipped.some((s) => s.includes('hot.npy')) && skipped.some((s) => s.includes('serial.npy')), `${name}: camera extras ${Object.keys(rx.arrays)}; ${skipped.join('; ')}`);
      check(r.camera?.gain_e_per_adu === 0.47 && r.camera?.offset_adu === 100, `${name}: camera scalars`);
    }
    msgs.push(`${name}: zernikes equal, render vs Python max rel err ${e.toExponential(1)} (${zb.shape.join('x')})`);
  }
  return msgs.join('; ');
});

test('psfzip: CPU render of Python 0.3 models', async () => {
  const old = await readPsfZip(await fetchBuf('psfzip/python_vectorial.psf.zip'));
  check(old.meta?.version === '0.2.0' && !old.renderer, 'existing box file');
  const ref = await fetchJson('psfzip/integration_ref.json'), msgs: string[] = [];
  for (const c of ref.cases) {
    const buf = await fetchBuf('psfzip/' + c.name + '.psf.zip'), file = await readPsfZip(buf);
    check(file.meta?.version === '0.3.0' && !!file.renderer, c.name + ': 0.3 renderer');
    if (c.angle == null) {
      const t = await readTif(await new ZipReader(buf).read('zstack.tif'));
      const stack = renderedZstack(file.renderer!, t.n);
      check(relErr(stack.stack, t.data) < 1e-5, c.name + ': Python zstack.tif');
    }
    let err = 0;
    for (let i = 0; i < ref.positions.length; i++) {
      const [x, y, z] = ref.positions[i], img = file.renderer!.render(x, y, z, c.angle ?? undefined);
      err = Math.max(err, relErr(img, c.images[i]));
    }
    check(err < 1e-5, c.name + ': error / peak ' + err);
    msgs.push(`${c.name} (angle ${c.angle ?? 'mean'}): ${err.toExponential(2)}`);
  }
  return msgs.join(', ');
});

test('psfzip: refuse unknown versions, integration modes and invalid spline coefficients', async () => {
  const file = await readPsfZip(await fetchBuf('psfzip/integrated_angles.psf.zip'));
  const reject = (edit: (meta: any) => void, feature: string) => {
    const meta = structuredClone(file.meta!);
    edit(meta);
    let message = '';
    try { psfFromMeta(meta, file.arrays); } catch (e) { message = (e as Error).message; }
    check(message.includes(feature), `expected ${feature}, got ${message}`);
  };
  reject(m => m.version = '0.4.0', '0.4.0');
  reject(m => m.channels[0].source.model.pixel_integration = 'unknown_rule', 'unknown_rule');
  reject(m => m.channels[0].model.integrated_spline.basis = 'unknown_basis', 'unknown_basis');
  reject(m => m.channels[0].model.integrated_spline.n_angles++, 'coefs_angle');
  reject(m => m.channels[0].arrays.coefs_angle = 'missing.npy', 'coefs_angle');
  reject(m => m.channels[0].arrays.coefs = 'second.npy', 'coefs_angle');
});
