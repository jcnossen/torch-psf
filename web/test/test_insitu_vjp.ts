// In-situ fit, GPU-resident route: the knot-build VJP (one backward pass, KnotEngine.vjp) against the forward-mode
// Jacobian contraction gK . dK of the previous route, and fitZernikeKnots (everything on the GPU, one readback per
// iteration) against the previous implementation (option jacobian). Reference data: test/.cache/smlm (as test_smlm).
import { test, check, fetchBuf, fetchJson } from './harness';
import { getDevice } from '../src/gpu/device';
import { splineGrid } from '../src/gpu/spline_lm';
import { KnotEngine } from '../src/gpu/knot_engine';
import { parseNpz } from '../src/io/npy';
import { knotsAndJacobian, fitZernikeKnots, KnotGridFull } from '../src/smlm/insitu';

const SU = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

async function readF32(dev: GPUDevice, src: GPUBuffer, n: number): Promise<Float32Array> {
  const st = dev.createBuffer({ size: Math.ceil(n * 4 / 16) * 16, usage: GPUBufferUsage.MAP_READ | CD });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, st, 0, n * 4);
  dev.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ);
  const a = new Float32Array(st.getMappedRange().slice(0, n * 4));
  st.unmap(); st.destroy();
  return a;
}

const relL2 = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let d = 0, n = 0; for (let i = 0; i < b.length; i++) { d += (a[i] - b[i]) ** 2; n += b[i] ** 2; } return Math.sqrt(d / n);
};

test('insitu VJP: GPU backward of the knot build = forward-mode Jacobian contraction gK . dK (and torch_psf vjp_vectorial)', async () => {
  const meta = await fetchJson('smlm/insitu.json');
  const z = await parseNpz(await fetchBuf('smlm/insitu_grad.npz'));
  const { device } = await getDevice();
  const g = splineGrid(15, 0, 0.8, 2, 0.04, 5, 3.0) as KnotGridFull;
  const cfg = { ...meta.optics, roisize: 0, npupil: 64, oversample: 3, zernike_noll: meta.noll };
  const zc0 = meta.noll.map((j: number) => Math.fround(meta.start_zernike[String(j)] ?? 0));
  const gK = new Float32Array(z.gK.data as Float32Array), nK = gK.length, nZ = zc0.length;
  const gzTorch = z.gz.data as Float32Array;
  // previous route: D = 18 directions, read back, contracted on the CPU (float64)
  let t = performance.now();
  const { knots: Kold, dK } = await knotsAndJacobian(device, cfg, zc0, g, meta.focus_um);
  const gzOld = new Float64Array(nZ);
  for (let q = 0; q < nZ; q++) { let s = 0; for (let k = 0; k < nK; k++) s += gK[k] * dK[q * nK + k]; gzOld[q] = s; }
  const tOld = performance.now() - t;
  // GPU route, one chunk (the default) and 8 chunks (the backward recomputes each chunk's fields)
  const bK = device.createBuffer({ size: nK * 4, usage: SU | CD | CS }), bG = device.createBuffer({ size: nK * 4, usage: SU | CD });
  const out = device.createBuffer({ size: 128, usage: SU | CD | CS });
  device.queue.writeBuffer(bG, 0, gK);
  const runNew = async (eng: KnotEngine) => {
    eng.setZernike(zc0);
    eng.build(bK);
    eng.vjp(bG, out);
    return readF32(device, out, nZ);
  };
  const res: { name: string; knots: Float32Array; gz: Float32Array; ms: number }[] = [];
  try {
    for (const [name, maxChunk] of [['one chunk', undefined], ['8 chunks', 4]] as const) {
      const eng = new KnotEngine(device, cfg, g, meta.focus_um, maxChunk ? { maxChunk } : {});
      try {
        await runNew(eng);                                   // warm-up (pipelines)
        const reps = 5;
        t = performance.now();
        let gz = new Float32Array(nZ);
        for (let r = 0; r < reps; r++) gz = await runNew(eng);
        const ms = (performance.now() - t) / reps;
        res.push({ name, knots: await readF32(device, bK, nK), gz, ms });
      } finally { eng.destroy(); }
    }
  } finally { bK.destroy(); bG.destroy(); out.destroy(); }
  let kd = 0, km = 0; for (let i = 0; i < nK; i++) { km = Math.max(km, Math.abs(Kold[i])); kd = Math.max(kd, Math.abs(res[0].knots[i] - Kold[i])); }
  let kd8 = 0; for (let i = 0; i < nK; i++) kd8 = Math.max(kd8, Math.abs(res[1].knots[i] - Kold[i]));
  const e1 = relL2(res[0].gz, gzOld), e8 = relL2(res[1].gz, gzOld), eT = relL2(res[0].gz, gzTorch), eOT = relL2(gzOld, gzTorch);
  let emax = 0; for (let q = 0; q < nZ; q++) emax = Math.max(emax, Math.abs(res[0].gz[q] - gzOld[q]) / Math.max(...Array.from(gzOld, Math.abs)));
  const msg = `knots GPU build vs previous: max |diff| / max ${(kd / km).toExponential(1)} (8 chunks ${(kd8 / km).toExponential(1)}); ` +
    `Zernike gradient VJP vs gK . dK: rel L2 ${e1.toExponential(1)}, max rel ${emax.toExponential(1)} (8 chunks ${e8.toExponential(1)}); ` +
    `vs torch_psf vjp_vectorial ${eT.toExponential(1)} (previous route ${eOT.toExponential(1)}); ` +
    `time build + VJP ${res[0].ms.toFixed(1)} ms (8 chunks ${res[1].ms.toFixed(1)} ms) vs knotsAndJacobian + contraction ${tOld.toFixed(0)} ms`;
  check(kd / km < 1e-6 && kd8 / km < 1e-6 && e1 < 1e-5 && e8 < 1e-5 && eT < 1e-3, msg);
  return msg;
});

test('insitu VJP: fitZernikeKnots on the GPU (one readback per iteration) = previous route (forward-mode Jacobian), 40 iterations', async () => {
  const meta = await fetchJson('smlm/insitu.json');
  const z = await parseNpz(await fetchBuf('smlm/insitu_grad.npz'));
  const { device } = await getDevice();
  const g = splineGrid(15, 0, 0.8, 2, 0.04, 5, 3.0) as KnotGridFull;
  const cfg = { ...meta.optics, roisize: 0, npupil: 64, oversample: 3, zernike_noll: meta.noll };
  const zc0 = meta.noll.map((j: number) => meta.start_zernike[String(j)] ?? 0);
  const N = z.rois.shape[0], rois = new Float32Array(z.rois.data as Float32Array);
  const iters = meta.iters;
  const rNew = await fitZernikeKnots(device, rois, N, cfg, zc0, g, meta.focus_um, { iters });
  const rOld = await fitZernikeKnots(device, rois, N, cfg, zc0, g, meta.focus_um, { iters, jacobian: true });
  const rms = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
  let hz = 0, hn = 0;
  for (let i = 0; i < iters; i++) {
    hz = Math.max(hz, rms(rNew.history[i].zcoef, rOld.history[i].zcoef));
    hn = Math.max(hn, Math.abs(rNew.history[i].nll - rOld.history[i].nll) / Math.abs(rOld.history[i].nll));
  }
  let dth = 0; for (let i = 0; i < N * 5; i++) if (i % 5 < 2) dth = Math.max(dth, Math.abs(rNew.theta[i] - rOld.theta[i]));
  let kd = 0, km = 0, k0 = 0;
  for (let i = 0; i < rOld.knots.length; i++) {
    km = Math.max(km, Math.abs(rOld.knots[i])); kd = Math.max(kd, Math.abs(rNew.knots[i] - rOld.knots[i])); k0 = Math.max(k0, Math.abs(rNew.K0[i] - rOld.K0[i]));
  }
  // per-iteration time after the first (the first localizes from 5 z starts with 30 LM iterations in both routes)
  const per = (r: typeof rNew) => (r.history[iters - 1].seconds - r.history[0].seconds) / (iters - 1);
  const ref = await fetchJson('smlm/insitu_fit.json');
  const msg = `${N} ROIs, ${iters} iterations: new ${rNew.seconds.toFixed(1)} s (${(1e3 * per(rNew)).toFixed(0)} ms / iteration), ` +
    `previous ${rOld.seconds.toFixed(1)} s (${(1e3 * per(rOld)).toFixed(0)} ms / iteration), speed-up ${(per(rOld) / per(rNew)).toFixed(1)}x; ` +
    `max over iterations: zcoef rms diff ${hz.toExponential(1)} w, NLL rel. diff ${hn.toExponential(1)}; final zcoef rms diff ${rms(rNew.zcoef, rOld.zcoef).toExponential(1)} w ` +
    `(vs torch_psf ${rms(rNew.zcoef, ref.final_zernike).toExponential(1)} w); K0 max |diff| / max ${(k0 / km).toExponential(1)}, last knots max |diff| / max ${(kd / km).toExponential(1)}, ` +
    `final theta max |dxy| ${dth.toExponential(1)} px`;
  check(rms(rNew.zcoef, rOld.zcoef) < 2e-4 && hz < 2e-4 && hn < 1e-5 && k0 / km < 1e-6 && kd / km < 1e-3 && rms(rNew.zcoef, ref.final_zernike) < 2e-3, msg);
  return msg;
});
