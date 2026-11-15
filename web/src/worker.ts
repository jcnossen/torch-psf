// Compute worker: all WebGPU work (PSF volumes, the bead fit, exports) and the TIFF reading run here, so the UI
// thread only renders. Messages: see the Req / Resp types in ui/protocol.ts.
import { getDevice, describeGpu } from './gpu/device';
import { VecEngine } from './gpu/ops';
import { TiffMovie, partIndex } from './io/tiff';
import { findBeads, extractRoi, BeadFitter, Stack, Bead } from './fit/beads';
import { StopToken, AbortError, Eta } from './core/task';
import { buildKnots } from './psf/spline';
import { writeNpz } from './io/npy';
import { writeTiffStack } from './io/tifwrite';
import { DEFAULT_NOLL, VectorialConfig } from './psf/vectorial';
import type { Req, VolumeSpec, FitForm, FitSummary } from './ui/protocol';
import { localizeMovie, MovieSource } from './smlm/movie';
import { estimateInsitu } from './smlm/estimate';
import { simulateDemo, ArrayMovie } from './smlm/simulate';
import { buildPsfZip } from './psf/makezip';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let token: StopToken | null = null;
let fitter: BeadFitter | null = null;
let lastForm: FitForm | null = null;
let lastNz = 0;
let lastProgress = 0;
let demo: ArrayMovie | null = null;
const sourceOf = (s: 'demo' | File[]): File[] | MovieSource => { if (s === 'demo') { if (!demo) throw new Error('no demo movie'); return demo; } return s; };

function post(m: unknown, transfer: Transferable[] = []) { ctx.postMessage(m, transfer); }
function progress(p: object, force = false) {
  const t = performance.now();
  if (!force && t - lastProgress < 100) return;     // throttled to ~10 / s
  lastProgress = t;
  post({ type: 'progress', p });
}

async function volume(spec: VolumeSpec): Promise<Float32Array> {
  const { device } = await getDevice();
  const cfg: VectorialConfig = { ...spec.optics, pixel_um: spec.dxy_um, roisize: spec.nxy, npupil: spec.npupil, oversample: 1, zernike_noll: spec.noll };
  const eng = new VecEngine(device, cfg, [], { maxChunk: 128 });
  try {
    eng.setZernike(spec.zcoef);
    const ps = Array.from({ length: spec.nz }, (_, k) => {
      const z = spec.zmin + (spec.zmax - spec.zmin) * k / Math.max(1, spec.nz - 1);
      if (spec.focus_um !== undefined) return { x: 0, y: 0, ze: z, f: spec.focus_um, I: 1, bg: 0 };
      return { x: 0, y: 0, ze: spec.ze_um, f: spec.ze_um - z, I: 1, bg: 0 };
    });
    eng.setImages(ps);
    await eng.run(ps.length, 'psf');
    return await eng.readPsf(ps.length);
  } finally { eng.destroy(); }
}

function groupStacks(files: File[]): File[][] {
  const g = new Map<string, File[]>();
  for (const f of files) { const [base] = partIndex(f.name); g.set(base, [...(g.get(base) ?? []), f]); }
  return [...g.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([, v]) => v);
}

async function fit(files: File[], form: FitForm): Promise<{ summary: FitSummary; transfer: Transferable[] }> {
  const { device } = await getDevice();
  const tok = token!;
  const groups = groupStacks(files);
  const stacks: Stack[] = [];
  const movies: TiffMovie[] = [];
  let total = 0;
  for (const g of groups) {
    tok.check();
    const m = await TiffMovie.open(g, (msg) => progress({ stage: 'header', done: 0, total: 1, info: { msg } }, true));
    movies.push(m); total += m.n;
  }
  // --- read (chunks of frames; "Enough" stops after the current chunk)
  const eta = new Eta();
  let done = 0, partialRead = false;
  const tRead = performance.now();
  for (let s = 0; s < movies.length && !partialRead; s++) {
    const m = movies[s];
    const raw = new Float32Array(m.n * m.H * m.W);
    let f = 0;
    for (; f < m.n; f += 16) {
      tok.check();
      if (tok.signal === 'enough') { partialRead = true; break; }
      const f1 = Math.min(m.n, f + 16);
      raw.set(await m.read(f, f1), f * m.H * m.W);
      done += f1 - f;
      progress({ stage: 'read', done, total, info: { file: groups[s][0].name }, ...eta.update(done, total) });
    }
    const nz = Math.min(f, m.n);
    if (nz < 8) { if (s === 0) throw new Error('fewer than 8 planes were read'); break; }
    const data = raw.subarray(0, nz * m.H * m.W);
    stacks.push({ name: groups[s].map((x) => x.name).join('+'), nz, H: m.H, W: m.W, data });
  }
  if (tok.signal === 'enough') tok.signal = 'run';   // "enough" applied to reading; the fit runs normally
  const readSeconds = (performance.now() - tRead) / 1000;
  // --- beads
  const R = form.roisize;
  const beads: Bead[] = [];
  stacks.forEach((s, si) => {
    for (const b of findBeads(s, R, form.minRel)) beads.push({ ...b, stack: si });
    for (let i = 0; i < s.data.length; i++) s.data[i] = Math.max(0, Math.fround(Math.fround(s.data[i] - form.offset) * form.gain));
  });
  if (!beads.length) throw new Error(`no beads found (ROI ${R} px fully inside the frame, peak above median + ${form.minRel} x range)`);
  let holdout: number[] = [];
  if (form.holdout === 'last-bead' && beads.length > 1) holdout = [beads.length - 1];
  if (form.holdout === 'last-stack' && stacks.length > 1) holdout = beads.map((b, k) => (b.stack === stacks.length - 1 ? k : -1)).filter((k) => k >= 0);
  if (form.holdout === 'last-stack' && stacks.length === 1 && beads.length > 1) holdout = [beads.length - 1];
  progress({ stage: 'beads', done: beads.length, total: beads.length, info: { beads: beads.length, stacks: stacks.length } }, true);
  fitter?.destroy();
  const rois = beads.map((b) => extractRoi(stacks[b.stack], b, R));
  fitter = new BeadFitter(device, stacks, beads, rois, {
    optics: form.optics, roisize: R, npupil: form.npupil, oversample: form.oversample, sign: form.sign, dz_um: form.dz_um,
    ze_um: form.ze_um, fitDz: form.fitDz, fitDrift: form.fitDrift, holdout, maxIter: form.maxIter,
  });
  lastForm = form; lastNz = stacks[0].nz;
  const res = await fitter.fit(tok, (p) => progress(p));
  const st = res.state;
  // PSF volume of the result over the z range of the first stack (2x oversampled in xy)
  const nz0 = stacks[0].nz;
  const zr = form.dz_um * Math.exp(st.logdz) * (nz0 - 1) / 2;
  const nxy = 2 * R;
  const vspec: VolumeSpec = { optics: form.optics, npupil: form.npupil, noll: DEFAULT_NOLL, zcoef: Array.from(st.zcoef), nxy, dxy_um: form.optics.pixel_um / 2,
    zmin: -zr, zmax: zr, nz: Math.min(101, nz0), ze_um: form.ze_um };
  const vol = await volume(vspec);
  const summary: FitSummary = {
    zcoef: Array.from(st.zcoef), noll: DEFAULT_NOLL, dz_scale: Math.exp(st.logdz), drift: Array.from(st.drift),
    beads: beads.map((b, k) => ({ stack: b.stack, y0: b.y0, x0: b.x0, x: st.x[k], y: st.y[k], zoff: st.zoff[k], heldout: holdout.includes(k), nz: stacks[b.stack].nz })),
    stacks: stacks.map((s) => ({ name: s.name, nz: s.nz, H: s.H, W: s.W })),
    nll_train_px: res.nll_train_px, nll_heldout_px: res.nll_heldout_px, iterations: res.iterations, heldout_iterations: res.heldout_iterations,
    converged: res.converged, partial: res.partial || partialRead, partial_read: partialRead, fit_seconds: res.seconds, read_seconds: readSeconds,
    frames_used: stacks.reduce((a, s) => a + s.nz, 0), frames_total: total, R,
    data: rois, model: res.model, volume: vol, volumeSpec: vspec,
  };
  return { summary, transfer: [vol.buffer] };
}

async function exportFile(kind: 'knots' | 'tif', p: { focus_um: number; splineRoi: number; zmax: number; zmin: number }): Promise<{ bytes: Uint8Array; name: string }> {
  if (!fitter || !lastForm) throw new Error('no fit result');
  const { device } = await getDevice();
  const f = lastForm, st = fitter.st;
  const cfg: VectorialConfig = { ...f.optics, roisize: f.roisize, npupil: f.npupil, oversample: 3, zernike_noll: DEFAULT_NOLL };
  if (kind === 'knots') {
    const k = await buildKnots(device, cfg, st.zcoef, { roisize: p.splineRoi, focus_um: p.focus_um, zmax: p.zmax, zmin: p.zmin, S: 2, hz: 0.04, zpad: 3, xmax: 3 });
    const sc = (v: number) => ({ data: new Float64Array([v]), shape: [] as number[] });
    const bytes = writeNpz({
      knots: { data: k.knots, shape: [k.nz, k.nv, k.nu] }, z0: sc(k.z0), hz: sc(k.hz), S: { data: new Int32Array([k.S]), shape: [] },
      u0: sc(k.u0), roisize: { data: new Int32Array([k.roisize]), shape: [] }, xmax: sc(k.xmax), focus_um: sc(k.focus_um),
      zernike_noll: { data: Int32Array.from(DEFAULT_NOLL), shape: [DEFAULT_NOLL.length] }, zernike_waves: { data: Float64Array.from(st.zcoef), shape: [st.zcoef.length] },
    });
    return { bytes, name: 'psf_spline.npz' };
  }
  // model z-stack: the normalized PSF (x = y = 0) at the planes of the first stack, z step with the fitted scale
  const eng = new VecEngine(device, { ...cfg, oversample: f.oversample }, []);
  try {
    eng.setZernike(st.zcoef);
    const nz = lastNz;
    const ps = Array.from({ length: nz }, (_, j) => ({ x: 0, y: 0, ze: f.ze_um, f: f.sign * f.dz_um * Math.exp(st.logdz) * (j - (nz - 1) / 2) + f.ze_um, I: 1, bg: 0 }));
    eng.setImages(ps);
    await eng.run(nz, 'psf');
    const img = await eng.readPsf(nz);
    return { bytes: writeTiffStack(img, nz, f.roisize, f.roisize, { spacing: (f.dz_um * Math.exp(st.logdz)).toFixed(6), unit: 'micron' }), name: 'psf_model_zstack.tif' };
  } finally { eng.destroy(); }
}

ctx.onmessage = async (ev: MessageEvent<Req>) => {
  const m = ev.data;
  if (m.type === 'stop') { if (m.signal === 'abort') token?.abort(); else token?.enough(); return; }
  try {
    if (m.type === 'init') {
      const { info } = await getDevice();
      post({ id: m.id, type: 'result', gpu: describeGpu(info) });
    } else if (m.type === 'volume') {
      const t0 = performance.now();
      const v = await volume(m.spec);
      post({ id: m.id, type: 'result', volume: v, ms: performance.now() - t0 }, [v.buffer]);
    } else if (m.type === 'fit') {
      token = new StopToken();
      const { summary, transfer } = await fit(m.files, m.form);
      post({ id: m.id, type: 'result', summary }, transfer);
    } else if (m.type === 'movieHeader') {
      const mv = await TiffMovie.open(m.files);
      post({ id: m.id, type: 'result', n: mv.n, H: mv.H, W: mv.W });
    } else if (m.type === 'movie') {
      token = new StopToken();
      const { device } = await getDevice();
      const summary = await localizeMovie(device, sourceOf(m.source), m.form, token, (p) => progress(p),
        (rows) => post({ type: 'locs', rows }, [rows.buffer]));
      post({ id: m.id, type: 'result', summary });
    } else if (m.type === 'estimate') {
      token = new StopToken();
      const { device } = await getDevice();
      const report = await estimateInsitu(device, sourceOf(m.source), m.form, token, (p) => progress(p, p.stage === 'fit'));
      post({ type: 'insitu', report });
      let summary = null;
      if (m.localize && token.signal !== 'abort') {
        token.signal = 'run';
        summary = await localizeMovie(device, sourceOf(m.source), { ...m.form, zcoef: report.zcoef, zpad: 5, knots: report.refine?.kept ? report.refine.knots ?? undefined : undefined }, token, (p) => progress(p),
          (rows) => post({ type: 'locs', rows }, [rows.buffer]));
      }
      post({ id: m.id, type: 'result', report, summary });
    } else if (m.type === 'demo') {
      const { device } = await getDevice();
      const d = await simulateDemo(device);
      demo = new ArrayMovie(d.adu, d.T, d.H, d.W);
      const copy = d.adu.slice();
      post({ id: m.id, type: 'result', adu: copy, T: d.T, H: d.H, W: d.W, zernike: d.zernike, optics: d.optics, params: d.params, n_events: d.truth.length / 5 }, [copy.buffer]);
    } else if (m.type === 'export') {
      const r = await exportFile(m.kind, m.params);
      post({ id: m.id, type: 'result', bytes: r.bytes, name: r.name }, [r.bytes.buffer]);
    } else if (m.type === 'psfzip') {
      const { device } = await getDevice();
      const r = await buildPsfZip(device, m.req);
      post({ id: m.id, type: 'result', bytes: r.bytes }, [r.bytes.buffer]);
    }
  } catch (e) {
    const err = e as Error;
    post({ id: m.id, type: 'error', message: err.message ?? String(e), abort: e instanceof AbortError });
  }
};
