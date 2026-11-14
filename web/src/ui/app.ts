// The web app (UI thread): presets + 3D view, the bead-fit form, progress / stop, results and exports.
// All GPU work runs in the compute worker (../worker.ts).
import { View3D, Volume } from './view3d';
import { drawArray } from './colormap';
import { DEFAULT_NOLL, Optics } from '../psf/vectorial';
import { makePsfJson } from '../io/psfjson';
import { WEBGPU_HELP } from '../gpu/device';
import type { FitForm, FitSummary, VolumeSpec, Req } from './protocol';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------- worker RPC
const worker = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
let nextId = 1;
const pending = new Map<number, { res: (v: any) => void; rej: (e: any) => void }>();
let onProgress: (p: any) => void = () => {};
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'progress') { onProgress(m.p); return; }
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  if (m.type === 'error') { const e = new Error(m.message); (e as any).abort = m.abort; p.rej(e); } else p.res(m);
};
type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
function call(req: DistOmit<Exclude<Req, { type: 'stop' }>, 'id'>, transfer: Transferable[] = []): Promise<any> {
  const id = nextId++;
  return new Promise((res, rej) => { pending.set(id, { res, rej }); worker.postMessage({ ...req, id }, transfer); });
}

// ---------------------------------------------------------------- presets
interface Preset { name: string; label: string; spec: VolumeSpec; note: string; }
const OIL: Optics = { pixel_um: 0.1, NA: 1.49, wavelength_um: 0.68, n_imm: 1.518, n_med: 1.33 };
const zc = (m: Record<number, number>) => DEFAULT_NOLL.map((j) => m[j] ?? 0);
const PRESETS: Preset[] = [
  { name: 'tetrapod', label: 'Tetrapod-like (Zernike approx.)',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({ 6: 0.2, 12: -0.2 }), nxy: 64, dxy_um: 0.075, zmin: -1.6, zmax: 1.6, nz: 65, ze_um: 0 },
    note: 'Tetrapod-like PSF approximated by a Zernike combination: primary astigmatism Noll 6 = +0.20 and secondary astigmatism Noll 12 = -0.20 waves rms (not the published optimized tetrapod pupil). NA 1.49, 680 nm, z = defocus of an emitter at the coverslip, range 3.2 um.' },
  { name: 'astig', label: 'Astigmatic 3D',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({ 6: 0.1 }), nxy: 48, dxy_um: 0.04, zmin: -0.6, zmax: 0.6, nz: 61, ze_um: 0 },
    note: 'Astigmatic PSF: Noll 6 = 0.10 waves rms (a weak cylindrical lens). NA 1.49, 680 nm, range 1.2 um.' },
  { name: 'focus', label: 'In-focus 2D',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({}), nxy: 40, dxy_um: 0.04, zmin: -0.4, zmax: 0.4, nz: 41, ze_um: 0 },
    note: 'Aberration-free vectorial PSF, NA 1.49, 680 nm, range 0.8 um.' },
];

// ---------------------------------------------------------------- DOM
export function startApp(root: HTMLElement) {
  root.innerHTML = TEMPLATE;
  const view = new View3D($('view3d'));
  (window as any).__view = view;
  let fitResult: FitSummary | null = null;
  let current: { vol: Volume; spec: VolumeSpec } | null = null;

  const status = $('gpu-status');
  call({ type: 'init' }).then(
    (r) => { status.textContent = 'WebGPU: ' + r.gpu; status.dataset.ok = '1'; showPreset('tetrapod'); },
    (e) => {
      status.textContent = 'WebGPU unavailable';
      status.dataset.ok = '0';
      $('nogpu').hidden = false;
      $('nogpu').textContent = String(e.message || e) || WEBGPU_HELP;
      ($('run') as HTMLButtonElement).disabled = true;
    });

  // --- presets and the view
  const sel = $<HTMLSelectElement>('preset');
  sel.innerHTML = PRESETS.map((p) => `<option value="${p.name}">${p.label}</option>`).join('') + '<option value="fit" disabled>Loaded file (fit result)</option>';
  sel.onchange = () => showPreset(sel.value);

  async function showPreset(name: string) {
    if (name === 'fit') { if (fitResult) setVolume(fitResult.volume, fitResult.volumeSpec, fitNote(fitResult)); return; }
    const p = PRESETS.find((q) => q.name === name)!;
    sel.value = name;
    $('preset-note').textContent = 'computing...';
    const r = await call({ type: 'volume', spec: p.spec });
    setVolume(r.volume, p.spec, p.note + ` (GPU: ${r.ms.toFixed(0)} ms for ${p.spec.nz} planes of ${p.spec.nxy} x ${p.spec.nxy})`);
    (window as any).__presetShown = name;
  }

  function setVolume(data: Float32Array, spec: VolumeSpec, note: string) {
    const v: Volume = { data, nx: spec.nxy, ny: spec.nxy, nz: spec.nz, dx_um: spec.dxy_um, dz_um: (spec.zmax - spec.zmin) / Math.max(1, spec.nz - 1), z0_um: spec.zmin };
    current = { vol: v, spec };
    view.setVolume(v);
    $('preset-note').textContent = note;
    drawSections();
  }

  function drawSections() {
    const s = view.sections();
    if (!s || !current) return;
    const v = current.vol, dec = view.opts.decades;
    drawArray($('sec-xy') as HTMLCanvasElement, s.xy, v.nx, v.ny, { decades: dec });
    drawArray($('sec-xz') as HTMLCanvasElement, flipRows(s.xz, v.nx, v.nz), v.nx, v.nz, { decades: dec });
    drawArray($('sec-yz') as HTMLCanvasElement, flipRows(s.yz, v.ny, v.nz), v.ny, v.nz, { decades: dec });
    const z = v.z0_um + s.iz * v.dz_um;
    $('sec-label').textContent = `xy at z = ${z.toFixed(3)} um; xz / yz through the slice planes (z up), log scale over ${dec} decades`;
  }

  for (const [i, id] of ['sx', 'sy', 'sz'].entries()) {
    $<HTMLInputElement>(id).oninput = (e) => { view.setSlice(i as 0 | 1 | 2, +(e.target as HTMLInputElement).value / 100); drawSections(); };
  }
  $<HTMLInputElement>('zexag').oninput = (e) => { view.opts.zExag = +(e.target as HTMLInputElement).value; $('zexag-v').textContent = view.opts.zExag.toFixed(1); view.rebuild(); };
  $<HTMLInputElement>('levels').onchange = (e) => {
    const lv = (e.target as HTMLInputElement).value.split(/[ ,]+/).map(Number).filter((x) => x > 0 && x < 100).map((x) => x / 100);
    if (lv.length) { view.opts.levels = lv; view.rebuild(); }
  };
  $<HTMLInputElement>('show-iso').onchange = (e) => { view.opts.showIso = (e.target as HTMLInputElement).checked; view.rebuild(); };
  $<HTMLInputElement>('show-slices').onchange = (e) => { view.opts.showSlices = (e.target as HTMLInputElement).checked; view.rebuild(); };
  $<HTMLInputElement>('decades').onchange = (e) => { view.opts.decades = +(e.target as HTMLInputElement).value; view.updateSlices(); drawSections(); };

  // --- files
  let files: File[] = [];
  const drop = $('drop');
  const setFiles = (fl: FileList | File[]) => {
    files = [...fl].filter((f) => /\.tiff?$/i.test(f.name));
    $('files').textContent = files.length ? files.map((f) => `${f.name} (${(f.size / 2 ** 20).toFixed(1)} MB)`).join(', ') : 'no TIFF selected';
  };
  (window as any).__setFiles = setFiles;
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer) setFiles(e.dataTransfer.files); };
  $<HTMLInputElement>('file').onchange = (e) => setFiles((e.target as HTMLInputElement).files!);

  // --- run / stop
  const runBtn = $<HTMLButtonElement>('run'), enoughBtn = $<HTMLButtonElement>('enough'), abortBtn = $<HTMLButtonElement>('abort');
  const busy = (b: boolean) => { runBtn.disabled = b; enoughBtn.disabled = !b; abortBtn.disabled = !b; };
  busy(false);
  enoughBtn.onclick = () => worker.postMessage({ type: 'stop', signal: 'enough' });
  abortBtn.onclick = () => worker.postMessage({ type: 'stop', signal: 'abort' });
  onProgress = (p) => {
    const frac = p.total ? Math.min(1, p.done / p.total) : 0;
    $('pbar').style.width = (100 * frac).toFixed(1) + '%';
    enoughBtn.textContent = p.stage === 'read' ? 'Enough frames' : p.stage === 'fit' ? 'Stop and keep current estimate' : 'Use what we have';
    const info = Object.entries(p.info ?? {}).map(([k, v]) => `${k} ${typeof v === 'number' ? (Math.abs(v) < 1e-2 || Math.abs(v) > 1e4 ? v.toExponential(2) : v.toFixed(5)) : v}`).join(', ');
    const t = p.elapsed_s !== undefined ? ` | ${p.elapsed_s.toFixed(1)} s` + (p.eta_s !== undefined ? `, ~${p.eta_s.toFixed(1)} s left (at most)` : '') : '';
    $('ptext').textContent = `${p.stage}: ${p.done} / ${p.total}${info ? ' | ' + info : ''}${t}`;
  };
  runBtn.onclick = async () => {
    if (!files.length) { $('ptext').textContent = 'Drop or choose a bead z-stack TIFF first.'; return; }
    busy(true);
    $('ptext').textContent = 'starting...';
    try {
      const r = await call({ type: 'fit', files, form: readForm() });
      fitResult = r.summary as FitSummary;
      showResult(fitResult);
      $('pbar').style.width = '100%';
      $('ptext').textContent = `done: ${fitResult.iterations} iterations in ${fitResult.fit_seconds.toFixed(2)} s` + (fitResult.partial ? ' (partial result)' : '');
      (window as any).__fitDone = true;
    } catch (e) {
      const err = e as Error & { abort?: boolean };
      $('ptext').textContent = err.abort ? 'aborted' : 'error: ' + err.message;
      $('pbar').style.width = '0%';
    } finally { busy(false); }
  };

  function readForm(): FitForm {
    const n = (id: string) => +$<HTMLInputElement>(id).value;
    return {
      optics: { NA: n('f-na'), wavelength_um: n('f-wl'), n_imm: n('f-nimm'), n_med: n('f-nmed'), pixel_um: n('f-px') },
      sign: n('f-sign'), dz_um: n('f-dz'), gain: n('f-gain'), offset: n('f-offset'), ze_um: n('f-ze'),
      roisize: n('f-roi'), npupil: n('f-npupil'), oversample: n('f-os'),
      fitDz: $<HTMLInputElement>('f-fitdz').checked, fitDrift: $<HTMLInputElement>('f-drift').checked,
      holdout: $<HTMLSelectElement>('f-holdout').value as FitForm['holdout'], maxIter: n('f-iter'), minRel: n('f-minrel'),
    };
  }

  // --- results
  const fitNote = (r: FitSummary) => `Fitted PSF (Noll 5-21) of ${r.stacks.map((s) => s.name).join(', ')}; z = defocus of the bead (stage coordinate, z-step scale applied), xy 2x oversampled.`;

  function showResult(r: FitSummary) {
    $('results').hidden = false;
    const opt = sel.querySelector('option[value="fit"]') as HTMLOptionElement;
    opt.disabled = false;
    sel.value = 'fit';
    setVolume(r.volume, r.volumeSpec, fitNote(r));
    const rows: [string, string][] = [
      ['beads (held out)', `${r.beads.length} (${r.beads.filter((b) => b.heldout).length})`],
      ['planes used / read', `${r.frames_used} / ${r.frames_total}${r.partial_read ? ' (reading stopped early)' : ''}`],
      ['iterations', `${r.iterations}${r.converged ? ' (converged)' : r.partial ? ' (stopped by user)' : ' (iteration limit)'} + ${r.heldout_iterations} held-out`],
      ['NLL / px, fit beads', r.nll_train_px.toFixed(5)],
      ['NLL / px, held-out', r.nll_heldout_px === null ? 'none held out' : r.nll_heldout_px.toFixed(5)],
      ['z-step scale', r.dz_scale.toFixed(4)],
      ['time: read / fit', `${r.read_seconds.toFixed(2)} s / ${r.fit_seconds.toFixed(2)} s`],
    ];
    $('res-table').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
    drawZernike($('zern') as HTMLCanvasElement, r.noll, r.zcoef);
    const bs = $<HTMLSelectElement>('bead');
    bs.innerHTML = r.beads.map((b, k) => `<option value="${k}">bead ${k + 1}${b.heldout ? ' (held out)' : ''}: stack ${b.stack + 1}, corner (${b.x0}, ${b.y0})</option>`).join('');
    const ho = r.beads.findIndex((b) => b.heldout);
    bs.value = String(ho >= 0 ? ho : 0);
    bs.onchange = () => drawBead();
    $<HTMLInputElement>('plane').oninput = () => drawBead();
    drawBead();
  }

  function drawBead() {
    const r = fitResult!;
    const k = +$<HTMLSelectElement>('bead').value;
    const R = r.R, nz = r.beads[k].nz, d = r.data[k], m = r.model[k];
    const pl = $<HTMLInputElement>('plane');
    pl.max = String(nz - 1);
    let j = +pl.value; if (j > nz - 1) { j = Math.floor(nz / 2); pl.value = String(j); }
    const xy = (a: Float32Array) => a.subarray(j * R * R, (j + 1) * R * R);
    let vmax = 0; for (let i = 0; i < R * R; i++) vmax = Math.max(vmax, d[j * R * R + i], m[j * R * R + i]);
    const res = new Float32Array(R * R).map((_, i) => d[j * R * R + i] - m[j * R * R + i]);
    drawArray($('b-data') as HTMLCanvasElement, xy(d), R, R, { vmax });
    drawArray($('b-model') as HTMLCanvasElement, xy(m), R, R, { vmax });
    drawArray($('b-res') as HTMLCanvasElement, res, R, R, { diverging: true, vmax: Math.max(1, Math.sqrt(vmax)) * 3 });
    // xz sections through the ROI centre row (z up)
    const row = Math.floor(R / 2);
    const xz = (a: Float32Array) => { const o = new Float32Array(nz * R); for (let z = 0; z < nz; z++) for (let x = 0; x < R; x++) o[(nz - 1 - z) * R + x] = a[(z * R + row) * R + x]; return o; };
    const dx = xz(d), mx = xz(m);
    let v2 = 0; for (let i = 0; i < dx.length; i++) v2 = Math.max(v2, dx[i], mx[i]);
    drawArray($('b-dxz') as HTMLCanvasElement, dx, R, nz, { vmax: v2 });
    drawArray($('b-mxz') as HTMLCanvasElement, mx, R, nz, { vmax: v2 });
    drawArray($('b-rxz') as HTMLCanvasElement, dx.map((v, i) => v - mx[i]), R, nz, { diverging: true, vmax: Math.max(1, Math.sqrt(v2)) * 3 });
    $('b-label').textContent = `plane ${j} of ${nz}; residual colour range +-3 sqrt(peak); bead ${r.beads[k].heldout ? 'held out (PSF frozen)' : 'used in the fit'}; ` +
      `x ${r.beads[k].x.toFixed(2)} px, y ${r.beads[k].y.toFixed(2)} px from the ROI centre, focus offset ${(r.beads[k].zoff * 1000).toFixed(1)} nm`;
  }

  // --- exports
  const download = (bytes: BlobPart, name: string, type = 'application/octet-stream') => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([bytes], { type }));
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  };
  const expParams = () => ({ focus_um: +$<HTMLInputElement>('e-focus').value, splineRoi: +$<HTMLInputElement>('e-roi').value, zmin: +$<HTMLInputElement>('e-zmin').value, zmax: +$<HTMLInputElement>('e-zmax').value });
  $('x-json').onclick = () => {
    const r = fitResult!, f = readForm();
    const j = makePsfJson({ optics: f.optics, noll: r.noll, zcoef: r.zcoef, focus_um: expParams().focus_um, npupil: f.npupil, oversample: f.oversample,
      provenance: { source: 'torch-psf web bead fit', created: new Date().toISOString(), stacks: r.stacks.map((s) => s.name), beads: r.beads.length,
        heldout_beads: r.beads.filter((b) => b.heldout).length, frames_used: r.frames_used, frames_total: r.frames_total, partial: r.partial,
        nll_train_px: r.nll_train_px, nll_heldout_px: r.nll_heldout_px, iterations: r.iterations, dz_scale: r.dz_scale,
        stage_sign: f.sign, dz_um: f.dz_um, bead_centre_um: f.ze_um, gain_e_per_adu: f.gain, offset_adu: f.offset } });
    download(JSON.stringify(j, null, 1), 'psf.json', 'application/json');
  };
  const viaWorker = async (kind: 'knots' | 'tif') => {
    $('x-status').textContent = 'computing...';
    try { const r = await call({ type: 'export', kind, params: expParams() }); download(r.bytes, r.name); $('x-status').textContent = `${r.name} (${(r.bytes.length / 1024).toFixed(0)} kB)`; }
    catch (e) { $('x-status').textContent = 'error: ' + (e as Error).message; }
  };
  $('x-knots').onclick = () => viaWorker('knots');
  $('x-tif').onclick = () => viaWorker('tif');
}

function flipRows(a: Float32Array, w: number, h: number) {
  const o = new Float32Array(a.length);
  for (let r = 0; r < h; r++) o.set(a.subarray(r * w, (r + 1) * w), (h - 1 - r) * w);
  return o;
}

function drawZernike(cv: HTMLCanvasElement, noll: number[], z: number[]) {
  const ctx = cv.getContext('2d')!;
  const W = cv.width, H = cv.height, pad = 28;
  ctx.clearRect(0, 0, W, H);
  const m = Math.max(0.02, ...z.map(Math.abs));
  const bw = (W - 2 * pad) / noll.length;
  const y0 = H / 2;
  ctx.strokeStyle = '#667'; ctx.beginPath(); ctx.moveTo(pad, y0); ctx.lineTo(W - pad / 2, y0); ctx.stroke();
  ctx.font = '11px sans-serif'; ctx.fillStyle = '#aab';
  ctx.fillText(`+${m.toFixed(3)} w`, 2, 12); ctx.fillText(`-${m.toFixed(3)} w`, 2, H - 4);
  noll.forEach((j, i) => {
    const h = (z[i] / m) * (H / 2 - 16);
    ctx.fillStyle = z[i] >= 0 ? '#e8a33d' : '#3d8be8';
    ctx.fillRect(pad + i * bw + 2, h >= 0 ? y0 - h : y0, bw - 4, Math.abs(h));
    ctx.fillStyle = '#ccd';
    ctx.fillText(String(j), pad + i * bw + bw / 2 - 6, h >= 0 ? y0 + 12 : y0 - 4);
  });
}

const num = (id: string, label: string, v: number, step = 'any', title = '') =>
  `<label title="${title}">${label}<input id="${id}" type="number" step="${step}" value="${v}"></label>`;

const TEMPLATE = `
<header><h1>torch-psf <span>web</span></h1><div id="gpu-status">Checking WebGPU...</div></header>
<div id="nogpu" class="warn" hidden></div>
<main>
  <section id="viewer">
    <div class="bar">
      <label>PSF <select id="preset"></select></label>
      <label title="relative levels of the peak, in %">isosurfaces (%) <input id="levels" value="50, 10, 1" size="9"></label>
      <label>z exaggeration <input id="zexag" type="range" min="0.25" max="4" step="0.25" value="1"><span id="zexag-v">1.0</span></label>
      <label>log decades <input id="decades" type="number" min="1" max="6" value="3" style="width:3em"></label>
      <label><input id="show-iso" type="checkbox" checked> surfaces</label>
      <label><input id="show-slices" type="checkbox" checked> slices</label>
    </div>
    <div id="view3d"></div>
    <div class="bar">slice planes: <label>x <input id="sx" type="range" min="0" max="100" value="50"></label>
      <label>y <input id="sy" type="range" min="0" max="100" value="50"></label>
      <label>z <input id="sz" type="range" min="0" max="100" value="50"></label></div>
    <p id="preset-note" class="note"></p>
    <div class="panels">
      <figure><canvas id="sec-xy" width="200" height="200"></canvas><figcaption>xy</figcaption></figure>
      <figure><canvas id="sec-xz" width="200" height="200"></canvas><figcaption>xz</figcaption></figure>
      <figure><canvas id="sec-yz" width="200" height="200"></canvas><figcaption>yz</figcaption></figure>
    </div>
    <p id="sec-label" class="note"></p>
  </section>
  <section id="fit">
    <h2>Bead z-stack &rarr; vectorial PSF</h2>
    <div id="drop"><input id="file" type="file" multiple accept=".tif,.tiff"><p>Drop bead z-stack TIFF(s) here or choose files. Several stacks: one file each; a multi-part stack: movie.tif, movie_X1.tif, ...</p><p id="files" class="note">no TIFF selected</p></div>
    <fieldset><legend>Optics and stage</legend>
      ${num('f-na', 'NA', 1.49)}${num('f-wl', 'wavelength (um)', 0.68)}${num('f-nimm', 'n immersion', 1.518)}${num('f-nmed', 'n medium', 1.33)}
      ${num('f-px', 'pixel size (um)', 0.1)}${num('f-dz', 'z step (um)', 0.02)}
      <label>stage sign<select id="f-sign"><option value="1">+1</option><option value="-1">-1</option></select></label>
      ${num('f-ze', 'bead centre height (um)', 0.05, 'any', 'emitter height of the (point) bead above the coverslip')}
    </fieldset>
    <fieldset><legend>Camera</legend>${num('f-gain', 'gain (e-/ADU)', 0.5)}${num('f-offset', 'offset (ADU)', 100)}</fieldset>
    <fieldset><legend>Fit</legend>
      ${num('f-roi', 'ROI (px)', 26, '1')}${num('f-npupil', 'pupil samples', 64, '1')}${num('f-os', 'oversampling', 2, '1')}
      ${num('f-iter', 'max iterations', 100, '1')}${num('f-minrel', 'bead threshold (rel.)', 0.3)}
      <label><input id="f-fitdz" type="checkbox" checked> fit z-step scale</label>
      <label><input id="f-drift" type="checkbox"> fit xy drift (per stack)</label>
      <label>held out<select id="f-holdout"><option value="last-bead">last bead</option><option value="last-stack">last stack</option><option value="none">none</option></select></label>
    </fieldset>
    <div class="buttons"><button id="run">Fit PSF</button><button id="enough">Use what we have</button><button id="abort">Abort</button></div>
    <div id="progress"><div id="pbar"></div></div><p id="ptext" class="note"></p>
    <div id="results" hidden>
      <h3>Result</h3>
      <table id="res-table"></table>
      <canvas id="zern" width="460" height="150"></canvas>
      <p class="note">Zernike coefficients (waves rms) by Noll index</p>
      <label>bead <select id="bead"></select></label> <label>plane <input id="plane" type="range" min="0" max="100" value="50"></label>
      <div class="panels small">
        <figure><canvas id="b-data" width="120" height="120"></canvas><figcaption>data xy</figcaption></figure>
        <figure><canvas id="b-model" width="120" height="120"></canvas><figcaption>model xy</figcaption></figure>
        <figure><canvas id="b-res" width="120" height="120"></canvas><figcaption>residual</figcaption></figure>
        <figure><canvas id="b-dxz" width="120" height="200"></canvas><figcaption>data xz</figcaption></figure>
        <figure><canvas id="b-mxz" width="120" height="200"></canvas><figcaption>model xz</figcaption></figure>
        <figure><canvas id="b-rxz" width="120" height="200"></canvas><figcaption>residual xz</figcaption></figure>
      </div>
      <p id="b-label" class="note"></p>
      <h3>Export</h3>
      <div class="exp">${num('e-focus', 'imaging focus (um)', 0.0, 'any', 'focal-plane position above the coverslip for the exported PSF / spline')}
        ${num('e-roi', 'spline ROI (px)', 13, '1')}${num('e-zmin', 'spline z from (um)', 0)}${num('e-zmax', 'to (um)', 0.8)}</div>
      <div class="buttons"><button id="x-json">PSF json</button><button id="x-knots">spline knots .npz</button><button id="x-tif">model z-stack .tif</button></div>
      <p id="x-status" class="note"></p>
    </div>
  </section>
</main>
<footer>Everything runs in your browser (WebGPU); no data leave the machine. Python reference: torch-psf.</footer>`;
