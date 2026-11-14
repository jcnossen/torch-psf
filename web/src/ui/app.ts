// The web app (UI thread): presets + 3D view, the bead-fit form, progress / stop, results and exports.
// All GPU work runs in the compute worker (../worker.ts).
import { View3D, Volume } from './view3d';
import { drawArray } from './colormap';
import { DEFAULT_NOLL, Optics } from '../psf/vectorial';
import { makePsfJson } from '../io/psfjson';
import { WEBGPU_HELP } from '../gpu/device';
import type { FitForm, FitSummary, VolumeSpec, Req, MovieForm } from './protocol';
import { Recon } from './recon';
import { locsToCsv, MovieSummary, NCOL } from '../smlm/movie';
import type { ViewLook } from './view3d';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------- worker RPC
const worker = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
let nextId = 1;
const pending = new Map<number, { res: (v: any) => void; rej: (e: any) => void }>();
let onProgress: (p: any) => void = () => {};
let onLocs: (rows: Float32Array) => void = () => {};
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'progress') { onProgress(m.p); return; }
  if (m.type === 'locs') { onLocs(m.rows); return; }
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
  // foldable panels: the open / closed state is remembered per panel (localStorage may be unavailable)
  for (const d of root.querySelectorAll<HTMLDetailsElement>('details.panel')) {
    const key = 'tpsf.panel.' + d.dataset.key;
    try { const v = localStorage.getItem(key); if (v !== null) d.open = v === '1'; } catch { /* no storage */ }
    d.addEventListener('toggle', () => { try { localStorage.setItem(key, d.open ? '1' : '0'); } catch { /* no storage */ } });
  }
  const view = new View3D($('view3d'));
  (window as any).__view = view;
  let fitResult: FitSummary | null = null;
  let current: { vol: Volume; spec: VolumeSpec } | null = null;
  let refreshPsfInfo = () => {};

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
    refreshPsfInfo();
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

  view.onSliceChange = (axis, f) => { $<HTMLInputElement>(['sx', 'sy', 'sz'][axis]).value = String(f * 100); drawSections(); };
  for (const [i, id] of ['sx', 'sy', 'sz'].entries()) {
    $<HTMLInputElement>(id).oninput = (e) => { view.setSlice(i as 0 | 1 | 2, +(e.target as HTMLInputElement).value / 100); drawSections(); };
  }
  $<HTMLInputElement>('zexag').oninput = (e) => { view.setZExag(+(e.target as HTMLInputElement).value); $('zexag-v').textContent = view.opts.zExag.toFixed(1); };
  const lookSel = $<HTMLSelectElement>('look');
  try { const v = localStorage.getItem('tpsf.look'); if (v === 'slices' || v === 'volume') lookSel.value = v; } catch { /* no storage */ }
  view.opts.look = lookSel.value as ViewLook;
  lookSel.onchange = () => { view.setLook(lookSel.value as ViewLook); try { localStorage.setItem('tpsf.look', lookSel.value); } catch { /* no storage */ } };
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
  let running: 'stack' | 'movie' = 'stack';
  onProgress = (p) => {
    const frac = p.total ? Math.min(1, p.done / p.total) : 0;
    const [bar, txt] = running === 'movie' ? ['mbar', 'mtext'] : ['pbar', 'ptext'];
    $(bar).style.width = (100 * frac).toFixed(1) + '%';
    if (running === 'stack') enoughBtn.textContent = p.stage === 'read' ? 'Enough frames' : p.stage === 'fit' ? 'Stop and keep current estimate' : 'Use what we have';
    const info = Object.entries(p.info ?? {}).map(([k, v]) => `${k} ${typeof v === 'number' ? (Math.abs(v) < 1e-2 || Math.abs(v) > 1e4 ? v.toExponential(2) : v.toFixed(5)) : v}`).join(', ');
    const t = p.elapsed_s !== undefined ? ` | ${p.elapsed_s.toFixed(1)} s` + (p.eta_s !== undefined ? `, ~${p.eta_s.toFixed(1)} s left (at most)` : '') : '';
    $(txt).textContent = `${p.stage}: ${p.done} / ${p.total}${info ? ' | ' + info : ''}${t}`;
  };
  runBtn.onclick = async () => {
    running = 'stack';
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

  const n = (id: string) => +$<HTMLInputElement>(id).value;
  const readOptics = () => ({ NA: n('f-na'), wavelength_um: n('f-wl'), n_imm: n('f-nimm'), n_med: n('f-nmed'), pixel_um: n('f-px') });

  // --- mode switch (shared optics / camera; mode-specific inputs and result panels)
  const setMode = (m: string) => {
    $('mode-stack').hidden = m !== 'stack';
    $('mode-movie').hidden = m !== 'movie';
    $('results').hidden = m !== 'stack' || !fitResult;
    $('p-recon').hidden = m !== 'movie' || !recon;
    $('p-locs').hidden = m !== 'movie' || !recon;
    try { localStorage.setItem('tpsf.mode', m); } catch { /* no storage */ }
  };
  for (const r of root.querySelectorAll<HTMLInputElement>('input[name=mode]')) r.onchange = () => setMode(r.value);
  try {
    const m0 = localStorage.getItem('tpsf.mode');
    if (m0 === 'movie') { (root.querySelector('input[name=mode][value=movie]') as HTMLInputElement).checked = true; setTimeout(() => setMode('movie')); }
  } catch { /* no storage */ }

  // --- SMLM movie: detect and localize with the PSF of the 3D view or a PSF json, reconstruction, csv
  let mfiles: File[] = [];
  let recon: Recon | null = null;
  let psfJson: { noll: number[]; zcoef: number[]; name: string } | null = null;
  const mdrop = $('mdrop');
  const setMFiles = (fl: FileList | File[]) => {
    mfiles = [...fl].filter((f) => /\.tiff?$/i.test(f.name));
    $('mfiles').textContent = mfiles.length ? mfiles.map((f) => `${f.name} (${(f.size / 2 ** 20).toFixed(1)} MB)`).join(', ') : 'no TIFF selected';
  };
  (window as any).__setMovieFiles = setMFiles;
  mdrop.ondragover = (e) => { e.preventDefault(); mdrop.classList.add('over'); };
  mdrop.ondragleave = () => mdrop.classList.remove('over');
  mdrop.ondrop = (e) => { e.preventDefault(); mdrop.classList.remove('over'); if (e.dataTransfer) setMFiles(e.dataTransfer.files); };
  $<HTMLInputElement>('mfile').onchange = (e) => setMFiles((e.target as HTMLInputElement).files!);
  const psfSel = $<HTMLSelectElement>('m-psf');
  const psfInfo = () => {
    if (psfSel.value === 'json') { $('m-psfinfo').textContent = psfJson ? `${psfJson.name}: Zernikes ${fmtZ(psfJson.noll, psfJson.zcoef)}` : 'choose a psf.json'; return; }
    $('m-psfinfo').textContent = current ? `${sel.options[sel.selectedIndex]?.text ?? ''}: Zernikes ${fmtZ(current.spec.noll, current.spec.zcoef)}` : '';
  };
  refreshPsfInfo = psfInfo;
  psfSel.onchange = () => { $('m-psfjson').hidden = psfSel.value !== 'json'; psfInfo(); };
  sel.addEventListener('change', psfInfo);
  $<HTMLInputElement>('m-psfjson').onchange = async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    try {
      const j = JSON.parse(await f.text());
      const noll = j.zernike_noll ?? j.psf?.zernike_noll, zw = j.zernike_waves ?? j.psf?.zernike_waves;
      if (!Array.isArray(noll) || !Array.isArray(zw)) throw new Error('no zernike_noll / zernike_waves');
      psfJson = { noll: noll.map(Number), zcoef: zw.map(Number), name: f.name };
    } catch (err) { psfJson = null; $('m-psfinfo').textContent = 'not a torch-psf PSF json: ' + (err as Error).message; return; }
    psfInfo();
  };
  const mRun = $<HTMLButtonElement>('m-run'), mEnough = $<HTMLButtonElement>('m-enough'), mAbort = $<HTMLButtonElement>('m-abort');
  const mBusy = (b: boolean) => { mRun.disabled = b; mEnough.disabled = !b; mAbort.disabled = !b; runBtn.disabled = b; };
  mBusy(false);
  mEnough.onclick = () => worker.postMessage({ type: 'stop', signal: 'enough' });
  mAbort.onclick = () => worker.postMessage({ type: 'stop', signal: 'abort' });
  let lastDraw = 0, drawTimer = 0;
  const drawRecon = () => {
    if (!recon) return;
    const r = recon.render($('recon') as HTMLCanvasElement, { pixel_nm: n('r-px'), blur_px: n('r-blur'), colorZ: $<HTMLInputElement>('r-colorz').checked,
      zlo: n('m-zlo'), zhi: n('m-zhi'), maxCrlbNm: n('r-crlb') });
    $('r-label').textContent = `${r.shown} of ${recon.n} localizations, ${r.w} x ${r.h} pixels of ${n('r-px')} nm` +
      ($<HTMLInputElement>('r-colorz').checked ? `; colour = z from ${n('m-zlo')} (blue) to ${n('m-zhi')} um (red)` : '');
    lastDraw = performance.now();
  };
  for (const id of ['r-px', 'r-blur', 'r-crlb', 'r-colorz']) $(id).addEventListener('change', drawRecon);
  onLocs = (rows) => {
    recon?.add(rows);
    if (performance.now() - lastDraw > 700) drawRecon();
    else { clearTimeout(drawTimer); drawTimer = window.setTimeout(drawRecon, 700); }
  };
  $('r-csv').onclick = () => { if (recon) download(locsToCsv(recon.rows()), 'localizations.csv', 'text/csv'); };
  mRun.onclick = async () => {
    if (!mfiles.length) { $('mtext').textContent = 'Drop or choose a movie TIFF first.'; return; }
    const src = psfSel.value === 'json' ? psfJson : current ? { noll: current.spec.noll, zcoef: current.spec.zcoef } : null;
    if (!src) { $('mtext').textContent = psfSel.value === 'json' ? 'Choose a PSF json first.' : 'No PSF in the 3D view yet.'; return; }
    const P = Math.round(n('m-roi'));
    if (P % 2 !== 1 || P < 5) { $('mtext').textContent = 'The ROI size must be odd (>= 5).'; return; }
    const form: MovieForm = { optics: readOptics(), gain: n('f-gain'), offset: n('f-offset'), readnoise_e: n('m-rn'),
      focus_um: n('m-focus'), zlo: n('m-zlo'), zhi: n('m-zhi'), roisize: P, threshold: n('m-thr'), isolation: Math.round(n('m-iso')),
      chunk: Math.round(n('m-chunk')), maxFrames: Math.round(n('m-frames')), npupil: 64, oversample: 2, noll: src.noll, zcoef: src.zcoef };
    running = 'movie';
    mBusy(true);
    recon = null;
    $('mtext').textContent = 'starting...';
    try {
      // the reconstruction needs the frame size: read it from the TIFF header in the worker's first progress message;
      // here a provisional size from the first locs is avoided by asking the worker (movie header) first
      const hdr = await call({ type: 'movieHeader', files: mfiles });
      recon = new Recon(hdr.H, hdr.W, form.optics.pixel_um);
      $('p-recon').hidden = false; $('p-locs').hidden = false;
      const r = await call({ type: 'movie', files: mfiles, form });
      const s = r.summary as MovieSummary;
      drawRecon();
      showLocs(s);
      $('mbar').style.width = '100%';
      $('mtext').textContent = `done: ${s.n_locs} localizations from ${s.frames_read} frames in ${s.seconds.toFixed(1)} s` + (s.partial ? ' (stopped early)' : '');
      (window as any).__movieDone = s;
    } catch (e) {
      const err = e as Error & { abort?: boolean };
      $('mtext').textContent = err.abort ? 'aborted' : 'error: ' + err.message;
      $('mbar').style.width = '0%';
    } finally { mBusy(false); }
  };
  function showLocs(s: MovieSummary) {
    const rows = recon!.rows(), N = rows.length / NCOL;
    const med = (k: number, f = 1) => { const a = new Float32Array(N); for (let i = 0; i < N; i++) a[i] = rows[i * NCOL + k] * f; a.sort(); return N ? a[N >> 1] : NaN; };
    const px = n('f-px') * 1000;
    const t: [string, string][] = [
      ['localizations', `${s.n_locs} (${(s.n_locs / Math.max(1, s.frames_read)).toFixed(2)} per frame)`],
      ['frames', `${s.frames_read} of ${s.frames_total}${s.partial ? ' (stopped early)' : ''}, ${s.W} x ${s.H} px`],
      ['median photons / bg', `${med(4).toFixed(0)} / ${med(5).toFixed(1)}`],
      ['median CRLB x / y / z', `${med(6, px).toFixed(1)} / ${med(7, px).toFixed(1)} / ${med(8, 1000).toFixed(1)} nm`],
      ['at the offset limit', (100 * s.at_limit_fraction).toFixed(2) + ' %'],
      ['time', `${s.seconds.toFixed(1)} s (${s.spots_per_s.toFixed(0)} spots / s, including reading)`],
    ];
    $('l-table').innerHTML = t.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
  }

  function readForm(): FitForm {
    return {
      optics: readOptics(),
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
    psfInfo();
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
  function download(bytes: BlobPart, name: string, type = 'application/octet-stream') {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([bytes], { type }));
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }
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

const fmtZ = (noll: number[], z: number[]) => noll.map((j, i) => [j, z[i]] as [number, number]).filter(([, v]) => Math.abs(v) >= 0.005)
  .map(([j, v]) => `${j}: ${v.toFixed(3)}`).join(', ') || 'none';

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
  <div class="col">
  <details class="panel" data-key="view" open><summary><h2>3D PSF</h2></summary>
    <div class="bar">
      <label>PSF <select id="preset"></select></label>
      <label>view <select id="look"><option value="slices">slices</option><option value="volume">volume</option></select></label>
      <label>z exaggeration <input id="zexag" type="range" min="0.25" max="4" step="0.25" value="1"><span id="zexag-v">1.0</span></label>
      <label>log decades <input id="decades" type="number" min="1" max="6" value="3" style="width:3em"></label>
    </div>
    <div id="view3d"></div>
    <div class="bar">slice planes (or drag them): <label>x <input id="sx" type="range" min="0" max="100" step="0.1" value="50"></label>
      <label>y <input id="sy" type="range" min="0" max="100" step="0.1" value="50"></label>
      <label>z <input id="sz" type="range" min="0" max="100" step="0.1" value="50"></label></div>
    <p id="preset-note" class="note"></p>
  </details>
  <details class="panel" data-key="sections" open><summary><h2>Sections</h2></summary>
    <div class="panels">
      <figure><canvas id="sec-xy" width="200" height="200"></canvas><figcaption>xy</figcaption></figure>
      <figure><canvas id="sec-xz" width="200" height="200"></canvas><figcaption>xz</figcaption></figure>
      <figure><canvas id="sec-yz" width="200" height="200"></canvas><figcaption>yz</figcaption></figure>
    </div>
    <p id="sec-label" class="note"></p>
  </details>
  <details class="panel" data-key="recon" open id="p-recon" hidden><summary><h2>Reconstruction</h2></summary>
    <div class="bar">
      ${num('r-px', 'pixel (nm)', 20)}${num('r-blur', 'blur (pixels)', 0.7)}${num('r-crlb', 'max CRLB x (nm, 0 = all)', 0)}
      <label><input id="r-colorz" type="checkbox" checked> colour by z</label>
      <button id="r-csv">localizations .csv</button>
    </div>
    <div class="recon-wrap"><canvas id="recon" width="10" height="10"></canvas></div>
    <p id="r-label" class="note"></p>
  </details>
  </div>
  <div class="col">
  <details class="panel" data-key="input" open><summary><h2>Input</h2></summary>
    <fieldset><legend>Optics (all modes)</legend>
      ${num('f-na', 'NA', 1.49)}${num('f-wl', 'wavelength (um)', 0.68)}${num('f-nimm', 'n immersion', 1.518)}${num('f-nmed', 'n medium', 1.33)}
      ${num('f-px', 'pixel size (um)', 0.1)}
    </fieldset>
    <fieldset><legend>Camera</legend>${num('f-gain', 'gain (e-/ADU)', 0.5)}${num('f-offset', 'offset (ADU)', 100)}</fieldset>
    <div class="modes" role="radiogroup">
      <label><input type="radio" name="mode" value="stack" checked> Bead z-stack &rarr; PSF</label>
      <label><input type="radio" name="mode" value="movie"> SMLM movie &rarr; localizations</label>
    </div>
    <div id="mode-stack">
      <div id="drop" class="drop"><input id="file" type="file" multiple accept=".tif,.tiff"><p>Drop bead z-stack TIFF(s) here or choose files. Several stacks: one file each; a multi-part stack: movie.tif, movie_X1.tif, ...</p><p id="files" class="note">no TIFF selected</p></div>
      <fieldset><legend>Stage</legend>
        ${num('f-dz', 'z step (um)', 0.02)}
        <label>stage sign<select id="f-sign"><option value="1">+1</option><option value="-1">-1</option></select></label>
        ${num('f-ze', 'bead centre height (um)', 0.05, 'any', 'emitter height of the (point) bead above the coverslip')}
      </fieldset>
      <fieldset><legend>Fit</legend>
        ${num('f-roi', 'ROI (px)', 26, '1')}${num('f-npupil', 'pupil samples', 64, '1')}${num('f-os', 'oversampling', 2, '1')}
        ${num('f-iter', 'max iterations', 100, '1')}${num('f-minrel', 'bead threshold (rel.)', 0.3)}
        <label><input id="f-fitdz" type="checkbox" checked> fit z-step scale</label>
        <label><input id="f-drift" type="checkbox"> fit xy drift (per stack)</label>
        <label>held out<select id="f-holdout"><option value="last-bead">last bead</option><option value="last-stack">last stack</option><option value="none">none</option></select></label>
      </fieldset>
      <div class="buttons"><button id="run">Fit PSF</button><button id="enough">Use what we have</button><button id="abort">Abort</button></div>
      <div class="progress"><div id="pbar"></div></div><p id="ptext" class="note"></p>
    </div>
    <div id="mode-movie" hidden>
      <div id="mdrop" class="drop"><input id="mfile" type="file" multiple accept=".tif,.tiff"><p>Drop an SMLM movie (DNA-PAINT / dSTORM TIFF; a multi-part movie.tif, movie_X1.tif, ... is joined). It is streamed, not loaded at once.</p><p id="mfiles" class="note">no TIFF selected</p></div>
      <fieldset><legend>Sample</legend>
        ${num('m-focus', 'focal plane (um)', 0.4, 'any', 'focal-plane position above the coverslip during the acquisition')}
        ${num('m-zlo', 'emitter z from (um)', 0)}${num('m-zhi', 'to (um)', 0.8)}${num('m-rn', 'read noise (e- rms)', 0, 'any', 'sCMOS read noise; 0 = Poisson only')}
      </fieldset>
      <fieldset><legend>PSF for localization</legend>
        <label>source <select id="m-psf"><option value="view">the PSF in the 3D view</option><option value="json">a PSF json file</option></select></label>
        <input id="m-psfjson" type="file" accept=".json" hidden>
        <span id="m-psfinfo" class="note"></span>
      </fieldset>
      <fieldset><legend>Detection and fit (torch_psf defaults)</legend>
        ${num('m-roi', 'ROI (px, odd)', 15, '2')}${num('m-thr', 'threshold (x noise)', 6)}${num('m-iso', 'isolation (px, 0 = off)', 0, '1')}
        ${num('m-chunk', 'frames per chunk', 500, '1', 'the noise is estimated per chunk (torch_psf default 500)')}${num('m-frames', 'max frames (0 = all)', 0, '1')}
      </fieldset>
      <div class="buttons"><button id="m-run">Detect and localize</button><button id="m-enough">Enough frames</button><button id="m-abort">Abort</button>
        <button id="m-insitu" disabled title="next step: in-situ PSF estimation from the spots (knot route) and optional spline refinement">Estimate PSF in situ (next)</button></div>
      <div class="progress"><div id="mbar"></div></div><p id="mtext" class="note"></p>
    </div>
  </details>
  <details class="panel" data-key="results" open id="results" hidden><summary><h2>Bead fit result</h2></summary>
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
  </details>
  <details class="panel" data-key="locs" open id="p-locs" hidden><summary><h2>Localizations</h2></summary>
    <table id="l-table"></table>
  </details>
  </div>
</main>
<footer>Everything runs in your browser (WebGPU); no data leave the machine. Python reference: torch-psf.</footer>`;
