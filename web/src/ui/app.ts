// The web app (UI thread), laid out as a workflow: a step bar on top (Input > Detect spots > Fit PSF > Localize >
// Export for an SMLM movie; Input > Fit PSF > Export for a bead z-stack; Input > Export for a localization file),
// the settings of the selected step in the left sidebar, and the views in cards: the movie and the reconstruction
// side by side with one shared zoom / pan, a compact PSF view, and the fit summary. Selecting a step outlines the
// cards it feeds. All GPU work runs in the compute worker (../worker.ts).
import { View3D, Volume } from './view3d';
import { drawArray } from './colormap';
import { DEFAULT_NOLL, Optics } from '../psf/vectorial';
import { makePsfJson } from '../io/psfjson';
import { readPsfFile } from '../io/psfzip';
import { beadZipRequest, insituZipRequest, ZipOptions } from './exportzip';
import type { PsfZipRequest } from '../psf/makezip';
import { reportSections } from '../report/sections';
import { WEBGPU_HELP, getDevice } from '../gpu/device';
import { TiffMovie } from '../io/tiff';
import type { FitForm, FitSummary, VolumeSpec, Req, MovieForm, InsituForm, MovieSourceSpec } from './protocol';
import { InsituReport, FREE_KNOT_WARNING } from '../smlm/estimate';
import { splineGrid } from '../gpu/spline_lm';
import { Recon } from './recon';
import { readLocFile, isLocFile } from '../io/locfile';
import { PointCloudView } from './pointcloud';
import { locsToCsv, MovieSummary, NCOL } from '../smlm/movie';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------- worker RPC
const worker = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
let nextId = 1;
const pending = new Map<number, { res: (v: any) => void; rej: (e: any) => void }>();
let onProgress: (p: any) => void = () => {};
let onLocs: (rows: Float32Array) => void = () => {};
let onInsitu: (r: InsituReport) => void = () => {};
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'progress') { onProgress(m.p); return; }
  if (m.type === 'locs') { onLocs(m.rows); return; }
  if (m.type === 'insitu') { onInsitu(m.report); return; }
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

// ---------------------------------------------------------------- PSF presets (3D view)
interface Preset { name: string; label: string; spec: VolumeSpec; note: string; }
const OIL: Optics = { pixel_um: 0.1, NA: 1.49, wavelength_um: 0.68, n_imm: 1.518, n_med: 1.33 };
const zc = (m: Record<number, number>) => DEFAULT_NOLL.map((j) => m[j] ?? 0);
const PRESETS: Preset[] = [
  { name: 'astig', label: 'Astigmatic 3D',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({ 6: 0.1 }), nxy: 48, dxy_um: 0.04, zmin: -0.6, zmax: 0.6, nz: 61, ze_um: 0 },
    note: 'Astigmatic PSF: Noll 6 = 0.10 waves rms (a weak cylindrical lens). NA 1.49, 680 nm; z = defocus of an emitter at the coverslip, range 1.2 um.' },
  { name: 'tetrapod', label: 'Tetrapod-like',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({ 6: 0.2, 12: -0.2 }), nxy: 64, dxy_um: 0.075, zmin: -1.6, zmax: 1.6, nz: 65, ze_um: 0 },
    note: 'Tetrapod-like PSF approximated by a Zernike combination: Noll 6 = +0.20, Noll 12 = -0.20 waves rms (not the published optimized tetrapod pupil). NA 1.49, 680 nm; z = defocus of an emitter at the coverslip, range 3.2 um.' },
  { name: 'focus', label: 'In-focus 2D',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({}), nxy: 40, dxy_um: 0.04, zmin: -0.4, zmax: 0.4, nz: 41, ze_um: 0 },
    note: 'Aberration-free vectorial PSF, NA 1.49, 680 nm, range 0.8 um.' },
];
const START_GUESS: Record<number, number> = { 6: 0.08 };   // the default start of the in-situ fit: a generic astigmatic PSF

// ---------------------------------------------------------------- workflow
type Mode = 'movie' | 'stack' | 'locs';
type Step = 'input' | 'detect' | 'fit' | 'localize' | 'export';
const STEPS: Record<Mode, [Step, string][]> = {
  movie: [['input', 'Input'], ['detect', 'Detect spots'], ['fit', 'Fit PSF'], ['localize', 'Localize'], ['export', 'Export']],
  stack: [['input', 'Input'], ['fit', 'Fit PSF'], ['export', 'Export']],
  locs: [['input', 'Input'], ['export', 'Export']],
};
/** the cards each step feeds (outlined while the step is selected) */
const FEEDS: Record<string, string[]> = {
  'movie.input': ['c-movie'], 'movie.detect': ['c-movie'], 'movie.fit': ['c-psf', 'c-fit'], 'movie.localize': ['c-recon'], 'movie.export': ['c-recon', 'c-fit'],
  'stack.input': ['c-psf'], 'stack.fit': ['c-psf', 'c-fit'], 'stack.export': ['c-fit'], 'locs.input': ['c-recon'], 'locs.export': ['c-recon'],
};
const CARDS: Record<Mode, string[]> = { movie: ['c-movie', 'c-recon', 'c-psf', 'c-fit'], stack: ['c-psf', 'c-fit'], locs: ['c-recon'] };

// ---------------------------------------------------------------- app
export function startApp(root: HTMLElement) {
  root.innerHTML = TEMPLATE;
  for (const d of root.querySelectorAll<HTMLDetailsElement>('details[data-key]')) {
    const key = 'tpsf.panel.' + d.dataset.key;
    try { const v = localStorage.getItem(key); if (v !== null) d.open = v === '1'; } catch { /* no storage */ }
    d.addEventListener('toggle', () => { try { localStorage.setItem(key, d.open ? '1' : '0'); } catch { /* no storage */ } });
  }
  const view = new View3D($('view3d'));
  (window as any).__view = view;
  const n = (id: string) => +$<HTMLInputElement>(id).value;
  const setNum = (id: string, v: number) => { $<HTMLInputElement>(id).value = String(+v.toPrecision(10)); };
  const pxUm = () => n('o-px') / 1000;
  const optics = (): Optics => ({ pixel_um: pxUm(), NA: n('o-na'), wavelength_um: n('o-wl') / 1000, n_imm: n('o-nimm'), n_med: n('o-nmed') });
  const note = (id: string, s: string) => { const e = $(id); e.textContent = s; e.title = s; };
  let fitResult: FitSummary | null = null;
  let insitu: InsituReport | null = null;
  let current: { vol: Volume; spec: VolumeSpec } | null = null;
  let truthZ: Record<number, number> | null = null;      // the demo's true Zernikes
  let locsDone = false;

  // ================================================================ step bar, modes
  let mode: Mode = 'movie', step: Step = 'input';
  const stepDone = (s: Step): boolean => {
    if (mode === 'movie') return s === 'input' ? !!player : s === 'detect' || s === 'fit' ? !!insitu : s === 'localize' ? locsDone : false;
    if (mode === 'stack') return s === 'input' ? files.length > 0 : s === 'fit' ? !!fitResult : false;
    return s === 'input' ? !!recon && recon.n > 0 : false;
  };
  function renderSteps() {
    const CHECK = '<svg viewBox="0 0 20 20"><path d="M5.5 10.2l3 3 6-6.6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    $('steps').innerHTML = STEPS[mode].map(([k, lab], i) =>
      `${i ? '<span class="chev" aria-hidden="true"></span>' : ''}<button class="step${stepDone(k) ? ' done' : ''}" data-step="${k}" ${k === step ? 'aria-current="step"' : ''}>` +
      `<span class="dot">${stepDone(k) ? CHECK : i + 1}</span><span class="sl">${lab}</span></button>`).join('');
    for (const b of $('steps').querySelectorAll<HTMLButtonElement>('button.step')) b.onclick = () => setStep(b.dataset.step as Step);
    const nav = $('steps'), cur = nav.querySelector<HTMLElement>('[aria-current]');   // narrow screens: keep the current step in view
    if (cur && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = cur.offsetLeft - nav.offsetLeft - (nav.clientWidth - cur.offsetWidth) / 2;
  }
  const inMode = (el: HTMLElement) => !el.dataset.mode || el.dataset.mode.split(' ').includes(mode);
  function applyLayout() {
    for (const s of root.querySelectorAll<HTMLElement>('.side-body > section')) s.hidden = s.dataset.step !== step || !inMode(s);
    for (const el of root.querySelectorAll<HTMLElement>('[data-mode]:not(.side-body > section)')) el.hidden = !inMode(el);
    for (const id of ['c-movie', 'c-recon', 'c-psf', 'c-fit']) $(id).hidden = !CARDS[mode].includes(id);
    $('c-recon').classList.toggle('wide', mode === 'locs');
    const feeds = FEEDS[`${mode}.${step}`] ?? [];
    for (const id of ['c-movie', 'c-recon', 'c-psf', 'c-fit']) $(id).classList.toggle('feed', feeds.includes(id));
    $('tab-movie').hidden = mode !== 'movie'; $('tab-stack').hidden = mode !== 'stack'; $('tab-locs').hidden = mode !== 'locs';
    for (const b of root.querySelectorAll<HTMLButtonElement>('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === mode));
    const nb = $<HTMLButtonElement>('next'), labels: Partial<Record<Step, string>> = { input: 'Next: Detect spots  →', detect: 'Next: Fit PSF  →', fit: 'Estimate PSF', localize: 'Localize' };
    nb.textContent = labels[step] ?? ''; nb.hidden = !labels[step];
    $<HTMLButtonElement>('m-run').hidden = step === 'fit';
    const sb = $<HTMLButtonElement>('snext'); sb.textContent = 'Next: Fit PSF  →'; sb.hidden = step !== 'input';
    $<HTMLButtonElement>('run').classList.toggle('primary', step !== 'input');
    $('fit-title').textContent = mode === 'stack' ? 'Bead fit' : 'PSF fit';
    // the run controls (progress, next / run all, stop) sit right under the title of the current step
    const sec = [...root.querySelectorAll<HTMLElement>('.side-body > section')].find((s) => !s.hidden);
    const anchor = sec?.querySelector<HTMLElement>(':scope > .sub') ?? sec?.querySelector<HTMLElement>(':scope > h2');
    if (anchor && anchor.nextElementSibling !== $('runbox')) anchor.after($('runbox'));
    const first = STEPS[mode][0][0] === step;
    $<HTMLButtonElement>('back').hidden = first; $<HTMLButtonElement>('sback').hidden = first;
    renderSteps();
    cloud?.request(); renderMovie();
  }
  function setStep(s: Step) { step = s; applyLayout(); try { localStorage.setItem('tpsf.step', s); } catch { /* no storage */ } }
  const prevStep = () => { const k = STEPS[mode].findIndex(([s]) => s === step); if (k > 0) setStep(STEPS[mode][k - 1][0]); };
  for (const id of ['back', 'sback']) $(id).onclick = prevStep;
  function setMode(m: Mode, keepStep = false) {
    mode = m;
    if (!keepStep || !STEPS[m].some(([k]) => k === step)) step = 'input';
    applyLayout();
    try { localStorage.setItem('tpsf.tab', m); } catch { /* no storage */ }
  }
  for (const b of root.querySelectorAll<HTMLButtonElement>('.tabs button')) b.onclick = () => setMode(b.dataset.tab as Mode);
  (window as any).__setStep = setStep; (window as any).__setMode = setMode;

  // --- WebGPU, then the demo movie and the default view
  const status = $('gpu-status');
  call({ type: 'init' }).then(
    (r) => { status.textContent = 'WebGPU · ' + r.gpu; status.title = r.gpu; status.dataset.ok = '1'; showPreset('astig'); loadDemo(); },
    (e) => {
      status.textContent = 'WebGPU unavailable'; status.dataset.ok = '0';
      $('nogpu').hidden = false; $('nogpu').textContent = String(e.message || e) || WEBGPU_HELP;
      for (const b of ['m-run', 'run', 'next', 'snext']) $<HTMLButtonElement>(b).disabled = true;
    });

  // ================================================================ PSF card
  const sel = $<HTMLSelectElement>('preset');
  const fillPresets = () => {
    const v = sel.value;
    sel.innerHTML = PRESETS.map((p) => `<option value="${p.name}">${p.label}</option>`).join('') +
      `<option value="insitu" ${insitu ? '' : 'disabled'}>In-situ estimate</option><option value="refined" ${insitu?.refine?.knots ? '' : 'disabled'}>In-situ refined spline${insitu?.refine && !insitu.refine.kept ? ' (not kept)' : ''}</option><option value="fit" ${fitResult ? '' : 'disabled'}>Bead fit result</option>`;
    if (v) sel.value = v;
  };
  fillPresets();
  sel.onchange = () => showPreset(sel.value);
  async function showPreset(name: string) {
    if (name === 'fit') { if (fitResult) setVolume(fitResult.volume, fitResult.volumeSpec, beadNote(fitResult)); return; }
    if (name === 'insitu') { if (insitu) showInsituVolume(); return; }
    if (name === 'refined') { if (insitu?.refine?.knots) showRefinedVolume(); return; }
    const p = PRESETS.find((q) => q.name === name)!;
    sel.value = name;
    note('preset-note', 'computing...');
    const r = await call({ type: 'volume', spec: p.spec });
    setVolume(r.volume, p.spec, p.note);
    (window as any).__presetShown = name;
  }
  function setVolume(data: Float32Array, spec: VolumeSpec, txt: string) {
    const v: Volume = { data, nx: spec.nxy, ny: spec.nxy, nz: spec.nz, dx_um: spec.dxy_um, dz_um: (spec.zmax - spec.zmin) / Math.max(1, spec.nz - 1), z0_um: spec.zmin };
    current = { vol: v, spec };
    view.setVolume(v);
    note('preset-note', txt);
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
    note('sec-label', `xy at z = ${(1000 * z).toFixed(0)} nm; xz / yz through the slice planes (z up); log scale over ${dec} decades. Drag the planes in the 3D view.`);
  }
  view.onSliceChange = (axis, f) => { $<HTMLInputElement>(['sx', 'sy', 'sz'][axis]).value = String(f * 100); drawSections(); };
  for (const [i, id] of ['sx', 'sy', 'sz'].entries()) $<HTMLInputElement>(id).oninput = (e) => { view.setSlice(i as 0 | 1 | 2, +(e.target as HTMLInputElement).value / 100); drawSections(); };
  $<HTMLInputElement>('zexag').oninput = (e) => { view.setZExag(+(e.target as HTMLInputElement).value); $('zexag-v').textContent = view.opts.zExag.toFixed(1); };
  $<HTMLInputElement>('decades').onchange = (e) => { view.opts.decades = +(e.target as HTMLInputElement).value; view.updateSlices(); drawSections(); };
  $('psf-wide').onclick = () => { const w = $('c-psf').classList.toggle('wide'); $('psf-wide').classList.toggle('on', w); };

  // ================================================================ movie
  let source: MovieSourceSpec = 'demo';
  let player: { n: number; H: number; W: number; frame: (t: number) => Promise<Float32Array> } | null = null;
  let recon: Recon | null = null;
  let psfJson: { noll: number[]; zcoef: number[]; name: string } | null = null;

  let loadSeq = 0;      // a movie opened while the demo is still being simulated wins
  async function loadDemo() {
    const seq = ++loadSeq;
    note('movie-note', 'simulating the demo movie...');
    const d = await call({ type: 'demo' });
    if (seq !== loadSeq) return;
    truthZ = d.zernike;
    const fp = d.H * d.W, adu: Uint16Array = d.adu;
    player = { n: d.T, H: d.H, W: d.W, frame: async (t) => Float32Array.from(adu.subarray(t * fp, (t + 1) * fp)) };
    source = 'demo';
    const o: Optics = d.optics, p = d.params;
    setNum('o-px', 1000 * o.pixel_um); setNum('o-na', o.NA); setNum('o-wl', 1000 * o.wavelength_um); setNum('o-nimm', o.n_imm); setNum('o-nmed', o.n_med);
    setNum('o-gain', p.gain); setNum('o-offset', p.offset); setNum('o-rn', p.readnoise); setNum('m-focus', p.focus_um); setNum('m-zlo', p.zlo); setNum('m-zhi', p.zhi);
    setCamType(1);
    camBadge('known (simulation)', 'info');
    $('cam-note').textContent = 'The demo camera is known. It cannot be measured from this movie: photon transfer needs a background that varies across the field, and the demo background is uniform.';
    $('cam-est').hidden = true; $('cam-offknown').closest('label')!.hidden = true;
    $('mname').textContent = 'Simulated demo movie';
    $('mfiles').textContent = `${d.T} frames · ${d.W} × ${d.H} px · simulated in your browser`;
    note('movie-note', `Simulated demo (not real data): ${d.n_events} blinking events on ${p.sites === 'filaments' ? p.filaments + ' filaments' : p.nsites + ' random sites'}, astigmatic PSF with Noll 5 / 6 / 11 = ${d.zernike[5]} / ${d.zernike[6]} / ${d.zernike[11]} waves, ` +
      `focal plane ${p.focus_um} µm, emitters ${p.zlo}–${p.zhi} µm, ~${p.photons} photons, bg ${p.bg} / px, gain ${p.gain} e⁻/ADU, offset ${p.offset}, read noise ${p.readnoise} e⁻. ` +
      `Estimate PSF starts from a generic astigmatic guess (Noll 6 = ${START_GUESS[6]}), not from the truth.`);
    newMovie();
    (window as any).__demoReady = true;
  }
  function newMovie() {
    locsByFrame.clear(); detsByFrame.clear(); insitu = null; locsDone = false; recon = null; showInsituEmpty(); fillPresets();
    mview.scale = 0; mview.cx = 0; mview.cy = 0;
    if (cloud && player) cloud.start(player.W, player.H, n('o-px'));
    $('r-empty').hidden = false;
    startPlayer();
    applyLayout();
  }

  // --- the movie view: frames drawn into an offscreen W x H canvas, shown with the same nm-based view transform as the
  // reconstruction (scale = nm per device pixel, cx / cy = view centre relative to the field centre), so that zoom and
  // pan can be shared (Linked)
  const mcv = $('movie') as HTMLCanvasElement, slider = $<HTMLInputElement>('m-frame'), playBtn = $<HTMLButtonElement>('m-play');
  const frameCv = document.createElement('canvas');
  const mview = { scale: 0, cx: 0, cy: 0 };
  let linked = true, rings = true;
  let playing = false, frameNow = 0, playTimer = 0, lims: [number, number] | null = null, drawing = false, haveFrame = false;
  const locsByFrame = new Map<number, number[]>();
  const detsByFrame = new Map<number, number[]>();     // spots detected for the PSF fit (rings in another colour)
  // while a run is going the movie shows the frame the worker finished last, with its spots / localizations
  let follow = false, followAt = 0, followNext = -1, followTimer = 0;
  function followFrame(t: number) {
    if (!follow) return;
    followNext = t;
    const wait = 250 - (performance.now() - followAt);
    if (wait <= 0) { followAt = performance.now(); showFrame(t); followNext = -1; }
    else if (!followTimer) followTimer = window.setTimeout(() => { followTimer = 0; if (followNext >= 0) followFrame(followNext); }, wait);
  }
  const fitScale = (cw: number, ch: number) => player ? Math.max(player.W * n('o-px') / cw, player.H * n('o-px') / ch) * 1.02 : 1;
  let queuedFrame = -1;     // a frame asked for while another was being drawn: drawn next (the last request wins)
  async function showFrame(t: number) {
    if (!player) return;
    if (drawing) { queuedFrame = t; return; }
    drawing = true;
    try {
      frameNow = t; slider.max = String(player.n - 1); slider.value = String(t);
      const a = await player.frame(t);
      if (!lims) { const s = Float32Array.from(a).sort(); lims = [s[Math.floor(0.01 * s.length)], s[Math.floor(0.999 * s.length)]]; }
      const g = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) g[i] = Math.max(0, a[i] - lims[0]);
      frameCv.width = player.W; frameCv.height = player.H;
      drawArray(frameCv, g, player.W, player.H, { vmax: lims[1] - lims[0], gray: true });
      haveFrame = true;
      renderMovie();
      $('m-frame-v').textContent = `frame ${t + 1} / ${player.n}`;
    } finally {
      drawing = false;
      if (queuedFrame >= 0) { const q = queuedFrame; queuedFrame = -1; showFrame(q); }
    }
  }
  function renderMovie() {
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(16, Math.round(mcv.clientWidth * dpr)), H = Math.max(16, Math.round(mcv.clientHeight * dpr));
    if (mcv.width !== W || mcv.height !== H) { mcv.width = W; mcv.height = H; }
    const ctx = mcv.getContext('2d')!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#0e1013'; ctx.fillRect(0, 0, W, H);
    if (!player || !haveFrame || !mcv.clientWidth) return;
    const px = n('o-px'), fit = fitScale(W, H);
    if (!mview.scale) mview.scale = fit;
    const s = mview.scale, X0 = player.W * px / 2, Y0 = player.H * px / 2;
    const ox = (-X0 - mview.cx) / s + W / 2, oy = (-Y0 - mview.cy) / s + H / 2;
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(px / s, 0, 0, px / s, ox, oy);
    ctx.drawImage(frameCv, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (rings) {
      ctx.strokeStyle = '#4fd1c5'; ctx.lineWidth = 1.3 * dpr;
      const loc = locsByFrame.get(frameNow), L = loc ?? detsByFrame.get(frameNow) ?? [], r = Math.max(5 * dpr, 2.6 * px / s);
      if (!loc) ctx.strokeStyle = '#e8a33d';      // detections of the spot collection (Estimate PSF), not localizations
      for (let i = 0; i < L.length; i += 2) { ctx.beginPath(); ctx.arc(ox + (L[i] + 0.5) * px / s, oy + (L[i + 1] + 0.5) * px / s, r, 0, 2 * Math.PI); ctx.stroke(); }
    }
    overlay(ctx, W, H, s, fit, player.W * px, player.H * px, mview.cx, mview.cy, dpr);
  }
  function startPlayer() {
    lims = null; haveFrame = false;
    clearInterval(playTimer);
    playTimer = window.setInterval(() => { if (playing && player && !document.hidden) showFrame((frameNow + 1) % player.n); }, 70);
    showFrame(0); drawSpark();
  }
  const PLAY = '<svg viewBox="0 0 20 20"><path d="M7 5l8 5-8 5z" fill="currentColor"/></svg>', PAUSE = '<svg viewBox="0 0 20 20"><path d="M6 5h3v10H6zM11 5h3v10h-3z" fill="currentColor"/></svg>';
  const setPlaying = (p: boolean) => { playing = p; playBtn.innerHTML = p ? PAUSE : PLAY; };
  setPlaying(false);       // no autoplay: a blinking movie is restless; play on request
  playBtn.onclick = () => { follow = false; setPlaying(!playing); };
  slider.oninput = () => { follow = false; setPlaying(false); showFrame(+slider.value); };
  // pan / zoom (shared with the reconstruction when Linked)
  const pushView = () => { if (linked && cloud) { cloud.scale = mview.scale; cloud.cx = mview.cx; cloud.cy = mview.cy; cloud.request(); } renderMovie(); };
  {
    let last: { x: number; y: number } | null = null;
    mcv.addEventListener('pointerdown', (e) => { setPlaying(false); last = { x: e.clientX, y: e.clientY }; mcv.setPointerCapture(e.pointerId); });
    mcv.addEventListener('pointermove', (e) => {
      if (!last) return;
      const dpr = window.devicePixelRatio || 1, dx = e.clientX - last.x, dy = e.clientY - last.y; last = { x: e.clientX, y: e.clientY };
      mview.cx -= dx * dpr * mview.scale; mview.cy -= dy * dpr * mview.scale; pushView();
    });
    const end = (e: PointerEvent) => { last = null; if (mcv.hasPointerCapture(e.pointerId)) mcv.releasePointerCapture(e.pointerId); };
    mcv.addEventListener('pointerup', end); mcv.addEventListener('pointercancel', end);
    mcv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = mcv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      const mx = (e.clientX - r.left) * dpr - mcv.width / 2, my = (e.clientY - r.top) * dpr - mcv.height / 2;
      const f = Math.exp(e.deltaY * 0.0015);
      mview.cx += mx * mview.scale * (1 - f); mview.cy += my * mview.scale * (1 - f); mview.scale *= f; pushView();
    }, { passive: false });
    mcv.addEventListener('dblclick', () => fitView());
    new ResizeObserver(() => renderMovie()).observe(mcv);
  }
  const fitView = () => { mview.scale = 0; mview.cx = 0; mview.cy = 0; if (linked) cloud?.reset(); renderMovie(); };
  $('m-fit').onclick = fitView;
  $('m-link').onclick = () => { linked = !linked; $('m-link').classList.toggle('on', linked); if (linked) pushView(); };
  $('m-rings').onclick = () => { rings = !rings; $('m-rings').classList.toggle('on', rings); renderMovie(); };
  /** localizations per frame under the frame slider */
  function drawSpark() {
    const cv = $('m-spark') as HTMLCanvasElement, dpr = window.devicePixelRatio || 1;
    const W = Math.max(16, Math.round(cv.clientWidth * dpr)), H = Math.max(8, Math.round(cv.clientHeight * dpr));
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d')!; ctx.clearRect(0, 0, W, H);
    if (!player || !locsByFrame.size) return;
    const bins = new Float32Array(W);
    for (const [f, L] of locsByFrame) bins[Math.min(W - 1, Math.floor(f / player.n * W))] += L.length / 2;
    const m = Math.max(...bins) || 1;
    ctx.strokeStyle = '#9aa4ae'; ctx.lineWidth = dpr; ctx.beginPath();
    for (let x = 0; x < W; x++) { const y = H - 1 - (H - 2) * bins[x] / m; if (x) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
    ctx.stroke();
  }

  // --- movie files (replace the demo)
  const setMFiles = async (fl: FileList | File[]) => {
    const fs = [...fl].filter((f) => /\.tiff?$/i.test(f.name));
    if (!fs.length) return;
    ++loadSeq;
    source = fs; truthZ = null;
    const mv = await TiffMovie.open(fs);
    player = { n: mv.n, H: mv.H, W: mv.W, frame: (t) => mv.read(t, t + 1) };
    $('mname').textContent = fs.length > 1 ? `${fs[0].name} + ${fs.length - 1}` : fs[0].name;
    $('mfiles').textContent = `${mv.n} frames · ${mv.W} × ${mv.H} px · ${mv.dtype} · ${(fs.reduce((s, f) => s + f.size, 0) / 2 ** 20).toFixed(1)} MB`;
    note('movie-note', `${fs.map((f) => f.name).join(' + ')}: ${mv.n} frames of ${mv.W} x ${mv.H} px (${mv.dtype}).`);
    camBadge('', ''); $('cam-note').textContent = ''; $('cam-est').hidden = false; $('cam-offknown').closest('label')!.hidden = false;
    if (mode !== 'movie') setMode('movie');
    newMovie();
    estimateCamera();
  };
  (window as any).__setMovieFiles = setMFiles;
  const routeFiles = (fl: File[]) => { if (fl.some(isLocFile)) loadLocFiles(fl); else if (mode === 'stack') setFiles(fl); else setMFiles(fl); };
  for (const id of ['mdrop', 'drop', 'ldrop']) {
    const d = $(id);
    d.ondragover = (e) => { e.preventDefault(); d.classList.add('over'); };
    d.ondragleave = () => d.classList.remove('over');
  }
  $<HTMLInputElement>('mfile').onchange = (e) => routeFiles([...(e.target as HTMLInputElement).files!]);
  $<HTMLInputElement>('lfile').onchange = (e) => loadLocFiles([...(e.target as HTMLInputElement).files!]);
  $('m-demo').onclick = (e) => { e.preventDefault(); loadDemo(); };

  // --- camera: type (excess noise) and gain / offset measured from the movie (torch_psf photon_transfer)
  let camK: number | null = null;     // photon-transfer slope (ADU per ADU): gain = F^2 / k
  function setCamType(f2: number) {
    for (const b of $('cam-type').querySelectorAll<HTMLButtonElement>('button')) b.setAttribute('aria-selected', String(+b.dataset.f! === f2));
    setNum('o-excess', f2);
    if (camK) setNum('o-gain', +(f2 / camK).toPrecision(4));
  }
  for (const b of $('cam-type').querySelectorAll<HTMLButtonElement>('button')) b.onclick = () => setCamType(+b.dataset.f!);
  function camBadge(txt: string, kind: string) { const b = $('cam-badge'); b.textContent = txt; b.dataset.kind = kind; b.hidden = !txt; }
  async function estimateCamera() {
    if (!player) return;
    camBadge('estimating…', 'busy');
    try {
      const known = $<HTMLInputElement>('cam-offknown').checked;
      const r = await call({ type: 'camera', source, opts: { excessNoise: n('o-excess'), offsetAdu: known ? n('o-offset') : null } });
      const c = r.camera;
      camK = c.slope_adu;
      setNum('o-gain', +c.gain_e_per_adu.toPrecision(4));
      if (known) setNum('o-rn', +c.readnoise_e.toPrecision(3)); else setNum('o-offset', +c.offset_adu.toFixed(1));
      camBadge(known ? 'gain, read noise estimated' : 'gain, offset estimated', 'ok');
      $('cam-note').textContent = `Photon-transfer fit on background pixels (${c.per_block.length} blocks of ${c.block_used ?? ''} frames; slope spread ${(100 * c.slope_spread).toFixed(1)} %). ` +
        (known ? 'Read noise from the intercept, given the offset.' : 'Offset and read noise cannot be separated from the movie alone: the offset absorbs the read noise. Enter the read noise, or tick "offset known" to estimate it.');
      (window as any).__camera = c;
    } catch (e) {
      camBadge('not estimated', 'warn');
      $('cam-note').textContent = 'Camera estimate failed: ' + (e as Error).message;
    }
  }
  $('cam-est').onclick = (e) => { e.preventDefault(); estimateCamera(); };
  $('cam-offknown').onchange = () => { if (source !== 'demo') estimateCamera(); };

  // --- start PSF of the fit / the PSF for "localize only"
  const psfSel = $<HTMLSelectElement>('m-psf');
  psfSel.onchange = () => { if (psfSel.value === 'json') $<HTMLInputElement>('m-psfjson').click(); };
  $<HTMLInputElement>('m-psfjson').onchange = async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    try {
      const j = await readPsfFile(f);
      if (!Array.isArray(j.noll) || !Array.isArray(j.zcoef)) throw new Error('no zernike_noll / zernike_waves');
      psfJson = { noll: j.noll.map(Number), zcoef: j.zcoef.map(Number), name: f.name };
      $('m-psfinfo').textContent = `${f.name} (${j.kind}): ${fmtZ(psfJson.noll, psfJson.zcoef)}` + (j.notes.length ? ` [${j.notes.join('; ')}]` : '');
    } catch (err) { psfJson = null; $('m-psfinfo').textContent = 'not a torch-psf PSF file: ' + (err as Error).message; }
  };
  const startPsf = (): { noll: number[]; zcoef: number[] } | null => {
    if (psfSel.value === 'json') return psfJson;
    if (psfSel.value === 'view') return current ? { noll: current.spec.noll, zcoef: current.spec.zcoef } : null;
    if (psfSel.value === 'insitu') return insitu ? { noll: insitu.noll, zcoef: insitu.zcoef } : null;
    return { noll: DEFAULT_NOLL, zcoef: zc(START_GUESS) };
  };
  const movieForm = (psf: { noll: number[]; zcoef: number[] }): InsituForm => ({
    optics: optics(), gain: n('o-gain'), offset: n('o-offset'), readnoise_e: n('o-rn'), focus_um: n('m-focus'), zlo: n('m-zlo'), zhi: n('m-zhi'),
    roisize: Math.round(n('m-roi')), threshold: n('m-thr'), isolation: Math.round(n('m-iso')), chunk: Math.round(n('m-chunk')), maxFrames: Math.round(n('m-frames')),
    npupil: 64, oversample: 3, noll: psf.noll, zcoef: psf.zcoef, nspots: Math.round(n('m-nspots')), nheldout: Math.round(n('m-nheld')), iters: Math.round(n('m-iters')), tile: 16,
    rejectOverlap: n('m-devk') > 0, devianceK: n('m-devk'), refine: $<HTMLInputElement>('m-refine').checked, refineOuter: 12,
    sample: 'spread', sampleChunk: Math.round(n('m-schunk')), devianceRule: 'median', excessNoise: n('o-excess'),
  });

  // --- progress, stop
  let running: 'stack' | 'movie' = 'movie';
  const mRun = $<HTMLButtonElement>('m-run'), nextBtn = $<HTMLButtonElement>('next');
  const busy = (b: boolean) => {
    root.classList.toggle('busy', b);
    for (const id of ['m-enough', 'm-abort', 'enough', 'abort']) $<HTMLButtonElement>(id).disabled = !b;
    for (const id of ['m-run', 'next', 'run', 'snext']) $<HTMLButtonElement>(id).disabled = b;
  };
  busy(false);
  for (const id of ['m-enough', 'enough']) $(id).onclick = () => worker.postMessage({ type: 'stop', signal: 'enough' });
  for (const id of ['m-abort', 'abort']) $(id).onclick = () => worker.postMessage({ type: 'stop', signal: 'abort' });
  const STAGES: Record<string, string> = { spots: 'collecting spots', fit: 'fitting the PSF', report: 'held-out check', localize: 'localizing with the fitted PSF', spline: 'building the spline', header: 'reading the TIFF header' };
  onProgress = (p) => {
    const frac = p.total ? Math.min(1, p.done / p.total) : 0;
    const [bar, txt] = running === 'movie' ? ['mbar', 'mtext'] : ['pbar', 'ptext'];
    $(bar).style.width = (100 * frac).toFixed(1) + '%';
    const fmt = (v: unknown) => typeof v === 'number' ? Number.isInteger(v) ? String(v) : (Math.abs(v) < 1e-2 || Math.abs(v) > 1e4 ? v.toExponential(3) : v.toFixed(4)) : String(v);
    const info = Object.entries(p.info ?? {}).filter(([k]) => k !== 'zcoef').map(([k, v]) => `${k.replace(/_/g, ' ')} ${fmt(v)}`).join(', ');
    const t = p.elapsed_s !== undefined ? ` | ${p.elapsed_s.toFixed(1)} s` + (p.eta_s !== undefined ? `, ~${p.eta_s.toFixed(0)} s left` : '') : '';
    $(txt).textContent = `${running === 'movie' ? (STAGES[p.stage] ?? p.stage) : p.stage}: ${p.done} / ${p.total}${info ? ' | ' + info : ''}${t}`;
    if (running === 'movie' && p.stage === 'fit' && p.info?.zcoef) drawZernikeBars(String(p.info.zcoef).split(' ').map(Number), startZ);
    if (running === 'movie' && p.view) {
      const xy: number[] = []; for (let i = 0; i < p.view.x.length; i++) xy.push(p.view.x[i], p.view.y[i]);
      detsByFrame.set(p.view.frame, xy); followFrame(p.view.frame);
    }
    if (running === 'movie' && p.stage === 'localize' && !$('r-empty').hidden) $('r-empty').textContent = 'Localizing… the first frames are on their way.';
    if (running === 'stack') $('enough').textContent = p.stage === 'read' ? 'Enough frames' : p.stage === 'fit' ? 'Stop, keep estimate' : 'Use what we have';
  };

  // ================================================================ reconstruction
  let lastDraw = 0, drawTimer = 0;
  // point-cloud view (pointcloud.ts): world-unit Gaussian splats at screen resolution, pan / zoom / orbit
  let cloud: PointCloudView | null = null;
  getDevice().then(({ device }) => {
    const c = new PointCloudView(device, $('recon') as HTMLCanvasElement,
      () => ({ mode: $<HTMLSelectElement>('r-mode').value as 'precision' | 'fixed', blurNm: n('r-blur'), capNm: 60, maxCrlbNm: n('r-crlb'),
        colorZ: $<HTMLInputElement>('r-colorz').checked, zlo: n('r-zlo'), zhi: n('r-zhi'), zExag: n('r-zex') }));
    cloud = c; (window as any).__cloud = c;
    if (player) c.start(player.W, player.H, n('o-px'));
    c.onChange = () => {
      $<HTMLInputElement>('r-tilt').value = String(Math.round(c.tilt)); $<HTMLInputElement>('r-rot').value = String(Math.round(c.rot));
      const L = c.last;
      $('r-label').textContent = recon && L.n ? `${fmtN(L.n)} localizations` + (L.drawn < L.visible ? ` · drawing ${fmtN(L.drawn)} of ${fmtN(L.visible)} in view` : '') : '';
      // linked: follow the reconstruction's zoom / pan
      if (linked && mode === 'movie' && (c.scale !== mview.scale || c.cx !== mview.cx || c.cy !== mview.cy)) { mview.scale = c.scale; mview.cx = c.cx; mview.cy = c.cy; renderMovie(); }
    };
  }, () => { /* no WebGPU: the page says so */ });
  const drawRecon = () => { cloud?.request(); drawSpark(); lastDraw = performance.now(); };
  for (const id of ['r-mode', 'r-blur', 'r-colorz', 'r-zlo', 'r-zhi']) $(id).addEventListener('change', () => cloud?.request());

  // --- viewer: a localization file (torch_psf / Picasso HDF5, ThunderSTORM csv) dropped anywhere on the page
  async function loadLocFiles(fs: File[]) {
    const lt = $('loc-status');
    setMode('locs');
    $('lname').textContent = fs.map((f) => f.name).join(' + ');
    lt.textContent = 'reading...';
    focusRecon();
    try {
      const L = await readLocFile(fs, n('o-px'));
      recon = new Recon(L.H, L.W, L.pixel_nm / 1000);
      recon.add(L.rows);
      if (L.has3d) {
        const z = new Float32Array(L.n); for (let i = 0; i < L.n; i++) z[i] = L.rows[i * NCOL + 3]; z.sort();
        setNum('r-zlo', +z[Math.floor(0.01 * (L.n - 1))].toFixed(3)); setNum('r-zhi', +z[Math.floor(0.99 * (L.n - 1))].toFixed(3));
      }
      $<HTMLInputElement>('r-colorz').checked = L.has3d;
      $<HTMLSelectElement>('r-mode').value = Number.isFinite(L.rows[6]) ? 'precision' : 'fixed';
      cloud?.start(L.W, L.H, L.pixel_nm); cloud?.setFilter(n('r-crlb')); cloud?.append(L.rows);
      $('r-empty').hidden = true;
      const med = (k: number, f = 1) => { const a = new Float32Array(L.n); for (let i = 0; i < L.n; i++) a[i] = L.rows[i * NCOL + k] * f; a.sort(); return a[L.n >> 1]; };
      lt.textContent = `${fmtN(L.n)} localizations · ${L.nFrames} frames · ${L.W} × ${L.H} px of ${L.pixel_nm} nm · ${L.has3d ? '3D' : '2D'} (${L.format})`;
      $('l-table').innerHTML = kvRows([
        ['file', `${fs.map((f) => f.name).join(' + ')} (${L.format})`],
        ['localizations', `${L.n} in ${L.nFrames} frames; field ${L.W} x ${L.H} px of ${L.pixel_nm} nm${L.has3d ? '; 3D' : '; 2D'}`],
        ['median photons / bg', `${med(4).toFixed(0)} / ${med(5).toFixed(1)}`],
        ['median CRLB x / y / z', Number.isFinite(L.rows[6]) ? `${med(6, L.pixel_nm).toFixed(1)} / ${med(7, L.pixel_nm).toFixed(1)} / ${L.has3d ? med(8, 1000).toFixed(1) : '-'} nm` : 'not in the file (fixed blur)'],
        ...(L.notes.length ? [['notes', L.notes.join('; ')] as [string, string]] : []),
      ]);
      renderSteps();
      (window as any).__locsLoaded = L;
    } catch (e) { lt.textContent = 'could not read the file: ' + (e as Error).message; }
  }
  (window as any).__loadLocFiles = loadLocFiles;
  /** The reconstruction card at the top, in view, focused. */
  function focusRecon() {
    const p = $('c-recon');
    p.scrollIntoView({ behavior: 'smooth', block: 'start' });
    ($('recon') as HTMLCanvasElement).focus({ preventScroll: true });
  }
  // page-wide drop: TIFF -> the movie (or the bead stack in bead mode); .h5 / .hdf5 / .csv (+ Picasso .yaml) -> the viewer
  document.addEventListener('dragover', (e) => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault(); });
  document.addEventListener('drop', (e) => {
    for (const id of ['mdrop', 'drop', 'ldrop']) $(id).classList.remove('over');
    if (!e.dataTransfer?.files.length) return;
    e.preventDefault();
    routeFiles([...e.dataTransfer.files]);
  });
  $('r-crlb').addEventListener('change', () => cloud?.setFilter(n('r-crlb')));
  $<HTMLInputElement>('r-tilt').oninput = () => { if (cloud) { cloud.tilt = n('r-tilt'); cloud.request(); } };
  $<HTMLInputElement>('r-rot').oninput = () => { if (cloud) { cloud.rot = n('r-rot'); cloud.request(); } };
  $<HTMLInputElement>('r-zex').oninput = () => { $('r-zex-v').textContent = n('r-zex').toFixed(1); cloud?.request(); };
  $('r-reset').onclick = () => { cloud?.reset(); if (linked) { mview.scale = 0; mview.cx = 0; mview.cy = 0; renderMovie(); } };
  onLocs = (rows) => {
    recon?.add(rows);
    cloud?.append(rows);
    $('r-empty').hidden = true;
    let last = -1;
    for (let o = 0; o < rows.length; o += NCOL) { const f = rows[o]; const L = locsByFrame.get(f) ?? []; L.push(rows[o + 1], rows[o + 2]); locsByFrame.set(f, L); if (f > last) last = f; }
    if (last >= 0) followFrame(last);
    if (performance.now() - lastDraw > 700) drawRecon(); else { clearTimeout(drawTimer); drawTimer = window.setTimeout(drawRecon, 700); }
  };
  $('r-csv').onclick = () => { if (recon) download(locsToCsv(recon.rows(), n('o-px')), 'localizations.csv', 'text/csv'); };

  // ================================================================ Estimate PSF / Localize
  let startZ: number[] | null = null;
  let insituForm: InsituForm | null = null, insituRows: [string, string][] = [], beadRows: [string, string][] = [];
  onInsitu = (r) => { insitu = r; fillPresets(); showInsitu(r); if (r.refine?.kept) showRefinedVolume(); else showInsituVolume(); renderSteps(); };
  async function runMovie(kind: 'estimate' | 'localize') {
    if (!player) { $('mtext').textContent = 'No movie yet.'; return; }
    const psf = kind === 'localize' && psfSel.value === 'guess' && insitu ? { noll: insitu.noll, zcoef: insitu.zcoef } : startPsf();
    if (!psf) { $('mtext').textContent = 'Choose the start PSF first (Fit PSF step).'; return; }
    const form = movieForm(psf);
    if (form.roisize % 2 !== 1 || form.roisize < 5) { $('mtext').textContent = 'The ROI size must be odd (>= 5).'; return; }
    running = 'movie'; busy(true);
    startZ = kind === 'estimate' ? [...psf.zcoef] : null;
    if (kind === 'estimate') insituForm = form;
    recon = new Recon(player.H, player.W, form.optics.pixel_um); locsByFrame.clear(); detsByFrame.clear(); locsDone = false;
    setPlaying(false); follow = true; followAt = 0;
    $('r-empty').hidden = false;
    $('r-empty').textContent = kind === 'estimate' ? 'Fills in at the last stage of Estimate PSF: localizing the movie with the fitted PSF.' : 'Fills in live while the movie is localized.';
    cloud?.start(player.W, player.H, 1000 * form.optics.pixel_um); cloud?.setFilter(n('r-crlb'));
    if (linked && cloud) { cloud.scale = mview.scale; cloud.cx = mview.cx; cloud.cy = mview.cy; }
    setNum('r-zlo', n('m-zlo')); setNum('r-zhi', n('m-zhi'));
    $('mtext').textContent = 'starting...';
    try {
      const src: MovieSourceSpec = source;
      const r = kind === 'estimate' ? await call({ type: 'estimate', source: src, form, localize: true }) : await call({ type: 'movie', source: src, form: form as MovieForm });
      follow = false; drawRecon(); setPlaying(false); showFrame(frameNow);
      const s = r.summary as MovieSummary | null;
      if (s) { showLocs(s); locsDone = true; }
      $('mbar').style.width = '100%';
      $('mtext').textContent = kind === 'estimate'
        ? `done: PSF from ${r.report.n_train} spots in ${r.report.iterations} iterations (${r.report.seconds.toFixed(1)} s)` + (s ? `; then ${s.n_locs} localizations (${s.seconds.toFixed(1)} s)` : '')
        : `done: ${s!.n_locs} localizations from ${s!.frames_read} frames in ${s!.seconds.toFixed(1)} s`;
      renderSteps();
      (window as any).__movieDone = r;
    } catch (e) {
      const err = e as Error & { abort?: boolean };
      $('mtext').textContent = err.abort ? 'aborted' : 'error: ' + err.message;
      $('mbar').style.width = '0%';
    } finally { busy(false); follow = false; }
  }
  mRun.onclick = () => runMovie('estimate');
  nextBtn.onclick = () => {
    if (step === 'input') setStep('detect');
    else if (step === 'detect') setStep('fit');
    else if (step === 'fit') runMovie('estimate');
    else if (step === 'localize') runMovie('localize');
  };

  async function showInsituVolume() {
    if (!insitu) return;
    const zlo = n('m-zlo'), zhi = n('m-zhi');
    const spec: VolumeSpec = { optics: optics(), npupil: 64, noll: insitu.noll, zcoef: insitu.zcoef, nxy: 48, dxy_um: 0.04, zmin: zlo, zmax: zhi, nz: 41, ze_um: 0, focus_um: n('m-focus') };
    const r = await call({ type: 'volume', spec });
    sel.value = 'insitu';
    setVolume(r.volume, spec, `In-situ estimate (Noll 5-21; ${fmtZ(insitu.noll, insitu.zcoef)}). z = emitter height above the coverslip (${zlo}-${zhi} µm) at the focal plane ${n('m-focus')} µm.`);
  }
  function showRefinedVolume() {
    const r = insitu!, K = r.refine!.knots!;
    const g = splineGrid(n('m-roi'), n('m-zlo'), n('m-zhi'), 2, 0.04, r.grid.zpad, 3.0);
    const spec: VolumeSpec = { optics: optics(), npupil: 64, noll: r.noll, zcoef: r.zcoef, nxy: g.nu, dxy_um: pxUm() / g.S, zmin: g.z0, zmax: g.z0 + (g.nz - 1) * g.hz, nz: g.nz, ze_um: 0, focus_um: n('m-focus') };
    sel.value = 'refined';
    const kept = r.refine!.kept;
    setVolume(K, spec, `In-situ refined spline: free knots regularised towards the vectorial fit; ${kept ? 'kept (held-out NLL better than the vectorial fit; the movie was localized with it)' : 'NOT kept: it explains the held-out spots worse than the vectorial fit, so the movie was localized with the vectorial PSF'}. The knots themselves, ` +
      `every ${(n('o-px') / g.S).toFixed(0)} nm laterally and ${(1000 * g.hz).toFixed(0)} nm in z. ${FREE_KNOT_WARNING}.`);
  }
  function showInsituEmpty() {
    $('p-insitu').hidden = true; $('i-empty').hidden = false; $('i-report').hidden = true; $('i-meta').textContent = '';
    $('i-table').innerHTML = ''; $('i-kv').innerHTML = '';
  }
  $('i-report-btn').onclick = () => { const r = $('i-report'); r.hidden = !r.hidden; $('i-report-btn').classList.toggle('on', !r.hidden); };
  function showInsitu(r: InsituReport) {
    $('p-insitu').hidden = false; $('i-empty').hidden = true;
    $('i-meta').textContent = `in situ · ${fmtN(r.n_train)} spots` + (r.deviance ? ` · ${r.n_rejected} overlaps rejected` : '');
    const rms = Math.sqrt(r.zcoef.reduce((s, v, i) => s + (r.noll[i] >= 5 ? v * v : 0), 0));
    const chi = r.heldout_chi2_median;
    const kv: [string, string, string][] = [
      ['Held-out χ² / px', chi === null ? 'n/a' : chi.toFixed(3), chi === null ? 'na' : chi > 0.8 && chi < 1.25 ? 'ok' : 'warn'],
      ['rms aberration', `${(1000 * rms).toFixed(0)} mλ`, 'na'],
      ['Overlaps rejected', r.deviance ? `${(100 * r.deviance.rejected_fraction).toFixed(1)} %` : 'off', r.deviance ? (r.deviance.rejected_fraction < 0.25 ? 'ok' : 'warn') : 'na'],
      ['Held-out NLL vs start', r.heldout ? r.heldout.diff.toFixed(4) : 'n/a', r.heldout ? (r.heldout.diff <= 0 ? 'ok' : 'warn') : 'na'],
      ['Spline refinement', r.refine ? (r.refine.kept ? 'kept' : 'not kept') : 'off', r.refine?.kept ? 'ok' : 'na'],
      ['Spots at offset limit', (100 * r.at_limit).toFixed(1) + ' %', r.at_limit < 0.02 ? 'ok' : 'warn'],
    ];
    if (r.warnings.length) kv.push(['Warnings', String(r.warnings.length), 'warn']);
    if (truthZ) {
      const t = DEFAULT_NOLL.map((j) => truthZ![j] ?? 0);
      const e = Math.sqrt(r.zcoef.reduce((s, v, i) => s + (v - t[i]) ** 2, 0) / t.length);
      kv.push(['Error vs demo truth', `${(1000 * e).toFixed(1)} mλ rms`, e < 0.01 ? 'ok' : 'warn']);
    }
    $('i-kv').innerHTML = kv.map(([a, b, k]) => `<div class="kvr"><span>${a}</span><b>${b}</b><i class="st st-${k}"></i></div>`).join('');
    const rows: [string, string][] = [
      ['spots (fit / held out)', `${r.n_train} / ${r.n_test} of ${r.n_spots}, from ${r.frames_read} of ${r.frames_total} frames (${r.n_chunks} chunks spread over the movie)`],
      ['overlapping spots rejected', !r.deviance ? 'off' : `${r.n_rejected} (${(100 * r.deviance.rejected_fraction).toFixed(0)} %; deviance per pixel > ${r.deviance.threshold.toFixed(3)} = median ${r.deviance.median.toFixed(3)} + k x ${r.deviance.scale.toFixed(3)}), then a second pass`],
      ['held-out median chi2 per px', chi === null ? 'n/a' : `${chi.toFixed(3)} (about 1 if the camera gain / offset / excess noise are right)`],
      ['iterations', `${r.iterations}${r.partial ? ' (stopped early)' : ''}, ${r.seconds.toFixed(1)} s`],
      ['held-out NLL / px, fitted - start', r.heldout ? `${r.heldout.diff.toFixed(4)} (negative: the fitted PSF explains spots it was not fitted to better)` : 'no held-out spots'],
      ['z scale vs the start PSF', `${r.z_scale.toFixed(3)} (offset ${(1000 * r.z_offset_um).toFixed(0)} nm)`],
      ['spots at the offset limit', (100 * r.at_limit).toFixed(2) + ' %'],
      ['Zernikes (waves rms)', fmtZ(r.noll, r.zcoef)],
    ];
    if (r.warnings.length) rows.push(['warnings', r.warnings.join('; ')]);
    if (r.refine) {
      const f = r.refine;
      rows.push(['spline refinement', `${f.kept ? 'kept' : 'not kept'}: held-out NLL / px ${f.heldout_refined === null ? 'n/a' : (f.heldout_refined - (r.heldout?.start ?? 0)).toFixed(4)} vs start ` +
        `(vectorial ${r.heldout ? r.heldout.diff.toFixed(4) : 'n/a'}); z scale vs vectorial ${f.z_scale_vs_vectorial.toFixed(3)}; ${f.outer} outer iterations, ${f.seconds.toFixed(1)} s` +
        (f.kept ? '. The movie is localized with the refined spline. ' + FREE_KNOT_WARNING : '. The refined spline can still be viewed: PSF menu, "In-situ refined spline (not kept)".')]);
    }
    if (truthZ) {
      const t = DEFAULT_NOLL.map((j) => truthZ![j] ?? 0);
      const e = (a: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - t[i]) ** 2, 0) / t.length);
      rows.push(['vs the demo truth (rms over Noll 5-21)', `start ${e(r.zcoef_start).toFixed(4)} w, fitted ${e(r.zcoef).toFixed(4)} w`]);
    }
    $('i-table').innerHTML = kvRows(rows);
    insituRows = rows;
    drawZernikeBars(r.zcoef, r.zcoef_start);
  }
  function drawZernikeBars(z: number[], start: number[] | null) {
    const truth = truthZ ? DEFAULT_NOLL.map((j) => truthZ![j] ?? 0) : null;
    $('p-insitu').hidden = false; $('i-empty').hidden = true;
    drawZernikeSets($('i-zern') as HTMLCanvasElement, DEFAULT_NOLL, [
      ...(start ? [{ z: start, color: '#c3cad2', label: 'start' }] : []),
      { z, color: '#2f6feb', label: 'fitted' },
      ...(truth ? [{ z: truth, color: '#1a7f37', label: 'truth (demo)' }] : []),
    ]);
  }
  const insituProv = (r: InsituReport) => ({ source: 'torch-psf web in-situ fit (knot route, torch_psf estimate_psf_insitu)',
    movie: source === 'demo' ? 'simulated demo' : (source as File[]).map((f) => f.name),
    n_train: r.n_train, n_test: r.n_test, iterations: r.iterations, partial: r.partial, heldout_nll_px_diff: r.heldout?.diff ?? null,
    z_scale_vs_start: r.z_scale, z_range_um: [n('m-zlo'), n('m-zhi')], gain_e_per_adu: n('o-gain'), offset_adu: n('o-offset') });
  $('i-json').onclick = () => {
    if (!insitu) return;
    const j = makePsfJson({ optics: optics(), noll: insitu.noll, zcoef: insitu.zcoef, focus_um: n('m-focus'), npupil: 64, oversample: 3,
      provenance: { ...insituProv(insitu), created: new Date().toISOString() } });
    download(JSON.stringify(j, null, 1), 'psf_insitu.json', 'application/json');
  };
  /** fit_report.pdf of the in-situ fit (reproducible: no time stamp, for psf.zip) and its psf.json sections */
  async function insituPdf(reproducible: boolean) {
    const R = await import('../report/fromapp'), r = insitu!, f = insituForm!;
    const spec: VolumeSpec = { optics: f.optics, npupil: 64, noll: r.noll, zcoef: r.zcoef, nxy: 48, dxy_um: 0.04, zmin: f.zlo, zmax: f.zhi, nz: 41, ze_um: 0, focus_um: f.focus_um };
    const vol = await call({ type: 'volume', spec });
    const d = R.insituReportData(r, { form: f, results: insituRows, source: source === 'demo' ? 'simulated demo' : (source as File[]).map((x) => x.name).join(', '),
      truth: truthZ ? DEFAULT_NOLL.map((j) => truthZ![j] ?? 0) : null, locs: recon?.rows() ?? null, volume: { data: vol.volume, spec } });
    if (reproducible) d.created = '';
    return { bytes: await R.makeFitReport(d, { creationDate: !reproducible }), sections: reportSections(d) };
  }
  $('i-pdf').onclick = async () => {
    if (!insitu || !insituForm) return;
    const st = $('i-status');
    st.textContent = 'writing the report...';
    try {
      const { bytes } = await insituPdf(false);
      download(bytes, 'fit_report.pdf', 'application/pdf');
      st.textContent = `fit_report.pdf (${(bytes.length / 1024).toFixed(0)} kB)`;
    } catch (e) { st.textContent = 'error: ' + (e as Error).message; }
  };
  // --- psf.zip (both fits): built in the worker (psf/makezip.ts)
  const zipOpts = (p: string): ZipOptions => ({ planes: Math.round(n(p + '-zn')), k: Math.round(n(p + '-zk')), spline: $<HTMLInputElement>(p + '-spline').checked });
  const saveZip = async (st: HTMLElement, make: () => Promise<PsfZipRequest>) => {
    st.textContent = 'writing psf.zip...';
    try {
      const r = await call({ type: 'psfzip', req: await make() });
      download(r.bytes, 'psf.zip', 'application/zip');
      st.textContent = `psf.zip (${(r.bytes.length / 1024).toFixed(0)} kB)`;
    } catch (e) { st.textContent = 'error: ' + (e as Error).message; }
  };
  $('i-zip').onclick = () => {
    if (!insitu || !insituForm) return;
    const r = insitu, f = insituForm;
    saveZip($('i-status'), async () => { const p = await insituPdf(true); return insituZipRequest(r, f, zipOpts('i'), insituProv(r), p.bytes, p.sections); });
  };
  function showLocs(s: MovieSummary) {
    const rows = recon!.rows(), N = rows.length / NCOL;
    const med = (k: number, f = 1) => { const a = new Float32Array(N); for (let i = 0; i < N; i++) a[i] = rows[i * NCOL + k] * f; a.sort(); return N ? a[N >> 1] : NaN; };
    const px = n('o-px');
    $('l-table').innerHTML = kvRows([
      ['localizations', `${s.n_locs} (${(s.n_locs / Math.max(1, s.frames_read)).toFixed(2)} per frame) from ${s.frames_read} of ${s.frames_total} frames`],
      ['median photons / bg', `${med(4).toFixed(0)} / ${med(5).toFixed(1)}`],
      ['median CRLB x / y / z', `${med(6, px).toFixed(1)} / ${med(7, px).toFixed(1)} / ${med(8, 1000).toFixed(1)} nm`],
      ['at the offset limit', (100 * s.at_limit_fraction).toFixed(2) + ' %'],
      ['time', `${s.seconds.toFixed(1)} s (${s.spots_per_s.toFixed(0)} spots / s)`],
    ]);
  }

  // ================================================================ bead z-stack
  let files: File[] = [];
  const setFiles = (fl: FileList | File[]) => {
    files = [...fl].filter((f) => /\.tiff?$/i.test(f.name));
    $('files').textContent = files.length ? files.map((f) => `${f.name} (${(f.size / 2 ** 20).toFixed(1)} MB)`).join(', ') : 'no TIFF selected';
    renderSteps();
  };
  (window as any).__setFiles = setFiles;
  $<HTMLInputElement>('file').onchange = (e) => setFiles((e.target as HTMLInputElement).files!);
  $('snext').onclick = () => setStep('fit');
  $('run').onclick = async () => {
    running = 'stack';
    if (!files.length) { $('ptext').textContent = 'Open a bead z-stack TIFF first.'; return; }
    busy(true);
    $('ptext').textContent = 'starting...';
    try {
      const r = await call({ type: 'fit', files, form: readForm() });
      fitResult = r.summary as FitSummary;
      fillPresets();
      showResult(fitResult);
      $('pbar').style.width = '100%';
      $('ptext').textContent = `done: ${fitResult.iterations} iterations in ${fitResult.fit_seconds.toFixed(2)} s` + (fitResult.partial ? ' (partial result)' : '');
      renderSteps();
      (window as any).__fitDone = true;
    } catch (e) {
      const err = e as Error & { abort?: boolean };
      $('ptext').textContent = err.abort ? 'aborted' : 'error: ' + err.message;
      $('pbar').style.width = '0%';
    } finally { busy(false); }
  };
  function readForm(): FitForm {
    return {
      optics: optics(), sign: n('f-sign'), dz_um: n('f-dz'), gain: n('o-gain'), offset: n('o-offset'), ze_um: n('f-ze'),
      roisize: n('f-roi'), npupil: n('f-npupil'), oversample: n('f-os'),
      fitDz: $<HTMLInputElement>('f-fitdz').checked, fitDrift: $<HTMLInputElement>('f-drift').checked,
      holdout: $<HTMLSelectElement>('f-holdout').value as FitForm['holdout'], maxIter: n('f-iter'), minRel: n('f-minrel'),
    };
  }
  const beadNote = (r: FitSummary) => `Bead fit (Noll 5-21) of ${r.stacks.map((s) => s.name).join(', ')}; z = defocus of the bead (stage coordinate, z-step scale applied), xy 2x oversampled.`;
  function showResult(r: FitSummary) {
    $('p-results').hidden = false; $('b-empty').hidden = true;
    $('i-meta').textContent = `${r.beads.length} beads · ${r.iterations} iterations`;
    sel.value = 'fit';
    setVolume(r.volume, r.volumeSpec, beadNote(r));
    const rows: [string, string][] = [
      ['beads (held out)', `${r.beads.length} (${r.beads.filter((b) => b.heldout).length})`],
      ['planes used / read', `${r.frames_used} / ${r.frames_total}${r.partial_read ? ' (reading stopped early)' : ''}`],
      ['iterations', `${r.iterations}${r.converged ? ' (converged)' : r.partial ? ' (stopped by user)' : ' (iteration limit)'} + ${r.heldout_iterations} held-out`],
      ['NLL / px, fit beads', r.nll_train_px.toFixed(5)],
      ['NLL / px, held-out', r.nll_heldout_px === null ? 'none held out' : r.nll_heldout_px.toFixed(5)],
      ['z-step scale', r.dz_scale.toFixed(4)],
      ['time: read / fit', `${r.read_seconds.toFixed(2)} s / ${r.fit_seconds.toFixed(2)} s`],
    ];
    $('res-table').innerHTML = kvRows(rows);
    beadRows = rows;
    drawZernikeSets($('zern') as HTMLCanvasElement, r.noll, [{ z: r.zcoef, color: '#2f6feb', label: 'fitted' }]);
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
  const expParams = () => ({ focus_um: n('e-focus'), splineRoi: n('e-roi'), zmin: n('e-zmin'), zmax: n('e-zmax') });
  const beadProv = (r: FitSummary, f: FitForm) => ({ source: 'torch-psf web bead fit', stacks: r.stacks.map((s) => s.name), beads: r.beads.length,
    heldout_beads: r.beads.filter((b) => b.heldout).length, frames_used: r.frames_used, frames_total: r.frames_total, partial: r.partial,
    nll_train_px: r.nll_train_px, nll_heldout_px: r.nll_heldout_px, iterations: r.iterations, dz_scale: r.dz_scale,
    stage_sign: f.sign, dz_um: f.dz_um, bead_centre_um: f.ze_um, gain_e_per_adu: f.gain, offset_adu: f.offset });
  $('x-json').onclick = () => {
    const r = fitResult!, f = readForm();
    const j = makePsfJson({ optics: f.optics, noll: r.noll, zcoef: r.zcoef, focus_um: expParams().focus_um, npupil: f.npupil, oversample: f.oversample,
      provenance: { ...beadProv(r, f), created: new Date().toISOString() } });
    download(JSON.stringify(j, null, 1), 'psf.json', 'application/json');
  };
  const viaWorker = async (kind: 'knots' | 'tif') => {
    $('x-status').textContent = 'computing...';
    try { const r = await call({ type: 'export', kind, params: expParams() }); download(r.bytes, r.name); $('x-status').textContent = `${r.name} (${(r.bytes.length / 1024).toFixed(0)} kB)`; }
    catch (e) { $('x-status').textContent = 'error: ' + (e as Error).message; }
  };
  async function beadPdf(r: FitSummary, f: FitForm, reproducible: boolean) {
    const R = await import('../report/fromapp');
    const d = R.beadReportData(r, { form: f, results: beadRows, settings: {
      'stage sign / z step (\u00b5m)': [f.sign, f.dz_um], 'bead centre height (\u00b5m)': f.ze_um, 'camera gain (e/ADU) / offset (ADU)': [f.gain, f.offset],
      'ROI size (px)': f.roisize, 'pupil sampling / oversampling': [f.npupil, f.oversample], 'fit z-step scale': f.fitDz, 'fit xy drift': f.fitDrift,
      'held-out beads': f.holdout, 'max. iterations': f.maxIter } });
    if (reproducible) d.created = '';
    return { bytes: await R.makeFitReport(d, { creationDate: !reproducible }), sections: reportSections(d) };
  }
  $('x-zip').onclick = () => {
    const r = fitResult!, f = readForm(), e = expParams();
    saveZip($('x-status'), async () => { const p = await beadPdf(r, f, true); return beadZipRequest(r, f, e, zipOpts('x'), beadProv(r, f), p.bytes, p.sections); });
  };
  $('x-pdf').onclick = async () => {
    const r = fitResult!, f = readForm();
    $('x-status').textContent = 'writing the report...';
    try {
      const { bytes } = await beadPdf(r, f, false);
      download(bytes, 'fit_report.pdf', 'application/pdf');
      $('x-status').textContent = `fit_report.pdf (${(bytes.length / 1024).toFixed(0)} kB)`;
    } catch (e) { $('x-status').textContent = 'error: ' + (e as Error).message; }
  };
  $('x-knots').onclick = () => viaWorker('knots');
  $('x-tif').onclick = () => viaWorker('tif');

  // ================================================================ start
  let m0: Mode = 'movie', s0: Step = 'input';
  try { const t = localStorage.getItem('tpsf.tab'); m0 = t === 'stack' ? 'stack' : 'movie'; const s = localStorage.getItem('tpsf.step') as Step | null; if (s) s0 = s; } catch { /* no storage */ }
  setMode(m0); if (STEPS[m0].some(([k]) => k === s0)) setStep(s0);
  showInsituEmpty();
}

function download(bytes: BlobPart, name: string, type = 'application/octet-stream') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([bytes], { type }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

const fmtZ = (noll: number[], z: number[]) => noll.map((j, i) => [j, z[i]] as [number, number]).filter(([, v]) => Math.abs(v) >= 0.005)
  .map(([j, v]) => `${j}: ${v.toFixed(3)}`).join(', ') || 'none';
const fmtN = (v: number) => v >= 1e6 ? (v / 1e6).toFixed(2) + ' M' : v >= 1e4 ? (v / 1e3).toFixed(0) + ' k' : String(v);
const kvRows = (rows: [string, string][]) => rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');

function flipRows(a: Float32Array, w: number, h: number) {
  const o = new Float32Array(a.length);
  for (let r = 0; r < h; r++) o.set(a.subarray(r * w, (r + 1) * w), (h - 1 - r) * w);
  return o;
}

function niceBar(nm: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(nm))), m = nm / p;
  return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * p;
}

/** Image overlay shared by the movie view: zoom factor, scale bar, and a minimap when zoomed in. */
function overlay(ctx: CanvasRenderingContext2D, W: number, H: number, scale: number, fit: number, fw: number, fh: number, cx: number, cy: number, dpr: number) {
  const z = fit / scale;
  ctx.font = `600 ${11 * dpr}px Inter, 'Segoe UI', system-ui, sans-serif`;
  const zt = `${z.toFixed(z < 10 ? 1 : 0)}×`, tw = ctx.measureText(zt).width;
  ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(8 * dpr, 8 * dpr, tw + 12 * dpr, 18 * dpr);
  ctx.fillStyle = '#fff'; ctx.fillText(zt, 14 * dpr, 21 * dpr);
  const barNm = niceBar(scale * W / 5), barPx = barNm / scale;
  ctx.fillStyle = '#fff'; ctx.fillRect(12 * dpr, H - 14 * dpr, barPx, 3 * dpr);
  ctx.font = `500 ${10 * dpr}px Inter, 'Segoe UI', system-ui, sans-serif`;
  ctx.fillText(barNm >= 1000 ? `${barNm / 1000} µm` : `${barNm} nm`, 12 * dpr, H - 19 * dpr);
  if (z > 1.05) {
    const m = 64 * dpr, mx = W - m - 8 * dpr, my = H - m - 8 * dpr, k = m / Math.max(fw, fh);
    ctx.fillStyle = 'rgba(0,0,0,.6)'; ctx.fillRect(mx, my, m, m);
    ctx.strokeStyle = 'rgba(255,255,255,.45)'; ctx.lineWidth = dpr; ctx.strokeRect(mx + .5, my + .5, fw * k, fh * k);
    const vw = W * scale, vh = H * scale, vx = fw / 2 + cx - vw / 2, vy = fh / 2 + cy - vh / 2;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2 * dpr;
    ctx.strokeRect(mx + Math.max(0, vx * k), my + Math.max(0, vy * k), Math.min(fw, vw) * k, Math.min(fh, vh) * k);
  }
}

/** Zernike coefficients (waves rms) by Noll index; several sets side by side (e.g. start, fitted, truth). */
function drawZernikeSets(cv: HTMLCanvasElement, noll: number[], sets: { z: number[]; color: string; label: string }[]) {
  const dpr = window.devicePixelRatio || 1;
  const cw = cv.clientWidth || cv.width, ch = cv.clientHeight || cv.height;
  cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
  const ctx = cv.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cw, H = ch, pad = 38;
  ctx.clearRect(0, 0, W, H);
  const m = Math.max(0.02, ...sets.flatMap((s) => s.z.map(Math.abs)));
  const bw = (W - pad - 6) / noll.length, y0 = H / 2 + 6, k = sets.length;
  ctx.strokeStyle = '#aeb6bf'; ctx.beginPath(); ctx.moveTo(pad, y0 + .5); ctx.lineTo(W - 4, y0 + .5); ctx.stroke();
  ctx.font = "10px Inter, 'Segoe UI', system-ui, sans-serif"; ctx.fillStyle = '#8c959f';
  ctx.fillText(`+${m.toFixed(3)}`, 0, 30); ctx.fillText(`-${m.toFixed(3)}`, 0, H - 16);
  const amp = H / 2 - 24;
  noll.forEach((j, i) => {
    sets.forEach((s, q) => {
      const h = (s.z[i] / m) * amp;
      ctx.fillStyle = s.color;
      ctx.fillRect(pad + i * bw + 2 + q * (bw - 4) / k, h >= 0 ? y0 - h : y0, Math.max(1, (bw - 4) / k - 1), Math.abs(h));
    });
    ctx.fillStyle = '#8c959f';
    ctx.fillText(String(j), pad + i * bw + bw / 2 - 5, H - 2);
  });
  let x = pad + 2;
  ctx.font = "11px Inter, 'Segoe UI', system-ui, sans-serif";
  for (const s of sets) { ctx.fillStyle = s.color; ctx.fillRect(x, 4, 10, 10); ctx.fillStyle = '#656d76'; ctx.fillText(s.label, x + 14, 13); x += 24 + ctx.measureText(s.label).width; }
}


// ---------------------------------------------------------------- references (numbered list at the bottom of the page;
// cite('key') puts a superscript number next to the control that uses the method)
const REFS: { key: string; text: string; url: string }[] = [
  { key: 'torchpsf', text: 'torch-psf: vectorial and spline PSF models, in-situ PSF estimation and localization in PyTorch (the Python reference implementation; every step of this app is a port of it and tested against it).', url: 'https://github.com/jcnossen/torch-psf' },
  { key: 'thunderstorm', text: 'M. Ovesný, P. Křížek, J. Borkovec, Z. Švindrych, G. M. Hagen. ThunderSTORM: a comprehensive ImageJ plug-in for PALM and STORM data analysis and super-resolution imaging. Bioinformatics 30, 2389–2390 (2014).', url: 'https://doi.org/10.1093/bioinformatics/btu202' },
  { key: 'picasso', text: 'J. Schnitzbauer, M. T. Strauss, T. Schlichthaerle, F. Schueder, R. Jungmann. Super-resolution microscopy with DNA-PAINT. Nature Protocols 12, 1198–1228 (2017).', url: 'https://doi.org/10.1038/nprot.2017.024' },
  { key: 'janesick', text: 'J. R. Janesick. Photon Transfer: DN → λ. SPIE Press (2007).', url: 'https://doi.org/10.1117/3.725073' },
  { key: 'heintzmann', text: 'R. Heintzmann, P. K. Relich, R. P. J. Nieuwenhuizen, K. A. Lidke, B. Rieger. Calibrating photon counts from a single image. arXiv:1611.05654 (2016).', url: 'https://arxiv.org/abs/1611.05654' },
  { key: 'mortensen', text: 'K. I. Mortensen, L. S. Churchman, J. A. Spudich, H. Flyvbjerg. Optimized localization analysis for single-molecule tracking and super-resolution microscopy. Nature Methods 7, 377–381 (2010).', url: 'https://doi.org/10.1038/nmeth.1447' },
  { key: 'huang', text: 'F. Huang, T. M. P. Hartwich, F. E. Rivera-Molina et al., J. Bewersdorf. Video-rate nanoscopy using sCMOS camera-specific single-molecule localization algorithms. Nature Methods 10, 653–658 (2013).', url: 'https://doi.org/10.1038/nmeth.2488' },
  { key: 'richards', text: 'B. Richards, E. Wolf. Electromagnetic diffraction in optical systems. II. Structure of the image field in an aplanatic system. Proc. R. Soc. Lond. A 253, 358–379 (1959).', url: 'https://doi.org/10.1098/rspa.1959.0200' },
  { key: 'noll', text: 'R. J. Noll. Zernike polynomials and atmospheric turbulence. J. Opt. Soc. Am. 66, 207–211 (1976).', url: 'https://doi.org/10.1364/JOSA.66.000207' },
  { key: 'siemons', text: 'M. Siemons, C. N. Hulleman, R. Ø. Thorsen, C. S. Smith, S. Stallinga. High precision wavefront control in point spread function engineering for single emitter localization. Optics Express 26, 8397–8416 (2018).', url: 'https://doi.org/10.1364/OE.26.008397' },
  { key: 'inspr', text: 'F. Xu, D. Ma, K. P. MacPherson et al. Three-dimensional nanoscopy of whole cells and tissues with in situ point spread function retrieval. Nature Methods 17, 531–540 (2020).', url: 'https://doi.org/10.1038/s41592-020-0816-x' },
  { key: 'uipsf', text: 'S. Liu, J. Chen, J. Hellgoth et al., J. Ries. Universal inverse modeling of point spread functions for SMLM localization and microscope characterization. Nature Methods 21, 1082–1093 (2024).', url: 'https://doi.org/10.1038/s41592-024-02282-x' },
  { key: 'smith', text: 'C. S. Smith, N. Joseph, B. Rieger, K. A. Lidke. Fast, single-molecule localization that achieves theoretically minimum uncertainty. Nature Methods 7, 373–375 (2010).', url: 'https://doi.org/10.1038/nmeth.1449' },
  { key: 'adam', text: 'D. P. Kingma, J. Ba. Adam: a method for stochastic optimization. ICLR (2015); arXiv:1412.6980.', url: 'https://arxiv.org/abs/1412.6980' },
  { key: 'lbfgs', text: 'D. C. Liu, J. Nocedal. On the limited memory BFGS method for large scale optimization. Mathematical Programming 45, 503–528 (1989).', url: 'https://doi.org/10.1007/BF01589116' },
  { key: 'websmlm', text: 'J. Hohlbein, K. J. A. Martens, C. Smith. webSMLM: a browser-based single-molecule localization microscopy tool (software, v0.13.0, Zenodo); the principles of the point-cloud renderer.', url: 'https://doi.org/10.5281/zenodo.23089005' },
];
const cite = (...keys: string[]) => `<sup class="cite">${keys.map((k) => {
  const i = REFS.findIndex((r) => r.key === k);
  if (i < 0) throw new Error('unknown reference ' + k);
  return `<a href="#ref-${i + 1}" title="${REFS[i].text.replace(/"/g, '&quot;')}">${i + 1}</a>`;
}).join('')}</sup>`;
const REF_LIST = `<section class="refs" id="references"><h3>References</h3><p class="note">The methods behind the numbers next to the controls.</p><ol>${REFS.map((r, i) =>
  `<li id="ref-${i + 1}">${r.text} <a href="${r.url}" target="_blank" rel="noopener">${r.url.replace(/^https:\/\/(doi\.org\/)?/, '')}</a></li>`).join('')}</ol></section>`;

const field = (id: string, label: string, v: number, unit = '', step = 'any', title = '') =>
  `<label class="field" title="${title}"><span class="fl">${label}</span><span class="inp"><input id="${id}" type="number" step="${step}" value="${v}">${unit ? `<i>${unit}</i>` : ''}</span></label>`;
const check = (id: string, label: string, on = false, title = '') => `<label class="chk" title="${title}"><input id="${id}" type="checkbox" ${on ? 'checked' : ''}> ${label}</label>`;

const ZIP_TITLE = 'psf.zip (torch-psf 0.2.0): psf.json, z-stack, preview and fit report; read by torch_psf.io.load and by this app';
/** psf.zip options: z-stack planes and sampling (1 / k camera px), spline channel. */
const zipRow = (p: string, splineTitle: string) => `<details class="adv" data-key="zip-${p}"><summary>psf.zip options <span class="dim">z-stack resolution, spline</span></summary>` +
  field(p + '-zn', 'z-stack planes', 25, '', '1', 'zstack.tif planes over the z range (torch_psf default 25)') +
  field(p + '-zk', 'Sampling 1 / k px', 1, 'k', '1', 'zstack.tif sample spacing 1 / k camera px, from k x k sub-pixel renders (torch_psf default k = 1)') +
  check(p + '-spline', 'Spline channel', false, `channel 0 as a spline whose source is the vectorial model: ${splineTitle}`) + '</details>';

const TEMPLATE = `
<header class="appbar">
  <div class="brand">
    <svg class="logo" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="11" opacity=".22"/><circle cx="12" cy="12" r="7.2" opacity=".45"/><circle cx="12" cy="12" r="3.6"/></svg>
    <div><div class="name">torch-psf <span>web</span></div>
      <a class="gh" href="https://github.com/jcnossen/torch-psf" target="_blank" rel="noopener">github.com/jcnossen/torch-psf</a></div>
  </div>
  <nav class="steps" id="steps" aria-label="workflow"></nav>
  <div id="gpu-status">Checking WebGPU...</div>
</header>
<div id="nogpu" class="warn" hidden></div>
<div class="shell">
<aside class="side">
 <div class="side-body">
  <section data-step="input">
    <h2>Input</h2><p class="sub">Load data and describe the microscope.</p>
    <div class="seg tabs" role="tablist"><button data-tab="movie" role="tab">SMLM movie</button><button data-tab="stack" role="tab">Bead z-stack</button><button data-tab="locs" role="tab">Localizations</button></div>
    <div id="tab-movie">
      <div id="mdrop" class="filecard drop"><div class="ficon"></div><div class="fbody"><div id="mname" class="fname">Simulated demo movie</div><div id="mfiles" class="fmeta"></div>
        <div class="flinks"><label class="link">Open…<input id="mfile" type="file" multiple accept=".tif,.tiff,.h5,.hdf5,.csv,.yaml" hidden></label><a href="#" id="m-demo">Demo movie</a></div></div></div>
      <p class="hint">Or drop a TIFF movie, or a localization file (HDF5${cite('picasso')}, CSV${cite('thunderstorm')}), anywhere on the page.</p>
    </div>
    <div id="tab-stack" hidden>
      <div id="drop" class="filecard drop"><div class="ficon"></div><div class="fbody"><div class="fname">Bead z-stack</div><div id="files" class="fmeta">no TIFF selected</div>
        <div class="flinks"><label class="link">Open…<input id="file" type="file" multiple accept=".tif,.tiff" hidden></label></div></div></div>
      <p class="hint">Several stacks: one file each; a multi-part stack: movie.tif, movie_X1.tif, ...</p>
      ${field('f-dz', 'z step', 0.02, 'µm')}
    </div>
    <div id="tab-locs" hidden>
      <div id="ldrop" class="filecard drop"><div class="ficon"></div><div class="fbody"><div id="lname" class="fname">Localization file</div><div id="loc-status" class="fmeta">torch-psf / Picasso${cite('picasso')} HDF5 (+ yaml), ThunderSTORM${cite('thunderstorm')} CSV</div>
        <div class="flinks"><label class="link">Open…<input id="lfile" type="file" multiple accept=".h5,.hdf5,.csv,.yaml" hidden></label></div></div></div>
    </div>
    <h3 class="grp">Optics</h3>
    ${field('o-px', 'Pixel size', 108, 'nm')}${field('o-na', 'Numerical aperture', 1.49)}${field('o-wl', 'Emission wavelength', 680, 'nm')}
    <div class="field"><span class="fl">Refractive index</span><span class="pair"><span class="inp"><input id="o-nimm" type="number" step="any" value="1.518" title="immersion"></span><span class="inp"><input id="o-nmed" type="number" step="any" value="1.33" title="sample medium"></span></span></div>
    <div class="pairlab"><span>immersion</span><span>sample</span></div>
    <div class="grp-row"><h3 class="grp">Camera${cite('janesick', 'heintzmann')}</h3><span id="cam-badge" class="badge" hidden></span></div>
    <div class="seg" id="cam-type"><button data-f="1" aria-selected="true">sCMOS / CCD</button><button data-f="2">EMCCD${cite('mortensen')}</button></div>
    ${field('o-gain', 'Gain', 0.5, 'e⁻/ADU')}${field('o-offset', 'Offset', 100, 'ADU')}${field('o-rn', 'Read noise' + cite('huang'), 0, 'e⁻', 'any', 'read noise (e- rms), used in every likelihood (shifted Poisson)')}
    ${check('cam-offknown', 'Offset known (spec or dark frames): estimate the read noise' + cite('heintzmann'), false, 'photon transfer cannot separate the offset from the read noise; with a known offset the read noise follows from the intercept (torch_psf photon_transfer(offset_adu=...))')}
    <p id="cam-note" class="hint"></p>
    <div class="flinks"><a href="#" id="cam-est">Estimate gain and offset again</a></div>
    <details class="adv" data-key="adv-input"><summary>Advanced <span class="dim">frame range, excess noise</span></summary>
      ${field('m-frames', 'Max frames', 0, '0 = all', '1')}
      ${field('o-excess', 'Excess noise F²' + cite('mortensen'), 1, '', 'any', 'torch_psf Camera.excess_noise: photons are divided by F^2 (Poisson-equivalent), read-noise variance by F^4. EMCCD: 2')}
    </details>
  </section>

  <section data-step="detect" data-mode="movie">
    <h2>Detect spots</h2><p class="sub">Candidate emitters for the PSF fit and the localization.</p>
    ${field('m-roi', 'ROI size', 15, 'px, odd', '2')}${field('m-thr', 'Threshold', 6, '× noise')}${field('m-iso', 'Isolation', 0, 'px, 0 = off', '1')}
    ${field('m-chunk', 'Noise estimate per', 500, 'frames', '1', 'the detection noise is estimated per chunk (torch_psf default 500)')}
    <p class="hint">Difference-of-Gaussians filter, a robust noise estimate per chunk of frames, local maxima above the threshold (torch_psf detect_spots). After a run, the detections are drawn as rings on the movie.</p>
  </section>

  <section data-step="fit" data-mode="movie">
    <h2>Fit PSF</h2><p class="sub">Vectorial PSF${cite('richards', 'siemons')} fitted to isolated spots of the movie itself (in situ${cite('torchpsf', 'inspr', 'uipsf')}).</p>
    <h3 class="grp">Sample</h3>
    ${field('m-focus', 'Focal plane', 0.4, 'µm', 'any', 'focal-plane position above the coverslip during the acquisition')}
    <div class="field"><span class="fl">Emitter heights</span><span class="pair"><span class="inp"><input id="m-zlo" type="number" step="any" value="0"></span><span class="inp"><input id="m-zhi" type="number" step="any" value="0.8"><i>µm</i></span></span></div>
    <h3 class="grp">Fit</h3>
    <label class="field"><span class="fl">Start PSF</span><select id="m-psf"><option value="guess">generic astigmatic</option><option value="view">PSF in the 3D view</option><option value="insitu">last in-situ estimate</option><option value="json">PSF file (psf.zip / json)…</option></select></label>
    <input id="m-psfjson" type="file" accept=".json,.zip" hidden><p id="m-psfinfo" class="hint"></p>
    ${check('m-refine', 'Spline refinement' + cite('torchpsf', 'lbfgs'), false, 'free spline knots regularised towards the fitted vectorial PSF (torch_psf InSituEstimator refine=True); kept only if the held-out NLL improves')}
    <details class="adv" data-key="adv-fit"><summary>Advanced <span class="dim">spot counts, iterations, overlaps</span></summary>
      ${field('m-nspots', 'Spots for the fit', 8000, '', '1')}${field('m-nheld', 'Held-out spots', 4000, '', '1')}${field('m-iters', 'Iterations' + cite('adam'), 80, '', '1')}
      ${field('m-schunk', 'Sampling chunk', 50, 'frames', '1', 'the PSF fit samples spots from chunks spread over the whole movie (bit-reversed order; torch_psf InSituEstimator chunk=50)')}
      ${field('m-devk', 'Overlap rejection k', 3, '0 = off', 'any', 'after the first pass, spots whose Poisson deviance per pixel is above median + k x scale are dropped and the fit is repeated (torch_psf InSituEstimator deviance_k)')}
    </details>
    <p class="hint">Estimate PSF also localizes the movie with the fitted PSF. Defaults of torch_psf InSituEstimator; every step is a port of the Python code and is tested against it.</p>
  </section>

  <section data-step="localize" data-mode="movie">
    <h2>Localize</h2><p class="sub">Fit every detected spot with the PSF (x, y, z, photons, background) by maximum likelihood, with Cramér-Rao bounds${cite('smith', 'huang')}.</p>
    <p class="hint">Runs automatically after Estimate PSF, with the fitted PSF (the refined spline if it was kept). Localize again here after changing the detection settings; it uses the last estimate, or the start PSF if there is none.</p>
  </section>

  <section data-step="fit" data-mode="stack">
    <h2>Fit PSF</h2><p class="sub">Vectorial PSF${cite('richards', 'siemons')} fitted to the bead z-stack(s).</p>
    <label class="field"><span class="fl">Stage sign</span><select id="f-sign"><option value="1">+1</option><option value="-1">-1</option></select></label>
    ${field('f-ze', 'Bead centre height', 0.05, 'µm', 'any', 'emitter height of the (point) bead above the coverslip')}
    ${check('f-fitdz', 'Fit the z-step scale', true)}${check('f-drift', 'Fit xy drift per stack')}
    <label class="field"><span class="fl">Held out</span><select id="f-holdout"><option value="last-bead">last bead</option><option value="last-stack">last stack</option><option value="none">none</option></select></label>
    <details class="adv" data-key="adv-stack"><summary>Advanced <span class="dim">ROI, sampling, iterations</span></summary>
      ${field('f-roi', 'ROI', 26, 'px', '1')}${field('f-npupil', 'Pupil samples', 64, '', '1')}${field('f-os', 'Oversampling', 2, '', '1')}
      ${field('f-iter', 'Max iterations', 100, '', '1')}${field('f-minrel', 'Bead threshold', 0.3, 'rel.')}
    </details>
  </section>

  <section data-step="export">
    <h2>Export</h2><p class="sub">Download the results.</p>
    <div data-mode="movie locs" class="btncol">
      <button id="r-csv" title="ThunderSTORM csv (nm, frames from 1), as torch_psf Dataset.save_csv">Localizations (.csv)</button><p class="hint">CSV in the ThunderSTORM format${cite('thunderstorm')}.</p>
      <button id="i-zip" data-mode="movie" title="${ZIP_TITLE}">PSF (psf.zip)</button>
      <div data-mode="movie">${zipRow('i', 'the refined free knots when kept, else the fit tabulated on its knot grid')}</div>
      <button id="i-json" data-mode="movie">PSF (torch-psf json)</button>
      <button id="i-pdf" data-mode="movie" title="PDF report of the in-situ PSF estimate">Fit report (PDF)</button><p id="i-status" class="hint" data-mode="movie"></p>
    </div>
    <div data-mode="stack">
      ${field('e-focus', 'Imaging focus', 0.0, 'µm', 'any', 'focal-plane position above the coverslip for the exported PSF / spline')}
      ${field('e-roi', 'Spline ROI', 13, 'px', '1')}
      <div class="field"><span class="fl">Spline z range</span><span class="pair"><span class="inp"><input id="e-zmin" type="number" step="any" value="0"></span><span class="inp"><input id="e-zmax" type="number" step="any" value="0.8"><i>µm</i></span></span></div>
      <div class="btncol"><button id="x-zip" title="${ZIP_TITLE}">PSF (psf.zip)</button>${zipRow('x', 'the model tabulated on the spline grid above (ROI, z range)')}<button id="x-json">PSF (torch-psf json)</button><button id="x-knots">Spline knots (.npz)</button><button id="x-tif">Model z-stack (.tif)</button><button id="x-pdf" title="PDF report of the bead fit">Fit report (PDF)</button></div>
      <p id="x-status" class="hint"></p>
    </div>
  </section>
 </div>
 <div class="side-foot" id="runbox">
  <div class="foot" data-mode="movie">
    <div class="progress"><div id="mbar"></div></div><p id="mtext" class="hint">Estimate PSF collects isolated spots, fits a vectorial PSF to them, then localizes the movie with it.</p>
    <div class="btnrow idle"><button id="back" class="back" title="previous step">←</button><button id="next" class="primary"></button><button id="m-run" title="Estimate the PSF and localize the movie">Run all</button></div>
    <div class="btnrow running"><button id="m-enough" title="stop now and keep the current estimate">Stop early</button><button id="m-abort">Abort</button></div>
  </div>
  <div class="foot" data-mode="stack">
    <div class="progress"><div id="pbar"></div></div><p id="ptext" class="hint"></p>
    <div class="btnrow idle"><button id="sback" class="back" title="previous step">←</button><button id="snext" class="primary"></button><button id="run">Fit PSF</button></div>
    <div class="btnrow running"><button id="enough">Use what we have</button><button id="abort">Abort</button></div>
  </div>
 </div>
</aside>

<main class="grid">
  <section class="card" id="c-movie">
    <header><h3>Movie</h3><span class="meta" id="m-frame-v"></span><span class="tools">
      <button class="tb on" id="m-rings" title="show the localizations of this frame">Rings</button>
      <button class="tb on" id="m-link" title="movie and reconstruction share zoom and pan">Linked</button>
      <button class="tb" id="m-fit" title="fit the whole field (double-click the image)">Fit</button></span></header>
    <div class="view"><canvas id="movie" class="img" tabindex="0"></canvas></div>
    <div class="timeline"><button id="m-play" class="playbtn" title="play / pause"></button>
      <div class="tl"><canvas id="m-spark"></canvas><input id="m-frame" type="range" min="0" max="0" value="0"></div></div>
    <p id="movie-note" class="note clamp"></p>
  </section>

  <section class="card" id="c-recon">
    <header><h3>Reconstruction${cite('websmlm')}</h3><span class="meta" id="r-label"></span><span class="tools">
      <button class="tb" id="r-reset" title="top view, whole field">Reset view</button></span></header>
    <div class="view cloudwrap"><canvas id="recon" class="cloud" tabindex="0"></canvas><div id="r-empty" class="empty">Fills in live while the movie is localized.</div></div>
    <div class="ctrls">
      ${check('r-colorz', 'Colour by z', true)}<span class="mini"><input id="r-zlo" type="number" step="any" value="0">–<input id="r-zhi" type="number" step="any" value="0.8"> µm</span>
      <label class="mini">Splat <select id="r-mode"><option value="precision">precision</option><option value="fixed">fixed blur</option></select></label>
      <label class="mini" title="fixed blur, or the minimum sigma in precision mode">σ <input id="r-blur" type="number" step="any" value="8"> nm</label>
      <label class="mini" title="drop localizations with CRLB x above this (0 = keep all)">CRLB ≤ <input id="r-crlb" type="number" step="any" value="0"> nm</label>
    </div>
    <details class="adv" data-key="locs"><summary>3D view, details <span class="dim">drag: pan · wheel: zoom · shift-drag: orbit</span></summary>
      <div class="ctrls">
        <label class="mini">Tilt <input id="r-tilt" type="range" min="0" max="180" step="1" value="0"></label>
        <label class="mini">Rotate <input id="r-rot" type="range" min="0" max="360" step="1" value="0"></label>
        <label class="mini">z × <input id="r-zex" type="range" min="1" max="5" step="0.5" value="1"><span id="r-zex-v">1.0</span></label>
      </div>
      <table id="l-table" class="kvt"></table></details>
  </section>

  <section class="card" id="c-psf">
    <header><h3>PSF${cite('richards', 'noll')}</h3><span class="meta"><select id="preset"></select></span><span class="tools">
      <button class="tb" id="psf-wide" title="larger view">⤢</button></span></header>
    <div class="psfbox">
      <div id="view3d"></div>
      <div class="psfright">
        <div class="sections">
          <figure><canvas id="sec-xy" class="img" width="184" height="184"></canvas><figcaption>xy</figcaption></figure>
          <figure><canvas id="sec-xz" class="img" width="184" height="184"></canvas><figcaption>xz</figcaption></figure>
          <figure><canvas id="sec-yz" class="img" width="184" height="184"></canvas><figcaption>yz</figcaption></figure>
        </div>
        <div class="sliders"><label>x <input id="sx" type="range" min="0" max="100" step="0.1" value="50"></label>
          <label>y <input id="sy" type="range" min="0" max="100" step="0.1" value="50"></label>
          <label>z <input id="sz" type="range" min="0" max="100" step="0.1" value="50"></label></div>
        <div class="ctrls"><label class="mini">log decades <input id="decades" type="number" min="1" max="6" value="3"></label>
          <label class="mini">z × <input id="zexag" type="range" min="0.25" max="4" step="0.25" value="1"><span id="zexag-v">1.0</span></label></div>
      </div>
    </div>
    <p id="sec-label" class="note clamp1"></p>
    <p id="preset-note" class="note clamp1"></p>
  </section>

  <section class="card" id="c-fit">
    <header><h3 id="fit-title">PSF fit</h3><span class="meta" id="i-meta"></span><span class="tools">
      <button class="tb" id="i-report-btn" data-mode="movie">Report</button></span></header>
    <div id="fit-insitu" data-mode="movie">
      <div id="i-empty" class="emptytxt">Run <b>Estimate PSF</b> to fit the PSF to this movie.</div>
      <div id="p-insitu" class="fitbox" hidden>
        <div><canvas id="i-zern" width="330" height="160"></canvas><p class="note">Zernike coefficients (waves rms) by Noll index${cite('noll')}</p></div>
        <div id="i-kv" class="kv"></div>
      </div>
      <div id="i-report" hidden><table id="i-table" class="kvt"></table></div>
    </div>
    <div id="fit-bead" data-mode="stack">
      <div id="b-empty" class="emptytxt">Open a bead z-stack and run <b>Fit PSF</b>.</div>
      <div id="p-results" hidden>
        <table id="res-table" class="kvt"></table>
        <canvas id="zern" width="460" height="150"></canvas>
        <p class="note">Zernike coefficients (waves rms) by Noll index</p>
        <div class="ctrls"><label class="mini">Bead <select id="bead"></select></label><label class="mini">Plane <input id="plane" type="range" min="0" max="100" value="50"></label></div>
        <div class="panels">
          <figure><canvas id="b-data" class="img" width="96" height="96"></canvas><figcaption>data xy</figcaption></figure>
          <figure><canvas id="b-model" class="img" width="96" height="96"></canvas><figcaption>model xy</figcaption></figure>
          <figure><canvas id="b-res" class="img" width="96" height="96"></canvas><figcaption>residual</figcaption></figure>
          <figure><canvas id="b-dxz" class="img" width="96" height="160"></canvas><figcaption>data xz</figcaption></figure>
          <figure><canvas id="b-mxz" class="img" width="96" height="160"></canvas><figcaption>model xz</figcaption></figure>
          <figure><canvas id="b-rxz" class="img" width="96" height="160"></canvas><figcaption>residual xz</figcaption></figure>
        </div>
        <p id="b-label" class="note"></p>
      </div>
    </div>
  </section>
</main>
</div>
${REF_LIST}
<footer>Everything runs in your browser (WebGPU); no data leave the machine. Python reference: <a href="https://github.com/jcnossen/torch-psf" target="_blank" rel="noopener">torch-psf</a>.</footer>`;
