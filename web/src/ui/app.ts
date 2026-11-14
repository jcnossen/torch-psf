// The web app (UI thread). Input panel with two tabs: "SMLM movie" (default; a simulated demo movie is loaded at
// start-up; one button: Estimate PSF = spots -> in-situ vectorial fit -> localization with the fitted PSF ->
// reconstruction) and "Bead z-stack" (the bead fit). Only the essential numbers are shown; the rest is under
// "Advanced". All GPU work runs in the compute worker (../worker.ts).
import { View3D, Volume } from './view3d';
import { drawArray } from './colormap';
import { DEFAULT_NOLL, Optics } from '../psf/vectorial';
import { makePsfJson, readPsfJson } from '../io/psfjson';
import { WEBGPU_HELP } from '../gpu/device';
import { TiffMovie } from '../io/tiff';
import type { FitForm, FitSummary, VolumeSpec, Req, MovieForm, InsituForm, MovieSourceSpec } from './protocol';
import { InsituReport, FREE_KNOT_WARNING } from '../smlm/estimate';
import { splineGrid } from '../gpu/spline_lm';
import { Recon } from './recon';
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
  { name: 'tetrapod', label: 'Tetrapod-like (Zernike approx.)',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({ 6: 0.2, 12: -0.2 }), nxy: 64, dxy_um: 0.075, zmin: -1.6, zmax: 1.6, nz: 65, ze_um: 0 },
    note: 'Tetrapod-like PSF approximated by a Zernike combination: Noll 6 = +0.20, Noll 12 = -0.20 waves rms (not the published optimized tetrapod pupil). NA 1.49, 680 nm; z = defocus of an emitter at the coverslip, range 3.2 um.' },
  { name: 'focus', label: 'In-focus 2D',
    spec: { optics: OIL, npupil: 64, noll: DEFAULT_NOLL, zcoef: zc({}), nxy: 40, dxy_um: 0.04, zmin: -0.4, zmax: 0.4, nz: 41, ze_um: 0 },
    note: 'Aberration-free vectorial PSF, NA 1.49, 680 nm, range 0.8 um.' },
];
const START_GUESS: Record<number, number> = { 6: 0.08 };   // the default start of the in-situ fit: a generic astigmatic PSF

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
  const setNum = (id: string, v: number) => { $<HTMLInputElement>(id).value = String(v); };
  const optics = (): Optics => ({ pixel_um: n('o-px'), NA: n('o-na'), wavelength_um: n('o-wl'), n_imm: n('o-nimm'), n_med: n('o-nmed') });
  let fitResult: FitSummary | null = null;
  let insitu: InsituReport | null = null;
  let current: { vol: Volume; spec: VolumeSpec } | null = null;
  let truthZ: Record<number, number> | null = null;      // the demo's true Zernikes

  // --- WebGPU, then the demo movie and the default view
  const status = $('gpu-status');
  call({ type: 'init' }).then(
    (r) => { status.textContent = 'WebGPU: ' + r.gpu; status.dataset.ok = '1'; showPreset('astig'); loadDemo(); },
    (e) => {
      status.textContent = 'WebGPU unavailable'; status.dataset.ok = '0';
      $('nogpu').hidden = false; $('nogpu').textContent = String(e.message || e) || WEBGPU_HELP;
      for (const b of ['m-run', 'run']) $<HTMLButtonElement>(b).disabled = true;
    });

  // --- 3D view
  const sel = $<HTMLSelectElement>('preset');
  const fillPresets = () => {
    const v = sel.value;
    sel.innerHTML = PRESETS.map((p) => `<option value="${p.name}">${p.label}</option>`).join('') +
      `<option value="insitu" ${insitu ? '' : 'disabled'}>In-situ estimate</option><option value="refined" ${insitu?.refine?.knots ? '' : 'disabled'}>In-situ refined spline</option><option value="fit" ${fitResult ? '' : 'disabled'}>Bead fit result</option>`;
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
    $('preset-note').textContent = 'computing...';
    const r = await call({ type: 'volume', spec: p.spec });
    setVolume(r.volume, p.spec, p.note);
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
  view.onSliceChange = (axis, f) => { $<HTMLInputElement>(['sx', 'sy', 'sz'][axis]).value = String(f * 100); drawSections(); };
  for (const [i, id] of ['sx', 'sy', 'sz'].entries()) $<HTMLInputElement>(id).oninput = (e) => { view.setSlice(i as 0 | 1 | 2, +(e.target as HTMLInputElement).value / 100); drawSections(); };
  $<HTMLInputElement>('zexag').oninput = (e) => { view.setZExag(+(e.target as HTMLInputElement).value); $('zexag-v').textContent = view.opts.zExag.toFixed(1); };
  $<HTMLInputElement>('decades').onchange = (e) => { view.opts.decades = +(e.target as HTMLInputElement).value; view.updateSlices(); drawSections(); };

  // ================================================================ SMLM movie tab
  let source: MovieSourceSpec = 'demo';
  let player: { n: number; H: number; W: number; frame: (t: number) => Promise<Float32Array> } | null = null;
  let recon: Recon | null = null;
  let psfJson: { noll: number[]; zcoef: number[]; name: string } | null = null;

  async function loadDemo() {
    $('movie-note').textContent = 'simulating the demo movie...';
    const d = await call({ type: 'demo' });
    truthZ = d.zernike;
    const fp = d.H * d.W, adu: Uint16Array = d.adu;
    player = { n: d.T, H: d.H, W: d.W, frame: async (t) => Float32Array.from(adu.subarray(t * fp, (t + 1) * fp)) };
    source = 'demo';
    locsByFrame.clear();
    const o: Optics = d.optics, p = d.params;
    setNum('o-px', o.pixel_um); setNum('o-na', o.NA); setNum('o-wl', o.wavelength_um); setNum('o-nimm', o.n_imm); setNum('o-nmed', o.n_med);
    setNum('o-gain', p.gain); setNum('o-offset', p.offset); setNum('o-rn', p.readnoise); setNum('m-focus', p.focus_um); setNum('m-zlo', p.zlo); setNum('m-zhi', p.zhi);
    $('mfiles').textContent = 'now: the simulated demo movie';
    $('movie-note').textContent = `Simulated demo movie (generated in your browser, not real data): ${d.T} frames of ${d.W} x ${d.H} px, ` +
      `${d.n_events} blinking events on ${p.sites === 'filaments' ? p.filaments + ' filaments' : p.nsites + ' random sites'}, astigmatic PSF with Noll 5 / 6 / 11 = ${d.zernike[5]} / ${d.zernike[6]} / ${d.zernike[11]} waves, ` +
      `focal plane ${p.focus_um} um, emitters ${p.zlo}-${p.zhi} um, ~${p.photons} photons, bg ${p.bg} / px, camera gain ${p.gain} e-/ADU, offset ${p.offset}, read noise ${p.readnoise} e-. ` +
      `"Estimate PSF" starts from a generic astigmatic guess (Noll 6 = ${START_GUESS[6]}), not from the truth.`;
    startPlayer();
    (window as any).__demoReady = true;
  }

  // --- movie player: autoplay (loop), pause on interaction, frame slider, localizations overlaid after a run
  const mcv = $('movie') as HTMLCanvasElement, slider = $<HTMLInputElement>('m-frame'), playBtn = $<HTMLButtonElement>('m-play');
  let playing = true, frameNow = 0, playTimer = 0, lims: [number, number] | null = null, drawing = false;
  const locsByFrame = new Map<number, number[]>();
  async function showFrame(t: number) {
    if (!player || drawing) return;
    drawing = true;
    try {
      frameNow = t; slider.max = String(player.n - 1); slider.value = String(t);
      const a = await player.frame(t);
      if (!lims) { const s = Float32Array.from(a).sort(); lims = [s[Math.floor(0.01 * s.length)], s[Math.floor(0.999 * s.length)]]; }
      const g = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) g[i] = Math.max(0, a[i] - lims[0]);
      drawArray(mcv, g, player.W, player.H, { vmax: lims[1] - lims[0], gray: true });
      const ctx = mcv.getContext('2d')!, sc = mcv.width / player.W;
      ctx.strokeStyle = '#e8a33d'; ctx.lineWidth = 1.5;
      const L = locsByFrame.get(t) ?? [];
      for (let i = 0; i < L.length; i += 2) { ctx.beginPath(); ctx.arc((L[i] + 0.5) * sc, (L[i + 1] + 0.5) * sc, 3.5 * sc, 0, 2 * Math.PI); ctx.stroke(); }
      $('m-frame-v').textContent = `${t + 1} / ${player.n}`;
    } finally { drawing = false; }
  }
  function startPlayer() {
    lims = null;
    const sz = Math.min(512, Math.max(256, 4 * player!.W));
    mcv.width = sz; mcv.height = Math.round(sz * player!.H / player!.W);
    clearInterval(playTimer);
    playTimer = window.setInterval(() => { if (playing && player && !document.hidden) showFrame((frameNow + 1) % player.n); }, 70);
    showFrame(0);
  }
  const setPlaying = (p: boolean) => { playing = p; playBtn.textContent = p ? 'Pause' : 'Play'; };
  playBtn.onclick = () => setPlaying(!playing);
  slider.oninput = () => { setPlaying(false); showFrame(+slider.value); };
  mcv.onpointerdown = () => setPlaying(false);

  // --- tabs (after the player: the tab switch shows / hides its panel)
  const setTab = (t: string) => {
    for (const b of root.querySelectorAll<HTMLButtonElement>('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === t));
    $('tab-movie').hidden = t !== 'movie'; $('tab-stack').hidden = t !== 'stack';
    $('p-results').hidden = t !== 'stack' || !fitResult;
    $('p-movie').hidden = t !== 'movie';
    $('p-recon').hidden = t !== 'movie' || !recon;
    $('p-insitu').hidden = t !== 'movie' || !insitu;
    try { localStorage.setItem('tpsf.tab', t); } catch { /* no storage */ }
  };
  for (const b of root.querySelectorAll<HTMLButtonElement>('.tabs button')) b.onclick = () => setTab(b.dataset.tab!);
  let tab0 = 'movie'; try { tab0 = localStorage.getItem('tpsf.tab') === 'stack' ? 'stack' : 'movie'; } catch { /* no storage */ }
  setTab(tab0);

  // --- movie files (replace the demo)
  const mdrop = $('mdrop');
  const setMFiles = async (fl: FileList | File[]) => {
    const files = [...fl].filter((f) => /\.tiff?$/i.test(f.name));
    if (!files.length) return;
    source = files; truthZ = null;
    $('mfiles').textContent = 'now: ' + files.map((f) => `${f.name} (${(f.size / 2 ** 20).toFixed(1)} MB)`).join(', ');
    const mv = await TiffMovie.open(files);
    player = { n: mv.n, H: mv.H, W: mv.W, frame: (t) => mv.read(t, t + 1) };
    locsByFrame.clear();
    $('movie-note').textContent = `${files.map((f) => f.name).join(' + ')}: ${mv.n} frames of ${mv.W} x ${mv.H} px (${mv.dtype}). Set the pixel size, NA and wavelength above, the camera and the focal plane under Advanced.`;
    startPlayer();
  };
  (window as any).__setMovieFiles = setMFiles;
  mdrop.ondragover = (e) => { e.preventDefault(); mdrop.classList.add('over'); };
  mdrop.ondragleave = () => mdrop.classList.remove('over');
  mdrop.ondrop = (e) => { e.preventDefault(); mdrop.classList.remove('over'); if (e.dataTransfer) setMFiles(e.dataTransfer.files); };
  $<HTMLInputElement>('mfile').onchange = (e) => setMFiles((e.target as HTMLInputElement).files!);
  $('m-demo').onclick = (e) => { e.preventDefault(); loadDemo(); };

  // --- start PSF of the fit / the PSF for "localize only"
  const psfSel = $<HTMLSelectElement>('m-psf');
  psfSel.onchange = () => { $('m-psfjson').hidden = psfSel.value !== 'json'; };
  $<HTMLInputElement>('m-psfjson').onchange = async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    try {
      const j = readPsfJson(await f.text());
      if (!Array.isArray(j.noll) || !Array.isArray(j.zcoef)) throw new Error('no zernike_noll / zernike_waves');
      psfJson = { noll: j.noll.map(Number), zcoef: j.zcoef.map(Number), name: f.name };
      $('m-psfinfo').textContent = `${f.name}: ${fmtZ(psfJson.noll, psfJson.zcoef)}`;
    } catch (err) { psfJson = null; $('m-psfinfo').textContent = 'not a torch-psf PSF json: ' + (err as Error).message; }
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
  });

  // --- progress, stop
  let running: 'stack' | 'movie' = 'movie';
  const mRun = $<HTMLButtonElement>('m-run'), mLoc = $<HTMLButtonElement>('m-loc'), mStop = $<HTMLButtonElement>('m-enough'), mAbort = $<HTMLButtonElement>('m-abort');
  const runBtn = $<HTMLButtonElement>('run'), enoughBtn = $<HTMLButtonElement>('enough'), abortBtn = $<HTMLButtonElement>('abort');
  const busy = (b: boolean) => { mRun.disabled = b; mLoc.disabled = b; runBtn.disabled = b; mStop.disabled = !b; mAbort.disabled = !b; enoughBtn.disabled = !b; abortBtn.disabled = !b; };
  busy(false);
  for (const b of [mStop, enoughBtn]) b.onclick = () => worker.postMessage({ type: 'stop', signal: 'enough' });
  for (const b of [mAbort, abortBtn]) b.onclick = () => worker.postMessage({ type: 'stop', signal: 'abort' });
  const STAGES: Record<string, string> = { spots: 'collecting spots', fit: 'fitting the PSF', report: 'held-out check', localize: 'localizing with the fitted PSF', spline: 'building the spline', header: 'reading the TIFF header' };
  onProgress = (p) => {
    const frac = p.total ? Math.min(1, p.done / p.total) : 0;
    const [bar, txt] = running === 'movie' ? ['mbar', 'mtext'] : ['pbar', 'ptext'];
    $(bar).style.width = (100 * frac).toFixed(1) + '%';
    const fmt = (v: unknown) => typeof v === 'number' ? (Math.abs(v) < 1e-2 || Math.abs(v) > 1e4 ? v.toExponential(3) : v.toFixed(4)) : String(v);
    const info = Object.entries(p.info ?? {}).filter(([k]) => k !== 'zcoef').map(([k, v]) => `${k.replace(/_/g, ' ')} ${fmt(v)}`).join(', ');
    const t = p.elapsed_s !== undefined ? ` | ${p.elapsed_s.toFixed(1)} s` + (p.eta_s !== undefined ? `, ~${p.eta_s.toFixed(0)} s left` : '') : '';
    $(txt).textContent = `${running === 'movie' ? (STAGES[p.stage] ?? p.stage) : p.stage}: ${p.done} / ${p.total}${info ? ' | ' + info : ''}${t}`;
    if (running === 'movie' && p.stage === 'fit' && p.info?.zcoef) drawZernikeBars(String(p.info.zcoef).split(' ').map(Number), startZ);
    if (running === 'stack') enoughBtn.textContent = p.stage === 'read' ? 'Enough frames' : p.stage === 'fit' ? 'Stop and keep current estimate' : 'Use what we have';
  };

  // --- reconstruction
  let lastDraw = 0, drawTimer = 0;
  const drawRecon = () => {
    if (!recon) return;
    const r = recon.render($('recon') as HTMLCanvasElement, { pixel_nm: n('r-px'), blur_px: n('r-blur'), colorZ: $<HTMLInputElement>('r-colorz').checked, zlo: n('m-zlo'), zhi: n('m-zhi'), maxCrlbNm: n('r-crlb') });
    $('r-label').textContent = `${r.shown} of ${recon.n} localizations, ${r.w} x ${r.h} pixels of ${n('r-px')} nm` + ($<HTMLInputElement>('r-colorz').checked ? `; colour = z from ${n('m-zlo')} (blue) to ${n('m-zhi')} um (red)` : '');
    lastDraw = performance.now();
  };
  for (const id of ['r-px', 'r-blur', 'r-crlb', 'r-colorz']) $(id).addEventListener('change', drawRecon);
  onLocs = (rows) => {
    recon?.add(rows);
    for (let o = 0; o < rows.length; o += NCOL) { const f = rows[o]; const L = locsByFrame.get(f) ?? []; L.push(rows[o + 1], rows[o + 2]); locsByFrame.set(f, L); }
    if (performance.now() - lastDraw > 700) drawRecon(); else { clearTimeout(drawTimer); drawTimer = window.setTimeout(drawRecon, 700); }
  };
  $('r-csv').onclick = () => { if (recon) download(locsToCsv(recon.rows(), 1000 * n('o-px')), 'localizations.csv', 'text/csv'); };

  // --- the one button: Estimate PSF (then localize the movie with it)
  let startZ: number[] | null = null;
  onInsitu = (r) => { insitu = r; fillPresets(); showInsitu(r); if (r.refine?.knots) showRefinedVolume(); else showInsituVolume(); };
  async function runMovie(kind: 'estimate' | 'localize') {
    if (!player) { $('mtext').textContent = 'No movie yet.'; return; }
    const psf = kind === 'localize' && psfSel.value === 'guess' && insitu ? { noll: insitu.noll, zcoef: insitu.zcoef } : startPsf();
    if (!psf) { $('mtext').textContent = 'Choose the start PSF first (Advanced).'; return; }
    const form = movieForm(psf);
    if (form.roisize % 2 !== 1 || form.roisize < 5) { $('mtext').textContent = 'The ROI size must be odd (>= 5).'; return; }
    running = 'movie'; busy(true);
    startZ = kind === 'estimate' ? [...psf.zcoef] : null;
    recon = new Recon(player.H, player.W, form.optics.pixel_um); locsByFrame.clear();
    $('p-recon').hidden = false;
    $('mtext').textContent = 'starting...';
    try {
      const src: MovieSourceSpec = source;
      const r = kind === 'estimate' ? await call({ type: 'estimate', source: src, form, localize: true }) : await call({ type: 'movie', source: src, form: form as MovieForm });
      drawRecon(); setPlaying(false); showFrame(frameNow);
      const s = r.summary as MovieSummary | null;
      if (s) showLocs(s);
      $('mbar').style.width = '100%';
      $('mtext').textContent = kind === 'estimate'
        ? `done: PSF from ${r.report.n_train} spots in ${r.report.iterations} iterations (${r.report.seconds.toFixed(1)} s)` + (s ? `; then ${s.n_locs} localizations (${s.seconds.toFixed(1)} s)` : '')
        : `done: ${s!.n_locs} localizations from ${s!.frames_read} frames in ${s!.seconds.toFixed(1)} s`;
      (window as any).__movieDone = r;
    } catch (e) {
      const err = e as Error & { abort?: boolean };
      $('mtext').textContent = err.abort ? 'aborted' : 'error: ' + err.message;
      $('mbar').style.width = '0%';
    } finally { busy(false); }
  }
  mRun.onclick = () => runMovie('estimate');
  mLoc.onclick = () => runMovie('localize');

  async function showInsituVolume() {
    if (!insitu) return;
    const zlo = n('m-zlo'), zhi = n('m-zhi');
    const spec: VolumeSpec = { optics: optics(), npupil: 64, noll: insitu.noll, zcoef: insitu.zcoef, nxy: 48, dxy_um: 0.04, zmin: zlo, zmax: zhi, nz: 41, ze_um: 0, focus_um: n('m-focus') };
    const r = await call({ type: 'volume', spec });
    sel.value = 'insitu';
    setVolume(r.volume, spec, `In-situ estimate (Noll 5-21; ${fmtZ(insitu.noll, insitu.zcoef)}). z = emitter height above the coverslip (${zlo}-${zhi} um) at the focal plane ${n('m-focus')} um.`);
  }
  function showRefinedVolume() {
    const r = insitu!, K = r.refine!.knots!;
    const g = splineGrid(n('m-roi'), n('m-zlo'), n('m-zhi'), 2, 0.04, r.grid.zpad, 3.0);
    const spec: VolumeSpec = { optics: optics(), npupil: 64, noll: r.noll, zcoef: r.zcoef, nxy: g.nu, dxy_um: n('o-px') / g.S, zmin: g.z0, zmax: g.z0 + (g.nz - 1) * g.hz, nz: g.nz, ze_um: 0, focus_um: n('m-focus') };
    sel.value = 'refined';
    setVolume(K, spec, `In-situ refined spline: free knots regularised towards the vectorial fit (kept: held-out NLL better). The knots themselves, ` +
      `every ${(1000 * n('o-px') / g.S).toFixed(0)} nm laterally and ${(1000 * g.hz).toFixed(0)} nm in z. ${FREE_KNOT_WARNING}.`);
  }
  function showInsitu(r: InsituReport) {
    $('p-insitu').hidden = false;
    const rows: [string, string][] = [
      ['spots (fit / held out)', `${r.n_train} / ${r.n_test} of ${r.n_spots}, from ${r.frames_read} of ${r.frames_total} frames`],
      ['overlapping spots rejected', r.deviance_threshold === null ? 'off' : `${r.n_rejected} (deviance per pixel > ${r.deviance_threshold.toFixed(3)}), then a second pass`],
      ['iterations', `${r.iterations}${r.partial ? ' (stopped early)' : ''}, ${r.seconds.toFixed(1)} s`],
      ['held-out NLL / px, fitted - start', r.heldout ? `${r.heldout.diff.toFixed(4)} (negative: the fitted PSF explains spots it was not fitted to better)` : 'no held-out spots'],
      ['z scale vs the start PSF', `${r.z_scale.toFixed(3)} (offset ${(1000 * r.z_offset_um).toFixed(0)} nm)`],
      ['spots at the offset limit', (100 * r.at_limit).toFixed(2) + ' %'],
    ];
    if (r.refine) {
      const f = r.refine;
      rows.push(['spline refinement', `${f.kept ? 'kept' : 'not kept'}: held-out NLL / px ${f.heldout_refined === null ? 'n/a' : (f.heldout_refined - (r.heldout?.start ?? 0)).toFixed(4)} vs start ` +
        `(vectorial ${r.heldout ? r.heldout.diff.toFixed(4) : 'n/a'}); z scale vs vectorial ${f.z_scale_vs_vectorial.toFixed(3)}; ${f.outer} outer iterations, ${f.seconds.toFixed(1)} s` +
        (f.kept ? '. The movie is localized with the refined spline. ' + FREE_KNOT_WARNING : '')]);
    }
    if (truthZ) {
      const t = DEFAULT_NOLL.map((j) => truthZ![j] ?? 0);
      const rms = (a: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - t[i]) ** 2, 0) / t.length);
      rows.push(['vs the demo truth (rms over Noll 5-21)', `start ${rms(r.zcoef_start).toFixed(4)} w, fitted ${rms(r.zcoef).toFixed(4)} w`]);
    }
    $('i-table').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
    drawZernikeBars(r.zcoef, r.zcoef_start);
  }
  function drawZernikeBars(z: number[], start: number[] | null) {
    const truth = truthZ ? DEFAULT_NOLL.map((j) => truthZ![j] ?? 0) : null;
    drawZernikeSets($('i-zern') as HTMLCanvasElement, DEFAULT_NOLL, [
      ...(start ? [{ z: start, color: '#6b7385', label: 'start' }] : []),
      { z, color: '#e8a33d', label: 'fitted' },
      ...(truth ? [{ z: truth, color: '#3dbf7a', label: 'truth (demo)' }] : []),
    ]);
    $('p-insitu').hidden = false;
  }
  $('i-json').onclick = () => {
    if (!insitu) return;
    const j = makePsfJson({ optics: optics(), noll: insitu.noll, zcoef: insitu.zcoef, focus_um: n('m-focus'), npupil: 64, oversample: 3,
      provenance: { source: 'torch-psf web in-situ fit (knot route, torch_psf estimate_psf_insitu)', created: new Date().toISOString(),
        movie: source === 'demo' ? 'simulated demo' : (source as File[]).map((f) => f.name),
        n_train: insitu.n_train, n_test: insitu.n_test, iterations: insitu.iterations, partial: insitu.partial, heldout_nll_px_diff: insitu.heldout?.diff ?? null,
        z_scale_vs_start: insitu.z_scale, z_range_um: [n('m-zlo'), n('m-zhi')], gain_e_per_adu: n('o-gain'), offset_adu: n('o-offset') } });
    download(JSON.stringify(j, null, 1), 'psf_insitu.json', 'application/json');
  };
  function showLocs(s: MovieSummary) {
    const rows = recon!.rows(), N = rows.length / NCOL;
    const med = (k: number, f = 1) => { const a = new Float32Array(N); for (let i = 0; i < N; i++) a[i] = rows[i * NCOL + k] * f; a.sort(); return N ? a[N >> 1] : NaN; };
    const px = n('o-px') * 1000;
    $('l-table').innerHTML = ([
      ['localizations', `${s.n_locs} (${(s.n_locs / Math.max(1, s.frames_read)).toFixed(2)} per frame) from ${s.frames_read} of ${s.frames_total} frames`],
      ['median photons / bg', `${med(4).toFixed(0)} / ${med(5).toFixed(1)}`],
      ['median CRLB x / y / z', `${med(6, px).toFixed(1)} / ${med(7, px).toFixed(1)} / ${med(8, 1000).toFixed(1)} nm`],
      ['at the offset limit', (100 * s.at_limit_fraction).toFixed(2) + ' %'],
      ['time', `${s.seconds.toFixed(1)} s (${s.spots_per_s.toFixed(0)} spots / s)`],
    ] as [string, string][]).map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
  }

  // ================================================================ bead z-stack tab
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
  runBtn.onclick = async () => {
    running = 'stack';
    if (!files.length) { $('ptext').textContent = 'Drop or choose a bead z-stack TIFF first.'; return; }
    busy(true);
    $('ptext').textContent = 'starting...';
    try {
      const r = await call({ type: 'fit', files, form: readForm() });
      fitResult = r.summary as FitSummary;
      fillPresets();
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
    return {
      optics: optics(), sign: n('f-sign'), dz_um: n('f-dz'), gain: n('o-gain'), offset: n('o-offset'), ze_um: n('f-ze'),
      roisize: n('f-roi'), npupil: n('f-npupil'), oversample: n('f-os'),
      fitDz: $<HTMLInputElement>('f-fitdz').checked, fitDrift: $<HTMLInputElement>('f-drift').checked,
      holdout: $<HTMLSelectElement>('f-holdout').value as FitForm['holdout'], maxIter: n('f-iter'), minRel: n('f-minrel'),
    };
  }
  const beadNote = (r: FitSummary) => `Bead fit (Noll 5-21) of ${r.stacks.map((s) => s.name).join(', ')}; z = defocus of the bead (stage coordinate, z-step scale applied), xy 2x oversampled.`;
  function showResult(r: FitSummary) {
    $('p-results').hidden = false;
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
    $('res-table').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
    drawZernikeSets($('zern') as HTMLCanvasElement, r.noll, [{ z: r.zcoef, color: '#e8a33d', label: 'fitted' }]);
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

function download(bytes: BlobPart, name: string, type = 'application/octet-stream') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([bytes], { type }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

const fmtZ = (noll: number[], z: number[]) => noll.map((j, i) => [j, z[i]] as [number, number]).filter(([, v]) => Math.abs(v) >= 0.005)
  .map(([j, v]) => `${j}: ${v.toFixed(3)}`).join(', ') || 'none';

function flipRows(a: Float32Array, w: number, h: number) {
  const o = new Float32Array(a.length);
  for (let r = 0; r < h; r++) o.set(a.subarray(r * w, (r + 1) * w), (h - 1 - r) * w);
  return o;
}

/** Zernike coefficients (waves rms) by Noll index; several sets side by side (e.g. start, fitted, truth). */
function drawZernikeSets(cv: HTMLCanvasElement, noll: number[], sets: { z: number[]; color: string; label: string }[]) {
  const ctx = cv.getContext('2d')!;
  const W = cv.width, H = cv.height, pad = 34;
  ctx.clearRect(0, 0, W, H);
  const m = Math.max(0.02, ...sets.flatMap((s) => s.z.map(Math.abs)));
  const bw = (W - pad - 6) / noll.length, y0 = H / 2 + 4, k = sets.length;
  ctx.strokeStyle = '#667'; ctx.beginPath(); ctx.moveTo(pad, y0); ctx.lineTo(W - 4, y0); ctx.stroke();
  ctx.font = '11px sans-serif'; ctx.fillStyle = '#aab';
  ctx.fillText(`+${m.toFixed(3)}`, 0, 28); ctx.fillText(`-${m.toFixed(3)}`, 0, H - 16);
  const amp = H / 2 - 22;
  noll.forEach((j, i) => {
    sets.forEach((s, q) => {
      const h = (s.z[i] / m) * amp;
      ctx.fillStyle = s.color;
      ctx.fillRect(pad + i * bw + 2 + q * (bw - 4) / k, h >= 0 ? y0 - h : y0, Math.max(1, (bw - 4) / k - 1), Math.abs(h));
    });
    ctx.fillStyle = '#ccd';
    ctx.fillText(String(j), pad + i * bw + bw / 2 - 6, H - 2);
  });
  let x = pad + 4;
  for (const s of sets) { ctx.fillStyle = s.color; ctx.fillRect(x, 4, 10, 10); ctx.fillStyle = '#ccd'; ctx.fillText(s.label, x + 14, 13); x += 24 + ctx.measureText(s.label).width; }
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
      <label>z exaggeration <input id="zexag" type="range" min="0.25" max="4" step="0.25" value="1"><span id="zexag-v">1.0</span></label>
      <label>log decades <input id="decades" type="number" min="1" max="6" value="3" style="width:3em"></label>
    </div>
    <div id="view3d"></div>
    <div class="bar">slice planes (or drag them): <label>x <input id="sx" type="range" min="0" max="100" step="0.1" value="50"></label>
      <label>y <input id="sy" type="range" min="0" max="100" step="0.1" value="50"></label>
      <label>z <input id="sz" type="range" min="0" max="100" step="0.1" value="50"></label></div>
    <p id="preset-note" class="note"></p>
  </details>
  <details class="panel" data-key="movie" open id="p-movie"><summary><h2>Movie</h2></summary>
    <canvas id="movie" width="384" height="384"></canvas>
    <div class="bar"><button id="m-play">Pause</button><input id="m-frame" type="range" min="0" max="0" value="0" style="flex:1"><span id="m-frame-v"></span></div>
    <p id="movie-note" class="note"></p>
  </details>
  <details class="panel" data-key="recon" open id="p-recon" hidden><summary><h2>Reconstruction</h2></summary>
    <div class="bar">
      ${num('r-px', 'pixel (nm)', 20)}${num('r-blur', 'blur (pixels)', 0.7)}${num('r-crlb', 'max CRLB x (nm, 0 = all)', 0)}
      <label><input id="r-colorz" type="checkbox" checked> colour by z</label>
      <button id="r-csv" title="ThunderSTORM csv (nm, frames from 1), as torch_psf Dataset.save_csv">localizations .csv</button>
    </div>
    <div class="recon-wrap"><canvas id="recon" width="10" height="10"></canvas></div>
    <p id="r-label" class="note"></p>
    <table id="l-table"></table>
  </details>
  <details class="panel" data-key="sections"><summary><h2>PSF sections</h2></summary>
    <div class="panels">
      <figure><canvas id="sec-xy" width="200" height="200"></canvas><figcaption>xy</figcaption></figure>
      <figure><canvas id="sec-xz" width="200" height="200"></canvas><figcaption>xz</figcaption></figure>
      <figure><canvas id="sec-yz" width="200" height="200"></canvas><figcaption>yz</figcaption></figure>
    </div>
    <p id="sec-label" class="note"></p>
  </details>
  </div>
  <div class="col">
  <details class="panel" data-key="input" open><summary><h2>Input</h2></summary>
    <div class="tabs" role="tablist"><button data-tab="movie" role="tab">SMLM movie</button><button data-tab="stack" role="tab">Bead z-stack</button></div>
    <div class="essentials">${num('o-px', 'pixel size (um)', 0.108)}${num('o-na', 'NA', 1.49)}${num('o-wl', 'wavelength (um)', 0.68)}</div>
    <details class="adv" data-key="adv-optics"><summary>Advanced: refractive indices, camera</summary>
      <div class="advgrid">${num('o-nimm', 'n immersion', 1.518)}${num('o-nmed', 'n medium', 1.33)}${num('o-gain', 'camera gain (e-/ADU)', 0.5)}${num('o-offset', 'camera offset (ADU)', 100)}${num('o-rn', 'read noise (e- rms, localization)', 0)}</div>
    </details>
    <div id="tab-movie">
      <div id="mdrop" class="drop"><input id="mfile" type="file" multiple accept=".tif,.tiff"><p>Drop your SMLM movie (TIFF, streamed) to replace the demo. <a href="#" id="m-demo">Back to the demo</a></p><p id="mfiles" class="note"></p></div>
      <div class="buttons"><button id="m-run" class="primary">Estimate PSF</button><button id="m-enough">Stop early</button><button id="m-abort">Abort</button></div>
      <div class="progress"><div id="mbar"></div></div><p id="mtext" class="note">Estimate PSF collects isolated spots, fits a vectorial PSF to them (in situ), then localizes the movie with it.</p>
      <details class="adv" data-key="adv-movie"><summary>Advanced</summary>
        <div class="advgrid">
          ${num('m-focus', 'focal plane (um)', 0.4, 'any', 'focal-plane position above the coverslip during the acquisition')}${num('m-zlo', 'emitter z from (um)', 0)}${num('m-zhi', 'to (um)', 0.8)}
          ${num('m-roi', 'ROI (px, odd)', 15, '2')}${num('m-thr', 'detection threshold (x noise)', 6)}${num('m-iso', 'isolation (px, 0 = off)', 0, '1')}
          ${num('m-nspots', 'spots for the fit', 8000, '1')}${num('m-nheld', 'held-out spots', 4000, '1')}${num('m-iters', 'iterations', 80, '1')}
          ${num('m-chunk', 'frames per chunk', 500, '1', 'the detection noise is estimated per chunk (torch_psf default 500)')}${num('m-frames', 'max frames (0 = all)', 0, '1')}
          <label title="free spline knots regularised towards the fitted vectorial PSF (torch_psf InSituEstimator refine=True); kept only if the held-out NLL improves"><input id="m-refine" type="checkbox"> spline refinement</label>
          ${num('m-devk', 'overlap rejection k (0 = off)', 3, 'any', 'after the first pass, spots with a Poisson deviance per pixel above 1 + k sqrt(2 / P^2) are dropped and the fit is repeated (torch_psf InSituEstimator deviance_k)')}
          <label>start PSF <select id="m-psf"><option value="guess">generic astigmatic (Noll 6 = 0.08)</option><option value="view">the PSF in the 3D view</option><option value="insitu">the last in-situ estimate</option><option value="json">a PSF json file</option></select></label>
          <input id="m-psfjson" type="file" accept=".json" hidden><span id="m-psfinfo" class="note"></span>
        </div>
        <div class="buttons"><button id="m-loc" title="localize with the start PSF (or the last estimate), without estimating the PSF">Localize only</button></div>
        <p class="note">Defaults of torch_psf InSituEstimator / PSF.localize; every step is a port of the Python code and is tested against it.</p>
      </details>
    </div>
    <div id="tab-stack" hidden>
      <div id="drop" class="drop"><input id="file" type="file" multiple accept=".tif,.tiff"><p>Drop bead z-stack TIFF(s): several stacks one file each; a multi-part stack movie.tif, movie_X1.tif, ...</p><p id="files" class="note">no TIFF selected</p></div>
      <div class="essentials">${num('f-dz', 'z step (um)', 0.02)}</div>
      <div class="buttons"><button id="run" class="primary">Fit PSF</button><button id="enough">Use what we have</button><button id="abort">Abort</button></div>
      <div class="progress"><div id="pbar"></div></div><p id="ptext" class="note"></p>
      <details class="adv" data-key="adv-stack"><summary>Advanced</summary>
        <div class="advgrid">
          <label>stage sign<select id="f-sign"><option value="1">+1</option><option value="-1">-1</option></select></label>
          ${num('f-ze', 'bead centre height (um)', 0.05, 'any', 'emitter height of the (point) bead above the coverslip')}
          ${num('f-roi', 'ROI (px)', 26, '1')}${num('f-npupil', 'pupil samples', 64, '1')}${num('f-os', 'oversampling', 2, '1')}
          ${num('f-iter', 'max iterations', 100, '1')}${num('f-minrel', 'bead threshold (rel.)', 0.3)}
          <label><input id="f-fitdz" type="checkbox" checked> fit z-step scale</label>
          <label><input id="f-drift" type="checkbox"> fit xy drift (per stack)</label>
          <label>held out<select id="f-holdout"><option value="last-bead">last bead</option><option value="last-stack">last stack</option><option value="none">none</option></select></label>
        </div>
      </details>
    </div>
  </details>
  <details class="panel" data-key="insitu" open id="p-insitu" hidden><summary><h2>Estimated PSF</h2></summary>
    <canvas id="i-zern" width="460" height="160"></canvas>
    <p class="note">Zernike coefficients (waves rms) by Noll index</p>
    <table id="i-table"></table>
    <div class="buttons"><button id="i-json">PSF json</button></div>
  </details>
  <details class="panel" data-key="results" open id="p-results" hidden><summary><h2>Bead fit result</h2></summary>
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
  </div>
</main>
<footer>Everything runs in your browser (WebGPU); no data leave the machine. Python reference: torch-psf.</footer>`;
