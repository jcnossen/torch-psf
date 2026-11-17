// PSF grab: capture page for PSF estimation (camera recordings, focus sweeps, dropped TIFF / image / video files),
// with a live spectrum + sharpness display and a zip export (frames.tif + meta.json [+ still.jpg]).
// Everything runs in the browser; nothing is uploaded.
import '../ui/style.css';
import './psfgrab.css';
import { Analyzer } from './analyzer';
import { fftSizeFor, FrameAnalysis } from './spectrum';
import { drawSpectrum, drawTrace } from './plot';
import { Region, regionAt, grabDrawable } from './grab';
import { tiffStackBlob } from './tiffstack';
import { zipBlobs, BlobMember } from './zipblob';
import { openFiles, FileSource, nextFrameMeta } from './sources';
import {
  Rec, openCamera, listCameras, capsOf, settingsOf, frameSettings, SHOWN_KEYS, fmtCap, fmtNum, isRange, manualFocus,
  applySetting, hasImageCapture, takeStill, Still, plain,
} from './camera';

const VERSION = '0.1.0';
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const num = (id: string) => +$<HTMLInputElement>(id).value;

const el = {
  status: $('status'), viewCard: $('view-card'), anaCard: $('ana-card'), camCard: $('cam-card'), recCard: $('rec-card'), outCard: $('out-card'),
  viewer: $('viewer'), live: $<HTMLVideoElement>('live'), frameView: $<HTMLCanvasElement>('frame-view'), overlay: $<HTMLCanvasElement>('overlay'),
  viewinfo: $('viewinfo'), spec: $<HTMLCanvasElement>('spec'), trace: $<HTMLCanvasElement>('trace'), outTrace: $<HTMLCanvasElement>('out-trace'),
  sharp: $('sharp-val'), hf: $('hf-val'), anaMeta: $('ana-meta'), est: $('est'), progBox: $('prog-box'), progBar: $('prog-bar'), progMsg: $('prog-msg'),
  slider: $<HTMLInputElement>('frame-slider'), frameLabel: $('frame-label'),
};

const analyzer = new Analyzer();
type Mode = 'none' | 'camera' | 'file';
let mode: Mode = 'none';
let busy = false, stopFlag = false;
let cx = 0, cy = 0;                                    // crop centre (source pixels)
const liveHist: number[] = [];

function setStatus(msg: string, kind: '' | 'err' | 'ok' = '') { el.status.textContent = msg; el.status.dataset.kind = kind; }
function errMsg(e: unknown) { return String((e as Error)?.message ?? e); }

// ---------------------------------------------------------------- persisted per-viewer conveniences
const store = {
  get(k: string) { try { return localStorage.getItem('psfgrab.' + k); } catch { return null; } },
  set(k: string, v: string) { try { localStorage.setItem('psfgrab.' + k, v); } catch { /* private mode */ } },
};
for (const id of ['label', 'notes', 'crop', 'down', 'nframes', 'cam-res', 'cam-fps', 'ana-size', 'sw-steps', 'sw-m', 'sw-settle', 'still']) {
  const e = $<HTMLInputElement>(id), v = store.get(id);
  if (v !== null) { if (e.type === 'checkbox') e.checked = v === '1'; else e.value = v; }
  e.addEventListener('change', () => store.set(id, e.type === 'checkbox' ? (e.checked ? '1' : '0') : e.value));
}

// ---------------------------------------------------------------- geometry: source size, regions, overlay
function srcSize(): [number, number] {
  if (mode === 'camera') return [el.live.videoWidth, el.live.videoHeight];
  if (mode === 'file' && file) return [file.W, file.H];
  return [0, 0];
}
const cropSize = () => +$<HTMLSelectElement>('crop').value;
const downF = () => +$<HTMLSelectElement>('down').value;
const anaSize = () => +$<HTMLSelectElement>('ana-size').value;
function recRegion(): Region { const [W, H] = srcSize(); return regionAt(W, H, cx, cy, cropSize(), downF()); }
function anaRegion(): Region {
  const r = recRegion(), [W, H] = srcSize();
  return regionAt(W, H, r.x + r.w / 2, r.y + r.h / 2, fftSizeFor(r.w, r.h, anaSize()));   // the square that is analysed
}

function fitRect(): { s: number; ox: number; oy: number } {
  const [W, H] = srcSize(), cw = el.viewer.clientWidth, ch = el.viewer.clientHeight;
  const s = Math.min(cw / W, ch / H);
  return { s, ox: (cw - W * s) / 2, oy: (ch - H * s) / 2 };
}

function drawOverlay() {
  const [W, H] = srcSize();
  const cv = el.overlay, dpr = Math.min(window.devicePixelRatio || 1, 3);
  const cw = el.viewer.clientWidth, ch = el.viewer.clientHeight;
  cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
  const c = cv.getContext('2d')!;
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, cw, ch);
  if (!W || !H) return;
  const { s, ox, oy } = fitRect();
  const r = recRegion(), a = anaRegion();
  c.lineWidth = 2; c.strokeStyle = '#2f6feb'; c.strokeRect(ox + r.x * s, oy + r.y * s, r.w * s, r.h * s);
  c.setLineDash([5, 4]); c.lineWidth = 1.5; c.strokeStyle = 'rgba(255,255,255,.85)';
  c.strokeRect(ox + a.x * s, oy + a.y * s, a.w * s, a.h * s);
  c.setLineDash([]);
  updateEstimate();
}

function setViewerAspect() {
  const [W, H] = srcSize();
  if (W && H) el.viewer.style.aspectRatio = `${W} / ${H}`;
  drawOverlay();
}
new ResizeObserver(() => drawOverlay()).observe(el.viewer);
// a tap (click) moves the crop; dragging still scrolls the page on a phone
el.overlay.addEventListener('click', (e) => {
  const [W, H] = srcSize();
  if (!W || busy) return;
  const b = el.overlay.getBoundingClientRect(), { s, ox, oy } = fitRect();
  cx = Math.max(0, Math.min(W, (e.clientX - b.left - ox) / s));
  cy = Math.max(0, Math.min(H, (e.clientY - b.top - oy) / s));
  drawOverlay();
  if (mode === 'file') analyzeCurrentFile();
});
for (const id of ['crop', 'down', 'ana-size']) $(id).addEventListener('change', () => { drawOverlay(); if (mode === 'file') analyzeCurrentFile(); });
for (const id of ['nframes', 'f-from', 'f-to', 'f-step', 'sw-steps', 'sw-m']) $(id).addEventListener('input', updateEstimate);

function outCount(): number {
  if (mode === 'file') { const [a, b, s] = fileRange(); return a > b ? 0 : Math.floor((b - a) / s) + 1; }
  return Math.max(1, Math.floor(num('nframes')));
}
function updateEstimate() {
  const [W] = srcSize();
  if (!W) { el.est.textContent = ''; return; }
  const r = recRegion(), f = downF(), ow = Math.floor(r.w / f), oh = Math.floor(r.h / f);
  const C = mode === 'file' && file ? file.channels : 3, B = mode === 'file' && file && !file.u8 ? 4 : 1;
  const mb = (n: number) => n * ow * oh * C * B / 2 ** 20;
  let txt = `${ow} x ${oh} px x ${C === 3 ? 'RGB' : '1 channel'} ${B === 1 ? '8-bit' : 'float32'}: ${mb(outCount()).toFixed(0)} MB for ${outCount()} frames`;
  if (mode === 'camera' && sweepRange) txt += `; focus sweep ${Math.round(num('sw-steps')) * Math.round(num('sw-m'))} frames, ${mb(Math.round(num('sw-steps')) * Math.round(num('sw-m'))).toFixed(0)} MB`;
  el.est.textContent = txt + '. Kept in memory until downloaded (the zip is at most this size).';
  el.est.dataset.kind = mb(outCount()) > 600 ? 'warn' : '';
}

// ---------------------------------------------------------------- analysis display
function showAnalysis(a: FrameAnalysis & { ms?: number }, trace: number[] | null, cursor = -1) {
  drawSpectrum(el.spec, a.profiles, a.n, `${a.n} x ${a.n}`);
  el.sharp.textContent = fmtSharp(a.sharpness);
  el.hf.textContent = a.hfRatio.map((v) => v.toFixed(3)).join(', ');
  el.anaMeta.textContent = a.ms !== undefined ? `${a.ms.toFixed(0)} ms` : '';
  if (trace) drawTrace(el.trace, trace, cursor, mode === 'camera' ? 'last 120 analyses' : 'frame');
}
const fmtSharp = (v: number) => (v >= 100 || v < 0.01) && v !== 0 ? v.toExponential(3) : v.toPrecision(4);

// ================================================================ camera
let stream: MediaStream | null = null, track: MediaStreamTrack | null = null;
let caps: Rec = {}, sweepRange: ReturnType<typeof manualFocus> = null;
let wake: { release(): Promise<void> } | null = null;
let liveTimer = 0, lastAna = 0;

$('btn-camera').addEventListener('click', () => startCamera());
$('btn-cam-stop').addEventListener('click', () => stopCamera());
for (const id of ['cam-dev', 'cam-res', 'cam-fps']) $(id).addEventListener('change', () => startCamera(true));

async function startCamera(keepDevice = false) {
  if (busy) return;
  const dev = keepDevice ? $<HTMLSelectElement>('cam-dev').value : '';
  closeFile(); stopCamera(false);
  const [w, h] = $<HTMLSelectElement>('cam-res').value.split('x').map(Number);
  setStatus('opening the camera ...');
  try {
    stream = await openCamera({ deviceId: dev || undefined, width: w, height: h, frameRate: +$<HTMLSelectElement>('cam-fps').value });
  } catch (e) {
    const name = (e as DOMException)?.name;
    setStatus(name === 'NotAllowedError' ? 'Camera permission was denied. Allow the camera for this site in the browser settings and try again.'
      : name === 'NotFoundError' ? 'No camera found.' : `Could not open the camera: ${errMsg(e)}`, 'err');
    return;
  }
  track = stream.getVideoTracks()[0];
  mode = 'camera';
  el.live.srcObject = stream;
  el.live.hidden = false; el.frameView.hidden = true;
  try { await el.live.play(); } catch { /* autoplay: muted + playsinline */ }
  if (!el.live.videoWidth) await new Promise((r) => el.live.addEventListener('loadedmetadata', r, { once: true }));
  cx = el.live.videoWidth / 2; cy = el.live.videoHeight / 2;
  track.addEventListener('ended', () => { if (mode === 'camera') { setStatus('The camera stopped.', 'err'); stopCamera(); } });
  for (const c of [el.viewCard, el.anaCard, el.camCard, el.recCard]) c.hidden = false;
  $('file-nav').hidden = true; $('cam-n').hidden = false; $('file-range').hidden = true; $('ana-all-row').hidden = true;
  $('rec-title').textContent = 'Record';
  $('btn-rec').textContent = 'Record';
  liveHist.length = 0;
  setViewerAspect();
  await fillDevices();
  refreshCaps(true);
  // some Android Chrome versions report the image-capture capabilities (focus, zoom, torch) only once frames flow
  setTimeout(() => { if (track === stream?.getVideoTracks()[0] && !busy) refreshCaps(true); }, 900);
  try { wake = await (navigator as unknown as { wakeLock?: { request(t: string): Promise<{ release(): Promise<void> }> } }).wakeLock?.request('screen') ?? null; } catch { wake = null; }
  setStatus(`Camera: ${track.label || 'camera'}. Tap the image to move the crop.`, 'ok');
  liveLoop();
  clearInterval(liveTimer);
  liveTimer = window.setInterval(() => { if (mode === 'camera' && !busy) refreshCaps(false); }, 1000);
}

function stopCamera(updateUi = true) {
  clearInterval(liveTimer);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null; track = null;
  el.live.srcObject = null;
  wake?.release().catch(() => { /* ignore */ }); wake = null;
  if (mode === 'camera') mode = 'none';
  if (updateUi) { for (const c of [el.viewCard, el.anaCard, el.camCard, el.recCard]) c.hidden = true; setStatus(''); }
}

el.live.addEventListener('resize', () => { if (mode === 'camera') { const [W, H] = srcSize(); cx = Math.min(cx, W); cy = Math.min(cy, H); setViewerAspect(); } });

/** Waits for the next camera / video frame: requestVideoFrameCallback metadata, or an animation frame (null). */
async function waitFrame(v: HTMLVideoElement): Promise<Rec | null> {
  const m = await nextFrameMeta(v, 2000);
  if (m) return m as unknown as Rec;
  if (!(v as unknown as { requestVideoFrameCallback?: unknown }).requestVideoFrameCallback) await new Promise((r) => requestAnimationFrame(r));
  return null;
}

async function liveLoop() {
  const s = stream;
  while (stream === s && s) {
    await waitFrame(el.live);
    if (stream !== s) break;
    const t = performance.now();
    if (busy || analyzer.busy || t - lastAna < 120 || !el.live.videoWidth) continue;
    lastAna = t;
    const a = anaRegion(), p = grabDrawable(el.live, a, a.w, a.h);
    analyzer.run(p.planes, p.w, p.h, anaSize(), true).then((res) => {
      if (mode !== 'camera') return;
      liveHist.push(res.sharpness); if (liveHist.length > 120) liveHist.shift();
      showAnalysis(res, liveHist);
    }).catch((e) => setStatus(`analysis: ${errMsg(e)}`, 'err'));
  }
}

async function fillDevices() {
  const sel = $<HTMLSelectElement>('cam-dev'), cur = track ? String(settingsOf(track).deviceId ?? '') : '';
  const devs = await listCameras();
  sel.innerHTML = '';
  devs.forEach((d, i) => { const o = document.createElement('option'); o.value = d.deviceId; o.textContent = d.label || `camera ${i + 1}`; sel.appendChild(o); });
  if (cur) sel.value = cur;
}

const CONTROL_KEYS = ['focusMode', 'focusDistance', 'exposureMode', 'exposureTime', 'exposureCompensation', 'iso', 'whiteBalanceMode', 'colorTemperature', 'zoom', 'torch'];

/** The capabilities table (+ the controls on a full refresh). */
function refreshCaps(full: boolean) {
  if (!track) return;
  const st = settingsOf(track);
  if (full) {
    caps = capsOf(track);
    const hasCaps = Object.keys(caps).length > 0;
    $('caps-note').textContent = hasCaps ? 'Supported range or options, and the current value.'
      : 'This browser does not report track capabilities (MediaStreamTrack.getCapabilities); only the current settings are shown.';
    sweepRange = manualFocus(caps);
    $('sweep').hidden = !sweepRange;
    if (sweepRange) {
      const f = $<HTMLInputElement>('sw-from'), t = $<HTMLInputElement>('sw-to');
      f.value = String(sweepRange.min); t.value = String(sweepRange.max);
      f.min = t.min = String(sweepRange.min); f.max = t.max = String(sweepRange.max);
      if (sweepRange.step) f.step = t.step = String(sweepRange.step);
    }
    $('still-row').hidden = !hasImageCapture();
    buildControls(st);
    updateEstimate();
  }
  const keys = [...SHOWN_KEYS.filter((k) => k in caps || k in st), ...Object.keys(caps).filter((k) => !SHOWN_KEYS.includes(k) && k !== 'deviceId' && k !== 'groupId')];
  const rows = keys.map((k) => `<tr><td>${k}</td><td>${esc(fmtCap(caps[k]))}</td><td>${esc(fmtCap(st[k]))}</td></tr>`);
  rows.push(`<tr><td>ImageCapture</td><td>${hasImageCapture() ? 'yes (full-resolution still)' : 'no'}</td><td></td></tr>`);
  rows.push(`<tr><td>focus sweep</td><td>${sweepRange ? 'yes (manual focusDistance)' : 'no (needs focusMode manual + focusDistance)'}</td><td></td></tr>`);
  $('caps').innerHTML = `<tr><th>setting</th><th>supported</th><th>current</th></tr>${rows.join('')}`;
  $('cam-meta').textContent = `${st.width ?? '?'} x ${st.height ?? '?'} @ ${typeof st.frameRate === 'number' ? fmtNum(st.frameRate) : '?'} fps`;
  el.viewinfo.textContent = `${el.live.videoWidth} x ${el.live.videoHeight} px. Tap the image to move the crop (blue: recorded, dashed: analysed).`;
  for (const k of CONTROL_KEYS) {
    const inp = document.querySelector<HTMLInputElement | HTMLSelectElement>(`[data-ctl="${k}"]`);
    if (!inp || document.activeElement === inp || st[k] === undefined || Date.now() - +(inp.dataset.t ?? 0) < 2000) continue;
    if (inp instanceof HTMLInputElement && inp.type === 'checkbox') inp.checked = !!st[k]; else inp.value = String(st[k]);
    const lab = document.querySelector(`[data-val="${k}"]`); if (lab) lab.textContent = fmtCap(st[k]);
  }
}
const esc = (s: string) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]!));

function buildControls(st: Rec) {
  const box = $('controls');
  box.innerHTML = '';
  for (const k of CONTROL_KEYS) {
    const cap = caps[k];
    if (cap === undefined) continue;
    const row = document.createElement('div'); row.className = 'ctl';
    let inp: HTMLInputElement | HTMLSelectElement;
    if (k === 'torch') {
      const lab = document.createElement('label'); lab.className = 'chk';
      inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = !!st.torch;
      lab.append(inp, document.createTextNode(' torch (flash light)')); row.appendChild(lab);
    } else if (Array.isArray(cap)) {
      if (cap.length < 2) continue;
      row.classList.add('field');
      const s = document.createElement('select');
      for (const v of cap) { const o = document.createElement('option'); o.value = o.textContent = String(v); s.appendChild(o); }
      if (st[k] !== undefined) s.value = String(st[k]);
      const l = document.createElement('span'); l.className = 'fl'; l.textContent = k;
      row.append(l, s); inp = s;
    } else if (isRange(cap) && cap.max > cap.min) {
      const head = document.createElement('div'); head.className = 'ctlhead';
      head.innerHTML = `<span>${k}</span><span class="mono" data-val="${k}">${fmtCap(st[k])}</span>`;
      const r = document.createElement('input'); r.type = 'range';
      r.min = String(cap.min); r.max = String(cap.max); r.step = String(cap.step || (cap.max - cap.min) / 200);
      if (typeof st[k] === 'number') r.value = String(st[k]);
      row.append(head, r); inp = r;
    } else continue;
    inp.dataset.ctl = k;
    let pending = 0;
    const apply = async () => {
      if (!track) return;
      const v = inp instanceof HTMLInputElement && inp.type === 'checkbox' ? inp.checked : inp instanceof HTMLInputElement && inp.type === 'range' ? +inp.value : inp.value;
      const lab = document.querySelector(`[data-val="${k}"]`); if (lab && typeof v === 'number') lab.textContent = fmtNum(v);
      try { await applySetting(track, k, v); } catch (e) { setStatus(`${k}: ${errMsg(e)}`, 'err'); }
      refreshCaps(false);
    };
    inp.addEventListener('input', () => { inp.dataset.t = String(Date.now()); clearTimeout(pending); pending = window.setTimeout(apply, 80); });
    inp.addEventListener('change', apply);
    box.appendChild(row);
  }
}

// ================================================================ files
let file: FileSource | null = null;
let fileCur = 0, fileTrace: number[] = [], fileToken = 0;

$('btn-open').addEventListener('click', () => $<HTMLInputElement>('file').click());
$<HTMLInputElement>('file').addEventListener('change', (e) => {
  const fs = [...((e.target as HTMLInputElement).files ?? [])];
  (e.target as HTMLInputElement).value = '';
  if (fs.length) openFile(fs);
});
const drop = $('drop');
for (const t of [drop, document.body]) {
  t.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  t.addEventListener('dragleave', () => drop.classList.remove('over'));
  t.addEventListener('drop', (e) => {
    e.preventDefault(); drop.classList.remove('over');
    const fs = [...((e as DragEvent).dataTransfer?.files ?? [])];
    if (fs.length) openFile(fs);
  });
}

function closeFile() {
  file?.close(); file = null;
  if (mode === 'file') mode = 'none';
}

async function openFile(fs: File[]) {
  if (busy) return;
  stopCamera(false); closeFile();
  setStatus(`opening ${fs[0].name} ...`);
  try {
    file = await openFiles(fs, (m) => setStatus(m));
  } catch (e) { setStatus(errMsg(e), 'err'); return; }
  mode = 'file';
  const f = file;
  cx = f.W / 2; cy = f.H / 2;
  el.frameView.width = f.W; el.frameView.height = f.H;
  el.live.hidden = true; el.frameView.hidden = false;
  for (const c of [el.viewCard, el.anaCard, el.recCard]) c.hidden = false;
  el.camCard.hidden = true; $('sweep').hidden = true; $('still-row').hidden = true;
  $('file-nav').hidden = false; $('cam-n').hidden = true; $('file-range').hidden = false; $('ana-all-row').hidden = false;
  $('fps-field').hidden = f.kind !== 'video';
  if (f.kind === 'video') $<HTMLInputElement>('fps').value = String((f as unknown as { fps: number }).fps);
  $('rec-title').textContent = 'Export frames';
  $('btn-rec').textContent = 'Export as TIFF + JSON';
  setFileFrames();
  setViewerAspect();
  setStatus(`${f.name}: ${f.W} x ${f.H} px, ${f.n} frames, ${f.channels === 3 ? 'RGB' : 'grayscale'}${f.u8 ? ' 8-bit' : ''} (${f.kind}).`, 'ok');
  await showFileFrame(0);
}

function setFileFrames() {
  const f = file!;
  el.slider.max = String(f.n - 1); el.slider.value = '0';
  $<HTMLInputElement>('f-from').value = '0'; $<HTMLInputElement>('f-to').value = String(f.n - 1); $<HTMLInputElement>('f-step').value = '1';
  $<HTMLInputElement>('f-from').max = $<HTMLInputElement>('f-to').max = String(f.n - 1);
  fileTrace = new Array(f.n).fill(NaN);
  el.viewinfo.textContent = `${f.W} x ${f.H} px. Tap the image to move the crop (blue: exported, dashed: analysed).`;
  updateEstimate();
}

$<HTMLInputElement>('fps').addEventListener('change', () => {
  const f = file as unknown as { setFps?: (x: number) => void } | null, v = num('fps');
  if (f?.setFps && v > 0) { f.setFps(v); setFileFrames(); showFileFrame(0); }
});

function fileRange(): [number, number, number] {
  const n = file?.n ?? 0, cl = (v: number) => Math.max(0, Math.min(n - 1, Math.floor(v)));
  return [cl(num('f-from')), cl(num('f-to')), Math.max(1, Math.floor(num('f-step')))];
}

el.slider.addEventListener('input', () => showFileFrame(+el.slider.value));

async function showFileFrame(i: number) {
  if (!file || busy) return;
  const tok = ++fileToken, f = file;
  try {
    await f.load(i);
    if (tok !== fileToken || f !== file) return;
    fileCur = i;
    f.draw(el.frameView.getContext('2d')!);
    el.frameLabel.textContent = `${i + 1} / ${f.n}`;
    await analyzeCurrentFile();
  } catch (e) { setStatus(errMsg(e), 'err'); }
}

async function analyzeCurrentFile() {
  if (!file) return;
  const a = anaRegion(), p = file.grab(a, 1);
  try {
    const res = await analyzer.run(p.planes, p.w, p.h, anaSize(), true);
    fileTrace[fileCur] = res.sharpness;
    showAnalysis(res, fileTrace, fileCur);
  } catch (e) { setStatus(`analysis: ${errMsg(e)}`, 'err'); }
}

$('btn-ana-all').addEventListener('click', async () => {
  if (!file || busy) return;
  const f = file, [a, b, s] = fileRange();
  await runBusy(async () => {
    for (let i = a; i <= b && !stopFlag; i += s) {
      await f.load(i);
      const r = anaRegion(), p = f.grab(r, 1);
      const res = await analyzer.run(p.planes, p.w, p.h, anaSize(), true);
      fileTrace[i] = res.sharpness;
      progress((i - a) / Math.max(1, b - a), `analysed frame ${i + 1} of ${f.n}`);
      if (i % 4 === 0 || i + s > b) { showAnalysis(res, fileTrace, i); }
    }
    await f.load(fileCur);
  });
});

// ================================================================ recording
interface Recording {
  mode: 'frames' | 'focus_sweep' | 'file';
  planes: (Uint8Array | Float32Array)[];  // page t * C + c
  N: number; C: number; w: number; h: number;
  region: Region; downscale: number;
  frames: Rec[];
  source: Rec;
  sweep?: Rec;
  still?: Still | null;
  stillError?: string;
  created: string; t0: number;
  analysis?: Promise<void>;
  source_W?: number; source_H?: number;    // camera image size at the start (a change aborts the recording)
}
let rec: Recording | null = null;

function progress(frac: number, msg: string) {
  el.progBar.style.width = `${Math.round(100 * Math.max(0, Math.min(1, frac)))}%`;
  el.progMsg.textContent = msg;
}

async function runBusy(f: () => Promise<void>) {
  busy = true; stopFlag = false;
  document.body.classList.add('busy');
  el.progBox.hidden = false; progress(0, '');
  try { await f(); } catch (e) { setStatus(errMsg(e), 'err'); }
  finally { busy = false; document.body.classList.remove('busy'); el.progBox.hidden = true; }
}
$('btn-stop').addEventListener('click', () => { stopFlag = true; });

$('btn-rec').addEventListener('click', () => { if (mode === 'camera') recordCamera(); else if (mode === 'file') exportFile(); });
$('btn-sweep').addEventListener('click', () => focusSweep());

function newRecording(m: Recording['mode'], C: number): Recording {
  const r = recRegion(), f = downF();
  // drop the previous recording (and its zip) first (memory)
  rec = null; lastZip = null; el.outCard.hidden = true;
  return { mode: m, planes: [], N: 0, C, w: Math.floor(r.w / f), h: Math.floor(r.h / f), region: r, downscale: f, frames: [], source: {},
    created: new Date().toISOString(), t0: performance.now() };
}

function cameraSource(): Rec {
  const t = track!;
  return { kind: 'camera', label: t.label, settings: plain(settingsOf(t)), capabilities: plain(caps), constraints: plain(t.getConstraints()),
    video_element: { videoWidth: el.live.videoWidth, videoHeight: el.live.videoHeight } };
}

/** Grab the record region of the current camera frame into r (with its metadata). */
function grabCamera(r: Recording, meta: Rec | null, extra: Rec = {}) {
  const p = grabDrawable(el.live, r.region, r.w, r.h);
  r.planes.push(...p.planes);
  const m: Rec = { index: r.N, t_ms: +(performance.now() - r.t0).toFixed(2) };
  if (meta) for (const k of ['mediaTime', 'presentedFrames', 'captureTime', 'expectedDisplayTime', 'presentationTime', 'processingDuration', 'rtpTimestamp']) if (meta[k] !== undefined) m[k] = meta[k];
  m.settings = plain(frameSettings(track!));
  r.frames.push({ ...m, ...extra });
  r.N++;
}

function checkSize(r: Recording) {
  if (el.live.videoWidth !== r.source_W || el.live.videoHeight !== r.source_H) throw new Error('the camera image size changed during the recording (device rotated?); recording stopped');
}

async function recordCamera() {
  if (!track || busy) return;
  const N = outCount();
  await runBusy(async () => {
    const r = newRecording('frames', 3);
    r.source = cameraSource(); r.source_W = el.live.videoWidth; r.source_H = el.live.videoHeight;
    for (let i = 0; i < N && !stopFlag; i++) {
      const m = await waitFrame(el.live);
      checkSize(r);
      grabCamera(r, m);
      if (i % 3 === 0 || i === N - 1) progress((i + 1) / N, `frame ${i + 1} / ${N}`);
    }
    if (!r.N) return;
    await maybeStill(r);
    finish(r);
  });
}

async function maybeStill(r: Recording) {
  if (!$<HTMLInputElement>('still').checked || !hasImageCapture() || !track) return;
  progress(1, 'taking the still photo ...');
  try { r.still = await takeStill(track); } catch (e) { r.still = null; r.stillError = errMsg(e); }
}

async function focusSweep() {
  if (!track || busy || !sweepRange) return;
  const K = Math.max(2, Math.round(num('sw-steps'))), M = Math.max(1, Math.round(num('sw-m'))), settle = Math.max(0, num('sw-settle'));
  const from = num('sw-from'), to = num('sw-to');
  if (!isFinite(from) || !isFinite(to)) { setStatus('focus sweep: enter the focus distance range', 'err'); return; }
  const range = sweepRange, tol = Math.max(range.step ?? 0, Math.abs(range.max - range.min) * 1e-3);
  const t = track, before = settingsOf(t);
  await runBusy(async () => {
    const r = newRecording('focus_sweep', 3);
    r.source = cameraSource(); r.source_W = el.live.videoWidth; r.source_H = el.live.videoHeight;
    r.sweep = { steps: K, frames_per_step: M, from, to, settle_ms: settle, tolerance: tol, focusDistance_capability: plain(range),
      settings_before: plain(before), averaged: false };
    try {
      for (let k = 0; k < K && !stopFlag; k++) {
        const target = from + (to - from) * k / (K - 1);
        const t0 = performance.now();
        await t.applyConstraints({ advanced: [{ focusMode: 'manual', focusDistance: target } as unknown as MediaTrackConstraintSet] });
        let settled = false, waited = 0;
        for (;;) {
          await waitFrame(el.live);
          waited = performance.now() - t0;
          const fd = settingsOf(t).focusDistance;
          if (waited >= settle && (typeof fd !== 'number' || Math.abs(fd - target) <= tol)) { settled = true; break; }
          if (waited > settle + 3000 || stopFlag) break;
        }
        await waitFrame(el.live); await waitFrame(el.live);       // frames already in the pipeline at the old focus
        for (let m = 0; m < M && !stopFlag; m++) {
          const meta = await waitFrame(el.live);
          checkSize(r);
          grabCamera(r, meta, { step: k, step_frame: m, target_focusDistance: target, settled, settle_ms: +waited.toFixed(1) });
        }
        progress((k + 1) / K, `step ${k + 1} / ${K}: focusDistance ${fmtNum(target)} (reported ${fmtCap(settingsOf(t).focusDistance)})${settled ? '' : ', not settled'}`);
      }
    } finally {
      try {
        if (before.focusMode && before.focusMode !== 'manual') await t.applyConstraints({ advanced: [{ focusMode: before.focusMode } as unknown as MediaTrackConstraintSet] });
        else if (typeof before.focusDistance === 'number') await applySetting(t, 'focusDistance', before.focusDistance);
      } catch { /* best effort */ }
      refreshCaps(false);
    }
    if (r.N) { await maybeStill(r); finish(r); }
  });
}

async function exportFile() {
  if (!file || busy) return;
  const f = file, [a, b, s] = fileRange();
  await runBusy(async () => {
    const r = newRecording('file', f.channels);
    r.source = { kind: f.kind, name: f.name, ...f.info, width: f.W, height: f.H, frames: f.n };
    const total = Math.floor((b - a) / s) + 1;
    for (let i = a; i <= b && !stopFlag; i += s) {
      const fm = await f.load(i);
      const p = f.grab(r.region, r.downscale);
      r.planes.push(...p.planes);
      r.frames.push({ ...fm, index: r.N, source_frame: i, t_ms: +(performance.now() - r.t0).toFixed(2) });
      r.N++;
      if (r.N % 4 === 0 || i + s > b) progress(r.N / total, `frame ${i + 1} (${r.N} / ${total})`);
    }
    await f.load(fileCur);
    if (r.N) finish(r);
  });
}

/** Show the recording, then analyse each frame (sharpness, spectrum summary) for the JSON and the trace. */
function finish(r: Recording) {
  rec = r;
  el.outCard.hidden = false;
  const mb = r.planes.reduce((s, p) => s + p.byteLength, 0) / 2 ** 20;
  const dt = r.frames.length > 1 ? (+(r.frames[r.frames.length - 1].t_ms as number) - +(r.frames[0].t_ms as number)) / (r.frames.length - 1) : 0;
  let dropped = 0;
  for (let i = 1; i < r.frames.length; i++) {
    const a = r.frames[i - 1].presentedFrames, b = r.frames[i].presentedFrames;
    if (typeof a === 'number' && typeof b === 'number' && r.mode === 'frames') dropped += Math.max(0, b - a - 1);
  }
  $('out-meta').textContent = `${r.N} frames, ${r.w} x ${r.h}, ${mb.toFixed(0)} MB${dt ? `, ${(1000 / dt).toFixed(1)} fps` : ''}${dropped ? `, ${dropped} camera frames skipped` : ''}`;
  $('out-hint').textContent = `frames.tif: ${r.N} x ${r.C} pages of ${r.w} x ${r.h} ${r.planes[0] instanceof Float32Array ? 'float32' : 'uint8'}` +
    `${r.C === 3 ? ' (R, G, B)' : ''}; meta.json${r.still ? `; still.jpg (${(r.still.blob.size / 2 ** 20).toFixed(1)} MB)` : r.stillError ? `; still failed: ${r.stillError}` : ''}. A new recording replaces this one.`;
  setStatus(`Recorded ${r.N} frames.`, 'ok');
  const sharp = new Array(r.N).fill(NaN);
  drawTrace(el.outTrace, sharp, -1, 'sharpness per frame');
  r.analysis = (async () => {
    for (let i = 0; i < r.N; i++) {
      if (rec !== r) return;
      const pl = r.planes.slice(i * r.C, (i + 1) * r.C);
      try {
        const a = await analyzer.run(pl, r.w, r.h, 512, false);
        r.frames[i].sharpness = a.sharpness; r.frames[i].hfRatio = a.hfRatio; r.frames[i].mean = a.mean;
        if (i === 0) r.frames[i].analysis_square = { n: a.n, x0: a.x0, y0: a.y0 };
        sharp[i] = a.sharpness;
      } catch { /* leave NaN */ }
      if (i % 4 === 0 || i === r.N - 1) drawTrace(el.outTrace, sharp, -1, 'sharpness per frame');
    }
  })();
}

// ---------------------------------------------------------------- export
async function deviceInfo(): Promise<Rec> {
  const n = navigator as unknown as Rec & { userAgentData?: { brands: unknown; mobile: boolean; platform: string; getHighEntropyValues?(h: string[]): Promise<Rec> } };
  const o: Rec = {
    userAgent: navigator.userAgent, platform: navigator.platform, language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency, deviceMemory_GB: n.deviceMemory ?? null, maxTouchPoints: navigator.maxTouchPoints,
    devicePixelRatio: window.devicePixelRatio, screen: { width: screen.width, height: screen.height, orientation: screen.orientation?.type ?? null },
  };
  if (n.userAgentData) {
    o.userAgentData = { brands: plain(n.userAgentData.brands), mobile: n.userAgentData.mobile, platform: n.userAgentData.platform };
    try { o.userAgentData_high = plain(await n.userAgentData.getHighEntropyValues?.(['model', 'platformVersion', 'fullVersionList', 'architecture'])); } catch { /* optional */ }
  }
  return o;
}

function stamp(d: Date) {
  const p = (x: number) => String(x).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function buildZip(r: Recording): Promise<{ blob: Blob; name: string }> {
  progress(0, 'analysing the frames ...');
  await r.analysis;
  const tif = tiffStackBlob(r.planes, r.N, r.C, r.h, r.w, { axis: r.mode === 'focus_sweep' ? 'slices' : 'frames' });
  const label = ($<HTMLInputElement>('label').value || '').trim().replace(/[^\w.-]+/g, '_').slice(0, 60);
  const created = new Date(r.created);
  const meta: Rec = {
    format: 'psfgrab/1',
    app: { name: 'PSF grab', version: VERSION, url: location.href.split('#')[0] },
    created: r.created,
    timezone_offset_min: -created.getTimezoneOffset(),
    label, notes: $<HTMLTextAreaElement>('notes').value,
    device: await deviceInfo(),
    source: r.source,
    recording: {
      mode: r.mode, frames: r.N, channels: r.C === 3 ? ['R', 'G', 'B'] : ['gray'],
      width: r.w, height: r.h, dtype: r.planes[0] instanceof Float32Array ? 'float32' : 'uint8',
      region: r.region, downscale: r.downscale,
      region_note: 'region in source pixels (x, y top-left, w, h); downscale > 1 is a smoothed canvas resize (camera, video, images) or a block mean (TIFF)',
      tiff: { file: 'frames.tif', pages: r.N * r.C, page_order: 'frame * C + channel', tifffile_shape: r.C > 1 ? [r.N, r.C, r.h, r.w] : [r.N, r.h, r.w],
        imagej_axis: r.mode === 'focus_sweep' ? 'slices' : 'frames' },
      ...(r.sweep ? { focus_sweep: r.sweep } : {}),
    },
    analysis: {
      note: 'per frame: sharpness = var(4-neighbour Laplacian) / mean^2 of the channel mean; hfRatio per channel = share of the ' +
        'non-DC power above half the Nyquist frequency (Hann window, centred square of up to 512 px of the recorded frame); mean per channel',
    },
    still: r.still ? { file: 'still.jpg', bytes: r.still.blob.size, type: r.still.blob.type, requested: r.still.requested,
      photoCapabilities: r.still.photoCapabilities, photoSettings: r.still.photoSettings, ms: +r.still.ms.toFixed(1) } : r.stillError ? { error: r.stillError } : null,
    frames: r.frames,
  };
  const members: BlobMember[] = [
    { name: 'frames.tif', data: tif },
    { name: 'meta.json', data: new Blob([JSON.stringify(meta, null, 1)], { type: 'application/json' }) },
  ];
  if (r.still) members.push({ name: r.still.blob.type === 'image/png' ? 'still.png' : 'still.jpg', data: r.still.blob, compress: false });
  if (r.still && r.still.blob.type === 'image/png') (meta.still as Rec).file = 'still.png';
  const blob = await zipBlobs(members, (m) => progress(0.5, m));
  return { blob, name: `psfgrab_${label ? label + '_' : ''}${stamp(created)}.zip` };
}

let lastZip: { blob: Blob; name: string; rec: Recording } | null = null;
async function zipOf(r: Recording) {
  if (lastZip?.rec === r) return lastZip;
  const z = await buildZip(r);
  lastZip = { ...z, rec: r };
  return lastZip;
}

$('btn-zip').addEventListener('click', async () => {
  const r = rec;
  if (!r || busy) return;
  await runBusy(async () => {
    const z = await zipOf(r);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(z.blob); a.download = z.name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    setStatus(`${z.name} (${(z.blob.size / 2 ** 20).toFixed(1)} MB) is downloading.`, 'ok');
  });
});

const canShareFiles = (() => {
  try { return typeof navigator.canShare === 'function' && navigator.canShare({ files: [new File(['x'], 'x.zip', { type: 'application/zip' })] }); } catch { return false; }
})();
$('btn-share').hidden = !canShareFiles;
$('btn-share').addEventListener('click', async () => {
  const r = rec;
  if (!r || busy) return;
  let z: { blob: Blob; name: string } | null = null;
  await runBusy(async () => { z = await zipOf(r); });
  if (!z) return;
  const zz = z as { blob: Blob; name: string };
  try { await navigator.share({ files: [new File([zz.blob], zz.name, { type: 'application/zip' })], title: zz.name }); }
  catch (e) {
    const name = (e as DOMException)?.name;
    // building the zip can outlast the tap's user activation: the zip is kept, a second tap shares at once
    if (name === 'NotAllowedError') setStatus('The zip is ready: tap Share again.', 'ok');
    else if (name !== 'AbortError') setStatus(`share: ${errMsg(e)}`, 'err');
  }
});

// ---------------------------------------------------------------- start
drawSpectrum(el.spec, null, 512);
if (!window.isSecureContext) setStatus('This page is not served over HTTPS: the camera is not available (files still work).', 'err');
else if (!navigator.mediaDevices?.getUserMedia) setStatus('This browser has no camera API; files still work.', 'err');
