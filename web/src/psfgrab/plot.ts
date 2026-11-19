// Small canvas plots for PSF grab: the radially averaged log power spectrum per channel, and a value trace
// (sharpness over time or over frames).

export const CH_COLORS = ['#d1242f', '#1a7f37', '#2f6feb'];
const GRAY = '#1f2328', AXIS = '#8c959f', GRID = '#eef0f2', FONT = '10px Inter, system-ui, sans-serif';

function setup(cv: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const w = Math.max(50, cv.clientWidth), h = Math.max(50, cv.clientHeight);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const c = cv.getContext('2d')!;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.font = FONT;
  return { c, w, h };
}

function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target), p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

/** profiles: log10 ring means, index r = 0 .. n/2 (f = r / n cycles / px); r = 0 (the removed mean) is not drawn. */
export function drawSpectrum(cv: HTMLCanvasElement, profiles: Float32Array[] | null, n: number, label = '') {
  const { c, w, h } = setup(cv);
  const L = 34, R = 8, T = 8, B = 24, pw = w - L - R, ph = h - T - B;
  let lo = Infinity, hi = -Infinity;
  for (const p of profiles ?? []) for (let r = 1; r < p.length; r++) { if (p[r] < lo) lo = p[r]; if (p[r] > hi) hi = p[r]; }
  if (!isFinite(lo)) { lo = -2; hi = 4; }
  if (hi - lo < 1) { const m = (hi + lo) / 2; lo = m - 0.5; hi = m + 0.5; }
  lo = Math.floor(lo * 2) / 2; hi = Math.ceil(hi * 2) / 2;
  const X = (f: number) => L + (f / 0.5) * pw, Y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * ph;
  c.strokeStyle = GRID; c.lineWidth = 1; c.fillStyle = AXIS; c.textAlign = 'center'; c.textBaseline = 'top';
  for (let f = 0; f <= 0.5001; f += 0.1) { c.beginPath(); c.moveTo(X(f), T); c.lineTo(X(f), T + ph); c.stroke(); c.fillText(f.toFixed(1), X(f), T + ph + 3); }
  c.textAlign = 'right'; c.textBaseline = 'middle';
  const ys = niceStep(hi - lo, 4);
  for (let v = Math.ceil(lo / ys) * ys; v <= hi + 1e-9; v += ys) { c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(L + pw, Y(v)); c.stroke(); c.fillText(String(+v.toFixed(2)), L - 4, Y(v)); }
  c.textAlign = 'left'; c.textBaseline = 'bottom';
  c.fillText('cycles / px', L + pw - 52, h - 1);
  c.save(); c.translate(9, T + ph / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('log10 power', 0, 0); c.restore();
  if (label) { c.textAlign = 'right'; c.textBaseline = 'top'; c.fillText(label, L + pw - 2, T + 2); }
  if (!profiles) return;
  c.lineWidth = 1.5;
  profiles.forEach((p, k) => {
    c.strokeStyle = profiles.length === 1 ? GRAY : CH_COLORS[k % 3];
    c.beginPath();
    for (let r = 1; r < p.length; r++) { const x = X(r / n), y = Y(p[r]); if (r === 1) c.moveTo(x, y); else c.lineTo(x, y); }
    c.stroke();
  });
}

/** A trace of values (NaN = gap); cursor: highlighted index. */
export function drawTrace(cv: HTMLCanvasElement, vals: ArrayLike<number>, cursor = -1, xlabel = '') {
  const { c, w, h } = setup(cv);
  const L = 34, R = 8, T = 6, B = xlabel ? 18 : 6, pw = w - L - R, ph = h - T - B;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < vals.length; i++) { const v = vals[i]; if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  c.fillStyle = AXIS; c.strokeStyle = GRID;
  if (xlabel) { c.textAlign = 'right'; c.textBaseline = 'bottom'; c.fillText(xlabel, L + pw, h - 1); }
  if (!isFinite(lo)) return;
  if (hi - lo < 1e-12 * Math.max(1, Math.abs(hi))) { lo -= 0.5 * Math.abs(lo || 1); hi += 0.5 * Math.abs(hi || 1); }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const n = vals.length, X = (i: number) => L + (n > 1 ? i / (n - 1) : 0.5) * pw, Y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * ph;
  c.textAlign = 'right'; c.textBaseline = 'middle';
  const ys = niceStep(hi - lo, 3);
  for (let v = Math.ceil(lo / ys) * ys; v <= hi; v += ys) {
    c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(L + pw, Y(v)); c.stroke();
    c.fillText(Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0) : String(+v.toPrecision(3)), L - 4, Y(v));
  }
  if (cursor >= 0 && cursor < n) { c.strokeStyle = '#c7d7fb'; c.lineWidth = 2; c.beginPath(); c.moveTo(X(cursor), T); c.lineTo(X(cursor), T + ph); c.stroke(); }
  c.strokeStyle = '#2f6feb'; c.lineWidth = 1.5; c.beginPath();
  let pen = false;
  for (let i = 0; i < n; i++) {
    const v = vals[i];
    if (!isFinite(v)) { pen = false; continue; }
    if (pen) c.lineTo(X(i), Y(v)); else c.moveTo(X(i), Y(v));
    pen = true;
  }
  c.stroke();
  if (n <= 64) { c.fillStyle = '#2f6feb'; for (let i = 0; i < n; i++) if (isFinite(vals[i])) { c.beginPath(); c.arc(X(i), Y(vals[i]), 2, 0, 2 * Math.PI); c.fill(); } }
}
