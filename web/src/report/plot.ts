// A small plotting layer on top of io/pdf.ts: axes with ticks and labels, line plots, grouped bar charts with error
// bars, histograms, images with a colour map and colour bar, key / value tables and wrapped text. All sizes in points,
// rectangles with the origin at the top left of the page (as PdfPage).
import { PdfDoc, PdfPage, Color, FontName, StrokeStyle, textWidth } from '../io/pdf';
import { viridis, diverging } from '../ui/colormap';

export interface Rect { x: number; y: number; w: number; h: number }
export type Cmap = 'viridis' | 'diverging' | 'gray';

export const INK = '#222', MUTED = '#666', GRID = '#e3e3e3', FRAME = '#888';
/** Categorical series colours (blue, orange, green, red, purple). */
export const PALETTE = ['#3b6fb6', '#e08a2c', '#3a9a5b', '#c8453b', '#7a5bb0'];

// ---------------------------------------------------------------- ticks and numbers
/** About `target` round tick values covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
/** Tick label: as many decimals as the tick step needs; exponent notation for very large / small values. */
export function fmtTick(v: number, step: number): string {
  const a = Math.max(Math.abs(v), Math.abs(step));
  if (v !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(Math.max(0, Math.min(3, Math.round(-Math.log10(step / a) + 0.5) - 1))).replace('e+', 'e');
  return v.toFixed(Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9))));
}
/** A short number for tables and labels (3 significant digits, at least the integer part). */
export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return 'n/a';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(digits - 1).replace('e+', 'e');
  return a >= 10 ** digits ? v.toFixed(0) : String(+v.toPrecision(digits));
}
function finiteRange(arrs: ArrayLike<number>[]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (let i = 0; i < a.length; i++) { const v = a[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  return lo <= hi ? [lo, hi] : [0, 1];
}
/** [lo, hi] padded by a fraction of the range (a zero-width range is widened). */
export function padRange([lo, hi]: [number, number], f = 0.05): [number, number] {
  if (hi <= lo) { const d = Math.abs(lo) * 0.1 || 1; return [lo - d, hi + d]; }
  const d = (hi - lo) * f;
  return [lo - d, hi + d];
}

// ---------------------------------------------------------------- text helpers
/** Greedy word wrap to a width in points. */
export function wrapText(s: string, width: number, size: number, font: FontName = 'regular'): string[] {
  const out: string[] = [];
  for (const para of s.split('\n')) {
    let line = '';
    for (const word of para.split(/\s+/).filter(Boolean)) {
      const t = line ? line + ' ' + word : word;
      if (textWidth(t, size, font) <= width || !line) line = t;
      else { out.push(line); line = word; }
    }
    out.push(line);
  }
  return out;
}
/** Shorten a string with an ellipsis to fit a width. */
export function fitText(s: string, width: number, size: number, font: FontName = 'regular'): string {
  if (textWidth(s, size, font) <= width) return s;
  while (s.length > 1 && textWidth(s + '…', size, font) > width) s = s.slice(0, -1);
  return s + '…';
}

/** Paragraph(s) of wrapped text from (x, y) (top of the first line); returns the height used. */
export function paragraph(pg: PdfPage, x: number, y: number, w: number, s: string, o: { size?: number; font?: FontName; color?: Color; lead?: number } = {}): number {
  const size = o.size ?? 8.5, lead = o.lead ?? size * 1.3;
  const lines = wrapText(s, w, size, o.font);
  lines.forEach((l, i) => pg.text(x, y + size + i * lead, l, { size, font: o.font, color: o.color ?? INK }));
  return lines.length * lead + (lead - size) / 2;
}

/** Key / value table; values wrap. Returns the height used. */
export function table(pg: PdfPage, r: { x: number; y: number; w: number }, rows: [string, string][], o: { size?: number; keyFrac?: number; title?: string } = {}): number {
  const size = o.size ?? 8, lead = size * 1.28, keyW = r.w * (o.keyFrac ?? 0.38), pad = 3;
  let y = r.y;
  if (o.title) { pg.text(r.x, y + size + 1, o.title, { size: size + 1, font: 'bold', color: INK }); y += lead + 4; }
  rows.forEach(([k, v], i) => {
    const kl = wrapText(k, keyW - 2 * pad, size, 'regular'), vl = wrapText(v, r.w - keyW - 2 * pad, size);
    const h = Math.max(kl.length, vl.length) * lead + 2 * pad - 1;
    if (i % 2 === 0) pg.rect(r.x, y, r.w, h, { fill: '#f3f4f6' });
    kl.forEach((l, j) => pg.text(r.x + pad, y + pad + size + j * lead - 1, l, { size, color: MUTED }));
    vl.forEach((l, j) => pg.text(r.x + keyW + pad, y + pad + size + j * lead - 1, l, { size, color: INK }));
    y += h;
  });
  return y - r.y;
}
/** Height table() would use (for layout before drawing). */
export function tableHeight(w: number, rows: [string, string][], o: { size?: number; keyFrac?: number; title?: string } = {}): number {
  const size = o.size ?? 8, lead = size * 1.28, keyW = w * (o.keyFrac ?? 0.38), pad = 3;
  let h = o.title ? lead + 4 : 0;
  for (const [k, v] of rows) h += Math.max(wrapText(k, keyW - 2 * pad, size).length, wrapText(v, w - keyW - 2 * pad, size).length) * lead + 2 * pad - 1;
  return h;
}

// ---------------------------------------------------------------- axes
export interface AxesOpts {
  xlim: [number, number]; ylim: [number, number];
  title?: string; xlabel?: string; ylabel?: string;
  logy?: boolean;                                   // ylim in data units (> 0), drawn as log10
  xticks?: { v: number; label: string }[];          // explicit (categorical) ticks instead of nice numbers
  xtickSize?: number; grid?: boolean;
}

export class Axes {
  readonly inner: Rect;
  private ylo: number; private yhi: number;
  constructor(readonly pg: PdfPage, outer: Rect, readonly o: AxesOpts) {
    const fs = 7;
    this.ylo = o.logy ? Math.log10(o.ylim[0]) : o.ylim[0];
    this.yhi = o.logy ? Math.log10(o.ylim[1]) : o.ylim[1];
    const yt = this.yTicks();
    const ytw = Math.max(0, ...yt.map((t) => textWidth(t.label, fs)));
    const left = ytw + 6 + (o.ylabel ? 11 : 0), top = o.title ? 14 : 4, bottom = 14 + (o.xlabel ? 10 : 0);
    this.inner = { x: outer.x + left, y: outer.y + top, w: Math.max(10, outer.w - left - 4), h: Math.max(10, outer.h - top - bottom) };
    const R = this.inner;
    if (o.title) pg.text(R.x + R.w / 2, outer.y + 9, fitText(o.title, outer.w, 8.5, 'bold'), { size: 8.5, font: 'bold', align: 'center', color: INK });
    if (o.ylabel) pg.text(outer.x + 7, R.y + R.h / 2, fitText(o.ylabel, R.h, fs), { size: fs, angle: 90, align: 'center', color: MUTED });
    if (o.xlabel) pg.text(R.x + R.w / 2, R.y + R.h + 22, o.xlabel, { size: fs, align: 'center', color: MUTED });
    for (const t of yt) {
      const y = this.Y(t.v, true);
      if (o.grid !== false) pg.line(R.x, y, R.x + R.w, y, { color: GRID, width: 0.4 });
      pg.line(R.x - 2.5, y, R.x, y, { color: FRAME, width: 0.5 });
      pg.text(R.x - 4, y + 2.4, t.label, { size: fs, align: 'right', color: MUTED });
    }
    const xs = o.xtickSize ?? fs;
    for (const t of this.xTicks()) {
      const x = this.X(t.v);
      pg.line(x, R.y + R.h, x, R.y + R.h + 2.5, { color: FRAME, width: 0.5 });
      pg.text(x, R.y + R.h + 4 + xs, t.label, { size: xs, align: 'center', color: MUTED });
    }
    pg.rect(R.x, R.y, R.w, R.h, { stroke: { color: FRAME, width: 0.5 } });
  }
  private yTicks(): { v: number; label: string }[] {
    if (this.o.logy) {
      const out: { v: number; label: string }[] = [];
      const d = Math.max(1, Math.ceil((Math.floor(this.yhi) - Math.ceil(this.ylo)) / 5));
      for (let e = Math.ceil(this.ylo - 1e-9); e <= this.yhi + 1e-9; e += d) out.push({ v: e, label: e >= -2 && e <= 3 ? String(10 ** e) : `1e${e}` });
      return out;
    }
    const t = niceTicks(this.ylo, this.yhi, 5), st = t.length > 1 ? t[1] - t[0] : Math.abs(t[0]) || 1;
    return t.map((v) => ({ v, label: fmtTick(v, st) }));
  }
  private xTicks(): { v: number; label: string }[] {
    if (this.o.xticks) return this.o.xticks;
    const [a, b] = this.o.xlim, t = niceTicks(a, b, Math.max(3, Math.min(7, Math.round(this.inner.w / 45)))), st = t.length > 1 ? t[1] - t[0] : 1;
    return t.map((v) => ({ v, label: fmtTick(v, st) }));
  }
  X(v: number) { const [a, b] = this.o.xlim; return this.inner.x + ((v - a) / (b - a)) * this.inner.w; }
  /** y in data units (or log10 units with raw = true on a log axis). */
  Y(v: number, raw = false) {
    const t = this.o.logy && !raw ? (v > 0 ? Math.log10(v) : NaN) : v;
    return this.inner.y + this.inner.h - ((t - this.ylo) / (this.yhi - this.ylo)) * this.inner.h;
  }
  clip(fn: () => void) { const R = this.inner; this.pg.clip(R.x, R.y, R.w, R.h, fn); }
  line(xs: ArrayLike<number>, ys: ArrayLike<number>, st: StrokeStyle = {}) {
    const pts = new Float64Array(2 * xs.length);
    for (let i = 0; i < xs.length; i++) { pts[2 * i] = this.X(xs[i]); pts[2 * i + 1] = this.Y(ys[i]); }
    this.clip(() => this.pg.polyline(pts, { width: 1, ...st }));
  }
  hline(y: number, st: StrokeStyle = {}) { const R = this.inner, Y = this.Y(y); if (Y >= R.y && Y <= R.y + R.h) this.pg.line(R.x, Y, R.x + R.w, Y, st); }
  vline(x: number, st: StrokeStyle = {}) { const R = this.inner, X = this.X(x); if (X >= R.x && X <= R.x + R.w) this.pg.line(X, R.y, X, R.y + R.h, st); }
  /** Legend in a corner of the plot area. */
  legend(items: { label: string; color: Color; kind?: 'box' | 'line'; dash?: number[] }[], corner: 'tr' | 'tl' | 'br' | 'bl' = 'tr') {
    if (!items.length) return;
    const fs = 6.5, lead = 9, w = 16 + Math.max(...items.map((i) => textWidth(i.label, fs))), h = items.length * lead + 3, R = this.inner;
    const x = corner[1] === 'r' ? R.x + R.w - w - 4 : R.x + 4, y = corner[0] === 't' ? R.y + 4 : R.y + R.h - h - 4;
    this.pg.rect(x, y, w, h, { fill: '#ffffff', stroke: { color: GRID, width: 0.4 } });
    items.forEach((it, i) => {
      const yy = y + 2 + i * lead + lead / 2;
      if (it.kind === 'line') this.pg.line(x + 3, yy, x + 12, yy, { color: it.color, width: 1.2, dash: it.dash });
      else this.pg.rect(x + 3, yy - 3, 9, 6, { fill: it.color });
      this.pg.text(x + 14, yy + 2.3, it.label, { size: fs, color: INK });
    });
  }
}

// ---------------------------------------------------------------- charts
export interface LineSeries { x: ArrayLike<number>; y: ArrayLike<number>; color?: Color; label?: string; dash?: number[]; width?: number }
export function linePlot(pg: PdfPage, r: Rect, o: { series: LineSeries[]; title?: string; xlabel?: string; ylabel?: string; logy?: boolean; legend?: 'tr' | 'tl' | 'br' | 'bl' }): Axes {
  const ys = o.series.map((s) => o.logy ? Array.from(s.y, (v) => (v > 0 ? v : NaN)) : s.y);
  const xr = finiteRange(o.series.map((s) => s.x)), yr = finiteRange(ys);
  const ylim: [number, number] = o.logy ? [10 ** Math.floor(Math.log10(yr[0]) + 1e-9), 10 ** Math.ceil(Math.log10(yr[1]) - 1e-9)] : padRange(yr, 0.06);
  if (o.logy && ylim[1] <= ylim[0]) ylim[1] = ylim[0] * 10;
  const ax = new Axes(pg, r, { xlim: xr[1] > xr[0] ? xr : padRange(xr), ylim, title: o.title, xlabel: o.xlabel, ylabel: o.ylabel, logy: o.logy });
  o.series.forEach((s, i) => ax.line(s.x, s.y, { color: s.color ?? PALETTE[i % PALETTE.length], width: s.width ?? 1, dash: s.dash }));
  const lab = o.series.filter((s) => s.label);
  if (lab.length > 1 || (lab.length === 1 && o.legend)) ax.legend(o.series.map((s, i) => ({ label: s.label ?? '', color: s.color ?? PALETTE[i % PALETTE.length], kind: 'line' as const, dash: s.dash })).filter((s) => s.label), o.legend ?? 'tr');
  return ax;
}

export interface BarSeries { values: ArrayLike<number>; errors?: ArrayLike<number> | null; color?: Color; label?: string }
/** Grouped bars per category with optional +-error bars. */
export function barChart(pg: PdfPage, r: Rect, o: { categories: string[]; series: BarSeries[]; title?: string; xlabel?: string; ylabel?: string; minRange?: number }): Axes {
  const n = o.categories.length, k = o.series.length;
  let lo = 0, hi = 0;
  for (const s of o.series) for (let i = 0; i < n; i++) {
    const v = s.values[i], e = s.errors ? Math.abs(s.errors[i]) || 0 : 0;
    if (Number.isFinite(v)) { lo = Math.min(lo, v - e); hi = Math.max(hi, v + e); }
  }
  const m = o.minRange ?? 0;
  if (hi - lo < m) { const c = (hi + lo) / 2; lo = Math.min(lo, c - m / 2); hi = Math.max(hi, c + m / 2); }
  const fs = n > 24 ? 5 : 6.5;
  const ax = new Axes(pg, r, { xlim: [-0.5, n - 0.5], ylim: padRange([lo, hi], 0.08), title: o.title, xlabel: o.xlabel, ylabel: o.ylabel, xticks: o.categories.map((label, v) => ({ v, label })), xtickSize: fs });
  const bw = Math.min(0.8 / k, 0.4), Wpx = ax.inner.w / n;
  ax.hline(0, { color: '#999', width: 0.5 });
  o.series.forEach((s, q) => {
    const col = s.color ?? PALETTE[q % PALETTE.length];
    for (let i = 0; i < n; i++) {
      const v = s.values[i];
      if (!Number.isFinite(v)) continue;
      const xc = i + (q - (k - 1) / 2) * bw, x0 = ax.X(xc - bw / 2), x1 = ax.X(xc + bw / 2);
      const y0 = ax.Y(0), y1 = ax.Y(v);
      pg.rect(x0, Math.min(y0, y1), Math.max(0.3, x1 - x0), Math.abs(y1 - y0), { fill: col });
      const e = s.errors ? s.errors[i] : NaN;
      if (Number.isFinite(e) && e > 0) {
        const X = ax.X(xc), ya = ax.Y(v - e), yb = ax.Y(v + e), cap = Math.min(2.5, (Wpx * bw) / 3);
        const st = { color: INK, width: 0.6 };
        pg.line(X, ya, X, yb, st); pg.line(X - cap, ya, X + cap, ya, st); pg.line(X - cap, yb, X + cap, yb, st);
      }
    }
  });
  if (k > 1 || o.series[0]?.label) ax.legend(o.series.map((s, q) => ({ label: s.label ?? '', color: s.color ?? PALETTE[q % PALETTE.length] })), 'tr');
  return ax;
}

/** Quantile of a sorted array. */
function quantile(s: Float64Array, q: number) { if (!s.length) return NaN; const t = q * (s.length - 1), i = Math.floor(t); return i + 1 < s.length ? s[i] + (t - i) * (s[i + 1] - s[i]) : s[i]; }

export interface HistSeries { values: ArrayLike<number>; color?: Color; label?: string }
/** Histogram of one or more series on common bins, stacked (e.g. accepted + rejected). Values outside the range are
 *  counted in the edge bins; the default range is the 0.5 - 99.5 % quantiles of all values. */
export function histogram(pg: PdfPage, r: Rect, o: { series: HistSeries[]; bins?: number; range?: [number, number]; title?: string; xlabel?: string; ylabel?: string;
  logy?: boolean; vlines?: { x: number; label?: string; color?: Color }[] }): Axes {
  const all: number[] = [];
  for (const s of o.series) for (let i = 0; i < s.values.length; i++) if (Number.isFinite(s.values[i])) all.push(s.values[i]);
  const srt = Float64Array.from(all).sort();
  let [lo, hi] = o.range ?? [quantile(srt, 0.005), quantile(srt, 0.995)];
  if (!(hi > lo)) [lo, hi] = padRange([srt[0] ?? 0, srt[srt.length - 1] ?? 1], 0.05);
  const nb = o.bins ?? Math.max(8, Math.min(50, Math.round(Math.sqrt(all.length))));
  const counts = o.series.map((s) => {
    const c = new Float64Array(nb);
    for (let i = 0; i < s.values.length; i++) { const v = s.values[i]; if (Number.isFinite(v)) c[Math.max(0, Math.min(nb - 1, Math.floor(((v - lo) / (hi - lo)) * nb)))]++; }
    return c;
  });
  const tot = new Float64Array(nb);
  for (const c of counts) for (let b = 0; b < nb; b++) tot[b] += c[b];
  const cmax = Math.max(1, ...tot);
  const ax = new Axes(pg, r, { xlim: [lo, hi], ylim: o.logy ? [0.8, 10 ** Math.ceil(Math.log10(cmax * 1.2))] : [0, cmax * 1.08], title: o.title, xlabel: o.xlabel, ylabel: o.ylabel ?? 'count', logy: o.logy });
  const base = new Float64Array(nb).fill(o.logy ? 0.8 : 0);
  ax.clip(() => counts.forEach((c, q) => {
    const col = o.series[q].color ?? PALETTE[q % PALETTE.length];
    for (let b = 0; b < nb; b++) {
      if (!c[b]) continue;
      const x0 = ax.X(lo + (b / nb) * (hi - lo)), x1 = ax.X(lo + ((b + 1) / nb) * (hi - lo));
      const y0 = ax.Y(base[b]), y1 = ax.Y(base[b] + c[b]);
      pg.rect(x0, y1, Math.max(0.2, x1 - x0 - 0.3), y0 - y1, { fill: col });
      base[b] += c[b];
    }
  }));
  // legend on the emptier side; line labels half-way down, on the side of the line away from the legend
  const third = Math.max(1, Math.floor(nb / 3)), side = (a: number, b: number) => Math.max(...tot.subarray(a, b));
  const right = side(nb - third, nb) <= side(0, third);
  for (const v of o.vlines ?? []) {
    ax.vline(v.x, { color: v.color ?? INK, width: 0.8, dash: [3, 2] });
    if (v.label) pg.text(ax.X(v.x) + (right ? -2 : 2), ax.inner.y + ax.inner.h * 0.45, v.label, { size: 6.5, color: v.color ?? INK, align: right ? 'right' : 'left' });
  }
  const lab = o.series.filter((s) => s.label);
  if (lab.length) ax.legend(o.series.map((s, q) => ({ label: `${s.label} (${s.values.length})`, color: s.color ?? PALETTE[q % PALETTE.length] })).filter((_, q) => o.series[q].label), right ? 'tr' : 'tl');
  return ax;
}

// ---------------------------------------------------------------- images
export function cmapRgb(cmap: Cmap, t: number): [number, number, number] {   // t in [0, 1]
  if (cmap === 'diverging') return diverging(2 * t - 1);
  if (cmap === 'gray') { const g = 255 * Math.max(0, Math.min(1, t)); return [g, g, g]; }
  return viridis(t);
}

export interface Image2D {
  data: ArrayLike<number>; width: number; height: number;   // row-major, row 0 at the top (unless origin 'lower')
  origin?: 'upper' | 'lower';
}

/** Colour-mapped RGB pixels, nearest-neighbour upscaled by an integer factor so that small images stay sharp in
 *  viewers that smooth images; NaN pixels are drawn in nanColor. */
export function imageRgb(im: Image2D, cmap: Cmap, vmin: number, vmax: number, o: { minPixels?: number; nanColor?: [number, number, number] } = {}) {
  const { width: w, height: h } = im;
  const f = Math.max(1, Math.min(16, Math.ceil((o.minPixels ?? 160) / Math.max(w, h))));
  const W = w * f, H = h * f, out = new Uint8Array(W * H * 3), nan = o.nanColor ?? [255, 255, 255];
  const span = vmax - vmin || 1;
  const row = new Uint8Array(W * 3);
  for (let y = 0; y < h; y++) {
    const sy = im.origin === 'lower' ? h - 1 - y : y;
    for (let x = 0; x < w; x++) {
      const v = im.data[sy * w + x];
      const c = Number.isFinite(v) ? cmapRgb(cmap, (v - vmin) / span) : nan;
      for (let k = 0; k < f; k++) { const o3 = 3 * (x * f + k); row[o3] = c[0]; row[o3 + 1] = c[1]; row[o3 + 2] = c[2]; }
    }
    for (let k = 0; k < f; k++) out.set(row, 3 * W * (y * f + k));
  }
  return { rgb: out, width: W, height: H };
}

export function dataRange(a: ArrayLike<number>, symmetric = false): [number, number] {
  const [lo, hi] = finiteRange([a]);
  if (symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return [-m, m]; }
  return hi > lo ? [lo, hi] : [lo - 1, lo + 1];
}

/** Vertical colour bar with ticks and an optional label. */
export function colorbar(doc: PdfDoc, pg: PdfPage, r: Rect, cmap: Cmap, vmin: number, vmax: number, label?: string) {
  const n = 128, g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = vmax - ((vmax - vmin) * i) / (n - 1);
  const im = imageRgb({ data: g, width: 1, height: n }, cmap, vmin, vmax, { minPixels: 1 });
  pg.image(doc.addImage(im.rgb, im.width, im.height), r.x, r.y, r.w, r.h);
  pg.rect(r.x, r.y, r.w, r.h, { stroke: { color: FRAME, width: 0.4 } });
  const t = niceTicks(vmin, vmax, Math.max(3, Math.min(5, Math.floor(r.h / 22)))), st = t.length > 1 ? t[1] - t[0] : 1;
  for (const v of t) {
    const y = r.y + r.h - ((v - vmin) / (vmax - vmin || 1)) * r.h;
    pg.line(r.x + r.w, y, r.x + r.w + 2, y, { color: FRAME, width: 0.4 });
    pg.text(r.x + r.w + 3.5, y + 2.2, fmtTick(v, st), { size: 6, color: MUTED });
  }
  if (label) pg.text(r.x + r.w + 26, r.y + r.h / 2, fitText(label, r.h, 6.5), { size: 6.5, angle: 90, align: 'center', color: MUTED });
}

export interface ImagePlotOpts {
  cmap?: Cmap; vmin?: number; vmax?: number; symmetric?: boolean;
  title?: string; colorbar?: boolean; cbarLabel?: string;
  /** physical extent [x0, x1, y0, y1] of the image (pixel edges) for the axis ticks; none: no ticks */
  extent?: [number, number, number, number]; xlabel?: string; ylabel?: string;
  /** physical aspect (height / width of one pixel); default 1 */
  pixelAspect?: number;
}
/** An image in a panel: kept at its aspect ratio, centred, with title and optional colour bar. Returns the image rect. */
export function imagePlot(doc: PdfDoc, pg: PdfPage, r: Rect, im: Image2D, o: ImagePlotOpts = {}): Rect {
  const cmap = o.cmap ?? 'viridis';
  const dr = dataRange(im.data, o.symmetric ?? cmap === 'diverging');
  const vmin = o.vmin ?? dr[0], vmax = o.vmax ?? dr[1];
  const top = o.title ? 12 : 0, cbw = o.colorbar ? 34 + (o.cbarLabel ? 8 : 0) : 0;
  const ticks = !!o.extent, left = ticks ? 24 + (o.ylabel ? 9 : 0) : 0, bottom = ticks ? 12 + (o.xlabel ? 9 : 0) : 0;
  const aw = r.w - cbw - left, ah = r.h - top - bottom;
  const aspect = (im.height * (o.pixelAspect ?? 1)) / im.width;
  let w = aw, h = aw * aspect;
  if (h > ah) { h = ah; w = ah / aspect; }
  const x = r.x + left + (aw - w) / 2, y = r.y + top + (ah - h) / 2;
  if (o.title) pg.text(x + w / 2, y - 4, fitText(o.title, r.w, 7.5, 'bold'), { size: 7.5, font: 'bold', align: 'center', color: INK });
  const px = imageRgb(im, cmap, vmin, vmax);
  pg.image(doc.addImage(px.rgb, px.width, px.height), x, y, w, h);
  pg.rect(x, y, w, h, { stroke: { color: FRAME, width: 0.4 } });
  if (o.extent) {
    const [x0, x1, y0, y1] = o.extent, fs = 6;
    const tx = niceTicks(Math.min(x0, x1), Math.max(x0, x1), 4), sx = tx.length > 1 ? tx[1] - tx[0] : 1;
    for (const v of tx) { const X = x + ((v - x0) / (x1 - x0)) * w; pg.line(X, y + h, X, y + h + 2, { color: FRAME, width: 0.4 }); pg.text(X, y + h + 8, fmtTick(v, sx), { size: fs, align: 'center', color: MUTED }); }
    const ty = niceTicks(Math.min(y0, y1), Math.max(y0, y1), 4), sy = ty.length > 1 ? ty[1] - ty[0] : 1;
    // extent y0 is the top edge for origin 'upper', the bottom edge for origin 'lower'
    for (const v of ty) {
      const f = (v - y0) / (y1 - y0), Y = im.origin === 'lower' ? y + h - f * h : y + f * h;
      pg.line(x - 2, Y, x, Y, { color: FRAME, width: 0.4 }); pg.text(x - 3, Y + 2, fmtTick(v, sy), { size: fs, align: 'right', color: MUTED });
    }
    if (o.xlabel) pg.text(x + w / 2, y + h + 16, o.xlabel, { size: 6.5, align: 'center', color: MUTED });
    if (o.ylabel) pg.text(r.x + 6, y + h / 2, o.ylabel, { size: 6.5, angle: 90, align: 'center', color: MUTED });
  }
  if (o.colorbar) colorbar(doc, pg, { x: x + w + 5, y, w: 6, h }, cmap, vmin, vmax, o.cbarLabel);
  return { x, y, w, h };
}

/** Split a rectangle into a grid of cells (row-major). */
export function grid(r: Rect, rows: number, cols: number, gap = 10): Rect[] {
  const w = (r.w - gap * (cols - 1)) / cols, h = (r.h - gap * (rows - 1)) / rows, out: Rect[] = [];
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out.push({ x: r.x + j * (w + gap), y: r.y + i * (h + gap), w, h });
  return out;
}
