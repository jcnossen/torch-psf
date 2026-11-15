// PDF fit report of a PSF estimation (the fit_report.pdf of a psf.zip): settings and results tables, ROI counts and
// histograms, loss convergence, Zernike coefficients with error bars, pupil amplitude / phase, PSF xy / xz slices,
// example ROIs (data / model / difference) and Zernike-over-field maps. Every section is optional: sections without
// data are left out.
import { PdfDoc, PdfPage, A4, textWidth } from '../io/pdf';
import { nollToNm } from '../psf/vectorial';
import {
  Rect, Image2D, Cmap, INK, MUTED, PALETTE, table, tableHeight, paragraph, linePlot, barChart, histogram, imagePlot,
  dataRange, colorbar, grid, fitText, fmtNum, wrapText, LineSeries,
} from './plot';

export type SettingValue = string | number | boolean | null | undefined | number[];
export interface ReportImage extends Image2D { label?: string; extent?: [number, number, number, number]; xlabel?: string; ylabel?: string; pixelAspect?: number }

export interface FitReportData {
  title?: string;
  subtitle?: string;                       // e.g. the input file names
  created?: string;                        // ISO time; default now
  /** input settings (key -> value), shown as a table */
  settings?: Record<string, SettingValue>;
  /** results (key -> value), shown as a table next to the settings */
  results?: Record<string, SettingValue>;
  /** ROI bookkeeping: shown as a bar of accepted / rejected / held-out counts */
  rois?: { accepted: number; rejected?: number; heldout?: number; note?: string };
  zernike?: {
    noll: number[]; values: ArrayLike<number>; errors?: ArrayLike<number> | null;   // waves rms; errors: 1 sigma
    start?: ArrayLike<number> | null; truth?: ArrayLike<number> | null;
  };
  /** loss convergence; several curves (e.g. passes, train / held-out) as separate series */
  loss?: { series: { x?: ArrayLike<number>; y: ArrayLike<number>; label?: string }[]; xlabel?: string; ylabel?: string; logy?: boolean };
  histograms?: { title: string; xlabel?: string; accepted: ArrayLike<number>; rejected?: ArrayLike<number> | null; threshold?: number | null; bins?: number; range?: [number, number]; logy?: boolean }[];
  /** pupil maps; NaN outside the aperture. phase in waves */
  pupil?: { amplitude?: ReportImage | null; phase?: ReportImage | null };
  /** PSF slices: xy at several z (shared log or linear scale), and xz / yz sections (row 0 = top) */
  psf?: { xy?: ReportImage[]; xz?: ReportImage | null; yz?: ReportImage | null; log?: number /* decades; 0 = linear */ };
  /** example ROIs (data and model in photons); the difference is computed */
  examples?: { label?: string; data: ArrayLike<number>; model: ArrayLike<number>; width: number; height: number }[];
  /** Zernike coefficients over the field of view (waves rms), one map per Noll index */
  zernikeField?: { noll: number; map: ReportImage }[];
  notes?: string[];
}

const ZNAME: Record<string, string> = {
  '0,0': 'piston', '1,1': 'tilt', '2,0': 'defocus', '2,2': 'astig.', '3,1': 'coma', '3,3': 'trefoil', '4,0': 'spherical',
  '4,2': '2nd astig.', '4,4': 'quadrafoil', '5,1': '2nd coma', '5,3': '2nd trefoil', '5,5': 'pentafoil', '6,0': '2nd spherical',
};
export function zernikeName(j: number): string { const [n, m] = nollToNm(j); return ZNAME[`${n},${Math.abs(m)}`] ?? `n${n} m${m}`; }

function fmtSetting(v: SettingValue): string {
  if (v === null || v === undefined) return 'n/a';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number') return fmtNum(v, 4);
  if (Array.isArray(v)) return v.map((x) => fmtNum(x, 4)).join(', ');
  return v;
}
const rowsOf = (r?: Record<string, SettingValue>): [string, string][] => Object.entries(r ?? {}).filter(([, v]) => v !== undefined).map(([k, v]) => [k, fmtSetting(v)]);

/** Page flow: a cursor down the page, new pages as needed, header and footer on every page. */
class Flow {
  page!: PdfPage;
  y = 0;
  readonly M = 40;              // margin
  readonly W = A4[0] - 80;      // text width
  private readonly bottom = A4[1] - 46;
  constructor(readonly doc: PdfDoc, readonly header: string) { this.newPage(); }
  get x() { return this.M; }
  newPage() {
    this.page = this.doc.addPage(A4);
    this.page.text(this.M, 26, fitText(this.header, this.W * 0.75, 7), { size: 7, color: MUTED });
    this.page.line(this.M, 31, this.M + this.W, 31, { color: '#cccccc', width: 0.4 });
    this.y = 44;
  }
  need(h: number) { if (this.y + h > this.bottom) this.newPage(); }
  room() { return this.bottom - this.y; }
  section(title: string, minBody: number) {
    this.need(22 + minBody);
    this.page.text(this.M, this.y + 11, title, { size: 11, font: 'bold', color: INK });
    this.page.line(this.M, this.y + 15, this.M + this.W, this.y + 15, { color: PALETTE[0], width: 0.8 });
    this.y += 24;
  }
  /** Reserve a full-width band of height h and return it. */
  band(h: number): Rect { this.need(h); const r = { x: this.M, y: this.y, w: this.W, h }; this.y += h + 10; return r; }
  footers() {
    const n = this.doc.pages.length;
    this.doc.pages.forEach((p, i) => p.text(this.M + this.W, A4[1] - 24, `page ${i + 1} of ${n}`, { size: 7, color: MUTED, align: 'right' }));
  }
}

/** Shared colour range over several images (optionally log10 over `decades`). */
function sharedRange(ims: ArrayLike<number>[], decades: number): [number, number] {
  let hi = -Infinity, lo = Infinity;
  for (const a of ims) for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) { hi = Math.max(hi, a[i]); lo = Math.min(lo, a[i]); }
  if (!(hi > lo)) return [0, 1];
  return decades > 0 ? [Math.log10(hi) - decades, Math.log10(hi)] : [Math.min(0, lo), hi];
}
function logImage(im: ReportImage, decades: number, vmin: number): ReportImage {
  if (decades <= 0) return im;
  const d = Float32Array.from(im.data as ArrayLike<number>, (v) => (v > 0 ? Math.max(Math.log10(v), vmin) : vmin));
  return { ...im, data: d };
}

/** Diverging (symmetric about 0) colours only for maps with both signs; otherwise viridis on the data range. */
function crossesZero(a: ArrayLike<number>) { const [lo, hi] = dataRange(a); return lo < 0 && hi > 0; }

/** Build the report; returns the PDF bytes. */
export async function makeFitReport(d: FitReportData, opts: { compress?: boolean } = {}): Promise<Uint8Array> {
  const title = d.title ?? 'PSF fit report';
  const created = d.created ?? new Date().toISOString();
  const doc = new PdfDoc({ title, creator: 'torch-psf web', subject: d.subtitle });
  const F = new Flow(doc, `${title}  •  ${created.replace('T', ' ').replace(/\.\d+Z$/, ' UTC')}`);
  const pg = () => F.page;

  // ---- title
  pg().text(F.x, F.y + 16, title, { size: 17, font: 'bold', color: INK });
  F.y += 24;
  if (d.subtitle) F.y += paragraph(pg(), F.x, F.y, F.W, d.subtitle, { size: 9, color: MUTED }) + 2;
  F.y += 8;

  // ---- settings | results
  const set = rowsOf(d.settings), res = rowsOf(d.results);
  if (set.length || res.length) {
    const cw = (F.W - 16) / 2;
    const hs = tableHeight(cw, set, { title: 'Settings' }), hr = tableHeight(cw, res, { title: 'Results' });
    if (set.length && res.length && Math.max(hs, hr) <= F.room()) {   // side by side
      table(pg(), { x: F.x, y: F.y, w: cw }, set, { title: 'Settings' });
      table(pg(), { x: F.x + cw + 16, y: F.y, w: cw }, res, { title: 'Results' });
      F.y += Math.max(hs, hr) + 14;
    } else {                                                          // one below the other, rows flowing over pages
      for (const [t, rows] of [['Settings', set], ['Results', res]] as [string, [string, string][]][]) {
        if (!rows.length) continue;
        F.need(40);
        F.y += table(pg(), { x: F.x, y: F.y, w: F.W }, [], { title: t });
        for (const row of rows) { const h = tableHeight(F.W, [row], { keyFrac: 0.3 }); F.need(h); F.y += table(pg(), { x: F.x, y: F.y, w: F.W }, [row], { keyFrac: 0.3 }); }
        F.y += 14;
      }
    }
  }

  // ---- ROI counts
  if (d.rois) {
    const r = d.rois, parts = [['accepted', r.accepted, PALETTE[0]], ['rejected', r.rejected ?? 0, PALETTE[3]], ['held out', r.heldout ?? 0, PALETTE[2]]] as [string, number, string][];
    const tot = parts.reduce((s, p) => s + p[1], 0);
    if (tot > 0) {
      F.section('ROIs', 30);
      const b = F.band(r.note ? 42 : 30), bh = 12;
      let x = b.x;
      for (const [lab, v, col] of parts) {
        if (!v) continue;
        const w = (v / tot) * b.w;
        pg().rect(x, b.y, w, bh, { fill: col });
        const t = `${lab} ${v} (${((100 * v) / tot).toFixed(1)} %)`;
        if (textWidth(t, 7) < w - 4) pg().text(x + 3, b.y + 8.6, t, { size: 7, color: '#ffffff' });
        x += w;
      }
      pg().text(b.x, b.y + bh + 10, parts.filter((p) => p[1]).map(([lab, v]) => `${lab}: ${v}`).join('    ') + `    total: ${tot}`, { size: 7.5, color: INK });
      if (r.note) paragraph(pg(), b.x, b.y + bh + 13, b.w, r.note, { size: 7, color: MUTED });
    }
  }

  // ---- convergence and Zernikes
  const zs = d.zernike && d.zernike.noll.length ? d.zernike : null;
  const loss = d.loss?.series.filter((s) => s.y.length > 0) ?? [];
  if (zs || loss.length) {
    F.section(zs && loss.length ? 'Convergence and Zernike coefficients' : zs ? 'Zernike coefficients' : 'Convergence', 170);
    const b = F.band(180);
    const lw = zs && loss.length ? b.w * 0.36 : b.w;
    if (loss.length) {
      const series: LineSeries[] = loss.map((s, i) => ({ x: s.x ?? Float64Array.from({ length: s.y.length }, (_, k) => k), y: s.y, label: s.label, color: PALETTE[i % PALETTE.length] }));
      linePlot(pg(), { x: b.x, y: b.y, w: lw, h: b.h }, { series, title: 'Loss', xlabel: d.loss!.xlabel ?? 'iteration', ylabel: d.loss!.ylabel ?? 'loss', logy: d.loss!.logy });
    }
    if (zs) {
      const zx = loss.length ? b.x + lw + 14 : b.x, zw = loss.length ? b.w - lw - 14 : b.w;
      const series = [
        ...(zs.start ? [{ values: zs.start, color: '#b8bcc6', label: 'start' }] : []),
        { values: zs.values, errors: zs.errors, color: PALETTE[1], label: zs.errors ? 'fitted (± 1σ)' : 'fitted' },
        ...(zs.truth ? [{ values: zs.truth, color: PALETTE[2], label: 'truth' }] : []),
      ];
      barChart(pg(), { x: zx, y: b.y, w: zw, h: b.h }, { categories: zs.noll.map(String), series, title: 'Zernike coefficients', xlabel: 'Noll index', ylabel: 'waves rms', minRange: 0.02 });
    }
    if (zs) {   // values as a compact table: 3 columns of "Noll name value +- err"
      const n = zs.noll.length, cols = 3, per = Math.ceil(n / cols), cw = F.W / cols, lead = 10;
      F.need(per * lead + 6);
      for (let i = 0; i < n; i++) {
        const c = Math.floor(i / per), rr = i % per, x = F.x + c * cw, y = F.y + rr * lead + 8;
        const e = zs.errors ? zs.errors[i] : NaN;
        pg().text(x, y, `Z${zs.noll[i]}`, { size: 7.5, font: 'bold', color: INK });
        pg().text(x + 22, y, zernikeName(zs.noll[i]), { size: 7.5, color: MUTED });
        pg().text(x + cw - 14, y, `${zs.values[i] >= 0 ? '+' : '−'}${Math.abs(zs.values[i]).toFixed(4)}` + (Number.isFinite(e) ? ` ± ${e.toFixed(4)}` : ''), { size: 7.5, color: INK, align: 'right' });
      }
      F.y += per * lead + 14;
    }
  }

  // ---- histograms
  const hs = (d.histograms ?? []).filter((h) => h.accepted.length || h.rejected?.length);
  if (hs.length) {
    F.section('ROI statistics', 130);
    const cols = 3, rh = 132;
    for (let i = 0; i < hs.length; i += cols) {
      const b = F.band(rh), cells = grid(b, 1, cols, 14);
      hs.slice(i, i + cols).forEach((h, j) => histogram(pg(), cells[j], {
        series: [{ values: h.accepted, color: PALETTE[0], label: h.rejected?.length ? 'accepted' : undefined },
          ...(h.rejected?.length ? [{ values: h.rejected, color: PALETTE[3], label: 'rejected' }] : [])],
        title: h.title, xlabel: h.xlabel, bins: h.bins, range: h.range, logy: h.logy,
        vlines: Number.isFinite(h.threshold) ? [{ x: h.threshold!, label: 'threshold', color: PALETTE[3] }] : [],
      }));
    }
  }

  // ---- pupil
  const pup = [d.pupil?.amplitude && { im: d.pupil.amplitude, cmap: 'viridis' as Cmap, title: 'Pupil amplitude', cb: 'relative' },
    d.pupil?.phase && { im: d.pupil.phase, cmap: 'diverging' as Cmap, title: 'Pupil phase (aberration)', cb: 'waves' }].filter(Boolean) as { im: ReportImage; cmap: Cmap; title: string; cb: string }[];
  if (pup.length) {
    F.section('Pupil', 190);
    const b = F.band(200), cells = grid({ ...b, w: Math.min(b.w, pup.length * 250) }, 1, pup.length, 20);
    pup.forEach((p, i) => imagePlot(doc, pg(), cells[i], p.im, { cmap: p.cmap, title: p.im.label ?? p.title, colorbar: true, cbarLabel: p.cb, extent: p.im.extent, xlabel: p.im.xlabel, ylabel: p.im.ylabel }));
  }

  // ---- PSF slices
  const xy = d.psf?.xy ?? [], sec = [d.psf?.xz, d.psf?.yz].filter(Boolean) as ReportImage[];
  if (xy.length || sec.length) {
    const dec = d.psf?.log ?? 3;
    F.section('PSF', 120);
    if (xy.length) {
      const [vmin, vmax] = sharedRange(xy.map((s) => s.data), dec);
      const n = Math.min(xy.length, 7), cw = (F.W - 40) / n, b = F.band(Math.min(cw, 90) + 14);
      for (let i = 0; i < n; i++) imagePlot(doc, pg(), { x: b.x + i * cw, y: b.y, w: cw - 4, h: b.h }, logImage(xy[i], dec, vmin), { vmin, vmax, title: xy[i].label });
      colorbar(doc, pg(), { x: b.x + n * cw + 2, y: b.y + 12, w: 6, h: b.h - 14 }, 'viridis', vmin, vmax, dec > 0 ? 'log10' : undefined);
    }
    if (sec.length) {
      const cw = Math.min(F.W, 260 * sec.length) / sec.length - 16, aspect = Math.max(...sec.map((s) => (s.height * (s.pixelAspect ?? 1)) / s.width));
      const [vmin, vmax] = sharedRange(sec.map((s) => s.data), dec), b = F.band(Math.min(220, Math.max(80, (cw - 75) * aspect + 36)));
      const cells = grid({ ...b, w: Math.min(b.w, 260 * sec.length) }, 1, sec.length, 16);
      sec.forEach((s, i) => imagePlot(doc, pg(), cells[i], logImage(s, dec, vmin), { vmin, vmax, title: s.label, colorbar: true, cbarLabel: dec > 0 ? 'log10 intensity' : 'intensity',
        extent: s.extent, xlabel: s.xlabel, ylabel: s.ylabel, pixelAspect: s.pixelAspect }));
    }
    if (dec > 0) F.y += paragraph(pg(), F.x, F.y - 4, F.W, `Log scale over ${dec} decades below the maximum; xy and sections each on their own shared scale.`, { size: 7, color: MUTED }) + 4;
  }

  // ---- example ROIs
  const ex = d.examples ?? [];
  if (ex.length) {
    F.section('Example ROIs: data, model, data - model', 90);
    const per = 2, cw = F.W / per, iw = (cw - 52) / 3, rh = Math.min(iw, 90) + 26;
    for (let i = 0; i < ex.length; i += per) {
      const b = F.band(rh);
      ex.slice(i, i + per).forEach((e, j) => {
        const x0 = b.x + j * cw, n = e.width * e.height;
        const diff = new Float32Array(n);
        let vmax = 0;
        for (let k = 0; k < n; k++) { diff[k] = e.data[k] - e.model[k]; vmax = Math.max(vmax, e.data[k], e.model[k]); }
        const dm = dataRange(diff, true)[1];
        const im = (a: ArrayLike<number>) => ({ data: a, width: e.width, height: e.height });
        pg().text(x0, b.y + 8, fitText(e.label ?? `ROI ${i + j + 1}`, cw - 10, 7.5, 'bold'), { size: 7.5, font: 'bold', color: INK });
        const r = { y: b.y + 12, h: b.h - 12 };
        imagePlot(doc, pg(), { x: x0, w: iw, ...r }, im(e.data), { vmin: 0, vmax, title: 'data' });
        imagePlot(doc, pg(), { x: x0 + iw + 2, w: iw, ...r }, im(e.model), { vmin: 0, vmax, title: 'model' });
        imagePlot(doc, pg(), { x: x0 + 2 * (iw + 2), w: iw + 34, ...r }, im(diff), { cmap: 'diverging', vmin: -dm, vmax: dm, title: 'data - model', colorbar: true });
      });
    }
    F.y += paragraph(pg(), F.x, F.y - 4, F.W, 'Data and model on a shared linear scale (photons) per ROI; the difference on a symmetric diverging scale.', { size: 7, color: MUTED }) + 4;
  }

  // ---- Zernikes over the field
  const zf = d.zernikeField ?? [];
  if (zf.length) {
    F.section('Zernike coefficients over the field', 130);
    F.y += paragraph(pg(), F.x, F.y - 6, F.W, 'Maps with both signs on a symmetric diverging scale, the others on viridis over their own range.', { size: 7, color: MUTED }) + 6;
    const cols = 3, cw = F.W / cols, rh = Math.min(cw - 70, 140) + 34;
    for (let i = 0; i < zf.length; i += cols) {
      const b = F.band(rh);
      zf.slice(i, i + cols).forEach((z, j) => imagePlot(doc, pg(), { x: b.x + j * cw, y: b.y, w: cw - 6, h: b.h }, z.map, {
        cmap: crossesZero(z.map.data) ? 'diverging' : 'viridis', title: z.map.label ?? `Z${z.noll} ${zernikeName(z.noll)}`, colorbar: true, cbarLabel: 'waves rms', extent: z.map.extent, xlabel: z.map.xlabel, ylabel: z.map.ylabel,
      }));
    }
  }

  // ---- notes
  if (d.notes?.length) {
    F.section('Notes', 20);
    for (const s of d.notes) { F.need(wrapText('• ' + s, F.W, 8).length * 10.4 + 2); F.y += paragraph(pg(), F.x, F.y, F.W, '• ' + s, { size: 8 }) + 2; }
  }

  F.footers();
  return doc.save(opts);
}
