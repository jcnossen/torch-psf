// Minimal dependency-free PDF 1.4 writer: pages, text in the standard Type1 fonts (Helvetica, Helvetica-Bold,
// Helvetica-Oblique with WinAnsi encoding; Greek letters and a few math signs via Symbol; nothing embedded), vector
// lines / polylines / rectangles, clipping and 8-bit RGB images. Streams are Flate-compressed with the browser's
// CompressionStream('deflate') (zlib format, as FlateDecode expects) and written uncompressed without it.
// Page coordinates are in points (1/72 inch) with the origin at the TOP-left corner, y down; the writer flips them.
import { HELVETICA_W, HELVETICA_BOLD_W, SYMBOL } from './pdffonts';

export type Color = string | [number, number, number];   // '#rgb', '#rrggbb' or [r, g, b] in 0..255
export type FontName = 'regular' | 'bold' | 'italic';
export interface StrokeStyle { color?: Color; width?: number; dash?: number[]; cap?: 0 | 1 | 2; join?: 0 | 1 | 2 }
export interface TextStyle { size?: number; font?: FontName; color?: Color; align?: 'left' | 'center' | 'right'; angle?: number }
export interface PdfImage { readonly name: string; readonly width: number; readonly height: number }

export const A4: [number, number] = [595.28, 841.89];

const FONT_RES: Record<FontName | 'symbol', string> = { regular: 'F1', bold: 'F2', italic: 'F3', symbol: 'F4' };
const FONT_BASE: [string, string][] = [['F1', 'Helvetica'], ['F2', 'Helvetica-Bold'], ['F3', 'Helvetica-Oblique'], ['F4', 'Symbol']];

// Unicode -> WinAnsi (cp1252) for the code points 0x80..0x9f of cp1252; 0xa0..0xff map to themselves.
const CP1252: Record<number, number> = {
  0x20ac: 128, 0x201a: 130, 0x0192: 131, 0x201e: 132, 0x2026: 133, 0x2020: 134, 0x2021: 135, 0x02c6: 136, 0x2030: 137,
  0x0160: 138, 0x2039: 139, 0x0152: 140, 0x017d: 142, 0x2018: 145, 0x2019: 146, 0x201c: 147, 0x201d: 148, 0x2022: 149,
  0x2013: 150, 0x2014: 151, 0x02dc: 152, 0x2122: 153, 0x0161: 154, 0x203a: 155, 0x0153: 156, 0x017e: 158, 0x0178: 159,
};

/** Split a string into runs of the text font (WinAnsi bytes) and the Symbol font; unsupported characters become '?'. */
function runs(s: string): { sym: boolean; bytes: number[]; w: number[] }[] {
  const out: { sym: boolean; bytes: number[]; w: number[] }[] = [];
  for (const ch of s) {
    const u = ch.codePointAt(0)!;
    let sym = false, b: number;
    if ((u >= 32 && u < 127) || (u >= 0xa0 && u <= 0xff)) b = u;
    else if (CP1252[u] !== undefined) b = CP1252[u];
    else if (SYMBOL[u]) { sym = true; b = SYMBOL[u][0]; }
    else b = u === 9 ? 32 : 63;   // tab -> space, else '?'
    const last = out[out.length - 1];
    if (last && last.sym === sym) last.bytes.push(b);
    else out.push({ sym, bytes: [b], w: [] });
  }
  return out;
}

let symW: Map<number, number> | null = null;   // Symbol code -> width
function glyphWidth(b: number, sym: boolean, font: FontName): number {
  if (sym) {
    if (!symW) { symW = new Map(); for (const k in SYMBOL) symW.set(SYMBOL[k][0], SYMBOL[k][1]); }
    return symW.get(b) ?? 500;
  }
  const t = font === 'bold' ? HELVETICA_BOLD_W : HELVETICA_W;
  return (b >= 32 && b <= 255 ? t[b - 32] : 0) || 556;
}

/** Width of a string in points. */
export function textWidth(s: string, size = 10, font: FontName = 'regular'): number {
  let w = 0;
  for (const r of runs(s)) for (const b of r.bytes) w += glyphWidth(b, r.sym, font);
  return (w * size) / 1000;
}

/** A number for a content stream: finite, at most 3 decimals, never in exponent notation. */
function num(v: number): string {
  if (!Number.isFinite(v)) return '0';
  const r = Math.round(Math.max(-1e7, Math.min(1e7, v)) * 1000) / 1000;
  return Object.is(r, -0) ? '0' : String(r);
}

export function parseColor(c: Color): [number, number, number] {
  if (Array.isArray(c)) return c;
  let h = c.replace('#', '');
  if (h.length === 3) h = h.split('').map((x) => x + x).join('');
  const v = parseInt(h, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
const rgb = (c: Color) => parseColor(c).map((v) => num(v / 255)).join(' ');

/** A PDF literal string from bytes: ( ) \ escaped, bytes outside printable ASCII as octal. */
function lit(bytes: number[]): string {
  let s = '(';
  for (const b of bytes) {
    if (b === 40 || b === 41 || b === 92) s += '\\' + String.fromCharCode(b);
    else if (b < 32 || b > 126) s += '\\' + b.toString(8).padStart(3, '0');
    else s += String.fromCharCode(b);
  }
  return s + ')';
}

export class PdfPage {
  readonly ops: string[] = [];
  readonly images = new Set<PdfImage>();   // drawn on this page (its /XObject resources)
  constructor(readonly width: number, readonly height: number) {}
  private Y(y: number) { return this.height - y; }

  private stroke(st: StrokeStyle) {
    this.ops.push(`${rgb(st.color ?? '#000')} RG ${num(st.width ?? 0.5)} w [${(st.dash ?? []).map(num).join(' ')}] 0 d ${st.cap ?? 0} J ${st.join ?? 0} j`);
  }
  line(x1: number, y1: number, x2: number, y2: number, st: StrokeStyle = {}) {
    this.stroke(st);
    this.ops.push(`${num(x1)} ${num(this.Y(y1))} m ${num(x2)} ${num(this.Y(y2))} l S`);
  }
  /** Polyline through [x0, y0, x1, y1, ...]; non-finite points break the line into segments. */
  polyline(pts: ArrayLike<number>, st: StrokeStyle = {}, close = false) {
    this.stroke({ join: 1, ...st });
    const p: string[] = [];
    let pen = false;
    for (let i = 0; i + 1 < pts.length; i += 2) {
      const x = pts[i], y = pts[i + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; continue; }
      p.push(`${num(x)} ${num(this.Y(y))} ${pen ? 'l' : 'm'}`);
      pen = true;
    }
    if (p.length) this.ops.push(p.join(' ') + (close ? ' h S' : ' S'));
  }
  /** Filled polygon through [x0, y0, ...]. */
  polygon(pts: ArrayLike<number>, fill: Color, st?: StrokeStyle) {
    if (st) this.stroke(st);
    const p: string[] = [];
    for (let i = 0; i + 1 < pts.length; i += 2) p.push(`${num(pts[i])} ${num(this.Y(pts[i + 1]))} ${i ? 'l' : 'm'}`);
    this.ops.push(`${rgb(fill)} rg ${p.join(' ')} h ${st ? 'B' : 'f'}`);
  }
  rect(x: number, y: number, w: number, h: number, o: { fill?: Color; stroke?: StrokeStyle } = {}) {
    if (!o.fill && !o.stroke) return;
    if (o.stroke) this.stroke(o.stroke);
    const r = `${num(x)} ${num(this.Y(y + h))} ${num(w)} ${num(h)} re`;
    this.ops.push(o.fill ? `${rgb(o.fill)} rg ${r} ${o.stroke ? 'B' : 'f'}` : `${r} S`);
  }
  /** Text with its baseline at y. angle: degrees counter-clockwise (90 = reading upwards). */
  text(x: number, y: number, s: string, st: TextStyle = {}) {
    const size = st.size ?? 9, font = st.font ?? 'regular';
    const a = ((st.angle ?? 0) * Math.PI) / 180, c = Math.cos(a), sn = Math.sin(a);
    const w = st.align && st.align !== 'left' ? textWidth(s, size, font) * (st.align === 'center' ? 0.5 : 1) : 0;
    const X = x - w * c, Y = this.Y(y) - w * sn;
    const parts = runs(s).map((r) => `/${FONT_RES[r.sym ? 'symbol' : font]} ${num(size)} Tf ${lit(r.bytes)} Tj`);
    if (!parts.length) return;
    this.ops.push(`BT ${rgb(st.color ?? '#000')} rg ${num(c)} ${num(sn)} ${num(-sn)} ${num(c)} ${num(X)} ${num(Y)} Tm ${parts.join(' ')} ET`);
  }
  /** Draw an image into the box (x, y) - (x + w, y + h); row 0 of the image at the top. */
  image(img: PdfImage, x: number, y: number, w: number, h: number) {
    this.images.add(img);
    this.ops.push(`q ${num(w)} 0 0 ${num(h)} ${num(x)} ${num(this.Y(y + h))} cm /${img.name} Do Q`);
  }
  /** Run fn with drawing clipped to the rectangle. */
  clip(x: number, y: number, w: number, h: number, fn: () => void) {
    this.ops.push(`q ${num(x)} ${num(this.Y(y + h))} ${num(w)} ${num(h)} re W n`);
    fn();
    this.ops.push('Q');
  }
}

interface ImageObj extends PdfImage { rgb: Uint8Array }

export interface PdfInfo { title?: string; author?: string; subject?: string; creator?: string }

const enc = (s: string) => { const b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i) & 255; return b; };
/** A PDF text string (UTF-16BE hex with a byte order mark) for the document info. */
function textString(s: string): string {
  let h = '<FEFF';
  for (let i = 0; i < s.length; i++) h += s.charCodeAt(i).toString(16).padStart(4, '0').toUpperCase();
  return h + '>';
}

async function deflate(b: Uint8Array): Promise<Uint8Array | null> {
  if (typeof CompressionStream === 'undefined') return null;
  try {
    const s = new Blob([b]).stream().pipeThrough(new CompressionStream('deflate'));
    return new Uint8Array(await new Response(s).arrayBuffer());
  } catch { return null; }
}

export class PdfDoc {
  readonly pages: PdfPage[] = [];
  private images: ImageObj[] = [];
  constructor(readonly info: PdfInfo = {}) {}

  addPage(size: [number, number] = A4): PdfPage {
    const p = new PdfPage(size[0], size[1]);
    this.pages.push(p);
    return p;
  }
  /** Register an RGB image (8 bits per channel, row-major, row 0 at the top); draw it with page.image. */
  addImage(rgbBytes: Uint8Array, width: number, height: number): PdfImage {
    if (rgbBytes.length !== width * height * 3) throw new Error(`pdf image: ${rgbBytes.length} bytes for ${width} x ${height} RGB`);
    const im: ImageObj = { name: 'Im' + (this.images.length + 1), width, height, rgb: rgbBytes };
    this.images.push(im);
    return im;
  }
  /** RGB image from RGBA pixels (canvas ImageData); alpha is composited on white. */
  addImageData(d: { data: Uint8ClampedArray | Uint8Array; width: number; height: number }): PdfImage {
    const n = d.width * d.height, out = new Uint8Array(3 * n);
    for (let i = 0; i < n; i++) {
      const a = d.data[4 * i + 3] / 255;
      for (let c = 0; c < 3; c++) out[3 * i + c] = Math.round(d.data[4 * i + c] * a + 255 * (1 - a));
    }
    return this.addImage(out, d.width, d.height);
  }

  /** The PDF file. compress = false (or no CompressionStream) writes all streams uncompressed; creationDate = false
   *  leaves out /CreationDate (a reproducible file, as in psf.zip). */
  async save(opts: { compress?: boolean; creationDate?: boolean } = {}): Promise<Uint8Array> {
    const compress = opts.compress ?? true;
    if (!this.pages.length) this.addPage();
    const chunks: Uint8Array[] = [];
    const offsets: number[] = [0];
    let pos = 0;
    const put = (b: Uint8Array | string) => { const u = typeof b === 'string' ? enc(b) : b; chunks.push(u); pos += u.length; };
    // object numbers: 1 catalog, 2 pages, 3 font resources, 4-7 fonts, 8 info, then images, then (page, content) pairs
    const nImg = this.images.length, img0 = 9, page0 = img0 + nImg;
    const nObj = page0 + 2 * this.pages.length;
    const obj = (id: number, body: string) => { offsets[id] = pos; put(`${id} 0 obj\n${body}\nendobj\n`); };
    const stream = async (id: number, dict: string, data: Uint8Array) => {
      const z = compress ? await deflate(data) : null;
      offsets[id] = pos;
      put(`${id} 0 obj\n<< ${dict} /Length ${(z ?? data).length}${z ? ' /Filter /FlateDecode' : ''} >>\nstream\n`);
      put(z ?? data);
      put('\nendstream\nendobj\n');
    };
    put('%PDF-1.4\n%\xe2\xe3\xcf\xd3\n');
    obj(1, '<< /Type /Catalog /Pages 2 0 R >>');
    obj(2, `<< /Type /Pages /Count ${this.pages.length} /Kids [${this.pages.map((_, i) => `${page0 + 2 * i} 0 R`).join(' ')}] >>`);
    obj(3, `<< ${FONT_BASE.map(([r], i) => `/${r} ${4 + i} 0 R`).join(' ')} >>`);
    const imgId = new Map(this.images.map((im, i) => [im.name, img0 + i]));
    FONT_BASE.forEach(([, base], i) => obj(4 + i, `<< /Type /Font /Subtype /Type1 /BaseFont /${base}${base === 'Symbol' ? '' : ' /Encoding /WinAnsiEncoding'} >>`));
    const d = new Date(), p2 = (v: number) => String(v).padStart(2, '0');
    const date = `D:${d.getUTCFullYear()}${p2(d.getUTCMonth() + 1)}${p2(d.getUTCDate())}${p2(d.getUTCHours())}${p2(d.getUTCMinutes())}${p2(d.getUTCSeconds())}Z`;
    const inf = this.info;
    obj(8, `<< /Producer ${textString('torch-psf web pdf.ts')} ${opts.creationDate === false ? '' : ` /CreationDate (${date})`}${inf.title ? ' /Title ' + textString(inf.title) : ''}` +
      `${inf.author ? ' /Author ' + textString(inf.author) : ''}${inf.subject ? ' /Subject ' + textString(inf.subject) : ''}${inf.creator ? ' /Creator ' + textString(inf.creator) : ''} >>`);
    for (let i = 0; i < nImg; i++) {
      const im = this.images[i];
      await stream(img0 + i, `/Type /XObject /Subtype /Image /Width ${im.width} /Height ${im.height} /ColorSpace /DeviceRGB /BitsPerComponent 8`, im.rgb);
    }
    for (let i = 0; i < this.pages.length; i++) {
      const pg = this.pages[i], id = page0 + 2 * i;
      const xo = [...pg.images].map((im) => `/${im.name} ${imgId.get(im.name)} 0 R`).join(' ');
      const res = `<< /ProcSet [/PDF /Text${xo ? ' /ImageC' : ''}] /Font 3 0 R${xo ? ` /XObject << ${xo} >>` : ''} >>`;
      obj(id, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${num(pg.width)} ${num(pg.height)}] /Resources ${res} /Contents ${id + 1} 0 R >>`);
      await stream(id + 1, '', enc(pg.ops.join('\n')));
    }
    const xref = pos;
    let x = `xref\n0 ${nObj}\n0000000000 65535 f \n`;
    for (let i = 1; i < nObj; i++) x += String(offsets[i]).padStart(10, '0') + ' 00000 n \n';
    put(x + `trailer\n<< /Size ${nObj} /Root 1 0 R /Info 8 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
    const out = new Uint8Array(pos);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }
}
