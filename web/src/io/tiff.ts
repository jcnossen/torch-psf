// Lazy TIFF / BigTIFF reader for dropped files: only byte ranges are read (Blob.slice), never the whole file.
// Supports little / big endian, uint8 / uint16 / uint32 / int16 / float32 single-channel, strips or tiles,
// uncompressed (fast path), Deflate / Adobe Deflate (DecompressionStream), LZW and PackBits, horizontal predictor,
// multi-part movies (movie.tif, movie_X1.tif, ... or movie_1.tif, ...) with one global frame index.

export class TiffError extends Error { constructor(m: string) { super(m); this.name = 'TiffError'; } }

export interface FrameInfo {
  offsets: number[];        // strip or tile offsets
  counts: number[];         // byte counts
}

interface Ifd {
  W: number; H: number; bps: number; fmt: number; comp: number; spp: number; planar: number; pred: number;
  rps: number; tiled: boolean; tw: number; th: number; offsets: number[]; counts: number[]; desc: string; next: number;
}

export type Dtype = 'uint8' | 'uint16' | 'uint32' | 'int16' | 'int32' | 'float32';

const COMP_NAMES: Record<number, string> = { 6: 'old JPEG', 7: 'JPEG', 34712: 'JPEG2000', 34925: 'LZMA', 50000: 'Zstd', 50001: 'WebP' };

class Part {
  le = true; big = false;
  size: number;
  private cache: { off: number; buf: DataView } | null = null;
  constructor(public blob: Blob, public name: string) { this.size = blob.size; }

  async bytes(off: number, len: number): Promise<DataView> {
    const c = this.cache;
    if (c && off >= c.off && off + len <= c.off + c.buf.byteLength) return new DataView(c.buf.buffer, c.buf.byteOffset + off - c.off, len);
    const n = Math.max(len, 65536);
    const ab = await this.blob.slice(off, Math.min(this.size, off + n)).arrayBuffer();
    this.cache = { off, buf: new DataView(ab) };
    if (ab.byteLength < len) throw new TiffError(`${this.name}: unexpected end of file`);
    return new DataView(ab, 0, len);
  }

  async raw(off: number, len: number): Promise<ArrayBuffer> { return this.blob.slice(off, off + len).arrayBuffer(); }

  async header(): Promise<number> {
    const d = await this.bytes(0, 16);
    const bo = d.getUint16(0, false);
    if (bo === 0x4949) this.le = true; else if (bo === 0x4d4d) this.le = false;
    else throw new TiffError(`${this.name}: not a TIFF file`);
    const magic = d.getUint16(2, this.le);
    if (magic === 42) return d.getUint32(4, this.le);
    if (magic === 43) { this.big = true; return Number(d.getBigUint64(8, this.le)); }
    throw new TiffError(`${this.name}: not a TIFF file (magic ${magic})`);
  }

  async ifd(off: number): Promise<Ifd> {
    const big = this.big, le = this.le;
    const cntSize = big ? 8 : 2, entSize = big ? 20 : 12, ptrSize = big ? 8 : 4;
    let d = await this.bytes(off, cntSize);
    const n = big ? Number(d.getBigUint64(0, le)) : d.getUint16(0, le);
    d = await this.bytes(off, cntSize + n * entSize + ptrSize);
    const tags: Record<number, number[] | string> = {};
    const tsz: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8 };
    const pending: [number, number, number, number][] = [];
    for (let i = 0; i < n; i++) {
      const e = cntSize + i * entSize;
      const tag = d.getUint16(e, le), type = d.getUint16(e + 2, le);
      const count = big ? Number(d.getBigUint64(e + 4, le)) : d.getUint32(e + 4, le);
      const vo = e + (big ? 12 : 8);
      const bytes = (tsz[type] ?? 1) * count;
      if (![256, 257, 258, 259, 270, 273, 277, 278, 279, 284, 317, 322, 323, 324, 325, 339].includes(tag)) continue;
      if (bytes <= ptrSize) tags[tag] = this.values(d, vo, type, count);
      else pending.push([tag, type, count, big ? Number(d.getBigUint64(vo, le)) : d.getUint32(vo, le)]);
    }
    for (const [tag, type, count, p] of pending) {
      const nb = (tsz[type] ?? 1) * count;
      const dv = new DataView(await this.raw(p, nb));
      tags[tag] = this.values(dv, 0, type, count);
    }
    const next = big ? Number(d.getBigUint64(cntSize + n * entSize, le)) : d.getUint32(cntSize + n * entSize, le);
    const num = (t: number, def: number) => { const v = tags[t]; return Array.isArray(v) ? v[0] : def; };
    const arr = (t: number) => { const v = tags[t]; return Array.isArray(v) ? v : []; };
    const W = num(256, 0), H = num(257, 0);
    const tiled = tags[322] !== undefined;
    return {
      W, H, bps: num(258, 1), fmt: num(339, 1), comp: num(259, 1), spp: num(277, 1), planar: num(284, 1), pred: num(317, 1),
      rps: Math.min(num(278, H), H), tiled, tw: tiled ? num(322, 0) : W, th: tiled ? num(323, 0) : num(278, H),
      offsets: tiled ? arr(324) : arr(273), counts: tiled ? arr(325) : arr(279),
      desc: typeof tags[270] === 'string' ? (tags[270] as string) : '', next,
    };
  }

  private values(d: DataView, o: number, type: number, count: number): number[] | string {
    const le = this.le;
    if (type === 2) {
      const b = new Uint8Array(d.buffer, d.byteOffset + o, count);
      return new TextDecoder().decode(b).replace(/\0+$/, '');
    }
    const out: number[] = new Array(count);
    for (let i = 0; i < count; i++) {
      switch (type) {
        case 1: case 7: out[i] = d.getUint8(o + i); break;
        case 3: out[i] = d.getUint16(o + 2 * i, le); break;
        case 4: case 13: out[i] = d.getUint32(o + 4 * i, le); break;
        case 16: out[i] = Number(d.getBigUint64(o + 8 * i, le)); break;
        case 8: out[i] = d.getInt16(o + 2 * i, le); break;
        case 9: out[i] = d.getInt32(o + 4 * i, le); break;
        case 5: out[i] = d.getUint32(o + 8 * i, le) / d.getUint32(o + 8 * i + 4, le); break;
        default: out[i] = 0;
      }
    }
    return out;
  }
}

interface PartFrames { part: Part; frames: FrameInfo[] | null; first: number; stride: number; n: number; ifd0: Ifd; }

/** Sort key of a multi-part file name: movie.tif -> 0, movie_X1.tif / movie_1.tif -> 1, ... */
export function partIndex(name: string): [string, number] {
  const m = /^(.*?)(?:_X?(\d+))?\.(?:ome\.)?tiff?$/i.exec(name);
  if (!m) return [name, 0];
  return [m[1], m[2] ? parseInt(m[2], 10) : 0];
}

export class TiffMovie {
  n = 0; H = 0; W = 0;
  dtype: Dtype = 'uint16';
  fastPath = true;
  compression = 1;
  description = '';
  private parts: PartFrames[] = [];
  private ifd0!: Ifd;

  static async open(files: Blob[] | File[], onProgress?: (msg: string) => void): Promise<TiffMovie> {
    const m = new TiffMovie();
    const named = (files as (Blob & { name?: string })[]).map((f, i) => ({ f, name: f.name ?? `part${i}.tif` }));
    named.sort((a, b) => {
      const [ba, ia] = partIndex(a.name), [bb, ib] = partIndex(b.name);
      return ba === bb ? ia - ib : ba < bb ? -1 : 1;
    });
    for (const { f, name } of named) {
      onProgress?.(`reading TIFF header of ${name}`);
      const p = new Part(f, name);
      m.parts.push(await m.scan(p));
    }
    const p0 = m.parts[0].ifd0;
    m.ifd0 = p0; m.H = p0.H; m.W = p0.W; m.compression = p0.comp; m.description = p0.desc;
    for (const pf of m.parts) {
      if (pf.ifd0.W !== m.W || pf.ifd0.H !== m.H || pf.ifd0.bps !== p0.bps) throw new TiffError(`${pf.part.name}: frame shape differs from the first part`);
      pf.first = m.n; m.n += pf.n;
      if (pf.frames) m.fastPath = false;
    }
    return m;
  }

  private check(i: Ifd, name: string) {
    if (COMP_NAMES[i.comp]) throw new TiffError(`${name}: ${COMP_NAMES[i.comp]} compression is not supported (use uncompressed, Deflate, LZW or PackBits)`);
    if (![1, 5, 8, 32946, 32773].includes(i.comp)) throw new TiffError(`${name}: compression ${i.comp} is not supported`);
    if (i.spp !== 1) throw new TiffError(`${name}: ${i.spp} samples per pixel (RGB / multi-sample) is not supported`);
    if (i.pred !== 1 && i.pred !== 2) throw new TiffError(`${name}: predictor ${i.pred} is not supported`);
    const key = `${i.bps}:${i.fmt}`;
    const dt: Record<string, Dtype> = { '8:1': 'uint8', '16:1': 'uint16', '32:1': 'uint32', '16:2': 'int16', '32:2': 'int32', '32:3': 'float32' };
    if (!dt[key]) throw new TiffError(`${name}: ${i.bps}-bit sample format ${i.fmt} is not supported`);
    this.dtype = dt[key];
  }

  private frameBytes(i: Ifd) { return i.W * i.H * i.bps / 8; }

  private contiguous(i: Ifd): boolean {
    if (i.comp !== 1 || i.tiled || i.pred !== 1) return false;
    for (let k = 1; k < i.offsets.length; k++) if (i.offsets[k] !== i.offsets[k - 1] + i.counts[k - 1]) return false;
    return i.counts.reduce((s, c) => s + c, 0) >= this.frameBytes(i);
  }

  private async scan(p: Part): Promise<PartFrames> {
    let off = await p.header();
    const i0 = await p.ifd(off);
    this.check(i0, p.name);
    const ome = /<OME[\s>]/.test(i0.desc);
    if (ome) {
      const sc = /SizeC="(\d+)"/.exec(i0.desc);
      if (sc && parseInt(sc[1]) > 1) throw new TiffError(`${p.name}: multi-channel OME-TIFF (SizeC ${sc[1]}) is not supported`);
    }
    const fb = this.frameBytes(i0);
    const ij = /images=(\d+)/.exec(i0.desc);
    // ImageJ: frames after the first IFD are contiguous, possibly without their own IFDs (> 4 GB stacks)
    if (ij && this.contiguous(i0)) {
      const n = parseInt(ij[1]);
      const last = i0.offsets[0] + n * fb;
      if (last <= p.size) return { part: p, frames: null, first: 0, stride: fb, n, ifd0: i0 };
    }
    // fast path: constant data stride and IFD stride over the first IFDs, extrapolated and verified on the last frame
    if (this.contiguous(i0) && i0.next) {
      const i1 = await p.ifd(i0.next);
      if (this.contiguous(i1) && i1.next) {
        const i2 = await p.ifd(i1.next);
        const s = i1.offsets[0] - i0.offsets[0], si = i1.next - i0.next;
        if (this.contiguous(i2) && i2.offsets[0] - i1.offsets[0] === s && s >= fb && si > 0) {
          const cand: number[] = [];
          const meta = /"shape":\s*\[([\d,\s]+)\]/.exec(i0.desc);      // tifffile
          if (meta) { const d = meta[1].split(',').map(Number); if (d.length >= 3) cand.push(d.slice(0, -2).reduce((a, b) => a * b, 1)); }
          const om = /<Pixels[^>]*>/.exec(i0.desc);                           // OME: SizeZ * SizeT * SizeC
          if (om) cand.push(['SizeZ', 'SizeT', 'SizeC'].map((k) => +(new RegExp(k + '="(\\d+)"').exec(om[0])?.[1] ?? 1)).reduce((a, b) => a * b, 1));
          const nd = Math.floor((p.size - i0.offsets[0] - fb) / s) + 1, ni = Math.floor((p.size - i0.next) / si) + 1;
          cand.push(Math.min(nd, ni), nd, ni, Math.min(nd, ni) - 1);
          for (const n of cand) {
            if (n < 3) continue;
            const lastIfd = i0.next + (n - 2) * si;          // IFD of frame k >= 1 at i0.next + (k - 1) si
            if (lastIfd + 2 > p.size) continue;
            try {
              const il = await p.ifd(lastIfd);
              if (il.W === i0.W && il.H === i0.H && il.offsets[0] === i0.offsets[0] + (n - 1) * s && il.next === 0) {
                return { part: p, frames: null, first: 0, stride: s, n, ifd0: i0 };
              }
            } catch { /* not an IFD: full walk */ }
          }
        }
      }
    }
    // full walk of the IFD chain
    const frames: FrameInfo[] = [];
    let i = i0;
    for (;;) {
      if (i.W !== i0.W || i.H !== i0.H) break;   // e.g. a thumbnail / reduced image
      frames.push({ offsets: i.offsets, counts: i.counts });
      if (!i.next) break;
      i = await p.ifd(i.next);
    }
    if (frames.length === 1 && ij && this.contiguous(i0)) {
      const n = parseInt(ij[1]);
      return { part: p, frames: null, first: 0, stride: fb, n, ifd0: i0 };
    }
    return { part: p, frames, first: 0, stride: 0, n: frames.length, ifd0: i0 };
  }

  /** Frames f0 .. f1 - 1 as float32 [f1 - f0, H, W] (optionally cropped to rows y0 .. y0 + h, cols x0 .. x0 + w). */
  async read(f0: number, f1: number, crop?: { y0: number; x0: number; h: number; w: number }): Promise<Float32Array> {
    const { H, W } = this;
    const c = crop ?? { y0: 0, x0: 0, h: H, w: W };
    const out = new Float32Array((f1 - f0) * c.h * c.w);
    let f = f0;
    while (f < f1) {
      const pf = this.parts.find((q) => f >= q.first && f < q.first + q.n);
      if (!pf) throw new TiffError(`frame ${f} out of range (${this.n} frames)`);
      const loc = f - pf.first;
      if (!pf.frames) {
        // contiguous frames: one slice for the run within this part (uncompressed, rows of the crop only)
        const nrun = Math.min(f1 - f, pf.n - loc);
        const bpp = pf.ifd0.bps / 8;
        const base = pf.ifd0.offsets[0] + loc * pf.stride;
        const startRow = c.y0 * W * bpp;
        const len = (nrun - 1) * pf.stride + (c.y0 + c.h) * W * bpp - startRow;
        const ab = await pf.part.raw(base + startRow, len);
        for (let k = 0; k < nrun; k++) {
          const fo = k * pf.stride;
          this.convert(ab, fo, c.h * W, pf.part.le, (row, vals) => out.set(vals.subarray(c.x0, c.x0 + c.w), ((f - f0 + k) * c.h + row) * c.w), W);
        }
        f += nrun;
      } else {
        const fr = await this.decodeFrame(pf, pf.frames[loc]);
        for (let y = 0; y < c.h; y++) out.set(fr.subarray((c.y0 + y) * W + c.x0, (c.y0 + y) * W + c.x0 + c.w), ((f - f0) * c.h + y) * c.w);
        f += 1;
      }
    }
    return out;
  }

  /** Convert nrows * W samples starting at byte offset `off` of `ab`, row by row. */
  private convert(ab: ArrayBuffer, off: number, nsamp: number, le: boolean, put: (row: number, v: Float32Array) => void, W: number) {
    const v = this.samples(ab, off, nsamp, le);
    for (let r = 0; r < nsamp / W; r++) put(r, v.subarray(r * W, (r + 1) * W));
  }

  private samples(ab: ArrayBuffer, off: number, n: number, le: boolean): Float32Array {
    const out = new Float32Array(n);
    const dv = new DataView(ab, off);
    switch (this.dtype) {
      case 'uint8': { const b = new Uint8Array(ab, off, n); for (let i = 0; i < n; i++) out[i] = b[i]; break; }
      case 'uint16':
        if (le && off % 2 === 0) { const b = new Uint16Array(ab, off, n); out.set(b); }
        else for (let i = 0; i < n; i++) out[i] = dv.getUint16(2 * i, le);
        break;
      case 'int16': for (let i = 0; i < n; i++) out[i] = dv.getInt16(2 * i, le); break;
      case 'uint32': for (let i = 0; i < n; i++) out[i] = dv.getUint32(4 * i, le); break;
      case 'int32': for (let i = 0; i < n; i++) out[i] = dv.getInt32(4 * i, le); break;
      case 'float32':
        if (le && off % 4 === 0) out.set(new Float32Array(ab, off, n));
        else for (let i = 0; i < n; i++) out[i] = dv.getFloat32(4 * i, le);
        break;
    }
    return out;
  }

  private async decodeFrame(pf: PartFrames, fr: FrameInfo): Promise<Float32Array> {
    const i = pf.ifd0, { W, H } = this, bpp = i.bps / 8;
    const out = new Float32Array(W * H);
    const tiled = i.tiled;
    const tw = i.tw, th = tiled ? i.th : i.rps;
    const ntx = Math.ceil(W / tw);
    // read the frame's byte range once when the blocks are close together
    const lo = Math.min(...fr.offsets), hi = Math.max(...fr.offsets.map((o, k) => o + fr.counts[k]));
    const all = await pf.part.raw(lo, hi - lo);
    for (let k = 0; k < fr.offsets.length; k++) {
      let blk = new Uint8Array(all, fr.offsets[k] - lo, fr.counts[k]);
      blk = await decompress(blk, i.comp);
      const bw = tw, bh = tiled ? th : Math.min(th, H - k * th);
      const need = bw * bh * bpp;
      if (blk.length < need) { const t = new Uint8Array(need); t.set(blk); blk = t; }
      const ab = blk.buffer.slice(blk.byteOffset, blk.byteOffset + need);
      const v = this.samples(ab, 0, bw * bh, pf.part.le);
      if (i.pred === 2) for (let r = 0; r < bh; r++) for (let x = 1; x < bw; x++) v[r * bw + x] = wrap(v[r * bw + x] + v[r * bw + x - 1], this.dtype);
      const ox = tiled ? (k % ntx) * tw : 0, oy = tiled ? Math.floor(k / ntx) * th : k * th;
      for (let r = 0; r < bh && oy + r < H; r++) {
        const w = Math.min(bw, W - ox);
        out.set(v.subarray(r * bw, r * bw + w), (oy + r) * W + ox);
      }
    }
    return out;
  }
}

function wrap(v: number, dt: Dtype): number {
  switch (dt) {
    case 'uint8': return v & 0xff;
    case 'uint16': return v & 0xffff;
    case 'int16': return ((v + 32768) & 0xffff) - 32768;
    case 'uint32': return v >>> 0;
    case 'int32': return v | 0;
    default: return v;
  }
}

export async function decompress(b: Uint8Array, comp: number): Promise<Uint8Array> {
  if (comp === 1) return b;
  if (comp === 8 || comp === 32946) {
    const s = new Blob([b]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(s).arrayBuffer());
  }
  if (comp === 5) return lzwDecode(b);
  if (comp === 32773) return packbitsDecode(b);
  throw new TiffError('unsupported compression ' + comp);
}

export function packbitsDecode(b: Uint8Array): Uint8Array {
  const out: number[] = [];
  let i = 0;
  while (i < b.length) {
    const n = (b[i++] << 24) >> 24;
    if (n >= 0) { for (let k = 0; k <= n && i < b.length; k++) out.push(b[i++]); }
    else if (n !== -128) { const v = b[i++]; for (let k = 0; k < 1 - n; k++) out.push(v); }
  }
  return Uint8Array.from(out);
}

/** TIFF LZW (MSB-first codes, early change). */
export function lzwDecode(b: Uint8Array): Uint8Array {
  const dict: Uint8Array[] = [];
  const reset = () => { dict.length = 0; for (let i = 0; i < 256; i++) dict.push(Uint8Array.of(i)); dict.push(new Uint8Array(0), new Uint8Array(0)); };
  reset();
  let out = new Uint8Array(Math.max(1024, b.length * 3));
  let op = 0;
  const put = (s: Uint8Array) => {
    if (op + s.length > out.length) { const t = new Uint8Array(Math.max(out.length * 2, op + s.length)); t.set(out); out = t; }
    out.set(s, op); op += s.length;
  };
  let bit = 0, width = 9;
  let prev: Uint8Array | null = null;
  const nbits = b.length * 8;
  while (bit + width <= nbits) {
    let code = 0;
    for (let k = 0; k < width; k++) { const p = bit + k; code = (code << 1) | ((b[p >> 3] >> (7 - (p & 7))) & 1); }
    bit += width;
    if (code === 257) break;
    if (code === 256) { reset(); width = 9; prev = null; continue; }
    let s: Uint8Array;
    if (code < dict.length) {
      s = dict[code];
      if (prev) { const e = new Uint8Array(prev.length + 1); e.set(prev); e[prev.length] = s[0]; dict.push(e); }
    } else {
      if (!prev) throw new TiffError('LZW: bad code');
      s = new Uint8Array(prev.length + 1); s.set(prev); s[prev.length] = prev[0]; dict.push(s);
    }
    put(s);
    prev = s;
    const L = dict.length;   // early change: the width grows one code before the table needs it
    width = L >= 2047 ? 12 : L >= 1023 ? 11 : L >= 511 ? 10 : 9;
  }
  return out.subarray(0, op);
}
