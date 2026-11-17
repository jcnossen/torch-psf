// Zip container (PKWARE APPNOTE): a writer for arbitrary members, stored or deflated (CompressionStream
// 'deflate-raw' where the browser has it, else stored), with fixed member times (1980-01-01 00:00, as torch_psf.psfzip),
// so the output is byte-reproducible; and a reader of the central directory (zip64 sizes / offsets included) that
// extracts single members on demand (stored or deflated). No zip64 writing (members and archives < 4 GB).

export interface ZipMember { name: string; data: Uint8Array; store?: boolean }
export interface ZipEntry { name: string; method: number; csize: number; usize: number; offset: number; crc: number }

let crcTable: Uint32Array | null = null;
export function crc32(b: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcTable[n] = c >>> 0; }
  }
  let c = 0xffffffff;
  for (let i = 0; i < b.length; i++) c = crcTable[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const DOS_TIME = 0, DOS_DATE = (0 << 9) | (1 << 5) | 1;      // 1980-01-01 00:00:00
const EXT_ATTR = 0o644 << 16;                                 // -rw-r--r-- (as torch_psf.psfzip)

export const canDeflate = (): boolean => typeof CompressionStream !== 'undefined';

async function streamBytes(b: Uint8Array, t: { readable: ReadableStream; writable: WritableStream }): Promise<Uint8Array> {
  const s = new Blob([b as BlobPart]).stream().pipeThrough(t as unknown as ReadableWritablePair<Uint8Array, Uint8Array>);
  return new Uint8Array(await new Response(s).arrayBuffer());
}
export const deflateRaw = (b: Uint8Array) => streamBytes(b, new CompressionStream('deflate-raw'));
export const inflateRaw = (b: Uint8Array) => streamBytes(b, new DecompressionStream('deflate-raw'));

interface ZipItemHead { name: string; method: 0 | 8; crc: number; csize: number; usize: number }

/** Local header and central-directory record of one member whose body starts at byte `off`. */
function memberHeaders(it: ZipItemHead, off: number): [Uint8Array, Uint8Array] {
  const fn = new TextEncoder().encode(it.name);
  const utf8 = /[^\x00-\x7f]/.test(it.name) ? 0x800 : 0;
  const lh = new Uint8Array(30 + fn.length), dv = new DataView(lh.buffer);
  dv.setUint32(0, 0x04034b50, true); dv.setUint16(4, 20, true); dv.setUint16(6, utf8, true); dv.setUint16(8, it.method, true);
  dv.setUint16(10, DOS_TIME, true); dv.setUint16(12, DOS_DATE, true);
  dv.setUint32(14, it.crc, true); dv.setUint32(18, it.csize, true); dv.setUint32(22, it.usize, true);
  dv.setUint16(26, fn.length, true); lh.set(fn, 30);
  const ch = new Uint8Array(46 + fn.length), cv = new DataView(ch.buffer);
  cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, (3 << 8) | 20, true); cv.setUint16(6, 20, true); cv.setUint16(8, utf8, true);
  cv.setUint16(10, it.method, true); cv.setUint16(12, DOS_TIME, true); cv.setUint16(14, DOS_DATE, true);
  cv.setUint32(16, it.crc, true); cv.setUint32(20, it.csize, true); cv.setUint32(24, it.usize, true);
  cv.setUint16(28, fn.length, true); cv.setUint32(38, EXT_ATTR >>> 0, true); cv.setUint32(42, off, true); ch.set(fn, 46);
  return [lh, ch];
}

/** End-of-central-directory record. */
function endRecord(count: number, csize: number, off: number): Uint8Array {
  const end = new Uint8Array(22), ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, count, true); ev.setUint16(10, count, true);
  ev.setUint32(12, csize, true); ev.setUint32(16, off, true);
  return end;
}

/** Assemble a zip from members whose (possibly compressed) bodies are given: method 0 (stored) or 8 (deflate). */
export function buildZip(items: { name: string; body: Uint8Array; method: 0 | 8; crc: number; usize: number }[]): Uint8Array {
  const parts: Uint8Array[] = [], central: Uint8Array[] = [];
  let off = 0;
  for (const it of items) {
    const [lh, ch] = memberHeaders({ ...it, csize: it.body.length }, off);
    parts.push(lh, it.body); central.push(ch);
    off += lh.length + it.body.length;
  }
  const csize = central.reduce((s, c) => s + c.length, 0);
  const all = [...parts, ...central, endRecord(central.length, csize, off)];
  const out = new Uint8Array(all.reduce((s, c) => s + c.length, 0));
  let p = 0;
  for (const c of all) { out.set(c, p); p += c.length; }
  return out;
}

/** CRC-32 continued over one more chunk: crc32Update(crc32Update(0, a), b) === crc32(a + b). */
export function crc32Update(crc: number, b: Uint8Array): number {
  if (!crcTable) crc32(new Uint8Array(0));
  const t = crcTable!;
  let c = (crc ^ 0xffffffff) >>> 0;
  for (let i = 0; i < b.length; i++) c = t[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** As buildZip, but the bodies are Blobs and so is the result: the member data are referenced, not copied, so a
 *  large archive does not need a second copy in memory. */
export function buildZipBlob(items: { name: string; body: Blob; method: 0 | 8; crc: number; usize: number }[]): Blob {
  const parts: BlobPart[] = [], central: Uint8Array[] = [];
  let off = 0;
  for (const it of items) {
    const [lh, ch] = memberHeaders({ ...it, csize: it.body.size }, off);
    parts.push(lh as BlobPart, it.body); central.push(ch);
    off += lh.length + it.body.size;
  }
  const csize = central.reduce((s, c) => s + c.length, 0);
  return new Blob([...parts, ...central as BlobPart[], endRecord(central.length, csize, off) as BlobPart], { type: 'application/zip' });
}

/** Stored-only zip (synchronous). */
export function writeZipStored(members: ZipMember[]): Uint8Array {
  return buildZip(members.map((m) => ({ name: m.name, body: m.data, method: 0, crc: crc32(m.data), usize: m.data.length })));
}

/** Zip with every member deflated (unless store: true, or no CompressionStream: then stored), in the given order. */
export async function writeZip(members: ZipMember[], opts: { deflate?: boolean } = {}): Promise<Uint8Array> {
  const deflate = (opts.deflate ?? true) && canDeflate();
  const items = [];
  for (const m of members) {
    const crc = crc32(m.data);
    if (deflate && !m.store) items.push({ name: m.name, body: await deflateRaw(m.data), method: 8 as const, crc, usize: m.data.length });
    else items.push({ name: m.name, body: m.data, method: 0 as const, crc, usize: m.data.length });
  }
  return buildZip(items);
}

/** The central directory of a zip: one entry per member (sizes and offsets from zip64 extra fields when needed). */
export function zipEntries(buf: ArrayBuffer | Uint8Array): ZipEntry[] {
  const b = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let e = b.length - 22;
  while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new Error('zip: no central directory');
  const n = dv.getUint16(e + 10, true);
  let p = dv.getUint32(e + 16, true);
  const out: ZipEntry[] = [];
  const dec = new TextDecoder();
  for (let i = 0; i < n; i++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('zip: bad central directory entry');
    const method = dv.getUint16(p + 10, true), crc = dv.getUint32(p + 16, true);
    let csize = dv.getUint32(p + 20, true), usize = dv.getUint32(p + 24, true);
    const fnl = dv.getUint16(p + 28, true), exl = dv.getUint16(p + 30, true), cml = dv.getUint16(p + 32, true);
    let lho = dv.getUint32(p + 42, true);
    const name = dec.decode(b.subarray(p + 46, p + 46 + fnl));
    if (csize === 0xffffffff || usize === 0xffffffff || lho === 0xffffffff) {   // zip64 extra field
      let q = p + 46 + fnl;
      const qe = q + exl;
      while (q < qe) {
        const id = dv.getUint16(q, true), sz = dv.getUint16(q + 2, true);
        if (id === 1) {
          let r = q + 4;
          if (usize === 0xffffffff) { usize = Number(dv.getBigUint64(r, true)); r += 8; }
          if (csize === 0xffffffff) { csize = Number(dv.getBigUint64(r, true)); r += 8; }
          if (lho === 0xffffffff) lho = Number(dv.getBigUint64(r, true));
        }
        q += 4 + sz;
      }
    }
    out.push({ name, method, csize, usize, offset: lho, crc });
    p += 46 + fnl + exl + cml;
  }
  return out;
}

/** Random access to the members of a zip in memory. */
export class ZipReader {
  readonly entries: Map<string, ZipEntry>;
  private b: Uint8Array;
  constructor(buf: ArrayBuffer | Uint8Array) {
    this.b = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    this.entries = new Map(zipEntries(this.b).map((e) => [e.name, e]));
  }
  get names(): string[] { return [...this.entries.keys()]; }
  has(name: string): boolean { return this.entries.has(name); }
  /** The uncompressed bytes of a member (a copy). */
  async read(name: string): Promise<Uint8Array> {
    const e = this.entries.get(name);
    if (!e) throw new Error(`zip: no member ${name}`);
    const dv = new DataView(this.b.buffer, this.b.byteOffset, this.b.byteLength);
    if (dv.getUint32(e.offset, true) !== 0x04034b50) throw new Error(`zip: bad local header of ${name}`);
    const start = e.offset + 30 + dv.getUint16(e.offset + 26, true) + dv.getUint16(e.offset + 28, true);
    const body = this.b.slice(start, start + e.csize);
    if (e.method === 0) return body;
    if (e.method === 8) return inflateRaw(body);
    throw new Error(`zip: unsupported compression method ${e.method} (${name})`);
  }
  async text(name: string): Promise<string> { return new TextDecoder().decode(await this.read(name)); }
}

/** True if the bytes start like a zip file (local file header or an empty archive). */
export function isZip(b: Uint8Array): boolean {
  return b.length >= 4 && b[0] === 0x50 && b[1] === 0x4b && ((b[2] === 3 && b[3] === 4) || (b[2] === 5 && b[3] === 6));
}
