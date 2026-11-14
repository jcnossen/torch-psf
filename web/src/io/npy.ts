// .npy / .npz read and write (npz: a zip of .npy files; written uncompressed ("stored"), read stored or deflated).

export interface NdArray { dtype: string; shape: number[]; data: Float32Array | Float64Array | Int32Array | Uint32Array | Uint16Array | Uint8Array | Int16Array | Int8Array | BigInt64Array; }

const DT: Record<string, [string, number]> = {
  '<f4': ['f4', 4], '<f8': ['f8', 8], '<i4': ['i4', 4], '<u4': ['u4', 4], '<u2': ['u2', 2], '|u1': ['u1', 1],
  '<i2': ['i2', 2], '|i1': ['i1', 1], '<i8': ['i8', 8],
};

function typed(code: string, buf: ArrayBuffer, off: number, n: number): NdArray['data'] {
  const copy = buf.slice(off, off + n * DT[code][1]);
  switch (code) {
    case '<f4': return new Float32Array(copy);
    case '<f8': return new Float64Array(copy);
    case '<i4': return new Int32Array(copy);
    case '<u4': return new Uint32Array(copy);
    case '<u2': return new Uint16Array(copy);
    case '|u1': return new Uint8Array(copy);
    case '<i2': return new Int16Array(copy);
    case '|i1': return new Int8Array(copy);
    case '<i8': return new BigInt64Array(copy);
  }
  throw new Error('npy: unsupported dtype ' + code);
}

export function parseNpy(buf: ArrayBuffer): NdArray {
  const b = new Uint8Array(buf);
  if (b[0] !== 0x93 || String.fromCharCode(...b.slice(1, 6)) !== 'NUMPY') throw new Error('not a .npy file');
  const major = b[6];
  const dv = new DataView(buf);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hoff = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(b.slice(hoff, hoff + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)![1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error('npy: fortran order not supported');
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)![1].split(',').map((s) => s.trim()).filter((s) => s).map(Number);
  const n = shape.reduce((a, c) => a * c, 1);
  if (!DT[descr]) throw new Error('npy: unsupported dtype ' + descr);
  return { dtype: DT[descr][0], shape, data: typed(descr, buf, hoff + hlen, n) };
}

const DESCR: Record<string, string> = {
  Float32Array: '<f4', Float64Array: '<f8', Int32Array: '<i4', Uint32Array: '<u4', Uint16Array: '<u2', Uint8Array: '|u1',
  Int16Array: '<i2', Int8Array: '|i1', BigInt64Array: '<i8',
};

export function writeNpy(data: NdArray['data'], shape: number[]): Uint8Array {
  const descr = DESCR[data.constructor.name];
  if (!descr) throw new Error('writeNpy: unsupported array type');
  let h = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const total = 10 + h.length + 1;
  h += ' '.repeat((64 - (total % 64)) % 64) + '\n';
  const out = new Uint8Array(10 + h.length + data.byteLength);
  out.set([0x93, 78, 85, 77, 80, 89, 1, 0]);
  new DataView(out.buffer).setUint16(8, h.length, true);
  out.set(new TextEncoder().encode(h), 10);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 10 + h.length);
  return out;
}

// --- zip (stored entries, CRC-32)
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

export function writeNpz(arrays: Record<string, { data: NdArray['data']; shape: number[] }>): Uint8Array {
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let off = 0;
  for (const [name, a] of Object.entries(arrays)) {
    const body = writeNpy(a.data, a.shape);
    const fn = enc.encode(name + '.npy');
    const crc = crc32(body);
    const lh = new Uint8Array(30 + fn.length);
    const dv = new DataView(lh.buffer);
    dv.setUint32(0, 0x04034b50, true); dv.setUint16(4, 20, true); dv.setUint16(8, 0, true);
    dv.setUint32(14, crc, true); dv.setUint32(18, body.length, true); dv.setUint32(22, body.length, true);
    dv.setUint16(26, fn.length, true); lh.set(fn, 30);
    const ch = new Uint8Array(46 + fn.length);
    const cv = new DataView(ch.buffer);
    cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true);
    cv.setUint32(16, crc, true); cv.setUint32(20, body.length, true); cv.setUint32(24, body.length, true);
    cv.setUint16(28, fn.length, true); cv.setUint32(42, off, true); ch.set(fn, 46);
    parts.push(lh, body); central.push(ch);
    off += lh.length + body.length;
  }
  const csize = central.reduce((s, c) => s + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, central.length, true); ev.setUint16(10, central.length, true);
  ev.setUint32(12, csize, true); ev.setUint32(16, off, true);
  const all = [...parts, ...central, end];
  const out = new Uint8Array(all.reduce((s, c) => s + c.length, 0));
  let p = 0;
  for (const c of all) { out.set(c, p); p += c.length; }
  return out;
}

export async function inflateRaw(b: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream('deflate-raw');
  const s = new Blob([b]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(s).arrayBuffer());
}

export async function parseNpz(buf: ArrayBuffer): Promise<Record<string, NdArray>> {
  const b = new Uint8Array(buf);
  const dv = new DataView(buf);
  let e = b.length - 22;
  while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new Error('npz: no zip directory');
  const n = dv.getUint16(e + 10, true);
  let p = dv.getUint32(e + 16, true);
  const out: Record<string, NdArray> = {};
  for (let i = 0; i < n; i++) {
    const method = dv.getUint16(p + 10, true);
    let csize = dv.getUint32(p + 20, true);
    const fnl = dv.getUint16(p + 28, true), exl = dv.getUint16(p + 30, true), cml = dv.getUint16(p + 32, true);
    let lho = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(b.slice(p + 46, p + 46 + fnl));
    if (csize === 0xffffffff || lho === 0xffffffff) {   // zip64 extra field
      let q = p + 46 + fnl;
      const qe = q + exl;
      while (q < qe) {
        const id = dv.getUint16(q, true), sz = dv.getUint16(q + 2, true);
        if (id === 1) {
          let r = q + 4;
          if (dv.getUint32(p + 24, true) === 0xffffffff) r += 8;
          if (csize === 0xffffffff) { csize = Number(dv.getBigUint64(r, true)); r += 8; }
          if (lho === 0xffffffff) lho = Number(dv.getBigUint64(r, true));
        }
        q += 4 + sz;
      }
    }
    const dstart = lho + 30 + dv.getUint16(lho + 26, true) + dv.getUint16(lho + 28, true);
    let body = b.slice(dstart, dstart + csize);
    if (method === 8) body = await inflateRaw(body);
    else if (method !== 0) throw new Error('npz: unsupported compression ' + method);
    out[name.replace(/\.npy$/, '')] = parseNpy(body.buffer);
    p += 46 + fnl + exl + cml;
  }
  return out;
}
