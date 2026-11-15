// .npy / .npz read and write (npz: a zip of .npy files; written uncompressed ("stored"), read stored or deflated).
import { crc32, inflateRaw, writeZipStored, ZipReader } from './zip';

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

// --- npz: a zip of .npy members (zip.ts)
export { crc32, inflateRaw };

export function writeNpz(arrays: Record<string, { data: NdArray['data']; shape: number[] }>): Uint8Array {
  return writeZipStored(Object.entries(arrays).map(([name, a]) => ({ name: name + '.npy', data: writeNpy(a.data, a.shape) })));
}

/** Every .npy member of a zip (name without .npy -> array); other members are ignored. */
export async function parseNpz(buf: ArrayBuffer): Promise<Record<string, NdArray>> {
  const z = new ZipReader(buf);
  const out: Record<string, NdArray> = {};
  for (const name of z.names) if (name.endsWith('.npy')) out[name.replace(/\.npy$/, '')] = parseNpy((await z.read(name)).buffer as ArrayBuffer);
  return out;
}
