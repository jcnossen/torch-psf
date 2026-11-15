// ImageJ hyperstack TIFF writer (float32, little endian, uncompressed, one strip per page) with the metadata that
// tifffile.imwrite(..., imagej=True, metadata={axes, unit, spacing, Info}, resolution=...) writes, as used for the
// psf.zip zstack.tif (docs/PSF_FORMAT.md, torch_psf.psfzip.zstack_tif_bytes): the ImageJ description (images, channels,
// slices, hyperstack, mode, unit, spacing), X / YResolution (px per unit) and the Info string in the IJMetadata tags
// (50838 / 50839). Pages in ImageJ order (channel fastest): data [Z, C, Y, X] as is.

/** Python repr() of a float (shortest round trip; scientific below 1e-4 and from 1e16, '.0' for integral values). */
export function pyFloat(x: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const [m, e] = x.toExponential().split('e');
  const ex = +e;
  if (ex < -4 || ex >= 16) return `${m}e${ex < 0 ? '-' : '+'}${String(Math.abs(ex)).padStart(2, '0')}`;
  const s = String(x);
  return /[.e]/.test(s) ? s : s + '.0';
}

/** tifffile.rational(x): Fraction(x).limit_denominator(min(2^32 - 1, (2^32 - 1) / x)) as [numerator, denominator]. */
export function rational(x: number): [number, number] {
  if (!(x > 0)) return [0, 1];
  let k = 0;
  while (!Number.isInteger(x * 2 ** k)) k++;
  const N0 = BigInt(x * 2 ** k), D0 = 2n ** BigInt(k);          // the exact value of x, reduced
  const M = BigInt(Math.min(4294967295, Math.floor(4294967295 / x)));
  if (D0 <= M) return [Number(N0), Number(D0)];
  let p0 = 0n, q0 = 1n, p1 = 1n, q1 = 0n, n = N0, d = D0;
  for (;;) {                                                     // Python's Fraction.limit_denominator
    const a = n / d, q2 = q0 + a * q1;
    if (q2 > M) break;
    [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
    [n, d] = [d, n - a * d];
  }
  const kk = (M - q0) / q1;
  if (2n * d * (q0 + kk * q1) <= D0) return [Number(p1), Number(q1)];
  return [Number(p0 + kk * p1), Number(q0 + kk * q1)];
}

export interface HyperstackMeta {
  unit?: string;                 // ImageJ unit (default 'um')
  spacing?: number;              // z step (unit); omitted when undefined
  resolution?: number;           // px per unit (XResolution = YResolution); omitted when undefined
  info?: string;                 // ImageJ Info (IJMetadata 'info')
}

/** data [Z, C, H, W] float32 -> TIFF bytes. */
export function writeImageJHyperstack(data: Float32Array, Z: number, C: number, H: number, W: number, meta: HyperstackMeta = {}): Uint8Array {
  const n = Z * C, frameBytes = H * W * 4;
  if (data.length < n * H * W) throw new Error('writeImageJHyperstack: data too short');
  let desc = `ImageJ=1.11a\nimages=${n}\n`;
  if (C > 1) desc += `channels=${C}\n`;
  if (Z > 1) desc += `slices=${Z}\n`;
  desc += 'hyperstack=true\nmode=grayscale\n';
  desc += `unit=${meta.unit ?? 'um'}\n`;
  if (meta.spacing !== undefined) desc += `spacing=${pyFloat(meta.spacing)}\n`;
  const enc = new TextEncoder();
  const descB = enc.encode(desc + '\0');
  // IJMetadata: little-endian header 'JIJI' + ('ofni', count 1), body = the Info string in UTF-16LE
  let ijData: Uint8Array | null = null, ijCounts: number[] = [];
  if (meta.info !== undefined) {
    const s = meta.info, body = new Uint8Array(2 * s.length);
    for (let i = 0; i < s.length; i++) { const c = s.charCodeAt(i); body[2 * i] = c & 255; body[2 * i + 1] = c >> 8; }
    const head = new Uint8Array(12);
    head.set(enc.encode('JIJIofni'), 0);
    new DataView(head.buffer).setUint32(8, 1, true);
    ijData = new Uint8Array(12 + body.length); ijData.set(head); ijData.set(body, 12);
    ijCounts = [12, body.length];
  }
  const res = meta.resolution !== undefined ? rational(meta.resolution) : null;
  // layout: header | description | IJ data | IJ counts | resolution rationals | pixel data (16-aligned) | IFDs
  const align = (v: number, a: number) => Math.ceil(v / a) * a;
  const descOff = 8;
  const ijOff = align(descOff + descB.length, 2);
  const ijcOff = align(ijOff + (ijData?.length ?? 0), 4);
  const resOff = ijcOff + 4 * ijCounts.length;
  const dataOff = align(resOff + (res ? 8 : 0), 16);
  const nent = 10 + (res ? 3 : 0), nent0 = nent + 1 + (ijData ? 2 : 0);   // page 0 also has the description and the IJ tags
  const ifd0Size = 2 + nent0 * 12 + 4, ifdSize = 2 + nent * 12 + 4;
  const ifdOff = align(dataOff + n * frameBytes, 2);
  const total = ifdOff + ifd0Size + Math.max(0, n - 1) * ifdSize;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  out.set([0x49, 0x49, 42, 0]);
  dv.setUint32(4, ifdOff, true);
  out.set(descB, descOff);
  if (ijData) { out.set(ijData, ijOff); ijCounts.forEach((c, i) => dv.setUint32(ijcOff + 4 * i, c, true)); }
  if (res) { dv.setUint32(resOff, res[0], true); dv.setUint32(resOff + 4, res[1], true); }
  out.set(new Uint8Array(data.buffer, data.byteOffset, n * frameBytes), dataOff);
  let p = ifdOff;
  for (let k = 0; k < n; k++) {
    const first = k === 0;
    const start = p;
    dv.setUint16(p, first ? nent0 : nent, true); p += 2;
    const ent = (tag: number, type: number, count: number, value: number) => {
      dv.setUint16(p, tag, true); dv.setUint16(p + 2, type, true); dv.setUint32(p + 4, count, true);
      if (type === 3 && count === 1) dv.setUint16(p + 8, value, true); else dv.setUint32(p + 8, value, true);
      p += 12;
    };
    ent(256, 4, 1, W);
    ent(257, 4, 1, H);
    ent(258, 3, 1, 32);
    ent(259, 3, 1, 1);
    ent(262, 3, 1, 1);
    if (first) ent(270, 2, descB.length, descOff);
    ent(273, 4, 1, dataOff + k * frameBytes);
    ent(277, 3, 1, 1);
    ent(278, 4, 1, H);
    ent(279, 4, 1, frameBytes);
    if (res) { ent(282, 5, 1, resOff); ent(283, 5, 1, resOff); ent(296, 3, 1, 1); }
    ent(339, 3, 1, 3);
    if (first && ijData) { ent(50838, 4, ijCounts.length, ijcOff); ent(50839, 1, ijData.length, ijOff); }
    const size = first ? ifd0Size : ifdSize;
    dv.setUint32(start + size - 4, k + 1 < n ? start + size : 0, true);
    p = start + size;
  }
  return out;
}
