// ImageJ hyperstack TIFF writer that returns a Blob assembled from the frame planes themselves (no copy of the pixel
// data in JS memory): uint8 or float32 pages, little endian, uncompressed, one strip per page.
//
// Page order is t-major, channel-minor (page = t * C + c), ImageJ's default XYCZT order, so tifffile reads the file
// as an array of shape (N, C, H, W) ((N, H, W) for C = 1) and Fiji opens it as a C-channel hyperstack. With C = 3 the
// channels are R, G, B ('mode=composite': Fiji shows them as red, green and blue). The N axis is written as frames
// ('frames=N', a time series) or as slices ('slices=N', a focus sweep).

export interface StackOpts {
  axis?: 'frames' | 'slices';
  info?: string;                 // free text in the ImageJ description (a single line, no '=' needed)
}

export function imagejDescription(N: number, C: number, opts: StackOpts = {}): string {
  let d = `ImageJ=1.11a\nimages=${N * C}\n`;
  if (C > 1) d += `channels=${C}\n`;
  if (N > 1) d += `${opts.axis ?? 'frames'}=${N}\n`;
  d += `hyperstack=true\nmode=${C === 3 ? 'composite' : 'grayscale'}\nloop=false\n`;
  if (opts.info) d += 'Info=' + opts.info.replace(/[\r\n\0]+/g, ' ') + '\n';
  return d;
}

/** planes: N * C arrays of H * W samples (all Uint8Array or all Float32Array), page order t * C + c. */
export function tiffStackBlob(planes: (Uint8Array | Float32Array)[], N: number, C: number, H: number, W: number, opts: StackOpts = {}): Blob {
  const n = N * C;
  if (planes.length !== n) throw new Error(`tiffStackBlob: ${planes.length} planes for N=${N} x C=${C}`);
  const f32 = planes[0] instanceof Float32Array;
  const bps = f32 ? 32 : 8, fmt = f32 ? 3 : 1;
  const frameBytes = H * W * bps / 8;
  for (const p of planes) {
    if ((p instanceof Float32Array) !== f32 || p.length !== H * W) throw new Error('tiffStackBlob: planes differ in type or size');
  }
  const descB = new TextEncoder().encode(imagejDescription(N, C, opts) + '\0');
  const descOff = 8, dataOff = Math.ceil((descOff + descB.length) / 16) * 16;
  const nent = 10, nent0 = 11;                      // page 0 also has the ImageDescription
  const ifd0Size = 2 + nent0 * 12 + 4, ifdSize = 2 + nent * 12 + 4;
  const ifdOff = dataOff + n * frameBytes;          // even (frameBytes * n and dataOff are even unless H W odd and 8-bit)
  const ifdStart = ifdOff + (ifdOff & 1);
  const total = ifdStart + ifd0Size + Math.max(0, n - 1) * ifdSize;
  if (total >= 2 ** 32) throw new Error(`the TIFF would be ${(total / 2 ** 30).toFixed(1)} GB (> 4 GB): record fewer or smaller frames`);

  const head = new Uint8Array(dataOff), hv = new DataView(head.buffer);
  head.set([0x49, 0x49, 42, 0]);
  hv.setUint32(4, ifdStart, true);
  head.set(descB, descOff);

  const ifds = new Uint8Array(total - ifdOff), dv = new DataView(ifds.buffer);
  let p = ifdStart - ifdOff;
  for (let k = 0; k < n; k++) {
    const first = k === 0, start = p, size = first ? ifd0Size : ifdSize;
    dv.setUint16(p, first ? nent0 : nent, true); p += 2;
    const ent = (tag: number, type: number, count: number, value: number) => {
      dv.setUint16(p, tag, true); dv.setUint16(p + 2, type, true); dv.setUint32(p + 4, count, true);
      if (type === 3 && count === 1) dv.setUint16(p + 8, value, true); else dv.setUint32(p + 8, value, true);
      p += 12;
    };
    ent(256, 4, 1, W);
    ent(257, 4, 1, H);
    ent(258, 3, 1, bps);
    ent(259, 3, 1, 1);                              // no compression
    ent(262, 3, 1, 1);                              // BlackIsZero
    if (first) ent(270, 2, descB.length, descOff);
    ent(273, 4, 1, dataOff + k * frameBytes);
    ent(277, 3, 1, 1);
    ent(278, 4, 1, H);
    ent(279, 4, 1, frameBytes);
    ent(339, 3, 1, fmt);
    dv.setUint32(start + size - 4, k + 1 < n ? ifdOff + start + size : 0, true);
    p = start + size;
  }
  return new Blob([head as BlobPart, ...planes as BlobPart[], ifds as BlobPart], { type: 'image/tiff' });
}
