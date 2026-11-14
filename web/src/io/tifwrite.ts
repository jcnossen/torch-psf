// Minimal TIFF writer: float32 z-stack [n, H, W], little endian, uncompressed, one strip per page, ImageJ
// description (so ImageJ / Fiji and tifffile read it as a stack).

export function writeTiffStack(data: Float32Array, n: number, H: number, W: number, extra: Record<string, number | string> = {}): Uint8Array {
  const desc = `ImageJ=1.11a\nimages=${n}\nslices=${n}\n` + Object.entries(extra).map(([k, v]) => `${k}=${v}\n`).join('') + '\0';
  const descB = new TextEncoder().encode(desc);
  const nent = 10, ifdSize = 2 + nent * 12 + 4;
  const frameBytes = H * W * 4;
  const head = 8, descOff = head, dataOff = Math.ceil((descOff + descB.length) / 16) * 16;
  const ifdOff = dataOff + n * frameBytes;
  const total = ifdOff + n * ifdSize;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  out.set([0x49, 0x49, 42, 0]);
  dv.setUint32(4, ifdOff, true);
  out.set(descB, descOff);
  out.set(new Uint8Array(data.buffer, data.byteOffset, n * frameBytes), dataOff);
  for (let k = 0; k < n; k++) {
    let p = ifdOff + k * ifdSize;
    dv.setUint16(p, nent, true); p += 2;
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
    ent(270, 2, k === 0 ? descB.length : 1, k === 0 ? descOff : descOff + descB.length - 1);
    ent(273, 4, 1, dataOff + k * frameBytes);
    ent(278, 4, 1, H);
    ent(279, 4, 1, frameBytes);
    ent(339, 3, 1, 3);
    dv.setUint32(p, k + 1 < n ? ifdOff + (k + 1) * ifdSize : 0, true);
  }
  return out;
}
