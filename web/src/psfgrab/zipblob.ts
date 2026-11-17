// Zip export of Blob members (the recording TIFF can be hundreds of MB): CRC and deflate run as streams over the
// Blob, the archive is a Blob of the member bodies (io/zip.ts buildZipBlob), so the pixel data are never copied into
// one big JS buffer.
import { buildZipBlob, canDeflate, crc32Update } from '../io/zip';

export interface BlobMember { name: string; data: Blob; compress?: boolean }

async function forEachChunk(b: Blob, f: (c: Uint8Array) => void): Promise<void> {
  const r = b.stream().getReader();
  for (;;) {
    const { done, value } = await r.read();
    if (done) break;
    f(value as Uint8Array);
  }
}

export async function crc32Blob(b: Blob): Promise<number> {
  let c = 0;
  await forEachChunk(b, (x) => { c = crc32Update(c, x); });
  return c;
}

async function deflateBlob(b: Blob): Promise<Blob> {
  const s = b.stream().pipeThrough(new CompressionStream('deflate-raw') as unknown as ReadableWritablePair<Uint8Array, Uint8Array>);
  const parts: BlobPart[] = [];
  const r = s.getReader();
  for (;;) {
    const { done, value } = await r.read();
    if (done) break;
    parts.push(value as BlobPart);
  }
  return new Blob(parts);
}

/** Zip of the members in order; compress (default true) deflates where CompressionStream exists and it helps. */
export async function zipBlobs(members: BlobMember[], onProgress?: (msg: string) => void): Promise<Blob> {
  const items = [];
  let total = 0;
  for (const m of members) {
    onProgress?.(`checksum ${m.name}`);
    const crc = await crc32Blob(m.data);
    let body = m.data, method: 0 | 8 = 0;
    if ((m.compress ?? true) && canDeflate()) {
      onProgress?.(`compressing ${m.name}`);
      const d = await deflateBlob(m.data);
      if (d.size < m.data.size) { body = d; method = 8; }
    }
    total += body.size + 100;
    if (total >= 2 ** 32 || m.data.size >= 2 ** 32) throw new Error('the zip would exceed 4 GB (no zip64): record fewer or smaller frames');
    items.push({ name: m.name, body, method, crc, usize: m.data.size });
  }
  return buildZipBlob(items);
}
