// TIFF reader against tifffile-written cases (tests/data/web/tiff/*, expected arrays in tiff.npz).
import { test, check, fetchBuf, fetchJson } from './harness';
import { TiffMovie } from '../src/io/tiff';
import { parseNpz } from '../src/io/npy';

const expectKey = (k: string) => k.startsWith('u8') ? 'u8' : k.startsWith('f32') ? 'f32' : k.startsWith('i16') ? 'i16' : k.startsWith('u32') ? 'u32' : 'u16';

test('tiff: sparse Micro-Manager >4 GB index, duplicate description and lazy pixels', async () => {
  const limit = 2 ** 32, lastIfd = limit - 256, lastPixels = limit - 16;
  const segments: [number, Uint8Array][] = [];
  const header = new DataView(new ArrayBuffer(112));
  header.setUint16(0, 0x4949); header.setUint16(2, 42, true); header.setUint32(4, 1024, true);
  header.setUint32(8, 54773648, true); header.setUint32(12, 64, true);
  header.setUint32(64, 3453623, true); header.setUint32(68, 2, true);
  header.setUint32(88, 1024, true); header.setUint32(100, 1, true); header.setUint32(108, lastIfd, true);
  segments.push([0, new Uint8Array(header.buffer)]);
  const ifd = (at: number, pixels: number, next: number, duplicate: boolean) => {
    const tags = [[256, 4, 1, 4], [257, 4, 1, 4], [258, 3, 1, 16], [259, 3, 1, 1],
      [273, 4, 1, pixels], [277, 3, 1, 1], [278, 4, 1, 4], [279, 4, 1, 32]];
    if (duplicate) tags.push([270, 2, 100_000_000, 3_000_000_000], [270, 2, 2, 65]);
    const d = new DataView(new ArrayBuffer(2 + tags.length * 12 + 4));
    d.setUint16(0, tags.length, true);
    tags.forEach(([tag, type, count, value], i) => {
      const o = 2 + i * 12;
      d.setUint16(o, tag, true); d.setUint16(o + 2, type, true);
      d.setUint32(o + 4, count, true); d.setUint32(o + 8, value, true);
    });
    d.setUint32(2 + tags.length * 12, next, true);
    segments.push([at, new Uint8Array(d.buffer)]);
    const data = new DataView(new ArrayBuffer(32));
    for (let i = 0; i < 16; i++) data.setUint16(i * 2, (duplicate ? 10 : 100) + i, true);
    segments.push([pixels, new Uint8Array(data.buffer)]);
  };
  ifd(1024, 2048, lastIfd, true); ifd(lastIfd, lastPixels, 0, false);
  let bytes = 0, maxRead = 0;
  class SparseTiff extends Blob {
    readonly name = 'sparse_MMStack_Pos0.ome.tif';
    get size() { return limit + 64; }
    async arrayBuffer(): Promise<ArrayBuffer> { throw new Error('whole-file read is forbidden'); }
    slice(start = 0, end = this.size) {
      const length = Math.min(this.size, end) - start;
      check(length <= 8 * 2 ** 20, `unbounded TIFF read: ${length}`);
      check(!(start <= 3_000_000_000 && end > 3_000_000_000), 'read superseded OME XML');
      bytes += length; maxRead = Math.max(maxRead, length);
      const data = new Uint8Array(length);
      for (const [at, s] of segments) {
        const lo = Math.max(start, at), hi = Math.min(end, at + s.length);
        if (lo < hi) data.set(s.subarray(lo - at, hi - at), lo - start);
      }
      return new Blob([data]);
    }
  }
  const messages: string[] = [];
  const movie = await TiffMovie.open([new SparseTiff()], (m) => messages.push(m));
  check(movie.n === 2 && bytes < 8192, `indexing read ${bytes} bytes for ${movie.n} frames`);
  check(messages.some((m) => m.includes('frame index')), 'missing indexing progress');
  const data = await movie.read(1, 2);
  check(data.every((v, i) => v === 100 + i), 'incorrect pixels crossing the 4 GB boundary');
  const first = await movie.read(0, 1);
  check(first.every((v, i) => v === 10 + i), 'incorrect first frame');
  return `${movie.n} frames; ${bytes} bytes read, largest slice ${maxRead} bytes`;
});

test('tiff: compressions, BigTIFF, tiles, strips, dtypes, byte order, multi-part', async () => {
  const meta = await fetchJson('tiff.json');
  const ref = await parseNpz(await fetchBuf('tiff.npz'));
  const bad: string[] = [];
  const info: string[] = [];
  const cmp = async (name: string, files: File[], key: string) => {
    const m = await TiffMovie.open(files);
    const exp = ref[key];
    const [n, H, W] = exp.shape;
    if (m.n !== n || m.H !== H || m.W !== W) { bad.push(`${name}: shape ${m.n}x${m.H}x${m.W}`); return; }
    const all = await m.read(0, n);
    for (let i = 0; i < all.length; i++) if (all[i] !== Number(exp.data[i])) { bad.push(`${name}: value ${i}: ${all[i]} != ${exp.data[i]}`); return; }
    const crop = await m.read(2, 4, { y0: 5, x0: 3, h: 7, w: 11 });
    for (let f = 0; f < 2; f++) for (let y = 0; y < 7; y++) for (let x = 0; x < 11; x++)
      if (crop[(f * 7 + y) * 11 + x] !== Number(exp.data[((f + 2) * H + y + 5) * W + x + 3])) { bad.push(`${name}: crop`); return; }
    info.push(`${name}${m.fastPath ? '*' : ''}`);
  };
  for (const c of meta.cases) {
    const buf = await fetchBuf('tiff/' + c.file);
    await cmp(c.file, [new File([buf], c.file)], expectKey(c.key));
  }
  const parts = await Promise.all(meta.multipart.map(async (f: string) => new File([await fetchBuf('tiff/' + f)], f)));
  await cmp('multipart', parts, 'u16');
  check(bad.length === 0, bad.join('; '));
  return `${info.length} files ok (* = fast path): ${info.join(' ')}`;
});
