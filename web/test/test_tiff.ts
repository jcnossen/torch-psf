// TIFF reader against tifffile-written cases (tests/data/web/tiff/*, expected arrays in tiff.npz).
import { test, check, fetchBuf, fetchJson } from './harness';
import { TiffMovie } from '../src/io/tiff';
import { parseNpz } from '../src/io/npy';

const expectKey = (k: string) => k.startsWith('u8') ? 'u8' : k.startsWith('f32') ? 'f32' : k.startsWith('i16') ? 'i16' : k.startsWith('u32') ? 'u32' : 'u16';

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
