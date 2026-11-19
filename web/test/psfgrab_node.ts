// Unit checks of the PSF grab pure-TS parts, run in node (no browser): `npm run test:psfgrab` bundles this file with
// esbuild into test/out/ and runs it. FFT against a naive DFT, the paired real FFT, the power normalisation (Parseval),
// the radial average, sharpness / hfRatio ordering, the region helpers, the TIFF hyperstack writer (read back with
// the app's TiffMovie) and the Blob zip writer (against buildZip / ZipReader).
import { fft1d, fft2d, dftNaive } from '../src/psfgrab/fft';
import { hann, powerPair, radialAverage, analyzePlanes, sharpness, fftSizeFor } from '../src/psfgrab/spectrum';
import { regionAt, cropBin } from '../src/psfgrab/grab';
import { tiffStackBlob, imagejDescription } from '../src/psfgrab/tiffstack';
import { zipBlobs, crc32Blob } from '../src/psfgrab/zipblob';
import { buildZip, buildZipBlob, crc32, crc32Update, ZipReader } from '../src/io/zip';
import { TiffMovie } from '../src/io/tiff';

let failed = 0, passed = 0;
async function test(name: string, f: () => void | Promise<void>) {
  try { await f(); passed++; console.log(`PASS ${name}`); }
  catch (e) { failed++; console.log(`FAIL ${name}: ${(e as Error)?.stack ?? e}`); }
}
function check(c: boolean, msg: string) { if (!c) throw new Error(msg); }

// deterministic RNG (mulberry32) + Box-Muller
function rng(seed: number) {
  let a = seed >>> 0;
  const u = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const g = () => Math.sqrt(-2 * Math.log(u() + 1e-300)) * Math.cos(2 * Math.PI * u());
  return { u, g };
}
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

await test('fft1d matches the naive DFT (n = 2 .. 256, forward and inverse)', () => {
  const r = rng(1);
  for (const n of [2, 4, 8, 16, 64, 256]) {
    const re = Float64Array.from({ length: n }, r.g), im = Float64Array.from({ length: n }, r.g);
    const [er, ei] = dftNaive(re, im);
    const ar = Float64Array.from(re), ai = Float64Array.from(im);
    fft1d(ar, ai, n);
    const tol = 1e-9 * n;
    check(maxAbs(ar, er) < tol && maxAbs(ai, ei) < tol, `n=${n}: max error ${maxAbs(ar, er)}, ${maxAbs(ai, ei)}`);
    fft1d(ar, ai, n, 0, true);                       // inverse (unscaled) gives n x
    for (let i = 0; i < n; i++) { ar[i] /= n; ai[i] /= n; }
    check(maxAbs(ar, re) < tol && maxAbs(ai, im) < tol, `n=${n}: inverse round trip`);
  }
  let threw = false; try { fft1d(new Float64Array(12), new Float64Array(12), 12); } catch { threw = true; }
  check(threw, 'non power of two accepted');
});

await test('fft2d of a plane wave is two delta peaks', () => {
  const n = 32, kx = 3, ky = 5, re = new Float64Array(n * n), im = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) re[y * n + x] = Math.cos(2 * Math.PI * (kx * x + ky * y) / n);
  fft2d(re, im, n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const m = Math.hypot(re[y * n + x], im[y * n + x]);
    const peak = (x === kx && y === ky) || (x === n - kx && y === n - ky);
    check(peak ? Math.abs(m - n * n / 2) < 1e-6 : m < 1e-6, `bin (${x}, ${y}): ${m}`);
  }
});

await test('powerPair: two real images in one FFT = separate transforms', () => {
  const n = 64, r = rng(2);
  const a = Float64Array.from({ length: n * n }, r.g), b = Float64Array.from({ length: n * n }, () => r.u() * 5);
  const [pa, pb] = powerPair(a, b, n);
  const [qa] = powerPair(a, null, n), [qb] = powerPair(b, null, n);
  const scale = Math.max(...qa, ...qb);
  check(maxAbs(pa, qa) < 1e-9 * scale && maxAbs(pb!, qb) < 1e-9 * scale, `pair error ${maxAbs(pa, qa)}, ${maxAbs(pb!, qb)}`);
});

await test('power normalisation: mean P = sum (w x)^2 / sum w^2 (Parseval); white noise reads its variance', () => {
  const n = 128, r = rng(3), s = 7, w = hann(n);
  const x = Float64Array.from({ length: n * n }, () => s * r.g());
  const wx = new Float64Array(n * n);
  let sw2 = 0, swx2 = 0;
  for (let y = 0; y < n; y++) for (let i = 0; i < n; i++) { const v = x[y * n + i] * w[y] * w[i]; wx[y * n + i] = v; swx2 += v * v; sw2 += (w[y] * w[i]) ** 2; }
  const [p] = powerPair(wx, null, n);
  const mean = p.reduce((q, v) => q + v, 0) / p.length;
  check(Math.abs(mean - swx2 / sw2) < 1e-9 * mean, `Parseval: ${mean} vs ${swx2 / sw2}`);
  check(Math.abs(mean / (s * s) - 1) < 0.05, `white noise power ${mean} vs variance ${s * s}`);
  // through analyzePlanes: the high-frequency rings of white noise sit at log10(s^2)
  const a = analyzePlanes([Float32Array.from(x)], n, n, n);
  const hiRings = Array.from(a.profiles[0].subarray(n / 4, n / 2)), avg = hiRings.reduce((q, v) => q + v, 0) / hiRings.length;
  check(Math.abs(avg - Math.log10(s * s)) < 0.05, `profile level ${avg} vs ${Math.log10(s * s)}`);
});

await test('radialAverage: a constant spectrum stays constant; a ring lands in its bin', () => {
  const n = 64;
  const ones = new Float64Array(n * n).fill(2.5);
  const r1 = radialAverage(ones, n);
  check(r1.length === n / 2 + 1 && r1.every((v) => Math.abs(v - 2.5) < 1e-12), 'constant spectrum');
  const ring = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const kx = x < n / 2 ? x : x - n, ky = y < n / 2 ? y : y - n;
    if (Math.round(Math.hypot(kx, ky)) === 11) ring[y * n + x] = 1;
  }
  const r2 = radialAverage(ring, n);
  check(r2[11] === 1 && r2.every((v, i) => i === 11 || v === 0), 'ring at r = 11');
});

await test('analyzePlanes: a cosine grating peaks at its frequency in every channel; R, G, B kept apart', () => {
  const W = 300, H = 280, f = [0.125, 0.25, 0.0625];    // cycles / px for R, G, B
  const planes = f.map((fr) => Float32Array.from({ length: W * H }, (_, i) => 100 + 50 * Math.cos(2 * Math.PI * fr * (i % W))));
  const a = analyzePlanes(planes, W, H, 512);
  check(a.n === 256 && a.x0 === 22 && a.y0 === 12, `square ${a.n} at ${a.x0}, ${a.y0}`);
  a.profiles.forEach((p, c) => {
    let best = 1;
    for (let r = 1; r < p.length; r++) if (p[r] > p[best]) best = r;
    check(best === Math.round(f[c] * a.n), `channel ${c}: peak ring ${best}, expected ${f[c] * a.n}`);
  });
  check(a.mean.every((m) => Math.abs(m - 100) < 1), `means ${a.mean}`);
  check(fftSizeFor(1920, 1080, 512) === 512 && fftSizeFor(300, 200, 512) === 128 && fftSizeFor(5, 5, 512) === 8, 'fftSizeFor');
});

await test('sharpness and hfRatio: noise > edges > blurred edges', () => {
  const n = 128, r = rng(4);
  const noise = Float32Array.from({ length: n * n }, () => 100 + 20 * r.g());
  const sharpImg = Float32Array.from({ length: n * n }, (_, i) => ((i % n) >> 4) % 2 ? 150 : 50);   // 16 px bars
  const blur = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let s = 0; for (let d = -4; d <= 4; d++) s += sharpImg[y * n + Math.max(0, Math.min(n - 1, x + d))]; blur[y * n + x] = s / 9; }
  const sN = sharpness(noise, n, n), sS = sharpness(sharpImg, n, n), sB = sharpness(blur, n, n);
  check(sN > sS && sS > sB && sB > 0, `sharpness noise ${sN}, edges ${sS}, blurred ${sB}`);
  const hN = analyzePlanes([noise], n, n).hfRatio[0], hS = analyzePlanes([sharpImg], n, n).hfRatio[0], hB = analyzePlanes([blur], n, n).hfRatio[0];
  check(hN > hS && hS > hB, `hfRatio noise ${hN}, edges ${hS}, blurred ${hB}`);
  check(Math.abs(hN - 0.75) < 0.05, `white-noise hfRatio ${hN} (expected ~0.75, the annulus share weighted by r)`);
  check(sharpness(new Float32Array(100).fill(3), 10, 10) === 0, 'flat image');
});

await test('regionAt / cropBin', () => {
  const r = regionAt(1920, 1080, 100, 100, 512, 2);
  check(r.x === 0 && r.y === 0 && r.w === 512 && r.h === 512, `clamped region ${JSON.stringify(r)}`);
  const c = regionAt(1920, 1080, 960, 540, 0, 4);
  check(c.x === 0 && c.y === 0 && c.w === 1920 && c.h === 1080, `full frame ${JSON.stringify(c)}`);
  const o = regionAt(1001, 701, 500.5, 350.5, 0, 4);
  check(o.w === 1000 && o.h === 700, `multiple of the downscale ${JSON.stringify(o)}`);
  const s = regionAt(300, 200, 150, 100, 512);
  check(s.w === 300 && s.h === 200, 'crop larger than the frame');
  const W = 6, img = Float32Array.from({ length: 6 * 4 }, (_, i) => i);
  const b = cropBin(img, W, { x: 2, y: 0, w: 4, h: 4 }, 2);
  check(b.length === 4 && b[0] === (2 + 3 + 8 + 9) / 4 && b[3] === (16 + 17 + 22 + 23) / 4, `cropBin ${b}`);
});

await test('tiffStackBlob: uint8 RGB hyperstack and float32 stack read back with TiffMovie', async () => {
  const N = 4, C = 3, H = 5, W = 7;                    // odd sizes: odd data length, padded IFD start
  const planes = Array.from({ length: N * C }, (_, k) => Uint8Array.from({ length: H * W }, (_, i) => (k * 31 + i * 7) & 255));
  const blob = tiffStackBlob(planes, N, C, H, W, { axis: 'slices' });
  const mv = await TiffMovie.open([blob]);
  check(mv.n === N * C && mv.W === W && mv.H === H && mv.dtype === 'uint8', `shape ${mv.n} x ${mv.H} x ${mv.W} ${mv.dtype}`);
  check(/channels=3/.test(mv.description) && /slices=4/.test(mv.description) && /mode=composite/.test(mv.description), mv.description);
  const all = await mv.read(0, N * C);
  for (let k = 0; k < N * C; k++) for (let i = 0; i < H * W; i++) check(all[k * H * W + i] === planes[k][i], `page ${k} px ${i}`);
  // subarray views (as the recorder holds them) are written as their own bytes
  const big = new Uint8Array(1000).map((_, i) => i & 255);
  const views = [big.subarray(10, 10 + H * W), big.subarray(500, 500 + H * W)];
  const mv2 = await TiffMovie.open([tiffStackBlob(views, 2, 1, H, W)]);
  const a2 = await mv2.read(0, 2);
  check(a2[0] === 10 && a2[H * W] === (500 & 255), 'subarray planes');
  const f = [Float32Array.from({ length: 12 }, (_, i) => i / 3 - 1), Float32Array.from({ length: 12 }, (_, i) => -i * 1.5)];
  const mv3 = await TiffMovie.open([tiffStackBlob(f, 2, 1, 3, 4)]);
  check(mv3.dtype === 'float32' && mv3.n === 2, `float ${mv3.dtype} ${mv3.n}`);
  const a3 = await mv3.read(0, 2);
  check(maxAbs(a3, [...f[0], ...f[1]]) === 0, 'float values');
  check(imagejDescription(1, 1) === 'ImageJ=1.11a\nimages=1\nhyperstack=true\nmode=grayscale\nloop=false\n', 'single page description');
});

await test('zip: buildZipBlob = buildZip bytes; crc32Update chunks; zipBlobs readable by ZipReader', async () => {
  const r = rng(5);
  const a = Uint8Array.from({ length: 100000 }, () => (r.u() * 256) | 0), b = new TextEncoder().encode('{"x": 1}'), c = new Uint8Array(0);
  check(crc32Update(crc32Update(crc32Update(0, a.subarray(0, 333)), a.subarray(333, 70000)), a.subarray(70000)) === crc32(a), 'chunked crc');
  check(await crc32Blob(new Blob([a.subarray(0, 5), a.subarray(5)])) === crc32(a), 'crc32Blob');
  const items = [{ name: 'frames.tif', data: a }, { name: 'meta.json', data: b }, { name: 'empty', data: c }, { name: 'äö.txt', data: b }];
  const z1 = buildZip(items.map((m) => ({ name: m.name, body: m.data, method: 0 as const, crc: crc32(m.data), usize: m.data.length })));
  const z2 = new Uint8Array(await buildZipBlob(items.map((m) => ({ name: m.name, body: new Blob([m.data]), method: 0 as const, crc: crc32(m.data), usize: m.data.length }))).arrayBuffer());
  check(z1.length === z2.length && maxAbs(z1, z2) === 0, 'blob zip differs from buildZip');
  const z3 = new Uint8Array(await (await zipBlobs(items.map((m) => ({ name: m.name, data: new Blob([m.data]), compress: false })))).arrayBuffer());
  const zr = new ZipReader(z3);
  check(zr.names.join() === items.map((m) => m.name).join(), `names ${zr.names}`);
  for (const m of items) { const d = await zr.read(m.name); check(d.length === m.data.length && maxAbs(d, m.data) === 0, `member ${m.name}`); }
  // deflate needs CompressionStream('deflate-raw') (browsers; node >= 21.2)
  let raw = true; try { new CompressionStream('deflate-raw' as CompressionFormat); } catch { raw = false; }
  if (raw) {
    const z4 = new ZipReader(new Uint8Array(await (await zipBlobs([{ name: 't', data: new Blob([new Uint8Array(50000)]) }])).arrayBuffer()));
    check(z4.entries.get('t')!.method === 8 && (await z4.read('t')).length === 50000, 'deflated member');
  } else console.log('  (deflate-raw not available in this node: deflated members not checked)');
});

console.log(`${passed} passed, ${failed} failed`);
if (failed) throw new Error(`${failed} psfgrab unit check(s) failed`);
