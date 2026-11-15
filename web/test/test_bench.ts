// Timing (info only): one LM iteration's GPU work for the DESIGN.md "typical bead stack", 12 beads x 100 planes,
// ROI 26 px, npupil 64, oversampling 3 and 2: normal equations (value + 20 derivative directions) and one NLL pass.
import { test } from './harness';
import { getDevice } from '../src/gpu/device';
import { VecEngine } from '../src/gpu/ops';
import { DEFAULT_NOLL } from '../src/psf/vectorial';
import { photonTransfer } from '../src/smlm/camera';

test('bench: GPU time per LM iteration, 1200 images of 26 x 26', async () => {
  const { device } = await getDevice();
  const out: string[] = [];
  for (const os of [3, 2]) {
    const eng = new VecEngine(device, { pixel_um: 0.108, NA: 1.49, wavelength_um: 0.68, n_imm: 1.518, n_med: 1.33, roisize: 26, npupil: 64, oversample: os, zernike_noll: DEFAULT_NOLL }, ['x', 'y', 'focus', 'zern']);
    const B = 1200, RR = 26 * 26;
    const p = new Float32Array(B * 8);
    for (let b = 0; b < B; b++) p.set([0.1, -0.2, 0.1, -0.5 + b / B, 1000, 10, 0, 0], 8 * b);
    eng.setImages(p);
    eng.setData(new Float32Array(B * RR).fill(12));
    await eng.run(B, 'ne');   // warm-up (shader compile)
    let t = performance.now();
    await eng.run(B, 'ne');
    const tne = performance.now() - t;
    t = performance.now();
    await eng.run(B, 'nll');
    const tnll = performance.now() - t;
    out.push(`os ${os}: normal equations ${tne.toFixed(0)} ms, NLL pass ${tnll.toFixed(0)} ms`);
    eng.destroy();
  }
  return out.join('; ');
});

test('bench: camera photon transfer on a synthetic 512 x 512 movie (2048 frames, no TIFF decoding)', async () => {
  const { device } = await getDevice();
  const H = 512, W = 512, HW = H * W, n = 2048, NB = 16;
  // 16 noise frames (Gaussian approximation of Poisson, background 20 .. 50 photons over the field, gain 0.5,
  // offset 100), cycled; read() costs one copy per frame
  let s = 1;
  const u = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
  const base = new Float32Array(NB * HW);
  for (let i = 0; i < base.length; i++) {
    const p = i % HW, bg = 20 + 30 * (p % W) / W, z = Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
    base[i] = Math.round((bg + Math.sqrt(bg) * z) / 0.5 + 100);
  }
  const mv = { n, H, W, async read(f0: number, f1: number) {
    const out = new Float32Array((f1 - f0) * HW);
    for (let f = f0; f < f1; f++) out.set(base.subarray((f % NB) * HW, (f % NB + 1) * HW), (f - f0) * HW);
    return out;
  } };
  const r = await photonTransfer(device, mv, {});
  return `block_used ${r.block_used} (requested ${r.block_requested}), ${r.per_block.length} blocks, ${r.seconds.toFixed(2)} s ` +
    `(${(r.seconds / r.per_block.length).toFixed(2)} s per block); gain ${r.gain_e_per_adu.toFixed(3)} (synthetic truth 0.5), offset ${r.offset_adu.toFixed(1)}`;
});
