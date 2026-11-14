// Timing (info only): one LM iteration's GPU work for the DESIGN.md "typical bead stack", 12 beads x 100 planes,
// ROI 26 px, npupil 64, oversampling 3 and 2: normal equations (value + 20 derivative directions) and one NLL pass.
import { test } from './harness';
import { getDevice } from '../src/gpu/device';
import { VecEngine } from '../src/gpu/ops';
import { DEFAULT_NOLL } from '../src/psf/vectorial';

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
