// vec_forward / vec_jacobian against SIMCODE's VectorialPSF (tests/data/web/forward.*): tolerances from DESIGN.md.
import { test, check, fetchBuf, fetchJson, maxAbs, maxAbsDiff } from './harness';
import { getDevice } from '../src/gpu/device';
import { parseNpz } from '../src/io/npy';
import { VecEngine } from '../src/gpu/ops';
import { DEFAULT_NOLL } from '../src/psf/vectorial';

test('forward: PSF images and Jacobian vs SIMCODE', async () => {
  const { device } = await getDevice();
  const meta = await fetchJson('forward.json');
  const npz = await parseNpz(await fetchBuf('forward.npz'));
  const msgs: string[] = [];
  let worstV = 0, worstJ = 0;
  for (let ci = 0; ci < meta.cases.length; ci++) {
    const o = meta.cases[ci].optics;
    const eng = new VecEngine(device, { ...o, zernike_noll: DEFAULT_NOLL }, ['x', 'y', 'focus', 'ze', 'zern']);
    const pos = npz[`c${ci}_pos`].data as Float32Array, foc = npz[`c${ci}_focus`].data as Float32Array;
    const ref = npz[`c${ci}_psf`].data as Float32Array, jac = npz[`c${ci}_jac`].data as Float32Array;
    eng.setZernike(npz[`c${ci}_zcoef`].data as Float32Array);
    const B = foc.length, RR = o.roisize * o.roisize, D = eng.D;
    const ps = Array.from({ length: B }, (_, b) => ({ x: pos[3 * b], y: pos[3 * b + 1], ze: pos[3 * b + 2], f: foc[b], I: 1, bg: 0 }));
    const out = await eng.psfAndJacobian(ps);        // [D, B, RR]
    let ev = 0, ej = 0, ejName = '';
    for (let b = 0; b < B; b++) {
      ev = Math.max(ev, maxAbsDiff(out, b * RR, ref, b * RR, RR) / maxAbs(ref, b * RR, RR));
      for (let t = 1; t < D; t++) {
        const ro = (b * (D - 1) + (t - 1)) * RR;     // reference [B, 21, R, R]
        const e = maxAbsDiff(out, (t * B + b) * RR, jac, ro, RR) / Math.max(maxAbs(jac, ro, RR), 1e-3 * maxAbs(ref, b * RR, RR));
        if (e > ej) { ej = e; ejName = eng.names[t]; }
      }
    }
    worstV = Math.max(worstV, ev); worstJ = Math.max(worstJ, ej);
    msgs.push(`case${ci} value ${ev.toExponential(1)} jac ${ej.toExponential(1)} (${ejName})`);
    eng.destroy();
  }
  check(worstV < 1e-4 && worstJ < 1e-3, msgs.join('; '));
  return msgs.join('; ');
});
