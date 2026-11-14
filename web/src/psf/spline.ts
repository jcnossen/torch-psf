// Spline knots of a vectorial PSF for the convolutional Catmull-Rom spline (torch_psf CubicSplinePSF / SIMCODE
// simcode/psf/conv_spline.py): the PSF on a big ROI evaluated at S x S sub-pixel shifts and nz emitter heights
// (focus fixed), interleaved into an S-times oversampled z-stack [nz, nv, nu]. Same construction and grid as SIMCODE.
import { VecEngine } from '../gpu/ops';
import { VectorialConfig } from './vectorial';

export interface SplineKnots {
  knots: Float32Array;   // [nz, nv, nu]
  nz: number; nv: number; nu: number;
  z0: number; hz: number; S: number; u0: number; roisize: number; xmax: number; focus_um: number;
}

export async function buildKnots(dev: GPUDevice, cfg: VectorialConfig, zcoef: ArrayLike<number>,
  o: { roisize: number; focus_um: number; zmax: number; S?: number; hz?: number; zpad?: number; xmax?: number; zmin?: number },
  check?: () => void): Promise<SplineKnots> {
  const P = o.roisize, S = o.S ?? 2, hz = o.hz ?? 0.04, zpad = o.zpad ?? 3;
  const xmax = o.xmax ?? P / 2 + 1;
  const zmin = o.zmin ?? 0;
  const c = Math.ceil((P - 1) / 2 + xmax) + 2;
  const Pl = 2 * c + 1;
  const eng = new VecEngine(dev, { ...cfg, roisize: Pl }, []);
  try {
    eng.setZernike(zcoef);
    const nz = Math.round((o.zmax - zmin) / hz) + 1 + 2 * zpad;
    const z0 = zmin - zpad * hz;
    const ps: { x: number; y: number; ze: number; f: number; I: number; bg: number }[] = [];
    for (let k = 0; k < nz; k++) for (let b = 0; b < S; b++) for (let a = 0; a < S; a++)
      ps.push({ x: a / S, y: b / S, ze: z0 + hz * k, f: o.focus_um, I: 1, bg: 0 });
    eng.setImages(ps);
    await eng.run(ps.length, 'psf', check);
    const imgs = await eng.readPsf(ps.length);    // [nz, S(b), S(a), Pl(i), Pl(j)]
    const nv = Pl * S, nu = Pl * S;
    const knots = new Float32Array(nz * nv * nu);
    // G[z, i S + (S - 1 - b), j S + (S - 1 - a)] = imgs[z, b, a, i, j]
    for (let k = 0; k < nz; k++) for (let b = 0; b < S; b++) for (let a = 0; a < S; a++) {
      const src = ((k * S + b) * S + a) * Pl * Pl;
      for (let i = 0; i < Pl; i++) for (let j = 0; j < Pl; j++)
        knots[(k * nv + i * S + (S - 1 - b)) * nu + j * S + (S - 1 - a)] = imgs[src + i * Pl + j];
    }
    return { knots, nz, nv, nu, z0, hz, S, u0: -(c * S + S - 1) / S, roisize: P, xmax, focus_um: o.focus_um };
  } finally {
    eng.destroy();
  }
}
