// Value-only CPU renderers for imported psf.zip channel models (torch_psf/pixint.py and vectorial.py).
import type { NdArray } from '../io/npy';
import { aberrationPhase, pupilData } from './vectorial';
import type { VectorialModel, SplineModel } from './zstack';

export interface ImportedRenderer {
  roisize: number;
  z_range: [number, number];
  feature: string;
  n_angles?: number;
  render(x: number, y: number, z: number, angle?: number): Float32Array;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
function bspline(t: number) {
  const a = Math.abs(t);
  return a < 1 ? 2 / 3 - a * a + a ** 3 / 2 : Math.max(2 - a, 0) ** 3 / 6;
}
// Integral from -infinity, using symmetry about zero.
function cumulative(t: number) {
  const a = Math.abs(t), tail = a >= 2 ? 0 : a >= 1 ? (2 - a) ** 4 / 24 : 0.5 - 2 * a / 3 + a ** 3 / 3 - a ** 4 / 8;
  return t < 0 ? tail : 1 - tail;
}

export function integratedRenderer(g: Omit<SplineModel, 'knots'> & { basis: string; value: string; n_angles: number },
    refs: Record<string, string>, arrays: Record<string, NdArray>): ImportedRenderer {
  if (g?.basis !== 'cubic_bspline' || g.value !== 'pixel_integral')
    throw new Error(`integrated_spline: unsupported basis/value ${g?.basis}/${g?.value}`);
  const { roisize: R, oversample: S, nz, n_angles: A } = g, n = (2 * g.c + 1) * S, size = nz * n * n;
  const t0 = (-(R - 1) / 2 - g.u0_px) * S - S / 2;
  if (![R, S, nz, n].every(v => Number.isSafeInteger(v) && v > 0) || nz < 4 || !Number.isInteger(A) || A < 0 ||
      !(g.dz_um > 0 && g.xy_limit >= 0 && t0 - g.xy_limit * S >= 1 &&
        Math.floor(t0 + g.xy_limit * S) + R * S + 2 < n))
    throw new Error('integrated_spline: invalid coefficient grid');
  const key = A ? 'coefs_angle' : 'coefs', arr = arrays[refs[key]], shape = A ? [A, nz, n, n] : [nz, n, n];
  if (refs[A ? 'coefs' : 'coefs_angle'] || !(arr?.data instanceof Float32Array) ||
      arr.shape.join() !== shape.join() || arr.data.length !== size * (A || 1))
    throw new Error(`integrated_spline: expected ${key} float32 [${shape}]`);
  const K = arr.data, mean = A ? new Float32Array(size) : K;
  if (A) for (let i = 0; i < size; i++) {
    let sum = 0;
    for (let a = 0; a < A; a++) sum += K[a * size + i];
    mean[i] = sum / A;
  }
  function axis(v: number) {
    const t = t0 - clamp(v, -g.xy_limit, g.xy_limit) * S, i = Math.floor(t) - 1;
    const w = Array.from({ length: S + 4 }, (_, k) => (cumulative(t - i - k + S) - cumulative(t - i - k)) / S);
    return { i, w };
  }
  return { roisize: R, z_range: g.z_range, feature: 'integrated_spline', n_angles: A, render(x, y, z, angle) {
    if (A && angle !== undefined && (!Number.isInteger(angle) || angle < 0 || angle >= A))
      throw new Error(`integrated_spline angle ${angle} outside 0..${A - 1}`);
    const coefs = angle === undefined ? mean : K, offset = A ? (angle ?? 0) * size : 0;
    const tz = clamp((z - g.z0_um) / g.dz_um, 1, nz - 2 - 1e-4), iz = Math.floor(tz) - 1;
    const X = axis(x), Y = axis(y), out = new Float32Array(R * R);
    for (let k = 0; k < 4; k++) {
      const wz = bspline(tz - iz - k);
      for (let p = 0; p < R; p++) for (let q = 0; q < R; q++) {
        let sum = 0;
        for (let j = 0; j < Y.w.length; j++) {
          const row = offset + ((iz + k) * n + Y.i + p * S + j) * n + X.i + q * S;
          let dot = 0;
          for (let i = 0; i < X.w.length; i++) dot += X.w[i] * coefs[row + i];
          sum += Y.w[j] * dot;
        }
        out[p * R + q] += wz * sum;
      }
    }
    return out;
  } };
}

// Gauss-Legendre quadrature on a unit-width pixel (Newton iteration on the Legendre polynomial).
function gaussNodes(n: number) {
  return Array.from({ length: n }, (_, i) => {
    let x = Math.cos(Math.PI * (i + 0.75) / (n + 0.5)), derivative = 0;
    for (let iter = 0; iter < 32; iter++) {
      let p = 1, prev = 0;
      for (let k = 1; k <= n; k++) [prev, p] = [p, ((2 * k - 1) * x * p - (k - 1) * prev) / k];
      derivative = n * (prev - x * p) / (1 - x * x);
      const step = p / derivative;
      x -= step;
      if (Math.abs(step) < 1e-15) break;
    }
    return { x: x / 2, w: 1 / ((1 - x * x) * derivative ** 2) };
  });
}

// Transform each complex row, transposing the output so the next pass takes the other axis.
function transformRows(src: Float64Array, rows: number, cols: number, F: Float32Array | Float64Array) {
  const samples = F.length / (2 * cols), dst = new Float64Array(2 * samples * rows);
  for (let s = 0; s < samples; s++) for (let r = 0; r < rows; r++) {
    let re = 0, im = 0;
    for (let i = 0; i < cols; i++) {
      const k = 2 * (r * cols + i), f = 2 * (s * cols + i);
      re += src[k] * F[f] - src[k + 1] * F[f + 1];
      im += src[k] * F[f + 1] + src[k + 1] * F[f];
    }
    const d = 2 * (s * rows + r);
    dst[d] = re; dst[d + 1] = im;
  }
  return dst;
}

// Python's padded spectral pixel filter. Evaluate the inverse DFT only at the cropped pixel centres.
function spectralPixels(R: number, os: number, pad: number, band: number) {
  if (os < 2 || band >= os / 2) throw new Error('spectral pixel_integration: insufficient oversample for the optical band limit');
  const n = (R + 2 * pad) * os, fc = Math.min(1.02 * band, os / 2 - 0.05);
  const forward = new Float64Array(2 * n * n), inverse = new Float64Array(2 * R * n);
  const freq = Float64Array.from({ length: n }, (_, k) => (k <= n / 2 ? k : k - n) * os / n);
  for (let k = 0; k < n; k++) {
    const f = freq[k], a = Math.PI * f;
    const taper = (1 + Math.cos(Math.PI * clamp((Math.abs(f) - fc) / (os / 2 - fc), 0, 1))) / 2;
    const gain = taper * (a ? Math.sin(a) / a : 1) / n;
    for (let j = 0; j < n; j++) {
      const phase = -2 * a * j / os, i = 2 * (k * n + j);
      forward[i] = Math.cos(phase); forward[i + 1] = Math.sin(phase);
    }
    for (let p = 0; p < R; p++) {
      const phase = 2 * a * (p + pad + (os - 1) / (2 * os)), i = 2 * (p * n + k);
      inverse[i] = gain * Math.cos(phase); inverse[i + 1] = gain * Math.sin(phase);
    }
  }
  return (density: Float64Array, blur?: number[]) => {
    const samples = new Float64Array(2 * n * n);
    for (let k = 0; k < density.length; k++) samples[2 * k] = density[k];
    const spectrum = transformRows(transformRows(samples, n, n, forward), n, n, forward);
    if (blur) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const q = blur[0] * freq[x] ** 2 + blur[1] * freq[y] ** 2 + 2 * blur[2] * freq[x] * freq[y];
      const gain = Math.exp(Math.min(3, -2 * Math.PI ** 2 * q)), i = 2 * (y * n + x);
      spectrum[i] *= gain; spectrum[i + 1] *= gain;
    }
    const image = transformRows(transformRows(spectrum, n, n, inverse), R, n, inverse);
    return Float32Array.from({ length: R * R }, (_, k) => image[2 * k]);
  };
}

interface VectorialTerms {
  psf_blur_px2?: number[]; psf_blur_angle_px2?: number[][];
  psf_dipole_M?: number[][]; psf_dipole_M_angle?: number[][][]; psf_dipole_w?: number[];
  diattenuation?: number;
}

function angleTerm(terms: number[][] | undefined, angle?: number) {
  if (!terms) return undefined;
  if (angle === undefined) return terms[0].map((_, i) => terms.reduce((s, row) => s + row[i], 0) / terms.length);
  if (!Number.isInteger(angle) || !terms[angle]) throw new Error(`vectorial angle ${angle} outside 0..${terms.length - 1}`);
  return terms[angle];
}

export function vectorialRenderer(m: VectorialModel & VectorialTerms, mode: 'spectral' | 'gauss_legendre',
    maps: { pupil_amp?: Float32Array; pupil_phase?: Float32Array } = {}): ImportedRenderer {
  const { roisize: R, oversample: os, optics: o } = m, pad = mode === 'spectral' ? 3 : 0, ns = (R + 2 * pad) * os;
  if (!Number.isInteger(R) || R < 1 || !Number.isInteger(os) || os < 1 || !Number.isInteger(m.npupil) || m.npupil < 2 ||
      !Object.values(o).every(v => Number.isFinite(v) && v > 0) || m.noll.length !== m.zcoef.length)
    throw new Error(`${mode}: invalid vectorial optics or sampling grid`);
  const pd = pupilData({ ...m, ...o, roisize: R + 2 * pad, zernike_noll: m.noll }), { N, P, k1, F } = pd;
  const W = aberrationPhase(pd, m.zcoef), nodes = mode === 'gauss_legendre' ? gaussNodes(os) : null;
  const matrixAngles = m.psf_dipole_M_angle?.map(M => M.flat());
  if (nodes && (m.psf_blur_px2 || m.psf_blur_angle_px2)) throw new Error('gauss_legendre pixel_integration does not support image-plane blur');
  if (maps.pupil_amp || m.diattenuation) {
    let before = 0, after = 0;
    for (let c = 0; c < 6; c++) for (let k = 0; k < N * N; k++) {
      const i = 2 * (c * N * N + k), e = P[i] ** 2 + P[i + 1] ** 2;
      const factor = (maps.pupil_amp?.[k] ?? 1) * Math.sqrt(1 + (c % 2 ? -1 : 1) * (m.diattenuation ?? 0));
      before += e; after += e * factor ** 2;
      P[i] *= factor; P[i + 1] *= factor;
    }
    pd.C *= before / after;
  }
  if (maps.pupil_phase) for (let k = 0; k < W.length; k++) W[k] += 2 * Math.PI * maps.pupil_phase[k];
  const spectral = nodes ? null : spectralPixels(R, os, pad, 2 * o.NA * o.pixel_um / o.wavelength_um);
  if (nodes) for (let s = 0; s < ns; s++) for (let i = 0; i < N; i++) {
    const phase = (Math.floor(s / os) - (R - 1) / 2 + nodes[s % os].x) * o.pixel_um * k1[i];
    F[2 * (s * N + i)] = Math.cos(phase);
    F[2 * (s * N + i) + 1] = Math.sin(phase);
  }
  return { roisize: R, z_range: m.z_range_um ?? [0, 1], feature: mode, n_angles: matrixAngles?.length ?? m.psf_blur_angle_px2?.length, render(x, y, z, angle) {
    const M = angleTerm(matrixAngles, angle) ?? m.psf_dipole_M?.flat(), w = m.psf_dipole_w;
    const blur = angleTerm(m.psf_blur_angle_px2, angle) ?? m.psf_blur_px2;
    const fields: Float64Array[] = [];
    const nn = N * N, ph = new Float64Array(2 * nn), density = new Float64Array(ns * ns);
    for (let k = 0; k < nn; k++) {
      const a = W[k] - o.pixel_um * (x * k1[k % N] + y * k1[Math.floor(k / N)]) - pd.kzImm[k] * m.focus_um + pd.kzMed[2 * k] * z;
      const amp = Math.exp(-pd.kzMed[2 * k + 1] * z);
      ph[2 * k] = amp * Math.cos(a); ph[2 * k + 1] = amp * Math.sin(a);
    }
    for (let c = 0; c < 6; c++) {
      const pupil = new Float64Array(2 * nn);
      for (let k = 0; k < nn; k++) {
        const a = 2 * k, h = 2 * c * nn + a;
        pupil[a] = P[h] * ph[a] - P[h + 1] * ph[a + 1];
        pupil[a + 1] = P[h] * ph[a + 1] + P[h + 1] * ph[a];
      }
      const field = transformRows(transformRows(pupil, N, N, F), ns, N, F);
      const weight = M ? M[Math.floor(c / 2) * 4] : (w?.[Math.floor(c / 2)] ?? 1) / 3;
      for (let k = 0; k < density.length; k++)
        density[k] += (field[2 * k] ** 2 + field[2 * k + 1] ** 2) * weight * pd.C;
      if (M) {
        for (let b = c % 2; b < c; b += 2) {
          const i = Math.floor(b / 2), j = Math.floor(c / 2), cross = (M[3 * i + j] + M[3 * j + i]) * pd.C;
          for (let k = 0; k < density.length; k++)
            density[k] += cross * (field[2 * k] * fields[b][2 * k] + field[2 * k + 1] * fields[b][2 * k + 1]);
        }
        fields.push(field);
      }
    }
    if (spectral) return spectral(density, blur);
    const out = new Float32Array(R * R);
    for (let y = 0; y < ns; y++) for (let x = 0; x < ns; x++)
      out[Math.floor(y / os) * R + Math.floor(x / os)] += nodes![y % os].w * nodes![x % os].w * density[y * ns + x];
    return out;
  } };
}
