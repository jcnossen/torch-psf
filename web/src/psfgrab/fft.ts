// Radix-2 complex FFT (in place, Float64Array re / im) and the 2D transform of a square power-of-two image.
// Pure TypeScript, no DOM: used by the PSF grab analysis worker and by the node unit checks (test/psfgrab_node.ts).

const plans = new Map<number, { rev: Uint32Array; cos: Float64Array; sin: Float64Array }>();

export const isPow2 = (n: number): boolean => n > 0 && (n & (n - 1)) === 0;

function plan(n: number) {
  let p = plans.get(n);
  if (p) return p;
  if (!isPow2(n)) throw new Error(`fft: length ${n} is not a power of two`);
  const bits = Math.round(Math.log2(n));
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) { cos[k] = Math.cos(-2 * Math.PI * k / n); sin[k] = Math.sin(-2 * Math.PI * k / n); }
  p = { rev, cos, sin };
  plans.set(n, p);
  return p;
}

/** Forward DFT X[k] = sum_j x[j] exp(-2 pi i j k / n) of re / im[off .. off + n), in place (inverse: conjugate
 *  exponent, no 1/n scaling). */
export function fft1d(re: Float64Array, im: Float64Array, n: number, off = 0, inverse = false): void {
  const { rev, cos, sin } = plan(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[off + i]; re[off + i] = re[off + j]; re[off + j] = t;
      t = im[off + i]; im[off + i] = im[off + j]; im[off + j] = t;
    }
  }
  const sgn = inverse ? -1 : 1;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let s = 0; s < n; s += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sgn * sin[k * step];
        const a = off + s + k, b = a + half;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

/** 2D forward FFT of an n x n complex image (row major), in place: rows, then columns (via a scratch column). */
export function fft2d(re: Float64Array, im: Float64Array, n: number): void {
  for (let y = 0; y < n; y++) fft1d(re, im, n, y * n);
  const cr = new Float64Array(n), ci = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { cr[y] = re[y * n + x]; ci[y] = im[y * n + x]; }
    fft1d(cr, ci, n);
    for (let y = 0; y < n; y++) { re[y * n + x] = cr[y]; im[y * n + x] = ci[y]; }
  }
}

/** Naive O(n^2) DFT (reference for the unit checks). */
export function dftNaive(re: Float64Array, im: Float64Array): [Float64Array, Float64Array] {
  const n = re.length, or = new Float64Array(n), oi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let j = 0; j < n; j++) {
      const a = -2 * Math.PI * j * k / n, c = Math.cos(a), s = Math.sin(a);
      sr += re[j] * c - im[j] * s; si += re[j] * s + im[j] * c;
    }
    or[k] = sr; oi[k] = si;
  }
  return [or, oi];
}
