// Frame analysis for PSF grab: Hann-windowed power spectrum of a square centre crop per colour channel, radially
// averaged (one ring per integer radius in FFT pixels), and two sharpness numbers. Pure TypeScript (no DOM).
//
// Conventions (also written into the export JSON):
//   power  P(k) = |FFT(w * (x - mean(x)))(k)|^2 / sum(w^2), w the 2D periodic Hann window: white noise of variance
//          s^2 gives P = s^2 on average, so the high-frequency floor of a shot-noise limited image reads as its variance.
//   ring r collects the frequencies with round(|k|) = r, k in FFT pixels; spatial frequency f = r / n cycles per pixel,
//          r = 0 .. n/2 (f = 0 .. 0.5, the Nyquist frequency).
//   profile = log10 of the ring mean of P.
//   hfRatio = sum_{n/4 < r <= n/2} r * ring(r) / sum_{2 <= r <= n/2} r * ring(r) (ring(r) the ring mean, r ~ the
//          ring's area): the share of the non-DC power above half the Nyquist frequency.
//   sharpness = var(L) / mean(I)^2, L the 4-neighbour Laplacian of the channel-mean image I (interior pixels):
//          scale free, larger is sharper.

import { fft2d, isPow2 } from './fft';

/** Periodic Hann window w[i] = 0.5 - 0.5 cos(2 pi i / n). */
export function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
  return w;
}

const windows = new Map<number, { w: Float64Array; norm: number }>();
function windowOf(n: number) {
  let e = windows.get(n);
  if (!e) {
    const w = hann(n);
    let s2 = 0;
    for (let i = 0; i < n; i++) s2 += w[i] * w[i];
    e = { w, norm: s2 * s2 };               // sum over the 2D window w(x) w(y) of w^2
    windows.set(n, e);
  }
  return e;
}

/** Copy an n x n square of a channel (row stride `stride`, top-left x0, y0), mean-subtracted and windowed. */
function windowed(src: ArrayLike<number>, stride: number, x0: number, y0: number, n: number, out: Float64Array) {
  const { w } = windowOf(n);
  let s = 0;
  for (let y = 0; y < n; y++) { const r = (y0 + y) * stride + x0; for (let x = 0; x < n; x++) s += src[r + x]; }
  const m = s / (n * n);
  for (let y = 0; y < n; y++) {
    const r = (y0 + y) * stride + x0, wy = w[y];
    for (let x = 0; x < n; x++) out[y * n + x] = (src[r + x] - m) * wy * w[x];
  }
}

/** Power spectra (n x n, FFT order, normalised as above) of up to two real n x n images with one complex FFT:
 *  z = a + i b, A(k) = (Z(k) + conj Z(-k)) / 2, B(k) = (Z(k) - conj Z(-k)) / 2i. */
export function powerPair(a: Float64Array, b: Float64Array | null, n: number): [Float64Array, Float64Array | null] {
  const re = Float64Array.from(a), im = b ? Float64Array.from(b) : new Float64Array(n * n);
  fft2d(re, im, n);
  const norm = windowOf(n).norm;
  const pa = new Float64Array(n * n), pb = b ? new Float64Array(n * n) : null;
  for (let ky = 0; ky < n; ky++) {
    const my = ((n - ky) % n) * n;
    for (let kx = 0; kx < n; kx++) {
      const i = ky * n + kx, j = my + ((n - kx) % n);
      const zr = re[i], zi = im[i], cr = re[j], ci = -im[j];       // Z(k), conj Z(-k)
      const ar = zr + cr, ai = zi + ci;
      pa[i] = (ar * ar + ai * ai) / (4 * norm);
      if (pb) { const br = zr - cr, bi = zi - ci; pb[i] = (br * br + bi * bi) / (4 * norm); }
    }
  }
  return [pa, pb];
}

/** Ring means of an n x n power spectrum in FFT order: out[r] for r = 0 .. n/2 (round(|k|) = r). */
export function radialAverage(power: Float64Array, n: number): Float64Array {
  const nb = n / 2 + 1, sum = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let y = 0; y < n; y++) {
    const ky = y < n / 2 ? y : y - n;
    for (let x = 0; x < n; x++) {
      const kx = x < n / 2 ? x : x - n;
      const r = Math.round(Math.sqrt(kx * kx + ky * ky));
      if (r < nb) { sum[r] += power[y * n + x]; cnt[r]++; }
    }
  }
  for (let r = 0; r < nb; r++) sum[r] = cnt[r] ? sum[r] / cnt[r] : 0;
  return sum;
}

/** var(Laplacian) / mean^2 of a w x h image (4-neighbour Laplacian over the interior). */
export function sharpness(img: ArrayLike<number>, w: number, h: number): number {
  if (w < 3 || h < 3) return 0;
  let s = 0, sl = 0, sl2 = 0, nl = 0;
  for (let i = 0; i < w * h; i++) s += img[i];
  const m = s / (w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const l = img[i - 1] + img[i + 1] + img[i - w] + img[i + w] - 4 * img[i];
      sl += l; sl2 += l * l; nl++;
    }
  }
  const v = sl2 / nl - (sl / nl) ** 2;
  return m > 0 ? v / (m * m) : 0;
}

export interface FrameAnalysis {
  n: number;                 // FFT size (square crop side, power of two)
  x0: number; y0: number;    // top-left of the analysed square in the given planes
  profiles: Float32Array[];  // per channel: log10 ring-mean power, length n/2 + 1
  hfRatio: number[];         // per channel
  sharpness: number;         // of the channel mean over the analysed square
  mean: number[];            // per channel mean over the square (0..255 for 8-bit input)
}

/** Largest power of two <= min(w, h, maxN) (at least 8). */
export function fftSizeFor(w: number, h: number, maxN: number): number {
  let n = 8;
  while (n * 2 <= Math.min(w, h, maxN)) n *= 2;
  return n;
}

/** Analyse the centred n x n square of planar channels (each w x h, row major). */
export function analyzePlanes(planes: ArrayLike<number>[], w: number, h: number, maxN = 512): FrameAnalysis {
  const n = fftSizeFor(w, h, maxN);
  if (!isPow2(n) || n > w || n > h) throw new Error(`analyzePlanes: image ${w} x ${h} is smaller than 8 x 8`);
  const x0 = (w - n) >> 1, y0 = (h - n) >> 1;
  const win = planes.map(() => new Float64Array(n * n));
  planes.forEach((p, c) => windowed(p, w, x0, y0, n, win[c]));
  const spectra: Float64Array[] = [];
  for (let c = 0; c < planes.length; c += 2) {
    const [pa, pb] = powerPair(win[c], c + 1 < planes.length ? win[c + 1] : null, n);
    spectra.push(pa); if (pb) spectra.push(pb);
  }
  const lo = 2, mid = n / 4, hi = n / 2;
  const profiles: Float32Array[] = [], hfRatio: number[] = [];
  for (const p of spectra) {
    const ring = radialAverage(p, n);
    const prof = new Float32Array(ring.length);
    for (let r = 0; r < ring.length; r++) prof[r] = Math.log10(Math.max(ring[r], 1e-12));
    profiles.push(prof);
    // ring sums (mean x count ~ 2 pi r) for the high-frequency share
    let all = 0, high = 0;
    for (let r = lo; r <= hi; r++) { const v = ring[r] * r; all += v; if (r > mid) high += v; }
    hfRatio.push(all > 0 ? high / all : 0);
  }
  const lum = new Float64Array(n * n), mean: number[] = [];
  for (const p of planes) {
    let s = 0;
    for (let y = 0; y < n; y++) { const r = (y0 + y) * w + x0; for (let x = 0; x < n; x++) { const v = p[r + x]; lum[y * n + x] += v / planes.length; s += v; } }
    mean.push(s / (n * n));
  }
  return { n, x0, y0, profiles, hfRatio, sharpness: sharpness(lum, n, n), mean };
}

/** Planar channels (Float32Array per channel) from interleaved RGBA bytes. */
export function rgbaToPlanes(rgba: ArrayLike<number>, w: number, h: number): Float32Array[] {
  const r = new Float32Array(w * h), g = new Float32Array(w * h), b = new Float32Array(w * h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) { r[i] = rgba[j]; g[i] = rgba[j + 1]; b[i] = rgba[j + 2]; }
  return [r, g, b];
}
