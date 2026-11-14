// Super-resolution reconstruction of localizations: 2D histogram (or Gaussian-blurred histogram) on a grid of
// pixel_nm, optionally coloured by z (mean z per pixel, hue from blue (low) to red (high)).
import { NCOL } from '../smlm/movie';

export interface ReconOptions {
  pixel_nm: number;        // render pixel size
  blur_px: number;         // Gaussian sigma in render pixels (0 = plain histogram)
  colorZ: boolean;
  zlo: number; zhi: number; // colour range (um)
  maxCrlbNm: number;       // drop localizations with crlb_x > this (0 = keep all)
}

export class Recon {
  private chunks: Float32Array[] = [];
  n = 0;
  constructor(public H: number, public W: number, public pixel_um: number) {}
  add(rows: Float32Array) { this.chunks.push(rows); this.n += rows.length / NCOL; }
  rows(): Float32Array {
    const out = new Float32Array(this.n * NCOL);
    let o = 0; for (const c of this.chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  render(cv: HTMLCanvasElement, opt: ReconOptions): { w: number; h: number; shown: number } {
    const sc = (this.pixel_um * 1000) / opt.pixel_nm;
    const w = Math.max(1, Math.min(4096, Math.ceil(this.W * sc))), h = Math.max(1, Math.min(4096, Math.ceil(this.H * sc)));
    let cnt = new Float32Array(w * h), zs = new Float32Array(w * h);
    let shown = 0;
    const maxC = opt.maxCrlbNm > 0 ? opt.maxCrlbNm / (this.pixel_um * 1000) : Infinity;
    for (const c of this.chunks) {
      for (let o = 0; o < c.length; o += NCOL) {
        if (c[o + 6] > maxC) continue;
        // image px (pixel centres at integers) -> render pixel; the image spans -0.5 .. W - 0.5
        const x = Math.floor((c[o + 1] + 0.5) * sc), y = Math.floor((c[o + 2] + 0.5) * sc);
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        cnt[y * w + x] += 1; zs[y * w + x] += c[o + 3]; shown++;
      }
    }
    if (opt.blur_px > 0) { cnt = blur(cnt, w, h, opt.blur_px); zs = blur(zs, w, h, opt.blur_px); }
    // brightness: 99.5 % quantile of the non-empty pixels, gamma 0.6
    const nz: number[] = [];
    for (let i = 0; i < cnt.length; i += Math.max(1, Math.floor(cnt.length / 200000))) if (cnt[i] > 0) nz.push(cnt[i]);
    nz.sort((a, b) => a - b);
    const vmax = nz.length ? nz[Math.floor(0.995 * (nz.length - 1))] : 1;
    cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d')!;
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const v = Math.min(1, Math.pow(cnt[i] / vmax, 0.6));
      let r: number, g: number, b: number;
      if (opt.colorZ && cnt[i] > 0) {
        const t = Math.max(0, Math.min(1, (zs[i] / cnt[i] - opt.zlo) / Math.max(1e-9, opt.zhi - opt.zlo)));
        [r, g, b] = hue(t);
      } else { [r, g, b] = hot(v); }
      const k = opt.colorZ ? v : 1;
      img.data[4 * i] = 255 * r * k; img.data[4 * i + 1] = 255 * g * k; img.data[4 * i + 2] = 255 * b * k; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return { w, h, shown };
  }
}

function blur(a: Float32Array, w: number, h: number, s: number): Float32Array {
  const r = Math.ceil(3 * s), k = new Float32Array(2 * r + 1);
  let sum = 0; for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * s * s)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const t = new Float32Array(a.length), o = new Float32Array(a.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let j = -r; j <= r; j++) { const xx = x + j; if (xx >= 0 && xx < w) v += k[j + r] * a[y * w + xx]; }
    t[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let j = -r; j <= r; j++) { const yy = y + j; if (yy >= 0 && yy < h) v += k[j + r] * t[yy * w + x]; }
    o[y * w + x] = v;
  }
  return o;
}

/** blue (0) -> cyan -> green -> yellow -> red (1) */
function hue(t: number): [number, number, number] {
  const hh = (1 - t) * 240 / 60, x = 1 - Math.abs((hh % 2) - 1);
  if (hh < 1) return [1, x, 0]; if (hh < 2) return [x, 1, 0]; if (hh < 3) return [0, 1, x]; return [0, x, 1];
}
function hot(v: number): [number, number, number] {
  return [Math.min(1, 3 * v), Math.max(0, Math.min(1, 3 * v - 1)), Math.max(0, Math.min(1, 3 * v - 2))];
}
