// Pixel grabbing for PSF grab: a region of a drawable (video element, ImageBitmap, canvas) drawn to a canvas and read
// back with getImageData as planar R, G, B bytes; and the same crop / box-downscale for float frames (TIFF input).

export interface Region { x: number; y: number; w: number; h: number }     // source pixels
export interface Planes { w: number; h: number; planes: (Uint8Array | Float32Array)[] }

/** The record / analysis region: a size x size square (or the full frame when size is 0) centred at (cx, cy), shifted
 *  to lie inside the W x H frame, with w and h multiples of `mult` (the downscale factor). */
export function regionAt(W: number, H: number, cx: number, cy: number, size: number, mult = 1): Region {
  let w = size > 0 ? Math.min(size, W) : W, h = size > 0 ? Math.min(size, H) : H;
  w -= w % mult; h -= h % mult;
  const x = Math.max(0, Math.min(W - w, Math.round(cx - w / 2)));
  const y = Math.max(0, Math.min(H - h, Math.round(cy - h / 2)));
  return { x, y, w, h };
}

let canvas: HTMLCanvasElement | null = null;
let ctx: CanvasRenderingContext2D | null = null;

/** region of `src` -> ow x oh pixels (smoothed when downscaling) -> planar R, G, B uint8. */
export function grabDrawable(src: CanvasImageSource, r: Region, ow: number, oh: number): Planes {
  if (!canvas) {
    canvas = document.createElement('canvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: false });
    if (!ctx) throw new Error('no 2D canvas');
  }
  const c = ctx!;
  if (canvas.width !== ow || canvas.height !== oh) { canvas.width = ow; canvas.height = oh; }
  const scaled = ow !== r.w || oh !== r.h;
  c.imageSmoothingEnabled = scaled;
  if (scaled) c.imageSmoothingQuality = 'high';
  c.drawImage(src, r.x, r.y, r.w, r.h, 0, 0, ow, oh);
  const d = c.getImageData(0, 0, ow, oh).data;
  return { w: ow, h: oh, planes: splitRGBA(d, ow * oh) };
}

export function splitRGBA(d: ArrayLike<number>, n: number): Uint8Array[] {
  const R = new Uint8Array(n), G = new Uint8Array(n), B = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { R[i] = d[j]; G[i] = d[j + 1]; B[i] = d[j + 2]; }
  return [R, G, B];
}

/** Crop rows / columns of a W-wide float frame and average f x f blocks. */
export function cropBin(src: Float32Array, W: number, r: Region, f: number): Float32Array {
  const ow = Math.floor(r.w / f), oh = Math.floor(r.h / f), out = new Float32Array(ow * oh), s = 1 / (f * f);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      let a = 0;
      for (let dy = 0; dy < f; dy++) { const row = (r.y + y * f + dy) * W + r.x + x * f; for (let dx = 0; dx < f; dx++) a += src[row + dx]; }
      out[y * ow + x] = a * s;
    }
  }
  return out;
}

/** Float plane -> uint8 (rounded, clamped). */
export function toU8(p: Float32Array): Uint8Array {
  const o = new Uint8Array(p.length);
  for (let i = 0; i < p.length; i++) o[i] = p[i] < 0 ? 0 : p[i] > 255 ? 255 : Math.round(p[i]);
  return o;
}
