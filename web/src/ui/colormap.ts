// Colour maps (viridis-like sequential, blue-white-red diverging) and canvas image helpers.

const VIR = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121],
  [109, 205, 89], [180, 222, 44], [253, 231, 37]];

export function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t)) * (VIR.length - 1);
  const i = Math.min(VIR.length - 2, Math.floor(t)), f = t - i;
  const a = VIR[i], b = VIR[i + 1];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}

export function diverging(t: number): [number, number, number] {   // t in [-1, 1]
  t = Math.max(-1, Math.min(1, t));
  return t < 0 ? [255 * (1 + t) + 33 * -t, 255 * (1 + t) + 102 * -t, 255 * (1 + t) + 172 * -t]
    : [255 * (1 - t) + 178 * t, 255 * (1 - t) + 24 * t, 255 * (1 - t) + 43 * t];
}

/** RGBA pixels of a 2D array (h x w) with a log10 scale over `decades` below vmax, or linear when decades = 0. */
export function toRgba(a: ArrayLike<number>, w: number, h: number, opts: { vmax?: number; decades?: number; diverging?: boolean; alpha?: boolean; gray?: boolean } = {}): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  let vmax = opts.vmax ?? 0;
  if (!opts.vmax) for (let i = 0; i < w * h; i++) vmax = Math.max(vmax, Math.abs(a[i]));
  vmax = vmax || 1;
  const dec = opts.decades ?? 0;
  for (let i = 0; i < w * h; i++) {
    let c: [number, number, number], t = 1;
    if (opts.diverging) c = diverging(a[i] / vmax);
    else if (dec > 0) { t = 1 + Math.log10(Math.max(a[i] / vmax, 10 ** -dec)) / dec; c = viridis(t); }
    else if (opts.gray) { t = Math.min(1, Math.max(0, a[i] / vmax)); const g = Math.round(255 * Math.sqrt(t)); c = [g, g, g]; }
    else { t = a[i] / vmax; c = viridis(t); }
    out[4 * i] = c[0]; out[4 * i + 1] = c[1]; out[4 * i + 2] = c[2];
    out[4 * i + 3] = opts.alpha ? 255 * Math.min(1, Math.max(0, t) * 1.4) : 255;   // alpha: the low end fades out
  }
  return out;
}

/** Draw a 2D array into a canvas (nearest-neighbour upscaling to the canvas size). */
export function drawArray(cv: HTMLCanvasElement, a: ArrayLike<number>, w: number, h: number, opts: Parameters<typeof toRgba>[3] = {}) {
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  off.getContext('2d')!.putImageData(new ImageData(toRgba(a, w, h, opts), w, h), 0, 0);
  const ctx = cv.getContext('2d')!;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(off, 0, 0, cv.width, cv.height);
}
