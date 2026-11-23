import { VecEngine } from '../gpu/ops';
import type { VectorialModel } from '../psf/zstack';
import type { MovieObserver } from './movie';

export const profileHalfRangeUm = 0.8;
const radius = 4;
const side = 2 * radius + 1;

// Register bright local maxima by subpixel peak position. Favor peaks with a
// quiet surround for pupil estimation; the subsequent joint fit retains all beads.
function centeredPatch(im: Float32Array, w: number, peak: number) {
  const shift = (j: number) => {
    const a = im[peak - j],
      b = im[peak],
      c = im[peak + j];
    return Math.max(-0.5, Math.min(0.5, (a - c) / (2 * Math.min(-1e-12, a + c - 2 * b))));
  };
  const dx = shift(1),
    dy = shift(w);
  const origin = peak + Math.floor(dx) + Math.floor(dy) * w,
    fx = dx - Math.floor(dx),
    fy = dy - Math.floor(dy);
  const patch = new Float64Array(side * side);
  for (let y = -radius; y <= radius; y++)
    for (let x = -radius; x <= radius; x++) {
      const j = origin + y * w + x;
      patch[(y + radius) * side + x + radius] =
        (1 - fy) * ((1 - fx) * im[j] + fx * im[j + 1]) + fy * ((1 - fx) * im[j + w] + fx * im[j + w + 1]);
    }
  return patch;
}
function normalize(p: Float64Array) {
  const mean = p.reduce((a, b) => a + b, 0) / p.length;
  const norm = Math.sqrt(p.reduce((s, x) => s + (x - mean) ** 2, 0));
  for (let i = 0; i < p.length; i++) p[i] = (p[i] - mean) / Math.max(1e-12, norm);
  return p;
}
const spectralSide = 32;
const frequencies = Array.from({ length: 7 * 15 }, (_, i) => [(i % 15) - 7, 1 + Math.floor(i / 15)]).filter(
  ([x, y]) => Math.hypot(x, y) >= 2 && Math.hypot(x, y) <= 7,
);
const cos = Array.from({ length: 15 }, (_, k) =>
  Float64Array.from({ length: 32 }, (_, i) => Math.cos(((k - 7) * 2 * Math.PI * i) / 32)),
);
const sin = Array.from({ length: 15 }, (_, k) =>
  Float64Array.from({ length: 32 }, (_, i) => Math.sin(((k - 7) * 2 * Math.PI * i) / 32)),
);
function powers(image: Float32Array, w: number, windowed: boolean) {
  const n = windowed ? spectralSide : w,
    output = new Float64Array(frequencies.length);
  const tilesX = Math.floor(w / n),
    tilesY = Math.floor(image.length / w / n);
  const taper = Float64Array.from({ length: n }, (_, i) =>
    windowed ? Math.sin((Math.PI * (i + 0.5)) / n) ** 2 : 1,
  );
  for (let ty = 0; ty < tilesY; ty++)
    for (let tx = 0; tx < tilesX; tx++) {
      const ox = tx * n,
        oy = ty * n;
      let mean = 0;
      if (windowed) {
        for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) mean += image[(oy + y) * w + ox + x];
        mean /= n * n;
      }
      const re = new Float64Array(15 * n),
        im = new Float64Array(15 * n);
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
          const v = (image[(oy + y) * w + ox + x] - mean) * taper[y] * taper[x];
          for (let k = 0; k < 15; k++) {
            re[k * n + y] += v * cos[k][x];
            im[k * n + y] += v * sin[k][x];
          }
        }
      frequencies.forEach(([x, y], j) => {
        let a = 0,
          b = 0;
        for (let i = 0; i < n; i++) {
          const c = cos[y + 7][i],
            s = sin[y + 7][i],
            k = (x + 7) * n + i;
          a += re[k] * c - im[k] * s;
          b += re[k] * s + im[k] * c;
        }
        // Subtract the expected shot-noise floor for Hann-windowed photon counts.
        output[j] += a * a + b * b - (windowed ? mean * ((n * 3) / 8) ** 2 : 0);
      });
    }
  const sorted = Array.from(output).sort((a, b) => a - b);
  const cap = Math.max(0, sorted[Math.floor(0.96 * sorted.length)]) * 1.5;
  for (let j = 0; j < output.length; j++) output[j] = Math.max(0, Math.min(cap, output[j]));
  return normalize(output);
}

export function observedShape(im: Float32Array, w: number, h: number) {
  const peaks: { index: number; score: number }[] = [];
  for (let y = 6; y < h - 6; y++)
    for (let x = 6; x < w - 6; x++) {
      const i = y * w + x,
        height = im[i];
      if (![-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1].every((d) => im[i + d] < height)) continue;
      const ring: number[] = [];
      for (let d = -4; d <= 4; d++) {
        ring.push(im[i - 4 * w + d], im[i + 4 * w + d]);
        if (Math.abs(d) < 4) ring.push(im[i + d * w - 4], im[i + d * w + 4]);
      }
      ring.sort((a, b) => a - b);
      const contrast = height - ring[16];
      if (contrast > 5 * Math.sqrt(Math.max(1, ring[16])))
        peaks.push({ index: i, score: contrast / (Math.sqrt(Math.max(1, height)) + ring[28] - ring[4]) });
    }
  if (peaks.length < 6) return;
  peaks.sort((a, b) => b.score - a.score);
  const patches = peaks
    .slice(0, Math.min(120, Math.max(6, Math.floor(peaks.length / 12))))
    .map((p) => normalize(centeredPatch(im, w, p.index)));
  const shape = new Float64Array(side * side);
  for (let j = 0; j < shape.length; j++) {
    const values = patches.map((p) => p[j]).sort((a, b) => a - b);
    shape[j] = values[Math.floor(values.length / 2)];
  }
  return { patch: normalize(shape), spectrum: w >= 32 && h >= 32 ? powers(im, w, true) : undefined };
}

export async function fitMovieShapes(
  dev: GPUDevice,
  model: VectorialModel,
  diameterNm: number,
  observations: NonNullable<ReturnType<typeof observedShape>>[],
  observe?: MovieObserver,
) {
  const R = 21,
    height = diameterNm / 2000,
    node = height * Math.sqrt(3 / 5);
  const nodes = node
    ? [
        [node, 0, 0],
        [-node, 0, 0],
        [0, node, 0],
        [0, -node, 0],
        [0, 0, node],
        [0, 0, -node],
      ]
    : [[0, 0, 0]];
  const engine = new VecEngine(
    dev,
    {
      ...model.optics,
      roisize: R,
      npupil: model.npupil,
      oversample: model.oversample,
      zernike_noll: model.noll,
    },
    [],
  );
  // Use the same finite-bead model as the photometry fit at each trial focus.
  const focus = Array.from({ length: 33 }, (_, i) => (i - 16) * profileHalfRangeUm / 16);
  const images = focus.flatMap((z) =>
    nodes.map(([x, y, dz]) => ({
      x: x / model.optics.pixel_um,
      y: y / model.optics.pixel_um,
      ze: height + dz,
      f: height + z,
      I: 1,
      bg: 0,
    })),
  );
  const best = Array(model.noll.length).fill(0);
  let cost = Infinity,
    focusTrace: number[] = [],
    trial = 0;
  const steps = [0.08, 0.04, 0.02],
    total = 2 + 2 * (steps.length + 1) * best.length;
  try {
    engine.setImages(images);
    async function score(coef: number[]) {
      engine.setZernike(coef);
      await engine.run(images.length, 'psf');
      const bank = await engine.readPsf(images.length);
      const templates = focus.map((_, z) => {
        const psf = new Float32Array(R * R);
        for (let n = 0; n < nodes.length; n++)
          for (let i = 0; i < psf.length; i++) psf[i] += bank[(z * nodes.length + n) * R * R + i];
        let peak = (R * R - 1) / 2;
        for (let y = 6; y < R - 6; y++)
          for (let x = 6; x < R - 6; x++) if (psf[y * R + x] > psf[peak]) peak = y * R + x;
        return { patch: normalize(centeredPatch(psf, R, peak)), spectrum: powers(psf, R, false) };
      });
      const zs: number[] = [];
      let residual = 0;
      for (const obs of observations) {
        let min = Infinity,
          plane = 0;
        templates.forEach((p, z) => {
          const error =
            p.patch.reduce((s, v, j) => s + (v - obs.patch[j]) ** 2, 0) +
            (obs.spectrum ? 3 * p.spectrum.reduce((s, v, j) => s + (v - obs.spectrum![j]) ** 2, 0) : 0);
          if (error < min) {
            min = error;
            plane = z;
          }
        });
        zs.push(focus[plane]);
        residual += min;
      }
      residual /= observations.length;
      if (residual < cost) {
        cost = residual;
        best.splice(0, best.length, ...coef);
        focusTrace = zs;
      }
      model.zcoef = best.slice();
      await observe?.(dev, {
        stage: 'profile',
        index: ++trial,
        total,
        model,
        emitterHeightUm: height,
        focusUm: height + focusTrace[0],
        score: cost,
      });
    }
    await score(best.slice());
    for (const step of steps)
      for (let mode = 0; mode < best.length; mode++) {
        const center = best.slice();
        for (const sign of [-1, 1]) {
          const candidate = center.slice();
          candidate[mode] += sign * step;
          await score(candidate);
        }
      }
    // Opposite focus can give nearly the same image when the even pupil modes
    // reverse sign. Score the conjugate solution before the final fine search.
    await score(best.map((value, i) => ([5, 6, 11].includes(model.noll[i]) ? -value : value)));
    for (let mode = 0; mode < best.length; mode++) {
      const center = best.slice();
      for (const sign of [-1, 1]) {
        const candidate = center.slice();
        candidate[mode] += sign * 0.01;
        await score(candidate);
      }
    }
    return focusTrace;
  } finally {
    engine.destroy();
  }
}
