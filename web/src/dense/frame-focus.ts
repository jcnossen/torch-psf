import { fft2d } from '../psfgrab/fft';
import { VecEngine } from '../gpu/ops';
import type { VectorialModel } from '../psf/zstack';

const grid = 32,
  tile = 64,
  step = 0.005;
const frequencies: [number, number][] = [];
for (let y = 0; y <= 10; y++)
  for (let x = -10; x <= 10; x++)
    if ((y > 0 || x > 0) && Math.hypot(x, y) >= 3 && Math.hypot(x, y) <= 10) frequencies.push([x, y]);
const window = Float64Array.from(
  { length: tile },
  (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (tile - 1)),
);
const noise = window.reduce((s, v) => s + v * v, 0) ** 2;
const wrap = (v: number, n: number) => (v + n) % n;
function median(a: number[]) {
  a.sort((x, y) => x - y);
  const i = a.length >> 1;
  return a.length % 2 ? a[i] : (a[i - 1] + a[i]) / 2;
}

// Sixteen Hann-windowed camera patches estimate orientation-dependent image power.
export function framePower(im: Float32Array, w: number, h: number) {
  const out = new Float64Array(frequencies.length);
  const re = new Float64Array(tile * tile),
    imag = re.slice();
  let shot = 0;
  for (let iy = 0; iy < 4; iy++)
    for (let ix = 0; ix < 4; ix++) {
      const top = Math.floor((iy * (h - tile)) / 3),
        left = Math.floor((ix * (w - tile)) / 3);
      let mean = 0;
      for (let y = 0; y < tile; y++) for (let x = 0; x < tile; x++) mean += im[(y + top) * w + x + left];
      mean /= tile * tile;
      shot += mean * noise;
      for (let y = 0; y < tile; y++)
        for (let x = 0; x < tile; x++)
          re[y * tile + x] = (im[(y + top) * w + x + left] - mean) * window[x] * window[y];
      imag.fill(0);
      fft2d(re, imag, tile);
      frequencies.forEach(([x, y], j) => {
        const k = 2 * y * tile + wrap(2 * x, tile);
        out[j] += re[k] ** 2 + imag[k] ** 2;
      });
    }
  return out.map((v) => Math.log(Math.max(1, (v - shot) / 16)));
}

export async function exposureFocus(
  dev: GPUDevice,
  model: VectorialModel,
  diameter: number,
  rows: Float64Array[],
  period: number,
  centers: number[],
  focus: number[],
) {
  const low = Math.min(...focus) - 0.24,
    high = Math.max(...focus) + 0.24,
    height = diameter / 2000;
  const nz = Math.round((high - low) / step) + 1,
    nf = frequencies.length;
  const eng = new VecEngine(
    dev,
    {
      ...model.optics,
      roisize: 33,
      npupil: model.npupil,
      oversample: model.oversample,
      zernike_noll: model.noll,
    },
    [],
  );
  const templates: Float64Array[] = [];
  try {
    eng.setZernike(model.zcoef);
    for (let start = 0; start < nz; start += 32) {
      const n = Math.min(32, nz - start);
      eng.setImages(
        Array.from({ length: n }, (_, j) => ({
          x: 0,
          y: 0,
          ze: height,
          f: height + low + (start + j) * step,
          I: 1,
          bg: 0,
        })),
      );
      await eng.run(n, 'psf');
      const psfs = await eng.readPsf(n);
      for (let j = 0; j < n; j++) {
        const re = new Float64Array(grid * grid),
          imag = re.slice();
        for (let y = 0; y < 33; y++)
          for (let x = 0; x < 33; x++)
            re[wrap(y - 16, grid) * grid + wrap(x - 16, grid)] += psfs[(j * 33 + y) * 33 + x];
        fft2d(re, imag, grid);
        templates.push(
          Float64Array.from(frequencies, ([x, y]) => {
            const k = y * grid + wrap(x, grid);
            return Math.log(Math.max(1e-5, re[k] ** 2 + imag[k] ** 2));
          }),
        );
      }
    }
  } finally {
    eng.destroy();
  }
  const clip = (k: number) => Math.min(nz - 1, Math.max(0, k));
  let j = 0;
  const idx = rows.map((_, t) => {
    while (j + 1 < centers.length && centers[j + 1] <= t) j++;
    const k = Math.min(j + 1, centers.length - 1);
    const f = Math.max(0, Math.min(1, (t - centers[j]) / Math.max(1, centers[k] - centers[j])));
    return clip(Math.round((focus[j] + f * (focus[k] - focus[j]) - low) / step));
  });
  const diff = new Float64Array(nf),
    costs = new Float64Array(rows.length);
  let constantCost = 0;
  for (let iter = 0; iter < 6; iter++) {
    for (let phase = 0; phase < period; phase++) {
      const group: number[] = [];
      for (let t = phase; t < rows.length; t += period) group.push(t);
      const bias = Float64Array.from({ length: nf }, (_, k) =>
        median(group.map((t) => rows[t][k] - templates[idx[t]][k])),
      );
      const constant = new Float64Array(nz);
      for (const t of group) {
        let best = Infinity;
        for (let z = 0; z < nz; z++) {
          let mean = 0;
          for (let k = 0; k < nf; k++) {
            diff[k] = rows[t][k] - bias[k] - templates[z][k];
            mean += diff[k];
          }
          mean /= nf;
          let cost = 0;
          for (let k = 0; k < nf; k++) cost += Math.min(1, (diff[k] - mean) ** 2);
          if (iter === 5) constant[z] += cost / nf;
          if (cost < best) {
            best = cost;
            idx[t] = z;
          }
        }
        costs[t] = best / nf;
      }
      if (iter === 5) constantCost += Math.min(...constant);
    }
    const offset = median(
      centers.map((c, k) => {
        const start = c - (period - 1) / 2;
        let mean = 0;
        for (let i = 0; i < period; i++) mean += low + idx[start + i] * step;
        return focus[k] - mean / period;
      }),
    );
    for (let t = 0; t < idx.length; t++) idx[t] = clip(idx[t] + Math.round(offset / step));
  }
  // Refuse low-information data or traces that run out of the optical table.
  const typicalCost = median(Array.from(costs));
  const fittedCost = costs.reduce((a,b)=>a+b,0);
  if (
    typicalCost > 0.25 || (typicalCost > 0.12 && fittedCost > 0.8 * constantCost) ||
    idx.filter((i) => i === 0 || i === nz - 1).length > 0.05 * idx.length
  )
    return;
  return idx.map((i) => low + i * step);
}
