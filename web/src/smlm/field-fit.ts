import type { VectorialConfig } from '../psf/vectorial';
import { KnotEngine } from '../gpu/knot_engine';
import { LocResult, SplineLocalizer } from '../gpu/spline_lm';
import { StopToken } from '../core/task';
import type { KnotGridFull } from './insitu';
import { PupilField, FieldMethod, fieldPupil, NAT_MODES, CONSTANTS, FIELD_TERMS } from './field-model';

function groups(model: PupilField, xs: ArrayLike<number>, ys: ArrayLike<number>) {
  const { cx, cy, sx, sy } = model.norm, t = model.tile;
  const nx = Math.ceil(2 * sx / t), ny = Math.ceil(2 * sy / t);
  const tiles = new Map<number, { x: number; y: number; ids: number[] }>();
  for (let i = 0; i < xs.length; i++) {
    const ix = Math.max(0, Math.min(nx - 1, Math.floor((xs[i] - cx + sx) / t)));
    const iy = Math.max(0, Math.min(ny - 1, Math.floor((ys[i] - cy + sy) / t)));
    const id = iy * nx + ix;
    if (!tiles.has(id)) tiles.set(id, { x: cx - sx + (ix + .5) * t, y: cy - sy + (iy + .5) * t, ids: [] });
    tiles.get(id)!.ids.push(i);
  }
  return tiles;
}

function subset(rois: Float32Array, ids: number[], size: number) {
  const out = new Float32Array(ids.length * size);
  ids.forEach((id, i) => out.set(rois.subarray(id * size, (id + 1) * size), i * size));
  return out;
}

/** Evaluate each spot with its tile pupil and restore the detection order. */
export async function localizePupilField(dev: GPUDevice, model: PupilField, cfg: VectorialConfig, base: number[],
  grid: KnotGridFull, focus: number, rois: Float32Array, xs: ArrayLike<number>, ys: ArrayLike<number>, variance: number,
  tok: StopToken, cache: Map<number, SplineLocalizer>) {
  const n = xs.length, result: LocResult = {
    theta: new Float32Array(n * 5), crlb: new Float32Array(n * 5),
    nll: new Float32Array(n), chi2: new Float32Array(n), deviance: new Float32Array(n)
  };
  for (const [id, { x, y, ids }] of groups(model, xs, ys)) {
    tok.check();
    if (!cache.has(id)) {
      // Limit resident GPU knot grids on wide-field movies.
      if (cache.size >= 16) {
        const oldest = cache.keys().next().value!;
        cache.get(oldest)!.destroy();
        cache.delete(oldest);
      }
      const eng = new KnotEngine(dev, cfg, grid, focus);
      try {
        eng.setZernike(fieldPupil(model, cfg.zernike_noll, base, x, y));
        cache.set(id, new SplineLocalizer(dev, grid, await eng.knots()));
      } finally { eng.destroy(); }
    }
    const r = await cache.get(id)!.localize(subset(rois, ids, grid.roisize ** 2), ids.length, { var: variance });
    for (const key of Object.keys(result) as (keyof LocResult)[]) {
      const width = r[key].length / ids.length;
      ids.forEach((idx, i) => result[key].set(r[key].subarray(i * width, (i + 1) * width), idx * width));
    }
  }
  return result;
}

/** Alternate local LM and shared optical updates on the browser's pixel spline. */
export async function fitPupilField(dev: GPUDevice, cfg: VectorialConfig, base: number[], grid: KnotGridFull, focus: number,
  rois: Float32Array, xs: number[], ys: number[], w: number, h: number, variance: number, tok: StopToken,
  progress: (i: number) => void, iters = 10, method: FieldMethod = 'nat') {
  const sx = Math.max(1, (w - 1) / 2), sy = Math.max(1, (h - 1) / 2);
  const count = method === 'nat' ? 13 : 25;
  const model: PupilField = { method, parameters: Array(count).fill(0), norm: { cx: (w - 1) / 2, cy: (h - 1) / 2, sx, sy }, tile: Math.min(64, 2 * Math.min(sx, sy) / 3) };
  let offset = 0;
  NAT_MODES.forEach((j, i) => {
    model.parameters[method === 'nat' ? CONSTANTS[i] : offset] = base[cfg.zernike_noll.indexOf(j)];
    offset += FIELD_TERMS[i].length;
  });
  const units = model.parameters.map((_, k) => model.parameters.map((__, j) => Number(j === k)));
  const zero = base.map(() => 0);
  const tiles = Array.from(groups(model, xs, ys).values(), ({ x, y, ids }) => {
    if (method === 'independent') {
      // The Python fit_field_knots training pupil is evaluated at the mean spot position.
      x = ids.reduce((s, i) => s + xs[i], 0) / ids.length;
      y = ids.reduce((s, i) => s + ys[i], 0) / ids.length;
    }
    return {
      x, y, n: ids.length, data: subset(rois, ids, grid.roisize ** 2), theta: undefined as Float32Array | undefined,
      basis: units.map(gamma => fieldPupil({ ...model, parameters: gamma }, cfg.zernike_noll, zero, x, y))
    };
  });
  const eng = new KnotEngine(dev, cfg, grid, focus);
  const loc = new SplineLocalizer(dev, grid, null);
  const size = cfg.zernike_noll.length * 4;
  const rb = dev.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const adj = dev.createBuffer({ size: grid.nz * grid.nu * grid.nv * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const m = new Float64Array(count), v = new Float64Array(count);
  try {
    for (let it = 0; it < iters; it++) {
      const grad = new Float64Array(count);
      for (const t of tiles) {
        tok.check();
        eng.setZernike(fieldPupil(model, cfg.zernike_noll, base, t.x, t.y));
        eng.build(loc.knotBuffer);
        t.theta = (await loc.localize(t.data, t.n, { var: variance, theta0: t.theta, iters: it ? 5 : 30 })).theta;
        const { gK } = await loc.knotGradient(t.data, t.n, t.theta, variance);
        dev.queue.writeBuffer(adj, 0, gK.buffer, gK.byteOffset, gK.byteLength);
        eng.vjp(adj, rb);
        await rb.mapAsync(GPUMapMode.READ);
        try {
          const gz = new Float32Array(rb.getMappedRange());
          t.basis.forEach((row, k) => { grad[k] += row.reduce((s, c, j) => s + c * gz[j], 0); });
        } finally { rb.unmap(); }
      }
      for (let k = 0; k < count; k++) {
        const g = grad[k] / xs.length;
        m[k] = .9 * m[k] + .1 * g; v[k] = .999 * v[k] + .001 * g * g;
        model.parameters[k] -= .002 * (m[k] / (1 - .9 ** (it + 1))) / (Math.sqrt(v[k] / (1 - .999 ** (it + 1))) + 1e-12);
      }
      progress(it + 1);
      if (tok.stopEarly) break;
    }
    return model;
  } finally { adj.destroy(); rb.destroy(); loc.destroy(); eng.destroy(); }
}
