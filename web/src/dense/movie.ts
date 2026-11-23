import { observedShape, fitMovieShapes } from './movie-shapes';
import { VecEngine } from '../gpu/ops';
import { TiffMovie } from '../io/tiff';
import { BeadFitter, extractRoi, findBeads } from '../fit/beads';
import type { VectorialModel } from '../psf/zstack';
import type { Optics } from '../psf/vectorial';
import { buildPsfZip } from '../psf/makezip';
import { makeFitReport, pupilMaps, volumeSlices } from '../report/fromapp';
import shader from './joint.wgsl?raw';

export interface Settings {
  optics: Optics;
  diameterNm: number;
  stepNm: number;
  offset: number;
  gain: number;
  period?: number;
}
export interface MovieFit {
  width: number;
  height: number;
  xy: number[][];
  intensity: number[][];
  drift: number[][];
  model: VectorialModel;
  background: number[];
  settings: Settings;
  stackCalibrated: boolean;
  boundaryFrames: number;
  focusWindowShifts: number;
}
const R = 13,
  K = 3,
  SIDE = R * K,
  NZ = 25;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
function quantile(a: Float32Array, q: number) {
  const every = Math.max(1, Math.floor(a.length / 20000));
  const sample = Array.from({ length: Math.ceil(a.length / every) }, (_, i) => a[i * every]).sort(
    (x, y) => x - y,
  );
  return sample[Math.floor(q * (sample.length - 1))];
}
function photons(a: Float32Array, s: Settings) {
  for (let i = 0; i < a.length; i++) a[i] = Math.max(0, (a[i] - s.offset) * s.gain);
  return a;
}
export function stackStepNm(description: string) {
  const unit = (u: string) =>
    (
      ({ nm: 1, um: 1000, µm: 1000, μm: 1000, micron: 1000, microns: 1000, mm: 1e6 }) as Record<
        string,
        number
      >
    )[u.toLowerCase()];
  const xml = new DOMParser().parseFromString(description, 'application/xml');
  const pixels = xml.getElementsByTagNameNS('*', 'Pixels')[0];
  if (pixels) {
    const positions = Array.from(pixels.getElementsByTagNameNS('*', 'Plane'))
      .filter((plane) => plane.hasAttribute('PositionZ') && plane.hasAttribute('TheZ'))
      .map((plane) => ({
        index: Number(plane.getAttribute('TheZ')),
        nm: Number(plane.getAttribute('PositionZ')) * unit(plane.getAttribute('PositionZUnit') || ''),
      }))
      .filter((plane) => Number.isFinite(plane.nm))
      .sort((a, b) => a.index - b.index);
    if (positions.length > 1) {
      const steps = positions
        .slice(1)
        .map((p, i) => (p.nm - positions[i].nm) / (p.index - positions[i].index));
      const step = steps.reduce((a, b) => a + b, 0) / steps.length;
      if (Number.isFinite(step) && step !== 0) {
        if (steps.some((s) => Math.abs(s - step) > Math.max(0.01, Math.abs(step) * 0.03)))
          throw Error('The calibration Z positions are uneven. Resample the stack to a uniform Z grid.');
        return step;
      }
    }
  }
  if (pixels?.hasAttribute('PhysicalSizeZ'))
    return (
      Number(pixels.getAttribute('PhysicalSizeZ')) * unit(pixels.getAttribute('PhysicalSizeZUnit') || 'µm')
    );
  return (
    Number(description.match(/^spacing=([^\r\n]+)/m)?.[1]) *
    unit(description.match(/^unit=([^\r\n]+)/m)?.[1] || '')
  );
}
function checkAxis(m: TiffMovie, stack: boolean) {
  const size = (key: string, imageJ: string) =>
    Number(
      m.description.match(new RegExp(key + '="(\\d+)"'))?.[1] ??
        m.description.match(new RegExp('^' + imageJ + '=(\\d+)', 'm'))?.[1] ??
        1,
    );
  if (size('SizeC', 'channels') > 1 || (stack ? size('SizeT', 'frames') : size('SizeZ', 'slices')) > 1)
    throw Error(
      stack ? 'Supply a single-channel ZYX calibration stack.' : 'Supply a single-channel TYX movie.',
    );
}
async function calibrate(
  dev: GPUDevice,
  file: File,
  s: Settings,
  m: VectorialModel,
  observe?: MovieObserver,
) {
  const t = await TiffMovie.open([file]);
  checkAxis(t, true);
  const dz = s.stepNm || stackStepNm(t.description);
  if (!Number.isFinite(dz) || !dz || t.n < 3)
    throw Error('Calibration requires at least three planes and a signed Z step in nm.');
  const stack = { name: file.name, nz: t.n, H: t.H, W: t.W, data: photons(await t.read(0, t.n), s) };
  const beads = findBeads(stack, R, 0.12)
    .slice(0, 48)
    .map((b) => ({ ...b, stack: 0 }));
  if (!beads.length) throw Error('No calibration bead ROIs were found.');
  m.noll = [5, 6, 7, 8, 9, 10, 11];
  const fit = new BeadFitter(
    dev,
    [stack],
    beads,
    beads.map((b) => extractRoi(stack, b, R)),
    {
      optics: s.optics,
      roisize: R,
      npupil: m.npupil,
      oversample: m.oversample,
      sign: Math.sign(dz),
      dz_um: Math.abs(dz) / 1000,
      ze_um: s.diameterNm / 2000,
      fitDz: false,
      fitDrift: true,
      holdout: [],
      maxIter: 35,
      zernike_noll: m.noll,
    },
  );
  try {
    const result = await fit.fit(
      undefined,
      observe
        ? async (progress) => {
            await observe(dev, {
              stage: 'calibration',
              index: progress.done,
              total: progress.total,
              model: { ...m, zcoef: fit.st.zcoef },
              emitterHeightUm: s.diameterNm / 2000,
              focusUm: s.diameterNm / 2000,
              score: Number(progress.info?.nll_px),
            });
          }
        : undefined,
    );
    m.zcoef = Array.from(result.state.zcoef);
    if (!Array.from(m.zcoef).every(Number.isFinite))
      throw Error('Calibration failed to fit finite pupil coefficients.');
  } finally {
    fit.destroy();
  }
  return (Math.abs(dz) * (t.n - 1)) / 2000;
}
// Pixel-integrated vectorial PSFs, sampled at subpixel shifts. The six sphere
// points reproduce the three second moments of a uniformly fluorescent bead.
async function table(
  dev: GPUDevice,
  m: VectorialModel,
  diameter: number,
  half: number,
  offsetUm = 0,
  onlyPlane?: number,
) {
  const eng = new VecEngine(
    dev,
    { ...m.optics, roisize: R, npupil: m.npupil, oversample: m.oversample, zernike_noll: m.noll },
    [],
    { maxChunk: 64 },
  );
  const a = (diameter / 2000) * Math.sqrt(3 / 5),
    center = diameter / 2000;
  const nodes = a
    ? [
        [a, 0, 0],
        [-a, 0, 0],
        [0, a, 0],
        [0, -a, 0],
        [0, 0, a],
        [0, 0, -a],
      ]
    : [[0, 0, 0]];
  const out = new Float32Array(NZ * SIDE * SIDE);
  try {
    eng.setZernike(m.zcoef);
    for (let z = onlyPlane ?? 0; z <= (onlyPlane ?? NZ - 1); z++) {
      const images = [];
      for (let iy = 0; iy < K; iy++)
        for (let ix = 0; ix < K; ix++)
          for (const [dx, dy, dz] of nodes)
            images.push({
              x: (ix - 1) / K + dx / m.optics.pixel_um,
              y: (iy - 1) / K + dy / m.optics.pixel_um,
              ze: center + dz,
              f: center + offsetUm + half * ((2 * z) / (NZ - 1) - 1),
              I: 1,
              bg: 0,
            });
      eng.setImages(images);
      await eng.run(images.length, 'psf');
      const data = await eng.readPsf(images.length);
      for (let iy = 0; iy < K; iy++)
        for (let ix = 0; ix < K; ix++)
          for (let n = 0; n < nodes.length; n++) {
            const start = ((iy * K + ix) * nodes.length + n) * R * R;
            for (let y = 0; y < R; y++)
              for (let x = 0; x < R; x++)
                out[(z * SIDE + y * K + K - 1 - iy) * SIDE + x * K + K - 1 - ix] +=
                  data[start + y * R + x] / nodes.length;
          }
    }
  } finally {
    eng.destroy();
  }
  return out;
}
function seeds(mean: Float32Array, w: number, h: number, frames: number, kernel: Float32Array) {
  const bg = quantile(mean, 0.015),
    noise = Math.max(1, Math.sqrt(Math.max(bg, 1) / frames));
  const residual = mean.slice(),
    xy: number[][] = [];
  const occupied = new Uint8Array(w * h);
  const peaks = () => {
    const p: number[] = [];
    for (let y = 2; y < h - 2; y++)
      for (let x = 2; x < w - 2; x++) {
        const i = y * w + x,
          v = residual[i];
        if (v < bg + 5 * noise) continue;
        if ([-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1].every((d) => v > residual[i + d])) p.push(i);
      }
    return p.sort((a, b) => residual[b] - residual[a]);
  };
  const c = (SIDE - 1) / 2,
    mid = (((NZ - 1) / 2) * SIDE + c) * SIDE + c;
  for (let pass = 0; pass < 3; pass++)
    for (const i of peaks()) {
      const height = residual[i] - bg;
      if (height < 5 * noise) continue;
      const x = i % w,
        y = Math.floor(i / w);
      if (occupied[i]) continue;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) occupied[i + dy * w + dx] = 1;
      const shift = (d: number) =>
        clamp(
          (0.5 * (residual[i - d] - residual[i + d])) /
            Math.min(-noise, residual[i - d] + residual[i + d] - 2 * residual[i]),
          -0.5,
          0.5,
        );
      xy.push([x + shift(1), y + shift(w), height / kernel[mid]]);
      for (let dy = -6; dy <= 6; dy++)
        for (let dx = -6; dx <= 6; dx++)
          if (x + dx >= 0 && x + dx < w && y + dy >= 0 && y + dy < h)
            residual[i + dy * w + dx] -= (height * kernel[mid + (dy * SIDE + dx) * K]) / kernel[mid];
    }
  if (!xy.length) throw Error('No bead candidates detected. Check camera calibration and frame contrast.');
  return xy;
}
class Joint {
  buffers: GPUBuffer[] = [];
  read: GPUBuffer;
  params: GPUBuffer;
  lut: GPUBuffer;
  bead: GPUBuffer;
  obs: GPUBuffer;
  out: GPUBuffer;
  groups: GPUBindGroup[];
  pipes: GPUComputePipeline[];
  constructor(
    private dev: GPUDevice,
    private w: number,
    private h: number,
    private xy: number[][],
    lut: Float32Array,
  ) {
    const make = (size: number, usage: number) => {
      const b = dev.createBuffer({ size, usage });
      this.buffers.push(b);
      return b;
    };
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const put = (data: Float32Array | Uint32Array) => {
      const b = make(data.byteLength, storage);
      dev.queue.writeBuffer(b, 0, data);
      return b;
    };
    this.params = make(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.lut = put(lut);
    const cols = Math.ceil(w / 8),
      rows = Math.ceil(h / 8),
      tiles: number[][] = Array.from({ length: cols * rows }, () => []);
    xy.forEach(([x, y], i) => {
      for (
        let ty = Math.max(0, Math.floor((y - 12) / 8));
        ty <= Math.min(rows - 1, Math.floor((y + 12) / 8));
        ty++
      )
        for (
          let tx = Math.max(0, Math.floor((x - 12) / 8));
          tx <= Math.min(cols - 1, Math.floor((x + 12) / 8));
          tx++
        )
          tiles[ty * cols + tx].push(i);
    });
    const index = new Uint32Array(tiles.length + 1 + tiles.reduce((n, v) => n + v.length, 0));
    let offset = tiles.length + 1;
    tiles.forEach((v, i) => {
      index[i] = offset;
      index.set(v, offset);
      offset += v.length;
    });
    index[tiles.length] = offset;
    this.bead = make(xy.length * 16, storage);
    this.obs = make(w * h * 4, storage);
    this.out = make(xy.length * 36, storage | GPUBufferUsage.COPY_SRC);
    this.read = make(xy.length * 36, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const buffers = [
      this.params,
      this.lut,
      put(index),
      this.bead,
      this.obs,
      make(w * h * 4, storage),
      this.out,
    ];
    const module = dev.createShaderModule({ code: shader });
    this.pipes = ['render', 'fit'].map((entryPoint) =>
      dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } }),
    );
    this.groups = [
      [0, 1, 2, 3, 5],
      [0, 1, 3, 4, 5, 6],
    ].map((ids, i) =>
      dev.createBindGroup({
        layout: this.pipes[i].getBindGroupLayout(0),
        entries: ids.map((binding) => ({ binding, resource: { buffer: buffers[binding] } })),
      }),
    );
  }
  async step(b: Float32Array, d: number[], bg: number) {
    const { dev, w, h, xy } = this;
    const data = new ArrayBuffer(48);
    new Uint32Array(data).set([w, h, xy.length, SIDE, Math.ceil(w / 8), NZ]);
    new Float32Array(data).set([K, 1, ...d, bg], 6);
    dev.queue.writeBuffer(this.params, 0, data);
    dev.queue.writeBuffer(this.bead, 0, b);
    const enc = dev.createCommandEncoder();
    this.pipes.forEach((pipe, i) => {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, this.groups[i]);
      pass.dispatchWorkgroups(Math.ceil((i ? xy.length : w * h) / 64));
      pass.end();
    });
    enc.copyBufferToBuffer(this.out, 0, this.read, 0, this.read.size);
    dev.queue.submit([enc.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(this.read.getMappedRange().slice(0));
    this.read.unmap();
    return result;
  }
  destroy() {
    this.buffers.forEach((b) => b.destroy());
  }
}
export interface MovieIteration {
  stage: 'calibration' | 'profile' | 'frame' | 'final';
  index: number;
  total: number;
  model: VectorialModel;
  emitterHeightUm: number;
  focusUm: number;
  score?: number;
  driftNm?: number[];
}
export type PhotometrySnapshot = Pick<
  MovieFit,
  'xy' | 'width' | 'height' | 'drift' | 'intensity' | 'settings'
>;
export type MovieObserver = (device: GPUDevice, update: MovieIteration) => Promise<void>;
/** Camera-corrected input plane, delivered before its processing step. */
export type MovieFrameObserver = (
  pixels: Float32Array,
  width: number,
  height: number,
  index: number,
  total: number,
  stage: 'read' | 'fit',
) => void;
export async function fitMovie(
  dev: GPUDevice,
  file: File,
  stack: File | null,
  s: Settings,
  log: (msg: string) => void,
  observe?: MovieObserver,
  onFrame?: MovieFrameObserver,
  photometry?: (snapshot: PhotometrySnapshot) => void,
) {
  const t = await TiffMovie.open([file]);
  checkAxis(t, false);
  if (t.n < 2) throw Error('A movie requires multiple frames.');
  const model: VectorialModel = {
    optics: s.optics,
    noll: [5, 6, 7, 8, 9, 10, 11],
    zcoef: Array(7).fill(0),
    roisize: R,
    npupil: 32,
    oversample: 2,
    focus_um: s.diameterNm / 2000,
    z_range_um: [-0.4, 0.4],
  };
  log('Reading movie');
  const mean = new Float32Array(t.H * t.W);
  const period = Math.min(t.n, s.period ?? 1),
    groups = Array.from({ length: period }, () => new Float32Array(mean.length));
  const sampleFrames = new Set(
    Array.from({ length: Math.min(11, t.n) }, (_, i) =>
      Math.round((i * (t.n - 1)) / (Math.min(11, t.n) - 1)),
    ),
  );
  const shapes: NonNullable<ReturnType<typeof observedShape>>[] = [];
  const sampleIndices: number[] = [];
  for (let i = 0; i < t.n; i++) {
    const f = photons(await t.read(i, i + 1), s);
    if (!stack && sampleFrames.has(i)) {
      const shape = observedShape(f, t.W, t.H);
      if (shape) {
        shapes.push(shape);
        sampleIndices.push(i);
      }
    }
    onFrame?.(f, t.W, t.H, i + 1, t.n, 'read');
    for (let j = 0; j < f.length; j++) {
      mean[j] += f[j] / t.n;
      groups[i % period][j] += f[j] / Math.ceil((t.n - (i % period)) / period);
    }
  }
  log('Calibrating optics');
  const half = stack ? await calibrate(dev, stack, s, model, observe) : 0.35;
  model.z_range_um = [-half, half];
  const dz = (2 * half) / (NZ - 1);
  const shapeTrace =
    !stack && shapes.length >= 3
      ? await fitMovieShapes(dev, model, s.diameterNm, shapes, observe)
      : undefined;
  const shapeFocus = shapeTrace?.[0];
  const initialCenter = shapeFocus ?? 0;
  let lut = await table(dev, model, s.diameterNm, half, initialCenter);
  const xy = seeds(mean, t.W, t.H, t.n, lut);
  const bins = new Map<number, number[][]>();
  const bin = (x: number, y: number) => Math.floor(y / 2) * t.W + Math.floor(x / 2);
  const insert = (p: number[]) => {
    const k = bin(p[0], p[1]);
    if (!bins.has(k)) bins.set(k, []);
    bins.get(k)!.push(p);
  };
  xy.forEach(insert);
  if (period > 1)
    for (const group of groups)
      for (const p of seeds(group, t.W, t.H, t.n / period, lut)) {
        let near = false;
        for (const dy of [-2, 0, 2])
          for (const dx of [-2, 0, 2])
            if (
              bins
                .get(bin(p[0] + dx, p[1] + dy))
                ?.some((q) => (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < 1.2 ** 2)
            )
              near = true;
        if (!near) {
          xy.push(p);
          insert(p);
        }
      }
  const gpu = new Joint(dev, t.W, t.H, xy, lut);
  const b = new Float32Array(xy.length * 4);
  xy.forEach((v, i) => b.set(v, 4 * i));
  const base = b.slice(),
    bg = quantile(mean, 0.01);
  let d = [0, 0, (NZ - 1) / 2],
    best = Infinity,
    bestCoef = 0;
  try {
    dev.queue.writeBuffer(gpu.obs, 0, mean);
    // Profile the mean image over focus and a single astigmatism coefficient.
    // Through-focus calibration supplies the signed, multi-mode pupil when available.
    let trial = 0;
    const trialCount = (stack ? 1 : 3) * Math.floor((NZ - 2) / 2) + (stack ? 0 : 4 * model.noll.length);
    for (const coef of shapeFocus !== undefined ? [] : stack ? [0] : [0, 0.075, 0.15]) {
      if (coef) {
        model.zcoef = [0, coef, 0, 0, 0, 0, 0];
        lut = await table(dev, model, s.diameterNm, half);
        dev.queue.writeBuffer(gpu.lut, 0, lut);
      }
      for (let z = 2; z < NZ - 1; z += 2) {
        b.set(base);
        let cost = 0;
        for (let j = 0; j < 3; j++) {
          const stat = await gpu.step(b, [0, 0, z], bg);
          cost = 0;
          for (let i = 0; i < xy.length; i++) {
            b[4 * i + 2] *= stat[9 * i];
            cost += stat[9 * i + 7];
          }
        }
        trial++;
        if (observe)
          await observe(dev, {
            stage: 'profile',
            index: trial,
            total: trialCount,
            model,
            emitterHeightUm: s.diameterNm / 2000,
            focusUm: s.diameterNm / 2000 + (z - (NZ - 1) / 2) * dz,
            score: cost,
          });
        if (cost < best) {
          best = cost;
          d[2] = z;
          if (!stack) bestCoef = coef;
        }
      }
    }
    if (!stack && shapeFocus === undefined) {
      const coefficients = [0, bestCoef, 0, 0, 0, 0, 0];
      // Score independent signed perturbations of each pupil mode at the best focus.
      for (const delta of [0.04, 0.02])
        for (let mode = 0; mode < coefficients.length; mode++) {
          const center = coefficients[mode];
          for (const sign of [-1, 1]) {
            const candidate = coefficients.slice();
            candidate[mode] = center + sign * delta;
            model.zcoef = candidate;
            const offset = d[2] * SIDE * SIDE;
            const bank = await table(dev, model, s.diameterNm, half, 0, d[2]);
            dev.queue.writeBuffer(gpu.lut, offset * 4, bank.subarray(offset, offset + SIDE * SIDE));
            b.set(base);
            let cost = 0;
            for (let j = 0; j < 3; j++) {
              const stat = await gpu.step(b, d, bg);
              cost = 0;
              for (let i = 0; i < xy.length; i++) {
                b[4 * i + 2] *= stat[9 * i];
                cost += stat[9 * i + 7];
              }
            }
            if (cost < best) {
              best = cost;
              coefficients[mode] = candidate[mode];
            }
            trial++;
            if (observe)
              await observe(dev, {
                stage: 'profile',
                index: trial,
                total: trialCount,
                model,
                emitterHeightUm: s.diameterNm / 2000,
                focusUm: s.diameterNm / 2000 + (d[2] - (NZ - 1) / 2) * dz,
                score: cost,
              });
          }
        }
      model.zcoef = coefficients;
      lut = await table(dev, model, s.diameterNm, half);
      dev.queue.writeBuffer(gpu.lut, 0, lut);
    }
    b.set(base);
    log('Refining bead positions');
    for (let it = 0; it < 10; it++) {
      const stat = await gpu.step(b, d, bg);
      for (let i = 0; i < xy.length; i++) {
        const j = 9 * i,
          k = 4 * i;
        b[k + 2] *= stat[j];
        for (let u = 0; u < 2; u++)
          b[k + u] = clamp(
            b[k + u] + 0.45 * clamp(stat[j + 1 + u] / Math.max(1, stat[j + 4 + u]), -0.4, 0.4),
            base[k + u] - 1.25,
            base[k + u] + 1.25,
          );
      }
    }
    const groupFlux = groups.map(() => b.slice());
    for (let round = 0; round < 7; round++) {
      log('Refining grouped positions � ' + (round + 1) + ' / 7');
      const deriv = new Float64Array(xy.length * 4);
      for (let j = 0; j < period; j++) {
        const g = groupFlux[j];
        dev.queue.writeBuffer(gpu.obs, 0, groups[j]);
        for (let i = 0; i < xy.length; i++) {
          g[4 * i] = b[4 * i];
          g[4 * i + 1] = b[4 * i + 1];
        }
        for (let it = 0; it < 3; it++) {
          const st = await gpu.step(g, d, bg);
          for (let i = 0; i < xy.length; i++) {
            const k = 4 * i,
              o = 9 * i;
            g[k + 2] *= st[o];
            if (it === 2) {
              deriv[k] += st[o + 1];
              deriv[k + 1] += st[o + 2];
              deriv[k + 2] += st[o + 4];
              deriv[k + 3] += st[o + 5];
            }
          }
        }
      }
      for (let i = 0; i < xy.length; i++)
        for (let u = 0; u < 2; u++) {
          const k = 4 * i + u;
          b[k] = clamp(
            b[k] + 0.6 * clamp(deriv[k] / Math.max(1, deriv[k + 2]), -0.4, 0.4),
            base[k] - 1.5,
            base[k] + 1.5,
          );
        }
    }
    const ref = b.slice(),
      intensity: number[][] = [],
      drift: number[][] = [],
      background: number[] = [];
    let focusWindowShifts = 0,
      windowCenterUm = initialCenter;
    const fittedXY = xy.map((_, i) => [ref[4 * i], ref[4 * i + 1]]);
    let minFocusUm = -half,
      maxFocusUm = half;
    for (let frame = 0; frame < t.n; frame++) {
      if (shapeTrace) {
        let j = 0;
        while (j + 1 < sampleIndices.length && sampleIndices[j + 1] <= frame) j++;
        const k = Math.min(shapeTrace.length - 1, j + 1);
        const f = clamp((frame - sampleIndices[j]) / Math.max(1, sampleIndices[k] - sampleIndices[j]), 0, 1);
        const target = shapeTrace[j] + f * (shapeTrace[k] - shapeTrace[j]);
        d[2] = (NZ - 1) / 2 + (target - windowCenterUm) / dz;
      }
      log('Fitting frame ' + (frame + 1) + ' / ' + t.n);
      const im = photons(await t.read(frame, frame + 1), s),
        back = quantile(im, 0.01);
      onFrame?.(im, t.W, t.H, frame + 1, t.n, 'fit');
      background.push(back);
      dev.queue.writeBuffer(gpu.obs, 0, im);
      b.set(ref);
      for (let i = 0; i < xy.length; i++) b[i * 4 + 2] = groupFlux[frame % period][i * 4 + 2];
      for (let it = 0; it < 12; it++) {
        // Re-tabulate around the current physical focus before interpolation approaches
        // either edge. Keep the same Z spacing and preserve the absolute trajectory.
        if (d[2] < 2 || d[2] > NZ - 3) {
          focusWindowShifts++;
          windowCenterUm += (d[2] - (NZ - 1) / 2) * dz;
          d[2] = (NZ - 1) / 2;
          lut = await table(dev, model, s.diameterNm, half, windowCenterUm);
          dev.queue.writeBuffer(gpu.lut, 0, lut);
        }
        const stat = await gpu.step(b, d, back),
          g = [0, 0, 0],
          h = [0, 0, 0];
        for (let i = 0; i < xy.length; i++) {
          const j = i * 9;
          b[i * 4 + 2] *= stat[j];
          for (let u = 0; u < 3; u++) {
            g[u] += stat[j + 1 + u];
            h[u] += stat[j + 4 + u];
          }
        }
        if (it >= 2 && it < 10)
          for (let u = 0; u < 3; u++) {
            const update = 0.6 * clamp(g[u] / Math.max(1, h[u]), u === 2 ? -1 : -0.25, u === 2 ? 1 : 0.25);
            d[u] = u === 2 ? d[u] + update : clamp(d[u] + update, -4, 4);
          }
      }
      intensity.push(xy.map((_, i) => b[i * 4 + 2]));
      const focusUm = windowCenterUm + (d[2] - (NZ - 1) / 2) * dz;
      drift.push([d[0], d[1], focusUm * 1000]);
      minFocusUm = Math.min(minFocusUm, focusUm);
      maxFocusUm = Math.max(maxFocusUm, focusUm);
      if (photometry && (frame === 11 || frame === 35) && frame + 1 < t.n)
        photometry({ xy: fittedXY, width: t.W, height: t.H, intensity, drift, settings: s });
      if (observe)
        await observe(dev, {
          stage: 'frame',
          index: frame + 1,
          total: t.n,
          model,
          emitterHeightUm: s.diameterNm / 2000,
          focusUm: s.diameterNm / 2000 + drift[frame][2] / 1000,
          driftNm: [d[0] * s.optics.pixel_um * 1000, d[1] * s.optics.pixel_um * 1000, drift[frame][2]],
        });
    }
    model.z_range_um = [minFocusUm, maxFocusUm];
    model.focus_um = s.diameterNm / 2000 + drift.reduce((sum, p) => sum + p[2], 0) / drift.length / 1000;
    if (observe)
      await observe(dev, {
        stage: 'final',
        index: t.n,
        total: t.n,
        model,
        emitterHeightUm: s.diameterNm / 2000,
        focusUm: model.focus_um,
      });
    return {
      width: t.W,
      height: t.H,
      xy: fittedXY,
      intensity,
      drift,
      background,
      model,
      settings: s,
      stackCalibrated: !!stack,
      boundaryFrames: 0, // Retained for earlier consumers. Axial motion is no longer clipped.
      focusWindowShifts,
    } satisfies MovieFit;
  } finally {
    gpu.destroy();
  }
}
export async function exportMovieReport(dev: GPUDevice, r: MovieFit, source?: string) {
  const m = r.model,
    optics = m.optics;
  const settings = {
    bead_diameter_nm: r.settings.diameterNm,
    stack_calibrated: r.stackCalibrated,
    'pixel size (nm)': 1000 * optics.pixel_um,
    NA: optics.NA,
    'emission wavelength (nm)': 1000 * optics.wavelength_um,
    'immersion / sample index': [optics.n_imm, optics.n_med],
    'gain (photons/ADU)': r.settings.gain,
    'offset (ADU)': r.settings.offset,
  };
  const results = {
    'fitted candidates': r.xy.length,
    frames: r.drift.length,
    'mean stage focus (nm)': 1000 * m.focus_um,
    focus_window_shifts: r.focusWindowShifts,
  };
  return buildPsfZip(dev, {
    vectorial: m,
    report: { estimator: 'dense bead movie', settings, results },
    pdf: (volume) =>
      makeFitReport({
        title: 'Dense bead movie PSF',
        subtitle: source,
        settings,
        results,
        zernike: { noll: m.noll, values: m.zcoef },
        pupil: pupilMaps(optics, m.noll, Array.from(m.zcoef), m.npupil),
        psf: volumeSlices(
          volume.stack,
          {
            optics,
            npupil: m.npupil,
            noll: m.noll,
            zcoef: Array.from(m.zcoef),
            nxy: volume.W,
            nz: volume.Z,
            dxy_um: optics.pixel_um * volume.sample_px,
            zmin: volume.z_um[0],
            zmax: volume.z_um[volume.Z - 1],
            ze_um: 0,
            focus_um: m.focus_um,
          },
          'z',
        ),
        notes: [
          r.stackCalibrated
            ? 'The calibration stack supplied the fitted pupil.'
            : 'Movie-only calibration profiled Noll modes 5 through 11. The axial sign can be ambiguous near focus.',
          'PSF slices show emitter height at the fitted mean stage focus.',
        ],
      }),
  });
}
export async function exportPsf(dev: GPUDevice, r: MovieFit) {
  return (await exportMovieReport(dev, r)).bytes;
}
