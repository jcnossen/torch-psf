import { VecEngine } from '../gpu/ops';
import { fft2d } from '../psfgrab/fft';
import { solve } from '../fit/linalg';
import type { VectorialModel } from '../psf/zstack';
import type { MovieObserver } from './movie';

const grid = 32,
  radius = 3,
  samples = (2 * radius + 1) ** 2,
  tau = 2 * Math.PI;
const freq: number[][] = [];
for (let y = 0; y <= 12; y++)
  for (let x = -12; x <= 12; x++) {
    const r = Math.hypot(x, y);
    if ((y > 0 || x > 0) && r >= 2 && r <= 12) freq.push([x, y]);
  }
const wrap = (i: number, n: number) => ((i % n) + n) % n;
function norm2(a: Float64Array) {
  let s = 0;
  for (const x of a) s += x * x;
  return s;
}

// Average complete illumination cycles to obtain repeated observations of the
// same object through different optical transfer functions. Store spectra only.
export class CycleSpectra {
  readonly frames: number[] = [];
  readonly spectra: Float64Array[] = [];
  private selected: Set<number>;
  private side: number;
  private plane: Float64Array;
  constructor(
    private width: number,
    private height: number,
    n: number,
    private period: number,
  ) {
    const cycles = Math.floor(n / period),
      count = Math.min(16, cycles);
    this.side = Math.min(width, height, 512);
    this.selected = new Set(
      this.side >= 96 && count >= 5
        ? Array.from({ length: count }, (_, i) => Math.round((i * (cycles - 1)) / (count - 1)))
        : [],
    );
    this.plane = new Float64Array(this.selected.size ? this.side ** 2 : 0);
  }
  add(im: Float32Array, t: number) {
    if (!this.selected.has(Math.floor(t / this.period))) return;
    const n = this.side,
      x0 = Math.floor((this.width - n) / 2),
      y0 = Math.floor((this.height - n) / 2);
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) this.plane[y * n + x] += im[(y + y0) * this.width + x + x0];
    if (t % this.period !== this.period - 1) return;
    const mean = this.plane.reduce((s, x) => s + x, 0) / this.plane.length;
    const size = 2 ** Math.ceil(Math.log2(n)),
      re = new Float64Array(size * size),
      imPart = re.slice();
    const win = Array.from({ length: n }, (_, i) => Math.sin((Math.PI * i) / (n - 1)) ** 2);
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) re[y * size + x] = (this.plane[y * n + x] - mean) * win[x] * win[y];
    fft2d(re, imPart, size);
    const block = new Float64Array(2 * freq.length * samples);
    let k = 0;
    for (const [x, y] of freq)
      for (let dy = -radius; dy <= radius; dy++)
        for (let dx = -radius; dx <= radius; dx++) {
          const j =
            wrap(Math.round((y * size) / grid + (dy * size) / n), size) * size +
            wrap(Math.round((x * size) / grid + (dx * size) / n), size);
          block[k++] = re[j];
          block[k++] = imPart[j];
        }
    this.spectra.push(block);
    this.frames.push(t - (this.period - 1) / 2);
    this.plane.fill(0);
  }
}

// For each frequency, estimate its complex temporal transfer up to one unknown
// complex object amplitude from the leading eigenvector of the cross spectrum.
function targets(data: CycleSpectra) {
  const n = data.frames.length,
    target = new Float64Array(2 * n * freq.length),
    power = new Float64Array(freq.length);
  for (let f = 0; f < freq.length; f++) {
    const cov = new Float64Array(2 * n * n);
    let ref = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        let ar = 0,
          ai = 0;
        for (let k = 0; k < samples; k++) {
          const p = 2 * (f * samples + k),
            a = data.spectra[i],
            b = data.spectra[j];
          ar += a[p] * b[p] + a[p + 1] * b[p + 1];
          ai += a[p + 1] * b[p] - a[p] * b[p + 1];
        }
        cov[2 * (i * n + j)] = cov[2 * (j * n + i)] = ar;
        cov[2 * (i * n + j) + 1] = ai;
        cov[2 * (j * n + i) + 1] = -ai;
        if (i === j && ar > cov[2 * (ref * n + ref)]) ref = i;
      }
    let v = new Float64Array(2 * n);
    v[2 * ref] = 1;
    for (let it = 0; it < 12; it++) {
      const u = new Float64Array(2 * n);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          const k = 2 * (i * n + j);
          u[2 * i] += cov[k] * v[2 * j] - cov[k + 1] * v[2 * j + 1];
          u[2 * i + 1] += cov[k] * v[2 * j + 1] + cov[k + 1] * v[2 * j];
        }
      power[f] = Math.sqrt(norm2(u));
      v = u.map((x) => x / Math.max(power[f], 1e-16));
    }
    for (let t = 0; t < n; t++) {
      const k = 2 * (t * freq.length + f);
      target[k] = v[2 * t];
      target[k + 1] = v[2 * t + 1];
    }
  }
  const cap = power.slice().sort()[Math.floor(0.9 * power.length)];
  const weight = power.map((v) => Math.max(0.03, Math.min(1, v / Math.max(cap, 1e-12))) ** 0.25);
  return { target, weight };
}

export interface PupilTrial {
  sign:number;
  iterations:number;
  stop:string;
  cost:number;
  coefficients:number[];
}
export interface DiversityDiagnostic {
  focusSpanUm:number;
  decision:string;
  trials:PupilTrial[];
  constantCost?:number;
}
export async function refineDiversePupil(
  dev: GPUDevice,
  m: VectorialModel,
  diameter: number,
  data: CycleSpectra,
  initial: number[],
  observe?: MovieObserver,
  cycleOffsets?: number[][],
  record?:(trial:DiversityDiagnostic)=>void,
) {
  const n = data.frames.length,
    nm = m.noll.length,
    nf = freq.length,
    npar = nm + n + 2 * (n - 1);
  const focusSpanUm=Math.max(...initial)-Math.min(...initial);
  if(n<5 || focusSpanUm<.04){
    record?.({focusSpanUm,decision:n<5?'Too few complete cycles':'Initial focus span below 40 nm',trials:[]});
    return;
  }
  const { target, weight } = targets(data),
    base = Array.from(m.zcoef),
    height = diameter / 2000;
  const engine = new VecEngine(
    dev,
    { ...m.optics, roisize: 33, npupil: m.npupil, oversample: m.oversample, zernike_noll: m.noll },
    [],
  );
  async function transfer(x: Float64Array, sign: number) {
    engine.setZernike(x.subarray(0, nm));
    const groups = cycleOffsets ?? data.frames.map(() => [0]);
    const images = groups.flatMap((offsets, t) =>
      offsets.map((offset) => ({
        x: 0,
        y: 0,
        ze: height,
        f: height + x[nm + t] + sign * offset,
        I: 1,
        bg: 0,
      })),
    );
    engine.setImages(images);
    await engine.run(images.length, 'psf');
    const planes = await engine.readPsf(images.length);
    const bank = new Float64Array(n * 33 * 33);
    let source = 0;
    groups.forEach((offsets, t) => {
      for (const _ of offsets) for (let j = 0; j < 33 * 33; j++) bank[t * 33 * 33 + j] += planes[source++];
      for (let j = 0; j < 33 * 33; j++) bank[t * 33 * 33 + j] /= offsets.length;
    });
    const h = new Float64Array(2 * n * nf);
    for (let t = 0; t < n; t++) {
      const re = new Float64Array(grid * grid),
        im = re.slice();
      for (let y = 0; y < 33; y++)
        for (let i = 0; i < 33; i++)
          re[wrap(y - 16, grid) * grid + wrap(i - 16, grid)] += bank[(t * 33 + y) * 33 + i];
      fft2d(re, im, grid);
      const ix = nm + n + 2 * (t - 1),
        dx = t ? x[ix] : 0,
        dy = t ? x[ix + 1] : 0;
      freq.forEach(([fx, fy], j) => {
        const k = wrap(fy, grid) * grid + wrap(fx, grid),
          a = (-tau * (fx * dx + fy * dy)) / grid;
        const c = Math.cos(a),
          s = Math.sin(a),
          o = 2 * (t * nf + j);
        h[o] = re[k] * c - im[k] * s;
        h[o + 1] = re[k] * s + im[k] * c;
      });
    }
    return h;
  }
  // Project out the unknown sample coefficient independently at each frequency.
  function residual(h: Float64Array, x: Float64Array) {
    const out = new Float64Array(h.length + n - 1);
    for (let f = 0; f < nf; f++) {
      let ar = 0,
        ai = 0,
        den = 0;
      for (let t = 0; t < n; t++) {
        const i = 2 * (t * nf + f);
        den += h[i] ** 2 + h[i + 1] ** 2;
        ar += h[i] * target[i] + h[i + 1] * target[i + 1];
        ai += h[i] * target[i + 1] - h[i + 1] * target[i];
      }
      ar /= Math.max(den, 1e-12);
      ai /= Math.max(den, 1e-12);
      for (let t = 0; t < n; t++) {
        const i = 2 * (t * nf + f);
        out[i] = weight[f] * (h[i] * ar - h[i + 1] * ai - target[i]);
        out[i + 1] = weight[f] * (h[i] * ai + h[i + 1] * ar - target[i + 1]);
      }
    }
    for (let t = 1; t < n; t++)
      out[h.length + t - 1] = Math.sqrt((0.002 * nf) / (n - 1)) * (x[nm + t] - x[nm + t - 1]);
    return out;
  }
  const iterations = 26;
  async function branch(sign: number, branchIndex: number) {
    let x = new Float64Array(npar);
    base.forEach((v, j) => (x[j] = v * ([5, 6, 11].includes(m.noll[j]) ? sign : 1)));
    x.set(
      initial.map((z) => sign * z),
      nm,
    );
    let h = await transfer(x, sign),
      r = residual(h, x),
      cost = norm2(r),
      lambda = 0.01;
    let used=0,stop='Iteration limit';
    for (let it = 0; it < iterations; it++) {
      used=it+1;
      const jac: Float64Array[] = [],
        eps = 0.0005;
      const shifted = x.slice();
      for (let t = 0; t < n; t++) shifted[nm + t] += eps;
      const hz = await transfer(shifted, sign);
      for (let j = 0; j < npar; j++) {
        const xp = x.slice();
        xp[j] += eps;
        let hp: Float64Array;
        if (j < nm) hp = await transfer(xp, sign);
        else {
          hp = h.slice();
          const z = j < nm + n,
            t = z ? j - nm : 1 + Math.floor((j - nm - n) / 2);
          if (z) hp.set(hz.subarray(2 * t * nf, 2 * (t + 1) * nf), 2 * t * nf);
          else
            for (let f = 0; f < nf; f++) {
              const k = 2 * (t * nf + f),
                a = (-tau * freq[f][(j - nm - n) % 2] * eps) / grid;
              hp[k] = h[k] * Math.cos(a) - h[k + 1] * Math.sin(a);
              hp[k + 1] = h[k] * Math.sin(a) + h[k + 1] * Math.cos(a);
            }
        }
        jac.push(residual(hp, xp).map((v, k) => (v - r[k]) / eps));
      }
      const A = new Float64Array(npar * npar),
        b = new Float64Array(npar);
      for (let i = 0; i < npar; i++) {
        const ji = jac[i];
        let g = 0;
        for (let k = 0; k < r.length; k++) g -= ji[k] * r[k];
        b[i] = g;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          const jj = jac[j];
          for (let k = 0; k < r.length; k++) s += ji[k] * jj[k];
          A[i * npar + j] = A[j * npar + i] = s;
        }
      }
      let accepted = false,
        improvement = 0;
      for (let attempt = 0; attempt < 5; attempt++) {
        const lhs = A.slice();
        for (let i = 0; i < npar; i++) lhs[i * npar + i] += lambda * Math.max(A[i * npar + i], 1e-5);
        const delta = solve(lhs, b.slice(), npar);
        const scale = Math.max(
          1,
          ...delta.map((v, j) => Math.abs(v) / (j < nm ? 0.07 : j < nm + n ? 0.16 : 0.45)),
        );
        const xp = x.map((v, j) => v + delta[j] / scale),
          hp = await transfer(xp, sign),
          rp = residual(hp, xp),
          cp = norm2(rp);
        if (cp < cost) {
          improvement = cost - cp;
          cost = cp;
          x = xp;
          h = hp;
          r = rp;
          lambda = Math.max(1e-7, lambda / 3);
          accepted = true;
          break;
        }
        lambda *= 8;
      }
      if (observe)
        await observe(dev, {
          stage: 'diversity',
          index: branchIndex * iterations + it + 1,
          total: 2 * iterations,
          model: { ...m, zcoef: Array.from(x.subarray(0, nm)) },
          emitterHeightUm: height,
          focusUm: height + x[nm],
          score: cost / nf,
        });
      if(!accepted || improvement<1e-7*nf){
        stop=accepted?'Score improvement below tolerance':'No downhill step';break;
      }
    }
    return { x, sign, cost: cost / nf, used, stop };
  }
  try {
    const a = await branch(1, 0),
      b = await branch(-1, 1);
    const best = a.cost <= b.cost ? a : b,
      constant = best.x.slice();
    constant.fill(best.x[nm], nm, nm + n);
    const flatCost = norm2(residual(await transfer(constant, best.sign), constant)) / nf;
    const accepted=best.cost<.08 && best.cost<.85*flatCost;
    record?.({focusSpanUm,constantCost:flatCost,decision:accepted?'Accepted':'Failed score or focus-diversity comparison',
      trials:[a,b].map(v=>({sign:v.sign,iterations:v.used,stop:v.stop,cost:v.cost,coefficients:Array.from(v.x.subarray(0,nm))}))});
    if(!accepted)return;
    return {
      coefficients: Array.from(best.x.subarray(0, nm)),
      focus: Array.from(best.x.subarray(nm, nm + n)),
      cost: best.cost,
      alternative: Math.max(a.cost, b.cost),
      constantCost: flatCost,
    };
  } finally {
    engine.destroy();
  }
}
