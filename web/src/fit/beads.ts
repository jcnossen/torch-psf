// Bead z-stack -> vectorial PSF: bead finding, ROI extraction and the Poisson LM fit (nested Schur complement).
//
// Model (the same as tests/make_web_reference.py fit_beads_web, which is the Python reference):
//   mu_bj = I_bj psf(x_b + dx_s(j), y_b + dy_s(j), z_e, f_bj) + bg_bj
//   f_bj  = sign dz exp(logdz) (j - (nz - 1) / 2) - zoff_b + z_e
//   dx_s, dy_s = sum_{d=1..4} c_sd tt^d, tt = linspace(-1, 1, nz)   (per-stack drift, optional)
// Parameters: per image (I, bg); per bead (x, y, zoff); global (Zernikes, optional logdz, optional drift).
// The GPU gives the Poisson normal equations per image over the "atomic" directions (I, bg, x, y, focus, Zernikes);
// the CPU maps them onto the parameters (chain rule) and eliminates per-image, then per-bead blocks (Schur
// complement), and solves the small global system.
import { VecEngine } from '../gpu/ops';
import { VectorialConfig, Optics, DEFAULT_NOLL } from '../psf/vectorial';
import { solve, inverse, lgamma } from './linalg';
import { StopToken, Progress, Eta } from '../core/task';

export interface Stack { name: string; nz: number; H: number; W: number; data: Float32Array; }   // photons [nz, H, W]
export interface Bead { stack: number; y0: number; x0: number; }                               // ROI corner (px)

/** Beads = isolated maxima of the mean projection over z (peaked at the bead centre also for wide 3D PSFs such as
 *  the tetrapod, whose max projection has one maximum per lobe): local maximum over an R x R neighbourhood (R rounded
 *  up to odd), above median + minRel (max - median) of the projection; plateau duplicates removed; the R x R ROI fully
 *  inside the frame. Same rule as tests/make_web_reference.py find_beads. */
export function findBeads(s: Stack, R: number, minRel = 0.3): Bead[] {
  const { nz, H, W, data } = s;
  const acc = new Float64Array(H * W);
  for (let z = 0; z < nz; z++) for (let i = 0; i < H * W; i++) acc[i] += data[z * H * W + i];
  const mp = new Float32Array(H * W);
  for (let i = 0; i < H * W; i++) mp[i] = acc[i] / nz;
  const med = median(mp);
  let mx = -Infinity; for (const v of mp) mx = Math.max(mx, v);
  const thr = med + minRel * (mx - med);
  const size = R | 1;
  const h = Math.floor(size / 2), r2 = Math.floor(R / 2);
  const pts: [number, number, number][] = [];
  for (let y = r2; y < H - r2; y++) for (let x = r2; x < W - r2; x++) {
    const v = mp[y * W + x];
    if (!(v > thr)) continue;
    let isMax = true;
    for (let dy = -h; dy <= h && isMax; dy++) for (let dx = -h; dx <= h; dx++) {
      const yy = Math.min(H - 1, Math.max(0, y + dy)), xx = Math.min(W - 1, Math.max(0, x + dx));
      if (mp[yy * W + xx] > v) { isMax = false; break; }
    }
    if (isMax) pts.push([v, y, x]);
  }
  pts.sort((a, b) => b[0] - a[0] || a[1] - b[1] || a[2] - b[2]);
  const keep: [number, number][] = [];
  for (const [, y, x] of pts) if (!keep.some(([qy, qx]) => Math.max(Math.abs(y - qy), Math.abs(x - qx)) <= h)) keep.push([y, x]);
  keep.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return keep.map(([y, x]) => ({ stack: -1, y0: y - r2, x0: x - r2 }));
}

/** Exact median (histogram for integer-valued data, else sort). */
export function median(d: Float32Array): number {
  let integer = true, lo = Infinity, hi = -Infinity;
  for (let i = 0; i < d.length; i += 7) { const v = d[i]; if (v !== Math.round(v)) { integer = false; break; } }
  for (const v of d) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (integer && hi - lo < 1 << 22) {
    for (const v of d) if (v !== Math.round(v)) { integer = false; break; }
  }
  const n = d.length;
  if (integer && hi - lo < 1 << 22) {
    const h = new Uint32Array(hi - lo + 1);
    for (const v of d) h[v - lo]++;
    const at = (k: number) => { let c = 0; for (let i = 0; i < h.length; i++) { c += h[i]; if (c > k) return i + lo; } return hi; };
    return n % 2 ? at((n - 1) / 2) : (at(n / 2 - 1) + at(n / 2)) / 2;
  }
  const s = Float32Array.from(d).sort();
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

export function extractRoi(s: Stack, b: Bead, R: number): Float32Array {
  const out = new Float32Array(s.nz * R * R);
  for (let z = 0; z < s.nz; z++) for (let y = 0; y < R; y++) {
    const o = (z * s.H + b.y0 + y) * s.W + b.x0;
    out.set(s.data.subarray(o, o + R), (z * R + y) * R);
  }
  return out;
}

export interface BeadFitOptions {
  optics: Optics;
  roisize: number;
  npupil: number;
  oversample: number;
  sign: number;           // stage direction (+1 / -1)
  dz_um: number;          // nominal z step
  ze_um: number;          // bead centre height above the coverslip (point emitter)
  fitDz: boolean;         // fit a z-step scale
  fitDrift: boolean;      // per-stack xy drift (polynomial degree 1-4 in the plane index)
  holdout: number[];      // bead indices left out of the PSF fit (refitted with the PSF frozen)
  maxIter: number;
  zernike_noll?: number[];
}

export interface BeadFitState {
  zcoef: Float64Array;
  logdz: number;
  drift: Float64Array;    // [nstack, 2, 4]
  x: Float64Array; y: Float64Array; zoff: Float64Array;   // per bead
  I: Float64Array[]; bg: Float64Array[];                  // per bead, per plane
}

export interface BeadFitResult {
  state: BeadFitState;
  nll_train_px: number;
  nll_heldout_px: number | null;
  iterations: number;
  heldout_iterations: number;
  partial: boolean;
  converged: boolean;
  seconds: number;
  /** model images of every bead [nz, R, R] (expected counts mu) */
  model: Float32Array[];
}

type GlobalParam = { kind: 'zern'; i: number } | { kind: 'logdz' } | { kind: 'drift'; s: number; c: number; d: number };

export class BeadFitter {
  readonly eng: VecEngine;
  readonly R: number;
  readonly opt: BeadFitOptions;
  readonly st: BeadFitState;
  private lg: number[];   // sum lgamma(d + 1) per bead
  nstack: number;

  constructor(dev: GPUDevice, public stacks: Stack[], public beads: Bead[], public rois: Float32Array[], opt: BeadFitOptions) {
    this.opt = opt;
    this.R = opt.roisize;
    const noll = opt.zernike_noll ?? DEFAULT_NOLL;
    const cfg: VectorialConfig = { ...opt.optics, roisize: opt.roisize, npupil: opt.npupil, oversample: opt.oversample, zernike_noll: noll };
    this.eng = new VecEngine(dev, cfg, ['x', 'y', 'focus', 'zern']);
    this.nstack = stacks.length;
    const nb = beads.length, R = this.R, RR = R * R;
    const st: BeadFitState = {
      zcoef: new Float64Array(noll.length), logdz: 0, drift: new Float64Array(this.nstack * 8),
      x: new Float64Array(nb), y: new Float64Array(nb), zoff: new Float64Array(nb), I: [], bg: [],
    };
    this.lg = [];
    for (let k = 0; k < nb; k++) {
      const nz = stacks[beads[k].stack].nz, d = rois[k];
      let best = -Infinity, peak = 0;
      const I = new Float64Array(nz), bg = new Float64Array(nz);
      let lgs = 0;
      for (let j = 0; j < nz; j++) {
        const o = j * RR;
        let m = -Infinity, sum = 0;
        const border: number[] = [];
        for (let i = 0; i < R; i++) border.push(d[o + i], d[o + (R - 1) * R + i], d[o + i * R], d[o + i * R + R - 1]);
        border.sort((a, b) => a - b);
        bg[j] = Math.max(0.1, border[Math.floor((border.length - 1) / 2)]);
        for (let p = 0; p < RR; p++) { m = Math.max(m, d[o + p]); sum += d[o + p]; lgs += lgamma(d[o + p] + 1); }
        I[j] = Math.max(10, sum - bg[j] * RR);
        if (m > best) { best = m; peak = j; }
      }
      st.zoff[k] = (peak - (nz - 1) / 2) * opt.dz_um * opt.sign;
      st.I.push(I); st.bg.push(bg);
      this.lg.push(lgs);
    }
    this.st = st;
  }

  private focus(k: number, j: number, logdz = this.st.logdz, zoff = this.st.zoff[k]) {
    const nz = this.stacks[this.beads[k].stack].nz;
    return this.opt.sign * this.opt.dz_um * Math.exp(logdz) * (j - (nz - 1) / 2) - zoff + this.opt.ze_um;
  }

  private driftBasis(nz: number, j: number, d: number) { const tt = nz > 1 ? -1 + 2 * j / (nz - 1) : 0; return tt ** (d + 1); }

  private imageParams(sel: number[], st: BeadFitState): Float32Array {
    const out: number[] = [];
    for (const k of sel) {
      const s = this.beads[k].stack, nz = this.stacks[s].nz;
      for (let j = 0; j < nz; j++) {
        let dx = 0, dy = 0;
        if (this.opt.fitDrift) for (let d = 0; d < 4; d++) {
          const b = this.driftBasis(nz, j, d);
          dx += st.drift[s * 8 + d] * b; dy += st.drift[s * 8 + 4 + d] * b;
        }
        const f = this.opt.sign * this.opt.dz_um * Math.exp(st.logdz) * (j - (nz - 1) / 2) - st.zoff[k] + this.opt.ze_um;
        out.push(st.x[k] + dx, st.y[k] + dy, this.opt.ze_um, f, st.I[k][j], st.bg[k][j], 0, 0);
      }
    }
    return new Float32Array(out);
  }

  private cloneState(s: BeadFitState): BeadFitState {
    return { zcoef: Float64Array.from(s.zcoef), logdz: s.logdz, drift: Float64Array.from(s.drift), x: Float64Array.from(s.x),
      y: Float64Array.from(s.y), zoff: Float64Array.from(s.zoff), I: s.I.map((a) => Float64Array.from(a)), bg: s.bg.map((a) => Float64Array.from(a)) };
  }

  private nimg(sel: number[]) { return sel.reduce((n, k) => n + this.stacks[this.beads[k].stack].nz, 0); }

  private async nll(sel: number[], st: BeadFitState, tok?: StopToken): Promise<number> {
    this.eng.setZernike(st.zcoef);
    this.eng.setImages(this.imageParams(sel, st));
    const r = (await this.eng.run(this.nimg(sel), 'nll', () => tok?.check()))!;
    let s = 0; for (const v of r) s += v;
    return s;
  }

  /** One LM stage over the beads `sel` with the global parameters `glob` free (the others fixed). */
  private async lm(sel: number[], glob: GlobalParam[], maxIter: number, tok: StopToken | undefined,
    onIter: (it: number, nll: number, lambda: number) => void): Promise<{ it: number; converged: boolean; nll: number }> {
    const R = this.R, RR = R * R, eng = this.eng, nA = eng.nA, nG = glob.length;
    const data = new Float32Array(this.nimg(sel) * RR);
    let off = 0;
    for (const k of sel) { data.set(this.rois[k], off); off += this.rois[k].length; }
    eng.setData(data);
    let st = this.st;
    let L = await this.nll(sel, st, tok);
    let lambda = 1e-2;
    const hist: number[] = [L];
    let it = 0, converged = false;
    const pairs = eng.slotPairs, np_ = pairs.length;
    for (; it < maxIter; it++) {
      if (tok?.stopEarly) { tok.check(); break; }
      eng.setZernike(st.zcoef);
      eng.setImages(this.imageParams(sel, st));
      const NE = (await eng.run(this.nimg(sel), 'ne', () => tok?.check()))!;
      // --- per-bead / per-image reduction (float64)
      const SGG = new Float64Array(nG * nG), rG = new Float64Array(nG), dGG = new Float64Array(nG);
      const beadBlocks: { Sll: Float64Array; SlG: Float64Array; rl: Float64Array; dll: Float64Array; imgs: any[] }[] = [];
      let b = 0;
      for (const k of sel) {
        const s = this.beads[k].stack, nz = this.stacks[s].nz;
        const Sll = new Float64Array(9), SlG = new Float64Array(3 * nG), rl = new Float64Array(3), dll = new Float64Array(3);
        const imgs: any[] = [];
        for (let j = 0; j < nz; j++, b++) {
          const ne = NE.subarray(b * eng.nslot, (b + 1) * eng.nslot);
          const H = new Float64Array(nA * nA);
          for (let p = 0; p < np_; p++) { const [a1, a2] = pairs[p]; H[a1 * nA + a2] = ne[p]; H[a2 * nA + a1] = ne[p]; }
          const g = new Float64Array(nA); for (let a = 0; a < nA; a++) g[a] = ne[np_ + a];
          // global map Ag [nA, nG]
          const Ag = new Float64Array(nA * nG);
          glob.forEach((q, c) => {
            if (q.kind === 'zern') Ag[(5 + q.i) * nG + c] = 1;
            else if (q.kind === 'logdz') Ag[4 * nG + c] = this.opt.sign * this.opt.dz_um * Math.exp(st.logdz) * (j - (nz - 1) / 2);
            else if (q.s === s) Ag[(2 + q.c) * nG + c] = this.driftBasis(nz, j, q.d);
          });
          const al = [2, 3, 4], sl = [1, 1, -1];
          const HA = new Float64Array(nA * nG);
          for (let r = 0; r < nA; r++) for (let c = 0; c < nG; c++) {
            let v = 0; for (let a = 0; a < nA; a++) { const w = Ag[a * nG + c]; if (w) v += H[r * nA + a] * w; } HA[r * nG + c] = v;
          }
          const HGG = new Float64Array(nG * nG), gG = new Float64Array(nG);
          for (let c1 = 0; c1 < nG; c1++) {
            for (let a = 0; a < nA; a++) { const w = Ag[a * nG + c1]; if (w) { gG[c1] += w * g[a]; for (let c2 = 0; c2 < nG; c2++) HGG[c1 * nG + c2] += w * HA[a * nG + c2]; } }
          }
          const Hqq = [H[0], H[1], H[nA], H[nA + 1]];
          const Hql = [0, 1].map((q) => al.map((a, i) => H[q * nA + a] * sl[i]));
          const HqG = [0, 1].map((q) => HA.subarray(q * nG, (q + 1) * nG));
          const Hll = al.map((a1, i) => al.map((a2, i2) => H[a1 * nA + a2] * sl[i] * sl[i2]));
          const HlG = al.map((a, i) => Array.from(HA.subarray(a * nG, (a + 1) * nG), (v) => v * sl[i]));
          const gq = [g[0], g[1]], gl = al.map((a, i) => g[a] * sl[i]);
          // damped 2 x 2 inverse
          const d0 = Hqq[0] * (1 + lambda) + 1e-12, d3 = Hqq[3] * (1 + lambda) + 1e-12;
          const det = d0 * d3 - Hqq[1] * Hqq[2];
          const iq = [d3 / det, -Hqq[1] / det, -Hqq[2] / det, d0 / det];
          // X = Hqq^-1 [Hql | HqG | gq]
          const Xl = [0, 1].map((r) => [0, 1, 2].map((c) => iq[r * 2] * Hql[0][c] + iq[r * 2 + 1] * Hql[1][c]));
          const XG = [0, 1].map((r) => Array.from({ length: nG }, (_, c) => iq[r * 2] * HqG[0][c] + iq[r * 2 + 1] * HqG[1][c]));
          const xg = [0, 1].map((r) => iq[r * 2] * gq[0] + iq[r * 2 + 1] * gq[1]);
          for (let i = 0; i < 3; i++) {
            dll[i] += Hll[i][i];
            for (let i2 = 0; i2 < 3; i2++) Sll[i * 3 + i2] += Hll[i][i2] - Hql[0][i] * Xl[0][i2] - Hql[1][i] * Xl[1][i2];
            for (let c = 0; c < nG; c++) SlG[i * nG + c] += HlG[i][c] - Hql[0][i] * XG[0][c] - Hql[1][i] * XG[1][c];
            rl[i] += gl[i] - Hql[0][i] * xg[0] - Hql[1][i] * xg[1];
          }
          for (let c1 = 0; c1 < nG; c1++) {
            dGG[c1] += HGG[c1 * nG + c1];
            for (let c2 = 0; c2 < nG; c2++) SGG[c1 * nG + c2] += HGG[c1 * nG + c2] - HqG[0][c1] * XG[0][c2] - HqG[1][c1] * XG[1][c2];
            rG[c1] += gG[c1] - HqG[0][c1] * xg[0] - HqG[1][c1] * xg[1];
          }
          imgs.push({ iq, Hql, HqG, gq });
        }
        beadBlocks.push({ Sll, SlG, rl, dll, imgs });
      }
      // eliminate the bead blocks
      const bead2: { iS: Float64Array }[] = [];
      for (const bb of beadBlocks) {
        const S = Float64Array.from(bb.Sll);
        for (let i = 0; i < 3; i++) S[i * 4] += lambda * bb.dll[i] + 1e-12;
        const iS = inverse(S, 3);
        bead2.push({ iS });
        // SGG -= SlG^T iS SlG ; rG -= SlG^T iS rl
        const Y = new Float64Array(3 * nG);   // iS SlG
        for (let i = 0; i < 3; i++) for (let c = 0; c < nG; c++) { let v = 0; for (let m = 0; m < 3; m++) v += iS[i * 3 + m] * bb.SlG[m * nG + c]; Y[i * nG + c] = v; }
        const y = [0, 1, 2].map((i) => iS[i * 3] * bb.rl[0] + iS[i * 3 + 1] * bb.rl[1] + iS[i * 3 + 2] * bb.rl[2]);
        for (let c1 = 0; c1 < nG; c1++) {
          for (let c2 = 0; c2 < nG; c2++) { let v = 0; for (let i = 0; i < 3; i++) v += bb.SlG[i * nG + c1] * Y[i * nG + c2]; SGG[c1 * nG + c2] -= v; }
          let v = 0; for (let i = 0; i < 3; i++) v += bb.SlG[i * nG + c1] * y[i]; rG[c1] -= v;
        }
      }
      // try steps with increasing damping (the global and bead damping enter only through lambda here)
      let accepted = false;
      for (let tries = 0; tries < 10; tries++) {
        const A = Float64Array.from(SGG);
        for (let c = 0; c < nG; c++) A[c * nG + c] += lambda * dGG[c] + 1e-12;
        const dG = nG ? solve(A, rG, nG) : new Float64Array(0);
        const ns = this.cloneState(st);
        glob.forEach((q, c) => {
          if (q.kind === 'zern') ns.zcoef[q.i] += clampAbs(dG[c], 0.1);
          else if (q.kind === 'logdz') ns.logdz += clampAbs(dG[c], 0.2);
          else ns.drift[q.s * 8 + q.c * 4 + q.d] += clampAbs(dG[c], 1);
        });
        sel.forEach((k, bi) => {
          const bb = beadBlocks[bi], iS = bead2[bi].iS;
          const r = [0, 1, 2].map((i) => bb.rl[i] - Array.from({ length: nG }, (_, c) => bb.SlG[i * nG + c] * dG[c]).reduce((a, v) => a + v, 0));
          const dl = [0, 1, 2].map((i) => iS[i * 3] * r[0] + iS[i * 3 + 1] * r[1] + iS[i * 3 + 2] * r[2]);
          ns.x[k] += clampAbs(dl[0], 1); ns.y[k] += clampAbs(dl[1], 1); ns.zoff[k] += clampAbs(dl[2], 0.2);
          bb.imgs.forEach((im: any, j: number) => {
            const rq = [0, 1].map((q) => im.gq[q] - im.Hql[q][0] * dl[0] - im.Hql[q][1] * dl[1] - im.Hql[q][2] * dl[2]
              - Array.from({ length: nG }, (_, c) => im.HqG[q][c] * dG[c]).reduce((a: number, v: number) => a + v, 0));
            const dq0 = im.iq[0] * rq[0] + im.iq[1] * rq[1], dq1 = im.iq[2] * rq[0] + im.iq[3] * rq[1];
            ns.I[k][j] = Math.max(1e-2, ns.I[k][j] + dq0);
            ns.bg[k][j] = Math.max(1e-3, ns.bg[k][j] + dq1);
          });
        });
        const Ln = await this.nll(sel, ns, tok);
        if (Number.isFinite(Ln) && Ln < L) {
          st = ns; this.adopt(ns); L = Ln; lambda = Math.max(1e-7, lambda / 10); accepted = true; break;
        }
        lambda *= 10;
      }
      hist.push(L);
      onIter(it + 1, L, lambda);
      if (!accepted) { converged = true; it++; break; }
      const n = hist.length;
      if (n > 5 && (hist[n - 6] - L) < 1e-6 * Math.abs(L + this.lgSum(sel))) { converged = true; it++; break; }
    }
    return { it, converged, nll: L };
  }

  private adopt(ns: BeadFitState) {
    const s = this.st;
    s.zcoef.set(ns.zcoef); s.logdz = ns.logdz; s.drift.set(ns.drift);
    s.x.set(ns.x); s.y.set(ns.y); s.zoff.set(ns.zoff);
    ns.I.forEach((a, k) => s.I[k].set(a)); ns.bg.forEach((a, k) => s.bg[k].set(a));
  }

  private lgSum(sel: number[]) { return sel.reduce((s, k) => s + this.lg[k], 0); }

  /** NLL per pixel (with the lgamma(d + 1) constant) of the beads sel at the current state. */
  async nllPerPx(sel: number[]): Promise<number> {
    const RR = this.R * this.R;
    const data = new Float32Array(this.nimg(sel) * RR);
    let off = 0;
    for (const k of sel) { data.set(this.rois[k], off); off += this.rois[k].length; }
    this.eng.setData(data);
    const L = await this.nll(sel, this.st);
    return (L + this.lgSum(sel)) / (this.nimg(sel) * RR);
  }

  async fit(tok?: StopToken, onProgress?: (p: Progress) => void): Promise<BeadFitResult> {
    const t0 = performance.now();
    const nb = this.beads.length;
    const ho = new Set(this.opt.holdout.filter((k) => k >= 0 && k < nb));
    const train = [...Array(nb).keys()].filter((k) => !ho.has(k));
    if (!train.length) throw new Error('no beads left for the fit (all held out)');
    const held = [...ho];
    const glob: GlobalParam[] = this.st.zcoef.length ? Array.from(this.st.zcoef, (_, i) => ({ kind: 'zern' as const, i })) : [];
    if (this.opt.fitDz) glob.push({ kind: 'logdz' });
    const trainStacks = new Set(train.map((k) => this.beads[k].stack));
    const driftOf = (stacks: Set<number>) => {
      const g: GlobalParam[] = [];
      if (this.opt.fitDrift) for (const s of stacks) for (let c = 0; c < 2; c++) for (let d = 0; d < 4; d++) g.push({ kind: 'drift', s, c, d });
      return g;
    };
    glob.push(...driftOf(trainStacks));
    const eta = new Eta();
    const RR = this.R * this.R;
    const ntr = this.nimg(train) * RR;
    const r1 = await this.lm(train, glob, this.opt.maxIter, tok, (it, L, lam) => {
      onProgress?.({ stage: 'fit', done: it, total: this.opt.maxIter, info: { nll_px: (L + this.lgSum(train)) / ntr, lambda: lam }, ...eta.update(it, this.opt.maxIter) });
    });
    const partial = !!tok?.stopEarly;
    if (tok) tok.check();
    const nllTrain = (r1.nll + this.lgSum(train)) / ntr;
    let nllHo: number | null = null, itHo = 0;
    if (held.length) {
      // held-out beads: per-bead parameters refitted with the PSF frozen (and the drift of stacks with no training bead)
      const hs = new Set(held.map((k) => this.beads[k].stack));
      const g2 = driftOf(new Set([...hs].filter((s) => !trainStacks.has(s))));
      const tok2 = new StopToken();
      if (tok?.signal === 'abort') tok2.abort();
      const nho = this.nimg(held) * RR;
      const r2 = await this.lm(held, g2, 60, tok2, (it, L) => {
        onProgress?.({ stage: 'held-out', done: it, total: 60, info: { heldout_nll_px: (L + this.lgSum(held)) / nho } });
      });
      itHo = r2.it;
      nllHo = (r2.nll + this.lgSum(held)) / nho;
    }
    const model = await this.modelImages();
    return { state: this.st, nll_train_px: nllTrain, nll_heldout_px: nllHo, iterations: r1.it, heldout_iterations: itHo,
      partial, converged: r1.converged, seconds: (performance.now() - t0) / 1000, model };
  }

  /** Expected counts mu of every bead at the current state: [nz, R, R] each. */
  async modelImages(): Promise<Float32Array[]> {
    const all = [...this.beads.keys()];
    this.eng.setZernike(this.st.zcoef);
    const P = this.imageParams(all, this.st);
    this.eng.setImages(P);
    const B = this.nimg(all), RR = this.R * this.R;
    await this.eng.run(B, 'psf');
    const psf = await this.eng.readPsf(B);
    const out: Float32Array[] = [];
    let b = 0;
    for (const k of all) {
      const nz = this.stacks[this.beads[k].stack].nz;
      const m = new Float32Array(nz * RR);
      for (let j = 0; j < nz; j++, b++) {
        const I = P[8 * b + 4], bg = P[8 * b + 5];
        for (let p = 0; p < RR; p++) m[j * RR + p] = I * psf[b * RR + p] + bg;
      }
      out.push(m);
    }
    return out;
  }

  destroy() { this.eng.destroy(); }
}

function clampAbs(v: number, m: number) { return Math.max(-m, Math.min(m, v)); }
