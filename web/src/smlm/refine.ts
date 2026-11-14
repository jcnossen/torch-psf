// Optional cubic-spline refinement of the in-situ PSF: a port of torch_psf.insitu.fit_free_knots as used by
// InSituEstimator(refine=True): free knots K, regularised towards the vectorial spline (ref) and smooth, alternating
// (1) emitter parameters at fixed knots (Levenberg-Marquardt; zstarts and 30 iterations the first time, 5 afterwards)
// and (2) the knots at fixed emitter parameters with a fresh torch.optim.LBFGS (max_iter = inner, history 10, strong
// Wolfe line search, lr 1), `outer` times.
//   loss = NLL(ROIs) + lam_ref sum (K - ref)^2 / (|ref| + 1e-4)
//          + lam_s sum of squared second differences along z, y, x, each weighted by 1 / (|K0| + 1e-4)
// The NLL and its knot gradient come from the GPU (SplineLocalizer.knotGradient), the rest is CPU (float64).
import { SplineLocalizer } from '../gpu/spline_lm';
import { KnotGridFull } from './insitu';
import { StopToken } from '../core/task';

type Vec = Float64Array;
const dot = (a: Vec, b: Vec) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const absMax = (a: Vec) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };
const absSum = (a: Vec) => { let m = 0; for (let i = 0; i < a.length; i++) m += Math.abs(a[i]); return m; };

/** torch.optim.lbfgs._cubic_interpolate */
function cubic(x1: number, f1: number, g1: number, x2: number, f2: number, g2: number, bounds?: [number, number]): number {
  const [lo, hi] = bounds ?? (x1 <= x2 ? [x1, x2] : [x2, x1]);
  const d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2);
  const d2s = d1 * d1 - g1 * g2;
  if (d2s >= 0) {
    const d2 = Math.sqrt(d2s);
    const m = x1 <= x2 ? x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)) : x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2));
    return Math.min(Math.max(m, lo), hi);
  }
  return (lo + hi) / 2;
}

type Obj = (t: number) => Promise<{ f: number; g: Vec }>;

/** torch.optim.lbfgs._strong_wolfe (c1 1e-4, c2 0.9, tolerance_change 1e-9, max_ls 25) along d from x (obj(t) evaluates x + t d). */
async function strongWolfe(obj: Obj, t: number, d: Vec, f: number, g: Vec, gtd: number): Promise<{ f: number; g: Vec; t: number; evals: number }> {
  const c1 = 1e-4, c2 = 0.9, tolChange = 1e-9, maxLs = 25;
  const dNorm = absMax(d);
  let { f: fNew, g: gNew } = await obj(t);
  let evals = 1, gtdNew = dot(gNew, d);
  let tPrev = 0, fPrev = f, gPrev = g, gtdPrev = gtd;
  let done = false, ls = 0;
  let br: number[] = [], brF: number[] = [], brG: Vec[] = [], brGtd: number[] = [];
  while (ls < maxLs) {
    if (fNew > f + c1 * t * gtd || (ls > 1 && fNew >= fPrev)) { br = [tPrev, t]; brF = [fPrev, fNew]; brG = [gPrev, gNew]; brGtd = [gtdPrev, gtdNew]; break; }
    if (Math.abs(gtdNew) <= -c2 * gtd) { br = [t]; brF = [fNew]; brG = [gNew]; done = true; break; }
    if (gtdNew >= 0) { br = [tPrev, t]; brF = [fPrev, fNew]; brG = [gPrev, gNew]; brGtd = [gtdPrev, gtdNew]; break; }
    const minStep = t + 0.01 * (t - tPrev), maxStep = t * 10, tmp = t;
    t = cubic(tPrev, fPrev, gtdPrev, t, fNew, gtdNew, [minStep, maxStep]);
    tPrev = tmp; fPrev = fNew; gPrev = gNew; gtdPrev = gtdNew;
    ({ f: fNew, g: gNew } = await obj(t)); evals++; gtdNew = dot(gNew, d); ls++;
  }
  if (ls === maxLs) { br = [0, t]; brF = [f, fNew]; brG = [g, gNew]; }
  let insuf = false;
  let [lowPos, highPos] = brF[0] <= brF[brF.length - 1] ? [0, 1] : [1, 0];
  while (!done && ls < maxLs) {
    if (Math.abs(br[1] - br[0]) * dNorm < tolChange) break;
    t = cubic(br[0], brF[0], brGtd[0], br[1], brF[1], brGtd[1]);
    const bmax = Math.max(...br), bmin = Math.min(...br), eps = 0.1 * (bmax - bmin);
    if (Math.min(bmax - t, t - bmin) < eps) {
      if (insuf || t >= bmax || t <= bmin) { t = Math.abs(t - bmax) < Math.abs(t - bmin) ? bmax - eps : bmin + eps; insuf = false; }
      else insuf = true;
    } else insuf = false;
    ({ f: fNew, g: gNew } = await obj(t)); evals++; gtdNew = dot(gNew, d); ls++;
    if (fNew > f + c1 * t * gtd || fNew >= brF[lowPos]) {
      br[highPos] = t; brF[highPos] = fNew; brG[highPos] = gNew; brGtd[highPos] = gtdNew;
      [lowPos, highPos] = brF[0] <= brF[1] ? [0, 1] : [1, 0];
    } else {
      if (Math.abs(gtdNew) <= -c2 * gtd) done = true;
      else if (gtdNew * (br[highPos] - br[lowPos]) >= 0) { br[highPos] = br[lowPos]; brF[highPos] = brF[lowPos]; brG[highPos] = brG[lowPos]; brGtd[highPos] = brGtd[lowPos]; }
      br[lowPos] = t; brF[lowPos] = fNew; brG[lowPos] = gNew; brGtd[lowPos] = gtdNew;
    }
  }
  return { f: brF[lowPos], g: brG[lowPos], t: br[lowPos], evals };
}

/** One step() of a fresh torch.optim.LBFGS (lr 1, max_iter, max_eval = max_iter * 5 / 4, tolerance_grad 1e-7,
 *  tolerance_change 1e-9, history_size 10, line_search_fn 'strong_wolfe') on x (updated in place). */
export async function lbfgsStep(x: Vec, closure: (x: Vec) => Promise<{ f: number; g: Vec }>, maxIter: number, history = 10): Promise<number> {
  const maxEval = Math.floor(maxIter * 5 / 4), tolGrad = 1e-7, tolChange = 1e-9;
  let { f: loss, g } = await closure(x);
  const orig = loss;
  let evals = 1;
  if (absMax(g) <= tolGrad) return orig;
  const dirs: Vec[] = [], stps: Vec[] = [], ro: number[] = [];
  let d: Vec = new Float64Array(x.length), t = 0, Hdiag = 1, prevG: Vec | null = null, prevLoss = loss;
  for (let it = 1; it <= maxIter; it++) {
    if (it === 1) { d = g.map((v) => -v); Hdiag = 1; }
    else {
      const y = new Float64Array(x.length), s = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) { y[i] = g[i] - prevG![i]; s[i] = d[i] * t; }
      const ys = dot(y, s);
      if (ys > 1e-10) {
        if (dirs.length === history) { dirs.shift(); stps.shift(); ro.shift(); }
        dirs.push(y); stps.push(s); ro.push(1 / ys); Hdiag = ys / dot(y, y);
      }
      const q = g.map((v) => -v), al = new Array(dirs.length);
      for (let i = dirs.length - 1; i >= 0; i--) { al[i] = dot(stps[i], q) * ro[i]; for (let k = 0; k < q.length; k++) q[k] -= al[i] * dirs[i][k]; }
      d = q.map((v) => v * Hdiag);
      for (let i = 0; i < dirs.length; i++) { const be = dot(dirs[i], d) * ro[i]; for (let k = 0; k < d.length; k++) d[k] += stps[i][k] * (al[i] - be); }
    }
    prevG = Float64Array.from(g); prevLoss = loss;
    t = it === 1 ? Math.min(1, 1 / absSum(g)) : 1;
    const gtd = dot(g, d);
    if (gtd > -tolChange) break;
    const x0 = Float64Array.from(x);
    const r = await strongWolfe(async (tt) => { const xt = new Float64Array(x0.length); for (let i = 0; i < xt.length; i++) xt[i] = x0[i] + tt * d[i]; return closure(xt); }, t, d, loss, g, gtd);
    loss = r.f; g = r.g; t = r.t;
    for (let i = 0; i < x.length; i++) x[i] = x0[i] + t * d[i];
    evals += r.evals;
    if (it === maxIter || evals >= maxEval || absMax(g) <= tolGrad) break;
    let dm = 0; for (let i = 0; i < d.length; i++) dm = Math.max(dm, Math.abs(d[i] * t));
    if (dm <= tolChange || Math.abs(loss - prevLoss) < tolChange) break;
  }
  return orig;
}

export interface RefineResult { knots: Float32Array; theta: Float32Array; history: { it: number; loss: number; seconds: number }[]; seconds: number; }

/** torch_psf.insitu.fit_free_knots(spline, knots0, rois=rois, ref=ref, lam_ref, lam_s, outer, inner) without a bead stack. */
export async function fitFreeKnots(dev: GPUDevice, g: KnotGridFull, K0: Float32Array, ref: Float32Array, rois: Float32Array, N: number,
  o: { lamRef?: number; lamS?: number; outer?: number; inner?: number; tok?: StopToken; log?: (h: { it: number; loss: number; seconds: number }) => void; var?: number } = {}): Promise<RefineResult> {
  const lamRef = o.lamRef ?? 3000, lamS = o.lamS ?? 1000, outer = o.outer ?? 12, inner = o.inner ?? 8, eps = 1e-4;
  const nz = g.nz, nv = g.nv, nu = g.nu, nK = nz * nv * nu;
  const wref = new Float64Array(nK), wsm = new Float64Array(nK);
  for (let i = 0; i < nK; i++) { wref[i] = 1 / (Math.abs(ref[i]) + eps); wsm[i] = 1 / (Math.abs(K0[i]) + eps); }
  const K = Float64Array.from(K0);
  const loc = new SplineLocalizer(dev, g, Float32Array.from(K));
  let th: Float32Array | null = null;
  const hist: { it: number; loss: number; seconds: number }[] = [];
  const t0 = performance.now();
  // regulariser value and gradient (float64; the same sums as the torch closure)
  const reg = (x: Vec, gr: Vec): number => {
    let r = 0;
    for (let i = 0; i < nK; i++) { const dd = x[i] - ref[i]; r += lamRef * wref[i] * dd * dd; gr[i] += 2 * lamRef * wref[i] * dd; }
    const second = (stride: number, n: number, idx: (a: number) => number[]) => {
      // second differences along one axis: for every line, D_k = x[k+1] - 2 x[k] + x[k-1] (k = 1 .. n-2), weight wsm at k
      for (const base of idx(stride)) for (let k = 1; k < n - 1; k++) {
        const a = base + (k - 1) * stride, b = base + k * stride, c = base + (k + 1) * stride;
        const D = x[c] - 2 * x[b] + x[a], w = lamS * wsm[b];
        r += w * D * D; gr[a] += 2 * w * D; gr[b] -= 4 * w * D; gr[c] += 2 * w * D;
      }
    };
    const lines = (axis: number): number[] => {
      const out: number[] = [];
      if (axis === 0) { for (let v = 0; v < nv; v++) for (let u = 0; u < nu; u++) out.push(v * nu + u); }
      else if (axis === 1) { for (let z = 0; z < nz; z++) for (let u = 0; u < nu; u++) out.push(z * nv * nu + u); }
      else { for (let z = 0; z < nz; z++) for (let v = 0; v < nv; v++) out.push((z * nv + v) * nu); }
      return out;
    };
    if (lamS > 0) { second(nv * nu, nz, () => lines(0)); second(nu, nv, () => lines(1)); second(1, nu, () => lines(2)); }
    return r;
  };
  try {
    for (let it = 0; it < outer; it++) {
      o.tok?.check();
      loc.setKnots(Float32Array.from(K));
      th = th ? (await loc.run(rois, N, th, { iters: 5, var: o.var })).theta : (await loc.localize(rois, N, { var: o.var })).theta;
      const thFix = th;
      const closure = async (x: Vec) => {
        loc.setKnots(Float32Array.from(x));
        const { gK, nll } = await loc.knotGradient(rois, N, thFix, o.var ?? 0);
        const gr = Float64Array.from(gK);
        const f = nll + (lamRef > 0 || lamS > 0 ? reg(x, gr) : 0);
        return { f, g: gr };
      };
      const loss = await lbfgsStep(K, closure, inner);
      const h = { it, loss, seconds: (performance.now() - t0) / 1000 };
      hist.push(h); o.log?.(h);
      if (o.tok?.stopEarly) break;
    }
  } finally { loc.destroy(); }
  return { knots: Float32Array.from(K), theta: th!, history: hist, seconds: (performance.now() - t0) / 1000 };
}
