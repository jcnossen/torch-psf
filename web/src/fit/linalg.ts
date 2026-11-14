// Small dense linear algebra (float64, row-major) for the CPU side of the Schur-complement LM step.

/** Solve A x = b (n x n, row-major) by Gaussian elimination with partial pivoting; A and b are not modified. */
export function solve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const M = Float64Array.from(A), x = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let p = c, best = Math.abs(M[c * n + c]);
    for (let r = c + 1; r < n; r++) { const v = Math.abs(M[r * n + c]); if (v > best) { best = v; p = r; } }
    if (best === 0) { M[c * n + c] = 1e-30; }
    if (p !== c) {
      for (let k = 0; k < n; k++) { const t = M[c * n + k]; M[c * n + k] = M[p * n + k]; M[p * n + k] = t; }
      const t = x[c]; x[c] = x[p]; x[p] = t;
    }
    const d = M[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = M[r * n + c] / d;
      if (f === 0) continue;
      for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k];
      x[r] -= f * x[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let k = r + 1; k < n; k++) s -= M[r * n + k] * x[k];
    x[r] = s / M[r * n + r];
  }
  return x;
}

/** Inverse of a small n x n matrix (via solve on the unit vectors). */
export function inverse(A: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let c = 0; c < n; c++) {
    const e = new Float64Array(n); e[c] = 1;
    const x = solve(A, e, n);
    for (let r = 0; r < n; r++) out[r * n + c] = x[r];
  }
  return out;
}

/** log Gamma(x) for x > 0 (Lanczos, g = 7). */
export function lgamma(x: number): number {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
