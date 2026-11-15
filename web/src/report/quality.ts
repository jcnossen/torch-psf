// ROI quality range of the fit report: a per-ROI quality score q (higher = worse), the acceptance cutoff, and the
// few ROIs shown (the best, the last accepted just below the cutoff, the first rejected just above). Shared by the
// worker (in-situ estimate: the ROIs stay there, only the shown ones are sent) and the main thread (bead fit).
// Same selection as torch_psf.psfreport (section 'quality').

export interface QualityRoi {
  q: number; rank: number;                 // rank: 0 = the lowest q
  data: Float32Array; model: Float32Array; width: number; height: number;   // photons; the model of the fit that gave q
  photons?: number | null; bg?: number | null; z?: number | null; label?: string;
}

export interface QualityRange {
  q: Float32Array;                         // every ROI's score, for the histogram
  cutoff: number | null;
  /** 'rejection': the estimator rejected q >= cutoff; 'reference': nothing rejected, a reference line only */
  cutoffKind: 'rejection' | 'reference' | null;
  qLabel: string;                          // x-axis label
  unit: string;                            // what the ROIs are, e.g. 'training spots', 'beads'
  notes: string[];                         // what q and the cutoff are
  best: QualityRoi[]; lastAccepted: QualityRoi[]; firstRejected: QualityRoi[];
}

/** Indices (into q) of the n lowest q, the n accepted with the highest q (not among the best) and the n not accepted
 *  with the lowest q (not among the best); rank[i] = position of q[i] in ascending order (stable). */
export function qualityBands(q: ArrayLike<number>, accepted: (i: number) => boolean, n = 3) {
  const o = Array.from({ length: q.length }, (_, i) => i).sort((a, b) => (q[a] - q[b]) || (a - b));
  const rank = new Int32Array(q.length);
  o.forEach((i, r) => { rank[i] = r; });
  const best = o.slice(0, n), inBest = new Set(best);
  const lastAccepted = o.filter((i) => accepted(i) && !inBest.has(i)).slice(-n);
  const firstRejected = o.filter((i) => !accepted(i) && !inBest.has(i)).slice(0, n);
  return { best, lastAccepted, firstRejected, rank };
}

/** numpy.median */
export function median(a: ArrayLike<number>): number {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** Reference line where the estimator rejects nothing: median + 3 x 1.4826 MAD. */
export function referenceCutoff(q: ArrayLike<number>): number {
  const m = median(q);
  return m + 3 * 1.4826 * median(Float64Array.from(q, (v) => Math.abs(v - m)));
}
