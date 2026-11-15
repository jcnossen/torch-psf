// Stop token for long tasks: 'run' -> keep going; 'enough' -> stop early and continue with the partial result;
// 'abort' -> cancel and discard. Checked between chunks / iterations, never in the middle of GPU work.

export type StopSignal = 'run' | 'enough' | 'abort';

export class StopToken {
  signal: StopSignal = 'run';
  enough() { if (this.signal === 'run') this.signal = 'enough'; }
  abort() { this.signal = 'abort'; }
  /** Throws an AbortError when aborted. */
  check() { if (this.signal === 'abort') throw new AbortError(); }
  get stopEarly() { return this.signal !== 'run'; }
}

export class AbortError extends Error { constructor() { super('aborted'); this.name = 'AbortError'; } }

export interface Progress {
  stage: string;
  done: number;
  total: number;
  /** stage-specific counters, e.g. { nll_px, lambda, heldout_nll_px } */
  info?: Record<string, number | string>;
  elapsed_s?: number;
  eta_s?: number;
  /** a frame the task has just finished and its spot positions (camera px), for the live movie view */
  view?: { frame: number; x: number[]; y: number[] };
}

/** Moving-average rate estimator for the ETA. */
export class Eta {
  private t0 = performance.now();
  private hist: [number, number][] = [];
  update(done: number, total: number): { elapsed_s: number; eta_s?: number } {
    const t = performance.now();
    this.hist.push([t, done]);
    if (this.hist.length > 10) this.hist.shift();
    const elapsed_s = (t - this.t0) / 1000;
    if (this.hist.length < 2) return { elapsed_s };
    const [ta, da] = this.hist[0];
    const rate = (done - da) / ((t - ta) / 1000);
    return { elapsed_s, eta_s: rate > 0 ? (total - done) / rate : undefined };
  }
}
