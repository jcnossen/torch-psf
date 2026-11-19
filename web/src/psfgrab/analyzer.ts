// Main-thread side of the analysis worker: one request in flight per call, results matched by id.
import type { FrameAnalysis } from './spectrum';

export class Analyzer {
  private w = new Worker(new URL('./analysis.worker.ts', import.meta.url), { type: 'module' });
  private next = 1;
  private pending = new Map<number, { res: (a: FrameAnalysis & { ms: number }) => void; rej: (e: Error) => void }>();
  busy = 0;
  constructor() {
    this.w.onmessage = (e: MessageEvent) => {
      const { id, result, ms, error } = e.data;
      const p = this.pending.get(id);
      if (!p) return;
      this.pending.delete(id); this.busy--;
      if (error) p.rej(new Error(error)); else p.res({ ...result, ms });
    };
  }
  /** planes are transferred when `transfer` (the caller must not use them afterwards), else copied. */
  run(planes: (Uint8Array | Float32Array)[], w: number, h: number, maxN: number, transfer = false): Promise<FrameAnalysis & { ms: number }> {
    const id = this.next++;
    this.busy++;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.w.postMessage({ id, w, h, planes, maxN }, transfer ? [...new Set(planes.map((p) => p.buffer as ArrayBuffer))] : []);
    });
  }
}
