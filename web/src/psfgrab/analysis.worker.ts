// PSF grab analysis worker: spectrum + sharpness of one frame (spectrum.ts analyzePlanes) per message, so the
// FFTs do not block the camera view. Request { id, w, h, planes, maxN } -> { id, result } or { id, error }.
import { analyzePlanes } from './spectrum';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
ctx.onmessage = (e: MessageEvent) => {
  const { id, w, h, planes, maxN } = e.data as { id: number; w: number; h: number; planes: ArrayLike<number>[]; maxN: number };
  try {
    const t0 = performance.now();
    const result = analyzePlanes(planes, w, h, maxN);
    ctx.postMessage({ id, result, ms: performance.now() - t0 }, result.profiles.map((p) => p.buffer));
  } catch (err) {
    ctx.postMessage({ id, error: String((err as Error)?.message ?? err) });
  }
};
