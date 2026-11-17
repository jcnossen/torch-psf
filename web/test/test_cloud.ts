import { test, check } from './harness';
import { GpuCloud, CloudUniforms } from '../src/gpu/cloud';

test('cloud: presented pixels survive redraw, resize and clear', async () => {
  const adapter = await navigator.gpu.requestAdapter();
  check(!!adapter, 'no GPU adapter');
  const features: GPUFeatureName[][] = [[]];
  if (adapter!.features.has('float32-blendable')) features.push(['float32-blendable']);
  for (const requiredFeatures of features) {
    const current = await navigator.gpu.requestAdapter();
    const device = await current!.requestDevice({ requiredFeatures });
    device.pushErrorScope('validation');
    try {
      const canvas = document.createElement('canvas');
      const cloud = new GpuCloud(device, canvas);
      cloud.clear(64, 64);
      cloud.append(new Float32Array([32, 32, 0, 1, 1, 1, 0, 0]), 0);
      const u: CloudUniforms = { W: 64, H: 64, scale: 1, cx: 0, cy: 0, X0: 32, Y0: 32, Z0: 0,
        tilt: 0, rot: 0, zExag: 1, mode: 'fixed', blur: 1, cap: 60, keep: 100,
        vmaxPx: 0.1, colorZ: false, zlo: 0, zhi: 1 };
      const snapshot = () => {
        const copy = document.createElement('canvas');
        copy.width = canvas.width; copy.height = canvas.height;
        const ctx = copy.getContext('2d')!;
        ctx.drawImage(canvas, 0, 0);
        return ctx.getImageData(Math.floor(canvas.width / 2), Math.floor(canvas.height / 2), 1, 1).data;
      };
      for (const [W, H] of [[64, 64], [64, 64], [96, 80]]) {
        canvas.width = W; canvas.height = H;
        cloud.draw({ ...u, W, H });
        // Check the visible canvas, not just the accumulation texture or the draw count.
        const p = snapshot();
        check(p[0] > 100 && p[3] === 255, `blank presented point at ${W}x${H}: ${p}`);
      }
      cloud.clear(64, 64);
      cloud.draw({ ...u, W: 96, H: 80 });
      const p = snapshot();
      check(p[0] === 0 && p[1] === 0 && p[2] === 0 && p[3] === 255, `stale pixels after clear: ${p}`);
      const error = await device.popErrorScope();
      check(!error, error?.message ?? '');
    } finally { device.destroy(); }
  }
  return `presented pixels checked with float16${features.length > 1 ? ' and float32' : ''} accumulation`;
});
