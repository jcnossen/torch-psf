// WebGPU adapter / device setup. There is no CPU fallback: without WebGPU the app shows a message.

export interface GpuInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallback: boolean;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupStorageSize: number;
}

export class WebGpuUnavailable extends Error {
  constructor(msg: string) { super(msg); this.name = 'WebGpuUnavailable'; }
}

export const WEBGPU_HELP =
  'This page needs WebGPU: Chrome / Edge 113+ (Windows, macOS, ChromeOS; on Linux start Chrome with ' +
  '--enable-unsafe-webgpu --enable-features=Vulkan), Safari 26+, or Firefox 141+ on Windows. ' +
  'Without WebGPU, use the Python package (pip install torch-psf) for the fit.';

let cached: Promise<{ device: GPUDevice; info: GpuInfo }> | null = null;

/** One shared device per realm (page or worker). */
export function getDevice(): Promise<{ device: GPUDevice; info: GpuInfo }> {
  if (!cached) cached = initDevice();
  return cached;
}

async function initDevice(): Promise<{ device: GPUDevice; info: GpuInfo }> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) throw new WebGpuUnavailable('navigator.gpu is not available. ' + WEBGPU_HELP);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new WebGpuUnavailable('No WebGPU adapter was found. ' + WEBGPU_HELP);
  const L = adapter.limits;
  // Ask for what the adapter has (the defaults, 128 / 256 MiB and 16 KiB, are lower than most GPUs support).
  // float32-blendable: the point-cloud accumulation texture in float32 when the adapter has it (else float16)
  const features: GPUFeatureName[] = adapter.features.has('float32-blendable' as GPUFeatureName) ? ['float32-blendable' as GPUFeatureName] : [];
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxStorageBuffersPerShaderStage: Math.min(L.maxStorageBuffersPerShaderStage, 10),
    },
  });
  device.lost.then((e) => console.error('WebGPU device lost:', e.message));
  const ai = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  const info: GpuInfo = {
    vendor: ai?.vendor ?? '',
    architecture: ai?.architecture ?? '',
    device: ai?.device ?? '',
    description: ai?.description ?? '',
    isFallback: (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? false,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxBufferSize: device.limits.maxBufferSize,
    maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
  };
  return { device, info };
}

export function describeGpu(i: GpuInfo): string {
  const name = [i.vendor, i.architecture, i.device, i.description].filter((s) => s).join(' ');
  return `${name || 'unknown adapter'}${i.isFallback ? ' (software fallback adapter)' : ''}, ` +
    `max storage binding ${(i.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`;
}
