// GPU point-cloud renderer (see kernels/cloud.wgsl). Points are appended as they stream in: the point and flag
// buffers have spare capacity and are only reallocated (doubling, old contents copied on the GPU) when they are full,
// so a live update writes just the new points. Filters (max CRLB) switch the per-point enabled flag on the GPU.
// Spatial index (for zooming in on large data sets): a uniform grid over the field (x, y); the buffer holds a part
// sorted by grid cell (counting sort; cellStart[c] .. cellStart[c + 1]) followed by an unsorted tail of the newest
// points. A draw issues one instanced draw per run of visible cells (cells whose projected 3D bounding box meets the
// screen) plus the tail, so off-screen points cost nothing. The tail is folded into the sorted part (full re-sort and
// one upload) when it exceeds max(65536, half the sorted part): the result is the same as a fresh index.
import src from './kernels/cloud.wgsl?raw';

const SU = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
const STRIDE = 8;          // floats per point

export interface CloudUniforms {
  W: number; H: number; scale: number; cx: number; cy: number; X0: number; Y0: number; Z0: number;
  tilt: number; rot: number; zExag: number; mode: 'precision' | 'fixed'; blur: number; cap: number;
  keep: number;              // point cap (above it a fixed-rank subset of the visible points is drawn)
  vmaxPx: number; colorZ: boolean; zlo: number; zhi: number;
}

export class GpuCloud {
  private ctx: GPUCanvasContext;
  private surface: OffscreenCanvas;
  private output: ImageBitmapRenderingContext;
  private fmt: GPUTextureFormat;
  private accFmt: GPUTextureFormat;
  private pts: GPUBuffer; private flags: GPUBuffer; private uni: GPUBuffer;
  private cap = 0;
  n = 0;
  private host: Float32Array = new Float32Array(0);    // CPU copy of the points, in buffer order
  private nSorted = 0;                                // points in the sorted (indexed) part
  private grid: { nx: number; ny: number; cs: number; x0: number; y0: number; start: Uint32Array; zlo: Float32Array; zhi: Float32Array } | null = null;
  private field = { w: 0, h: 0 };                     // field size (nm) for the grid
  lastDraw = { visible: 0, drawn: 0, calls: 0 };
  private splat: GPURenderPipeline; private color: GPURenderPipeline;
  private acc: GPUTexture | null = null;
  reallocations = 0;

  constructor(private dev: GPUDevice, canvas: HTMLCanvasElement) {
    // Present through bitmaprenderer: direct WebGPU canvas presentation can stay blank in Chrome even
    // when the render passes succeed. Transferring the offscreen bitmap keeps presentation on the GPU.
    this.surface = new OffscreenCanvas(canvas.width, canvas.height);
    this.ctx = this.surface.getContext('webgpu') as GPUCanvasContext;
    this.output = canvas.getContext('bitmaprenderer')!;
    this.fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: dev, format: this.fmt, alphaMode: 'opaque' });
    this.accFmt = dev.features.has('float32-blendable') ? 'rgba32float' : 'rgba16float';
    const module = dev.createShaderModule({ code: src });
    const add: GPUBlendComponent = { operation: 'add', srcFactor: 'one', dstFactor: 'one' };
    const bgl = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ] });
    const bgl2 = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ] });
    this.splat = dev.createRenderPipeline({ layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format: this.accFmt, blend: { color: add, alpha: add } }] },
      primitive: { topology: 'triangle-list' } });
    this.color = dev.createRenderPipeline({ layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl2] }),
      vertex: { module, entryPoint: 'vs_full' }, fragment: { module, entryPoint: 'fs_color', targets: [{ format: this.fmt }] },
      primitive: { topology: 'triangle-list' } });
    this.uni = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | CD });
    this.pts = dev.createBuffer({ size: 16, usage: SU | CD | CS });
    this.flags = dev.createBuffer({ size: 16, usage: SU | CD | CS });
    this.grow(65536);
  }

  /** Capacity for at least `need` points: doubles, copies the old contents on the GPU (the only reallocation). */
  private grow(need: number) {
    if (need <= this.cap) return;
    let c = Math.max(this.cap, 65536); while (c < need) c *= 2;
    const lim = this.dev.limits.maxStorageBufferBindingSize;
    c = Math.min(c, Math.floor(lim / (STRIDE * 4)));
    if (c < need) throw new Error(`point cloud: ${need} points exceed the GPU's storage limit`);
    const p = this.dev.createBuffer({ size: c * STRIDE * 4, usage: SU | CD | CS });
    const f = this.dev.createBuffer({ size: c * 4, usage: SU | CD | CS });
    if (this.n) {
      const enc = this.dev.createCommandEncoder();
      enc.copyBufferToBuffer(this.pts, 0, p, 0, this.n * STRIDE * 4);
      enc.copyBufferToBuffer(this.flags, 0, f, 0, this.n * 4);
      this.dev.queue.submit([enc.finish()]);
    }
    this.pts.destroy(); this.flags.destroy();
    this.pts = p; this.flags = f;
    const h = new Float32Array(c * STRIDE); h.set(this.host.subarray(0, this.n * STRIDE)); this.host = h;
    if (this.cap) this.reallocations++;
    this.cap = c;
  }

  clear(fieldWnm: number, fieldHnm: number) { this.n = 0; this.nSorted = 0; this.grid = null; this.field = { w: fieldWnm, h: fieldHnm }; }

  /** Sort all points by grid cell (counting sort) and upload them in that order (the tail folded into the index). */
  private rebuild(maxCrlbNm: number) {
    const n = this.n, F = this.field;
    const cs = Math.max(F.w, F.h) / 128, nx = Math.max(1, Math.ceil(F.w / cs)), ny = Math.max(1, Math.ceil(F.h / cs));
    const cell = new Uint32Array(n), count = new Uint32Array(nx * ny + 1);
    for (let i = 0; i < n; i++) {
      const cx = Math.min(nx - 1, Math.max(0, Math.floor(this.host[i * STRIDE] / cs))), cy = Math.min(ny - 1, Math.max(0, Math.floor(this.host[i * STRIDE + 1] / cs)));
      cell[i] = cy * nx + cx; count[cell[i] + 1]++;
    }
    for (let c = 0; c < nx * ny; c++) count[c + 1] += count[c];
    const start = count.slice(), pos = count.slice(0, nx * ny);
    const sorted = new Float32Array(this.host.length);
    const zlo = new Float32Array(nx * ny).fill(Infinity), zhi = new Float32Array(nx * ny).fill(-Infinity);
    for (let i = 0; i < n; i++) {
      const c = cell[i], j = pos[c]++;
      sorted.set(this.host.subarray(i * STRIDE, (i + 1) * STRIDE), j * STRIDE);
      const z = this.host[i * STRIDE + 2]; if (z < zlo[c]) zlo[c] = z; if (z > zhi[c]) zhi[c] = z;
    }
    this.host = sorted; this.nSorted = n;
    this.grid = { nx, ny, cs, x0: 0, y0: 0, start, zlo, zhi };
    this.dev.queue.writeBuffer(this.pts, 0, this.host.buffer, 0, n * STRIDE * 4);
    this.setFilter(maxCrlbNm);
    this.rebuilds++;
  }
  rebuilds = 0;
  /** Fold the unsorted tail into the index (when the stream pauses), so zoomed-in views cull everything. */
  fold(maxCrlbNm: number) { if (this.n > this.nSorted && this.field.w > 0) this.rebuild(maxCrlbNm); }
  get capacity() { return this.cap; }

  /** Append points [k, 8] = x, y, z, sx, sy, sz (nm), z (um, colour), rank; enabled by the filter maxCrlbNm. */
  append(p: Float32Array, maxCrlbNm: number) {
    const k = p.length / STRIDE;
    if (!k) return;
    this.grow(this.n + k);
    this.dev.queue.writeBuffer(this.pts, this.n * STRIDE * 4, p.buffer, p.byteOffset, p.byteLength);
    const fl = new Uint32Array(k);
    for (let i = 0; i < k; i++) fl[i] = maxCrlbNm > 0 && p[i * STRIDE + 3] > maxCrlbNm ? 0 : 1;
    this.dev.queue.writeBuffer(this.flags, this.n * 4, fl);
    this.host.set(p, this.n * STRIDE);
    this.n += k;
    if (this.n - this.nSorted > Math.max(65536, this.nSorted / 2) && this.field.w > 0) this.rebuild(maxCrlbNm);
  }

  /** Runs of [first, count) instances to draw for the current view: visible grid cells (sorted part) + the tail. */
  private ranges(u: CloudUniforms, ct: number, st: number, cr: number, sr: number): [number, number][] {
    const out: [number, number][] = [];
    const g = this.grid;
    if (g && this.nSorted) {
      const margin = 3 * Math.max(u.cap, u.blur);                   // splat reach (nm)
      const halfW = u.W / 2 * u.scale + margin, halfH = u.H / 2 * u.scale + margin;
      let runStart = -1, runEnd = -1;
      for (let cy = 0; cy < g.ny; cy++) for (let cx = 0; cx < g.nx; cx++) {
        const c = cy * g.nx + cx, a = g.start[c], b = g.start[c + 1];
        if (a === b) continue;
        // projected bounding box of the cell's 3D box (8 corners)
        let umin = Infinity, umax = -Infinity, vmin = Infinity, vmax = -Infinity;
        for (let q = 0; q < 8; q++) {
          const X = (cx + (q & 1)) * g.cs - u.X0, Y = (cy + ((q >> 1) & 1)) * g.cs - u.Y0, Z = ((q & 4) ? g.zhi[c] : g.zlo[c]) - u.Z0;
          const uu = X * cr + Y * sr, vv = (-X * sr + Y * cr) * ct - Z * u.zExag * st;
          umin = Math.min(umin, uu); umax = Math.max(umax, uu); vmin = Math.min(vmin, vv); vmax = Math.max(vmax, vv);
        }
        const vis = umax >= u.cx - halfW && umin <= u.cx + halfW && vmax >= u.cy - halfH && vmin <= u.cy + halfH;
        if (!vis) continue;
        if (runEnd === a) runEnd = b; else { if (runStart >= 0) out.push([runStart, runEnd - runStart]); runStart = a; runEnd = b; }
      }
      if (runStart >= 0) out.push([runStart, runEnd - runStart]);
    } else if (this.nSorted) out.push([0, this.nSorted]);
    if (this.n > this.nSorted) out.push([this.nSorted, this.n - this.nSorted]);
    return out;
  }

  /** Re-evaluate the enabled flags of all points (filter changed): one flag upload, no point upload. */
  setFilter(maxCrlbNm: number) {
    if (!this.n) return;
    const fl = new Uint32Array(this.n);
    for (let i = 0; i < this.n; i++) fl[i] = maxCrlbNm > 0 && this.host[i * STRIDE + 3] > maxCrlbNm ? 0 : 1;
    this.dev.queue.writeBuffer(this.flags, 0, fl);
  }

  /** Points the filter keeps (CPU count, for the cap). */
  enabledCount(maxCrlbNm: number) {
    if (!(maxCrlbNm > 0)) return this.n;
    let c = 0; for (let i = 0; i < this.n; i++) if (this.host[i * STRIDE + 3] <= maxCrlbNm) c++;
    return c;
  }

  draw(u: CloudUniforms) {
    const dev = this.dev, W = Math.round(u.W), H = Math.round(u.H);
    if (this.surface.width !== W) this.surface.width = W;
    if (this.surface.height !== H) this.surface.height = H;
    if (!this.acc || this.acc.width !== W || this.acc.height !== H) {
      this.acc?.destroy();
      this.acc = dev.createTexture({ size: [W, H], format: this.accFmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    }
    const ct = Math.cos(u.tilt * Math.PI / 180), st = Math.sin(u.tilt * Math.PI / 180), cr = Math.cos(u.rot * Math.PI / 180), sr = Math.sin(u.rot * Math.PI / 180);
    const runs = this.ranges(u, ct, st, cr, sr);
    let visible = 0; for (const [, c] of runs) visible += c;
    // u.keep = the point cap: draw a fixed-rank subset of the visible points above it, weighted to keep the brightness
    const keep = visible > u.keep ? u.keep / visible : 1;
    this.lastDraw = { visible, drawn: Math.round(visible * keep), calls: runs.length };
    dev.queue.writeBuffer(this.uni, 0, new Float32Array([W, H, u.scale, u.cx, u.cy, u.X0, u.Y0, u.Z0, ct, st, cr, sr,
      u.zExag, u.mode === 'precision' ? 0 : 1, u.blur, u.cap, keep, 1 / keep, u.vmaxPx, u.colorZ ? 1 : 0, u.zlo, u.zhi, 0, 0]));
    const enc = dev.createCommandEncoder();
    const p1 = enc.beginRenderPass({ colorAttachments: [{ view: this.acc.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    if (this.n) {
      p1.setPipeline(this.splat);
      p1.setBindGroup(0, dev.createBindGroup({ layout: this.splat.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uni } }, { binding: 1, resource: { buffer: this.pts } }, { binding: 2, resource: { buffer: this.flags } }] }));
      for (const [first, count] of runs) p1.draw(6, count, 0, first);
    }
    p1.end();
    const p2 = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    p2.setPipeline(this.color);
    p2.setBindGroup(0, dev.createBindGroup({ layout: this.color.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uni } }, { binding: 3, resource: this.acc.createView() }] }));
    p2.draw(3);
    p2.end();
    dev.queue.submit([enc.finish()]);
    this.output.transferFromImageBitmap(this.surface.transferToImageBitmap());
  }
}
