// Spline PSF localization on the GPU: torch_psf.fit.localize (Levenberg-Marquardt, zstarts starting heights, best
// likelihood wins), crlb and the chi2 of torch_psf.pipeline.localize_spots, for a ConvSplinePSF given by its knots.
import src from './kernels/spline_lm.wgsl?raw';

const SU = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

/** Grid of a torch_psf ConvSplinePSF (knots [nz, nv, nu]). */
export interface SplineGrid {
  roisize: number; S: number; hz: number; z0: number; nz: number; nv: number; nu: number; xmax: number;
  zlo: number; zhi: number;            // z_range (um)
}

/** ConvSplinePSF geometry for roisize P, z_range, oversample S, dz_um hz, zpad, xy_limit (torch_psf defaults). */
export function splineGrid(P: number, zlo: number, zhi: number, S = 2, hz = 0.04, zpad = 2, xmax = 3.0): SplineGrid & { c: number; Pl: number } {
  const h = (P - 1) >> 1;
  const c = Math.ceil(h + xmax) + 2, Pl = 2 * c + 1;
  const nz = Math.round((zhi - zlo) / hz) + 1 + 2 * zpad;
  return { roisize: P, S, hz, z0: zlo - zpad * hz, nz, nv: Pl * S, nu: Pl * S, xmax, zlo, zhi, c, Pl };
}

export interface LocResult {
  theta: Float32Array;   // [N, 5] (x px, y px from the ROI centre, z um, photons, bg)
  nll: Float32Array;     // [N]
  crlb: Float32Array;    // [N, 5]
  chi2: Float32Array;    // [N]
}

/** torch.quantile(x, q) with linear interpolation. */
function quantile(sorted: Float32Array, q: number): number {
  const pos = q * (sorted.length - 1), lo = Math.floor(pos), hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export class SplineLocalizer {
  private dev: GPUDevice;
  private pipe: GPUComputePipeline;
  readonly g: SplineGrid;
  private knots: GPUBuffer;

  constructor(dev: GPUDevice, grid: SplineGrid, knots: Float32Array) {
    this.dev = dev; this.g = grid;
    const module = dev.createShaderModule({ code: src });
    this.pipe = dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'lm' } });
    this.knots = dev.createBuffer({ size: knots.byteLength, usage: SU | CD });
    dev.queue.writeBuffer(this.knots, 0, knots);
  }
  destroy() { this.knots.destroy(); }
  setKnots(knots: Float32Array) { this.dev.queue.writeBuffer(this.knots, 0, knots); }

  /** The spline's z_limits: (z0 + hz, z0 + (nz - 2) hz). */
  get zLimits(): [number, number] { const g = this.g; return [g.z0 + g.hz, g.z0 + (g.nz - 2) * g.hz]; }

  /** torch_psf.fit.initial_theta: x = y = 0, z, photons = sum - bg P^2 (>= 50), bg = 0.2 quantile of the ROI. */
  initialTheta(rois: Float32Array, N: number, z: number): Float32Array {
    const P2 = this.g.roisize ** 2, th = new Float32Array(N * 5);
    for (let n = 0; n < N; n++) {
      const r = rois.subarray(n * P2, (n + 1) * P2);
      const s = Float32Array.from(r).sort();
      const bg = Math.fround(quantile(s, 0.2));
      let sum = 0; for (let i = 0; i < P2; i++) sum += r[i];
      th[n * 5 + 2] = z; th[n * 5 + 3] = Math.max(50, Math.fround(sum - bg * P2)); th[n * 5 + 4] = bg;
    }
    return th;
  }

  /** One LM run from theta0 [N, 5] (lm_poisson; iters 30, lam0 1). var: scalar read-noise variance (e^2). */
  async run(rois: Float32Array, N: number, theta0: Float32Array, o: { iters?: number; fixed?: number[]; var?: number } = {}): Promise<LocResult> {
    const g = this.g, P2 = g.roisize ** 2;
    const mk = (size: number, usage = SU | CD) => this.dev.createBuffer({ size: Math.max(16, size), usage });
    const bR = mk(N * P2 * 4), bT = mk(N * 20), bO = mk(N * 48, SU | CS), bU = mk(80, GPUBufferUsage.UNIFORM | CD);
    const rb = this.dev.createBuffer({ size: Math.max(16, N * 48), usage: GPUBufferUsage.MAP_READ | CD });
    try {
      this.dev.queue.writeBuffer(bR, 0, rois.buffer, rois.byteOffset, N * P2 * 4);
      this.dev.queue.writeBuffer(bT, 0, theta0.buffer, theta0.byteOffset, N * 20);
      const [zmin, zmax] = this.zLimits;
      const oS0 = (-((g.roisize - 1) >> 1) - (-(((Math.ceil((g.roisize - 1) / 2 + g.xmax) + 2) * g.S + g.S - 1) / g.S))) * g.S;
      let fixed = 0; for (const k of o.fixed ?? []) fixed |= 1 << k;
      const bg = this.dev.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: bU } }, { binding: 1, resource: { buffer: this.knots } },
        { binding: 2, resource: { buffer: bR } }, { binding: 3, resource: { buffer: bT } }, { binding: 4, resource: { buffer: bO } }] });
      const enc = this.dev.createCommandEncoder();
      const per = 65535;
      for (let first = 0; first < N; first += per) {
        const dv = new DataView(new ArrayBuffer(80));
        const u32 = [N, g.roisize, g.S, g.nz, g.nv, g.nu, o.iters ?? 30, fixed];
        u32.forEach((v, i) => dv.setUint32(4 * i, v, true));
        [g.z0, g.hz, oS0, g.xmax, zmin, zmax, 1.0, o.var ?? 0].forEach((v, i) => dv.setFloat32(32 + 4 * i, v, true));
        dv.setUint32(64, first, true);
        // one uniform write per dispatch: separate submits keep the order
        this.dev.queue.writeBuffer(bU, 0, dv.buffer);
        const e = first === 0 ? enc : this.dev.createCommandEncoder();
        const pass = e.beginComputePass();
        pass.setPipeline(this.pipe); pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.min(per, N - first));
        pass.end();
        if (first + per < N) this.dev.queue.submit([e.finish()]);
        else { e.copyBufferToBuffer(bO, 0, rb, 0, N * 48); this.dev.queue.submit([e.finish()]); }
      }
      await rb.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap();
      const r: LocResult = { theta: new Float32Array(N * 5), nll: new Float32Array(N), crlb: new Float32Array(N * 5), chi2: new Float32Array(N) };
      for (let n = 0; n < N; n++) {
        for (let k = 0; k < 5; k++) { r.theta[n * 5 + k] = out[n * 12 + k]; r.crlb[n * 5 + k] = out[n * 12 + 6 + k]; }
        r.nll[n] = out[n * 12 + 5]; r.chi2[n] = out[n * 12 + 11];
      }
      return r;
    } finally { for (const b of [bR, bT, bO, bU, rb]) b.destroy(); }
  }

  /** torch_psf.fit.localize without theta0: zstarts heights evenly inside z_range (linspace(lo, hi, zstarts + 2)[1:-1]),
   *  the best likelihood wins (strictly lower replaces). */
  async localize(rois: Float32Array, N: number, o: { iters?: number; zstarts?: number; var?: number; theta0?: Float32Array } = {}): Promise<LocResult> {
    if (o.theta0) return this.run(rois, N, o.theta0, o);
    const zs = o.zstarts ?? 5, { zlo, zhi } = this.g;
    let best: LocResult | null = null;
    for (let k = 1; k <= zs; k++) {
      const z = Math.fround(zlo + (zhi - zlo) * k / (zs + 1));
      const r = await this.run(rois, N, this.initialTheta(rois, N, z), o);
      if (!best) { best = r; continue; }
      for (let n = 0; n < N; n++) if (r.nll[n] < best.nll[n]) {
        best.nll[n] = r.nll[n]; best.chi2[n] = r.chi2[n];
        for (let q = 0; q < 5; q++) { best.theta[n * 5 + q] = r.theta[n * 5 + q]; best.crlb[n * 5 + q] = r.crlb[n * 5 + q]; }
      }
    }
    return best!;
  }
}
