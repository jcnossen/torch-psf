// Spline knots from the vectorial model on the GPU, and the vector-Jacobian product of that build with respect to the
// Zernike coefficients: torch_psf CubicSplinePSF.knots_from_vectorial / vjp_vectorial (one build plus one backward).
//
// Build (value only, the kernels of VecEngine): images n = (k S + b) S + a with the emitter at x = a / S, y = b / S px,
// z_e = knot z of plane k, focus fixed; phase -> dft1 -> dft2 -> bin, and knot_put writes each image straight into a
// knot buffer [nz, Pl S, Pl S] (the localizer's).
// Backward, given the knot gradient gK on the GPU: vjp_ge (adjoint of knot_put, bin and |E|^2 / 3), vjp_dft2t and
// vjp_dft1t (the transposed matrix DFTs; dft1t also forms dL/dW = Im sum_c A_c conj(PB_c) per pupil point), vjp_acc
// (sum over the images), vjp_zern (projection on the Zernike basis, x 2 pi). With all images in one chunk the fields
// E0 of the build are reused; otherwise each chunk's fields are recomputed before its backward.
import { VecEngine, ImageParams } from './ops';
import { VectorialConfig } from '../psf/vectorial';
import putSrc from './kernels/knot_put.wgsl?raw';
import geSrc from './kernels/vjp_ge.wgsl?raw';
import dft2tSrc from './kernels/vjp_dft2t.wgsl?raw';
import dft1tSrc from './kernels/vjp_dft1t.wgsl?raw';
import accSrc from './kernels/vjp_acc.wgsl?raw';
import zernSrc from './kernels/vjp_zern.wgsl?raw';

const SU = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

/** Knot grid of a torch_psf CubicSplinePSF built from a vectorial model of roisize Pl (knots [nz, Pl S, Pl S]). */
export interface KnotGeometry { S: number; Pl: number; nz: number; nv: number; nu: number; z0: number; hz: number; }

export class KnotEngine extends VecEngine {
  readonly g: KnotGeometry;
  readonly nImg: number;
  readonly nK: number;
  readonly nZ: number;
  private chunk: number;
  private nChunks: number;
  private fieldsValid = false;     // one chunk: E0 holds the fields of the last build (the backward overwrites them)

  /** cfg.roisize is replaced by g.Pl. */
  constructor(dev: GPUDevice, cfg: VectorialConfig, g: KnotGeometry, focus_um: number, opts: { budgetMB?: number; maxChunk?: number } = {}) {
    super(dev, { ...cfg, roisize: g.Pl }, [], { budgetMB: opts.budgetMB, maxChunk: opts.maxChunk ?? Math.ceil(g.nz * g.S * g.S / 4) });
    if (g.nv !== g.Pl * g.S || g.nu !== g.Pl * g.S) throw new Error('KnotEngine: knot grid must be [nz, Pl S, Pl S]');
    this.g = g;
    const S = g.S;
    this.nImg = g.nz * S * S;
    this.nK = g.nz * g.nv * g.nu;
    this.nZ = cfg.zernike_noll.length;
    this.chunk = Math.min(this.chunk1, this.nImg);
    this.nChunks = Math.ceil(this.nImg / this.chunk);
    const ps: ImageParams[] = [];
    for (let k = 0; k < g.nz; k++) for (let b = 0; b < S; b++) for (let a = 0; a < S; a++)
      ps.push({ x: a / S, y: b / S, ze: g.z0 + g.hz * k, f: focus_um, I: 1, bg: 0 });
    this.setImages(ps);
    const NN = this.pd.N * this.pd.N;
    const mk = (name: string, size: number) => {
      this.bufs[name] = dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage: SU | CD | CS, label: name });
    };
    mk('Z', this.nZ * NN * 4);
    mk('gWb', this.chunk * NN * 4);
    mk('gW', NN * 4);
    mk('zg', this.nZ * 4);
    dev.queue.writeBuffer(this.bufs.Z, 0, Float32Array.from(this.pd.Z));
    this.addPipelines({ put: putSrc, ge: geSrc, dft2t: dft2tSrc, dft1t: dft1tSrc, acc: accSrc, zern: zernSrc });
  }

  /** Uniform slots: encodeChunk of the build 2c, 2c + 1; of the recomputation in the backward 2 (nChunks + c) (+ 1);
   *  the knot kernels 4 nChunks + c. Each chunk is a separate submit, so the slots are only written between submits. */
  private knotUniform(c: number, b0: number, B: number): GPUBuffer {
    return this.uniform(4 * this.nChunks + c, { D: 1, B, b0, theta0: 0, S: this.g.S });
  }

  /** Knots [nz, nv, nu] at the current Zernikes (setZernike) into `knots` (a storage buffer of nK floats). Submitted,
   *  not awaited. */
  build(knots: GPUBuffer) {
    if (this.destroyed) throw new Error('engine destroyed');
    const RR = this.R * this.R;
    for (let c = 0; c < this.nChunks; c++) {
      const b0 = c * this.chunk, B = Math.min(this.chunk, this.nImg - b0);
      const enc = this.dev.createCommandEncoder();
      this.encodeChunk(enc, c, b0, B, 1, 'none');
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipes.put);
      pass.setBindGroup(0, this.dev.createBindGroup({ layout: this.pipes.put.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.knotUniform(c, b0, B) } },
        { binding: 1, resource: { buffer: this.bufs.psfJ } }, { binding: 2, resource: { buffer: knots } }] }));
      pass.dispatchWorkgroups(Math.ceil(RR / 64), B);
      pass.end();
      this.dev.queue.submit([enc.finish()]);
    }
    this.fieldsValid = this.nChunks === 1;
  }

  /** (d knots / d zcoef)^T gK for the knot gradient gK [nz, nv, nu] (a storage buffer) at the Zernikes of the last
   *  build(): nZ floats written to `out` at byte offset outOffset (a COPY_DST buffer). Submitted, not awaited. */
  vjp(gK: GPUBuffer, out: GPUBuffer, outOffset = 0) {
    if (this.destroyed) throw new Error('engine destroyed');
    if (this.nChunks === 1 && !this.fieldsValid) throw new Error('KnotEngine.vjp: call build() first (once per build)');
    const { N, ns } = this.pd;
    const NN = N * N, nss = ns * ns;
    for (let c = 0; c < this.nChunks; c++) {
      const b0 = c * this.chunk, B = Math.min(this.chunk, this.nImg - b0);
      const enc = this.dev.createCommandEncoder();
      if (c === 0) enc.clearBuffer(this.bufs.gW);
      if (this.nChunks > 1) this.encodeChunk(enc, this.nChunks + c, b0, B, 1, 'none');   // the fields E0 of this chunk
      const u = this.knotUniform(c, b0, B);
      const pass = enc.beginComputePass();
      const run = (pipe: string, bufs: (string | GPUBuffer)[], x: number, y: number, z = 1) => {
        pass.setPipeline(this.pipes[pipe]);
        pass.setBindGroup(0, this.dev.createBindGroup({ layout: this.pipes[pipe].getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: u } },
          ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: typeof b === 'string' ? this.bufs[b] : b } }))] }));
        pass.dispatchWorkgroups(x, y, z);
      };
      run('ge', [gK, 'E0'], Math.ceil(nss / 64), B);
      run('dft2t', ['F', 'E0', 'T'], Math.ceil(N / 16), Math.ceil(ns / 16), B);
      run('dft1t', ['F', 'T', 'P', 'kzmed', 'pk', 'imgp', 'gWb'], Math.ceil(N / 16), Math.ceil(N / 16), B);
      run('acc', ['gWb', 'gW'], Math.ceil(NN / 64), 1);
      if (c === this.nChunks - 1) run('zern', ['Z', 'gW', 'zg'], this.nZ, 1);
      pass.end();
      if (c === this.nChunks - 1) enc.copyBufferToBuffer(this.bufs.zg, 0, out, outOffset, this.nZ * 4);
      this.dev.queue.submit([enc.finish()]);
    }
    this.fieldsValid = false;
  }

  /** Knots at the current Zernikes, read back [nz, nv, nu]. */
  async knots(): Promise<Float32Array> {
    const bytes = this.nK * 4;
    const kb = this.dev.createBuffer({ size: bytes, usage: SU | CS });
    const rb = this.dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | CD });
    try {
      this.build(kb);
      const enc = this.dev.createCommandEncoder();
      enc.copyBufferToBuffer(kb, 0, rb, 0, bytes);
      this.dev.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap();
      return out;
    } finally { kb.destroy(); rb.destroy(); }
  }
}

/** Knots [nz, nv, nu] of the vectorial model at zcoef (torch_psf knots_from_vectorial, knot z = emitter height). */
export async function vectorialKnots(dev: GPUDevice, cfg: VectorialConfig, zcoef: ArrayLike<number>, g: KnotGeometry, focus_um: number): Promise<Float32Array> {
  const eng = new KnotEngine(dev, cfg, g, focus_um);
  try { eng.setZernike(zcoef); return await eng.knots(); } finally { eng.destroy(); }
}
