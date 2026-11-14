// GPU engine of the vectorial PSF: per-image pupil phase, two matrix DFTs (value + derivative directions), pixel
// integration, and the Poisson normal equations / NLL per image. Work is submitted in bounded chunks of images.
import common from './kernels/common.wgsl?raw';
import phaseSrc from './kernels/phase.wgsl?raw';
import dft1Src from './kernels/dft1.wgsl?raw';
import dft2Src from './kernels/dft2.wgsl?raw';
import binSrc from './kernels/bin.wgsl?raw';
import neSrc from './kernels/normal_eq.wgsl?raw';
import nllSrc from './kernels/nll.wgsl?raw';
import {
  VectorialConfig, PupilData, pupilData, aberrationPhase, Direction, directionMultipliers,
} from '../psf/vectorial';

const SU = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

/** One image of a batch: emitter (x, y px from the ROI centre, z_e um), focus f (um), intensity, background. */
export interface ImageParams { x: number; y: number; ze: number; f: number; I: number; bg: number; }

export class VecEngine {
  readonly cfg: VectorialConfig;
  readonly pd: PupilData;
  readonly D: number;            // value + derivative directions
  readonly names: string[];
  readonly nA: number;           // atomic columns of the normal equations (D + 1)
  readonly nslot: number;
  readonly R: number;
  private dev: GPUDevice;
  private pipes: Record<string, GPUComputePipeline> = {};
  private bufs: Record<string, GPUBuffer> = {};
  private chunkJ: number;        // images per chunk with derivatives
  private chunk1: number;        // images per chunk, value only
  private Btot = 0;              // capacity of the global per-image buffers
  private uni: GPUBuffer[] = [];
  readonly slotPairs: [number, number][];
  private destroyed = false;

  constructor(dev: GPUDevice, cfg: VectorialConfig, dirs: Direction[], opts: { budgetMB?: number; maxChunk?: number } = {}) {
    this.dev = dev;
    this.cfg = cfg;
    this.pd = pupilData(cfg);
    const { G, names } = directionMultipliers(cfg, this.pd, dirs);
    this.names = names;
    this.D = names.length;
    this.nA = this.D + 1;
    this.R = cfg.roisize;
    const pairs: [number, number][] = [];
    for (let a = 0; a < this.nA; a++) for (let b = a; b < this.nA; b++) pairs.push([a, b]);
    this.slotPairs = pairs;
    this.nslot = pairs.length + this.nA + 1;
    const { N, ns } = this.pd;
    const NN = N * N;
    // chunk sizes from a memory budget (the T buffer dominates) and the binding-size limit
    const budget = Math.min((opts.budgetMB ?? 384) * 2 ** 20, dev.limits.maxStorageBufferBindingSize, dev.limits.maxBufferSize);
    const perJ = Math.max(this.D * 6 * ns * N * 8, 6 * NN * 8, 6 * ns * ns * 8);
    const maxChunk = opts.maxChunk ?? 64;
    this.chunkJ = Math.max(1, Math.min(maxChunk, Math.floor(budget / perJ)));
    this.chunk1 = Math.max(1, Math.min(4 * maxChunk, Math.floor(budget / Math.max(6 * ns * N * 8, 6 * NN * 8))));
    const cap = Math.max(this.chunkJ, this.chunk1);
    const capT = Math.max(this.chunkJ * this.D, this.chunk1);
    const mk = (name: string, size: number, usage = SU | CD | CS) => {
      this.bufs[name] = dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage, label: name });
    };
    mk('P', this.pd.P.byteLength);
    mk('kzmed', this.pd.kzMed.byteLength);
    mk('pk', NN * 16);
    mk('F', this.pd.F.byteLength);
    mk('G', G.byteLength);
    mk('PB', cap * 6 * NN * 8);
    mk('T', capT * 6 * ns * N * 8);
    mk('E0', cap * 6 * ns * ns * 8);
    mk('I', capT * ns * ns * 4);
    mk('psfJ', capT * this.R * this.R * 4);
    const slots = new Uint32Array(this.nslot * 2);
    pairs.forEach(([a, b], k) => { slots[2 * k] = a; slots[2 * k + 1] = b; });
    for (let a = 0; a < this.nA; a++) { slots[2 * (pairs.length + a)] = a; slots[2 * (pairs.length + a) + 1] = 0xffff; }
    slots[2 * (this.nslot - 1)] = 0xffff; slots[2 * (this.nslot - 1) + 1] = 0xffff;
    mk('slots', slots.byteLength);
    const q = dev.queue;
    q.writeBuffer(this.bufs.P, 0, this.pd.P);
    q.writeBuffer(this.bufs.kzmed, 0, this.pd.kzMed);
    q.writeBuffer(this.bufs.F, 0, this.pd.F);
    q.writeBuffer(this.bufs.G, 0, G);
    q.writeBuffer(this.bufs.slots, 0, slots);
    this.setZernike(new Float64Array(cfg.zernike_noll.length));
    const mod = (src: string) => dev.createShaderModule({ code: common + '\n' + src });
    for (const [k, src] of Object.entries({ phase: phaseSrc, dft1: dft1Src, dft2: dft2Src, bin: binSrc, ne: neSrc, nll: nllSrc })) {
      this.pipes[k] = dev.createComputePipeline({ layout: 'auto', compute: { module: mod(src), entryPoint: 'main' }, label: k });
    }
  }

  /** Global Zernike coefficients (waves), in the order of cfg.zernike_noll. */
  setZernike(zcoef: ArrayLike<number>) {
    const W = aberrationPhase(this.pd, zcoef);
    const NN = this.pd.N * this.pd.N, N = this.pd.N;
    const pk = new Float32Array(NN * 4);
    for (let k = 0; k < NN; k++) {
      pk[4 * k] = this.pd.k1[k % N];
      pk[4 * k + 1] = this.pd.k1[Math.floor(k / N)];
      pk[4 * k + 2] = this.pd.kzImm[k];
      pk[4 * k + 3] = W[k];
    }
    this.dev.queue.writeBuffer(this.bufs.pk, 0, pk);
  }

  /** Capacity of the global per-image buffers (image parameters, data, outputs). */
  private ensure(B: number) {
    if (B <= this.Btot) return;
    const RR = this.R * this.R;
    for (const k of ['imgp', 'data', 'NE', 'nllOut', 'psfAll']) this.bufs[k]?.destroy();
    const mk = (name: string, size: number) => {
      this.bufs[name] = this.dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage: SU | CD | CS, label: name });
    };
    mk('imgp', B * 32);
    mk('data', B * RR * 4);
    mk('NE', B * this.nslot * 4);
    mk('nllOut', B * 4);
    mk('psfAll', B * RR * 4);
    this.Btot = B;
  }

  setImages(p: ImageParams[] | Float32Array) {
    let a: Float32Array;
    if (p instanceof Float32Array) a = p;
    else {
      a = new Float32Array(p.length * 8);
      p.forEach((q, i) => { a.set([q.x, q.y, q.ze, q.f, q.I, q.bg, 0, 0], 8 * i); });
    }
    this.ensure(a.length / 8);
    this.dev.queue.writeBuffer(this.bufs.imgp, 0, a);
  }

  /** Data (photons) [B, R*R]; must be called after setImages for the same B. */
  setData(d: Float32Array) {
    this.ensure(d.length / (this.R * this.R));
    this.dev.queue.writeBuffer(this.bufs.data, 0, d);
  }

  private uniform(i: number, v: { D: number; B: number; b0: number; theta0: number }): GPUBuffer {
    while (this.uni.length <= i) this.uni.push(this.dev.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | CD }));
    const ab = new ArrayBuffer(48), u32 = new Uint32Array(ab), f32 = new Float32Array(ab);
    u32.set([this.pd.N, this.pd.ns, this.R, this.cfg.oversample, v.D, v.B, v.b0, v.theta0]);
    f32[8] = this.pd.C; f32[9] = this.cfg.pixel_um;
    u32[10] = this.nA; u32[11] = this.nslot;
    this.dev.queue.writeBuffer(this.uni[i], 0, ab);
    return this.uni[i];
  }

  private bg(pipe: string, ub: GPUBuffer, bufs: string[]): GPUBindGroup {
    return this.dev.createBindGroup({
      layout: this.pipes[pipe].getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: ub } }, ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: this.bufs[b] } }))],
    });
  }

  /** Encode the image pipeline for images b0 .. b0 + B - 1 with D directions (D = 1: value only). */
  private encodeChunk(enc: GPUCommandEncoder, uslot: number, b0: number, B: number, D: number, what: 'ne' | 'nll' | 'none') {
    const { N, ns } = this.pd;
    const NN = N * N, RR = this.R * this.R;
    const u0 = this.uniform(2 * uslot, { D, B, b0, theta0: 0 });
    const pass = enc.beginComputePass();
    const run = (pipe: string, ub: GPUBuffer, bufs: string[], x: number, y: number, z = 1) => {
      pass.setPipeline(this.pipes[pipe]);
      pass.setBindGroup(0, this.bg(pipe, ub, bufs));
      pass.dispatchWorkgroups(x, y, z);
    };
    run('phase', u0, ['P', 'kzmed', 'pk', 'imgp', 'PB'], Math.ceil(NN / 64), B);
    run('dft1', u0, ['F', 'PB', 'G', 'T'], Math.ceil(N / 16), Math.ceil(ns / 16), B * D);
    run('dft2', u0, ['F', 'T', 'E0', 'I'], Math.ceil(ns / 16), Math.ceil(ns / 16), B);
    if (D > 1) {
      const u1 = this.uniform(2 * uslot + 1, { D, B, b0, theta0: 1 });
      run('dft2', u1, ['F', 'T', 'E0', 'I'], Math.ceil(ns / 16), Math.ceil(ns / 16), B * (D - 1));
    }
    run('bin', u0, ['I', 'psfJ'], Math.ceil(RR / 64), B * D);
    if (what === 'ne') run('ne', u0, ['psfJ', 'data', 'imgp', 'slots', 'NE'], Math.ceil(this.nslot / 64), B);
    if (what === 'nll') run('nll', u0, ['psfJ', 'data', 'imgp', 'nllOut'], B, 1);
    pass.end();
    enc.copyBufferToBuffer(this.bufs.psfJ, 0, this.bufs.psfAll, b0 * RR * 4, B * RR * 4);
  }

  private async readback(name: string, bytes: number, extra?: (enc: GPUCommandEncoder, dst: GPUBuffer) => void): Promise<ArrayBuffer> {
    const st = this.dev.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.dev.createCommandEncoder();
    if (extra) extra(enc, st); else enc.copyBufferToBuffer(this.bufs[name], 0, st, 0, Math.ceil(bytes / 4) * 4);
    this.dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = st.getMappedRange().slice(0, bytes);
    st.unmap(); st.destroy();
    return out;
  }

  /**
   * Run all images (setImages / setData first) in chunks.
   * mode 'ne': normal equations per image (needs data) -> NE [B, nslot]; 'nll': NLL per image -> [B];
   * 'psf': value only. The PSF of every image is always in psfAll (read with readPsf).
   * check() is called between chunks (cancellation); it may throw.
   */
  async run(B: number, mode: 'ne' | 'nll' | 'psf', check?: () => void): Promise<Float32Array | null> {
    if (this.destroyed) throw new Error('engine destroyed');
    this.ensure(B);
    const D = mode === 'ne' ? this.D : 1;
    const chunk = mode === 'ne' ? this.chunkJ : this.chunk1;
    for (let b0 = 0, i = 0; b0 < B; b0 += chunk, i++) {
      check?.();
      const enc = this.dev.createCommandEncoder();
      this.encodeChunk(enc, 0, b0, Math.min(chunk, B - b0), D, mode === 'ne' ? 'ne' : mode === 'nll' ? 'nll' : 'none');
      this.dev.queue.submit([enc.finish()]);
      await this.dev.queue.onSubmittedWorkDone();   // bounded chunks: the next one is submitted after this one finished
    }
    if (mode === 'ne') return new Float32Array(await this.readback('NE', B * this.nslot * 4));
    if (mode === 'nll') return new Float32Array(await this.readback('nllOut', B * 4));
    return null;
  }

  async readPsf(B: number): Promise<Float32Array> {
    return new Float32Array(await this.readback('psfAll', B * this.R * this.R * 4));
  }

  /** Images per call of psfAndJacobian. */
  get jacobianChunk(): number { return this.chunkJ; }

  /** Test / export helper: PSF and all derivative images for B <= one chunk: [D, B, R, R]. */
  async psfAndJacobian(params: ImageParams[]): Promise<Float32Array> {
    const B = params.length;
    if (B > this.chunkJ) throw new Error('psfAndJacobian: more images than one chunk');
    this.setImages(params);
    const enc = this.dev.createCommandEncoder();
    this.encodeChunk(enc, 0, 0, B, this.D, 'none');
    this.dev.queue.submit([enc.finish()]);
    const RR = this.R * this.R;
    // psfJ is [D, chunk-B, RR] with the chunk's B
    return new Float32Array(await this.readback('psfJ', this.D * B * RR * 4));
  }

  destroy() {
    this.destroyed = true;
    for (const b of Object.values(this.bufs)) b.destroy();
    for (const b of this.uni) b.destroy();
  }
}
