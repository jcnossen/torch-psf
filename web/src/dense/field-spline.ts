import { VecEngine, type ImageParams } from '../gpu/ops';
import type { VectorialModel } from '../psf/zstack';
import type { MovieField } from './movie-field';
import shader from './field-spline.wgsl?raw';

// Mirror-boundary tridiagonal solve along one axis of [focus, y, x].
// This is the same system as torch_psf.spline.bspline_prefilter.
function prefilter(a: Float64Array, n: number, stride: number) {
  const upper = new Float64Array(n), inv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    inv[i] = 1 / (4 - (i ? (i === n-1 ? 2 : 1)*upper[i-1] : 0));
    upper[i] = (i === 0 ? 2 : 1)*inv[i];
  }
  for (let block = 0; block < a.length; block += n*stride) {
    for (let off = 0; off < stride; off++) {
      let k = block + off;
      a[k] *= 6*inv[0];
      for (let i = 1; i < n; i++) {
        k += stride;
        a[k] = (6*a[k] - (i === n-1 ? 2 : 1)*a[k-stride])*inv[i];
      }
      for (let i = n-2; i >= 0; i--) {
        k -= stride;
        a[k] -= upper[i]*a[k+stride];
      }
    }
  }
}

// Cubic B-splines of already pixel-integrated bead images. The axial coordinate
// is objective focus drift, while the six fluorescent volume nodes stay fixed.
export async function fieldSplineRenderer(dev: GPUDevice, model: VectorialModel, field: MovieField,
  diameterNm: number, size: number, focusRangeUm: [number, number]) {
  if (!Number.isInteger(size) || size <= 0 || size % 2 === 0 || !Number.isFinite(diameterNm) || diameterNm < 0)
    throw Error('Expected an odd positive stamp size and a nonnegative bead diameter');
  const S = 2, step = .04, padded = size+4, side = padded*S;
  const [lo, hi] = focusRangeUm;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo) throw Error('Invalid focus interval');
  const z0 = lo-2*step, nz = Math.ceil((hi-lo)/step)+5, volume = nz*side*side;
  const rows = model.noll.map((j,i) => {
    const index = field.zernike_noll.indexOf(j);
    return index < 0 ? [model.zcoef[i]] : field.coef[index];
  });
  const varies = (indices: number[]) => rows.some(row => indices.some(k => !!row[k]));
  const nx = varies(field.order === 2 ? [3,4,5] : [2]) ? 5 : 1;
  const ny = varies(field.order === 2 ? [1,2,4] : [1]) ? 5 : 1;
  const { cx, cy, sx, sy } = field.feats_norm;
  const bytes = volume*nx*ny*4;
  if (bytes > Math.min(dev.limits.maxStorageBufferBindingSize, dev.limits.maxBufferSize))
    throw Error('Field spline bank exceeds GPU storage capacity');
  const bank = dev.createBuffer({size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  const radius = diameterNm/2000, a = radius*Math.sqrt(.6), lateral = a/model.optics.pixel_um;
  const nodes = radius ? [[lateral,0,0],[-lateral,0,0],[0,lateral,0],[0,-lateral,0],[0,0,a],[0,0,-a]] : [[0,0,0]];
  try {
    const eng = new VecEngine(dev, {...model.optics, roisize:padded, npupil:model.npupil,
      oversample:model.oversample, zernike_noll:model.noll, apod:model.apod, saf:model.saf}, [], {maxChunk:16});
    try {
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const u = nx === 1 ? 0 : 2*x/(nx-1)-1, v = ny === 1 ? 0 : 2*y/(ny-1)-1;
        const features = field.order === 2 ? [1,v,v*v,u,u*v,u*u] : [1,v,u];
        eng.setZernike(rows.map(row => row.reduce((value,b,k) => value+b*features[k],0)));
        const table = new Float64Array(volume);
        const perPlane = S*S*nodes.length;
        // Submit several adjacent planes together; only one GPU readback per batch.
        for (let first = 0; first < nz; first += 8) {
          const stop = Math.min(nz,first+8);
          const params: ImageParams[] = [];
          for (let z = first; z < stop; z++)
            for (let iy = 0; iy < S; iy++) for (let ix = 0; ix < S; ix++)
              for (const [dx,dy,dz] of nodes)
                params.push({x:(ix-(S-1)/2)/S+dx,y:(iy-(S-1)/2)/S+dy,ze:radius+dz,f:radius+z0+z*step,I:1,bg:0});
          eng.setImages(params);
          await eng.run(params.length,'psf');
          const samples = await eng.readPsf(params.length);
          for (let z = first; z < stop; z++)
            for (let iy = 0; iy < S; iy++) for (let ix = 0; ix < S; ix++)
              for (let k = 0; k < nodes.length; k++) {
                let src = ((z-first)*perPlane+(iy*S+ix)*nodes.length+k)*padded*padded;
                for (let py = 0; py < padded; py++) {
                  let dst = (z*side+py*S+S-1-iy)*side+S-1-ix;
                  for (let px = 0; px < padded; px++, dst += S) table[dst] += samples[src++]/nodes.length;
                }
              }
        }
        prefilter(table,nz,side*side);
        prefilter(table,side,side);
        prefilter(table,side,1);
        dev.queue.writeBuffer(bank,(y*nx+x)*volume*4,Float32Array.from(table));
      }
    } finally { eng.destroy(); }
    const pipe = await dev.createComputePipelineAsync({layout:'auto',compute:{module:dev.createShaderModule({code:shader}),entryPoint:'main'}});
    async function render(xy: number[][], drift: number[], visit: (j:number,x:number,y:number,stamp:Float32Array)=>void) {
      const focus = drift[2]/1000;
      if (focus < lo-1e-8 || focus > hi+1e-8) throw Error('Focus outside cached field spline interval');
      const area = size*size;
      for (let first = 0; first < xy.length; first += 256) {
        const count = Math.min(256,xy.length-first);
        const points = new Float32Array(count*4);
        for (let j = 0; j < count; j++) {
          const x = xy[first+j][0]+drift[0], y = xy[first+j][1]+drift[1];
          points.set([x-Math.round(x),y-Math.round(y),((x-cx)/sx+1)*(nx-1)/2,((y-cy)/sy+1)*(ny-1)/2],4*j);
        }
        const buffer = (size:number,usage:number) => dev.createBuffer({size,usage});
        const U = GPUBufferUsage, cfg = buffer(32,U.UNIFORM|U.COPY_DST), p = buffer(points.byteLength,U.STORAGE|U.COPY_DST);
        const out = buffer(count*area*4,U.STORAGE|U.COPY_SRC), read = buffer(out.size,U.MAP_READ|U.COPY_DST);
        try {
          const meta = new ArrayBuffer(32);
          new Uint32Array(meta).set([size,side,nz,nx,ny,S]);
          new Float32Array(meta)[6] = (focus-z0)/step;
          dev.queue.writeBuffer(cfg,0,meta);
          dev.queue.writeBuffer(p,0,points);
          const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
          pass.setPipeline(pipe);
          pass.setBindGroup(0,dev.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:
            [cfg,bank,p,out].map((b,binding)=>({binding,resource:{buffer:b}}))}));
          pass.dispatchWorkgroups(Math.ceil(area/64),count);
          pass.end();
          enc.copyBufferToBuffer(out,0,read,0,out.size);
          dev.queue.submit([enc.finish()]);
          await read.mapAsync(GPUMapMode.READ);
          const stamps = new Float32Array(read.getMappedRange().slice(0));
          for (let j = 0; j < count; j++)
            visit(first+j,Math.round(xy[first+j][0]+drift[0]),Math.round(xy[first+j][1]+drift[1]),stamps.subarray(j*area,(j+1)*area));
        } finally { cfg.destroy(); p.destroy(); out.destroy(); read.destroy(); }
      }
    }
    return {render,destroy:()=>bank.destroy()};
  } catch (e) { bank.destroy(); throw e; }
}
