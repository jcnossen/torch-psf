/** Joint fit of a scanned, index-matched 3D bead volume. XY in pixels, Z in µm. */
import { VecEngine } from '../gpu/ops';
import { BeadFitter } from '../fit/beads';
import type { Optics } from '../psf/vectorial';
export interface GelScan {
    width: number;
    height: number;
    frames: Float32Array;
    stage: number[];
    scanIndex: number[];
    estimateFocus?: boolean;
    /** Numerical PSF sampling recorded by the acquisition or simulator. */
    sampling?: 'compact' | 'wide';
}
export interface Bead {
    x: number;
    y: number;
    z: number;
    photons: number;
}
export interface GelOptics {
    optics: Optics;
    diameterUm: number;
}
export const modes = [5, 6, 7, 8, 9, 10, 11];

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
export function median(a: number[]) {
    if (!a.length)
        return 0;
    a.sort((x, y) => x - y);
    return (a[(a.length - 1) >> 1] + a[a.length >> 1]) / 2;
}
function gelModel(radius:number, dz:number, halfZ:number, pythonGrid=false) {
    // One zero plane on each side reproduces grid_sample padding at fractional edges.
    const side=2*radius+1, lattice=side*2, area=lattice*lattice;
    const pad=pythonGrid ? 1 : 0, imageSide=pythonGrid ? 129 : side;
    const storedSide=pythonGrid ? 128 : side;
    const zMin=pythonGrid ? -128 : -halfZ, zMax=pythonGrid ? 127 : halfZ;
/** Half-pixel sampling preserves pixel integration at fractional bead positions. */
async function opticalBank(device: GPUDevice, settings: GelOptics, coefficients: number[]) {
    const engine = new VecEngine(device, { ...settings.optics, roisize: imageSide, npupil: pythonGrid ? 96 : 48, oversample: 2, zernike_noll: modes }, []);
    const bank = new Float32Array((2 * halfZ + 1) * area);
    try {
        engine.setZernike(coefficients);
        for (let first = zMin; first <= zMax; first += 10) {
            const images = [];
            for (let z = first; z <= Math.min(first + 9, zMax); z++)
                for (let y = 0; y < 2; y++)
                    for (let x = 0; x < 2; x++)
                        images.push({ x: -x / 2, y: -y / 2, ze: 5, f: 5 + z * dz, I: 1, bg: 0 });
            engine.setImages(images);
            await engine.run(images.length, 'psf');
            const data = await engine.readPsf(images.length);
            for (let j = 0; j < images.length; j++)
                for (let y = 0; y < storedSide; y++)
                    for (let x = 0; x < storedSide; x++)
                        bank[(first + halfZ + (j >> 2)) * area + (pad + 2 * y + ((j >> 1) & 1)) * lattice + pad + 2 * x + (j & 1)] = data[j * imageSide * imageSide + y * imageSide + x];
        }
    }
    finally {
        engine.destroy();
    }
    // Six sphere nodes match the second moments of a uniformly fluorescent bead.
    const offset = settings.diameterUm * Math.sqrt(3 / 5) / 2;
    if (!offset)
        return bank;
    const blurred = new Float32Array(bank.length);
    for (const [axis, step] of [[1, offset * 2 / settings.optics.pixel_um], [lattice, offset * 2 / settings.optics.pixel_um], [area, offset / dz]]) {
        for (const sign of [-1, 1]) {
            const shift = sign * step, k = Math.floor(shift), f = shift - k;
            for (let i = 0; i < bank.length; i++) {
                const n = axis === area ? 2 * halfZ + 1 : lattice, coord = Math.floor(i / axis) % n;
                if (coord + k >= 0 && coord + k + 1 < n)
                    blurred[i] += ((1 - f) * bank[i + k * axis] + f * bank[i + (k + 1) * axis]) / 6;
            }
        }
    }
    if (pythonGrid) {
        for (let z=0;z<2*halfZ+1;z++) for (let y=0;y<lattice;y++) {
            const i=z*area+y*lattice;
            if(z<1 || z>256 || y<1 || y>256) blurred.fill(0,i,i+lattice);
            else {blurred[i]=0; blurred.fill(0,i+257,i+lattice);}
        }
    }
    return blurred;
}
/** Visit the overlapping PSF footprint and its three translation derivatives. */
function footprint(w: number, h: number, bank: Float32Array, x: number, y: number, defocus: number, visit: (p: number, k: number, kx: number, ky: number, kz: number) => void) {
    const z = defocus / dz + halfZ, iz = Math.floor(z);
    if (iz < 0 || iz >= 2 * halfZ)
        return;
    const left = Math.ceil(x - radius), top = Math.ceil(y - radius);
    const xx = (left - x + radius) * 2, yy = (top - y + radius) * 2;
    const ix = Math.floor(xx), iy = Math.floor(yy), fx = xx - ix, fy = yy - iy, fz = z - iz;
    for (let py = Math.max(0, top); py < Math.min(h, top + side - 1); py++) {
        let i = iz * area + (iy + 2 * (py - top)) * lattice + ix + 2 * Math.max(0, -left);
        for (let px = Math.max(0, left); px < Math.min(w, left + side - 1); px++, i += 2) {
            const a = bank[i], b = bank[i + 1], c = bank[i + lattice], d = bank[i + lattice + 1];
            const j = i + area, e = bank[j], f = bank[j + 1], g = bank[j + lattice], v = bank[j + lattice + 1];
            const p = a + (b - a) * fx, q = c + (d - c) * fx, r = e + (f - e) * fx, s = g + (v - g) * fx;
            const low = p + (q - p) * fy, high = r + (s - r) * fy;
            visit(py * w + px, low + (high - low) * fz, -2 * ((1 - fz) * ((1 - fy) * (b - a) + fy * (d - c)) + fz * ((1 - fy) * (f - e) + fy * (v - g))), -2 * ((1 - fz) * (q - p) + fz * (s - r)), (low - high) / dz);
        }
    }
}
// One invocation owns a camera pixel. All beads at this focus contribute to the same expected count.
const sceneShader = `
struct Config { w:u32, h:u32, nt:u32, nb:u32 };
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read> psf:array<f32>;
@group(0) @binding(2) var<storage,read> beads:array<vec4f>;
@group(0) @binding(3) var<storage,read> poses:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=cfg.w||id.y>=cfg.h||id.z>=cfg.nt){return;}
  let pose=poses[id.z]; var v=pose.w;
  for(var j=0u;j<cfg.nb;j++){
    let b=beads[j];let z=(pose.z-b.z)/${dz}+${halfZ}.0;
    let xy=(vec2f(id.xy)-b.xy-pose.xy+vec2f(${Number(radius).toFixed(1)}))*2.0;
    if(z<0.0||z>=${2 * halfZ}.0||any(xy<vec2f(0.0))||any(xy>=vec2f(${2 * (side - 1)}.0))){continue;}
    let q=vec3f(xy,z);let f=fract(q);let ci=vec3u(floor(q));
    let i=ci.z*${area}u+ci.y*${lattice}u+ci.x;let hi=i+${area}u;
    let l=mix(mix(psf[i],psf[i+1u],f.x),mix(psf[i+${lattice}u],psf[i+${lattice + 1}u],f.x),f.y);
    let h=mix(mix(psf[hi],psf[hi+1u],f.x),mix(psf[hi+${lattice}u],psf[hi+${lattice + 1}u],f.x),f.y);
    v+=b.w*mix(l,h,f.z);
  }
  result[(id.z*cfg.h+id.y)*cfg.w+id.x]=v;
}`;
function gpuScene(device: GPUDevice, scan: GelScan) {
    const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: sceneShader }), entryPoint: 'main' } });
    const deriv = device.createComputePipeline({layout:'auto', compute:{module:device.createShaderModule({code:gradientShader}),entryPoint:'main'}});
    return async (bank: Float32Array, beads: Bead[], drift: number[][], bg: Float64Array, background: boolean, observed?:Float32Array) => {
        const { width: w, height: h, stage } = scan, nt = stage.length, buffers: GPUBuffer[] = [];
        const buffer = (size: number, usage: number) => {
            const b = device.createBuffer({ size, usage });
            buffers.push(b);
            return b;
        };
        const upload = (a: Float32Array | Uint32Array, usage: number) => {
            const b = buffer(a.byteLength, usage | GPUBufferUsage.COPY_DST);
            device.queue.writeBuffer(b, 0, a);
            return b;
        };
        try {
            const cfg = upload(new Uint32Array([w, h, nt, beads.length]), GPUBufferUsage.UNIFORM);
            const kernel = upload(bank, GPUBufferUsage.STORAGE);
            const points = upload(Float32Array.from(beads.flatMap(b => [b.x, b.y, b.z, b.photons])), GPUBufferUsage.STORAGE);
            const poses = upload(Float32Array.from(stage.flatMap((z, t) => [drift[t][0], drift[t][1], z + drift[t][2], background ? bg[t] : 0])), GPUBufferUsage.STORAGE);
            const bytes = w * h * nt * 4, gradBytes=observed?nt*beads.length*32:0;
            const result = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), read = buffer(bytes+gradBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
            const group = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [cfg, kernel, points, poses, result].map((b, binding) => ({ binding, resource: { buffer: b } })) });
            const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
            pass.setPipeline(pipe);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), nt);
            pass.end();
            encoder.copyBufferToBuffer(result, 0, read, 0, bytes);
            if(observed){
              const counts=upload(observed,GPUBufferUsage.STORAGE),g=buffer(gradBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
              const bgd=device.createBindGroup({layout:deriv.getBindGroupLayout(0),entries:[cfg,kernel,points,poses,result,counts,g].map((b,binding)=>({binding,resource:{buffer:b}}))});
              const gp=encoder.beginComputePass();gp.setPipeline(deriv);gp.setBindGroup(0,bgd);
              gp.dispatchWorkgroups(beads.length,nt);gp.end();
              encoder.copyBufferToBuffer(g,0,read,bytes,gradBytes);
            }
            device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            return new Float32Array(read.getMappedRange().slice(0));
        }
        finally {
            buffers.forEach(b => b.destroy());
        }
    };
}
// Per-emitter Poisson gradients include every neighbour through the common
// expected camera image. Each workgroup reduces one bead footprint at one stop.
const gradientShader=`
struct Config {w:u32,h:u32,nt:u32,nb:u32};
@group(0) @binding(0) var<uniform> cfg:Config;
@group(0) @binding(1) var<storage,read> bank:array<f32>;
@group(0) @binding(2) var<storage,read> beads:array<vec4f>;
@group(0) @binding(3) var<storage,read> poses:array<vec4f>;
@group(0) @binding(4) var<storage,read> mu:array<f32>;
@group(0) @binding(5) var<storage,read> counts:array<f32>;
@group(0) @binding(6) var<storage,read_write> grad:array<vec4f>;
var<workgroup> rhs:array<vec4f,64>;
var<workgroup> fisher:array<vec4f,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
  let b=beads[id.x];let pose=poses[id.y];let center=b.xy+pose.xy;
  let z=(pose.z-b.z)/${dz}+${halfZ}.0;
  let idx=2u*(id.y*cfg.nb+id.x);
  
  let start=vec2i(ceil(center-vec2f(${radius}))); let origin=max(vec2i(0),start); let extent=max(vec2i(0),min(start+vec2i(${side-1}),vec2i(i32(cfg.w),i32(cfg.h)))-origin);var r=vec4f(0);var h=vec4f(0);
  for(var j=lane;j<u32(extent.x*extent.y) && z>=0.0 && z<${2*halfZ}.0;j+=64u){
    let xy=origin+vec2i(i32(j%u32(extent.x)),i32(j/u32(extent.x)));
    if(any(xy<vec2i(0))||any(xy>=vec2i(i32(cfg.w),i32(cfg.h)))){continue;}
    let q=vec3f(2.0*(vec2f(xy)-center+vec2f(${Number(radius).toFixed(1)})),z);
    let cell=vec3u(floor(q));let f=fract(q);
    let i=cell.z*${area}u+cell.y*${lattice}u+cell.x;let hi=i+${area}u;
    let a=vec4f(bank[i],bank[i+1u],bank[i+${lattice}u],bank[i+${lattice+1}u]);
    let e=vec4f(bank[hi],bank[hi+1u],bank[hi+${lattice}u],bank[hi+${lattice+1}u]);
    let row=mix(vec4f(a.x,a.z,e.x,e.z),vec4f(a.y,a.w,e.y,e.w),f.x);
    let zz=mix(row.xz,row.yw,f.y);let k=mix(zz.x,zz.y,f.z);
    let dx=-2.0*mix(mix(a.y-a.x,a.w-a.z,f.y),mix(e.y-e.x,e.w-e.z,f.y),f.z);
    let dy=-2.0*mix(row.y-row.x,row.w-row.z,f.z);
    let v=vec4f(b.w*dx,b.w*dy,b.w*(zz.x-zz.y)/${dz},k);
    let p=(id.y*cfg.h+u32(xy.y))*cfg.w+u32(xy.x);let wt=1.0/max(mu[p],2.0);
    r+=v*((counts[p]-mu[p])*wt);h+=v*v*wt;
  }
  rhs[lane]=r;fisher[lane]=h;workgroupBarrier();
  for(var k=32u;k>0u;k/=2u){if(lane<k){rhs[lane]+=rhs[lane+k];fisher[lane]+=fisher[lane+k];}workgroupBarrier();}
  if(lane==0u){grad[idx]=rhs[0];grad[idx+1u]=fisher[0];}
}`;
// Filter in XYZ before finding maxima. Comparing the unfiltered peak to its
// neighbours suppresses dim emitters on the shoulder of a brighter neighbour.

    return {opticalBank, footprint, gpuScene, size:(2*halfZ+1)*area};
}
const compactModel=gelModel(13,.12,32), wideModel=gelModel(64.5,.1,129,true);
export const opticalBank=compactModel.opticalBank;
export function footprint(...args:Parameters<typeof compactModel.footprint>){
    return (args[2].length===wideModel.size ? wideModel : compactModel).footprint(...args);
}
function blur3(input: Float32Array, sizes: number[], sigma: number[]) {
    let data = input;
    let stride = 1;
    for (let axis = 0; axis < 3; axis++) {
        const size = sizes[axis], rad = Math.round(4 * sigma[axis]);
        const weights = Array.from({length:2*rad+1}, (_,j)=>Math.exp(-.5*((j-rad)/sigma[axis])**2));
        const sum = weights.reduce((a,b)=>a+b,0);
        const offsets = Array.from({length:size}, (_,i)=>weights.map((_,j)=>{
            const u = ((i+j-rad) % (2*size)+2*size) % (2*size);
            return ((u<size ? u : 2*size-1-u)-i)*stride;
        }));
        const out = new Float32Array(data.length);
        for(let i=0;i<data.length;i++){
            const indices=offsets[Math.floor(i/stride)%size];
            let value=0;
            for(let j=0;j<weights.length;j++) value+=weights[j]*data[i+indices[j]];
            out[i]=value/sum;
        }
        data=out;
        stride*=size;
    }
    return data;
}
function isolatedPeaks(scan: GelScan, means: Float32Array) {
    const { width: w, height: h } = scan, n = w * h;
    const order = scan.stage.map((_, i) => i).filter(i => scan.scanIndex[i] >= 0).sort((a, b) => scan.stage[a] - scan.stage[b]);
    const smooth = order.map(t => {
        const image = new Float32Array(n);
        for (let y = 0; y < h; y++)
            for (let x = 0; x < w; x++) {
                let sum = 0, wt = 0;
                for (let j = -1; j <= 1; j++)
                    for (let i = -1; i <= 1; i++) {
                        if (x + i < 0 || x + i >= w || y + j < 0 || y + j >= h)
                            continue;
                        const v = (i === 0 ? 2 : 1) * (j === 0 ? 2 : 1);
                        sum += v * means[t * n + (y + j) * w + x + i];
                        wt += v;
                    }
                image[y * w + x] = sum / wt;
            }
        return image;
    });
    const beads: Bead[] = [];
    for (let z = 0; z < order.length; z++)
        for (let y = 0; y < h; y++)
            for (let x = 0; x < w; x++) {
                const p = y * w + x, value = smooth[z][p];
                let peak = true, bg = 0, num = 0;
                for (let j = -3; j <= 3; j += 3)
                    for (let i = -3; i <= 3; i += 3) {
                        if (i || j) {
                            bg += smooth[z][clamp(y + j, 0, h - 1) * w + clamp(x + i, 0, w - 1)];
                            num++;
                        }
                    }
                if (value - bg / num < 34)
                    continue;
                for (let k = Math.max(0, z - 2); k <= Math.min(order.length - 1, z + 2) && peak; k++)
                    for (let j = Math.max(0, y - 1); j <= Math.min(h - 1, y + 1) && peak; j++)
                        for (let i = Math.max(0, x - 1); i <= Math.min(w - 1, x + 1); i++)
                            if (smooth[k][j * w + i] > value) {
                                peak = false;
                                break;
                            }
                if (peak)
                    beads.push({ x, y, z: scan.stage[order[z]], photons: Math.max(900, (value - bg / num) * 22) });
            }
    return beads;
}
function detect(scan: GelScan, means: Float32Array) {
    const {width:w,height:h}=scan, n=w*h;
    const order=scan.stage.map((_,i)=>i).filter(i=>scan.scanIndex[i]>=0).sort((a,b)=>scan.stage[a]-scan.stage[b]);
    const volume=new Float32Array(n*order.length);
    order.forEach((t,k)=>volume.set(means.subarray(t*n,(t+1)*n),k*n));
    const shape=[w,h,order.length], signal=blur3(volume,shape,[.65,.65,.85]);
    const bg=blur3(signal,shape,[3,3,3]);
    for(let i=0;i<signal.length;i++)signal[i]-=bg[i];
    const beads:Bead[]=[];
    for(let z=0;z<order.length;z++) for(let y=0;y<h;y++) for(let x=0;x<w;x++){
        const v=signal[z*n+y*w+x];
        if(v<=35)continue;
        let peak=true;
        for(let k=Math.max(0,z-2);k<=Math.min(order.length-1,z+2)&&peak;k++)
          for(let j=Math.max(0,y-1);j<=Math.min(h-1,y+1)&&peak;j++)
            for(let i=Math.max(0,x-1);i<=Math.min(w-1,x+1);i++)
              if(signal[k*n+j*w+i]>v){peak=false;break;}
        if(peak)beads.push({x,y,z:scan.stage[order[z]],photons:Math.max(900,v*22)});
    }
    return beads;
}
function solve(a: number[][], rhs: number[]) {
    const m = a.map((r, i) => [...r, rhs[i]]), n = m.length;
    for (let i = 0; i < n; i++) {
        let best = i;
        for (let j = i + 1; j < n; j++)
            if (Math.abs(m[j][i]) > Math.abs(m[best][i]))
                best = j;
        [m[best], m[i]] = [m[i], m[best]];
        const d = m[i][i];
        if (Math.abs(d) < 1e-14)
            return rhs.map(() => 0);
        for (let j = i; j <= n; j++)
            m[i][j] /= d;
        for (let k = 0; k < n; k++)
            if (k !== i) {
                const f = m[k][i];
                for (let j = i; j <= n; j++)
                    m[k][j] -= f * m[i][j];
            }
    }
    return m.map(r => r[n]);
}
/** Register successive revisits to the same three focus planes. The middle
 * image supplies XY gradients and its two neighbours supply the Z gradient. */
function referenceDrift(scan: GelScan, means: Float32Array) {
    const { width: w, height: h, stage, scanIndex } = scan, n = w * h;
    const refs: number[] = [];
    for (let t = 1; t + 1 < stage.length; t++)
        if (scanIndex[t - 1] < 0 && scanIndex[t] < 0 && scanIndex[t + 1] < 0 && stage[t - 1] < stage[t] && stage[t] < stage[t + 1])
            refs.push(t);
    const motion = stage.map(() => [0, 0, 0]);
    if (refs.length < 2)
        return motion;
    const known = [[0, 0, 0]];
    function sample(t: number, x: number, y: number) {
        const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy, p = t * n + iy * w + ix;
        return (1 - fy) * ((1 - fx) * means[p] + fx * means[p + 1]) + fy * ((1 - fx) * means[p + w] + fx * means[p + w + 1]);
    }
    for (let r = 1; r < refs.length; r++) {
        const prev = refs[r - 1], next = refs[r], delta = [0, 0, 0], step = (stage[prev + 1] - stage[prev - 1]) / 2;
        if (Math.abs(stage[prev] - stage[next]) > 1e-5 || step <= 0)
            throw Error('Reference triplets must revisit the same focus positions.');
        for (let iter = 0; iter < 8; iter++) {
            const A = delta.map(() => [0, 0, 0]), rhs = [0, 0, 0];
            for (let y = 3; y < h - 3; y++)
                for (let x = 3; x < w - 3; x++) {
                    const xx = x - delta[0], yy = y - delta[1];
                    if (xx < 1 || yy < 1 || xx >= w - 2 || yy >= h - 2)
                        continue;
                    const lo = sample(prev - 1, xx, yy), mid = sample(prev, xx, yy), hi = sample(prev + 1, xx, yy);
                    const f = delta[2] / step;
                    const expected = mid + .5 * f * (hi - lo) + .5 * f * f * (hi + lo - 2 * mid);
                    const v = [.5 * (sample(prev, xx - 1, yy) - sample(prev, xx + 1, yy)), .5 * (sample(prev, xx, yy - 1) - sample(prev, xx, yy + 1)), ((hi - lo) * .5 + f * (hi + lo - 2 * mid)) / step];
                    const err = means[next * n + y * w + x] - expected, weight = 1 / Math.max(20, expected);
                    for (let i = 0; i < 3; i++) {
                        rhs[i] += v[i] * err * weight;
                        for (let j = 0; j < 3; j++)
                            A[i][j] += v[i] * v[j] * weight;
                    }
                }
            const d = solve(A, rhs);
            for (let i = 0; i < 3; i++)
                delta[i] += .65 * clamp(d[i], i === 2 ? -.05 : -.35, i === 2 ? .05 : .35);
        }
        known.push(known[r - 1].map((v, i) => v + delta[i]));
    }
    for (let t = 0; t < stage.length; t++) {
        let r = 1;
        while (r < refs.length - 1 && refs[r] < t)
            r++;
        const f = (t - refs[r - 1]) / (refs[r] - refs[r - 1]);
        motion[t] = known[r].map((v, i) => (1 - f) * known[r - 1][i] + f * v);
    }
    const first = motion[0].slice();
    return motion.map(v => v.map((x, i) => x - first[i]));
}
/** Offset of the strongest smoothed PSF spot from the emitter's model coordinates. */
function detectorPeak(bank: Float32Array) {
    const size = 25, center = 12;
    const stamp = new Float32Array(size * size);
    let best = -Infinity, peak = [0, 0, 0];
    for (let iz = -127; iz <= 127; iz++) {
        const z = iz * .1;
        stamp.fill(0);
        footprint(size, size, bank, center, center, z, (p, value) => { stamp[p] = value; });
        for (let y = 2; y < size - 2; y++) {
            for (let x = 2; x < size - 2; x++) {
                const p = y * size + x;
                const core = 4 * stamp[p] + 2 * (stamp[p-1] + stamp[p+1] + stamp[p-size] + stamp[p+size])
                    + stamp[p-size-1] + stamp[p-size+1] + stamp[p+size-1] + stamp[p+size+1];
                if (core > best) {
                    best = core;
                    peak = [x-center, y-center, z];
                }
            }
        }
    }
    return peak;
}

/** Blind pupil seed from the median of independently centred through-focus patches. */
async function initialPupil(device: GPUDevice, scan: GelScan, means: Float32Array, beads: Bead[], settings: GelOptics) {
    const { width: w, height: h } = scan, R = 13, r = 6, depth = 11, mid = 5;
    const times = scan.stage.map((_, i) => i).filter(i => scan.scanIndex[i] >= 0).sort((a, b) => scan.stage[a] - scan.stage[b]);
    const spacing = median(times.slice(1).map((t, i) => scan.stage[t] - scan.stage[times[i]]));
    const patches: Float32Array[] = [];
    for (const b of beads) {
        const z = times.findIndex(t => scan.stage[t] === b.z), x = Math.round(b.x), y = Math.round(b.y);
        if (z < mid || z + mid >= times.length || x < r || y < r || x + r >= w || y + r >= h)
            continue;
        const roi = new Float32Array(depth * R * R);
        for (let j = 0; j < depth; j++) {
            const border: number[] = [];
            for (let py = 0; py < R; py++)
                for (let px = 0; px < R; px++) {
                    const v = means[(times[z + j - mid] * h + y + py - r) * w + x + px - r];
                    roi[(j * R + py) * R + px] = v;
                    if (!py || !px || py === R - 1 || px === R - 1)
                        border.push(v);
                }
            const bg = median(border);
            for (let p = 0; p < R * R; p++)
                roi[j * R * R + p] -= bg;
        }
        const peak = roi[(mid * R + r) * R + r];
        if (peak > 40)
            patches.push(roi.map(v => v / peak));
    }
    if (patches.length < 5)
        return modes.map(() => 0);
    const data = Float32Array.from(patches[0], (_, i) => Math.max(1, 30 + 600 * median(patches.map(p => p[i]))));
    const stack = { name: 'Registered gel patches', nz: depth, H: R, W: R, data };
    const fitter = new BeadFitter(device, [stack], [{ stack: 0, x0: 0, y0: 0 }], [data], { optics: settings.optics, roisize: R,
        npupil: 48, oversample: 2, sign: 1, dz_um: spacing, ze_um: 5, fitDz: false, fitDrift: false, holdout: [], maxIter: 28, zernike_noll: modes });
    try {
        // A registered patch is centered on peak contrast. Express its starting
        // optical focus in the same coordinate system as the vectorial model.
        if (scan.sampling === 'compact') {
            const flatBank = await compactModel.opticalBank(device, settings, modes.map(() => 0));
            fitter.st.zoff[0] -= detectorPeak(flatBank)[2];
        }
        const v = Array.from((await fitter.fit()).state.zcoef);
        return v.every(x => Number.isFinite(x) && Math.abs(x) < .45) ? v : modes.map(() => 0);
    }
    finally {
        fitter.destroy();
    }
}
export async function fitGel(device: GPUDevice, scan: GelScan, settings: GelOptics, progress: (s: string) => void,
    propose?: (scan:GelScan, settings:GelOptics, seeds:Bead[], drift:number[][], bank:Float32Array)=>Bead[]|null, initialCoefficients?: number[],
    observe?: (coefficients:number[],focus:number[])=>void, onPlane?: (index:number)=>void) {
    const { width: w, height: h, stage } = scan, n = w * h, T = stage.length;
    const period = scan.frames.length / (T*n);
    if (!Number.isInteger(period) || period < 1) throw Error('Incomplete illumination cycles.');
    const means = new Float32Array(T * n), bg = new Float64Array(T);
    for (let t = 0; t < T; t++) {
        for (let c = 0; c < period; c++)
            for (let p = 0; p < n; p++)
                means[t * n + p] += scan.frames[(period * t + c) * n + p] / period;
        const sorted = means.slice(t * n, (t + 1) * n).sort();
        bg[t] = sorted[Math.floor(n * .25)];
    }
    const meanPeaks = detect(scan, means);
    const { NA, n_med, wavelength_um, pixel_um } = settings.optics;
    const span = Math.max(...stage)-Math.min(...stage);
    const beamVolume = Math.PI*(1.22*wavelength_um/NA)**2*(2*n_med*wavelength_um/NA**2);
    const crowded = meanPeaks.length*beamVolume > w*h*pixel_um**2*span;
    const compact = scan.sampling === 'compact';
    const {opticalBank,gpuScene} = compact ? compactModel : wideModel;
    let beads = meanPeaks;
    if (beads.length < 8)
        throw Error('Fewer than eight 3D bead candidates. Check the scan spacing and photon calibration.');
    const drift = referenceDrift(scan, means);
    progress('Calibrating the initial pupil from registered through-focus patches');
    let coeff = initialCoefficients?.slice() ?? await initialPupil(device, scan, means, beads, settings);
    const snapshot=()=>observe?.(coeff,stage.map((z,t)=>z+drift[t][2]));
    snapshot();
    // The scan focus at maximum contrast can differ substantially from bead Z
    // under index mismatch. Locate the corresponding peak in the current PSF.
    let bank = await opticalBank(device,settings,coeff);
    let peak = compact ? detectorPeak(bank) : [0,0,0];
    function register(seed: Bead) {
        const t = stage.findIndex((z, i) => scan.scanIndex[i] >= 0 && z === seed.z);
        seed.x -= drift[t][0]+peak[0];
        seed.y -= drift[t][1]+peak[1];
        seed.z += drift[t][2]-peak[2];
        return seed;
    }
    beads.forEach(register);
    const separated = (a:Bead,b:Bead) => Math.hypot((a.x-b.x)*pixel_um,(a.y-b.y)*pixel_um,.65*(a.z-b.z)) > .21;
    const scene = gpuScene(device, scan);
    const render = (kernel: Float32Array, background = true) => scene(kernel, beads, drift, bg, background);
    const objective = (mu: Float32Array) => {
        let loss = 0;
        for (let i = 0; i < mu.length; i++) {
            const m = Math.max(1, mu[i]), v = means[i];
            loss += m - v + (v > 0 ? v * Math.log(v / m) : 0);
        }
        return loss / mu.length;
    };
    const history: number[] = [];
    const pause = () => new Promise<void>(r => setTimeout(r, 0));
    const geometry = async (count: number) => {
        for (let iter = 0; iter < count; iter++) {
            const packed=await scene(bank,beads,drift,bg,true,means);
            const mu=packed.subarray(0,T*n),grad=beads.map(()=>new Float64Array(8));
            const previous = beads.map(b => ({ ...b })), previousDrift = drift.map(d => d.slice()), previousBg = bg.slice();
            const axial = new Float64Array(T * 2);
            const knots = Math.max(3, Math.round((T - 1) / 30) + 1), motion = Array.from({ length: knots }, () => new Float64Array(6));
            for (let t = 0; t < T; t++) {
                const pos = t / (T - 1) * (knots - 1), lo = Math.min(knots - 2, Math.floor(pos)), frac = pos - lo;
                let sb = 0, hb = 0;
                for (let p = t * n; p < (t + 1) * n; p++) {
                    const wt = 1 / Math.max(mu[p], 2);
                    sb += (means[p] - mu[p]) * wt;
                    hb += wt;
                }
                bg[t] = Math.max(0, bg[t] + .45 * sb / hb);
                for (let j = 0; j < beads.length; j++) {
                    const offset=T*n+8*(t*beads.length+j),a=packed.subarray(offset,offset+8);
                    axial[2*t] -= a[2];
                    axial[2*t+1] += a[6];
                    for (let d = 0; d < 8; d++)
                        grad[j][d] += a[d];
                    for (let d = 0; d < 3; d++) {
                        const sign = d === 2 ? -1 : 1;
                        for (let m = 0; m < 2; m++) {
                            const f = m ? frac : 1 - frac;
                            motion[lo + m][d] += f * a[d] * sign;
                            motion[lo + m][3 + d] += f * a[4 + d];
                        }
                    }
                }
            }
            for (let j = 0; j < beads.length; j++) {
                const b = beads[j], a = grad[j];
                b.x += .55 * clamp(a[0] / (a[4] + 1e-6), -.8, .8);
                b.y += .55 * clamp(a[1] / (a[5] + 1e-6), -.8, .8);
                b.z += .55 * clamp(a[2] / (a[6] + 1e-6), -.12, .12);
                b.photons = Math.max(80, b.photons + .65 * clamp(a[3] / (a[7] + 1e-10), -4000, 4000));
            }
            for (let t = 0; t < T; t++) {
                const pos = t / (T - 1) * (knots - 1), lo = Math.min(knots - 2, Math.floor(pos)), frac = pos - lo;
                for (let d = 0; d < 3; d++) {
                    const delta = (k: number) => k ? clamp(motion[k][d] / (motion[k][d + 3] + 1e-8), d === 2 ? -.08 : -.4, d === 2 ? .08 : .4) : 0;
                    if (scan.estimateFocus && d === 2) {
                        if (t) drift[t][2] += .65 * clamp(axial[2*t] / (axial[2*t+1]+1e-8), -.12, .12);
                        continue;
                    }
                    drift[t][d] += .65 * ((1 - frac) * delta(lo) + frac * delta(lo + 1));
                }
            }
            // Nearby candidates share pixels, so the block-diagonal Fisher step can
            // overshoot. Backtrack the whole coupled scene, including background/drift.
            const oldLoss = objective(mu);
            let accepted = false;
            for (let backtrack = 0; backtrack < 7; backtrack++) {
                if (objective(await render(bank)) <= oldLoss) {
                    accepted = true;
                    break;
                }
                for (let j = 0; j < beads.length; j++)
                    for (const key of ['x', 'y', 'z', 'photons'] as const)
                        beads[j][key] = (beads[j][key] + previous[j][key]) / 2;
                for (let t = 0; t < T; t++) {
                    bg[t] = (bg[t] + previousBg[t]) / 2;
                    for (let d = 0; d < 3; d++)
                        drift[t][d] = (drift[t][d] + previousDrift[t][d]) / 2;
                }
            }
            if (!accepted) {
                for (let j = 0; j < beads.length; j++)
                    Object.assign(beads[j], previous[j]);
                for (let t = 0; t < T; t++)
                    drift[t] = previousDrift[t];
                bg.set(previousBg);
                break;
            }
            if (iter % 4 === 0)
                await pause();
        }
    };
    let rounds = crowded ? 9 : 4;
    const extraRounds = scan.estimateFocus && !crowded && !compact ? 2 : 0;
    for (let round = 0; round < rounds + extraRounds; round++) {
        // Estimate the pupil and drift on the 3D difference-of-Gaussians peaks first. An inaccurate
        // initial transfer function displaces or duplicates sideband peaks.
        if (round === 3 && crowded && propose) {
            progress('Resolving overlapping beads with the calibrated illumination sidebands');
            const peaks=propose(scan,settings,beads,drift,bank);
            if(peaks) {
                for(const b of peaks) if(beads.every(other=>separated(b,other))) beads.push(b);
            }
        }
        progress('Refining ' + beads.length + ' overlapping beads and XYZ drift · ' + (round + 1) + '/' + (rounds + extraRounds));
        await geometry(round === 0 ? 20 : rounds > 4 ? 24 : 12);
        snapshot();
        if (round === 1 || (crowded && round === 4) || (!crowded && rounds > 4 && round === 3)) {
            const mu = await render(bank), residual = means.map((v, i) => v - mu[i]);
            beads = beads.filter(b => b.photons > 750);
            const retained = beads.length;
            for (const b of detect(scan, residual).map(register)) {
                if (!beads.some(c => Math.hypot((b.x - c.x) * settings.optics.pixel_um, (b.y - c.y) * settings.optics.pixel_um, .65 * (b.z - c.z)) < .18))
                    beads.push(b);
            }
            // Many new peaks after subtraction indicate an unresolved crowded volume.
            if (!crowded && round === 1 && beads.length > 1.5 * retained)
                rounds = 7;
        }
        if (crowded && round === 6) {
            // Very faint components can model a bright neighbour's truncated
            // PSF. Compare to the local bead brightness before the final pupil fit.
            const kept=beads.filter(b=>{
                const nearby=beads.filter(c=>Math.abs(b.z-c.z)<1 && Math.hypot(b.x-c.x,b.y-c.y)*pixel_um<1);
                return b.photons > Math.max(800,.65*median(nearby.map(c=>c.photons)));
            });
            if(kept.length>=8) beads=kept;
        }
        if (round === rounds + extraRounds - 1)
            break;
        progress('Fitting shared pupil · Noll 5–11 · ' + (round + 1) + '/' + (rounds + extraRounds - 1));
        const mu = await render(bank), cols: Float32Array[] = [];
        for (let m = 0; m < modes.length; m++) {
            const pert = coeff.slice();
            pert[m] += .002;
            const shifted = await opticalBank(device, settings, pert);
            for(let i=0;i<shifted.length;i++) shifted[i]=(shifted[i]-bank[i])/.002;
            cols.push(await render(shifted, false));
        }
        const A = modes.map(() => modes.map(() => 0)), rhs = modes.map(() => 0);
        for (let i = 0; i < mu.length; i++) {
            const wt = 1 / Math.max(2, mu[i]);
            for (let j = 0; j < 7; j++) {
                const v = cols[j][i] * wt;
                rhs[j] += v * (means[i] - mu[i]);
                for (let k = 0; k <= j; k++)
                    A[j][k] += v * cols[k][i];
            }
        }
        const damping = .01 * A.reduce((s, r, j) => s + r[j], 0) / 7;
        for (let j = 0; j < 7; j++) {
            A[j][j] += damping;
            for (let k = 0; k < j; k++)
                A[k][j] = A[j][k];
        }
        const step = solve(A, rhs), scale = Math.max(1, ...step.map(x => Math.abs(x) / .06));
        const old = objective(mu);
        for (const f of [1, .5, .25]) {
            const next = coeff.map((v, i) => v + f * step[i] / scale), candidate = await opticalBank(device, settings, next);
            const loss = objective(await render(candidate));
            if (loss < old) {
                bank = candidate;
                coeff = next;
                if (compact) peak = detectorPeak(bank);
                snapshot();
                history.push(loss);
                break;
            }
        }
    }
    beads = beads.filter(b => b.photons > 800);
    progress('Solving per-exposure intensities on shared pixels');
    const flux = await gelPhotometry(device,scan, bank, beads, drift, await render(bank), progress, onPlane);
    return { beads, coefficients: coeff, history, drift: drift.map(([x, y, z]) => [x * settings.optics.pixel_um, y * settings.optics.pixel_um, z]), flux };
}

// The rows are the weighted bead images followed by six weighted exposure residuals.
// One tile computes both the Gram matrix and the six right hand sides.
const gramShader = `
struct Shape {rows:u32, pixels:u32, beads:u32, pad:u32};
@group(0) @binding(0) var<uniform> shape:Shape;
@group(0) @binding(1) var<storage,read> design:array<f32>;
@group(0) @binding(2) var<storage,read_write> gram:array<f32>;
var<workgroup> left:array<f32,256>;
var<workgroup> right:array<f32,256>;
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) block:vec3u,@builtin(local_invocation_id) lane:vec3u){
  let row=16u*block.y+lane.y; let col=16u*block.x+lane.x;
  if(block.x<block.y){return;}
  let i=lane.y*16u+lane.x;var sum=0.0;
  for(var p=0u;p<shape.pixels;p+=16u){
    let pixel=p+lane.x;
    left[i]=0.0;right[i]=0.0;
    if(pixel<shape.pixels){
      if(row<shape.beads){left[i]=design[row*shape.pixels+pixel];}
      let other=16u*block.x+lane.y;
      if(other<shape.rows){right[i]=design[other*shape.pixels+pixel];}
    }
    workgroupBarrier();
    for(var j=0u;j<16u;j++){sum+=left[16u*lane.y+j]*right[16u*lane.x+j];}
    workgroupBarrier();
  }
  if(row<shape.beads && col<shape.rows){gram[row*shape.rows+col]=sum;}
}`;

/** Cholesky elimination on a regularized symmetric system, with one RHS per exposure.
 * Use doubles for elimination on the host when neighbouring PSFs nearly coincide. */
function solveFlux(upper:Float32Array, active:{b:Bead;i:number}[], rows:Float32Array[]) {
    const n=active.length, stride=n+rows.length, a=Float64Array.from(upper);
    for(let i=0;i<n;i++) a[i*stride+i]+=1/Math.max(100,.8*active[i].b.photons)**2;
    for(let i=0;i<n;i++){
        const offset=i*stride, diag=Math.sqrt(a[offset+i]);
        if(!(diag>0)) throw Error('The regularized bead intensity matrix lost positive definiteness.');
        for(let k=i+1;k<stride;k++) a[offset+k]/=diag;
        for(let j=i+1;j<n;j++){
            const weight=a[offset+j], dest=j*stride;
            for(let k=j;k<stride;k++) a[dest+k]-=weight*a[offset+k];
        }
        a[offset+i]=diag;
    }
    for(let i=n-1;i>=0;i--){
        const start=i*stride;
        for(let c=0;c<rows.length;c++){
            let delta=a[start+n+c];
            for(let j=i+1;j<n;j++) delta-=a[start+j]*a[j*stride+n+c];
            delta/=a[start+i];
            a[start+n+c]=delta;
            rows[c][active[i].i]=Math.max(0,active[i].b.photons+delta);
        }
    }
}

async function gelPhotometry(device:GPUDevice,scan:GelScan,bank:Float32Array,beads:Bead[],
        drift:number[][],mu:Float32Array, progress?: (message:string)=>void, onPlane?: (index:number)=>void) {
    const {width:w,height:h,stage}=scan, pixels=w*h;
    const period=scan.frames.length/(stage.length*pixels);
    const pipe=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:gramShader}),entryPoint:'main'}});
    const flux:Float32Array[]=[];
    for(let t=0;t<stage.length;t++){
        if(t%8===0 || t===stage.length-1) progress?.(`Solving per-exposure intensities on shared pixels at focus stop ${t+1}/${stage.length}`);
        const focus=stage[t]+drift[t][2];
        const active=beads.map((b,i)=>({b,i}));
        const count=active.length,stride=count+period;
        const rows=Array.from({length:period},()=>Float32Array.from(beads,b=>b.photons));
        const devLimit=Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
        if(count>0){
            const bytes=4*count*stride;
            if(bytes>devLimit)throw Error('Too many overlapping beads at this focus for the GPU intensity matrix.');
            const cfg=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
            const gram=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
            const read=device.createBuffer({size:bytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
            const chunk=Math.min(pixels,Math.floor(devLimit/(4*stride)));
            const design=device.createBuffer({size:4*stride*chunk,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
            try {
                const group=device.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:[cfg,design,gram].map((buffer,binding)=>({binding,resource:{buffer}}))});
                const weights=mu.subarray(t*pixels,(t+1)*pixels).map(v=>1/Math.sqrt(Math.max(2,v)));
                let upper:Float32Array|undefined;
                for(let first=0;first<pixels;first+=chunk){
                    const width=Math.min(chunk,pixels-first), values=new Float32Array(stride*width);
                    for(let j=0;j<count;j++){
                        const b=active[j].b;
                        footprint(w,h,bank,b.x+drift[t][0],b.y+drift[t][1],focus-b.z,(p,k)=>{
                            if(p>=first && p<first+width)values[j*width+p-first]=k*weights[p];
                        });
                    }
                    for(let c=0;c<period;c++)for(let k=0;k<width;k++){
                        const p=first+k;
                        values[(count+c)*width+k]=(scan.frames[(period*t+c)*pixels+p]-mu[t*pixels+p])*weights[p];
                    }
                    device.queue.writeBuffer(cfg,0,new Uint32Array([stride,width,count,0]));
                    device.queue.writeBuffer(design,0,values);
                    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
                    pass.setPipeline(pipe);pass.setBindGroup(0,group);
                    pass.dispatchWorkgroups(Math.ceil(stride/16),Math.ceil(count/16));pass.end();
                    encoder.copyBufferToBuffer(gram,0,read,0,bytes);device.queue.submit([encoder.finish()]);
                    await read.mapAsync(GPUMapMode.READ);
                    const part=new Float32Array(read.getMappedRange());
                    if(!upper)upper=part.slice();
                    else for(let i=0;i<part.length;i++)upper[i]+=part[i];
                    read.unmap();
                }
                solveFlux(upper!,active,rows);
            } finally {for(const buffer of [cfg,gram,read,design])buffer.destroy();}
        }
        flux.push(...rows);
        onPlane?.(t);
    }
    return flux;
}
