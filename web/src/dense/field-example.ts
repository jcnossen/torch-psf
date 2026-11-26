import { getDevice } from '../gpu/device';

import { floatMovieBlob } from '../psfgrab/tiffstack';
import { fieldSplineRenderer } from './field-spline';
import { beadRenderer, eachPixel } from './field-kernels';
import type { MovieField } from './movie-field';
import type { Settings,MovieFit } from './movie';

const frameBuffers = new WeakMap<HTMLCanvasElement, ImageData>();

export function plot(canvas: HTMLCanvasElement, image: Float32Array, w: number, h: number) {
  const ctx = canvas.getContext('2d')!;
  let pixels = frameBuffers.get(canvas);
  if (!pixels || pixels.width !== w || pixels.height !== h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.imageRendering = 'pixelated';
    canvas.style.width = 'min(100%, 320px)';
    pixels = ctx.createImageData(w, h);
    frameBuffers.set(canvas, pixels);
  }
  const stride = Math.max(1, Math.floor(image.length / 4096));
  const sorted = new Float32Array(Math.ceil(image.length / stride));
  for (let i = 0; i < sorted.length; i++) sorted[i] = image[i * stride];
  sorted.sort();
  const lo = sorted[Math.floor(.01 * sorted.length)];
  const range = Math.max(1, sorted[Math.floor(.998 * sorted.length)] - lo);
  const rgba = pixels.data;
  for (let i = 0, k = 0; i < image.length; i++, k += 4) {
    rgba[k] = 255 * (image[i] - lo) / range;
    rgba[k + 1] = rgba[k + 2] = rgba[k];
    rgba[k + 3] = 255;
  }
  ctx.putImageData(pixels, 0, 0);
}

export function exampleMovie(parent:HTMLElement,settings:()=>Settings,select:(file:File)=>void){
  const el=document.createElement('details');
  el.innerHTML='<summary>Example: randomly generated bead movie</summary><label>Frames <input class="frames" type="number" min="1" value="24"></label><label>Beads/µm² <input class="density" type="number" min=".01" max="10" step=".1" value=".9"></label><label>Field variation (mλ at edge) <input class="strength" type="number" min="0" max="150" value="70"></label><label>Movie renderer <select class="renderer"><option value="spline">Field-interpolated B-spline</option><option value="direct">Vectorial</option></select></label><button type="button">Generate movie</button><p aria-live="polite"></p><canvas hidden></canvas><input type="range" aria-label="Example frame" hidden><a hidden download="dense-beads.tif">Download movie TIFF</a>';
  el.querySelectorAll('input').forEach(input=>input.setAttribute('form','generated-movie-options'));
  parent.after(el);
  const button=el.querySelector('button')!,msg=el.querySelector('p')!,view=el.querySelector('canvas')!,slider=el.querySelector<HTMLInputElement>('[type=range]')!,download=el.querySelector('a')!;
  let previewRequest = 0;
  let truth: {file:File;field:MovieField;drift:number[][]}|undefined;
  button.onclick=async()=>{
    const read=(name:string)=>el.querySelector<HTMLInputElement>(name)!.valueAsNumber;
    const n=read('.frames'),density=read('.density'),strength=read('.strength')/1000;
    if(!Number.isSafeInteger(n)||n<1||!(density>=.01&&density<=10&&strength>=0&&strength<=.15)){msg.textContent='Check the example settings.';return;}
    button.disabled=true;
    const previousPreview = slider.oninput, previousMax = slider.max, previousFrame = slider.value;
    const previewHidden = slider.hidden;
    try{
      const s=settings(),size=160,c=(size-1)/2;
      if(!(s.optics.pixel_um>0&&s.optics.NA>0&&s.optics.NA<=s.optics.n_imm&&s.gain>0&&s.diameterNm>=0))throw Error('Check acquisition settings.');
      const zcoef=[.04,.09,-.035,.025,.02,-.015,.04],noll=[5,6,7,8,9,10,11];
      const model={optics:s.optics,roisize:19,npupil:32,oversample:2,noll,zcoef,focus_um:s.diameterNm/2000,z_range_um:[-.4,.4] as [number,number]};
      const field:MovieField={type:'zernike_poly',order:1,zernike_noll:noll,coef:zcoef.map((v,j)=>[v,j===1?strength:0,j===0?strength:0]),feats_norm:{cx:c,cy:c,sx:c,sy:c}};
      const beads=Math.max(1,Math.round(density*(size*s.optics.pixel_um)**2));
      // Reserve renderer workspace, the previous selection, bead coordinates and drift.
      // Allow a second copy of the new movie while the browser assembles its TIFF.
      const budget = 2 * 2**30;
      const fixed = 256 * 2**20 + (truth?.file.size ?? 0) + (truth?.drift.length ?? 0)*256 + beads*256;
      const bytesPerFrame = 2*size*size*Float32Array.BYTES_PER_ELEMENT + 1024;
      const maxFrames = Math.max(0, Math.floor((budget-fixed)/bytesPerFrame));
      if (n > maxFrames) {
        msg.textContent = 'The 2 GiB generator budget allows up to '+maxFrames.toLocaleString()+' frames with these settings.';
        return;
      }
      const {device} = await getDevice();
      const xy=Array.from({length:beads},()=>[Math.random()*size,Math.random()*size]);
      const flux=xy.map(()=>3500+3000*Math.random());
      const normal=()=>Math.sqrt(-2*Math.log(Math.max(1e-12,Math.random())))*Math.cos(2*Math.PI*Math.random());
      const axes=[1.4,1.1,650].map(span=>{
        let x=0;const walk=Array.from({length:n},()=>x+=normal()),mean=walk.reduce((a,b)=>a+b)/n;
        const range = walk.reduce((hi,v)=>Math.max(hi,v),-Infinity) - walk.reduce((lo,v)=>Math.min(lo,v),Infinity);
        return walk.map(v=>range ? (v-mean)*span/range : 0);
      });
      const drift=Array.from({length:n},(_,t)=>axes.map(axis=>axis[t]));
      const planes: Blob[] = [];
      const image = new Float32Array(size*size);
      const useSpline = el.querySelector<HTMLSelectElement>('.renderer')!.value === 'spline';
      msg.textContent = useSpline ? 'Building field B-spline tables�' : 'Preparing vectorial renderer�';
      const focus = drift.map(row => row[2]/1000);
      const renderer = useSpline
        ? await fieldSplineRenderer(device, model, field, s.diameterNm, 19, [Math.min(...focus), Math.max(...focus)])
        : beadRenderer(device, model, field, s.diameterNm, 19);
      ++previewRequest;
      view.hidden=false;slider.hidden=false;slider.min='0';slider.max='0';slider.value='0';
      slider.oninput = async () => {
        const id = ++previewRequest, plane = planes[slider.valueAsNumber];
        if (!plane) return;
        try {
          const bytes = await plane.arrayBuffer();
          if (id === previewRequest) plot(view,new Float32Array(bytes),size,size);
        } catch (e) { if (id === previewRequest) msg.textContent = String(e); }
      };
      try {
        for(let t=0;t<n;t++){
          msg.textContent='Rendering frame '+(t+1)+' / '+n;
          image.fill(8);
          await renderer.render(xy,drift[t],(j,x,y,stamp)=>eachPixel(x,y,19,size,size,(p,k)=>{image[p]+=flux[j]*stamp[k];}));
          for(let p=0;p<image.length;p++){
            const mean=image[p];let counts=0;
            if(mean<30){let prod=Math.random();while(prod>Math.exp(-mean)){counts++;prod*=Math.random();}}
            else counts=Math.max(0,Math.round(mean+Math.sqrt(mean)*normal()));
            image[p]=counts/s.gain+s.offset;
          }
          planes.push(new Blob([image]));
          ++previewRequest;slider.max=String(t);slider.value=String(t);plot(view,image,size,size);
        }
      } finally {
        renderer.destroy();
      }
      const file=new File([floatMovieBlob(planes,size,size)],'dense-beads.tif',{type:'image/tiff'});
      if(download.href)URL.revokeObjectURL(download.href);
      download.href=URL.createObjectURL(file);download.hidden=false;
      truth={file,drift,field};select(file);msg.textContent=beads+' beads · ready to fit';button.textContent='Regenerate';
    } catch(e) {
      ++previewRequest;
      slider.oninput = previousPreview; slider.max = previousMax; slider.value = previousFrame;
      view.hidden = slider.hidden = previewHidden;
      if (!previewHidden) slider.dispatchEvent(new Event('input'));
      msg.textContent = String(e);
    } finally { button.disabled=false; }
  };
  return (file:File)=>truth?.file===file?truth:undefined;
}

export function fieldReport(parent:HTMLElement,fit:MovieFit,truth?:{field:MovieField;drift:number[][]}){
  const p=document.createElement('p'),r=fit.fieldReport;
  if(!r)return;
  p.textContent=r.reason??('Reserved-frame prediction error · shared pupil '+r.global_score?.toPrecision(4)+' · spatial pupil '+r.field_score?.toPrecision(4)+' · selected '+r.selected);
  parent.append(p);
  const panel=document.createElement('div');panel.style.cssText='display:flex;flex-wrap:wrap;gap:12px';
  fit.model.noll.forEach((j,i)=>{
    const fitted=fit.field?.coef[i]??[fit.model.zcoef[i],0,0],gt=truth?.field.coef[i];
    const figure=document.createElement('figure');figure.style.margin='0';
    const label=document.createElement('figcaption');label.textContent='Z'+j+' (mλ RMS)';figure.append(label);
    const range=Math.max(.001,...[fitted,...(gt?[gt]:[])].map(([c,v,u])=>Math.abs(c)+Math.abs(u)+Math.abs(v)));
    for(const [name,row] of [['Fit',fitted],...(gt?[['True',gt]]:[])] as [string,number[]][]){
      const canvas=document.createElement('canvas');canvas.width=80;canvas.height=80;
      const ctx=canvas.getContext('2d')!,im=ctx.createImageData(80,80);
      for(let y=0;y<80;y++)for(let x=0;x<80;x++){
        const val=(row[0]+row[1]*(2*y/79-1)+row[2]*(2*x/79-1))/range;
        im.data.set([255*(1+Math.min(0,val)),255*(1-Math.abs(val)),255*(1-Math.max(0,val)),255],4*(80*y+x));
      }
      ctx.putImageData(im,0,0);canvas.title=name;figure.append(canvas);
    }
    const scale=document.createElement('small');scale.textContent=(gt?'Fit / true · ':'Fit · ')+'±'+Math.round(range*1000);figure.append(document.createElement('br'),scale);panel.append(figure);
  });
  parent.append(panel);
  const canvas=document.createElement('canvas');canvas.width=500;canvas.height=120;
  const ctx=canvas.getContext('2d')!,zs=fit.drift.map(d=>d[2]),gs=truth?.drift.map(d=>d[2]);
  const traces=[zs,...(gs?[gs]:[])].map(v=>v.map(x=>x-v[0]));
  const lo=Math.min(...traces.flat()),hi=Math.max(...traces.flat());
  traces.forEach((trace,i)=>{ctx.strokeStyle=i?'#f18436':'#1778bd';ctx.beginPath();trace.forEach((v,t)=>{const x=8+480*t/(trace.length-1),y=112-104*(v-lo)/Math.max(1,hi-lo);if(t)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();});
  const label=document.createElement('p');label.textContent='Focus relative to frame 1 ('+Math.round(lo)+' to '+Math.round(hi)+' nm) · frame 1–'+zs.length+' · fit in blue'+(gs?', true in orange':'');parent.append(label,canvas);
}
