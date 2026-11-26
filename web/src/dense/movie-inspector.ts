import {plot} from './field-example';
import {TiffMovie} from '../io/tiff';
import {vectorialZstack} from '../psf/zstack';
import type {MovieFit,MovieFrameObserver} from './movie';

export function movieInspector(out:HTMLElement){
  const el=document.createElement('section');
  el.innerHTML='<p>Movie frame <output></output></p><canvas></canvas><input type="range" min="1" aria-label="Movie frame" hidden><progress aria-label="Frames processed"></progress>';
  const canvas=el.querySelector('canvas')!,slider=el.querySelector('input')!,label=el.querySelector('output')!,progress=el.querySelector('progress')!;
  out.prepend(el);
  let pending: Parameters<MovieFrameObserver> | undefined;
  let animation = 0;
  const frame: MovieFrameObserver = (...args) => {
    pending = args;
    if (animation) return;
    animation = requestAnimationFrame(() => {
      animation = 0;
      const [pixels, w, h, i, n, stage] = pending!;
      pending = undefined;
      plot(canvas, pixels, w, h);
      label.value = i + ' / ' + n;
      progress.max = n;
      progress.value = i;
      progress.setAttribute('aria-label', stage === 'fit' ? 'Frames fitted' : 'Frames read');
    });
  };
  async function finish(dev:GPUDevice,file:File,fit:MovieFit){
    cancelAnimationFrame(animation);
    pending = undefined;
    const movie=await TiffMovie.open([file]),w=fit.width,h=fit.height;
    progress.hidden=true;slider.hidden=false;slider.max=String(movie.n);slider.value='1';
    const caption=document.createElement('p');
    caption.textContent='Fitted bead centres on the camera frame. Click a location to inspect its PSF.';
    el.prepend(caption);
    let epoch=0,selected=[(w-1)/2,(h-1)/2];
    async function draw(){
      const id=++epoch,t=slider.valueAsNumber-1,raw=await movie.read(t,t+1);
      if(id!==epoch)return;
      plot(canvas,raw,w,h);label.value=(t+1)+' / '+movie.n;
      const g=canvas.getContext('2d')!;g.strokeStyle='#53ffe3';g.lineWidth=.45;g.beginPath();
      for(const xy of fit.xy){const x=xy[0]+fit.drift[t][0],y=xy[1]+fit.drift[t][1];g.moveTo(x+1.5,y);g.arc(x,y,1.5,0,2*Math.PI);}
      g.stroke();g.strokeStyle='orange';g.strokeRect(selected[0]-3,selected[1]-3,6,6);
    }
    const panel=document.createElement('figure'),title=document.createElement('figcaption'),xy=document.createElement('canvas'),xz=document.createElement('canvas');
    panel.append(title,xy,xz);el.append(panel);
    let pupilEpoch=0;
    async function local(){
      const id=++pupilEpoch,[x,y]=selected;
      const f=fit.field,n=f?.feats_norm;
      const coef=f&&n?f.coef.map(([c,v,u])=>c+v*(y-n.cy)/n.sy+u*(x-n.cx)/n.sx):fit.model.zcoef;
      const vol=await vectorialZstack(dev,{...fit.model,zcoef:coef},25);
      if(id!==pupilEpoch)return;
      const mid=Math.floor(vol.Z/2),s=vol.W,center=Math.floor(s/2);
      plot(xy,vol.stack.subarray(mid*s*s,(mid+1)*s*s),s,s);
      const slice=Float32Array.from({length:s*vol.Z},(_,i)=>vol.stack[(Math.floor(i/s)*s+center)*s+i%s]);
      plot(xz,slice,s,vol.Z);xy.style.width=xz.style.width='120px';
      title.textContent='Estimated PSF at ('+Math.round(x)+', '+Math.round(y)+') px · XY and XZ';
    }
    canvas.onclick=async e=>{const r=canvas.getBoundingClientRect();selected=[Math.min(w-1,Math.max(0,(e.clientX-r.left)/r.width*w)),Math.min(h-1,Math.max(0,(e.clientY-r.top)/r.height*h))];await Promise.all([draw(),local()]);};
    slider.oninput=()=>void draw();await Promise.all([draw(),local()]);
  }
  return {frame,finish};
}
