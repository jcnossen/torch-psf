import { VecEngine } from '../gpu/ops';
import type { MovieFit } from './movie';
import type { MovieField } from './movie-field';
import type {VectorialModel} from '../psf/zstack';
import { TiffMovie } from '../io/tiff';

// Render at tile centres with the vectorial pupil Jacobian correcting each bead
// to its continuous field position. Six cubature nodes integrate the bead volume.
export async function beadKernels(dev:GPUDevice, model:VectorialModel, field:MovieField, xy:number[][], drift:number[], diameter:number, size:number, visit:(bead:number,x:number,y:number,stamp:Float32Array)=>void){
  const r=diameter/2000,a=r*Math.sqrt(.6),px=model.optics.pixel_um;
  const nodes=[[a/px,0,0],[-a/px,0,0],[0,a/px,0],[0,-a/px,0],[0,0,a],[0,0,-a]];
  const {cx,cy,sx,sy}=field.feats_norm;
  const coords=xy.map(([x,y])=>{
    x+=drift[0];y+=drift[1];
    const u=Math.max(-1,Math.min(1,(x-cx)/sx)),v=Math.max(-1,Math.min(1,(y-cy)/sy));
    return {x,y,u,v,col:Math.round(2+2*u),row:Math.round(2+2*v)};
  });
  const engine=new VecEngine(dev,{...model.optics,npupil:model.npupil,oversample:model.oversample,roisize:size,zernike_noll:model.noll},['zern'],{maxChunk:48});
  const batch=Math.floor(engine.jacobianChunk/6),S=size*size;
  try{
    if(batch<1)throw Error('GPU cannot fit one finite-bead kernel.');
    for(let row=0;row<5;row++)for(let col=0;col<5;col++){
      const group=coords.map((p,j)=>p.col===col&&p.row===row?j:-1).filter(i=>i>=0);
      if(!group.length)continue;
      const uc=(col-2)/2,vc=(row-2)/2;
      engine.setZernike(field.coef.map(([c,v,u])=>c+v*vc+u*uc));
      for(let first=0;first<group.length;first+=batch){
        const ids=group.slice(first,first+batch);
        const params=ids.flatMap(j=>{const {x,y}=coords[j];return nodes.map(([dx,dy,dz])=>({x:x-Math.round(x)+dx,y:y-Math.round(y)+dy,ze:r+dz,f:r+drift[2]/1000,I:1,bg:0}));});
        const psf=await engine.psfAndJacobian(params);
        ids.forEach((j,b)=>{
          const p=coords[j],stamp=new Float32Array(S);
          const weights=[1,...field.coef.map(([,v,u])=>v*(p.v-vc)+u*(p.u-uc))];
          weights.forEach((weight,d)=>{
            for(let k=0;k<6;k++){
              const start=(d*params.length+b*6+k)*S;
              for(let i=0;i<S;i++)stamp[i]+=weight*psf[start+i]/6;
            }
          });
          for(let i=0;i<S;i++)stamp[i]=Math.max(0,stamp[i]);
          visit(j,Math.round(p.x),Math.round(p.y),stamp);
        });
      }
    }
  }finally{engine.destroy();}
}

export function eachPixel(x:number,y:number,side:number,w:number,h:number,cb:(pixel:number,k:number)=>void){
  const radius=(side-1)/2;
  for(let j=Math.max(0,radius-y);j<Math.min(side,h+radius-y);j++)
    for(let i=Math.max(0,radius-x);i<Math.min(side,w+radius-x);i++)
      cb((y+j-radius)*w+x+i-radius,j*side+i);
}

export async function fieldFlux(dev:GPUDevice,file:File,fit:MovieFit,log:(s:string)=>void){
  const movie=await TiffMovie.open([file]),w=fit.width,h=fit.height,side=fit.model.roisize;
  for(let t=0;t<movie.n;t++){
    log('Refitting overlapping bead intensities · frame '+(t+1)+' / '+movie.n);
    const image=await movie.read(t,t+1),flux=fit.intensity[t];
    const stamps:{x:number;y:number;k:Float32Array}[]=[];
    await beadKernels(dev,fit.model,fit.field!,fit.xy,fit.drift[t],fit.settings.diameterNm,side,(j,x,y,k)=>{stamps[j]={x,y,k};});
    const [bx,by]=fit.backgroundSlopes?.[t]??[0,0];
    for(let it=0;it<8;it++){
      const mu=Float32Array.from(image,(_,i)=>fit.background[t]+bx*(2*(i%w)/(w-1)-1)+by*(2*Math.floor(i/w)/(h-1)-1));
      stamps.forEach(({x,y,k},j)=>eachPixel(x,y,side,w,h,(p,s)=>{mu[p]+=k[s]*flux[j];}));
      stamps.forEach(({x,y,k},j)=>{
        let expected=0,assigned=0;
        eachPixel(x,y,side,w,h,(p,s)=>{expected+=k[s];assigned+=k[s]*Math.max(0,(image[p]-fit.settings.offset)*fit.settings.gain)/Math.max(.01,mu[p]);});
        if(expected>0)flux[j]*=assigned/expected;
      });
    }
  }
}
