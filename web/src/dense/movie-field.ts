import { VecEngine } from '../gpu/ops';
import { TiffMovie } from '../io/tiff';
import { fft2d } from '../psfgrab/fft';
import { solve } from '../fit/linalg';
import type { MovieFit } from './movie';

export interface MovieField {
  type:'zernike_poly'; order:1; zernike_noll:number[]; coef:number[][];
  feats_norm:{cx:number;cy:number;sx:number;sy:number};
}

export async function fitMovieField(device:GPUDevice, file:File, fit:MovieFit, log:(s:string)=>void) {
  const {width:w,height:h,model:m,drift,settings}=fit;
  const n=Math.min(12,drift.length);
  const times=Array.from({length:n},(_,i)=>Math.round(i*(drift.length-1)/(n-1)));
  const focus=times.map(t=>drift[t][2]/1000);
  if(n<6 || Math.min(h,w)<64 || Math.max(...focus)-Math.min(...focus)<.04)
    return {selected:'global',reason:'Need six exposures, a 64 px field and 40 nm of focus diversity.'};
  const movie=await TiffMovie.open([file]);
  const frames=await Promise.all(times.map(t=>movie.read(t,t+1)));
  const side=Math.min(64,2**Math.floor(Math.log2(Math.min(h,w)/2)));
  const win=Array.from({length:side},(_,i)=>Math.sin(Math.PI*i/(side-1))**2);
  const weightSum=win.reduce((a,b)=>a+b,0)**2;
  const bins:number[]=[], fx:number[]=[], fy:number[]=[];
  for(let y=0;y<side/2;y++) for(let x=0;x<side;x++){
    const u=(x<side/2?x:x-side)/side,v=y/side;
    if(Math.hypot(u,v)>=.05 && Math.hypot(u,v)<=.25){bins.push(y*side+x);fx.push(u);fy.push(v);}
  }
  const B=bins.length, M=m.noll.length;
  const held=times.map((_,i)=>i%4===2);
  const tiles:{features:number[];data:Float64Array}[]=[];
  for(let gy=0;gy<3;gy++) for(let gx=0;gx<3;gx++){
    const x=Math.round(gx*(w-side)/2),y=Math.round(gy*(h-side)/2);
    const data=new Float64Array(2*n*B),power=new Float64Array(B);
    frames.forEach((frame,t)=>{
      const re=new Float64Array(side**2),im=re.slice();
      let mean=0;
      for(let j=0;j<side;j++) for(let i=0;i<side;i++) mean+=frame[(y+j)*w+x+i]*win[i]*win[j]/weightSum;
      for(let j=0;j<side;j++) for(let i=0;i<side;i++) re[j*side+i]=(frame[(y+j)*w+x+i]-mean)*settings.gain*win[i]*win[j];
      fft2d(re,im,side);
      bins.forEach((b,k)=>{
        const a=2*Math.PI*(drift[times[t]][0]*fx[k]+drift[times[t]][1]*fy[k]), c=Math.cos(a),s=Math.sin(a);
        const p=2*(t*B+k);
        data[p]=c*re[b]-s*im[b];data[p+1]=s*re[b]+c*im[b];
        if(!held[t])power[k]+=(re[b]**2+im[b]**2)/held.filter(v=>!v).length;
      });
    });
    const floor=.15*power.reduce((a,b)=>a+b,0)/B;
    for(let p=0;p<data.length;p++)data[p]/=Math.sqrt(Math.max(1,power[(p>>1)%B]+floor));
    tiles.push({features:[1,(2*y+side-1)/(h-1)-1,(2*x+side-1)/(w-1)-1],data});
  }
  const eng=new VecEngine(device,{...m.optics,roisize:19,npupil:m.npupil,oversample:m.oversample,zernike_noll:m.noll},['zern'],{maxChunk:32});
  const radius=settings.diameterNm/2000,a=radius*Math.sqrt(.6);
  const nodes=[[a,0,0],[-a,0,0],[0,a,0],[0,-a,0],[0,0,a],[0,0,-a]];
  async function transfer(c:number[]){
    eng.setZernike(c);
    const H=Array.from({length:M+1},()=>new Float64Array(2*n*B));
    for(let t=0;t<n;t++){
      const params=nodes.map(([dx,dy,dz])=>({x:dx/m.optics.pixel_um,y:dy/m.optics.pixel_um,ze:radius+dz,f:radius+focus[t],I:1,bg:0}));
      const stamps=await eng.psfAndJacobian(params);
      for(let d=0;d<=M;d++){
        const re=new Float64Array(side*side),im=re.slice();
        for(let node=0;node<6;node++) for(let j=0;j<19;j++) for(let i=0;i<19;i++)
          re[j*side+i]+=stamps[((d*6+node)*19+j)*19+i]/6;
        fft2d(re,im,side);
        bins.forEach((bin,k)=>{const p=2*(t*B+k);H[d][p]=re[bin];H[d][p+1]=im[bin];});
      }
    }
    return H;
  }
  function residual(H:Float64Array,Y:Float64Array,test:boolean){
    const r=new Float64Array(2*n*B);
    for(let k=0;k<B;k++){
      let ar=0,ai=0,den=0;
      for(let t=0;t<n;t++)if(!held[t]){
        const p=2*(t*B+k),x=H[p],y=H[p+1];
        ar+=x*Y[p]+y*Y[p+1];ai+=x*Y[p+1]-y*Y[p];den+=x*x+y*y;
      }
      ar/=Math.max(den,1e-10);ai/=Math.max(den,1e-10);
      for(let t=0;t<n;t++)if(held[t]===test){
        const p=2*(t*B+k),x=H[p],y=H[p+1];
        r[p]=x*ar-y*ai-Y[p];r[p+1]=y*ar+x*ai-Y[p+1];
      }
    }
    return r;
  }
  const dot=(a:Float64Array,b:Float64Array)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;};
  async function system(c:number[],jac=false,test=false){
    const size=c.length,F=size/M,G=new Float64Array(size*size),rhs=new Float64Array(size);
    let cost=0;
    const cache=new Map<string,Float64Array[]>();
    for(const {features,data} of tiles){
      const pupil=m.noll.map((_,j)=>features.slice(0,F).reduce((s,v,i)=>s+v*c[i*M+j],0));
      const key=pupil.join(',');
      let H=cache.get(key);
      if(!H){H=await transfer(pupil);cache.set(key,H);}
      const r=residual(H[0],data,test);cost+=dot(r,r);
      if(!jac)continue;
      const J=H.slice(1).map(d=>{
        const e=residual(d.map((x,p)=>H![0][p]+.001*x),data,false);
        return e.map((x,p)=>(x-r[p])/.001);
      });
      for(let j=0;j<M;j++){
        const g=dot(J[j],r);
        for(let i=0;i<F;i++)rhs[i*M+j]-=features[i]*g;
        for(let k=0;k<M;k++){
          const v=dot(J[j],J[k]);
          for(let i=0;i<F;i++)for(let l=0;l<F;l++)G[(i*M+j)*size+l*M+k]+=features[i]*features[l]*v;
        }
      }
    }
    return {cost,G,rhs};
  }
  async function optimize(c:number[]){
    for(let it=0;it<5;it++){
      log('Fitting '+(c.length===M?'shared':'field-dependent')+' pupil · iteration '+(it+1));
      const {cost,G,rhs}=await system(c,true),N=c.length;
      for(let i=0;i<N;i++)G[i*N+i]=G[i*N+i]*1.02+.0001;
      const step=solve(G,rhs,N),limit=Math.max(1,...step.map(x=>Math.abs(x)/.045));
      let accepted=false;
      for(const a of [1,.5,.25]){
        const trial=c.map((v,i)=>v+a*step[i]/limit);
        if((await system(trial)).cost<cost){c=trial;accepted=true;break;}
      }
      if(!accepted)break;
    }
    return c;
  }
  try{
    const nominal=Array.from(m.zcoef),global=await optimize(nominal),field=await optimize([...global,...Array(2*M).fill(0)]);
    const den=tiles.length*held.filter(Boolean).length*B;
    const global_score=Math.min((await system(nominal,false,true)).cost,(await system(global,false,true)).cost)/den;
    const field_score=(await system(field,false,true)).cost/den;
    const selected=field_score<.99*global_score?'field':'global';
    if(selected==='field'){
      const cx=(w-1)/2,cy=(h-1)/2;
      fit.field={type:'zernike_poly',order:1,zernike_noll:[...m.noll],coef:m.noll.map((_,j)=>[field[j],field[M+j],field[2*M+j]]),feats_norm:{cx,cy,sx:cx,sy:cy}};
      m.zcoef=field.slice(0,M);
    }
    return {selected,global_score,field_score,held_frames:times.filter((_,t)=>held[t])};
  } finally {eng.destroy();}
}
