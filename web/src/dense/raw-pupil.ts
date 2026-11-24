import {type MovieFit,predictMovieFrame} from './movie';
import {TiffMovie} from '../io/tiff';
import {solve} from '../fit/linalg';

// Diagnostic refinement on training pixels only. Profile a local intensity scale,
// background and common XYZ displacement in each patch alongside the shared pupil.
export async function refineRawPupil(device:GPUDevice, source:File, fit:MovieFit){
  const movie=await TiffMovie.open([source]), side=Math.min(64,movie.W,movie.H);
  const nModes=fit.model.noll.length, patches=[];
  const count=Math.min(8,movie.n);
  for(let i=0;i<count;i++){
    const frame=Math.round(i*(movie.n-1)/(count-1));
    const x0=Math.round((.25+.5*(i%2))*(movie.W-side)), y0=Math.round((.25+.5*(Math.floor(i/2)%2))*(movie.H-side));
    const d=fit.drift[frame];
    const ids=fit.xy.map(([x,y],j)=>x+d[0]>x0-10 && y+d[1]>y0-10 && x+d[0]<x0+side+10 && y+d[1]<y0+side+10 ? j : -1).filter(j=>j>=0);
    if(!ids.length)continue;
    const raw=await movie.read(frame,frame+1),observed=new Float32Array(side*side);
    for(let y=0;y<side;y++)for(let x=0;x<side;x++)observed[y*side+x]=Math.max(0,(raw[(y0+y)*movie.W+x0+x]-fit.settings.offset)*fit.settings.gain);
    patches.push({fit:{...fit,width:side,height:side,xy:ids.map(j=>[fit.xy[j][0]-x0,fit.xy[j][1]-y0])},
      d,flux:ids.map(j=>fit.intensity[frame][j]),bg:fit.background[frame],observed});
  }
  let coefficients=Array.from(fit.model.zcoef);
  if(!patches.length)return coefficients;
  const nuisance=patches.map(()=>[0,0,0,0,0]); // px, px, nm, fractional flux, background photons
  type Patch=typeof patches[number];
  async function predict(p:Patch,c:number[],u:number[]){
    const region={...p.fit,model:{...fit.model,zcoef:c}};
    const drift=p.d.map((v,j)=>v+u[j]);
    return (await predictMovieFrame(device,region,drift,[p.flux.map(v=>v*(1+u[3]))],Math.max(0,p.bg+u[4])))[0];
  }
  function loss(y:Float32Array,mu:Float32Array){
    let v=0;
    for(let i=0;i<mu.length;i++){const m=Math.max(.01,mu[i]);v+=m-y[i]*Math.log(m);}
    return v;
  }
  const n=nModes+5*patches.length;
  for(let round=0;round<2;round++){
    let cost=0;
    const gram=new Float64Array(n*n),rhs=new Float64Array(n);
    for(let j=0;j<patches.length;j++){
      const p=patches[j],u=nuisance[j],base=await predict(p,coefficients,u);
      cost+=loss(p.observed,base);
      const cols:Float32Array[]=[], keys:number[]=[];
      for(let k=0;k<nModes+5;k++){
        const c=coefficients.slice(),v=u.slice();
        const eps=k<nModes?.001:[.04,.04,3,.01,.5][k-nModes];
        if(k<nModes)c[k]+=eps;else v[k-nModes]+=eps;
        const shifted=await predict(p,c,v);
        cols.push(shifted.map((m,i)=>(m-base[i])/eps));keys.push(k<nModes?k:nModes+5*j+k-nModes);
      }
      for(let i=0;i<base.length;i++){
        const w=1/Math.max(1,base[i]),r=(p.observed[i]-base[i])*w;
        for(let k=0;k<cols.length;k++){
          const a=cols[k][i],idx=keys[k];rhs[idx]+=a*r;
          for(let l=0;l<=k;l++)gram[idx*n+keys[l]]+=w*a*cols[l][i];
        }
      }
    }
    for(let i=0;i<n;i++){
      for(let j=0;j<i;j++)gram[j*n+i]=gram[i*n+j];
      gram[i*n+i]=1.05*gram[i*n+i]+1e-6;
    }
    const step=solve(gram,rhs,n);
    const scale=Math.max(1,...step.map((v,i)=>Math.abs(v)/(i<nModes?.02:[.25,.25,40,.2,10][(i-nModes)%5])));
    let accepted=false;
    for(const fraction of [1,.5,.25]){
      const c=coefficients.map((v,k)=>v+fraction*step[k]/scale);
      const u=nuisance.map((row,j)=>row.map((v,k)=>v+fraction*step[nModes+5*j+k]/scale));
      let next=0;
      for(let j=0;j<patches.length;j++)next+=loss(patches[j].observed,await predict(patches[j],c,u[j]));
      if(next<cost){coefficients=c;u.forEach((row,j)=>nuisance[j]=row);accepted=true;break;}
    }
    if(!accepted)break;
  }
  return coefficients;
}
