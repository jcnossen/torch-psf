type Peak = number[];
export function centralPatch(image:Float32Array,w:number,h:number){
  const side=Math.min(160,w,h), out=new Float32Array(side*side);
  const left=Math.floor((w-side)/2), top=Math.floor((h-side)/2);
  for(let y=0;y<side;y++)out.set(image.subarray((top+y)*w+left,(top+y)*w+left+side),y*side);
  return out;
}
function sample(image:Float32Array,w:number,h:number,x:number,y:number){
  x=Math.min(w-1,Math.max(0,x)); y=Math.min(h-1,Math.max(0,y));
  const i=Math.floor(x),j=Math.floor(y),a=x-i,b=y-j,k=j*w+i;
  const right=Math.min(w-1,i+1)-i,down=j+1<h?w:0;
  return (1-b)*((1-a)*image[k]+a*image[k+right])+b*((1-a)*image[k+down]+a*image[k+down+right]);
}
export function lateralRegistration(patches:Float32Array[]){
  const side=Math.sqrt(patches[0].length), drift=patches.map(()=>[0,0]);
  if(side<24 || patches.length<2)return drift;
  const n=Math.ceil((side-8)/2)**2;
  for(let round=0;round<10;round++){
    const target=new Float64Array(n);
    const rows=patches.map((im,t)=>{
      const v=new Float64Array(n),gx=v.slice(),gy=v.slice();
      const [dx,dy]=drift[t];let sum=0,squares=0,i=0;
      for(let y=4;y<side-4;y+=2)for(let x=4;x<side-4;x+=2,i++){
        const at=(a:number,b:number)=>sample(im,side,side,x+dx+a,y+dy+b);
        v[i]=at(0,0);gx[i]=(at(1,0)-at(-1,0))/2;gy[i]=(at(0,1)-at(0,-1))/2;
        sum+=v[i];squares+=v[i]**2;
      }
      const mean=sum/n,scale=Math.sqrt(Math.max(1e-12,squares/n-mean**2));
      for(let k=0;k<n;k++){v[k]=(v[k]-mean)/scale;gx[k]/=scale;gy[k]/=scale;target[k]+=v[k]/patches.length;}
      return {v,gx,gy};
    });
    let mx=0,my=0;
    rows.forEach(({v,gx,gy},t)=>{
      let xx=1,xy=0,yy=1,rx=0,ry=0;
      for(let k=0;k<n;k++){xx+=gx[k]**2;xy+=gx[k]*gy[k];yy+=gy[k]**2;rx+=gx[k]*(target[k]-v[k]);ry+=gy[k]*(target[k]-v[k]);}
      const det=xx*yy-xy*xy;
      drift[t][0]+=Math.max(-.5,Math.min(.5,(yy*rx-xy*ry)/det));
      drift[t][1]+=Math.max(-.5,Math.min(.5,(xx*ry-xy*rx)/det));
      mx+=drift[t][0]/rows.length;my+=drift[t][1]/rows.length;
    });
    for(const d of drift){d[0]-=mx;d[1]-=my;}
  }
  return drift;
}
export function addRegistered(group:Float32Array,im:Float32Array,w:number,h:number,d:number[],count:number){
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)group[y*w+x]+=sample(im,w,h,x+d[0],y+d[1])/count;
}
function peaks(im:Float32Array,w:number,h:number){
  const score=new Float32Array(im.length), threshold=new Float32Array(im.length);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){
    const i=y*w+x;
    const smooth=(4*im[i]+2*(im[i-1]+im[i+1]+im[i-w]+im[i+w])+im[i-w-1]+im[i-w+1]+im[i+w-1]+im[i+w+1])/16;
    let ring=0;
    for(let dy=-3;dy<=3;dy+=2)for(let dx=-3;dx<=3;dx+=2)if(Math.abs(dy)===3 || Math.abs(dx)===3)
      ring+=im[Math.max(0,Math.min(h-1,y+dy))*w+Math.max(0,Math.min(w-1,x+dx))];
    ring/=12;score[i]=smooth-ring;threshold[i]=Math.max(6*Math.sqrt(Math.max(1,ring)),.12*smooth);
  }
  const out:Peak[]=[];
  for(let y=2;y<h-2;y++)for(let x=2;x<w-2;x++){
    const i=y*w+x,s=score[i];if(s<=threshold[i])continue;
    if([-w-1,-w,-w+1,-1,1,w-1,w,w+1].some(o=>score[i+o]>s))continue;
    const off=(k:number)=>Math.max(-.5,Math.min(.5,.5*(score[i-k]-score[i+k])/Math.min(-1e-6,score[i-k]+score[i+k]-2*s)));
    out.push([x+off(1),y+off(w),s]);
  }
  return out;
}
function residualPeaks(image:Float32Array,w:number,h:number){
  let candidates=peaks(image,w,h);
  const all=candidates.slice();
  const bright=candidates.filter(([x,y])=>x>=6 && y>=6 && x<w-6 && y<h-6).sort((a,b)=>b[2]-a[2]).slice(0,96);
  if(bright.length<4)return all;
  const anchors=bright.map(([x,y,height])=>{
    const i=Math.round(y)*w+Math.round(x);let top=0,bottom=0;
    for(let dx=-5;dx<=5;dx++){top+=image[i-5*w+dx];bottom+=image[i+5*w+dx];}
    return {i,height,edge:Math.min(top,bottom)/11};
  });
  const stamp=new Float32Array(121);
  for(let dy=-5;dy<=5;dy++)for(let dx=-5;dx<=5;dx++){
    const v=anchors.map(a=>(image[a.i+dy*w+dx]-a.edge)/Math.max(1,a.height)).sort((a,b)=>a-b);
    stamp[(dy+5)*11+dx+5]=Math.max(0,(v[(v.length-1)>>1]+v[v.length>>1])/2);
  }
  const residual=image.slice();
  for(let round=0;round<2;round++){
    for(const [x,y,height] of candidates){
      const ix=Math.round(x),iy=Math.round(y);
      for(let dy=Math.max(-5,-iy);dy<=Math.min(5,h-1-iy);dy++)
        for(let dx=Math.max(-5,-ix);dx<=Math.min(5,w-1-ix);dx++)
          residual[(iy+dy)*w+ix+dx]-=.75*height*stamp[(dy+5)*11+dx+5];
    }
    candidates=peaks(residual,w,h).filter(([x,y,v])=>v>4*Math.sqrt(Math.max(1,image[Math.round(y)*w+Math.round(x)])));
    for(const p of candidates)all.push(p);
  }
  return all;
}
export class MovieCandidates{
  private votes:Float32Array;
  private amplitude:Float32Array;
  constructor(private w:number, private h:number){
    this.votes=new Float32Array(4*w*h);this.amplitude=new Float32Array(4*w*h);
  }
  add(image:Float32Array,drift:number[]){
    const W=this.w*2;
    for(const [x,y,a] of residualPeaks(image,this.w,this.h)){
      const u=2*(x-drift[0]),v=2*(y-drift[1]),i=Math.floor(u),j=Math.floor(v);
      for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){
        if(i+dx<0||i+dx>=W||j+dy<0||j+dy>=this.h*2)continue;
        const weight=(1-Math.abs(u-i-dx))*(1-Math.abs(v-j-dy)),k=(j+dy)*W+i+dx;
        this.votes[k]+=weight;this.amplitude[k]+=weight*a;
      }
    }
  }
  finish(frames:number){
    const W=this.w*2,H=this.h*2;
    for(const a of [this.votes,this.amplitude]){
      for(const stride of [W,1]){
        const b=a.slice();
        for(let y=0;y<H;y++)for(let x=0;x<W;x++){
          const k=y*W+x;
          a[k]=.5*b[k]+.25*((stride===W ? y>0 : x>0)?b[k-stride]:0)+.25*((stride===W ? y+1<H : x+1<W)?b[k+stride]:0);
        }
      }
    }
    const out:Peak[]=[];
    for(let y=0;y<H;y++)for(let x=0;x<W;x++){
      const k=y*W+x,v=this.votes[k];if(v<=Math.max(.5,.014*frames))continue;
      let peak=true;
      for(let dy=Math.max(-1,-y);dy<=Math.min(1,H-1-y);dy++)for(let dx=Math.max(-1,-x);dx<=Math.min(1,W-1-x);dx++)if(this.votes[k+dy*W+dx]>v)peak=false;
      if(peak)out.push([x/2,y/2,this.amplitude[k]/v]);
    }
    return out;
  }
}
