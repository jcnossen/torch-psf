// Each camera pixel sees every bead whose footprint touches its tile.
struct Params { w:u32,h:u32,nb:u32,side:u32, tileCols:u32,nz:u32, sampling:f32,dz:f32,
  dx:f32,dy:f32,zi:f32,bg:f32, bgSlope:vec2f }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> lut:array<f32>;
@group(0) @binding(2) var<storage,read> tiles:array<u32>;
@group(0) @binding(3) var<storage,read> beads:array<vec4f>;
@group(0) @binding(4) var<storage,read> obs:array<f32>;
@group(0) @binding(5) var<storage,read_write> model:array<f32>;
@group(0) @binding(6) var<storage,read_write> sums:array<f32>;
fn plane(q:vec2f,z:u32)->f32 {
 let ij=vec2u(q); let f=fract(q); let i=(z*p.side+ij.y)*p.side+ij.x;
 return mix(mix(lut[i],lut[i+1],f.x),mix(lut[i+p.side],lut[i+p.side+1],f.x),f.y);
}
fn kernel(d:vec2f,z:f32)->f32 {
 let q=d*p.sampling+0.5*f32(p.side-1);
 if(any(q<vec2f(0))||any(q>=vec2f(f32(p.side-1)))){return 0;}
 let u=clamp(z,0.,f32(p.nz-1)); let i=min(u32(u),p.nz-2);
 return mix(plane(q,i),plane(q,i+1),u-f32(i));
}
@compute @workgroup_size(64) fn render(@builtin(global_invocation_id) id:vec3u){
 let i=id.x; if(i>=p.w*p.h){return;}
 let xy=vec2u(i%p.w,i/p.w); let t=xy.y/8*p.tileCols+xy.x/8;
 var mu=p.bg+dot(p.bgSlope,2.*vec2f(xy)/vec2f(vec2u(p.w-1,p.h-1))-1.);
 for(var j=tiles[t];j<tiles[t+1];j++){let b=beads[tiles[j]];
 mu+=b.z*kernel(vec2f(xy)-b.xy-vec2f(p.dx,p.dy),p.zi);}
 model[i]=max(mu,0.001);
}
@compute @workgroup_size(64) fn fit(@builtin(global_invocation_id) id:vec3u){
 let n=id.x;if(n>=p.nb){return;}
 let b=beads[n];let center=b.xy+vec2f(p.dx,p.dy);
 let radius=i32(ceil(f32(p.side)/p.sampling/2.));let mid=vec2i(round(center));
 var g=vec3f(0);var h=vec3f(0);var num=0.;var den=0.;var cost=0.;
 for(var y=max(0,mid.y-radius);y<=min(i32(p.h)-1,mid.y+radius);y++){
 for(var x=max(0,mid.x-radius);x<=min(i32(p.w)-1,mid.x+radius);x++){
 let d=vec2f(f32(x),f32(y))-center;let k=kernel(d,p.zi);if(k<=0.){continue;}
 let i=u32(y)*p.w+u32(x);let mu=model[i];let v=obs[i];let r=v/mu;
 let deriv=b.z*vec3f((kernel(d-vec2f(0.1,0),p.zi)-kernel(d+vec2f(0.1,0),p.zi))*5.,
 (kernel(d-vec2f(0,0.1),p.zi)-kernel(d+vec2f(0,0.1),p.zi))*5.,
 (kernel(d,p.zi+0.25)-kernel(d,p.zi-0.25))*2.);
 g+=deriv*(r-1.);h+=deriv*deriv/mu;num+=k*r;den+=k;
 cost+=b.z*k/mu*(mu-v)*(mu-v)/mu;
 }}
 let o=n*9;sums[o]=num/max(den,1e-9);sums[o+1]=g.x;sums[o+2]=g.y;sums[o+3]=g.z;
 sums[o+4]=h.x;sums[o+5]=h.y;sums[o+6]=h.z;sums[o+7]=cost;sums[o+8]=den;
}

// Six global derivatives: camera translation XYZ and the three background terms.
// Sum emitter derivatives at a pixel before forming the Poisson normal matrix.
var<workgroup> partial:array<array<f32,27>,64>;
@compute @workgroup_size(8,8) fn exposure(@builtin(global_invocation_id) id:vec3u,
 @builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) block:vec3u){
 if(id.x<p.w && id.y<p.h){
  let pixel=id.y*p.w+id.x; let tile=block.y*p.tileCols+block.x;
  var grad=vec3f(0.);
  for(var j=tiles[tile];j<tiles[tile+1];j++){
   let b=beads[tiles[j]];let d=vec2f(id.xy)-b.xy-vec2f(p.dx,p.dy);
   grad+=b.z*vec3f((kernel(d-vec2f(.008,0),p.zi)-kernel(d+vec2f(.008,0),p.zi))/.016,
                    (kernel(d-vec2f(0,.008),p.zi)-kernel(d+vec2f(0,.008),p.zi))/.016,
                    (kernel(d,p.zi+.1)-kernel(d,p.zi-.1))/.2);
  }
  let uv=2.*vec2f(id.xy)/vec2f(vec2u(p.w-1,p.h-1))-1.;
  var g=array<f32,6>(grad.x,grad.y,grad.z,1.,uv.x,uv.y);
  let pred=max(model[pixel],2.);let resid=(obs[pixel]-pred)/pred;
  var k=6u;
  for(var i=0u;i<6;i++){
   partial[lane][i]=g[i]*resid;
   for(var j=0u;j<=i;j++){partial[lane][k]=g[i]*g[j]/pred;k++;}
  }
 }
 workgroupBarrier();
 for(var stride=32u;stride>0;stride/=2u){
  if(lane<stride){for(var k=0u;k<27;k++){ partial[lane][k]+=partial[lane+stride][k]; }}
  workgroupBarrier();
 }
 if(lane==0u){let off=(block.y*p.tileCols+block.x)*27u;
  for(var k=0u;k<27;k++){sums[off+k]=partial[0][k];}
 }
}
