struct Grid { roi:u32, side:u32, nz:u32, nx:u32, ny:u32, sampling:u32, focus:f32, pad:u32 }
@group(0) @binding(0) var<uniform> g: Grid;
@group(0) @binding(1) var<storage,read> bank: array<f32>;
@group(0) @binding(2) var<storage,read> spots: array<vec4f>;
@group(0) @binding(3) var<storage,read_write> output: array<f32>;
fn cubic(t:f32) -> vec4f {
  let s=1.-t;
  return vec4f(s*s*s, 4.-6.*t*t+3.*t*t*t, 1.+3.*t+3.*t*t-3.*t*t*t,t*t*t)/6.;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3u) {
  let k=id.x;
  if(k>=g.roi*g.roi){ return; }
  let p=spots[id.y];
  let tile=clamp(p.zw,vec2f(0),vec2f(f32(g.nx-1u),f32(g.ny-1u)));
  let center=f32(g.side-1u)*.5;
  let pix=vec2f(f32(k%g.roi),f32(k/g.roi));
  let uv=center+(pix-f32(g.roi-1u)*.5-p.xy)*f32(g.sampling);
  let wx=cubic(fract(uv.x)); let wy=cubic(fract(uv.y)); let wz=cubic(fract(g.focus));
  let base=(u32(g.focus)-1u)*g.side*g.side+(u32(uv.y)-1u)*g.side+u32(uv.x)-1u;
  let t=vec2u(tile); let f=fract(tile);
  var val=0.;
  for(var j=0u;j<2u;j++){ for(var i=0u;i<2u;i++){
    let weight=select(1.-f.x,f.x,i==1u)*select(1.-f.y,f.y,j==1u);
    if(weight==0.){continue;}
    let node=(t.y+j)*g.nx+t.x+i;
    let start=node*g.nz*g.side*g.side+base;
    var v=0.;
    for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){
      let o=start+(z*g.side+y)*g.side;
      v+=wz[z]*wy[y]*dot(wx,vec4f(bank[o],bank[o+1u],bank[o+2u],bank[o+3u]));
    }}
    val+=weight*v;
  }}
  output[id.y*g.roi*g.roi+k]=max(0.,val);
}
