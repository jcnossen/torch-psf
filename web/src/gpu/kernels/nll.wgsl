// Poisson NLL per image (without the lgamma(d + 1) constant): sum_p mu - d log mu, mu = max(I psf + bg, 1e-4).
@group(0) @binding(1) var<storage, read> psfJ: array<f32>;          // [D, B, RR] (only t = 0 used)
@group(0) @binding(2) var<storage, read> data: array<f32>;          // [Btot, RR]
@group(0) @binding(3) var<storage, read> imgp: array<vec4f>;        // [2 Btot]
@group(0) @binding(4) var<storage, read_write> out: array<f32>;     // [Btot]

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let b = wg.x;
  let RR = u.R * u.R;
  let gb = u.b0 + b;
  let q = imgp[2u * gb + 1u];
  var acc = 0.0;
  for (var p = l.x; p < RR; p += 64u) {
    let mu = max(q.x * psfJ[b * RR + p] + q.y, 1e-4);
    acc += mu - data[gb * RR + p] * log(mu);
  }
  red[l.x] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (l.x < s) { red[l.x] += red[l.x + s]; }
    workgroupBarrier();
  }
  if (l.x == 0u) { out[gb] = red[0]; }
}
