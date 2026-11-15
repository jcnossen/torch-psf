// Knot VJP, step 5: the Zernike gradient out[j] = 2 pi sum_k Z[j, k] dL/dW_k (W = 2 pi sum_j c_j Z_j); one workgroup
// per coefficient j.
@group(0) @binding(1) var<storage, read> Z: array<f32>;            // [nZ, NN] (masked basis)
@group(0) @binding(2) var<storage, read> gW: array<f32>;           // [NN]
@group(0) @binding(3) var<storage, read_write> zg: array<f32>;     // [nZ]

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let NN = u.N * u.N;
  let j = wg.x;
  var s = 0.0;
  for (var k = l.x; k < NN; k += 256u) { s += Z[j * NN + k] * gW[k]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (l.x < h) { part[l.x] += part[l.x + h]; }
    workgroupBarrier();
  }
  if (l.x == 0u) { zg[j] = part[0] * 6.283185307179586; }
}
