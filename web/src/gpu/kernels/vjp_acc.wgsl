// Knot VJP, step 4: dL / dW summed over the images of a chunk, added to gW (cleared before the first chunk).
@group(0) @binding(1) var<storage, read> gWb: array<f32>;          // [B, NN]
@group(0) @binding(2) var<storage, read_write> gW: array<f32>;     // [NN]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let NN = u.N * u.N;
  let k = g.x;
  if (k >= NN) { return; }
  var s = 0.0;
  for (var b = 0u; b < u.B; b++) { s += gWb[b * NN + k]; }
  gW[k] += s;
}
