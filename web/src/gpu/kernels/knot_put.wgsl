// Knot build (torch_psf CubicSplinePSF.build_planes): the value image psfJ[b] of image n = b0 + b, n = (k S + bs) S + a
// (emitter at x = a / S, y = bs / S px, knot plane k), goes to the knots [nz, R S, R S] at
//   knots[k, i S + (S - 1 - bs), j S + (S - 1 - a)] = psfJ[b, i, j].
@group(0) @binding(1) var<storage, read> psfJ: array<f32>;            // [B, R, R] (value direction only)
@group(0) @binding(2) var<storage, read_write> knots: array<f32>;     // [nz, R S, R S]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let R = u.R; let S = u.S;
  let p = g.x;
  let b = g.y;
  if (p >= R * R || b >= u.B) { return; }
  let n = u.b0 + b;
  let k = n / (S * S); let bs = (n / S) % S; let a = n % S;
  let i = p / R; let j = p % R;
  let nv = R * S;
  knots[(k * nv + i * S + (S - 1u - bs)) * nv + j * S + (S - 1u - a)] = psfJ[b * R * R + p];
}
