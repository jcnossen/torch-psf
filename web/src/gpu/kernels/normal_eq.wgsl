// Poisson normal equations per image over the atomic columns a (nA = D + 1):
//   a = 0: d mu / d I = psf;  a = 1: d mu / d bg = 1;  a >= 2: d mu / d theta_(a-1) = I d psf / d theta_(a-1)
// slots k: pairs (H = J^T diag(1/mu) J, upper triangle), then g = J^T (d / mu - 1), then sum(mu - d log mu).
@group(0) @binding(1) var<storage, read> psfJ: array<f32>;          // [D, B, RR]
@group(0) @binding(2) var<storage, read> data: array<f32>;          // [Btot, RR] photons
@group(0) @binding(3) var<storage, read> imgp: array<vec4f>;        // [2 Btot]
@group(0) @binding(4) var<storage, read> slots: array<vec2u>;       // [nslot]: (a1, a2); grad (a, 0xffff); nll (0xffff, 0xffff)
@group(0) @binding(5) var<storage, read_write> NE: array<f32>;      // [Btot, nslot]

fn col(a: u32, b: u32, p: u32, RR: u32, I0: f32) -> f32 {
  if (a == 0u) { return psfJ[b * RR + p]; }
  if (a == 1u) { return 1.0; }
  return I0 * psfJ[((a - 1u) * u.B + b) * RR + p];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let b = g.y;
  if (k >= u.nslot || b >= u.B) { return; }
  let RR = u.R * u.R;
  let gb = u.b0 + b;
  let q = imgp[2u * gb + 1u];
  let s = slots[k];
  var acc = 0.0;
  for (var p = 0u; p < RR; p++) {
    let mu = max(q.x * psfJ[b * RR + p] + q.y, 1e-4);
    let d = data[gb * RR + p];
    if (s.x == 0xffffu) {
      acc += mu - d * log(mu);
    } else if (s.y == 0xffffu) {
      acc += col(s.x, b, p, RR, q.x) * (d / mu - 1.0);
    } else {
      acc += col(s.x, b, p, RR, q.x) * col(s.y, b, p, RR, q.x) / mu;
    }
  }
  NE[gb * u.nslot + k] = acc;
}
