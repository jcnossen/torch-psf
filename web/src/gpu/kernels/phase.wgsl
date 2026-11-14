// PB[b, c, k] = P[c, k] exp(i (W - kx x - ky y - kz_imm f)) exp(i kz_med z_e): the per-image pupil fields.
@group(0) @binding(1) var<storage, read> P: array<vec2f>;        // [6, NN]
@group(0) @binding(2) var<storage, read> kzmed: array<vec2f>;    // [NN]
@group(0) @binding(3) var<storage, read> pk: array<vec4f>;       // [NN]: kx, ky, kz_imm, W (rad)
@group(0) @binding(4) var<storage, read> imgp: array<vec4f>;     // [2 Btot]: (x px, y px, z_e um, f um), (I, bg, 0, 0)
@group(0) @binding(5) var<storage, read_write> PB: array<vec2f>; // [B, 6, NN]

const TWO_PI: f32 = 6.283185307179586;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let NN = u.N * u.N;
  let k = g.x;
  let b = g.y;
  if (k >= NN || b >= u.B) { return; }
  let q = imgp[2u * (u.b0 + b)];
  let t = pk[k];
  let kz = kzmed[k];
  var ph = t.w - t.x * q.x * u.px - t.y * q.y * u.px - t.z * q.w + kz.x * q.z;
  ph = ph - TWO_PI * round(ph / TWO_PI);     // range reduction: GPU sin / cos are accurate on [-pi, pi]
  let e = vec2f(cos(ph), sin(ph)) * exp(-kz.y * q.z);
  for (var c = 0u; c < 6u; c++) {
    PB[(b * 6u + c) * NN + k] = cmul(P[c * NN + k], e);
  }
}
