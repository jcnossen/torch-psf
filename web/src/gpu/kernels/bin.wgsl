// Pixel integration: psfJ[t, b, i, j] = C * mean over the os x os subpixels of I[b, t].
@group(0) @binding(1) var<storage, read> I: array<f32>;            // [B, D, ns, ns]
@group(0) @binding(2) var<storage, read_write> psfJ: array<f32>;   // [D, B, R, R]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let R = u.R; let os = u.os; let ns = u.ns;
  let p = g.x;
  if (p >= R * R) { return; }
  let bt = g.y;                 // b * D + t
  let b = bt / u.D;
  let t = bt % u.D;
  let i = p / R; let j = p % R;
  var s = 0.0;
  let base = bt * ns * ns;
  for (var a = 0u; a < os; a++) {
    for (var c = 0u; c < os; c++) { s += I[base + (i * os + a) * ns + j * os + c]; }
  }
  psfJ[(t * u.B + b) * R * R + p] = s * u.C / f32(os * os);
}
