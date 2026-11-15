// Knot VJP, step 1 (adjoint of knot_put, bin.wgsl and |E|^2 / 3): the knot gradient gK at the pixel (i, j) of image
// n = b0 + b is the gradient of psf[n, i, j]; psf = C mean_os(I), I = sum_c |E_c|^2 / 3, so with gI = gK C / os^2 at
// every subpixel (v, w) of the pixel, dL = Re sum conj(gE) dE with gE_c = 2/3 gI E_c. Written in place over E0 (the
// chunk's fields, kept from the knot build or recomputed).
@group(0) @binding(1) var<storage, read> gK: array<f32>;              // [nz, R S, R S]
@group(0) @binding(2) var<storage, read_write> E: array<vec2f>;      // [B, 6, ns, ns]: E0 in, gE out

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let ns = u.ns; let nss = ns * ns; let R = u.R; let S = u.S; let os = u.os;
  let p = g.x;
  let b = g.y;
  if (p >= nss || b >= u.B) { return; }
  let v = p / ns; let w = p % ns;
  let i = v / os; let j = w / os;
  let n = u.b0 + b;
  let k = n / (S * S); let bs = (n / S) % S; let a = n % S;
  let nv = R * S;
  let gI = gK[(k * nv + i * S + (S - 1u - bs)) * nv + j * S + (S - 1u - a)] * (u.C / f32(os * os));
  let f = gI * (2.0 / 3.0);
  let e = b * 6u;
  for (var c = 0u; c < 6u; c++) {
    E[(e + c) * nss + p] = f * E[(e + c) * nss + p];
  }
}
