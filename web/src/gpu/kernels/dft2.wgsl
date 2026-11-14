// E[b, t, c, v, w] = sum_kx T[b, t, c, v, kx] Fx[w, kx]   (second matrix DFT), then the intensity:
//   t == 0: E0 is stored and I[b, 0] = sum_c |E0_c|^2 / 3
//   t > 0:  I[b, t] = 2/3 sum_c Re(conj(E0_c) E_c)   (the derivative of the intensity along direction t)
// Directions t = theta0 + wg.z % nt, with nt = D (t > 0 pass: D - 1).
@group(0) @binding(1) var<storage, read> F: array<vec2f>;          // [ns, N]
@group(0) @binding(2) var<storage, read> T: array<vec2f>;          // [B, D, 6, ns, N]
@group(0) @binding(3) var<storage, read_write> E0: array<vec2f>;   // [B, 6, ns, ns]
@group(0) @binding(4) var<storage, read_write> I: array<f32>;      // [B, D, ns, ns]

var<workgroup> sT: array<vec2f, 1536>;    // [6][16 v][16 kx]
var<workgroup> sF: array<vec2f, 256>;     // [16 w][16 kx]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let N = u.N; let ns = u.ns;
  let nt = select(u.D - 1u, 1u, u.theta0 == 0u);
  let b = wg.z / nt;
  let t = u.theta0 + wg.z % nt;
  let w = wg.x * 16u + l.x;
  let v = wg.y * 16u + l.y;
  var acc: array<vec2f, 6>;
  for (var c = 0u; c < 6u; c++) { acc[c] = vec2f(0.0); }
  let tb = (b * u.D + t) * 6u;
  for (var t0 = 0u; t0 < N; t0 += 16u) {
    let kx = t0 + l.x;
    let vv = wg.y * 16u + l.y;
    let ok = vv < ns && kx < N;
    let off = min(vv, ns - 1u) * N + min(kx, N - 1u);
    for (var c = 0u; c < 6u; c++) {
      sT[c * 256u + l.y * 16u + l.x] = select(vec2f(0.0), T[(tb + c) * ns * N + off], ok);
    }
    let ww = wg.x * 16u + l.y;
    sF[l.y * 16u + l.x] = select(vec2f(0.0), F[min(ww, ns - 1u) * N + min(kx, N - 1u)], ww < ns && kx < N);
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let f = sF[l.x * 16u + kk];
      for (var c = 0u; c < 6u; c++) { acc[c] += cmul(sT[c * 256u + l.y * 16u + kk], f); }
    }
    workgroupBarrier();
  }
  if (v >= ns || w >= ns) { return; }
  let p = v * ns + w;
  let nss = ns * ns;
  if (t == 0u) {
    var s = 0.0;
    for (var c = 0u; c < 6u; c++) {
      E0[(b * 6u + c) * nss + p] = acc[c];
      s += dot(acc[c], acc[c]);
    }
    I[(b * u.D) * nss + p] = s / 3.0;
  } else {
    var s = 0.0;
    for (var c = 0u; c < 6u; c++) { s += dot(E0[(b * 6u + c) * nss + p], acc[c]); }
    I[(b * u.D + t) * nss + p] = s * (2.0 / 3.0);
  }
}
