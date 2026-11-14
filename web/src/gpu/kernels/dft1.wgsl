// T[b, t, c, v, kx] = sum_ky Fy[v, ky] (G[t] PB[b, c])[ky, kx]   (first matrix DFT, all directions t < D)
@group(0) @binding(1) var<storage, read> F: array<vec2f>;        // [ns, N] exp(i u k)
@group(0) @binding(2) var<storage, read> PB: array<vec2f>;       // [B, 6, NN]
@group(0) @binding(3) var<storage, read> G: array<vec2f>;        // [D, NN]
@group(0) @binding(4) var<storage, read_write> T: array<vec2f>;  // [B, D, 6, ns, N]

var<workgroup> sF: array<vec2f, 256>;     // [16 v][16 ky]
var<workgroup> sB: array<vec2f, 1536>;    // [6][16 ky][16 kx]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let N = u.N; let NN = N * N; let ns = u.ns;
  let b = wg.z / u.D;
  let t = wg.z % u.D;
  let kx = wg.x * 16u + l.x;
  let v = wg.y * 16u + l.y;
  var acc: array<vec2f, 6>;
  for (var c = 0u; c < 6u; c++) { acc[c] = vec2f(0.0); }
  for (var t0 = 0u; t0 < N; t0 += 16u) {
    let vv = wg.y * 16u + l.y;
    let ky = t0 + l.x;
    sF[l.y * 16u + l.x] = select(vec2f(0.0), F[min(vv, ns - 1u) * N + min(ky, N - 1u)], vv < ns && ky < N);
    let kyb = t0 + l.y;
    let ok = kyb < N && kx < N;
    let idx = min(kyb, N - 1u) * N + min(kx, N - 1u);
    let gm = G[t * NN + idx];
    for (var c = 0u; c < 6u; c++) {
      sB[c * 256u + l.y * 16u + l.x] = select(vec2f(0.0), cmul(PB[(b * 6u + c) * NN + idx], gm), ok);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let f = sF[l.y * 16u + kk];
      for (var c = 0u; c < 6u; c++) { acc[c] += cmul(f, sB[c * 256u + kk * 16u + l.x]); }
    }
    workgroupBarrier();
  }
  if (v < ns && kx < N) {
    for (var c = 0u; c < 6u; c++) { T[(((b * u.D + t) * 6u + c) * ns + v) * N + kx] = acc[c]; }
  }
}
