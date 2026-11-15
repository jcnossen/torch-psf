// Knot VJP, step 2 (adjoint of dft2.wgsl, E = T Fx^T): A1[b, c, v, kx] = sum_w gE[b, c, v, w] conj(F[w, kx]).
@group(0) @binding(1) var<storage, read> F: array<vec2f>;          // [ns, N]
@group(0) @binding(2) var<storage, read> E: array<vec2f>;          // [B, 6, ns, ns] gE
@group(0) @binding(3) var<storage, read_write> T: array<vec2f>;    // [B, 6, ns, N]

var<workgroup> sE: array<vec2f, 1536>;    // [6][16 v][16 w]
var<workgroup> sF: array<vec2f, 256>;     // [16 w][16 kx]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let N = u.N; let ns = u.ns; let nss = ns * ns;
  let b = wg.z;
  let kx = wg.x * 16u + l.x;
  let v = wg.y * 16u + l.y;
  var acc: array<vec2f, 6>;
  for (var c = 0u; c < 6u; c++) { acc[c] = vec2f(0.0); }
  let eb = b * 6u;
  for (var t0 = 0u; t0 < ns; t0 += 16u) {
    let w = t0 + l.x;
    let ok = v < ns && w < ns;
    let off = min(v, ns - 1u) * ns + min(w, ns - 1u);
    for (var c = 0u; c < 6u; c++) {
      sE[c * 256u + l.y * 16u + l.x] = select(vec2f(0.0), E[(eb + c) * nss + off], ok);
    }
    let wf = t0 + l.y;
    let f = F[min(wf, ns - 1u) * N + min(kx, N - 1u)];
    sF[l.y * 16u + l.x] = select(vec2f(0.0), vec2f(f.x, -f.y), wf < ns && kx < N);
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let fc = sF[kk * 16u + l.x];
      for (var c = 0u; c < 6u; c++) { acc[c] += cmul(sE[c * 256u + l.y * 16u + kk], fc); }
    }
    workgroupBarrier();
  }
  if (v < ns && kx < N) {
    for (var c = 0u; c < 6u; c++) { T[((b * 6u + c) * ns + v) * N + kx] = acc[c]; }
  }
}
