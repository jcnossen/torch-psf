// Knot VJP, step 3 (adjoint of dft1.wgsl, T = Fy PB, and of phase.wgsl, PB_c = P_c exp(i (W + ...))):
//   A[b, c, ky, kx] = sum_v conj(F[v, ky]) A1[b, c, v, kx],
//   gW[b, k] = dL / dW_k = Re sum_c conj(A_c) i PB_c = Im sum_c A_c conj(PB_c)   (k = ky N + kx).
// PB is recomputed here exactly as in phase.wgsl.
@group(0) @binding(1) var<storage, read> F: array<vec2f>;          // [ns, N]
@group(0) @binding(2) var<storage, read> T: array<vec2f>;          // [B, 6, ns, N] (A1)
@group(0) @binding(3) var<storage, read> P: array<vec2f>;          // [6, NN]
@group(0) @binding(4) var<storage, read> kzmed: array<vec2f>;      // [NN]
@group(0) @binding(5) var<storage, read> pk: array<vec4f>;         // [NN]: kx, ky, kz_imm, W (rad)
@group(0) @binding(6) var<storage, read> imgp: array<vec4f>;       // [2 Btot]
@group(0) @binding(7) var<storage, read_write> gWb: array<f32>;    // [B, NN]

var<workgroup> sF: array<vec2f, 256>;     // [16 ky][16 v]
var<workgroup> sA: array<vec2f, 1536>;    // [6][16 v][16 kx]

const TWO_PI: f32 = 6.283185307179586;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let N = u.N; let NN = N * N; let ns = u.ns;
  let b = wg.z;
  let kx = wg.x * 16u + l.x;
  let ky = wg.y * 16u + l.y;
  var acc: array<vec2f, 6>;
  for (var c = 0u; c < 6u; c++) { acc[c] = vec2f(0.0); }
  for (var t0 = 0u; t0 < ns; t0 += 16u) {
    let vf = t0 + l.x;
    let f = F[min(vf, ns - 1u) * N + min(ky, N - 1u)];
    sF[l.y * 16u + l.x] = select(vec2f(0.0), vec2f(f.x, -f.y), vf < ns && ky < N);
    let va = t0 + l.y;
    let ok = va < ns && kx < N;
    let off = min(va, ns - 1u) * N + min(kx, N - 1u);
    for (var c = 0u; c < 6u; c++) {
      sA[c * 256u + l.y * 16u + l.x] = select(vec2f(0.0), T[(b * 6u + c) * ns * N + off], ok);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let fc = sF[l.y * 16u + kk];
      for (var c = 0u; c < 6u; c++) { acc[c] += cmul(fc, sA[c * 256u + kk * 16u + l.x]); }
    }
    workgroupBarrier();
  }
  if (ky >= N || kx >= N) { return; }
  let k = ky * N + kx;
  // the pupil field of phase.wgsl
  let q = imgp[2u * (u.b0 + b)];
  let t = pk[k];
  let kz = kzmed[k];
  var ph = t.w - t.x * q.x * u.px - t.y * q.y * u.px - t.z * q.w + kz.x * q.z;
  ph = ph - TWO_PI * round(ph / TWO_PI);
  let e = vec2f(cos(ph), sin(ph)) * exp(-kz.y * q.z);
  var s = 0.0;
  for (var c = 0u; c < 6u; c++) {
    let pb = cmul(P[c * NN + k], e);
    s += acc[c].y * pb.x - acc[c].x * pb.y;        // Im(A conj(PB))
  }
  gWb[b * NN + k] = s;
}
