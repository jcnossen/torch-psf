// Gradient of the Poisson NLL with respect to the spline knots at fixed emitter parameters: a port of
// torch_psf.estimation.knot_gradient (autograd through CubicSplinePSF.render; the adjoint of the knot read-out is a
// scatter-add). One workgroup per spot: (1) the per-pixel factor r = photons x (1 - d / mu) (0 where mu is clamped at
// 1e-6, as torch's clamp_min gradient), (2) every knot of the spot's 4 x Lw x Lw window gathers the pixels it
// touches (at most 2 x 2 for S = 2) and adds its sum to the global gradient with a float atomic (compare-exchange).
// Also the NLL of each spot (poisson_nll, constant dropped). With a read-noise variance var (photons^2, scalar) the
// shifted-Poisson approximation of poisson_nll: mv = (mu + var) clamped at 1e-6, l = mv - (d + var)+ log mv.

struct U {
  N: u32, P: u32, S: u32, nz: u32,
  nv: u32, nu: u32, rvar: f32, _b: u32,
  z0: f32, hz: f32, oS0: f32, xmax: f32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> knots: array<f32>;
@group(0) @binding(2) var<storage, read> rois: array<f32>;
@group(0) @binding(3) var<storage, read> theta: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> nll: array<f32>;

const WG: u32 = 128u;
var<workgroup> rpix: array<f32, 1024>;      // P <= 31
var<workgroup> part: array<f32, 128>;

fn cr(t: f32) -> vec4f {
  let t2 = t * t; let t3 = t2 * t;
  return vec4f(-0.5 * t + t2 - 0.5 * t3, 1.0 - 2.5 * t2 + 1.5 * t3, 0.5 * t + 2.0 * t2 - 1.5 * t3, -0.5 * t2 + 0.5 * t3);
}

fn addf(i: u32, v: f32) {
  var old = atomicLoad(&grad[i]);
  loop {
    let r = atomicCompareExchangeWeak(&grad[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

@compute @workgroup_size(128)
fn knot_grad(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let spot = wg.x + wg.y * nw.x;
  let lid = li.x;
  if (spot >= u.N) { return; }
  let P = u.P; let S = u.S; let Sf = f32(S);
  let x = theta[spot * 5u]; let y = theta[spot * 5u + 1u]; let z = theta[spot * 5u + 2u];
  let I = theta[spot * 5u + 3u]; let bg = theta[spot * 5u + 4u];
  let tz = clamp((z - u.z0) / u.hz, 1.0, f32(u.nz) - 2.0 - 1e-4);
  let ty = u.oS0 - clamp(y, -u.xmax, u.xmax) * Sf;
  let tx = u.oS0 - clamp(x, -u.xmax, u.xmax) * Sf;
  let iz = floor(tz); let iy = floor(ty); let ix = floor(tx);
  let wz = cr(tz - iz); let wy = cr(ty - iy); let wx = cr(tx - ix);
  let base = (i32(iz) - 1) * i32(u.nv * u.nu) + (i32(iy) - 1) * i32(u.nu) + (i32(ix) - 1);
  var nl = 0.0;
  for (var p = lid; p < P * P; p += WG) {
    let i = p / P; let j = p % P;
    var v = 0.0;
    for (var c = 0u; c < 4u; c++) {
      var pv = 0.0;
      for (var r = 0u; r < 4u; r++) {
        let row = base + i32(c * u.nv * u.nu + (i * S + r) * u.nu + j * S);
        pv += wy[r] * dot(vec4f(knots[row], knots[row + 1], knots[row + 2], knots[row + 3]), wx);
      }
      v += wz[c] * pv;
    }
    let mv0 = I * v + bg + u.rvar;
    let dv = max(rois[spot * P * P + p] + u.rvar, 0.0);
    let mv = max(mv0, 1e-6);
    nl += mv - dv * log(mv);
    rpix[p] = select(0.0, I * (1.0 - dv / mv), mv0 >= 1e-6);
  }
  part[lid] = nl;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid < s) { part[lid] += part[lid + s]; }
    workgroupBarrier();
  }
  if (lid == 0u) { nll[spot] = part[0]; }
  // knot window: c in 0..3, ry, rx in 0..Lw-1; pixel (i, j) uses rows i S .. i S + 3 and columns j S .. j S + 3
  let Lw = (P - 1u) * S + 4u;
  for (var e = lid; e < 4u * Lw * Lw; e += WG) {
    let c = e / (Lw * Lw); let ry = (e / Lw) % Lw; let rx = e % Lw;
    var g = 0.0;
    let i1 = min(ry / S, P - 1u); let j1 = min(rx / S, P - 1u);
    let i0 = select(0u, (ry - 3u + S - 1u) / S, ry >= 3u);   // ceil((ry - 3) / S)
    let j0 = select(0u, (rx - 3u + S - 1u) / S, rx >= 3u);
    for (var i = i0; i <= i1; i++) {
      let a = ry - i * S;
      if (a > 3u) { continue; }
      for (var j = j0; j <= j1; j++) {
        let b = rx - j * S;
        if (b > 3u) { continue; }
        g += rpix[i * P + j] * wy[a] * wx[b];
      }
    }
    if (g != 0.0) { addf(u32(base + i32(c * u.nv * u.nu + ry * u.nu + rx)), wz[c] * g); }
  }
}
