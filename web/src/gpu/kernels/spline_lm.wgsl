// Per-spot maximum-likelihood fit on a CubicSplinePSF: a port of torch_psf.fit.localize / lm_poisson / crlb with the
// spline of torch_psf.spline.CubicSplinePSF.render (Catmull-Rom knots, value and d/dx, d/dy, d/dz).
// One workgroup per spot; the threads share the pixels, the normal equations are reduced in workgroup memory and
// thread 0 does the 5 x 5 solve. Same damping schedule as lm_poisson: H + lam diag(H), lam x 0.3 after an accepted
// step, x 5 after a rejected one (clamped to 1e-4 .. 1e6); theta clamped as clamp_theta (margin 0.05).
// Output per spot: theta[5], nll, crlb[5] (at the final theta), chi2 (mean squared normalized residual), deviance per
// pixel (torch_psf.estimation.spot_deviance: 2 sum(d log(d / mu) - (d - mu)) / P^2).
// Outside the knot z range z is clamped and d/dz is that of the spline at the clamped position (torch_psf >= the class
// redesign: a fit that starts outside still sees a slope towards the inside).

struct U {
  N: u32, P: u32, S: u32, nz: u32,
  nv: u32, nu: u32, iters: u32, fixed: u32,       // fixed: bit mask of theta indices held at theta0
  z0: f32, hz: f32, oS0: f32, xmax: f32,
  zmin: f32, zmax: f32, lam0: f32, rvar: f32,      // z clamp (the spline's z_limits); scalar read-noise variance (e^2)
  first: u32, _a: u32, _b: u32, _c: u32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> knots: array<f32>;
@group(0) @binding(2) var<storage, read> rois: array<f32>;
@group(0) @binding(3) var<storage, read> theta0: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;  // [N, 13]

const WG: u32 = 128u;
const NR: u32 = 23u;                     // reduced values: gr[5], H[15] (upper triangle), nll, chi2, deviance
var<workgroup> red: array<f32, 2944>;    // NR x WG
var<workgroup> th: array<f32, 5>;
var<workgroup> tn: array<f32, 5>;
var<workgroup> flag: u32;

// Catmull-Rom weights (tau 0.5): w = [1, t, t^2, t^3] @ G, dw = [0, 1, 2t, 3t^2] @ G
fn cr(t: f32) -> vec4f {
  let t2 = t * t; let t3 = t2 * t;
  return vec4f(-0.5 * t + t2 - 0.5 * t3, 1.0 - 2.5 * t2 + 1.5 * t3, 0.5 * t + 2.0 * t2 - 1.5 * t3, -0.5 * t2 + 0.5 * t3);
}
fn dcr(t: f32) -> vec4f {
  let t2 = t * t;
  return vec4f(-0.5 + 2.0 * t - 1.5 * t2, -5.0 * t + 4.5 * t2, 0.5 + 4.0 * t - 4.5 * t2, -t + 1.5 * t2);
}

struct Ax { base: i32, wz: vec4f, wy: vec4f, wx: vec4f, dz: vec4f, dy: vec4f, dx: vec4f };

fn axes(x: f32, y: f32, z: f32) -> Ax {
  var a: Ax;
  let S = f32(u.S);
  let tz = clamp((z - u.z0) / u.hz, 1.0, f32(u.nz) - 2.0 - 1e-4);
  let ty = u.oS0 - clamp(y, -u.xmax, u.xmax) * S;
  let tx = u.oS0 - clamp(x, -u.xmax, u.xmax) * S;
  let iz = floor(tz); let iy = floor(ty); let ix = floor(tx);
  let fz = tz - iz; let fy = ty - iy; let fx = tx - ix;
  a.base = (i32(iz) - 1) * i32(u.nv * u.nu) + (i32(iy) - 1) * i32(u.nu) + (i32(ix) - 1);
  a.wz = cr(fz); a.wy = cr(fy); a.wx = cr(fx);
  a.dz = dcr(fz); a.dy = dcr(fy); a.dx = dcr(fx);
  return a;
}

// value, d/dx, d/dy, d/dz of the normalized PSF at pixel (i, j)
fn render(a: Ax, i: u32, j: u32) -> vec4f {
  var v = 0.0; var gx = 0.0; var gy = 0.0; var gz = 0.0;
  let S = u.S;
  for (var c = 0u; c < 4u; c++) {
    var pv = 0.0; var px = 0.0; var py = 0.0;          // over (a, b) for this z plane
    for (var r = 0u; r < 4u; r++) {
      let row = a.base + i32(c * u.nv * u.nu + (i * S + r) * u.nu + j * S);
      let k = vec4f(knots[row], knots[row + 1], knots[row + 2], knots[row + 3]);
      let sx = dot(k, a.wx); let sdx = dot(k, a.dx);
      pv += a.wy[r] * sx; px += a.wy[r] * sdx; py += a.dy[r] * sx;
    }
    v += a.wz[c] * pv; gx += a.wz[c] * px; gy += a.wz[c] * py; gz += a.dz[c] * pv;
  }
  let Sf = f32(S);
  return vec4f(v, -Sf * gx, -Sf * gy, gz / u.hz);
}

fn clampth(t: ptr<function, array<f32, 5>>) {
  let xm = u.xmax - 0.05;
  (*t)[0] = clamp((*t)[0], -xm, xm);
  (*t)[1] = clamp((*t)[1], -xm, xm);
  (*t)[2] = clamp((*t)[2], u.zmin, u.zmax);
  (*t)[3] = max((*t)[3], 1.0);
  (*t)[4] = max((*t)[4], 1e-3);
}

fn isfixed(k: u32) -> bool { return ((u.fixed >> k) & 1u) != 0u; }

// accumulate gr, H, nll, chi2 of this thread's pixels at theta t (shared array tn or th) into red[*, lid]
fn evaluate(spot: u32, t: array<f32, 5>, lid: u32) {
  var acc: array<f32, 23>;
  for (var q = 0u; q < NR; q++) { acc[q] = 0.0; }
  let a = axes(t[0], t[1], t[2]);
  let P = u.P;
  let I = t[3]; let bg = t[4];
  for (var p = lid; p < P * P; p += WG) {
    let i = p / P; let j = p % P;
    let g = render(a, i, j);
    let d = rois[spot * P * P + p];
    let mu0 = I * g.x + bg;
    var J = array<f32, 5>(I * g.y, I * g.z, I * g.w, g.x, 1.0);
    for (var k = 0u; k < 5u; k++) { if (isfixed(k)) { J[k] = 0.0; } }
    let mu = max(mu0, 1e-6);                 // lm_poisson: mu.clamp_min(mu_min) before gr / H
    let mv = mu + u.rvar;
    let dv = d + u.rvar;
    let rr = dv / mv - 1.0;
    var q = 0u;
    for (var k = 0u; k < 5u; k++) { acc[k] += J[k] * rr; }
    for (var k = 0u; k < 5u; k++) { for (var l = k; l < 5u; l++) { acc[5u + q] += J[k] * J[l] / mv; q++; } }
    // poisson_nll: var None -> mu.clamp(1e-6) - d log mu; with var: (mu + var).clamp - (d + var).clamp(0) log(.)
    var nl: f32;
    if (u.rvar == 0.0) { nl = mu - d * log(mu); }
    else { let m2 = max(mu0 + u.rvar, 1e-6); nl = m2 - max(d + u.rvar, 0.0) * log(m2); }
    acc[20] += nl;
    let r2 = d - mu0;
    acc[21] += r2 * r2 / max(mu0 + u.rvar, 1e-6);
    let dd = d + u.rvar; let mm = mu0 + u.rvar;
    acc[22] += 2.0 * (select(0.0, dd * log(max(dd, 1e-9) / max(mm, 1e-6)), dd > 0.0) - (dd - mm));
  }
  for (var q2 = 0u; q2 < NR; q2++) { red[q2 * WG + lid] = acc[q2]; }
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (lid < s) { for (var q2 = 0u; q2 < NR; q2++) { red[q2 * WG + lid] += red[q2 * WG + lid + s]; } }
    workgroupBarrier();
  }
}

fn hidx(k: u32, l: u32) -> u32 {          // index into the upper-triangle list for k <= l
  let a = min(k, l); let b = max(k, l);
  return a * 5u - (a * (a - 1u)) / 2u + (b - a);
}

// solve A x = b (5 x 5, symmetric positive definite after damping) with Jacobi scaling and Gaussian elimination
fn solve5(A0: array<f32, 25>, b0: array<f32, 5>) -> array<f32, 5> {
  var A = A0; var b = b0; var sc: array<f32, 5>;
  for (var k = 0u; k < 5u; k++) { sc[k] = 1.0 / sqrt(max(A[k * 5u + k], 1e-30)); }
  for (var k = 0u; k < 5u; k++) { b[k] *= sc[k]; for (var l = 0u; l < 5u; l++) { A[k * 5u + l] *= sc[k] * sc[l]; } }
  for (var c = 0u; c < 5u; c++) {
    var piv = c; var m = abs(A[c * 5u + c]);
    for (var r = c + 1u; r < 5u; r++) { if (abs(A[r * 5u + c]) > m) { m = abs(A[r * 5u + c]); piv = r; } }
    if (piv != c) {
      for (var l = 0u; l < 5u; l++) { let t = A[c * 5u + l]; A[c * 5u + l] = A[piv * 5u + l]; A[piv * 5u + l] = t; }
      let tb = b[c]; b[c] = b[piv]; b[piv] = tb;
    }
    let dd = A[c * 5u + c];
    if (dd == 0.0) { continue; }
    for (var r = c + 1u; r < 5u; r++) {
      let f = A[r * 5u + c] / dd;
      for (var l = c; l < 5u; l++) { A[r * 5u + l] -= f * A[c * 5u + l]; }
      b[r] -= f * b[c];
    }
  }
  var x: array<f32, 5>;
  for (var ci = 0; ci < 5; ci++) {
    let c = u32(4 - ci);
    var s = b[c];
    for (var l = c + 1u; l < 5u; l++) { s -= A[c * 5u + l] * x[l]; }
    x[c] = select(0.0, s / A[c * 5u + c], A[c * 5u + c] != 0.0);
  }
  for (var k = 0u; k < 5u; k++) { x[k] *= sc[k]; }
  return x;
}

@compute @workgroup_size(128)
fn lm(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let spot = u.first + wg.x + wg.y * nw.x;
  let lid = li.x;
  if (spot >= u.N) { return; }              // uniform for the whole workgroup
  var t: array<f32, 5>;
  for (var k = 0u; k < 5u; k++) { t[k] = theta0[spot * 5u + k]; }
  clampth(&t);
  evaluate(spot, t, lid);
  var gr: array<f32, 5>; var H: array<f32, 15>; var nll = red[20u * WG]; var chi2 = red[21u * WG]; var dev = red[22u * WG];
  for (var k = 0u; k < 5u; k++) { gr[k] = red[k * WG]; }
  for (var q = 0u; q < 15u; q++) { H[q] = red[(5u + q) * WG]; }
  var lam = u.lam0;
  workgroupBarrier();
  for (var it = 0u; it < u.iters; it++) {
    var tnew: array<f32, 5>;
    if (lid == 0u) {
      var A: array<f32, 25>;
      for (var k = 0u; k < 5u; k++) { for (var l = 0u; l < 5u; l++) { A[k * 5u + l] = H[hidx(k, l)]; } }
      for (var k = 0u; k < 5u; k++) { A[k * 5u + k] += max(A[k * 5u + k], 1e-12) * lam; }
      for (var k = 0u; k < 5u; k++) {
        if (isfixed(k)) { for (var l = 0u; l < 5u; l++) { A[k * 5u + l] = 0.0; A[l * 5u + k] = 0.0; } A[k * 5u + k] = 1.0; }
      }
      let step = solve5(A, gr);
      for (var k = 0u; k < 5u; k++) { tnew[k] = t[k] + step[k]; }
      clampth(&tnew);
      for (var k = 0u; k < 5u; k++) { tn[k] = tnew[k]; }
    }
    workgroupBarrier();
    for (var k = 0u; k < 5u; k++) { tnew[k] = tn[k]; }
    workgroupBarrier();
    evaluate(spot, tnew, lid);
    let nn = red[20u * WG];
    if (nn < nll) {
      t = tnew; nll = nn; chi2 = red[21u * WG]; dev = red[22u * WG];
      for (var k = 0u; k < 5u; k++) { gr[k] = red[k * WG]; }
      for (var q = 0u; q < 15u; q++) { H[q] = red[(5u + q) * WG]; }
      lam = lam * 0.3;
    } else {
      lam = lam * 5.0;
    }
    lam = clamp(lam, 1e-4, 1e6);
    workgroupBarrier();
  }
  if (lid == 0u) {
    // Cramer-Rao bound: Fisher = H (J J / (mu + var), mu clamped at 1e-6) at the final theta, all 5 parameters
    var Fm: array<f32, 25>;
    for (var k = 0u; k < 5u; k++) { for (var l = 0u; l < 5u; l++) { Fm[k * 5u + l] = H[hidx(k, l)]; } }
    let o = spot * 13u;
    for (var k = 0u; k < 5u; k++) {
      var e = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0); e[k] = 1.0;
      let col = solve5(Fm, e);
      outp[o + 6u + k] = sqrt(max(col[k], 0.0));
      outp[o + k] = t[k];
    }
    outp[o + 5u] = nll;
    outp[o + 11u] = chi2 / f32(u.P * u.P);
    outp[o + 12u] = dev / f32(u.P * u.P);
  }
}
