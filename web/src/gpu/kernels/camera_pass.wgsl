// Photon transfer (torch_psf.camera.photon_transfer), per-pixel sums: one invocation per pixel loops over the T frames
// of the block (x [T, HW] f32 ADU). A sample is kept if no detection is within R px in this frame or the previous one
// (occupancy bits) and it passes the clips of the earlier rounds, |x - mu_j| < lim_j (float32, as torch; mu_j and
// lim_j = 4 sd_j come from the CPU, which forms them from the sums of round j):
//   round 0..3: count, sum and sum of squares of y = x - x[t = 0] of the kept samples (sums of round r: r clips);
//   round 4:    count, sum and sum of squares of d = x[t] - x[t - 1] over pairs with both samples kept (4 clips).
// The sums are compensated (hi + lo, the exact rounding error of each addition goes into lo) and the squares are
// split so that every term is exact for integer values below 65536: for integer ADU the sums are exact, i.e. the
// same as the float64 sums they replace. Output acc[k * HW + p], k = count, sum hi, sum lo, sumsq hi, sumsq lo.

struct U { T: u32, HW: u32, round: u32, _a: u32 };

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> occ: array<u32>;
@group(0) @binding(3) var<storage, read> clip: array<f32>;           // mu [4][HW], then lim [4][HW]
@group(0) @binding(4) var<storage, read_write> acc: array<f32>;      // [5][HW]

// s.x + v as hi + lo: the rounding error of the float32 addition (TwoSum) is added to lo
fn add2(s: vec2f, v: f32) -> vec2f {
  let t = s.x + v;
  let bp = t - s.x;
  let e = (s.x - (t - bp)) + (v - bp);
  return vec2f(t, s.y + e);
}
// s + y^2 with y = a + b, a = 64 floor(y / 64): a^2, 2ab and b^2 are exact float32 products for integer |y| < 65536
fn addsq(s: vec2f, y: f32) -> vec2f {
  let a = floor(y * 0.015625) * 64.0;
  let b = y - a;
  return add2(add2(add2(s, a * a), 2.0 * a * b), b * b);
}

@compute @workgroup_size(256)
fn sums(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let p = g.x + g.y * nw.x * 256u;
  let HW = u.HW;
  if (p >= HW) { return; }
  let r = u.round;
  let nclip = min(r, 4u);
  var mu: array<f32, 4>;
  var lim: array<f32, 4>;
  for (var j = 0u; j < nclip; j++) { mu[j] = clip[j * HW + p]; lim[j] = clip[(4u + j) * HW + p]; }
  let x0 = x[p];
  var n = 0u;
  var s1 = vec2f(0.0, 0.0);
  var s2 = vec2f(0.0, 0.0);
  var prevKeep = false;
  var prevX = 0.0;
  var prevOcc = 0u;
  for (var t = 0u; t < u.T; t++) {
    let xv = x[t * HW + p];
    let o = (occ[(t >> 5u) * HW + p] >> (t & 31u)) & 1u;
    var k = (o | prevOcc) == 0u;
    for (var j = 0u; j < nclip; j++) { k = k && (abs(xv - mu[j]) < lim[j]); }
    if (r < 4u) {
      if (k) { n++; let y = xv - x0; s1 = add2(s1, y); s2 = addsq(s2, y); }
    } else if (k && prevKeep) {
      n++; let d = xv - prevX; s1 = add2(s1, d); s2 = addsq(s2, d);
    }
    prevKeep = k; prevX = xv; prevOcc = o;
  }
  acc[p] = f32(n);
  acc[HW + p] = s1.x; acc[2u * HW + p] = s1.y;
  acc[3u * HW + p] = s2.x; acc[4u * HW + p] = s2.y;
}
