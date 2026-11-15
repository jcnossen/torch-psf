// Photon transfer (torch_psf.camera.photon_transfer), occupancy: one invocation per detection (frame, y, x) sets the
// bit of frame t in every pixel of the (2R + 1)^2 square around it, clipped to the frame (= max_pool2d(occ, 2R + 1,
// stride 1, padding R) > 0). Bits: occ[(t / 32) * HW + p], bit t % 32.

struct U { HW: u32, H: u32, W: u32, R: u32, ndet: u32, _a: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> det: array<u32>;              // (frame, y, x) per detection
@group(0) @binding(2) var<storage, read_write> occ: array<atomic<u32>>;

@compute @workgroup_size(64)
fn occupancy(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = g.x + g.y * nw.x * 64u;
  if (i >= u.ndet) { return; }
  let t = det[3u * i];
  let y = i32(det[3u * i + 1u]);
  let x = i32(det[3u * i + 2u]);
  let R = i32(u.R);
  let bit = 1u << (t & 31u);
  let base = (t >> 5u) * u.HW;
  for (var yy = max(y - R, 0); yy <= min(y + R, i32(u.H) - 1); yy++) {
    for (var xx = max(x - R, 0); xx <= min(x + R, i32(u.W) - 1); xx++) {
      atomicOr(&occ[base + u32(yy) * u.W + u32(xx)], bit);
    }
  }
}
