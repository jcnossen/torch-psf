// Spot detection kernels, a port of torch_psf.detect.detect_spots (band-pass = difference of Gaussians):
//   blur_y / blur_x: separable Gaussian blur with zero padding (torch conv2d, padding = len(k) // 2), first along y
//                    (rows), then along x, as torch_psf.detect.blur;
//   dog:             f = blur(sigma0) - blur(sigma1);
//   hist_hi/hist_lo: radix-select histograms of order-preserving float keys (exact median of a chunk, as torch.median);
//   peak_map:        f == max_pool2d(f, nms, stride 1, padding nms // 2 with -inf) and f > threshold x noise;
//   peaks_final:     border rule, isolation (another peak within +-iso px in the same frame drops both), output list.
// Frames: [T, H, W] f32 photons, row-major; one invocation per pixel of a batch of T frames that starts at element
// offset soff of src (dst at doff, src2 at s2off).

struct P {
  T: u32, H: u32, W: u32, klen: u32,
  mode: u32, prefix: u32, kth: u32, nms: u32,
  thr: f32, med: f32, border: u32, iso: u32,
  frame0: u32, maxpk: u32, soff: u32, doff: u32,
  s2off: u32, _a: u32, _b: u32, _c: u32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> kern: array<f32>;
@group(0) @binding(4) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> peaks: array<u32>;     // (t, y, x, bits of f) per peak; the count is hist[0]
@group(0) @binding(6) var<storage, read> src2: array<f32>;

fn n_px() -> u32 { return p.T * p.H * p.W; }
fn gid(g: vec3u, n: vec3u) -> u32 { return g.x + g.y * n.x * 256u; }

@compute @workgroup_size(256)
fn blur_y(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let i = gid(g, n);
  if (i >= n_px()) { return; }
  let x = i % p.W; let y = (i / p.W) % p.H; let t = i / (p.W * p.H);
  let r = i32(p.klen / 2u);
  var s = 0.0;
  for (var k = 0; k < i32(p.klen); k++) {
    let yy = i32(y) + k - r;
    if (yy >= 0 && yy < i32(p.H)) { s += kern[k] * src[p.soff + (t * p.H + u32(yy)) * p.W + x]; }
  }
  dst[p.doff + i] = s;
}

@compute @workgroup_size(256)
fn blur_x(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let i = gid(g, n);
  if (i >= n_px()) { return; }
  let x = i % p.W; let row = i / p.W;
  let r = i32(p.klen / 2u);
  var s = 0.0;
  for (var k = 0; k < i32(p.klen); k++) {
    let xx = i32(x) + k - r;
    if (xx >= 0 && xx < i32(p.W)) { s += kern[k] * src[p.soff + row * p.W + u32(xx)]; }
  }
  dst[p.doff + i] = s;
}

@compute @workgroup_size(256)
fn dog(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let i = gid(g, n);
  if (i >= n_px()) { return; }
  dst[p.doff + i] = src[p.soff + i] - src2[p.s2off + i];
}

// order-preserving key of a float (larger float -> larger key)
fn fkey(v: f32) -> u32 {
  let b = bitcast<u32>(v);
  if ((b & 0x80000000u) != 0u) { return ~b; }
  return b | 0x80000000u;
}

fn val(i: u32) -> f32 {
  if (p.mode == 1u) { return abs(src[p.soff + i] - p.med); }
  return src[p.soff + i];
}

@compute @workgroup_size(256)
fn hist_hi(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let i = gid(g, n);
  if (i >= n_px()) { return; }
  atomicAdd(&hist[fkey(val(i)) >> 16u], 1u);
}

@compute @workgroup_size(256)
fn hist_lo(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let i = gid(g, n);
  if (i >= n_px()) { return; }
  let k = fkey(val(i));
  if ((k >> 16u) == p.prefix) { atomicAdd(&hist[k & 0xffffu], 1u); }
}

fn fval(t: u32, y: i32, x: i32) -> f32 {
  if (y < 0 || x < 0 || y >= i32(p.H) || x >= i32(p.W)) { return -3.4e38; }
  return src[p.soff + (t * p.H + u32(y)) * p.W + u32(x)];
}

@compute @workgroup_size(256)
fn peak_map(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let i = gid(g, n);
  if (i >= n_px()) { return; }
  let x = i32(i % p.W); let y = i32((i / p.W) % p.H); let t = i / (p.W * p.H);
  let v = fval(t, y, x);
  var pk = v > p.thr;
  let r = i32(p.nms / 2u);
  for (var dy = -r; dy <= r && pk; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      if (fval(t, y + dy, x + dx) > v) { pk = false; break; }
    }
  }
  dst[p.doff + i] = select(0.0, 1.0, pk);
}

@compute @workgroup_size(256)
fn peaks_final(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) n: vec3u) {
  let i = gid(g, n);
  if (i >= n_px()) { return; }
  if (src2[p.s2off + i] < 0.5) { return; }
  let x = i32(i % p.W); let y = i32((i / p.W) % p.H); let t = i / (p.W * p.H);
  let b = i32(p.border);
  if (x < b || x >= i32(p.W) - b || y < b || y >= i32(p.H) - b) { return; }
  if (p.iso > 0u) {
    let r = i32(p.iso);
    var cnt = 0.0;
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        let yy = y + dy; let xx = x + dx;
        if (yy >= 0 && xx >= 0 && yy < i32(p.H) && xx < i32(p.W)) { cnt += src2[p.s2off + (t * p.H + u32(yy)) * p.W + u32(xx)]; }
      }
    }
    if (cnt >= 1.5) { return; }
  }
  let k = atomicAdd(&hist[0], 1u);
  if (k < p.maxpk) {
    peaks[4u * k] = t + p.frame0; peaks[4u * k + 1u] = u32(y); peaks[4u * k + 2u] = u32(x);
    peaks[4u * k + 3u] = bitcast<u32>(src[p.soff + i]);
  }
}
