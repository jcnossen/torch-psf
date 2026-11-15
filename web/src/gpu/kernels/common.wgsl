// Shared uniform block of the vectorial kernels (prepended to each kernel source by ops.ts).
struct U {
  N: u32,        // pupil grid size
  ns: u32,       // oversampled image size = R * os
  R: u32,        // ROI size (px)
  os: u32,       // oversampling
  D: u32,        // number of directions (value + derivatives) in this pass
  B: u32,        // images in this chunk
  b0: u32,       // index of the chunk's first image in the global image arrays
  theta0: u32,   // dft2: first direction of this dispatch
  C: f32,        // normalization (psf = C * mean_os(I))
  px: f32,       // pixel size (um)
  nA: u32,       // normal equations: number of atomic columns (D + 1)
  nslot: u32,    // normal equations: slots per image
  S: u32,        // knot build: spline oversampling (knots [nz, R S, R S])
  _p0: u32,
  _p1: u32,
  _p2: u32,
};
@group(0) @binding(0) var<uniform> u: U;

fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
