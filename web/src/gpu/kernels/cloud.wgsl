// Point-cloud splat renderer (Reconstruction panel). Each localization is one instanced quad covering +-3 sigma
// (at least +-1 px) of its Gaussian; the fragment shader integrates the Gaussian over its screen pixel (erf
// differences) and adds (w, w z) into a float accumulation texture (additive blending: a density image). The points
// live in a storage buffer with spare capacity; the instance count is the number of points written so far, and a
// per-point flag (bit 0) enables / disables a point (filters) without re-uploading anything. A fixed rank per point
// selects a stable subset while dragging (weighted by 1 / keep). A second pass maps density (and the density-weighted
// z) to colours with a global display maximum.

struct Pt { x: f32, y: f32, z: f32, sx: f32, sy: f32, sz: f32, zc: f32, rank: f32 };  // nm (x, y, z, sigmas), um (zc)

struct U {
  W: f32, H: f32, scale: f32, cx: f32,          // canvas px, nm per px, view centre (nm, rotated plane)
  cy: f32, X0: f32, Y0: f32, Z0: f32,           // field centre (nm)
  ct: f32, st: f32, cr: f32, sr: f32,           // cos / sin of tilt and rotation
  zExag: f32, mode: f32, blur: f32, cap: f32,   // mode 0 = precision (CRLB), 1 = fixed blur; blur = fixed / min sigma
  keep: f32, wgt: f32, vmax: f32, colorZ: f32,  // subset fraction and its weight; display max (density per px)
  zlo: f32, zhi: f32, _a: f32, _b: f32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pts: array<Pt>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) c: vec2f,        // centre (canvas px)
  @location(1) s: vec2f,        // sigma (canvas px)
  @location(2) zc: f32,
};

fn erf(x0: f32) -> f32 {        // Abramowitz and Stegun 7.1.26
  let x = abs(x0);
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return select(-y, y, x0 >= 0.0);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(2.0, 2.0, 0.0, 1.0);            // off screen = not drawn
  if ((flags[ii] & 1u) == 0u) { return o; }
  let p = pts[ii];
  if (p.rank >= u.keep) { return o; }
  let Xp = p.x - u.X0; let Yp = p.y - u.Y0; let Zp = (p.z - u.Z0) * u.zExag;
  let uu = Xp * u.cr + Yp * u.sr; let vv = -Xp * u.sr + Yp * u.cr;
  let c = vec2f((uu - u.cx) / u.scale + u.W * 0.5, (vv * u.ct - Zp * u.st - u.cy) / u.scale + u.H * 0.5);
  var s: vec2f;
  if (u.mode < 0.5) {
    let sx2 = p.sx * p.sx; let sy2 = p.sy * p.sy; let sz = p.sz * u.zExag;
    s = vec2f(sqrt(sx2 * u.cr * u.cr + sy2 * u.sr * u.sr), sqrt((sx2 * u.sr * u.sr + sy2 * u.cr * u.cr) * u.ct * u.ct + sz * sz * u.st * u.st));
    s = clamp(s, vec2f(u.blur), vec2f(u.cap));
  } else { s = vec2f(u.blur); }
  s = s / u.scale;
  let h = max(3.0 * s, vec2f(1.0));
  let corners = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));
  let q = c + corners[vi] * h;
  o.pos = vec4f(q.x / u.W * 2.0 - 1.0, 1.0 - q.y / u.H * 2.0, 0.0, 1.0);
  o.c = c; o.s = max(s, vec2f(1e-3)); o.zc = p.zc;
  return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  let px = floor(i.pos.xy);
  let k = 1.0 / (sqrt(2.0) * i.s);
  let wx = 0.5 * (erf((px.x + 1.0 - i.c.x) * k.x) - erf((px.x - i.c.x) * k.x));
  let wy = 0.5 * (erf((px.y + 1.0 - i.c.y) * k.y) - erf((px.y - i.c.y) * k.y));
  let w = wx * wy * u.wgt;
  return vec4f(w, w * i.zc, 0.0, 0.0);
}

// ------------------------------------------------------------------ colour pass
@group(0) @binding(3) var acc: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[vi], 0.0, 1.0);
}

fn hue(t: f32) -> vec3f {        // blue (0) -> cyan -> green -> yellow -> red (1)
  let hh = (1.0 - t) * 4.0; let x = 1.0 - abs((hh % 2.0) - 1.0);
  if (hh < 1.0) { return vec3f(1.0, x, 0.0); }
  if (hh < 2.0) { return vec3f(x, 1.0, 0.0); }
  if (hh < 3.0) { return vec3f(0.0, 1.0, x); }
  return vec3f(0.0, x, 1.0);
}

@fragment
fn fs_color(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let a = textureLoad(acc, vec2i(pos.xy), 0);
  let d = a.x;
  if (d <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let v = min(1.0, pow(d / u.vmax, 0.6));
  if (u.colorZ > 0.5) {
    let t = clamp((a.y / d - u.zlo) / max(1e-9, u.zhi - u.zlo), 0.0, 1.0);
    return vec4f(hue(t) * v, 1.0);
  }
  return vec4f(clamp(3.0 * v, 0.0, 1.0), clamp(3.0 * v - 1.0, 0.0, 1.0), clamp(3.0 * v - 2.0, 0.0, 1.0), 1.0);
}
