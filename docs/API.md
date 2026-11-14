# torch-psf: stable API

The functions below are the reference implementation that other code (the browser app in `web/`, downstream
pipelines) ports or calls. Their signatures, defaults and math are stable: **after the first public release any change
to their math or defaults is announced** (release notes / the commit message, and to the maintainers of the ports)
before it is merged. Units: px for lateral positions (x = column, y = row, pixel centres at integers; inside a ROI
relative to its centre pixel), um for z, focus, wavelengths and pixel sizes, photons (e-) for intensities, photons / px
for backgrounds. Defaults: `device=None` = cuda if available, else cpu; float32.

## Changes

### 2026-11-14: class design (breaking; no aliases)

| old | new |
|---|---|
| `ConvSplinePSF` | `ConvCubicSplinePSF` (same knot layout and math) |
| `ConvSplinePSF.from_vectorial(vec, roisize, z_range, ...)` | `vec.to_spline(roisize, z_range, ...)` (any 3D model: `PSF.to_spline`) |
| `GaussianPSF(..., gamma_um=, d_um=, z_range=)` (astigmatic) | `AstigmaticGaussianPSF(roisize, sigma_px, gamma_um, d_um, z_range)`; `GaussianPSF(roisize, sigma_px)` is 2D only |
| `torch_psf.localize(psf, rois)` (top level) | `psf.fit(rois)` -> localization table; the low-level `torch_psf.fit.localize` (theta, nll) stays |
| `torch_psf.crlb(psf, theta)` (top level) | `psf.crlb(theta)`; `torch_psf.fit.crlb` stays |
| `localize_movie(movie, psf, camera)` | `psf.localize(movie, camera)` |
| `load_psf(path)` / `save_psf(path, psf)` | `PSF.load(path)` / `psf.save(path)` (implementation: `torch_psf.io.load` / `save`) |
| `estimate_psf_beads(stacks, dz_um, ...)` | `ZStackEstimator(model, start / optics, camera, ...).estimate(stacks, dz_um)` -> PSF |
| `estimate_psf_insitu(movie, ...)` | `InSituEstimator(start / optics, camera, ..., refine=False).estimate(movie)` -> PSF |
| `PSFEstimate` (.psf, .report, .history, .data, .save) | removed: estimators return the PSF; `estimator.report`, `.history`, `.data` |
| `fit_beads`, `fit_zernike_knots`, `fit_pupil_knots`, `fit_free_knots`, `fit_tracks` return PSFEstimate | they return `dict(psf, report, history, ...)` (same fields as the former `.data`) |
| `torch_psf.estimate.z_scale` | `torch_psf.insitu.z_scale` |
| psf.json type `gaussian` with gamma (3D) | type `astig_gaussian`; `gaussian` is 2D |
| in-situ default pipeline | now rejects spots the single-emitter model does not explain (Poisson deviance per pixel > 1 + 3 sqrt(2 / P^2) after a first pass; refit on the rest; `reject_overlap=True`, `deviance_k=3`) |
| (new) | `simulate.filament_sites`, `simulate_movie(sites=...)`, `table.to_csv`, `insitu.spot_deviance`, `PSF.expected`, `PSF.fisher`, `n_dims` |
| `ConvSplinePSF.render(..., grad=True)`: d/dz = 0 outside the knot z range | **math change**: x, y, z are clamped and the derivatives are those at the clamped position (d/dz not zeroed), as the earlier lookup-table code did; inside the range nothing changes |
| ConvSplinePSF: odd roisize only, fixed knot layout | even roisize allowed; `knots=` with `knot_u0_px` / `knot_z0_um` wraps a stored knot array of another layout (the offset limit is reduced to what it supports) |
| `load_psf(path)` | `PSF.load(path_or_dict)` (a parsed dict is accepted) |
| spline knots only from a vectorial model | `PSF.to_spline` works for every model; 2D models with an explicit `z_range` (constant in z) |
| `fit_tracks(...)` | adds `fix_zernike`, `fix_focus`; `report['nll_per_px']` is the NLL after the last step (`nll_per_px_before_last_step` the value the last step started from) |
| `fit_beads(...)` | adds `fit_dz_scale` (z-step scale), `fit_height` (bead height), `intensity='bead_x_plane'` (per-plane factors, e.g. bleaching); held-out beads (weight 0) are refitted (position / photons / bg, PSF fixed) before they are scored (`report['heldout_nll_per_px']`); `bead_diameter_um` as before. ZStackEstimator exposes the same options |

## Classes

```
PSF (abstract; torch_psf.psf)                 render(pos, grad=False)            [the only abstract method]
  roisize, z_range (None = 2D), n_dims,       expected(theta) -> mu [N, P, P]
  xy_limit, z_limits, dev, to(device),        fisher(theta, var=None), crlb(theta, var=None) -> [N, 5]
  resized(roisize)                            fit(rois | Spots, camera=None, theta0=None, corners=None, frames=None)
                                                  -> localization table (LOC_DTYPE)
                                              localize(movie, camera=None, frames=None, ...) -> table
                                              save(path) / PSF.load(path) (type dispatch; plain calibration json too)
                                              to_spline(roisize, z_range, oversample, dz_um, zpad, xy_limit)
├── GaussianPSF(roisize, sigma_px)                                   2D
├── AstigmaticGaussianPSF(roisize, sigma_px, gamma_um, d_um, z_range)  3D
├── VectorialPSF(roisize, pixel_um, NA, n_imm, n_med, wavelength_um, zernike_indices, npupil, oversample,
│                focus_um, z_range)  (also a torch.nn.Module: forward(pos, focus, ...), set_zernike, resized,
│                from_optics, optics, to_spline(..., pos_fn))
└── ConvCubicSplinePSF(roisize, z_range, oversample, dz_um, zpad, xy_limit)  knots [nz, nv, nu]; render(..., knots=);
                 knots_from_model, knots_from_vectorial, vjp_vectorial, from_zstack

Estimator (torch_psf.estimators)             report (dict), history (list), data (dict), progress / stop hooks
├── ZStackEstimator(model='vectorial' | 'spline' | 'astig_gauss', start=None, optics=None, camera=None, roisize=21,
│                   nbeads=8, sign=1.0, iters=300, holdout=None, drift=True, regularize=3000.0, ...)
│     .estimate(stacks, dz_um) -> VectorialPSF | ConvCubicSplinePSF | AstigmaticGaussianPSF
└── InSituEstimator(start=None, optics=None, camera=None, focus_um=None, roisize=15, z_range=(0, 0.8), nspots=8000,
                    nheldout=4000, method='zernike', iters=80, reject_overlap=True, deviance_k=3.0, refine=False, ...)
      .estimate(movie, frames=None) -> VectorialPSF (or the refined ConvCubicSplinePSF if it wins on held-out spots)
```
ZStackEstimator conventions: plane k at focus `sign dz_um (k - (nk - 1) / 2)`; the vectorial result has `focus_um = 0`;
the spline and astigmatic-Gaussian results have z = -focus (bead defocus in stage um). InSituEstimator steps:
detection (`extract_spots`), checkerboard split (training / held-out), `fit_zernike_knots` (or `fit_pupil_knots`),
overlap rejection and a second pass, optional refinement (`fit_free_knots` with ref = the vectorial knots, lam_ref
3000, lam_s 1000; kept only if the held-out NLL per pixel improves; a z-warping warning is added), held-out
comparison (`compare_heldout` of start / vectorial / refined), `z_scale` of the result against the start.

## Detection and spots (`torch_psf.detect`, `torch_psf.spots`)

`detect_spots(frames, threshold=6.0, sigma=(2.0, 6.0), nms=9, noise='mad', border=0, isolation=None, median=False)`
- `frames` [T, H, W] float (photons). Band-pass `f`: difference of Gaussians `blur(frames, 2) - blur(frames, 6)`;
  `blur` = separable Gaussian, kernel `exp(-x^2 / (2 s^2))` on integer x in `[-int(4 s), int(4 s)]`, normalized to sum
  1, **zero padding** (conv2d padding = kernel radius). `median=True`: `blur(frames - temporal median, sigma[0])`.
- Noise `sig`: `'mad'` = `1.4826 * median(|f - median(f)|)` over **all pixels of all frames passed in one call**
  (`torch.median`: the lower median for an even count; extract_spots calls it per chunk of `chunk` = 500 frames);
  `'std'` = standard deviation of `f` per frame.
- Peaks: `f == max_pool2d(f, nms, stride 1, padding nms // 2)` (plateaus give several peaks) and `f > threshold * sig`.
- Kept if `border <= x < W - border` and `border <= y < H - border`; with `isolation = r`: dropped when another peak
  (above the threshold, before the border rule) lies within the (2r + 1) x (2r + 1) box in the same frame (both
  peaks are dropped).
- Returns dict(frame, y, x (long, frame relative to frames[0]), score = f / sig at the peak), sorted by (frame, y, x).

`extract_spots(movie, roisize, camera=None, frames=None, chunk=500, device=None, max_spots=None, stop=None,
progress=None, **detect_kw)`
- Reads `chunk` frames at a time (`movie`: path / array / reader), converts with the camera
  (`(adu - offset) * gain`, per-pixel maps cropped by the ROI corner), calls `detect_spots` with `border` defaulting
  to `(roisize - 1) / 2 + 2`, and cuts ROIs centred on the peak pixel: corner `(x0, y0) = (x - h, y - h)`,
  h = (roisize - 1) / 2. Returns `Spots(rois [n, P, P], frame, corner [n, 2] (x0, y0), score, var, frames_read)`.

## Fitting (`torch_psf.fit`; used by `PSF.fit`, `PSF.crlb`)

`localize(psf, rois, theta0=None, var=None, iters=30, zstarts=5, batch=3000, fixed=(), weight=None)`
- theta = (x, y, z, photons, bg); model `mu = photons * psf.render(x, y, z) + bg`.
- Starting values (`initial_theta`): x = y = 0, bg = the 0.2 quantile of the ROI (`torch.quantile`, linear
  interpolation), photons = max(sum(roi) - bg P^2, 50). Without theta0 a 3D model is started at `zstarts` heights
  `linspace(z_range[0], z_range[1], zstarts + 2)[1:-1]` and the lowest NLL wins; 2D models (z_range None) fix z.
- Levenberg-Marquardt (`lm_poisson`): gradient `g = J^T ((d + v) / (mu + v) - 1)`, Hessian `H = J^T diag(1 / (mu + v))
  J` (v = read-noise variance, 0 without sCMOS), damping `H + lam diag(H)` (diag clamped >= 1e-12), lam starts at 1,
  x 0.3 after an accepted step (NLL decreased), x 5 after a rejected one, clamped to [1e-4, 1e6]; the step is solved in
  float64; mu clamped >= 1e-6; NLL = sum(mu' - (d + v) log mu'), mu' = mu + v. Parameters are clamped after every step
  (`clamp_theta`): |x|, |y| <= xy_limit - 0.05, z inside the model's z limits (a spline: `[z0 + dz, z0 + (nz - 2) dz]`),
  photons >= 1, bg >= 1e-3. Returns theta [N, 5], NLL [N].

`crlb(psf, theta, var=None, weight=None, ridge=0.0)`: sqrt of the diagonal of the inverse Fisher matrix
`F = J^T diag(1 / (mu + v)) J` (float64 inverse), [N, 5]; 2D models: z column NaN (z not estimated).

`at_limit(theta, psf, margin=0.06)`: `max(|x|, |y|) > xy_limit - margin`.

## The convolutional cubic spline (`torch_psf.spline.ConvCubicSplinePSF`)

`ConvCubicSplinePSF(roisize, z_range, oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0, device=None)`: knots
`[nz, nv, nu]`, nz = round((hi - lo) / dz) + 1 + 2 zpad, plane k at z0 + k dz, z0 = lo - zpad dz;
c = ceil(h + xy_limit) + 2 (h = (roisize - 1) / 2), Pl = 2c + 1, nv = nu = Pl * oversample; knot index j of u is at
u = u0 + j / S, u0 = -(c S + S - 1) / S.
- `render(pos, grad=False, knots=None)`: patch[i, j] = g(o_j - x, o_i - y, z) by the Catmull-Rom spline (tau 0.5,
  per-axis weights `[1, t, t^2, t^3] @ G`) over the 4 x 4 x 4 knots around the emitter, as one conv2d with stride S;
  x, y clamped to +-xy_limit; derivatives d/dx = -S (conv with d/du weights), d/dy likewise, d/dz = (conv with d/dt
  weights) / dz, set to 0 outside the knot z range. The knot block is read with `index_select`.
- `knots_from_vectorial(vectorial, pos_fn=None, chunk=4)`: renders the vectorial model (roisize Pl) at the S x S
  sub-pixel shifts (a / S, b / S) for every knot plane and interleaves them (knot j S + (S - 1 - a) <- pixel j, shift
  a). `pos_fn(z_knot) -> (emitter z, focus)`, default emitter height at `vectorial.focus_um`.
- `vjp_vectorial(vectorial, pos_fn, grad_knots, names=('zcoef',), chunk=4)`: (d knots / d parameters)^T grad_knots.
- `vectorial.to_spline(roisize, z_range=None, ...)` (any model: `PSF.to_spline`), `from_zstack(zstack, z0_um, dz_um, roisize)`.

## In-situ estimation (`torch_psf.insitu`, `torch_psf.estimate`)

`fit_zernike_knots(spline, rois, vectorial, pos_fn=None, iters=80, lr=2e-3, route='K', loc_iters=3, ...)`: per
iteration: build the knots from the current Zernikes; localize (30 LM iterations the first time, then `loc_iters`,
warm-started); the knot gradient of the total NLL (autograd through `render`) and one VJP to the Zernikes; an Adam step
(beta 0.9 / 0.999, eps 1e-12, bias-corrected) of size `lr` waves. Returns dict(psf, report, history, zcoef, theta, knots) (report: at_limit, z_scale,
z_offset_um, nll, seconds).

`fit_free_knots(spline, knots0, rois=None, stack=None, ref=None, lam_ref=0, lam_s=0, w_stack=1, outer=12, inner=8, ...)`:
`outer` rounds of (localize; L-BFGS with `inner` iterations, history 10, strong-Wolfe line search on the knots);
loss = NLL(rois) + w_stack NLL(stack) + lam_ref sum((K - ref)^2 / (|ref| + 1e-4)) + lam_s sum over z, y, x of
(second difference)^2 / (|knots0| + 1e-4). Report carries the free-knot z-warp warning.

`split_checkerboard(x, y, tile=16)`: `(floor(x / tile) + floor(y / tile)) % 2` (image px).

`compare_heldout(psfs, rois, var=None)`: for each PSF: localize every ROI (no theta0), mean NLL per pixel
(`nll.mean() / P^2`), the fraction at the offset limit, and the difference to the first PSF.

`z_scale(psf_new, psf_ref, z_range=None, n=33, photons=1e5, bg=1)`: noise-free emitters at x = y = 0 and n heights
`linspace(z_range)` rendered with psf_new, localized with psf_ref; least-squares line through the inner 80 % (indices
`int(0.1 n) .. n - int(0.1 n)`): returns (slope d z_ref / d z_new, offset at the centre of that range, um).

## Tables and simulation

`table.LOC_DTYPE`: `frame i4, x f4, y f4, z f4, photons f4, bg f4, crlb_x f4, crlb_y f4, crlb_z f4, crlb_photons f4,
crlb_bg f4, loglik f4 (= -NLL), chi2 f4 (mean of (d - mu)^2 / (mu + v) per pixel), at_limit ?` (image px, um, photons).

`simulate.simulate_emitters`, `simulate_bead_stack`, `simulate_movie`: seeded by a `torch.Generator`; the order of the
random draws is part of the API (reference data depend on it). `simulate_movie(..., sites=(x, y, z))` takes given
binding sites (px, px, um) instead of drawing them (no random draws for the sites then); `filament_sites(...)` draws
sites along smooth random filaments (a microtubule-like test structure).

`simulate.simulate_reference_movie(psf='astigmatic' | 'tetrapod' | VectorialPSF, T=200, H=64, W=64,
emitters_per_frame=5.0, mean_on=3.0, photons=3000.0, photons_logsd=0.3, bg=20.0, gain_e_per_adu=0.5, offset_adu=100.0,
readnoise_e=1.5, optics=None, seed=0, device=None)`: a small deterministic movie for reference cases: 4 x
emitters_per_frame binding sites (uniform in the frame minus 4 px, z uniform in the PSF's z range), two-state blinking
with the stationary on-fraction emitters_per_frame / nsites, log-normal photons, Poisson noise, then
`adu = round(photons / gain + offset + N(0, readnoise / gain))`. Presets (`simulate.PRESETS`): astigmatic = Z6 0.10,
Z5 0.02, Z11 0.03 waves, focus 0.4 um, z 0 ... 0.8 um; tetrapod = Z6 0.20, Z12 -0.20 waves, focus 1.0 um, z 0 ... 2 um;
optics 0.108 um pixels, NA 1.49, 0.68 um, n 1.518 / 1.33. Returns dict(movie_adu uint16 [T, H, W], truth [n, 6] =
(frame, x, y, z, photons, site), psf, camera, params). Deterministic for a given seed and device (CPU and CUDA random
streams differ).
