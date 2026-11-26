# torch-psf: stable API

The functions below are the reference implementation that other code (the browser app in `web/`, downstream
pipelines) ports or calls. Their signatures, defaults and math are stable: **after the first public release any change
to their math or defaults is announced** (release notes / the commit message, and to the maintainers of the ports)
before it is merged. Units: px for lateral positions (x = column, y = row, pixel centres at integers; inside a ROI
relative to its centre pixel), um for z, focus, wavelengths and pixel sizes, photons (e-) for intensities, photons / px
for backgrounds. Defaults: `device=None` = cuda if available, else cpu; float32.

## Dataset (`torch_psf.dataset.Dataset`; format: docs/DATASET_FORMAT.md)

`Dataset(fields, meta=None)`, `Dataset.from_arrays(pos, frame, crlb, photons, bg,
n_frames, meta, **fields)`; `ds['pos']` / `ds.pos`, `len(ds)`, `ds[mask | slice | indices | int]` (all fields),
`ds['name'] = tensor`, `add_fields(**)`, `keys()`, `copy()`, `to(device)`, `numpy()`; `n_dims`, `end_frame`,
`center_frame`, `pixel_nm`, `pos_nm`, `num_frames`, `indices_per_frame()`; filters returning a new object and in-place
`_` variants: `filter(mask)`, `filter_finite()`, `filter_photons(min, max)`, `filter_crlb(max_px, max_nm)`,
`filter_frames(start, stop)`, `crop(x0, y0, x1, y1, shift)`, `crop_from_edge(dist)`; `Dataset.concatenate(items,
set_group)`, `+`; `render(pixel_nm, region, sigma_nm, weights)` (2D histogram), `frc(block=300, pixel_nm=5)` (even / odd
blocks of frames, 5-ring smoothing, 1/7 threshold); files: `save` / `load` (.hdf5 / .h5 this format, .csv
ThunderSTORM, .smlm ShareLoc), `save_picasso` / `load_picasso`, `save_csv` / `load_csv`, `save_smlm` / `load_smlm`.

`ds.estimate_drift(n_frames=None, rows=None, dims=2, device=None) -> Drift` estimates drift with DME v2
(optional dependency). `tp.Drift(curve, info=None)` holds per-frame displacements in pixels and the run record.
`drift.at(center_frame)` samples the curve with endpoint clipping. Use `drift.save(path)` and `tp.Drift.load(path)`
to save and read the result. `ds.drift_corrected(drift, *, force=False) -> Dataset` returns corrected positions
and records `drift.info` in metadata. With `force=True`, a further correction replaces the previous record.
Concatenation raises if correction records differ, including a mix of raw and corrected inputs.

## Classes

```
PSF (abstract; torch_psf.psf)                 render(pos, grad=False)            [the only abstract method]
  roisize, z_range (None = 2D), n_dims,       expected(theta) -> mu [N, P, P]
  xy_limit, z_limits, dev, to(device),        fisher(theta, var=None), crlb(theta, var=None) -> [N, 5]
  resized(roisize)                            fit(rois | Spots, camera=None, theta0=None, corners=None, frames=None)
                                                  -> Dataset
                                              localize(movie, camera=None, frames=None, ...) -> Dataset
                                              save(path) / PSF.load(path) (type dispatch; plain calibration json too)
                                              to_spline(roisize, z_range, oversample, dz_um, zpad, xy_limit)
├── GaussianPSF(roisize, sigma_px)                                   2D
├── AstigmaticGaussianPSF(roisize, sigma_px, gamma_um, d_um, z_range)  3D
├── VectorialPSF(roisize, pixel_um, NA, n_imm, n_med, wavelength_um, zernike_indices, npupil, oversample,
│                focus_um, z_range, pixel_integration='midpoint' | 'exact' | 'gauss_legendre' | 'point')  (also a torch.nn.Module: forward(pos, focus, ...), set_zernike, resized,
│                from_optics, optics, to_spline(...))
├── SplinePSF(roisize, z_range, oversample, dz_um, zpad, xy_limit)  knots [T, nz, nv, nu]; render_splines(..., knots=);
│                to_spline, sample, vjp_vectorial, from_zstack
├── FieldPSF(base, field), TiledSplinePSF(field_psf, tile, ...)   field-dependent (see Field-dependent PSFs)
└── MultiChannelPSF: ChannelsPSF, FourPiPSF, FourPiSplinePSF   images [N, C, P, P] (see Multi-channel PSFs)

Estimator (torch_psf.estimators)             report (dict), history (list), data (dict), progress / stop hooks
├── ZStackEstimator(model='vectorial' | 'spline' | 'astig_gauss', start=None, optics=None, camera=None, roisize=21,
│                   nbeads=8, sign=1.0, iters=300, holdout=None, drift=True, regularize=3000.0, ...)
│     .estimate(stacks, dz_um) -> VectorialPSF | SplinePSF | AstigmaticGaussianPSF
└── InSituEstimator(start=None, optics=None, camera=None, focus_um=None, roisize=15, z_range=(0, 0.8), nspots=8000,
                    nheldout=4000, method='zernike', iters=80, reject_overlap=True, deviance_k=3.0, refine=False,
                    detector=None, zstarts=5, field=None, ...)
      .estimate(movie, frames=None) -> VectorialPSF (or the refined SplinePSF if it wins on held-out spots)
```

### Rendering and optical compositions

All models use `render(pos, grad=False, *, dipM=None, phi=None, xy=None, focus=None, angle=None)`.
Positions are `[N,3]` in lateral pixels and axial micrometres. The result is `[N,P,P]` for one channel
and `[N,C,P,P]` for a multichannel model. With `grad=True`, the tuple contains the image followed by
its x, y and z derivatives. Coupled phase models also return the phase derivative. Pass only options
supported by the model. `dipole_basis` accepts the same keyword interface and returns the six tensor
responses before contraction. `basis(pos, **options)` and `weights(pos, **options)` expose compatible
`[N,C,K,P,P]` and `[N,C,K]` arrays. Sum their products over K. Dipole off-diagonal images include the
factor of two.

```python
from torch_psf import FieldPSF, Polychromatic, ZMode

spectrum = Polychromatic(optics, wavelengths_um=[.64, .72], weights=[.3, .7])
field_model = FieldPSF(spectrum, field_zernikes)
scan = ZMode(field_model, 'stage', height=.02, scale=.9)
images = scan.render(pos, xy=field_positions, dipM=tensors)
spline = scan.to_spline(z_range=(-.3, .3), components='dipole')  # at the field reference
tiled = field_model.to_tiled(tile=128, blend='bilinear', components='dipole')
```

`ZMode(model, 'emitter', focus=f)` varies height at fixed focus. In stage mode the emitter stays at
`height` and the optical focus is `f - pos[:,2] / scale`. A coherent stage scan uses the signed
displacement of each arm and its calibrated `depth_um`. Each wavelength in `Polychromatic` uses
the wavelength-dependent wavevectors, pupil phase in waves scaled by the reference wavelength divided by the sampled wavelength, and its own intensity
normalization. A spline of this model contains the spectrum average. Its spline count is independent
of the number of wavelengths.

Independent spline channels on a shared grid use a channel index into one coefficient bank.
A shared optical model can take `focus` per emitter or per emitter and channel. For mapped channels
pass the ROI **centres** as `xy[N,C,2]` to render, or use `render_geometry(..., corners=..., offsets=...)`
for explicit acquisition geometry. Fitting continues to accept ROI corners.

VectorialPSF forms the emitter pupil, transforms its six fields onto the image grid, and contracts the
dipole tensor or returns its six basis images. FourPiPSF uses the same arm transforms and combines their
polarized fields coherently. Both use `torch_psf.integrate.PixelIntegrator` for the grid, blur, and pixel
sampling. The `point` mode returns centre density per px^2 and supplies the knots for integrated splines.
Like `spectral`, it requires at least two samples per pixel and a grid Nyquist frequency above the optical
band limit. Both use a three-pixel border per side, doubled for coherent intensities. Blurred `box` uses
three pixels per side, and `gauss_legendre` requires an unblurred image.

ZStackEstimator conventions: plane k at focus `sign dz_um (k - (nk - 1) / 2)`; the vectorial result has `focus_um = 0`;
the spline and astigmatic-Gaussian results have z = -focus (bead defocus in stage um). InSituEstimator steps:
detection (`extract_spots`), checkerboard split (training / held-out), `fit_zernike_knots` (or `fit_pupil_knots`),
overlap rejection and a second pass, optional refinement (`fit_free_knots` with ref = the vectorial knots, lam_ref
3000, lam_s 1000; kept only if the held-out NLL per pixel improves; a z-warping warning is added), held-out
comparison (`compare_heldout` of start / vectorial / refined), `z_scale` of the result against the start.

## Multi-channel PSFs (`torch_psf.multichannel`)

```
PSF
└── MultiChannelPSF (abstract)               render_channels(pos_c [N, C, 3], extra [N, k] | None, grad=False)
      n_channels, extra_params, n_params,          -> [N, C, P, P]   [the only abstract method]
      param_names, channel_offsets [C, 2],   render_geometry(pos, grad=False, offsets=None, **extra), expected(theta, offsets)
      intensity_ratio [C],                   mu_and_jacobian(theta, offsets) -> (mu, J [N, 4 + C + k, C, P, P])
      set_channel_offsets(v, trainable),     fisher / covariance / crlb(theta, var, offsets, fixed)
      channel_positions(pos, offsets)        fit_theta(rois [N, C, P, P], theta0, var, offsets, fixed, ...) -> (theta, nll)
      set_mappings(maps), mapped,            fit(rois, ..., offsets=None, fixed=()) -> Dataset
      mapped_positions(pos, offsets, corners)   (all of these take corners=[N, C, 2] with mappings)
                                             localize(movie): NotImplementedError (needs the channel mapping)
    ├── ChannelsPSF(models, channel_offsets=None, intensity_ratio=None, names=None, mappings=None)   independent
    │       channels (models may be FieldPSF / TiledSplinePSF); to_tiled(tile, ...)
    ├── FourPiPSF (also a torch.nn.Module)                   extra_params ('phi',)
    └── FourPiSplinePSF                                      extra_params ('phi',); render_splines(..., knots=)
```

- **theta** = `[x, y, z, I, bg_0 .. bg_{C-1}, *extra]` (4 + C + k columns): x, y (px) of the emitter relative to the
  centre of the reference ROI, z (um), I (photons), one background per channel (photons / px), then the extra
  emitter parameters by name (4Pi: `phi`, the global 4Pi phase, rad). 2D models keep the z column, held fixed (as
  `PSF`); `crlb` gives NaN there. `theta(pos, intensity, bg, **extra)` builds it.
- **Channel positions**: channel c renders the emitter at `pos_c = (x + dx_c + ox_c, y + dy_c + oy_c, z)` with
  `channel_offsets[c] = (dx, dy)` (px; fixed image offset of the channel's model, default zeros; a buffer, or a
  parameter after `set_channel_offsets(v, trainable=True)`) and the optional per-spot offsets `offsets [N, C, 2]` (e.g.
  ROI-corner differences from a channel mapping when the ROIs are cut per channel).
- **Model**: `mu[:, c] = I intensity_ratio[c] g_c(pos_c; extra) + bg_c`; `render` includes `intensity_ratio` (default
  ones; for 4Pi the channel transmissions `tau` are already in `g`).
- **Fit** (`fit_theta`): `torch_psf.fit.lm_poisson` on all channels jointly (the same Levenberg-Marquardt, read-noise
  shift and float64 step as `localize`). Clamp: every channel position inside `xy_limit - 0.05`, z in `z_limits`,
  I >= 1, bg >= 1e-3. Starts: x, y = minus the mean channel offset, bg_c = 0.2 quantile of ROI c, I = signal / summed
  model image; a 3D model without theta0 starts at `zstarts` heights times `default_extra_starts()` (4Pi: phi = 0,
  pi / 2, pi, 3 pi / 2), the lowest NLL wins. `fixed`: names (`'x'`, `'z'`, `'I'`, `'bg'` = all backgrounds,
  `'bg1'`, `'phi'`) or indices, held at their start. The 4Pi phase is returned in (-pi, pi]. `fit` returns a
  Dataset: pos (reference frame; image coordinates with `corners` of the reference ROI), crlb, photons, bg (mean over
  channels) and crlb_bg (of that mean), bg_channels / crlb_bg_channels [N, C], `phi` / `crlb_phi`, theta, loglik,
  chi2, at_limit.
- **4Pi**: `FourPiPSF` channel offsets are the same image shift of both arms (a phase ramp on both pupil fields, in
  each arm's frame), computed with one pupil evaluation per spot and one image transform per channel; with all
  offsets zero all channels share one evaluation. `to_spline` bakes the offsets into the splines (the spline's own
  offsets are zero); `FourPiSplinePSF.channel_offsets` shift each channel's spline evaluation instead (stored in its
  torch-psf json, not in psf.zip, which refuses them). The render and forward calls accept `phi=phase, dipM=tensor`; `expected(theta)` takes the multi-channel theta
  [N, 4 + C + 1].
- **psf.zip**: `PSFZip.read(path).multichannel()` = the coupled model, or a `ChannelsPSF` of the channel models with
  their `intensity_ratio` (offsets zero; the channel `mapping` gives per-spot offsets); `multichannel(field=True,
  mapping=True)`: the field-dependent channel models and the mappings as `ChannelMap`s (below). The format is
  unchanged.

### Channel mappings and several field-dependent channels (`torch_psf.mapping`; `docs/notes/multichannel_field.md`)

```
ChannelMap(order=1, norm=(cx, cy, sx, sy), inv=None, fwd=None, index=1, identity=False, fit='all')   nn.Module
    from_ref(xy) -> channel px (differentiable; inv [nf, 2] px), to_ref(xy) (exact for order 1, else fwd + Newton),
    jacobian(xy) [N, 2, 2], linear(), describe() (shift, rotation, scale, flip), flip(), refit_fwd(extent),
    to_block(index, extent, kind) / ChannelMap.from_block(block, index, crop)   psf.zip identity / affine / poly
    .identity_map(index, norm), .translation(d), .from_affine(A [2, 3]), .fit_points(ref_xy, ch_xy, order, norm)
register_points(ref_xy, ch_xy, frame_shape, flips=(...), order=1) -> (ChannelMap, (i_ref, i_ch), info)
channel_rois(images [c] [F, H, W], frame [N], xy_ref [N, 2], maps, roisize) -> (rois [N, C, P, P], corners [N, C, 2])
MultiChannelPSF.set_mappings(maps) / ChannelsPSF(models, ..., mappings=maps): render / expected / mu_and_jacobian /
    fisher / crlb / fit_theta / fit(..., corners=[N, C, 2])
PSFZip.read(path).multichannel(field=False, mapping=False)
```
- **Mapped geometry**: with mappings, channel c renders the emitter at `pos_c = from_ref_c(corner_0 + h + (x, y)) -
  corner_c - h` (+ channel_offsets + offsets; h = (P - 1) / 2), with the per-spot ROI corners of every channel in its
  crop coordinates (`corners [N, C, 2]`, required); d/dx, d/dy go through the mapping Jacobian (rotation, scale,
  flip), so fits, Fisher information and CRLB are exact for any affine / polynomial map. `fit` returns positions in
  reference-channel coordinates. Without mappings (default) the offset-only geometry is unchanged.
  `ChannelMap.translation(d)` reproduces `channel_offsets = d`.
- **Field-dependent channels**: a ChannelsPSF channel may be a `FieldPSF` / `TiledSplinePSF`; it is evaluated at its
  ROI centre `corner_c + h` in its own channel coordinates (its field block's coordinates). `ChannelsPSF.to_tiled(tile,
  roisize, z_range)` turns every FieldPSF channel into its tiled spline form.
- **ChannelMap**: the SIMCODE PolyMap features (u^i v^j, i outer; u = (x - cx) / sx), one `norm` for both directions,
  coefficients in px (`inv`: reference -> channel, the model direction; `fwd`: the inverse polynomial, refitted on a
  grid by `refit_fwd` / `to_block`). An order-1 map is written as an `affine` block (channel -> reference matrix), a
  higher one as a `poly` block with `roundtrip_px`; `from_block` reads identity / affine / poly (a grid mapping has no
  polynomial form). `set_fit('all' | 'affine' | 'shift')` masks the trainable coefficients.
- **Estimation** (`torch_psf.estimation.multichannel_beads`): `build_multichannel_beads(stacks[c][s], maps, patch,
  masks, ...)` -> `MultiChannelBeads` (detections of every channel merged in reference coordinates, clusters by box
  overlap in any channel, one `BeadClusters` per channel with the clusters usable there, global bead / cluster ids;
  `subset(ids)`, `channel(c)`); `fit_multichannel_beads(data, focus_um, psfs, maps, iters=300, lr=3e-3,
  fit=('zcoef',), share=(), fields=None, fit_mapping=True, map_mode='all', train=None, focus_offset='bead',
  channel_focus=True, intensity='free', ...)`: shared bead positions (reference channel) and focus offsets, per channel
  photons, backgrounds, PSF (Zernikes, NA / apod per channel or `share`d), field polynomial (channel coordinates) and
  focus offset (channel 0 fixed), the mappings of channels 1.. fitted jointly, a shared drift; Poisson likelihood
  summed over channels, Adam; held-out clusters refitted and scored per channel. `multichannel_nll(data, focus_um,
  psfs, maps, channel_focus=, fit_channel_focus=False)` scores fixed models. Helpers: `channel_stop_mask` (field stop
  robust to a background gradient), `detect_channels`, `register_channels` (flip / shift / affine from the bead
  detections), `simulate_multichannel_beads`.
- **Estimator**: `MultiChannelBeadEstimator(regions, names=None, start=None, optics=None, camera=None, ...,
  mapping_order=1, fit_mapping=True, joint=True, share=(), channel_focus=True, field=None, holdout=None, ...)`;
  `estimate(stacks, dz_um)` -> ChannelsPSF (VectorialPSF / FieldPSF per channel with `focus_um` = the channel focus
  offset, the fitted mappings, intensity ratios); `model_of(est.result | est.result_field)`, `channel_entries()`,
  `save_zip(path, camera=None, psf=None)`: one psf.zip with every channel's model, mapping, crop, flip, intensity
  ratio and field block (format unchanged). `joint=False` fits every channel on its own and refits the mapping to the
  fitted bead positions.

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

`solve_linear(psf, data, mask, group, rn2, ngroups=2, iters=3, I_min=0.0)` (variable projection, used by
`estimation.fit_tracks`): the per-frame intensities I [B, T] and per-group backgrounds bg [B, ngroups] of
`mu[b, t] = I[b, t] psf[b] + bg[b, group[b, t]]` by `iters` rounds of weighted linear least squares (weights
`1 / (mu + rn2)`), I clamped >= I_min; masked frames give I = 0.

## Splines

`torch_psf.to_spline(model, z_range, *, roisize, kernel, value, oversample, dz_um, zpad, xy_limit)`
samples an optical model. The method form is `model.to_spline(roisize, z_range, **grid)`.
Use `kernel='catmull'` for interpolating samples or `kernel='bspline'` for coefficients with a mirror-boundary
prefilter. With `integrated=False` each sample is a rendered pixel. With `integrated=True` the builder samples
point density and the evaluator integrates the spline over the pixel. Without kernel / value, an optical
model (or a wrapper of one) gets the integrated cubic B-spline at two knots per pixel
(`DEFAULT_SPLINE = ('bspline', 'density', 2)`: in the study/psf_unify benchmark it is faster and more accurate than
integrated Catmull-Rom at two knots per pixel); a model that renders only pixel values (the Gaussians) gets
Catmull-Rom on pixel values at two knots per pixel (`DEFAULT_PIXEL_SPLINE`). Calibration builds pass kernel, value
and the grid explicitly; the bead and in-situ estimators' knot routes and the viewer keep Catmull-Rom pixel-value splines. The 4Pi and field builders retain their
integration-mode-dependent defaults.

```python
from torch_psf import PSF, to_spline

angles = [PSF.load(calibration, angle=k) for k in (0, 1)]
spline = to_spline(angles, (0, 0.4), roisize=41, kernel='bspline', integrated=True,
                   oversample=4, dz_um=0.01, zpad=6, xy_limit=2.0)
image, dx, dy, dz = spline.for_angle(1).render(pos, grad=True)
```

- `SplinePSF.knots` is one tensor of shape `[T, nz, nv, nu]`. `axes` names the dimensions of the integer
  `spline_index`, whose entries select volumes. Knot spacings are `1 / S` pixels laterally and `hz` micrometres axially.
  The first knot is at `(u0, u0, z0)`.
- `render(pos, grad=True)` returns the value and the x, y and z derivatives evaluated at the clamped position.
  `render(pos, weights={axis: weights, ...})` accepts shared or per-spot axis weights and differentiates through
  the weights and the knot bank. `evaluate_splines(pos, indices)` evaluates individual (position, spline) pairs.
- Build with `components='dipole'` for `dipole_basis(pos)` and `render(pos, dipM=M)`. The six images are ordered
  xx, yy, zz, xy, xz, yz and cross images include both symmetric terms.
- The legacy file blocks store Catmull-Rom pixel samples and B-spline density coefficients.
  Other kernel/value combinations and general weighted layouts can be used in memory.
- A sequence of optical models supplies the angle axis. `for_angle(None)` selects the calibration average.
  `for_angle(k)` and `resized(roisize)` share the knot tensor. `to(device, dtype)` moves the model in place.
- `backend='sep'` contracts along z, x and y. `backend='conv'` uses grouped convolution. Explicit kernel/value
  builds default to `sep`. Historical pixel-sample builds and files retain convolution for numerical compatibility.
- A `FourPiSplinePSF` is a multichannel view over `.spline`, a single spline with phase and channel axes
  and an optional dipole axis. The phase weights are `(1, cos(phi), -sin(phi))`.
- `SplinePSF.from_zstack(...)` wraps measured planes. For optical stacks, `ZMode(resized_optics, 'stage', height=h, scale=s)` supplies the
  emitter height and focus for each plane. `sample(model, differentiable=True)` retains the optical graph.
  `vjp_vectorial(...)` pulls a coefficient gradient through the prefilter and the optical sampler in z chunks.

## PSF estimation (`torch_psf.estimation`; used by `ZStackEstimator`, `InSituEstimator`)

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

`heldout_tracks(tracks, psfs, readnoise_e=1.0, zmax_um=0.2, groups=None, ngroups=1, refit_psfs=None,
z_starts_um=(0.02, 0.08, 0.15), iters=300, lr=0.03, batch=3000)`: held-out check of a frozen PSF on track samples
(`spots.Tracks`, the data of `fit_tracks`): per sample x, y and z (sigmoid on [0, zmax_um]) refitted by Adam from each
start z, I per frame and bg per (sample, group) by `solve_linear` at every step, the start with the lowest NLL kept.
psfs: one model or one per group (anything with `render(pos)`, e.g. a VectorialPSF at its focus_um with dipM / blur
set); refit_psfs: faster models for the refit iterations (e.g. their `to_spline()`), the NLL then from psfs. NLL:
Gaussian approximation of Poisson + read noise (as `fit_tracks`). Returns nll [B], chi2 [B], xy, z, I, bg, nll_per_px,
chi2_per_px, n (numpy).

`paired_nll(nll_a, nll_b)`: paired per-sample difference b - a on the same samples: mean, se (ddof 1), z = mean / se,
frac_better.

`z_scale(psf_new, psf_ref, z_range=None, n=33, photons=1e5, bg=1)`: noise-free emitters at x = y = 0 and n heights
`linspace(z_range)` rendered with psf_new, localized with psf_ref; least-squares line through the inner 80 % (indices
`int(0.1 n) .. n - int(0.1 n)`): returns (slope d z_ref / d z_new, offset at the centre of that range, um).

## Field-dependent PSFs (`torch_psf.field`, single channel; `docs/notes/field_psf_lib.md`)

```
FieldZernikes(noll, order=1, norm=(cx, cy, sx, sy), ref=None, prior_sd=0.05, clamp=True)   torch.nn.Module
    .from_positions(noll, order, xy, margin=0)        norm = the bounding box of xy (the fitted support)
    forward(xy) / relative(xy) / offsets(xy, zidx)    [N, k] / [N, k] / [N, nz] waves (differentiable)
    penalty(), to_block(waves_at_ref), from_block(d) -> (field, const)
FieldPSF(base: VectorialPSF, field)                   PSF; render(pos, grad=False, xy=None), at(x, y),
    zernikes_at(xy), fit(rois, corners=, xy=), bind(xy), to_tiled(tile, extent, roisize, z_range, **grid)
TiledSplinePSF(field_psf, tile, extent, roisize, z_range, lazy=True, blend="nearest", **grid)
    render(pos, grad, xy), tile_index(xy), spline(t), fit(rois, corners=, xy=)
```
- W_j(x, y) = W_j(ref) + sum_{1 <= i + k <= o_j} c_{j,ik} (u^i v^k - u_ref^i v_ref^k), u = clip((x - cx) / sx, -1, 1),
  v = clip((y - cy) / sy, -1, 1): the PolyMap features of psf.zip (i outer, k inner), x = column, y = row in camera
  (channel crop) px. The base VectorialPSF is the model at the reference point (default (cx, cy)). Per-mode order
  o_j in {0, 1, 2}; coef [k, nf - 1] masked by it. The clip (per coordinate, the feats_norm box) is the
  no-extrapolation rule; `psfzip.FieldZernikePoly` applies the same clip.
- Ridge prior `penalty() = 0.5 sum (c / prior_sd)^2` (NLL units, c in waves at the box edge).
- psf.zip: `io.save_zip(path, field_psf)` writes the base model plus the channel `field` block
  (`zernike_poly`, unchanged format; per-mode orders are the zero pattern of `coef`); a TiledSplinePSF is written as
  its reference spline with the vectorial `source` and the block, its tile grid in `provenance['field_tiles']`.
  `PSF.load` / `io.load(path, channel=)` / `PSFZip.read(path).psf` (single channel) / `Channel.psf` return
  FieldPSF (vectorial channel) or TiledSplinePSF (spline channel with a vectorial source).
- Localization: `FieldPSF.fit` evaluates each ROI at its own position (corners + (P - 1) / 2, or `xy`);
  `TiledSplinePSF.fit` samples the spline at the fixed field coordinates of each ROI. `PSF.localize` of
  either works through `fit`.
- Estimation: `estimation.fit_field_knots(spline, rois, xy, vectorial, field, tile=256, extent=None, iters=60,
  lr=2e-3, lr_field=None, zstarts=5, var=None, fit_base=True, min_spots=30)`: per tile (tiles with fewer than
  min_spots spots merged into the nearest one) a knot build at base + field(c_t), c_t = the tile's mean spot
  position, localization, knot gradient and VJP g_t; d/d base = sum g_t, d/d coef = sum g_t d field(c_t) / d coef
  + the prior; Adam. `fit_dense_beads(..., fit=(..., 'field'), field=, xy=)`, `fit_phase_diversity(...,
  field=, xy=)`: every bead / patch at its own polynomial Zernikes. Estimators: `field=` (dict: modes, order,
  prior_sd, margin, iters, lr, tile, score_tile; `field.field_from_spec`) in InSituEstimator, DenseBeadEstimator,
  PhaseDiversityEstimator: a field fit started from the global fit, scored on the held-out data against it
  (`report['field']`), returned when it wins.
- Held-out helpers: `heldout_nll / compare_heldout / spot_deviance(..., zstarts=5, xy=None)`; xy = field positions
  of the ROIs for field-dependent models. `compare_heldout` also gives `dnll_per_spot_vs_first`, `dnll_se`,
  `frac_better`.

Detection for engineered PSFs (`torch_psf.matched_filter`): `MatchedFilterDetector(psf, z_um=None, dz_um=0.2,
size=None, nms=21)` (zero-mean unit-norm templates of the model at heights z, FFT correlation per frame, each template
response / its 1.4826 MAD, maximum over templates, max-pool peaks above `threshold`; returns detect_spots' dict plus
`z`); `extract_spots(..., detector=)`, `InSituEstimator(detector=, zstarts=)`.

## Simulation

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

### First-order nodal aberration theory for in-situ calibration

`InSituEstimator(method="nat", field={"backend": "bspline", "tile": 64})`
fits a global starting pupil using the existing selection and rejection workflow,
then alternates emitter localization and a shared first-order NAT field update.
The held-out checkerboard decides whether to retain the field. The estimator's
`iters` controls the global prefit; `field["iters"]` controls NAT outer updates.

For already cropped data use `estimation.fit_nat(vectorial, rois, xy, norm=(cx,cy,sx,sy))`.
ROIs are in photons; xy are their fixed camera centers in pixels. Use the actual
FOV center and half-widths when comparing coefficients across datasets. The
13 parameters use Vectorfit 0.3.4's first-order field expansion with Zernike
outputs in torch-psf RMS waves. For coefficients copied from Supplement Eqs. S43/S58, use
`nat_polynomial(gamma_in_waves, convention="paper")` to convert the pupil
normalization and the published quadratic astigmatism scale. The fit returns a
serializable continuous FieldPSF, local emitter estimates, and the update history.
`nat_field(vectorial, gamma, norm)` restores the field with the same starting optics.
Other input modes remain fixed. Initialization uses the supplied pupil. Pass
multiple 13-vectors in gamma0 for independent starts. The lowest final NLL wins.
Use max_rois to balance a smaller sample across 10 by 10 field bins, and fraction
to resample the global update (the supplement used 0.2).

The direct backend evaluates the physical pupil at each emitter. The bspline
backend builds one pixel-integrated cubic B-spline per occupied tile, sums the
Poisson gradient on its knots, and propagates it through the spline prefilter
and vectorial model to the 13 global parameters. Tile size controls field
approximation; oversample and dz_um control interpolation at each tile.
Validate the selected grid against direct held-out predictions. Vectorfit and
torch-psf use different optical renderers and numerical optimizers.

For a physical-objective refinement, initialize the direct path from the cached fit
and keep the same observations, camera variance, optics, and field normalization.

```python
from torch_psf.estimation import fit_nat

cached = fit_nat(base, rois, xy, norm=norm, var=var, backend="bspline")
physical = fit_nat(base, rois, xy, norm=norm, var=var, backend="direct",
                   gamma0=cached["report"]["gamma"])
```

The second stage evaluates each emitter at its camera position and refits its
local parameters. Its iteration budget controls the accuracy of the refinement.

For a nested comparison on identical data, set `field_varying=False` in
`fit_nat`. This estimates the six constant Zernike coefficients while fixing
the seven coefficients responsible for field dependence to zero.
