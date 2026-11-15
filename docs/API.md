# torch-psf: stable API

The functions below are the reference implementation that other code (the browser app in `web/`, downstream
pipelines) ports or calls. Their signatures, defaults and math are stable: **after the first public release any change
to their math or defaults is announced** (release notes / the commit message, and to the maintainers of the ports)
before it is merged. Units: px for lateral positions (x = column, y = row, pixel centres at integers; inside a ROI
relative to its centre pixel), um for z, focus, wavelengths and pixel sizes, photons (e-) for intensities, photons / px
for backgrounds. Defaults: `device=None` = cuda if available, else cpu; float32.

## Changes

### 2026-11-15: multi-channel PSFs (additive; zero-offset 4Pi renders bit-identical)

`MultiChannelPSF` (new, `torch_psf.multichannel`) is the base class of the multi-channel models: `ChannelsPSF` (new;
independent channels) and `FourPiPSF` / `FourPiSplinePSF` (now its subclasses). The 4Pi models gain `fit`,
`fit_theta`, `fisher`, `crlb`, `covariance`, `mu_and_jacobian` (before: `NotImplementedError`) and per-channel
`channel_offsets`; `FourPiSplinePSF.render(..., knots=)`; `PSFZip.multichannel()`. Unchanged: `FourPiPSF.render(pos,
grad, phi, M)`, `forward`, `expected(theta [N, 5], phi)`, `state` / `from_state` (key `channel_offsets` only when
nonzero), `to_spline`, the psf.zip layout. See [Multi-channel PSFs](#multi-channel-psfs-torch_psfmultichannel).

### 2026-11-15: module renames (breaking for submodule imports; no aliases; no behaviour change)

The top-level names are unchanged (`from torch_psf import Optics, Camera, photon_transfer, ...`). Only imports from the
submodules below change; the old module paths are removed (no re-exports).

| old module path | new module path |
|---|---|
| `torch_psf.insitu` (package: `beads`, `knots`, `tracks`, `heldout`, `zscale`) | `torch_psf.estimation` (same submodules and names: `torch_psf.estimation.fit_beads`, `torch_psf.estimation.knots.knot_loss`, ...) |
| `torch_psf.config.Optics` | `torch_psf.optics.Optics` |
| `torch_psf.config.default_device`, `resolve_device` | `torch_psf.optics.default_device`, `resolve_device` |
| `torch_psf.config.Camera` | `torch_psf.camera.Camera` |
| `torch_psf.calibrate.photon_transfer` | `torch_psf.camera.photon_transfer` |
| `torch_psf.varpro.solve_linear` | `torch_psf.fit.solve_linear` |
| module layers: config, zernike, fit, varpro -> detect, movie, spots, dataset, calibrate -> psf -> vectorial, spline, gaussian -> io, insitu, simulate -> estimators -> viewer, `__main__` | optics, zernike, fit -> detect, movie, spots, dataset, camera -> psf -> vectorial, spline, gaussian -> io, estimation, simulate -> estimators -> viewer, `__main__` |

### 2026-11-14: class design (breaking; no aliases)

| old | new |
|---|---|
| `ConvSplinePSF` | `CubicSplinePSF` (same knot layout and math) |
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

### 2026-11-14: Dataset (breaking)

| old | new |
|---|---|
| localization table (numpy structured array `table.LOC_DTYPE`: frame, x, y, z, photons, bg, crlb_x / y / z / photons / bg, loglik, chi2, at_limit) | `torch_psf.Dataset` (dict of tensors): `pos` [n, D] px (x, y, z; **z in px** of the lateral pixel size, always), `crlb` [n, D] (same units), `frame`, `n_frames` (1), `id`, `photons`, `bg`, `crlb_photons`, `crlb_bg`, `loglik`, `chi2`, `at_limit`; meta `pixel_nm`, `image_size`, `n_frames`, `method` |
| `PSF.fit` / `PSF.localize` return the table | they return a Dataset (`fit(..., pixel_um=None)`, `localize(..., pixel_um=None)` override the pixel size used for z) |
| `table.to_csv` | `Dataset.save('x.csv')` (ThunderSTORM) or `.hdf5` (docs/DATASET_FORMAT.md); `save_picasso` |
| `CubicSplinePSF` | has `pixel_um` (set by `VectorialPSF.to_spline`, stored in the spline file) |
| (follow-up) z unit | one unit per Dataset: z in px of the lateral pixel size, always. Every model has `pixel_um` (constructor argument of CubicSplinePSF, `from_zstack`, GaussianPSF, AstigmaticGaussianPSF; VectorialPSF has it anyway); fitting / localizing a 3D model without one raises ValueError (pass `pixel_um=` to fit / localize or set it on the model) |

### 2026-11-14: follow-ups (defaults changed: marked **default**)

| old | new |
|---|---|
| `VectorialPSF.to(device)` returned a resized copy; `CubicSplinePSF.to` / the Gaussians returned new objects | `.to(device)` has torch.nn.Module semantics for every PSF: in place, returns self (VectorialPSF: also dtype, `.cuda()`, `.cpu()`; the optional blur / dipM / dipw / diat tensors move along). A copy of another ROI size: `resized()` |
| `fit_beads(drift=True)`: linear drift per stack (px per plane, lr 1e-4) | **default** `drift_order=3`: cubic polynomial in t = plane index / half the stack (t in [-1, 1]), no constant term, lr 2e-3; `drift_order=1` = linear. Returned `drift` is [nstacks, order, 2] px (coefficients). On six real bead stacks (19 beads, 151 planes, one stack held out) the cubic drift gave a lower training NLL (-82.40 vs -82.27 per px) and a lower held-out NLL (-52.58 vs -52.56 per px) than the linear one |
| held-out beads refitted: position / photons / bg | also the drift of their stack |
| bead size and height independent | with `bead_diameter_um` and `fit_height` the sphere diameter is tied to 2 x the fitted centre height (a bead lies on the coverslip); without fit_height the diameter is fixed and the starting / fixed height is `emitter_z_um`; the result reports `bead_diameter_um` |
| `ZStackEstimator(...)` | adds `emitter_z_um` (starting bead centre height; **default** bead_diameter_um / 2, else 0), `fit_na`, `fit_n_med`, `drift_order` (**default** 3, passed to fit_beads; drift is on by default as before) |
| `Dataset.estimate_drift` stored `meta['drift_info']` = dict(method, n, version) | stores `meta['drift']` = the run record: method, entry point (`dme.dme_prior.dme_prior_estimate`), version, package path, parameters (n_frames, dims, n_rows, time, row rule), `dme_info` (the choices DME made), result per axis (ptp_px, rms_px); `return_info=True` also returns it. DME is found through `DME_PATH` (a checkout) if set, else the installed package. `set_drift(info=)` stores into `meta['drift']` |

### 2026-11-14: read noise in the in-situ likelihoods (no default number changes)

| old | new |
|---|---|
| `InSituEstimator` fitted, scored and rejected spots with the plain Poisson likelihood: the camera read noise (`Spots.var` = `Camera.variance`) was ignored in the Zernike / pupil fits, the deviance rejection, the free-knot refinement, the held-out NLL and the held-out chi2 fit | all of them use the read-noise variance (scalar readnoise^2 / excess_noise^2 in photons^2, or per-pixel sCMOS maps cropped per ROI) when a camera with read noise is given; the d + var ~ Poisson(mu + var) likelihood of `fit.poisson_nll` / `lm_poisson`. On `simulate_reference_movie` (gain 0.5, read noise 1.5 e-) the held-out chi2 of the true PSF goes from 1.11 to 1.02 and the median deviance from 1.14 to 1.05. Report numbers that change: `heldout_chi2_median`, `deviance` (median / threshold / rejected fraction), absolute held-out NLL values (the likelihood includes var; differences between models are comparable as before). No option or default value changed |
| `fit_zernike_knots`, `fit_pupil_knots`, `fit_free_knots`, `vectorial_gradient` | take `var=None` (scalar or [N, P, P], aligned with the ROIs) |
| `PSF.fit(camera=...)` / `PSF.localize(camera=...)` | unchanged: they already used `Camera.variance` (localize through `Spots.var`); chi2 ~1.00-1.02 on the reference movie |

The bead fit (`fit_beads` / `ZStackEstimator`) still uses the plain Poisson likelihood (bright beads: read noise negligible).

### 2026-11-14: real-data follow-ups (defaults changed: marked **default**)

| old | new |
|---|---|
| `TiffMovie`: frames = TIFF pages; an ImageJ TIFF > 4 GB (one IFD, frames contiguous) read as 1 frame; part files found by replacing the first '.tif' anywhere in the path | ImageJ / single-IFD files: frame count and memory map from the first image series; multi-page files (Andor SOLIS): one frame per page, read through one strided memory map when the page data are evenly spaced uncompressed blocks (else page by page); parts = `<base>_X1<ext>`, ... next to the file. Same pixels, faster reads |
| `extract_spots(...)` read chunks from the start of the range | adds `sample='sequential'` (as before) or `'spread'` (bit-reversed chunk order: with `max_spots` the chunks read are spread over the whole range; spots returned in frame order); `Spots.frame_ranges` = the chunks read; an empty result is an empty Spots |
| `InSituEstimator`: spots from the first 500-frame chunks of the range until 2 (nspots + nheldout) | **default** `sample='spread'`, `chunk=50`: spots from chunks spread over the whole range. report: `frames_requested`, `frames_read`, `frame_span`, `n_chunks`, `spots_per_frame`; a warning when fewer training spots than nspots were found |
| overlap rejection above 1 + deviance_k sqrt(2 / P^2) (exact Poisson only); `report['deviance_threshold']` | **default** `deviance_rule='median'`: rejected above median + deviance_k x max(1.4826 MAD, median sqrt(2 / P^2)) (robust to a gain error or EMCCD statistics, which scale all deviances); `'poisson'` = the earlier rule. `report['deviance']` = rule, median, scale, threshold, rejected_fraction, quantiles; warnings when more than `warn_rejected` (0.25) are rejected or the median deviance is > 1.3 |
| `Camera(type='emccd')`: a label; the excess noise had to be folded into the gain by hand | `Camera.excess_noise` (F^2; default 1 = no change): `to_photons` divides by it and `variance` by its square, so fits / CRLB / deviances use Poisson-equivalent photons (= photoelectrons / F^2; EMCCD: true gain with excess_noise=2, or gain / 2 with 1) |
| (new) | `torch_psf.calibrate.photon_transfer(movie, frames, nblocks, block, mask_radius, threshold, offset_adu, excess_noise)` (also `tp.photon_transfer`) -> (Camera, report): offset and gain from background pixels (spots masked by detection, 4-sigma clip, consecutive-frame differences) |
| no gain check in the reports | `chi2_median` (median over spots of the mean (d - mu)^2 / (mu + var) per pixel; ~1 if the camera model is right) in `Dataset.meta` of `PSF.fit` / `PSF.localize`, `fit_beads` / `ZStackEstimator.report` (`chi2_median`, `heldout_chi2_median`), `InSituEstimator.report['heldout_chi2_median']` (held-out spots fitted with the returned PSF; warning outside 0.7 .. 1.4) |
| `InSituEstimator` start from optics: Noll 5..21, npupil 64 fixed | `zernike_indices`, `npupil`, `oversample_pupil` (as ZStackEstimator) |
| focus / wavelength by hand | `InSituEstimator(scan_focus_um=[...], scan_wavelength_um=[...], scan_iters=None)`: each combination fitted (scan_iters, default iters / 2), the lowest held-out NLL is used; `report['scan']`; warning when the best value is at the edge |
| `PSF.localize` passed only detection options to fit; progress per 5000-frame block | `iters`, `zstarts`, `batch` passed to `fit`; progress after every chunk |
| `CubicSplinePSF.to_spline(other roisize)` raised | `CubicSplinePSF.resized(roisize)` (the same knots, shared; a larger ROI than the knots cover lowers xy_limit) and `to_spline(roisize)` = `resized` |
| `est.data['spline']` = the spline of the vectorial fit, also when the refined spline was returned | **changed**: `data['spline']` = the spline of the returned PSF (refined if kept); `data['vectorial_spline']` = the vectorial fit's spline; `data['psf']` = the returned PSF; `data['vectorial']` as before |
| `psf.save` stored the provenance passed to it | `PSF.provenance` (dict): the estimators set it to `{estimator, settings, report}` (ZStackEstimator: `{estimator, model, dz_um, report}`); `save(provenance=None)` writes it, `PSF.load` reads it back. The command line no longer passes `provenance=dict(report=...)` itself (the file still has `provenance.report`) |
| `Dataset.frc`: full complex128 FFTs of the whole field | float32 histograms and real FFTs (same result to 1e-8; ~4x less memory); `region=(x0, y0, x1, y1)` px and `mask` (rows) options |
| (new) | `Dataset.load_smlm(path, pixel_nm, frame_base=1, offset_px=0.5)` and `Dataset.load('x.smlm', pixel_nm=...)`: ShareLoc .smlm (binary table; ThunderSTORM conventions) |
| `detect_spots(noise='mad')`: one noise level per chunk | adds `noise='mad_frame'` (per-frame MAD; for movies whose density / background changes within a chunk) |

Not done: partial ROIs at the image border (detections closer than (P - 1) / 2 + 2 px to the edge are still dropped), an image-plane blur term in the in-situ fit.

### 2026-11-14: imports (no behaviour change; files identical)

| old | new |
|---|---|
| `PSF.load` / `torch_psf.io` imported the model classes | registry: each model class sets `type_name` (its psf.json `type`: `vectorial`, `spline`, `gaussian`, `astig_gaussian`) and is registered by `PSF.__init_subclass__`; `PSF.load` looks the type up (`PSF._registry`). Each class writes / reads its own parameters (`_file_dict`, `_from_file`); `psf.save` writes the common part. `torch_psf/__init__` imports all models, so they are registered whenever torch_psf (or any submodule) is imported. A new model class: set `type_name` and implement the two methods |
| `torch_psf.io.save` / `load` / `load_spline_source` / `load_camera` held the implementation | thin wrappers of `psf.save`, `PSF.load`, `CubicSplinePSF.load_source`, `Camera.from_json` (new) |
| function-level imports | all imports at module top, in layers (each importing only from earlier ones): config, zernike, fit, varpro -> detect, movie, spots, dataset -> psf -> spline -> vectorial, gaussian -> io, insitu, simulate -> estimators -> viewer, `__main__`. Still imported where used: the optional dme (`Dataset.estimate_drift`) and scikit-image (viewer meshes) |

## Dataset (`torch_psf.dataset.Dataset`; format: docs/DATASET_FORMAT.md)

`Dataset(fields, meta=None, drift=None, drift_applied=False)`, `Dataset.from_arrays(pos, frame, crlb, photons, bg,
n_frames, meta, **fields)`; `ds['pos']` / `ds.pos`, `len(ds)`, `ds[mask | slice | indices | int]` (all fields),
`ds['name'] = tensor`, `add_fields(**)`, `keys()`, `copy()`, `to(device)`, `numpy()`; `n_dims`, `end_frame`,
`center_frame`, `pixel_nm`, `pos_nm`, `num_frames`, `indices_per_frame()`; filters returning a new object and in-place
`_` variants: `filter(mask)`, `filter_finite()`, `filter_photons(min, max)`, `filter_crlb(max_px, max_nm)`,
`filter_frames(start, stop)`, `crop(x0, y0, x1, y1, shift)`, `crop_from_edge(dist)`; `Dataset.concatenate(items,
set_group)`, `+`; drift: `set_drift`, `drift_at`, `apply_drift`, `undo_drift`, `estimate_drift` (DME v2, optional
dependency); `render(pixel_nm, region, sigma_nm, weights)` (2D histogram), `frc(block=300, pixel_nm=5)` (even / odd
blocks of frames, 5-ring smoothing, 1/7 threshold); files: `save` / `load` (.hdf5 / .h5 this format, .csv
ThunderSTORM), `save_picasso` / `load_picasso`, `save_csv` / `load_csv`.

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
│                focus_um, z_range)  (also a torch.nn.Module: forward(pos, focus, ...), set_zernike, resized,
│                from_optics, optics, to_spline(..., pos_fn))
├── CubicSplinePSF(roisize, z_range, oversample, dz_um, zpad, xy_limit)  knots [nz, nv, nu]; render(..., knots=);
│                knots_from_model, knots_from_vectorial, vjp_vectorial, from_zstack
└── MultiChannelPSF: ChannelsPSF, FourPiPSF, FourPiSplinePSF   images [N, C, P, P] (see Multi-channel PSFs)

Estimator (torch_psf.estimators)             report (dict), history (list), data (dict), progress / stop hooks
├── ZStackEstimator(model='vectorial' | 'spline' | 'astig_gauss', start=None, optics=None, camera=None, roisize=21,
│                   nbeads=8, sign=1.0, iters=300, holdout=None, drift=True, regularize=3000.0, ...)
│     .estimate(stacks, dz_um) -> VectorialPSF | CubicSplinePSF | AstigmaticGaussianPSF
└── InSituEstimator(start=None, optics=None, camera=None, focus_um=None, roisize=15, z_range=(0, 0.8), nspots=8000,
                    nheldout=4000, method='zernike', iters=80, reject_overlap=True, deviance_k=3.0, refine=False, ...)
      .estimate(movie, frames=None) -> VectorialPSF (or the refined CubicSplinePSF if it wins on held-out spots)
```
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
      param_names, channel_offsets [C, 2],   render(pos, grad=False, offsets=None, **extra), expected(theta, offsets)
      intensity_ratio [C],                   mu_and_jacobian(theta, offsets) -> (mu, J [N, 4 + C + k, C, P, P])
      set_channel_offsets(v, trainable),     fisher / covariance / crlb(theta, var, offsets, fixed)
      channel_positions(pos, offsets)        fit_theta(rois [N, C, P, P], theta0, var, offsets, fixed, ...) -> (theta, nll)
                                             fit(rois, ..., offsets=None, fixed=()) -> Dataset
                                             localize(movie): NotImplementedError (needs the channel mapping)
    ├── ChannelsPSF(models, channel_offsets=None, intensity_ratio=None, names=None)   independent channels
    ├── FourPiPSF (also a torch.nn.Module)                   extra_params ('phi',)
    └── FourPiSplinePSF                                      extra_params ('phi',); render(..., knots=)
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
  offsets zero the earlier single evaluation runs (renders bit-identical). `to_spline` bakes the offsets into the
  tables (the spline's own offsets are zero); `FourPiSplinePSF.channel_offsets` shift each channel's table
  evaluation instead (stored in its torch-psf json, not in psf.zip, which refuses them). The earlier calls keep their
  meaning: `render(pos, grad, phi, M)`, `forward(pos, M, phi_extra, intensity, bg)`, `expected(theta [N, 5], phi)`
  (one background).
- **psf.zip**: `PSFZip.read(path).multichannel()` = the coupled model, or a `ChannelsPSF` of the channel models with
  their `intensity_ratio` (offsets zero; the channel `mapping` gives per-spot offsets). The format is unchanged.

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

## The convolutional cubic spline (`torch_psf.spline.CubicSplinePSF`)

`CubicSplinePSF(roisize, z_range, oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0, device=None)`: knots
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

`z_scale(psf_new, psf_ref, z_range=None, n=33, photons=1e5, bg=1)`: noise-free emitters at x = y = 0 and n heights
`linspace(z_range)` rendered with psf_new, localized with psf_ref; least-squares line through the inner 80 % (indices
`int(0.1 n) .. n - int(0.1 n)`): returns (slope d z_ref / d z_new, offset at the centre of that range, um).

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
