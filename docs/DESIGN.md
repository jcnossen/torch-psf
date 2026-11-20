# torch-psf: library design

The stable API (signatures, defaults, math) is documented in [API.md](API.md), the PSF file format in
[PSF_FORMAT.md](PSF_FORMAT.md), the localization format in [DATASET_FORMAT.md](DATASET_FORMAT.md) and the viewer API
in [VIEWER_API.md](VIEWER_API.md). This page gives the overall structure.

## Goal

`torch_psf` (distribution `torch-psf`): PSF models, single-emitter fitting and PSF estimation (bead z-stacks and in
situ) for SMLM, in PyTorch, batched on the GPU. SIMCODE imports its shared parts from here (`simcode.psf`) and keeps
only what is SIMCODE-specific. The browser app in `web/` is a WebGPU port of the parts a user needs to estimate a PSF
and localize a movie; the Python library is its reference.

## Conventions (the whole package)

- Image coordinates: x = column, y = row, in px, pixel centres at integers. ROI-relative positions are offsets from the
  ROI centre pixel. Emitter z = height above the coverslip in um. `focus_um` = focal-plane position relative to the
  coverslip. Lengths carry their unit in the name (`pixel_um`, `wavelength_um`, `dz_um`); z ranges are
  `z_range=(lo, hi)`. Intensities in photons (e-), backgrounds in photons / px.
- A PSF model is a class; detection, spot extraction and the estimation steps are functions; the estimators are
  classes with a report. There are no compatibility aliases.
- Every single-channel model has the interface of `PSF` (`torch_psf.psf`): `render(pos, grad=False)` with pos [N, 3] =
  (x px, y px, z um) -> the normalized PSF [N, P, P] (grad: also d/dx, d/dy per px and d/dz per um); `roisize`,
  `z_range` (None for 2D models), `xy_limit`; `expected(theta)`, `fisher` / `crlb`, `fit` (ROIs -> Dataset),
  `localize` (movie -> Dataset), `save` / `PSF.load`, `to_spline`.
- Fit parameters of an emitter: theta = (x, y, z, photons, bg); z is held fixed for 2D models.
- Device default: cuda if available, else cpu; float32.

## Modules

```
torch_psf/
  optics.py        Optics (pixel_um, NA, wavelength_um, n_imm, n_med), default_device, resolve_device
  camera.py        Camera (gain, offset, read noise: scalars or per-pixel maps; excess noise), photon_transfer
  movie.py         TiffMovie, ArrayMovie, open_movie (reader interface: n, shape, read(f0, f1, y0, x0, S))
  zernike.py       noll_to_nm, zernike, NOLL_NAMES
  fit.py           poisson_nll, lm_poisson, localize, fisher, crlb, solve_linear (variable projection)
  detect.py        detect_spots (band-pass, local maxima, threshold x noise, isolation, border)
  spots.py         Spots, extract_spots (streams over a movie), extract_rois, Tracks
  dataset.py       Dataset (the localizations; csv / hdf5 / Picasso / .smlm, FRC) and Drift (per-movie drift estimate)
  psf.py           PSF (base class; type registry used by PSF.load)
  vectorial.py     VectorialPSF (+ gauss_blur_os, dipole_tensor, photosel_tensor)
  spline.py        SplinePSF (+ geometry_matrix, emitter_height_pos_fn, stage_scan_pos_fn)
  gaussian.py      GaussianPSF (2D), AstigmaticGaussianPSF
  multichannel.py  MultiChannelPSF (base of the [N, C, P, P] models), ChannelsPSF
  fourpi.py        FourPiPSF, FourPiSplinePSF
  psfzip.py        psf.zip read / write (save_zip, PSFZip, convert_calibration)
  psfreport.py     fit_report.pdf of a psf.zip
  io.py            module-level save / load / load_camera / load_spline_source
  estimation/      beads (fit_beads), knots (fit_zernike_knots, fit_pupil_knots, fit_free_knots), tracks (fit_tracks),
                   heldout (split_checkerboard, compare_heldout), zscale
  simulate.py      simulated emitters, bead stacks and blinking movies with known truth
  estimators.py    ZStackEstimator, InSituEstimator
  viewer.py        the PSF viewer server (front end bundled from web/)
  __main__.py      command line: beads, insitu, localize, view
```

Layers (each module imports only from earlier layers): optics, zernike, fit -> detect, movie, spots, dataset, camera ->
psf -> multichannel, vectorial, spline, gaussian -> fourpi -> psfzip, psfreport, io, estimation, simulate ->
estimators -> viewer, `__main__`. Model classes register themselves by their file type name
(`PSF.__init_subclass__`); `torch_psf/__init__` imports all of them.

## PSF models

- `VectorialPSF` (also a `torch.nn.Module`): pupil-function model with dipole emission, Fresnel coefficients and SAF,
  the index mismatch, Zernike aberrations (`zcoef`, Noll), optional free pupil maps (`pupil_phase`, `pupil_amp`),
  image-plane blur, diattenuation and the emission-dipole tensor, chirp-z / matrix DFT, oversampling and pixel
  integration. Derivatives by forward-mode autodiff (`torch.func.jvp`).
- `SplinePSF` stores a knot bank [T, nz, nv, nu] and an integer table map. The evaluator gathers
  all requested (spot, table) pairs and applies separable interpolation or grouped convolution.
  Pixel tables sample rendered pixels and density tables integrate the spline across each pixel.
  The builder `to_spline` samples optical models, and `from_zstack` wraps measured planes.
  The in-situ estimators use `vjp_vectorial` to propagate knot gradients to the optical model.
- `GaussianPSF` (2D) and `AstigmaticGaussianPSF` (3D calibration curves).
- `MultiChannelPSF`: an emitter seen in C channels ([N, C, P, P]); theta = (x, y, z, I, bg_0 .. bg_{C-1}, extra);
  joint fit, Fisher matrix and CRLB; per-channel offsets. `ChannelsPSF` holds C independent single-channel models
  (biplane, multicolour); `FourPiPSF` is the coherent two-objective model (extra parameter: the 4Pi phase) and
  `FourPiSplinePSF` its spline table with the phase as an exact analytic dependence.

## Fitting, detection, estimation

- `fit.lm_poisson`: batched Levenberg-Marquardt on the Poisson likelihood with sCMOS read noise
  (d + var ~ Poisson(mu + var)); `localize` uses it with z multi-start for 3D models and clamps to `xy_limit` /
  `z_limits`. `PSF.fit` / `PSF.localize` wrap it and return a `Dataset`.
- `detect_spots` / `extract_spots`: detection per chunk of frames and ROIs centred on the detection pixel (corner =
  pixel - (P - 1) / 2), in photons via the `Camera`, with the read-noise variance per ROI.
- `ZStackEstimator`: `fit_beads` (Zernikes and optional NA / n_med, per-bead xy, per-plane photons / background,
  optional bead size, drift and z-step scale; held-out beads).
- `InSituEstimator`: spots sampled over the whole movie, checkerboard split into training / held-out spots, the
  knot-route Zernike fit (`fit_zernike_knots`; `fit_pupil_knots` adds free pupil maps), overlap rejection by the
  Poisson deviance of the first fit, optional focus / wavelength scan, optional free-knot refinement (`fit_free_knots`,
  kept only if the held-out NLL improves), held-out comparison and a chi2 check of the camera model.
- Every estimator returns a PSF whose `provenance` holds the estimator name and its report; `estimator.data` holds the
  intermediate objects used by the fit report.

## Files

- `psf.zip` (version 0.2.0, [PSF_FORMAT.md](PSF_FORMAT.md)): the shared PSF export, written by `save_zip` and the web
  app: psf.json with one or more channels (model, mapping, field dependence), the .npy arrays, zstack.tif, preview.png
  and fit_report.pdf.
- The single-model json + npz (version 0.1.0; `PSF.save`) and plain calibration json files (SIMCODE calib) are read by
  `PSF.load` as well; `convert_calibration` turns a calibration json into a psf.zip.

## Viewer and web app

- `python -m torch_psf view`: a standard-library HTTP server that computes PSF volumes, isosurfaces, images and CRLB
  curves with torch ([VIEWER_API.md](VIEWER_API.md)) and serves the viewer front end built from `web/`.
- `web/`: a static TypeScript + WebGPU app (WGSL kernels; no server, no Pyodide). Bead z-stack fit, in-situ fit and
  localization of a movie read lazily from the dropped TIFF, psf.zip export and fit report. Its tests compare every
  step with torch_psf on generated references (`tests/make_web_*reference.py`). Layout, kernels and validation:
  [web/README.md](../web/README.md).
