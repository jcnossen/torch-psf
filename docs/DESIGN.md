# torch-psf: library design

Status: implemented (2026-11-14); the stable API is documented in docs/API.md, the viewer API in docs/VIEWER_API.md.
Differences from the proposal below: lengths carry units in their names (pixel_um, wavelength_um, dz_um), z ranges are
z_range=(lo, hi) everywhere, every estimator returns one type (PSFEstimate: psf, report, history, data, save), the
top-level namespace is small (models, files, Optics / Camera, open_movie, detect_spots / extract_spots, localize / crlb,
localize_movie, estimate_psf_beads / estimate_psf_insitu), the viewer front end is part of web/ and bundled at
packaging time, the server returns binary arrays with a JSON header, and the planned real-data validation
runs outside this repository.

## Goal

`torch_psf` (distribution `torch-psf`, version 0.1.0): PSF models, single-emitter fitting and PSF estimation (beads and
in situ) for SMLM, in PyTorch, batched on the GPU. SIMCODE (~/simcode2) imports the shared parts later and keeps
only what is SIMCODE-specific.

## Conventions (the whole package)

- Image coordinates: x = column, y = row, in px, pixel centres at integers. ROI-relative positions are offsets from the
  ROI centre pixel. Emitter z = height above the coverslip in um. Focus `focus_um` = focal-plane position relative to
  the coverslip, in the sample's coordinate (SIMCODE convention). Wavelengths and lengths in um, photons (e-) for
  intensities and background (photons / px).
- A PSF model is a class. Everything else (detect, extract, fit, estimate) is a function that takes models and data
  objects. Names are snake_case. Classes are CamelCase. There are no compatibility aliases.
- Every PSF model has the same interface:
  - `render(pos, grad=False)`: pos [N, 3] = (x px, y px, z um) relative to the ROI centre. Returns the normalized PSF
    [N, P, P], or with grad=True a tuple (psf, d/dx, d/dy, d/dz), where d/dx and d/dy are per px and d/dz is per um.
  - `roisize`, `z_range` (um, the valid z range or None for 2D models), `xy_limit` (px: the largest supported offset,
    which the fitter clamps to).
  - `__call__(pos, intensity=None, bg=None)` = intensity * render(pos) + bg.
- Parameter vector of the fitter: theta = (x, y, z, photons, bg). For a 2D model, z is held fixed.

## Shared configuration objects (`torch_psf.optics`, `torch_psf.camera`)

These are small dataclasses with `to_dict()` and `from_dict()`. Their field names and units are the ones that SIMCODE
already uses, so SIMCODE's ConfigDict blocks (`simcode/auto/config.py` DEFAULTS `optics`, `camera`) map 1:1 onto
them, and SIMCODE can build them from its config instead of duplicating them.

| object | fields (units) | SIMCODE source |
|---|---|---|
| `Optics` | `pixel_um`, `NA`, `wavelength_um`, `n_imm`, `n_med` | auto config `optics` block (the same names); calib json `pixelsize_um` / `wavelength_um` / ... read by `load_psf` |
| `Camera` | `type` ('scmos' / 'emccd'), `gain_e_per_adu`, `offset_adu`, `readnoise_e` (each a scalar, or a per-pixel [H, W] map), `maps` (npz path with `offset_adu`, `readnoise_e` maps, as `camera_maps_*.npz`) | auto config `camera` block, calib json `camera` block |
| movie reader | the interface `n`, `shape`, `read(f0, f1, y0=0, x0=0, S=None) -> float32 ADU` | `simcode/fileio/movie.py` |

- `Camera.to_photons(adu, corner=None)` gives (adu - offset) * gain. `Camera.variance(shape, corner=None)` gives the
  read-noise variance in e^2 (sCMOS, per pixel). With per-pixel maps, ROIs carry their corner so the right part of the
  map is applied.
- Movie reading: `torch_psf.movie` contains the reader interface and the reader classes (`TiffMovie` for multi-part
  Andor TIFF, `ArrayMovie`, `open_movie`). It is moved from `simcode/fileio/movie.py` without changes. The dcimg reader,
  Prefetch and the 4Pi layout stay in SIMCODE. They implement the same interface, so any SIMCODE reader can be passed to
  torch_psf functions.
- The z reference is not part of `Optics`. It belongs to the PSF: `VectorialPSF.focus_um`, and the z grid of a spline.

## Modules

```
torch_psf/
  optics.py      Optics (+ default_device, resolve_device)
  camera.py      Camera (scalars or per-pixel maps, excess noise), photon_transfer (offset / gain from a movie)
  movie.py       TiffMovie, ArrayMovie, open_movie (reader interface)
  zernike.py     noll_to_nm, zernike, zernike_basis
  fit.py         lm_poisson, localize, fisher, crlb, poisson_nll, solve_linear (variable projection: closed-form
                 intensities / backgrounds)
  detect.py      detect_spots (DoG / local maxima / threshold / isolation)
  spots.py       Spots (rois + frame + corner + variance), extract_spots, link_tracks, Tracks
  dataset.py     Dataset (the localizations)
  psf.py         PSF (base class: render, expected, fisher / crlb, fit, localize, save / load, to_spline)
  vectorial.py   VectorialPSF (+ gauss_blur_os, dipole_tensor, photosel_tensor)
  spline.py      CubicSplinePSF (+ geometry_matrix, emitter_height_pos_fn, stage_scan_pos_fn)
  gaussian.py    GaussianPSF (2D), AstigmaticGaussianPSF
  io.py          save, load (own format + SIMCODE calib json), load_camera
  estimation/    PSF estimation steps: beads (fit_beads), knots (fit_zernike_knots, fit_pupil_knots, fit_free_knots),
                 tracks (fit_tracks, variable projection), heldout, zscale
  simulate.py    simulate_emitters, simulate_movie (blinking), simulate_bead_stack, filament_sites, simulate_reference_movie
  estimators.py  ZStackEstimator, InSituEstimator
  viewer.py, __main__.py   the viewer server, the command line
```

### vectorial: `VectorialPSF(torch.nn.Module)`
- The base is SIMCODE's `simcode/psf/vectorial.py`, with the same physics and constructor names (`roisize`, `pixelsize`,
  `NA`, `n_imm`, `n_med`, `wavelength`, `zernike_indices`, `npupil`, `oversample`), the same buffers and parameters
  (`zcoef`, `apod`, `saf`, `NA`, `n_med`) and the same `forward(pos, focus, zcoef_offset, intensity, bg, blur, dipw,
  dipM)`. It includes dipole emission with Fresnel / SAF, the index mismatch, per-emitter Zernike offsets, image-plane
  Gaussian or line blur (`blur`), diattenuation (`diat`), the emission-dipole / photoselection tensor (`dipM`), the
  chirp-z transform, oversampling and pixel integration.
- From the study it adds the free pupil maps `pupil_phase` (waves) and `pupil_amp` as buffers, which are neutral by
  default. With neutral maps the result is bit-identical to SIMCODE.
- New: `focus_um` attribute and `render(pos, grad)` (the derivatives come from forward-mode AD, `torch.func.jvp`, with a
  test against finite differences).
- SIMCODE's `gauss_sigma_um` branch becomes the separate `GaussianPSF` class. `load_psf` returns that class for a calib
  with `psf_gauss_sigma_um`.
- Tests: images equal SIMCODE's `VectorialPSF` to float tolerance, including blur, dipM, diat and Zernike offsets
  (reference npz made once with SIMCODE by `tests/make_reference.py`, so the tests do not need SIMCODE). Images also
  equal the study's `torch_convspline.vectorial` with pupil maps.

### spline: `ConvSplinePSF`
- This is the user's convolutional Catmull-Rom 3D spline. The knots are [nz, nv, nu] on a grid that is S times finer than
  the pixels in u and v and has spacing hz um in z. A P x P ROI is one grouped conv2d with stride S of the 4x4x4 kernels
  over the knot block around the emitter. The value and d/dx, d/dy and d/dz come from the same conv.
- It combines three earlier versions. SIMCODE's ConvSplinePSF provides the stride and the single linear-index gather.
  The study's KnotPSF provides the `index_select` read-out (fast knot gradient through index_add), the derivative that is
  zero outside the knot z range, and the offset limit `xmax` (default >= 3 px). The original torch_convspline provides
  `from_zstack`, a measured z-stack used directly as knots with S = 1, which replaces `SplinePSF`.
- Constructors: `ConvSplinePSF(knots, roisize, z0, hz, S, xmax)`, `ConvSplinePSF.from_vectorial(vec, roisize, z_range,
  S=2, hz=0.04, zpad=..., xmax=3.0, pos_fn=None)`. pos_fn maps knot z to (emitter z, focus). The default is the emitter
  height at `vec.focus_um`. A bead-stack mapping is a helper.
- `render(pos, grad, knots=None)`: with `knots` given, the ROIs are differentiable with respect to the knots (free-knot
  fits).
- `build_planes` / `vjp_vectorial(vec, grad_knots, names)`: the knot build from a vectorial model, and one
  vector-Jacobian product that moves a knot gradient onto the vectorial parameters (the in-situ "knot route").
- The old stride-1 helpers (`eval_image_spline2D/3D`, `ConvSplineInterp3D`, `SplinePSF`, the cupy `fast_slice`) are
  replaced by this class. The cupy dependency goes away.
- Tests: spline built from a vectorial model against the vectorial model itself (relative error, value and derivatives),
  knot gradient against autograd through advanced indexing, and SIMCODE's ConvSplinePSF images to float tolerance.

### fit: Poisson MLE and CRLB
- `lm_poisson(model_fn, data, theta0, var=None, lo, hi, fixed=(), iters, lam0)`: a batched Levenberg-Marquardt loop.
  model_fn(theta) returns (mu [N, M], J [N, M, p]). The noise model is sCMOS: mu + var ~ Poisson, with var the per-pixel
  read-noise variance in e^2 (SIMCODE `lm_poisson_j`). `fixed` holds parameters. It returns theta, log-likelihood,
  iteration count and the converged flag. `jacobian_fd` is a central-difference fallback for models without analytic
  derivatives (SIMCODE `lm_poisson`).
- `localize(psf, rois, theta0=None, var=None, zstarts=5, iters=30)`: theta = (x, y, z, photons, bg) for any model with
  `render(pos, grad=True)`. It uses multi-start in z for 3D models and clamps to `xy_limit` and `z_range`.
- `fisher(psf, theta, var)` and `crlb(...)`.
- SIMCODE's `fourpi/lm.py` then reduces to the 4Pi-specific model functions and imports `lm_poisson` / `fisher` from
  here.
- Tests: on simulated isolated emitters, the std of the error is within a few % of the mean CRLB and the bias is much
  smaller than the CRLB. One test is 2D (GaussianPSF and in-focus vectorial, z fixed) and one is astigmatic 3D
  (ConvSplinePSF from an astigmatic vectorial model).

### detect + spots
- `detect_spots(frames_photons, sigma=(2.0, 6.0) DoG or a single Gaussian, threshold (x noise), noise='mad'|'std',
  nms=9, border, isolation=None, subtract_median=None)` returns a table (frame, x, y, score). The study's
  `detect_crop` and SIMCODE's `auto/spots.py` peak finding (temporal-median subtraction, MAD noise) are both options of
  this one function.
- `extract_spots(movie, peaks, roisize, camera, chunk)` returns `Spots`: rois [n, P, P] in photons, frame, corner (x0,
  y0), the variance ROI for per-pixel read noise, and the peak position.
- `link_tracks(table, max_dist=0.8, max_gap=1, min_len, isolation)` and `extract_tracks` return `Tracks` (rois [B, T,
  P, P], mask, frames, corner, xy_init), as in SIMCODE `insitu/extract_tracks.py`, for the varpro fit.
- Test: recall and precision of detection on a simulated blinking movie with known truth.

### estimation (PSF estimation; `torch_psf.estimation`; every function returns a PSF object and a result record)
- `fit_beads(stack, focus_um [nplanes], model, bead_diameter=None, drift=True, holdout=...)` fits the vectorial model
  (Zernikes, optional NA / n_med) to bead z-stacks, with xy shared per bead, I and bg per plane, optional finite bead size
  (SIMCODE nball) and optional per-stack drift. Sources: study `fit_vec_stack`, SIMCODE `psf/fit_beads.py`.
- `fit_zernike_knots(spots, model, spline_grid, iters, ...)`: in-situ Zernike fit through the knot route. The inner step is
  localize on the spline. The outer step is the knot gradient plus one VJP and an Adam step on the Zernikes.
- `fit_pupil_knots(...)` adds the free pupil phase / log-amplitude maps, with piston / tilt / defocus and the mean
  projected out and an optional roughness penalty.
- `fit_free_knots(...)`: free knots with alternating localize and L-BFGS on the knots, with prior / roughness terms.
- `fit_tracks(tracks, model, camera, ...)`: the variable-projection fit of SIMCODE `insitu/fit_insitu2.py`. Per-frame
  intensities and per-group backgrounds come from `fit.solve_linear`. Per-sample xyz and the PSF parameters (Zernikes,
  focus, optional line blur) are fitted with Adam. SIMCODE-specific extras (per-pattern-angle photoselection) are hooks,
  not built in.
- `heldout`: checkerboard split of the spots by position (`split_checkerboard(tile)`, as SIMCODE `--split cb0/cb1`).
  It reports the held-out NLL per pixel against a reference PSF.
- Every result reports the fraction of spots at the offset limit (`at_xy_limit`) and at the z-range ends.
- Test: simulated movie with a known Zernike set (study scenario A). The in-situ fit recovers it within the tolerance
  that the study reached, and the bead fit recovers its own set.

### io: one PSF file
- Current format: `psf.zip`, specified in [PSF_FORMAT.md](PSF_FORMAT.md) (channels, mappings, field dependence,
  z-stack, preview, fit report). The json + npz described below is its version 0.1.0 and is still read and written.
- `save_psf(path.json, psf)`: JSON with `format: "torch-psf"`, `version`, `type` ('vectorial' | 'spline' |
  'gaussian'), the `optics` block (Optics names), `zernike_noll`, `zernike_waves`, `apod`, `saf`, `focus_um`,
  `psf_blur_px2`, `psf_dipole_M`, `diattenuation`, and an optional `camera` block. Arrays (pupil maps, spline knots and
  the spline grid) go in a sidecar `<name>.npz` that the JSON names. The PSF keys are the same as in SIMCODE's calib
  json.
- `load_psf(path)` reads that format and also SIMCODE's `models/insitu/calib_*.json`: `pixelsize_um`, Zernikes,
  `focus_um`, `psf_blur_px2`, `psf_dipole_M(_angle)` (angle=None / 0 / 1 as in `set_psf_extras`), `psf_gauss_sigma_um`.
  Keys it does not use (pattern, tirf) are ignored. `load_camera(calib)` returns the camera block as a `Camera` and
  resolves `maps`.
- Test: save / load round trip (vectorial with pupil maps, spline, gaussian) gives identical images. Every calib_*.json in
  SIMCODE loads, and the image of the loaded model equals SIMCODE's `set_psf_extras` model.

### pipeline
- `localize_movie(movie, psf, camera, detect=dict(...), roisize, frames=None, chunk)` returns a localization table: a
  numpy structured array `LOC_DTYPE` = frame, x, y (image px), z (um), photons, bg, crlb_x / y / z / photons / bg,
  loglik, chi2, at_limit. It has no Dataset class (SIMCODE has its own and converts from this table).
- `estimate_psf_beads(stacks, focus_um, optics, camera)` and `estimate_psf_insitu(movie, camera, optics, init_psf,
  method='zernike'|'pupil'|'tracks')` run detect, extract and the fit, and return (PSF, report).

## What SIMCODE then imports, and what it keeps

- Imports from torch_psf: `VectorialPSF` and helpers (replacing `simcode/psf/vectorial.py`), `ConvSplinePSF`
  (replacing `psf/conv_spline.py`; SIMCODE's `make_lut` becomes a short call), `lm_poisson` / `fisher` / `crlb`
  (`fourpi/lm.py` keeps only its 4Pi models), `solve_linear` (`insitu/varpro.py`), the movie readers (TIFF / array; the
  dcimg reader stays in SIMCODE and implements the same interface), `Optics` / `Camera` (built from the auto config
  blocks, which keep their names), `load_psf` / `load_camera` for the calib json, peak finding of `auto/spots.py`, track
  linking of `insitu/extract_tracks.py`, and the base of `psf/fit_beads.py` and `insitu/fit_insitu2.py`.
- Stays in SIMCODE: the illumination pattern (fit_pattern, pattern-angle photoselection setup, frame order), the
  networks and training, 4Pi (dcimg layouts, 4Pi models), camera-map estimation from movies (`insitu/camera_maps.py`,
  `camera_gain.py`; possible later move), Dataset, the pipeline stages / Units / reports.
- Migration order (later and separate; ~/simcode2 is not modified in this work): first the vectorial model (the
  reference test guarantees identical images), then the spline, LM, varpro and readers.

## study/, real-data scripts and paper after the rename

- `torch_convspline` is renamed to `torch_psf` in one commit, with no alias. The editable install in simcode-py311 is
  reinstalled under the new name. Every `study/insitu/*`, `study/realdata/insitu_*.py`, quick-start / README and the
  paper's code listings are updated in that commit. `KnotPSF` becomes `ConvSplinePSF`, and `localize` / `crlb` /
  `poisson_nll` move to `torch_psf.fit`.
- `study/insitu/common.py` keeps only the study scenarios (optics, Zernike sets, scenario B screen, F0, pos maps). The
  simulators and `detect_crop` go to `torch_psf.simulate` / `detect`. `study/insitu/fit.py` keeps only the study
  metrics (`evaluate`, `psf_rel_err`), and its estimators are replaced by `torch_psf.estimation` calls.
- Check: the study driver (`run_all.py`, seed 0, reduced size) and each real-data script run after the change. The
  study's headline numbers are reproduced within noise (RNG streams for detection / simulation are kept).

## End-to-end validation on real data (after the restructuring)

Dataset 1809_2: movie `/mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/simflux_5.5ms_500mW.tif` (71506 frames, 120 x 120),
bead stacks `/mnt/data/dmdsf/bn_dnapaint/PSF_calib/raw/zstack{2..7}-1.tif`, reference
`~/simcode2/models/insitu/calib_1809_2_psf3d.json` (+ `camera_maps_1809_2.npz`).

1. `load_camera(calib)` gives gain 0.56 and per-pixel offset / read-noise maps. `TiffMovie` reads the movie.
2. Beads: `estimate_psf_beads` on the stacks (dz 10 nm, stage sign as in SIMCODE, holdout stack 7) is compared with
   SIMCODE's bead fit `models/insitu/beads/psf_fit_sign-1_na149_ball_drift.*`: Zernikes and PSF images.
3. Movie: `detect_spots` / `extract_spots` (6-frame sums as SIMCODE spots, and single frames) and `link_tracks` give
   Tracks. Then `fit_tracks` (zernike + focus + line blur, the SIMCODE stage-A recipe) and `fit_zernike_knots` on
   single-frame spots.
4. Comparison with calib_1809_2_psf3d.json: Zernike vector difference (waves rms), focus, blur; PSF image difference
   over z 0-0.2 um (relative L2); CRLB curves x / y / z vs z at the dataset's median photons / bg; held-out NLL per pixel
   on the other checkerboard parity for both PSFs; fraction at the offset limit.
5. `localize_movie` with both PSFs (first 20000 frames): localization counts, CRLB distributions, the photon / z
   histograms, and the median difference of matched localizations (x, y, z). The result goes in a short report under
   `study/realdata/` (GPU 1 or 2 for the long runs).

## README quick example (top of README, ~15 lines)

```python
import torch, torch_psf as tp
psf = tp.load_psf('calib_1809_2_psf3d.json')          # or: psf, rep = tp.estimate_psf_beads(stack, focus_um, optics, camera)
camera = tp.load_camera('calib_1809_2_psf3d.json')
movie = tp.open_movie('movie.tif')
spline = tp.ConvSplinePSF.from_vectorial(psf, roisize=13, z_range=(0, 0.8))
locs = tp.localize_movie(movie, spline, camera, frames=(0, 1000))   # numpy table: frame, x, y, z, photons, bg, crlb_*
print(len(locs), locs['crlb_x'].mean())
pos = torch.tensor([[0.2, -0.1, 0.3]])                 # x, y (px from ROI centre), z (um)
img, dx, dy, dz = spline.render(pos, grad=True)       # [1, 13, 13] each
```

## examples/psf_viewer: interactive PSF viewer (example, not a core module)

- Start: `python -m torch_psf.viewer [--psf file.json] [--port 8000] [--device cpu|cuda]` (server in `torch_psf/viewer.py`,
  imported only when run; static page in `torch_psf/viewer_static/index.html`, shipped as package data), or
  `python examples/psf_viewer/serve.py`. Prints the localhost URL to open in a browser.
- Server: standard-library `http.server` (ThreadingHTTPServer), no new dependencies. JSON API:
  `GET /` the page; `POST /api/render` {model: vectorial | spline | gaussian, optics {NA, wavelength_um, n_med, n_imm,
  pixel_um}, focus_um, z_um, zernike {noll: waves}, roisize, photons, bg, readnoise_e} -> {xy, xz, yz, zstrip, dx, dy, dz
  (float arrays + shape), crlb {z, x, y, zz}}; `POST /api/load` {path} or an uploaded PSF file -> its parameters (fills the
  sliders); `GET /api/default`. Images computed with torch (CPU default, GPU optional), ROI ~ 21 px, z grid ~ 41 planes;
  a small LRU cache of the knot build for the spline model.
- Page: one HTML file with inline JS (no build step). The 3D view (three.js, see the refinement below) is the core and
  is always on; the 2D panels sit next to it. Sliders: NA, wavelength, n_med, pixel size, focus, z, Zernike astigmatism
  (Noll 5, 6), coma (7, 8), trefoil (9, 10), spherical (11); model choice; photons / background.
  Panels: xy at z, xz and yz sections, z-stack strip, d/dx, d/dy, d/dz images (diverging colour map), CRLB x / y / z vs z
  (nm, canvas line plot). Requests are debounced (~100 ms) so the sliders feel live.
- README: a screenshot made headlessly (Playwright / Chromium if available in the env; otherwise a described panel
  layout).
- Test: a smoke test that starts the server on a free port, calls `/api/render` once and checks shapes and finite values.

### Viewer refinement: the 3D view is the core

- Main panel (not optional): a three.js scene of the PSF volume (x, y, z), rotate / zoom / pan with the mouse. three.js
  is loaded as a pinned version from cdnjs (with the matching OrbitControls), with no build step.
  1. Nested translucent isosurfaces at relative levels 50 %, 10 %, 1 % of the peak (levels editable), each its own colour
     and opacity, depth-sorted for transparency. Meshes are computed server-side by marching cubes:
     `skimage.measure.marching_cubes` when scikit-image is installed, else a small numpy marching-cubes fallback in
     `torch_psf/viewer.py` (no hard dependency). Sent as JSON (float32 vertices / uint32 faces, base64) in physical units
     (um in x, y, z; z axis optionally exaggerated by a slider).
  2. Three orthogonal slice planes (xy, xz, yz) moved through the volume with sliders, textured with the PSF on a log
     colour scale (data textures sent once per volume; slicing is done in the browser from the volume, so moving a plane
     needs no server call).
  3. Ray-marched volume rendering (fragment shader over a 3D texture, log transfer function) as a toggle, only if it
     stays simple; 1 and 2 are the required parts.
- The 2D panels stay next to the 3D view: xy at the chosen z, xz / yz cross-sections, d/dx, d/dy, d/dz, CRLB vs z.
- Volume grid: 2x oversampled in x / y (pixel_um / 2) over the ROI, 10-20 nm in z over the PSF's z range (e.g. 0-1 um ->
  50-100 planes; spline model: evaluated at sub-pixel shifts or read from its knots), recomputed when a slider changes
  (debounced ~150 ms; a request id so stale responses are dropped). API: `POST /api/volume` -> {grid (x, y, z axes),
  volume (float32 base64), isosurfaces [{level, vertices, faces}]}; `/api/render` keeps the 2D panels.

## Web version (`web/`; later phase, after the restructuring and the viewer)

Goal: a static site (GitHub Pages, together with the viewer as the GitHub demo). The user drops a TIFF in the browser and
gets a PSF. Everything is client-side, with no installation and no server. The decision is TypeScript + WebGPU: the
physics and the fits run as WGSL compute shaders driven from TypeScript. There is no Pyodide. The Python library stays
the reference implementation.

### Scope
- v1, beads:
  - Input: a bead z-stack TIFF plus a form with NA, wavelength, n_imm / n_med, pixel size, z step and stage sign.
  - Bead finding: the brightest isolated maxima of the max-projection, ROIs of 20-26 px.
  - Fit: vectorial PSF with Zernike Noll 5-21 and focus. Per bead: xy, I and bg per plane.
  - Held-out check: one bead or stack is left out, and its NLL per px and residual images are shown.
  - Export: the torch-psf PSF json (same schema as `save_psf`), the spline knots (`.npz` written in JS: uncompressed
    zip of `.npy`), and the model z-stack `.tif`.
  - Display: the 3D view, using the same three.js scene module as the viewer.
- v2, in situ: a cropped DNA-PAINT movie. The steps are detect (WGSL DoG + max-pool), ROIs, then the knot-route
  Zernike fit on the conv spline (`fit_zernike_knots`).
  - The movie is never loaded whole: it is read with random access (see "Movie reading" below). Detection streams frame
    chunks through the GPU, and only the spot ROIs are kept. v2 on a full DNA-PAINT movie is therefore limited by
    compute time, not by memory.

### Movie reading in the browser (`src/io/tiff.ts`)
- Random access to the dropped local file:
  - `File.slice(start, end).arrayBuffer()` reads only that byte range, lazily, in all browsers.
  - `file.stream()` is used for one sequential pass where that is enough.
  - The File System Access API is an optional extra where it is available (re-open the same file without a new drop).
    It is not required.
- The IFD chain is parsed once, reading only the small headers (a few hundred bytes per IFD, read in batched slices).
  The result is a per-frame table of strip / tile offsets and byte counts, shape, dtype and compression.
- Fast path: when the frames are uncompressed and contiguous with a constant stride (the usual case for Micro-Manager,
  OME-TIFF, Andor SOLIS and ImageJ stacks), the IFD walk stops after the first few IFDs. The offsets are then
  extrapolated and checked on the last frame, so a 70k-frame header scan takes milliseconds.
- Reads:
  - A frame range is one slice when the frames are contiguous; otherwise one slice per strip run.
  - A crop (y0, x0, S) reads the needed rows only when uncompressed. For compressed data the whole strips / tiles that
    cover it are read.
- Multi-part movies: several dropped files (`movie.tif`, `movie_X1.tif`, ... or `_1`, `_2`, ...) are sorted by their
  part suffix and form one movie with a global frame index, as SIMCODE's `TiffMovie` does.
- Supported:
  - classic TIFF and BigTIFF (64-bit offsets), little- and big-endian;
  - uint8 / uint16 / uint32 / int16 / float32, single-channel;
  - strips or tiles;
  - uncompressed (fast path), Deflate / Adobe Deflate (via `DecompressionStream`, built into the browser), LZW and
    PackBits (small JS decoders);
  - OME-TIFF (the frame order is taken from the OME-XML when it is a simple T or Z series), ImageJ hyperstacks and
    Micro-Manager stacks.
- Not supported in v1: JPEG / JPEG2000 / LZMA / Zstd compression, multi-sample (RGB) or planar-configuration-2 data,
  and Hamamatsu dcimg. These give a clear error message.
- The pinned JS library (utif 3.1.0 or geotiff 2.1.3) is only used for decoding compressed strips, if at all. The IFD
  parsing and the fast path are the app's own small code, so the reading stays lazy.
- Memory: the IFD table is about 16-32 bytes per frame (71k frames: about 2 MB). The reads go through a ring of chunk
  buffers (e.g. 2 x 256 frames, 2 x 7 MB for 120 x 120 uint16; frames are converted to f32 only on the GPU). The GPU
  holds one chunk plus the DoG intermediates (about 3 x the chunk in f32) plus the accumulated ROIs (20k-200k spots x
  13 x 13 x 4 B = 14-135 MB). So 1-2 GB movies need about 50-200 MB in total.
- Expected throughput on a laptop (NVMe SSD, RTX 4060 laptop class):
  - Reading uses one `slice().arrayBuffer()` per 256-frame chunk at about 1-2 GB/s from the OS cache / SSD (Chrome's
    file-reading path, about 0.5 GB/s for a cold read on slower disks).
  - Detection (u16 -> f32, gain / offset, separable DoG, 9 x 9 max-pool, threshold, ROI gather) is far faster than
    reading for small frames.
  - 120 x 120 frames: about 30-60k frames/s read + detect, so the full 71.5k-frame 1809_2 movie (2 GB) takes about
    30-70 s.
  - 512 x 512 frames: about 2-4k frames/s. 2048 x 2048: about 100-200 frames/s.
  - Disk read is the bottleneck. Two chunks in flight overlap reading and GPU work.

### Architecture
```
web/
  package.json, package-lock.json     pinned: vite 5.x, typescript 5.x, three 0.160.x, utif 3.1.0 (or geotiff 2.1.3)
  src/gpu/        device.ts (adapter / limits / buffer pool), kernels/*.wgsl, ops.ts (typed dispatch wrappers)
  src/psf/        vectorial.ts, spline.ts, gaussian.ts      (parameters + GPU buffers, same names / units as torch_psf)
  src/fit/        lm.ts (batched LM), adam.ts, beads.ts, insitu.ts (v2)
  src/io/         tiff.ts, psfjson.ts (torch-psf json read / write), npy.ts, tifwrite.ts
  src/ui/         app.ts, form.ts, progress.ts, view3d.ts (shared with the viewer page)
  test/           node / browser tests; data in ../web/test/.cache/ (generated, git-ignored; tests/data/web/ only holds the one-bead real crop)
```
- Build: Vite + TypeScript, pinned in a lockfile, which produces a static `dist/`. GitHub Actions builds it and deploys
  to Pages. The WGSL files are imported as strings (`?raw`), so there is no shader build step.
- The viewer page (`torch_psf/viewer_static`) and the web app share `view3d.ts` (three.js scene: isosurfaces, slice
  planes). The viewer gets its compiled bundle from the same build. Isosurfaces in the web app use a marching-cubes
  compute shader, or JS marching cubes on the read-back volume, which is small.
- Work runs asynchronously on the GPU queue, chunked per iteration. The UI stays responsive.
  - Progress: iteration, NLL, and a PSF thumbnail every n iterations.
  - Cancel: a flag checked between dispatch batches. Cancel keeps the last iterate, which can be exported.

### Progress, stopping and cancellation (`src/ui/progress.ts`, `src/core/task.ts`)
- Every long operation (TIFF header scan, read + detect, ROI / track collection, bead fit, in-situ fit, spline build)
  is a `Task` with a progress record and a stop token.
  - The progress bar shows the stage-specific counters: frames read / detected of the total, spots collected (good
    isolated spots / tracks vs the target), and fit iteration with the current NLL per px, held-out NLL and step size.
  - It also shows the elapsed time and the estimated remaining time (moving average of the chunk / iteration rate).
- Two kinds of stop, both available at any time:
  - "Abort": cancel and discard. GPU buffers are released, and the app returns to the previous state.
  - "Use what we have": stop early and continue with the partial result.
    - When reading / detecting, this is the "Enough frames" button: reading stops after the current chunk, and the
      pipeline continues with the spots collected so far.
    - In a fit it is "Stop and keep current estimate": the last accepted iterate is kept, and the held-out check and
      export run on it.
- Automatic stop: reading stops when a target number of good isolated spots / tracks is reached. The target is set in
  the form. Defaults are the number the next step needs: about 8000 spots for the in-situ Zernike fit (as the
  library's default `nsamples`) plus about 6000 held-out spots, and all planes for beads. Fits stop automatically on
  convergence (relative NLL change < 1e-6 over 5 iterations) or the iteration limit.
- Implementation:
  - A stop token in the style of AbortController (`{signal: 'run' | 'enough' | 'abort'}`) is passed to every task. It
    is checked between chunks and iterations; never in the middle of GPU work.
  - GPU work is submitted in bounded chunks (one read chunk, one fit iteration or a fixed number of patches, about
    10-50 ms of GPU time each). The next chunk is submitted after `queue.onSubmittedWorkDone()`. This keeps the UI and
    rendering responsive and bounds the latency of a stop to one chunk.
  - Main-thread work is kept small. TIFF decoding of compressed strips and the IFD scan run in a Web Worker (WebGPU is
    also available in workers, so the whole compute pipeline can run in a dedicated worker). The UI thread only renders
    progress and the 3D view. Progress messages are throttled to about 10 per second.
  - Aborting rejects the task's promise with an `AbortError`, and each stage cleans up in `finally`. "Enough" resolves
    with the partial result and a flag `partial: true`, which is recorded in the exported json (`provenance.partial`,
    frames used, spots used).

### WGSL kernels
1. `pupil`: the per-dipole pupil fields P [6, N, N] (complex as vec2<f32>). It includes Fresnel t_s / t_p, the
   index-mismatch kz, SAF evanescence, apodization, the Zernike phase sum_j c_j Z_j (Z precomputed on the CPU and uploaded
   once), and the free pupil maps.
2. `vec_forward`: per image b = (emitter / plane), the phase exp(i(W - kx x - ky y - kz_imm f + kz_med z)), then the
   matrix DFT as two batched complex GEMMs (Fy [ns, N] @ Pb [N, N] @ Fx^T [N, ns]; tiled with workgroup shared memory).
   This is followed by |E|^2 summed over the dipoles (or with the dipole tensor M), the optional Fourier-domain blur
   (separate small FFT kernel, v2), and pixel integration (mean over os x os) with the energy normalization.
3. `vec_jacobian`: the forward-mode derivatives in the same pass.
   - Positions: d/dx multiplies Pb by -i kx dx, so dI/dx = 2 Re(conj(E) dE).
   - Focus and z: likewise with kz.
   - Zernike j: dPb/dc_j = i 2 pi Z_j Pb, giving one extra DFT per parameter. The 17 Zernikes plus focus are batched as
     extra GEMM columns.
   - Output: mu and J [B, P*P, p].
4. `poisson_lm`: per patch, the gradient g = J^T (d/mu - 1) and the Hessian H = J^T diag(1/mu) J (reductions in shared
   memory). Then the damped solve: a small p x p Cholesky per patch for the per-bead / per-spot parameters (p <= 5),
   and a CPU-side solve for the global Zernike block (18 x 18, read back each iteration; tiny). The scheme is a Schur
   complement: per-bead parameters are eliminated on the GPU, and the global step is solved on the CPU.
5. `spline_build`: knots from vectorial images at S x S sub-pixel shifts, which reuses `vec_forward` plus a
   permute / flip kernel.
6. `spline_render`: the conv-spline ROI with value and d/dx, d/dy, d/dz. Per output pixel this is the 4x4x4 Catmull-Rom
   weights times the knot block gathered by linear index (stride S). It also has an adjoint kernel (atomicAdd on f32 is
   not native in WGSL; instead scatter by the knot-owner pass or by u32 CAS loops) for knot / VJP gradients in v2.
7. v2 helpers: DoG blur, max-pool peak finding, ROI gather, Adam update.

### Expected runtime, typical bead stack (6 stacks x 100 planes x 26 x 26 px), laptop GPU
The example GPU is an RTX 4060 laptop class, about 10 TFLOP/s f32 peak, about 1-3 TFLOP/s achieved in WGSL GEMMs.
- Full settings (npupil 64, oversample 3, ROI 26 px, about 12 beads x 100 planes = 1200 images):
  - Forward: about 25 MFLOP per image, about 30 GFLOP per pass.
  - Forward + Jacobian (19 global + 5 local parameters): about 0.6 TFLOP per LM iteration, about 0.3-0.6 s.
  - 20-30 LM iterations: about 10-20 s.
- Reduced settings (npupil 48, oversample 2, 50 planes in the PSF range): about 2-4 s.
- Overheads: TIFF decode of 25 MB in JS takes about 0.5 s, and the first shader compile about 1 s.
- Integrated GPUs (Intel Iris / Apple M1): about 3-5x slower, still under about a minute.
- v2 in situ (4000 spots, 80 iterations): about 1-3 min on a laptop GPU.

### Browser support and fallback
- WebGPU is available in Chrome / Edge 113+ on Windows, macOS and ChromeOS, and in Chrome on Android 121+. On Linux it is
  behind a flag in some versions. Safari 26+ supports it (earlier versions had a developer flag). Firefox ships it on
  Windows (141+); other platforms are in progress.
- Without WebGPU there is no CPU fallback, because a second (JS) implementation of the physics is not worth the
  maintenance. The page detects `navigator.gpu` and adapter features. Without them it shows a clear message (supported
  browsers / flags), and the viewer still shows precomputed example PSFs (volumes and meshes shipped as static files) so
  the demo is not empty. For an actual fit on such a machine the user is pointed to the Python package
  (`pip install torch-psf`).
- Precision: everything is f32. f16 is not used.

### Validation against the Python library (the reference)
- Shared test data and expected outputs live in the repo under `web/test/.cache/ (generated, git-ignored; tests/data/web/ only holds the one-bead real crop)`:
  - two small stacks: a simulated stack with a known Zernike set + bead xy, and a crop of a real bead stack (PSF_calib
    zstack2-1, 2-3 beads);
  - the forward test cases: parameter sets and their torch_psf images / Jacobians (npz);
  - the library's fit results (json) for both stacks.
  They are generated by `tests/make_web_reference.py`, which uses torch_psf.
- Kernel tests (the web test runner in a headless Chrome with WebGPU, via Playwright):
  - `vec_forward` images equal torch_psf within relative 1e-4 (f32 GEMM order);
  - `vec_jacobian` equals torch_psf / finite differences within 1e-3;
  - `spline_render` equals ConvSplinePSF within 1e-5.
- Fit tests: the web fit of each test stack compared with the library fit. Pass criteria: Zernike difference < 0.005
  waves rms, focus < 5 nm, PSF image relative L2 < 1e-2 over the z range, held-out NLL per px within 0.1 %.
- `web/validate.py` (Python, torch_psf) takes an exported web json and the stack and prints the same comparisons. It
  is used for user stacks and for the real-data check. Every exported json must load with `torch_psf.load_psf`.
- CI: Python tests always. Web tests run in a GitHub Actions job with Chrome and WebGPU (SwiftShader / Dawn's CPU
  adapter for correctness only, not speed). If that adapter is unavailable in CI, the job only builds, and the tests are
  run locally on a GPU machine before a release.
