# PSF file format (psf.zip)

A `psf.zip` holds one PSF model of a setup together with a rendered z-stack, a preview image and a report of the
estimation. One file holds **all channels** of the setup (biplane, multicolour, dual view, 4Pi): the channels are linked
by coordinate mappings, and each channel's PSF may depend on the field position. Every program that writes the
members below can be read by every other one; members a reader does not need (the TIFF, PNG, PDF) can be ignored.

Writer / reader in Python: `torch_psf.io.save_zip(path, psf, ...)`, `PSF.load(path)` / `torch_psf.io.load(path)`
(single-channel and coupled files; `channel=` for one channel of an independent multi-channel file) and
`torch_psf.psfzip.PSFZip.read(path)` (all channels with their mappings and field dependence). Implementation:
`torch_psf/psfzip.py`, `torch_psf/psfreport.py`.

## Contents of the zip

| member | always | content |
|---|---|---|
| `psf.json` | yes | the model description (below); UTF-8 JSON |
| `<owner>/<name>.npy` | when the models have arrays | NumPy `.npy` arrays (pupil maps, spline knots, camera maps, mapping grids), referenced by member name from `psf.json`. `<owner>` is the channel name, `source`, `<channel>/source` or `camera` (a convention; readers use the names in `psf.json`) |
| `zstack.tif` | yes | the model rendered through z for every model type (ImageJ hyperstack, float32) |
| `preview.png` | yes | per channel: xy slices through z and an xz section |
| `fit_report.pdf` | yes | an overview of the estimation; model pages only when no estimator data was available |

There is no residuals file. Members may be stored or deflated (the Python writer deflates everything except the PNG).
Readers must not depend on the member order; the Python writer puts `psf.json` first.

## Versioning

- `format` is `"torch-psf"`; `version` is `"0.2.0"` or `"0.3.0"` for the zip. A reader accepts the minor versions it
  knows (this one: every `0.2.x` and `0.3.x`; unknown keys are ignored) and refuses others.
- `0.3.0` = `0.2.0` plus two things a `0.2` reader would read wrongly without noticing (it ignores unknown keys): a
  model with a `pixel_integration` other than `box` (a 0.2 reader renders `box`, up to ~0.6 % of the peak off) and the
  `integrated_spline` model type. A writer writes `0.3.0` exactly when a model block of the file (a channel, its
  `source`, the top-level `source`) is an `integrated_spline` or has `pixel_integration` other than `box`, and
  `0.2.0` otherwise, so every file a 0.2 reader can read correctly stays a 0.2.0 file (byte for byte as before), and
  the others are refused by 0.2 readers (the Python reader before this revision and the web app check the minor
  version).
- The 0.1.0 files (one JSON with `type` at the top level plus a sidecar `.npz` named by `arrays`) are still written by
  `PSF.save` and read by `PSF.load`; so are plain SIMCODE calibration JSONs (no `format`, the vectorial keys at the top
  level). A 0.2.0 channel `model` block holds exactly the keys of the 0.1.0 JSON of that model type, so a 0.1.0 file
  converts to a single-channel zip without renaming anything.

## Coordinates and units

- x = column, y = row, in camera pixels; pixel centres at integer coordinates.
- Emitter positions in a model are offsets from the ROI centre (px); z is in um (meaning per model type below); `I` in
  photons, `bg` in photons / px, phases in radians, Zernike coefficients in waves rms (Noll index and normalization),
  pupil phase maps in waves.
- Channel coordinates are **channel crop** coordinates (the channel's own image after cropping it from the camera
  chip, see `crop`); the reference channel is channel 0.
- All channel models use the same emitter coordinate z: a biplane focus offset lives in the models (the `focus_um` of a
  vectorial channel, the knots of a spline channel), not in the mapping.

## psf.json

| key | type | required | meaning |
|---|---|---|---|
| `format` | str | yes | `"torch-psf"` |
| `version` | str | yes | `"0.2.0"`, or `"0.3.0"` for the files that need it (Versioning) |
| `psf_type` | str | yes | reader hint: the torch-psf type of the whole file: a model type (single channel or coupled) or `"channels"` (independent channels) |
| `n_dims` | int | yes | 3, or 2 for models without a z axis (z ignored) |
| `emitter_params` | [str] | yes | the parameters of one emitter, in order: `["x","y","z","I","bg"]`, 2D `["x","y","I","bg"]`, 4Pi `["x","y","z","I","bg","phi"]` |
| `units` | {param: str} | no | the units above |
| `per_channel_params` | [str] | no | parameters with one value per channel (default for multi-channel files: `["bg"]`) |
| `optics` | dict | no | default optics of all channels (`pixel_um`, `NA`, `wavelength_um`, `n_imm`, `n_med`; may be partial, e.g. only `pixel_um` for a spline); a channel model's own `optics` takes precedence |
| `camera` | dict | no | camera (below) |
| `provenance` | dict | yes (may be empty) | free-form: estimator, settings, report (the estimators store their report here) |
| `channels` | [dict] | yes | one entry per output channel (below) |
| `source` | dict | no | the coupled model behind all channels (4Pi), `{type, model, arrays}` |
| `patterns` | dict | no | pattern / modulation metadata (SIMFLUX, SIMCODE) |
| `acquisition` | dict | no | sample / acquisition metadata that is not the PSF (TIRF depth, emitter height distribution, ...), free-form, kept verbatim |
| `zstack` | dict | yes | description of `zstack.tif` |
| `preview` | dict | yes | `{"file": "preview.png"}` |
| `report` | dict | yes | `{"file": "fit_report.pdf", "sections": [...]}` (the sections written); when the PSF was saved with an estimator report also `estimator` (name), `results` (the estimator's report: counts, losses, thresholds, ...) and `settings`, JSON values only (arrays longer than 10^4 elements stay in the PDF) |

### Channels

Two kinds of multi-channel model:

- **Independent channels** (biplane, multicolour, dual view): every channel is a model of its own, of any type, plus a
  mapping to the reference channel, an intensity ratio and optionally a field dependence. One emitter is seen in all
  channels (with `I` and the per-channel `bg`); different species imaged one after the other are separate files.
- **Coupled multi-output models** (4Pi): one physical model produces all C channel images of an emitter, with a shared
  extra emitter parameter (the 4Pi phase `phi`, declared in `emitter_params`). The vectorial 4Pi model is stored once
  in the top-level `source`; its channels are `source_output` entries. Its spline form is C `phase_spline` channels (the
  phase dependence is analytic), with the vectorial model in `source`.

An ordinary single-objective model is a file with one channel. A file has either independent channels or the outputs
of one coupled model, not both.

Channel entry:

| key | type | required | meaning |
|---|---|---|---|
| `name` | str | yes | unique name (`ch0`, `ch1`, ... by default; e.g. `plane0`, `red`, `top`) |
| `type` | str | yes | model type: `vectorial`, `spline`, `integrated_spline`, `gaussian`, `astig_gaussian` (independent), `phase_spline`, `source_output` (coupled) |
| `model` | dict | yes | the model's parameters: exactly the keys of the 0.1.0 JSON of that type, without `format` / `version` / `type` / `camera` / `provenance` / `arrays` (tables below) |
| `arrays` | {name: member} | no | the model's arrays by their 0.1.0 npz names (`pupil_phase`, `pupil_amp`, `knots`; `coefs` / `coefs_angle` for integrated_spline; `inc` / `cr` / `ci` for phase_spline) |
| `source` | {type, model, arrays} | no | independent channels: the model this channel was derived from (e.g. the vectorial model a spline was tabulated from) |
| `index` | int | source_output | the output index of the source model |
| `phase_param` | str | phase_spline | the emitter parameter of the phase (default `"phi"`) |
| `mapping` | dict | no | channel coordinates <-> reference channel (below; default identity; channel 0 is identity) |
| `crop` | {x0, y0, w, h} | no | the channel's crop on the camera chip (px): chip = channel + (x0, y0) |
| `flip` | str | no | `"none"`, `"lr"`, `"ud"`, `"both"`: the mirror of this channel's optics relative to channel 0 (information: the mapping already contains it) |
| `intensity_ratio` | float | no | expected photons in this channel = `intensity_ratio` x `I` (default 1). For coupled models 1: the channel weights (4Pi `tau`) are inside the model |
| `field` | dict | no | field dependence of this channel's PSF (below) |

Any other key is kept (written as given, ignored by readers that do not know it).

### Model types (`channels[c].model`, `source.model`)

**vectorial** (`VectorialPSF`): `optics` {pixel_um, NA, wavelength_um, n_imm, n_med}, `zernike_noll` [nz] (Noll
indices), `zernike_waves` [nz], `apod` [3] (radial amplitude exp(sum_n c_n rho^(2n))), `saf` [4] (supercritical annulus
tilt x / y, amplitude x / y), `focus_um` (focal plane relative to the coverslip), `psf_blur_px2` [3] or null (image
blur covariance sxx, syy, sxy in px^2), `psf_dipole_M` [3, 3] or null (emission dipole tensor), optional
`psf_dipole_w` [3], `diattenuation` or null, `npupil`, `oversample`, `roisize`, `z_range_um` [2] or null, optional
`pixel_integration` (`box` = the mean of oversample x oversample point samples per pixel, the default when absent;
`spectral` = the exact pixel integral of the band-limited image; `gauss_legendre` = oversample Gauss-Legendre nodes
per axis; a reader must refuse a value it does not know instead of rendering `box`; readers written before the key
existed ignore it and render `box`: off by up to ~0.6 % of the peak at oversample 3, 108 nm pixels); the SIMCODE
calibration names `pixelsize_um`, `wavelength_um`, `NA`, `n_imm`, `n_med` (duplicates of `optics`, so SIMCODE reads the
block as a calibration). Optional per-pattern-angle terms (SIMCODE): `psf_dipole_M_angle` [A, 3, 3],
`psf_blur_angle_px2` [A, 3]. z = emitter height above the coverslip (um) at focus `focus_um`.
Arrays (optional; absent = neutral): `pupil_phase` [npupil, npupil] float32 waves, `pupil_amp` [npupil, npupil]
float32 factor; pupil grid: sample i at (i - (npupil - 1) / 2) / (npupil / 2) of the pupil radius, x = column.

**spline** (`CubicSplinePSF`): `spline` {roisize, z_range [2] um, oversample S (knots per px), dz_um, zpad, xy_limit
(px), c (stored half width, px), z0_um (z of knot plane 0), nz, u0_px (lateral offset of knot 0 from the PSF centre,
px), pixel_um}. Array `knots` [nz, n, n] float32 (n = (2c + 1) S), Catmull-Rom knots of the pixel-integrated PSF
g(u, v, z), knot (k, i, j) at z = z0_um + k dz_um, v = u0_px + i / S, u = u0_px + j / S. z: the coordinate the spline
was tabulated in (emitter height for in-situ models; -focus for bead stacks).

**integrated_spline** (`IntegratedSplinePSF`, torch_psf/pixint.py): a different representation from `spline`, under its
own type name and array names so that readers that do not know it refuse it (an unknown `type`) instead of reading
its coefficients as `spline` knots. `integrated_spline` {basis `"cubic_bspline"`, value `"pixel_integral"`, roisize,
z_range [2] um (the heights it is accurate on), oversample S (knots per px), dz_um, zpad (knot planes outside z_range),
xy_limit (px, at roisize), c (stored half width, px), z0_um, nz, u0_px, pixel_um, n_angles (0 = one model)}, the grid
of `spline` (knot (k, i, j) at z = z0_um + k dz_um, v = u0_px + i / S, u = u0_px + j / S; u, v = offset of a point
from the emitter, px). Arrays: `coefs` [nz, n, n] float32 (n = (2c + 1) S), or with per-pattern-angle terms
`coefs_angle` [n_angles, nz, n, n] float32 (exactly one of the two). The coefficients c_kij are those of a
tensor-product cubic B-spline (kernel B(t) = 2/3 - t^2 + |t|^3 / 2 for |t| <= 1, (2 - |t|)^3 / 6 for 1 < |t| < 2,
0 beyond) that interpolates the point density of the PSF (fraction of the photons per px^2) at the knots:
f(u, v, z) = sum_kij c_kij B((z - z_k) / dz_um) B(S (v - v_i)) B(S (u - u_j)). The value of a pixel is the exact
integral of f over the pixel (1 x 1 px, centred at the pixel offset (o_x - x, o_y - y) from the emitter): per lateral
axis the integral of B over the pixel is a difference of its antiderivative (a piecewise quartic), so the patch is
separable and exact; it is not an interpolation of pixel-integrated values (the `spline` knots). Coefficients of
the knots beyond the stored grid are not defined: readers clamp |x|, |y| to the offset limit and z to the knot range
(it is accurate on z_range). Per-angle models (SIMCODE photoselection; the vectorial `psf_dipole_M_angle` /
`psf_blur_angle_px2`): one coefficient set per pattern angle, selected per frame by the parameter `angle` (Python:
`PSF.load(path, angle=a)`); the model without an angle is the mean of the angle coefficient sets (= the mean of the
angle images). `source` (optional): the vectorial model the coefficients were built from
(`IntegratedSplinePSF.from_vectorial`: the point density of the exact band-limited image, prefiltered with the cubic
B-spline, mirror boundary in all three axes).

**gaussian** (`GaussianPSF`, 2D): `gaussian` {sigma_px [2], pixel_um}, `roisize`. No arrays.

**astig_gaussian** (`AstigmaticGaussianPSF`): `gaussian` {sigma_px [2] (in focus), gamma_um, d_um, z_range_um [2],
pixel_um}, `roisize`; sigma_x(z) = sx0 sqrt(1 + ((z - gamma) / d)^2), sigma_y(z) = sy0 sqrt(1 + ((z + gamma) / d)^2).

**fourpi** (`FourPiPSF`, only as the top-level `source`): `optics` (top arm), `roisize`, `z_range_um`, `psf_dipole_M`,
`fourpi` = the parameter dict {meta {R, px, NA, n, lam, npupil, os, mirror, zidx, pol [C], NA_t, n_imm_t, NA_b, n_imm_b,
n_med, zmode, asym}, zt [nz] (top-arm Zernikes, waves), zb [nz] (bottom arm), phi [C] (channel phases, rad), tau [C]
(channel transmissions, sum 1), a (arm amplitude ratio), gamma (coherence), d_bot [3] (bottom-arm offset: px, px, um),
blur_px2, depth [2] (asymmetric arms only), raw {parameter: value} (optional: the internal parameters tau_logit,
a_log, gamma_logit, logblur, ...; when present a reader takes them instead of the natural values, which makes the
round trip bit-exact)}; see `torch_psf/fourpi.py` for the physics. No arrays.

**source_output**: channel `index` of the top-level source model. `model` is informative only: {pol, phi_rad, tau} of
that output.

**phase_spline** (`FourPiSplinePSF` channel c): `model` = {`spline` (as for a spline, the grid of all three
components), `tau` (the channel weight of the source model, for reference)}; arrays `inc`, `cr`, `ci`, each
[nz, n, n] float32 knots on that grid (float16 when written with `spline_knots='float16'`: half the size, relative
error ~5e-4; readers convert to float32). Written with `spline_knots='rebuild'` the channels have no `arrays` but a
`rebuild` {method: "FourPiPSF.to_spline", ...} entry: the reader tabulates the top-level `source` FourPiPSF on the
stored grid (a 4Pi spline is ~100 MB of knots otherwise). The channel image is

    g_c(x, y, z; phi) = inc_c(x, y, z) + cos(phi) cr_c(x, y, z) - sin(phi) ci_c(x, y, z)

with phi the emitter parameter named by `phase_param` (the channel phase phi_c is inside cr / ci). All phase_spline
channels of a file share the grid.

### Mapping (`channels[c].mapping`)

Maps channel-c crop coordinates to the reference channel 0 (`fwd`) and back (`inv`). Types:

- `{"type": "identity"}` (default).
- `{"type": "affine", "matrix": [[a, b, tx], [c, d, ty]]}`: (x0, y0) = M (x, y, 1); the inverse is computed.
- `{"type": "poly", ...}`: **the SIMCODE PolyMap JSON unchanged** (`simcode/channels/registration.py`), so a SIMCODE
  registration file drops in: `order`, `fwd` {c: [nf, 2]} (channel c -> 0), `inv` {c: [nf, 2]} (0 -> channel c),
  `feats_norm` {cx, cy, sx, sy}, plus whatever metadata keys SIMCODE wrote (kept). Features
  u^i v^j for i + j <= order, i the outer loop, j the inner one (order 2: 1, v, v^2, u, uv, u^2), u = (x - cx) / sx,
  v = (y - cy) / sy; nf = (order + 1)(order + 2) / 2; (x0, y0) = feats(x, y) @ fwd[c]. The keys of `fwd` / `inv` are
  channel indices as strings; the reader uses the entry of its own index (or the only entry). A whole SIMCODE file may
  be pasted into each channel. Without `feats_norm` the crop centre (w / 2, h / 2, w / 2, h / 2) is used (as SIMCODE's
  `PolyMap.from_dict(size=(W, H))`). The flip of the optics is already absorbed in the coefficients (SIMCODE
  `unflip_coefs`); `flip` on the channel records it.
- `{"type": "grid", "fwd": member, "inv": member}`: displacement fields [2, H, W] float32 (dx, dy) on the channel
  (`fwd`: x0 = x + dx(x, y)) and reference (`inv`) pixel grids, bilinear interpolation; `inv` optional (the reader then
  inverts by fixed-point iteration).

### Field dependence (`channels[c].field`)

- **`zernike_poly`** (implemented): `{"type": "zernike_poly", "order", "zernike_noll": [k], "coef": [k, nf] or a
  member name, "feats_norm": {cx, cy, sx, sy}, "reference_point": {x, y} (optional)}`. The waves of Zernike
  `zernike_noll[i]` at channel coordinates (x, y) are `feats(x, y) @ coef[i]`, with the PolyMap features above (in that
  channel's coordinates). The channel model's `zernike_waves` are the values at the reference point (default the
  `feats_norm` centre, where they equal `coef[:, 0]`); the writer warns when they differ. It applies to the channel's
  vectorial model, or to its `source` for a spline channel. **No extrapolation:** readers clip the normalised
  coordinates u = (x - cx) / sx and v = (y - cy) / sy to [-1, 1] before evaluating the features, so outside the
  `feats_norm` box the polynomial keeps its value at the box edge; writers set the box to the region the field was
  fitted on. A mode whose higher-order coefficients are all zero has a lower order (per-mode orders are implicit in
  `coef`). Python: `PSFZip.read(p).channels[c].model_at(x, y)` returns the vectorial model at that field point.
- **`grid`** (specified, not yet implemented): `{"type": "grid", "anchors": [{"x", "y", "type", "model", "arrays"}],
  "interpolation": "nearest" | "idw" | "zernike_linear"}`: complete models at anchor positions (channel coordinates),
  interpolated by nearest anchor, inverse-distance weighting of the rendered PSFs, or linear interpolation of the
  Zernike coefficients over a Delaunay triangulation of the anchors.

For a coupled model a field dependence belongs to the arms of the source; reserved key `source.field` = {arm:
field block} with arm `top` / `bottom` (not yet specified further).

### Patterns (`patterns`, optional)

Structured illumination does not change the PSF: one PSF, the patterns are metadata. `{"type": "simflux" | "simcode" |
other, "n_angles", "n_phases", "angles": [{"k_px": [kx, ky] (rad / px), "phases_rad": [...], "depth"}], "frame_params":
["angle"], ...}` (free-form beyond these keys). A per-angle PSF term (SIMCODE photoselection: the emission dipole
tensor per pattern angle; a per-angle blur) is a parameter of the one PSF, not separate PSFs: it lives in the vectorial
model (`psf_dipole_M_angle` [A, 3, 3], `psf_blur_angle_px2` [A, 3]) and is selected per frame by the known parameter
`angle` (Python: `PSF.load(path, angle=a)`; `angle=None` = the average).

### Acquisition (`acquisition`, optional)

Free-form metadata of the sample and the acquisition that a user of the PSF may need but that is not part of it, e.g.
SIMCODE's `tirf_depth_um`, `emitter_z_median_um`, `emitter_z_quantiles_um`, `emitter_z_range_um` (the height
distribution a network is trained on). Readers keep it verbatim (Python: `PSFZip.read(p).acquisition`).

### Converting calibration json files

`python -m torch_psf.psfzip calib.json [...] [-o out]` (Python: `torch_psf.psfzip.convert_calibration`) converts a
SIMCODE calib json or a torch-psf 0.1.0 PSF json + npz without loss: the PSF and the camera (with its maps) as usual;
of the remaining top-level keys `source` / `psf_v2` go to `provenance`, a `pattern` dict to `patterns`
(`type: "simcode"`, plus `pattern_note`), PSF-model keys the model does not use (the vectorial keys of a Gaussian
calibration) to `provenance.unused_model_keys`, everything else to `acquisition`; the optics keys of the json go to
the top-level `optics` also when the model does not use them; values that equal the json's at float32 precision are
written as in the json; the complete original json is also kept in `provenance.converted_from` = {file, json}.

### Camera (`camera`, optional)

The `torch_psf.Camera` fields: `gain_e_per_adu`, `offset_adu`, `readnoise_e`, `type` (`scmos` | `emccd`),
`excess_noise`. Per-pixel maps are arrays in the zip: `arrays` {field: member} (e.g. `camera/offset_adu.npy` [H, W]
float32), and the scalar field then holds the calibrated scalar given with the maps (e.g. the global offset a
simulator uses), or the map median when none was given; `maps` is null. Other entries of the maps npz are kept as
`extra_arrays` {name: member} (`camera/extra/<name>.npy`, any non-object dtype such as bool, int64 or a unicode
string, 0-d allowed; readers that do not need them skip dtypes they cannot parse); other keys of the camera dict
(notes etc.) are kept as given.

### Numbers and reproducibility

Scalars are JSON numbers (float64). Optics scalars that the model holds as float32 are written as the shortest
decimal of that float32 (1.33, not 1.3300000429). Arrays in the JSON are float64 decimals of the stored values.
Writers should make the zip byte-reproducible: fixed member times (torch-psf: 1980-01-01), no creation date or wall-clock
time stamp in the PDF (torch-psf dates the report only when SOURCE_DATE_EPOCH is set), no creation time in
provenance.

### zstack.tif

Always present, for every model type. ImageJ hyperstack, float32, axes **ZCYX** (Z planes, C channels, Y rows, X
columns; C and Z may be 1: with C = 1 the ImageJ description has no `channels=`, so readers that squeeze singleton
axes, e.g. tifffile's default series, report ZYX; read with squeeze=False for ZCYX), ImageJ metadata `spacing` = z step (um), `unit` = um, resolution = 1 / sample spacing
(px per um), `Info` = JSON {format, version, psf_type, z_um, channels, sample_px}. `psf.json` `zstack`: `{file, axes:
"ZCYX", shape [Z, C, Y, X], dtype, z_um [Z], sample_px, sample_um, channels [names], position {x: 0, y: 0}, params
(e.g. {phi: 0} for 4Pi), value}`.

- Planes: `zstack_planes` (default 25) evenly over the z range of the (reference) model, ends included; one plane at
  z = 0 for 2D models.
- Lateral sampling: `zstack_px` = 1 / k camera px (default 1). Sample (m, n) holds the pixel-integrated PSF of a camera
  pixel centred at ((n - (kP - 1) / 2) / k, (m - (kP - 1) / 2) / k) px from the emitter (rendered with k x k sub-pixel
  shifts), P = the ROI size (the largest of the channels). Values: the normalized PSF as rendered (fraction of the
  photons of the emitter in that camera pixel), so a plane sums to ~k^2 in focus.
- Channels on the C axis, in `channels` order; a coupled model at phi = 0.

### preview.png

One row per channel: five xy slices through z (normalized per panel) and an xz section through the centre row.

### fit_report.pdf

An overview of the estimation, never every ROI. Sections (written only when their data is available; `report.sections`
lists them; a PSF saved without estimator data gets `summary`, `aberrations` and `pupil` (models with Zernikes),
`psf`, and `field`):

| section | content | needs |
|---|---|---|
| summary | model, optics, camera, channels, estimator, settings, scalar results, warnings | always |
| spots | ROIs detected / used / held out / rejected and why (deviance histogram with the threshold), at the offset limit | report counts (`n_spots`, `n_train`, `n_test`, `n_rejected`, or `n_beads`, `n_planes`) |
| quality | the range of ROI quality: a histogram of a per-ROI score q (higher = worse) with the acceptance cutoff and marks at the ROIs shown, then data / model / data - model with q, its rank and photons / bg / z of the 3 best ROIs, the 3 last accepted (q just below the cutoff) and the 3 first rejected (just above; only when ROIs were rejected). InSitu: q = Poisson deviance per px of each training spot after its single-emitter fit with the first-fit PSF, cutoff = the overlap rejection threshold (`report.deviance.threshold`), the model of a rejected spot is that fit (not written with `reject_overlap=False`). ZStack: q = chi2 per px of a bead (mean over its planes; images: its sharpest plane); the estimator rejects no beads, so the line is a reference, median + 3 x 1.4826 MAD, labelled as such | InSitu: `report.deviance` and the spots (first-fit values `data.deviance`, `deviance_theta`, `deviance_psf`; else recomputed with `vectorial_spline`); ZStack: `data.rois` / `model` / `chi2` |
| fit_stats | histograms of chi2, photons, background, z | estimator data (InSitu: held-out spots refitted with the final spline; ZStack: every bead plane with the fitted model) |
| convergence | loss per iteration of each stage; focus / wavelength scan | `history` / `report.scan` |
| aberrations | Zernike bar chart (error bars when the report has `zernike_std`), focus, z offset / scale | a vectorial model (the model, a source, the arms of a 4Pi, or the fitted vectorial of the data) |
| pupil | phase and amplitude, Zernike-only and with the free pupil maps (4Pi: both arms) | as aberrations |
| psf | xy slices and xz sections | always |
| examples | a few ROIs over z: data, model, difference | estimator data |
| residuals | mean (data - model) / photons, binned by z | estimator data |
| field | Zernike maps over the field; the non-affine part of a poly mapping (quiver) and its round-trip error | `field` / poly `mapping` |

Python: `save_zip(path, psf, report=est)` with the estimator that made the PSF (`ZStackEstimator` / `InSituEstimator`:
`est.report`, `est.history`, `est.data`), or `report=` a dict {report, history, data, settings} or an estimator
report dict; without it, `psf.provenance['report']` / `['settings']`.

## Reading a psf.zip

1. Read `psf.json`, check `format` and `version`.
2. Channel types all `source_output`: build the `source` model; it renders all channels (C outputs, `emitter_params`
   extra parameters).
3. All `phase_spline`: stack the knots of the channels to [3 (inc, cr, ci), C, nz, n, n]; evaluate with the formula
   above; `source` (if present) is the vectorial model it was tabulated from.
4. Otherwise each channel is a model of its own: the `model` block is a 0.1.0 PSF JSON of that `type` (with its
   `arrays`); merge the channel's `source.model` keys under it when the reader wants the source parameters at the top
   level (the 0.1.0 spline-with-source layout). Apply `mapping`, `intensity_ratio`, `field` per channel.

Python's `PSF.read_file` performs 2-4 into a 0.1.0 dict (`torch_psf.psfzip.legacy`), so each model class reads a zip
with its existing `_from_file`.

## Example (biplane, two vectorial channels, a SIMCODE poly mapping and field-dependent Zernikes; lists shortened)

```json
{
 "format": "torch-psf", "version": "0.2.0", "psf_type": "channels", "n_dims": 3,
 "emitter_params": ["x", "y", "z", "I", "bg"], "per_channel_params": ["bg"],
 "units": {"x": "px", "y": "px", "z": "um", "I": "photons", "bg": "photons / px", "phi": "rad"},
 "optics": {"pixel_um": 0.10833, "NA": 1.49, "wavelength_um": 0.68, "n_imm": 1.518, "n_med": 1.33},
 "provenance": {},
 "channels": [
  {"name": "plane0", "type": "vectorial",
   "model": {"optics": {"...": "..."}, "zernike_noll": [5, 6, 7], "zernike_waves": [0.0, 0.1, 0.0],
             "apod": [0, 0, 0], "saf": [0, 0, 0, 0], "focus_um": 0.3, "psf_blur_px2": null, "psf_dipole_M": null,
             "diattenuation": null, "npupil": 24, "oversample": 1, "roisize": 13, "z_range_um": [0.0, 0.6],
             "pixelsize_um": 0.10833, "wavelength_um": 0.68, "NA": 1.49, "n_imm": 1.518, "n_med": 1.33},
   "arrays": {"pupil_phase": "plane0/pupil_phase.npy"},
   "mapping": {"type": "identity"}, "intensity_ratio": 1.0, "crop": {"x0": 0, "y0": 0, "w": 128, "h": 120},
   "field": {"type": "zernike_poly", "order": 2, "zernike_noll": [6, 11],
             "coef": [[0.1, -0.02, 0.0, 0.03, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0, 0.0, 0.01]],
             "feats_norm": {"cx": 64.0, "cy": 60.0, "sx": 64.0, "sy": 60.0}}},
  {"name": "plane1", "type": "vectorial", "model": {"focus_um": 0.6, "...": "..."}, "arrays": {},
   "mapping": {"type": "poly", "order": 2, "fwd": {"1": [[130.2, -1.5], "..."]}, "inv": {"1": [[130.0, 1.5], "..."]},
               "feats_norm": {"cx": 64.0, "cy": 60.0, "sx": 64.0, "sy": 60.0}, "n_pairs": 5000},
   "intensity_ratio": 0.9, "crop": {"x0": 128, "y0": 0, "w": 128, "h": 120}, "flip": "lr"}
 ],
 "zstack": {"file": "zstack.tif", "axes": "ZCYX", "shape": [25, 2, 13, 13], "dtype": "float32",
            "z_um": [0.0, 0.025, "..."], "sample_px": 1.0, "sample_um": 0.10833, "channels": ["plane0", "plane1"],
            "position": {"x": 0.0, "y": 0.0}, "params": {}, "value": "..."},
 "preview": {"file": "preview.png"},
 "report": {"file": "fit_report.pdf", "sections": ["summary", "aberrations", "pupil", "psf", "field"]}
}
```

A 4Pi spline: `"psf_type": "fourpi_spline"`, `"emitter_params": [..., "phi"]`, channels
`{"name": "ch0", "type": "phase_spline", "phase_param": "phi", "model": {"spline": {...}, "tau": 0.25}, "arrays":
{"inc": "ch0/inc.npy", "cr": "ch0/cr.npy", "ci": "ch0/ci.npy"}, ...}` x C, and `"source": {"type": "fourpi",
"model": {"optics", "roisize", "z_range_um", "psf_dipole_M", "fourpi": {...}}, "arrays": {}}`.

## Web app

The TypeScript writer / reader (`web/src/io/psfzip.ts`, `zip.ts`, `imagejtif.ts`; `web/src/psf/makezip.ts`) writes a
one-channel psf.zip (vectorial, or spline with the vectorial model as `source`) from the bead and in-situ fits, with
zstack.tif, preview.png and the client-side fit_report.pdf, and reads channel 0 (vectorial, or the `source` of a
spline channel) of any psf.zip. Not read yet: pupil maps, blur, dipole terms, mappings, field dependence (listed in
the import's info line). Cross-checked against torch_psf in `tests/test_web_psfzip.py` / `web/test/test_psfzip.ts`.

## Open points

- Zernike error bars: no estimator reports Zernike standard errors yet; the report draws them from
  `report['zernike_std']` (one per `zernike_noll` entry) when present.
- Field dependence `grid` and the field dependence of coupled (4Pi) models are specified but not implemented; the
  `zernike_poly` field and the identity / poly / grid mappings are (Python writer: a grid mapping's `fwd` / `inv` may
  be given as arrays; they are stored as `<channel>/mapping_fwd.npy` / `mapping_inv.npy`).
- The fitting code does not use mappings, intensity ratios or field dependence yet: they are read and written.
