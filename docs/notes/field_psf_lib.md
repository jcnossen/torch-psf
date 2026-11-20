# Field-dependent PSFs in the library, stage 2a: single channel, joint estimation, validation

Status: single channel, Python. Code: `torch_psf/field.py` (FieldZernikes, FieldPSF, TiledSplinePSF),
`torch_psf/estimation/field_knots.py` (fit_field_knots), `torch_psf/matched_filter.py` (MatchedFilterDetector), the
`field=` options of `fit_dense_beads` / `fit_phase_diversity` and of the three estimators; tests
`tests/test_field.py`; study scripts `study/fieldpsf2/`; outputs `~/fieldpsf2_out/` (not in the repo). Stage 1 (the
FD-DeepLoc measurements this builds on): `docs/notes/field_psf_study.md`.

## Summary

- **Library**: `torch_psf.field` (FieldZernikes: per-mode polynomial of the field position, psf.zip PolyMap
  features / feats_norm, clipped to the bounding box of the training positions, ridge prior, read / write of the
  unchanged psf.zip `zernike_poly` block; FieldPSF: the vectorial model with per-emitter Zernike offsets;
  TiledSplinePSF: one spline per tile, picked by spot position), `estimation.fit_field_knots` (the knot route per
  field tile, tile VJPs summed into the polynomial), `field=` in fit_dense_beads / fit_phase_diversity and in the
  three estimators (a held-out comparison global vs field built in), `MatchedFilterDetector` and `zstarts=` (stage-1
  recommendations). `PSF.load` / `PSFZip.read` return the field model.
- **FD-DeepLoc movie (in situ)**, held-out NLL per spot vs the global in-situ PSF (45k spots, stage-1 scoring):
  field polynomial **-14.97** (66 coefficients, astigmatism / spherical quadratic, the rest linear; 134 px tiled
  spline) and **-15.24** (105 coefficients, all quadratic) vs FD-DeepLoc's aber_map -5.8 / -7.5 (4 x 4 / 6 x 6 tile
  means), stage-1 per-tile PSFs -14.9 (4 x 4, 336 coefficients) / -16.1 (6 x 6, 756). The polynomial is 2x better
  than the map, equal to or better than the 4 x 4 tiles and within 0.8 - 1.1 per spot (93 - 95 %) of the 6 x 6 tiles;
  better than the global PSF in every tile. Its maps follow the tile fits (r 0.8 - 0.95 for the strongly varying
  modes) and agree better with the aber_map than the tiles do for coma.
- **2-colour beads, left channel**: held-out deviance per pixel on the beads2 clusters 1.338 -> **1.299** (dense
  fit) and 1.336 -> **1.297** (phase diversity) with the field polynomial (35 coefficients, NA / apod fitted
  jointly), better in 44 / 52 and 48 / 52 clusters; the polynomial reproduces the per-patch PD pupils of both stacks
  better than the stacks reproduce each other (Z5 r 0.88 - 0.90, Z6 0.72 - 0.74).
- **Simulation**: known linear fields are recovered (in situ to < 1 mλ with 16k spots; dense beads < 6 mλ; tests).
- Open: psf.zip has no rule for the support (the reader clips; spec sentence proposed), the in-situ tile
  approximation, and the remaining gap to the 6 x 6 tiles (section 6). Stage 2b design: section 7.

## 1. The model

```python
from torch_psf import FieldZernikes, FieldPSF, io, PSF
field = FieldZernikes.from_positions(noll, order={5: 2, 6: 2, 11: 2, 7: 1}, xy=train_xy, prior_sd=0.05)
psf = FieldPSF(vectorial, field)           # vectorial = the model at the reference point (the box centre)
psf.render(pos, xy=field_xy); psf.at(x, y); psf.fit(rois, corners=corners)
tiled = psf.to_tiled(tile=134, roisize=41, z_range=(0, 6))       # one CubicSplinePSF per tile, built on first use
io.save_zip('psf.zip', psf); PSF.load('psf.zip')                  # FieldPSF again (a spline channel: TiledSplinePSF)
```

- **Polynomial.** For each Noll index j of the field model, with order o_j in {0, 1, 2}:
  `W_j(x, y) = W_j(ref) + sum_{1 <= i + k <= o_j} c_jik (u^i v^k - u_ref^i v_ref^k)`, with the psf.zip PolyMap
  features (u^i v^k, i outer, k inner) and normalisation `u = (x - cx) / sx`, `v = (y - cy) / sy` (camera / channel
  crop px, x = column). The base VectorialPSF's zcoef is the value at the reference point (the box centre by
  default), so the field terms are per-emitter Zernike offsets through `VectorialPSF.forward(zcoef_offset=)`. Modes of
  the base model without a field term stay constant.
- **No extrapolation: bounding box.** u and v are clipped to [-1, 1] before the features are formed; the box is
  feats_norm, which `from_positions` sets to the **bounding box of the training positions** (spots, beads, patch
  centres). Outside it the polynomial is held at its value on the box edge (per coordinate). Not the convex hull:
  the hull is not expressible in the psf.zip block (see Problems), and for these data the box and the hull differ only
  at the corners of the field stop (beads) or not at all (in situ, spots everywhere). `psfzip.FieldZernikePoly` (the
  numpy reader of the block) applies the same clip; with SIMCODE's default feats_norm (the crop centre / half size) the
  clip is the crop, i.e. a no-op inside the image.
- **Ridge prior**: `penalty() = 0.5 sum (c / prior_sd)^2` over the non-constant coefficients, in NLL units (c in
  waves at the box edge; default prior_sd 0.05 waves). A weak Gaussian prior towards the field-constant model; with
  10^4 - 10^5 spots or 10^6 bead pixels it only matters for coefficients the data do not determine.
- **psf.zip** (format unchanged): the channel `field` block `zernike_poly` with `order` = the highest per-mode order,
  `coef [k, nf]` (column 0 = the value at u = v = 0, chosen so that the block equals the model's `zernike_waves` at the
  reference point, so the writer's consistency check passes), `feats_norm`, and `reference_point` only when it is not
  the box centre. Per-mode orders are the zero pattern of `coef` (`FieldZernikes.from_block` infers them). A
  TiledSplinePSF is written as a spline channel (the spline at the reference point) with the vectorial `source` and
  the block; its tile grid goes to `provenance['field_tiles']` (free-form) so that `PSF.load` rebuilds the same tiles.
- **Tiled spline form.** `TiledSplinePSF`: tiles of `tile` px over the field box, per tile a CubicSplinePSF of the
  vectorial model at the tile centre (built lazily); `fit` / `render(xy=)` pick the tile by the spot's field position
  (positions outside the box: the nearest tile). For a 41 px tetrapod ROI one tile spline is 7 MB of knots and takes
  ~0.5 s to build on a 3090; 144 tiles (134 px over 1608 px) are ~1 GB.
- **Localization** (`FieldPSF.fit`, `TiledSplinePSF.fit`, `estimation.localize_field`): every ROI at its own field
  position (the ROI centre, corners + (P - 1) / 2, or given xy); the Datasets keep the input order. PSF.localize of a
  field model works through fit.

## 2. Joint estimation

- **In situ** (`fit_field_knots`, `InSituEstimator(field=)`): the knot route generalised. The field box is cut into
  tiles (`tile` px; sparse tiles merged into the nearest kept one); per iteration and tile t one knot build at
  `base + field(c_t)`, c_t = the tile's mean spot position (exact on average for the linear terms), localization of
  the tile's spots on it (warm started; 30 LM iterations with `zstarts` starts the first time), the knot gradient of
  their NLL and one VJP through the build: g_t = dNLL_t / dzcoef. Then `dNLL / dbase = sum_t g_t` and
  `dNLL / dcoef = sum_t g_t . dfield(c_t) / dcoef` (autograd through the polynomial), plus the prior; Adam on both.
  Cost per iteration: one build + one VJP per tile plus the localizations (36 tiles of 268 px on the FD-DeepLoc movie:
  ~4.5 s per iteration on a shared 3090 for 36k spots). The estimator starts from its global fit (after the overlap
  rejection) and adds the field model (as a tiled spline, `score_tile` px) to the held-out candidates; it returns the
  FieldPSF when it wins. The stage-1 recommendations are in the library: `MatchedFilterDetector` (detection for
  engineered PSFs; `detector=`) and `zstarts=` (used by every multi-start localization of the estimator).
- **Dense beads** (`fit_dense_beads(fit=(..., 'field'), field=, xy=)`, `DenseBeadEstimator(field=)`): every bead
  is rendered with the Zernikes of its position (all its planes), the polynomial coefficients are an Adam group of the
  PSF (held during warmup), the prior is added to the loss. The estimator runs the global fit (NA / apod as
  configured) first and then the field fit from it (base, polynomial and NA / apod jointly), excluding the clusters
  the global fit rejected; both are scored on the held-out clusters by the same refit.
- **Phase diversity** (`fit_phase_diversity(field=, xy=)`, `PhaseDiversityEstimator(field=)`): every patch gets
  the Zernikes at its centre (`VectorialStack(zcoef_offset)`), fitted jointly over all patches (Poisson stage from the
  global pupil, the PSF held for `field_warmup` iterations); held-out patches scored with both.
- **Held-out comparison**: built into each estimator (`report['heldout']` / `report['field']`, `report['chosen']`),
  plus the shared helpers `compare_heldout / heldout_nll / spot_deviance(..., xy=, zstarts=)` (field positions per
  ROI; `compare_heldout` now also gives the per-spot difference to the first model with its standard error).

## 3. Simulation (tests)

`tests/test_field.py`: a linear Z5 (x) / Z6 (y) field of 0.03 - 0.04 waves over 256 x 256 px, 2500 spots, 2 x 2 tiles,
25 iterations from a constant field: coefficients within 0.008 waves (CPU test); with 16000 spots and 80
iterations (study run, GPU) within 0.0004 waves for 2 x 2 and 4 x 4 tiles. Dense beads (GPU test): two simulated
stacks of 160 x 160 px with a linear Z5 / Z6 field, the field fit recovers the Zernikes over the field to < 6 mλ and
beats the constant fit on the held-out stack. InSituEstimator(field=) (GPU test): a simulated movie with Z6 changing
by 0.12 waves across x: the field model is chosen on held-out spots and its coefficient recovered to 0.015 waves.

## 4. FD-DeepLoc movie, in situ

Script `study/fieldpsf2/insitu_field.py` (outputs `~/fieldpsf2_out/json/insitu_*.json`, per-spot NLL
`insitu_*_nll.npz`, logs `~/fieldpsf2_out/logs/`), figures `insitu_maps.py`.

Setup as stage 1 (same spots, optics, camera, 21 modes, 9 z starts, the same checkerboard of 32 px squares of the spot
positions: parity 0 training, parity 1 held out; the stage-1 global and tile PSFs were also trained on parity 0
only). Training: 40000 random parity-0 spots of the whole field, 4334 rejected as overlaps with the global PSF
(median + 3 robust scales), 35666 used. Field model: all 21 modes, order 2 for the astigmatism / spherical families
(Z5, Z6, Z11, Z12, Z13, Z22, Z23, Z24) and 1 for the other 13 (66 coefficients; variant 'linear': all order 1, 42;
'quad': all order 2, 105), prior sd 0.05 waves, the field box = the bounding box of the training spots (the whole
chip). Estimation tiles 268 px (6 x 6, the stage-1 layout), Adam from the stage-1 global fit: 60 iterations lr 2e-3
(5.5 min on a shared 3090) or 160 iterations lr 3e-3 (17 min; 'long'). Held-out: up to 1500 parity-1 spots per 6 x 6
tile, 45489 spots; every candidate scored with the stage-1 scoring (NLL of each spot after its full refit, 9 z
starts; mean difference to the global PSF per spot, standard error over spots). The stage-1 numbers are reproduced on
this held-out set (tile4 -14.9 vs -14.5, tile6 -16.1 vs -16.0, aber4 -5.8 vs -5.5, aber6 -7.5 vs -7.6).

| candidate | parameters | held-out NLL per spot vs global | fraction of spots better |
|---|---|---|---|
| global (stage 1) | 21 | 0 | - |
| aber_map, 4 x 4 tile means | (map) | -5.83 +- 0.11 | 0.61 |
| aber_map, 6 x 6 tile means | (map) | -7.54 +- 0.12 | 0.65 |
| stage-1 tiles 4 x 4 | 16 x 21 = 336 | -14.91 +- 0.12 | 0.83 |
| stage-1 tiles 6 x 6 | 36 x 21 = 756 | **-16.06** +- 0.13 | 0.85 |
| field 'linear' (60 it), 268 px tiled spline | 42 + 21 | -13.00 +- 0.11 | 0.81 |
| field 'mixed' (60 it), 268 px | 66 + 21 | -14.44 +- 0.12 | 0.83 |
| field 'mixed' (60 it), 134 px | 66 + 21 | -14.91 +- 0.12 | 0.84 |
| field 'mixed' long (160 it), 268 px | 66 + 21 | -14.52 +- 0.12 | 0.83 |
| field 'mixed' long (160 it), 134 px | 66 + 21 | **-14.97** +- 0.12 | 0.84 |
| field 'quad' long (160 it, all modes order 2), 134 px | 105 + 21 | **-15.24** +- 0.12 | 0.85 |

Paired differences on the same spots (field 'mixed' long, 134 px tiles): vs aber_map 6 x 6 **-7.44 +- 0.09** (better
for 74 % of the spots), vs aber_map 4 x 4 -9.15 +- 0.09, vs stage-1 tiles 4 x 4 -0.07 +- 0.04 (equal), vs stage-1
tiles 6 x 6 **+1.09 +- 0.04** (the tiles better for 56 % of the spots); 134 px vs 268 px tiles of the same polynomial
-0.45 +- 0.02.

Per tile (held-out spots of each stage-1 tile): the field model beats the global PSF in 16 / 16 (4 x 4) and 36 / 36
(6 x 6) tiles (-2.5 .. -27.7 / -1.6 .. -29.5 per spot) and the aber_map tile means in every tile of both grids; it
beats the stage-1 4 x 4 tile PSFs in 9 / 16 tiles and the 6 x 6 ones in 4 / 36.

Reading: a smooth polynomial with 66 coefficients, fitted jointly to all spots, captures **93 %** of the gain of
the 36 independent tile PSFs (756 coefficients) and all of the gain of the 4 x 4 tiles, and is 2x better than
FD-DeepLoc's map. The remaining 1.1 per spot to the 6 x 6 tiles is variation the polynomial does not have (the
'quad' run, every mode quadratic, recovers a quarter of it: -15.24, i.e. 0.27 per spot better than 'mixed', +0.82
+- 0.04 behind the 6 x 6 tiles and 0.33 ahead of the 4 x 4 tiles, better than the 6 x 6 tile PSF in 5 / 36 tiles and
the 4 x 4 one in 11 / 16), plus the tile approximation of the fit itself
(268 px estimation tiles; the scoring at 134 px gains 0.45 per spot over 268 px with the same polynomial, so the field
varies within a 268 px tile by enough to matter). 60 vs 160 iterations changes the score by 0.06 per spot.

**Maps** (`~/fieldpsf2_out/figs/field_maps_mixed_long.png`: per mode the polynomial with the stage-1 6 x 6 tile values
as squares, and the aber_map, each minus its field mean; `field_vs_tiles_mixed_long.png`: scatter;
`heldout_mixed_long.png`: per-tile held-out map; table `json/field_maps_mixed_long.json`). Correlation over the 36
tiles of the polynomial at the tile centres with the tile fits, and with the aber_map tile means (stage 1: tiles vs
map):

| mode | order | r(field, tiles) | r(field, aber_map) | r(tiles, aber_map) | std over tiles: field / tiles / map (mλ) |
|---|---|---|---|---|---|
| Z5 astig 45 | 2 | +0.71 | +0.50 | +0.51 | 13 / 25 / 42 |
| Z6 astig 0 | 2 | **+0.95** | +0.56 | +0.48 | 37 / 39 / 30 |
| Z7 coma y | 1 | +0.62 | **+0.85** | +0.54 | 26 / 27 / 55 |
| Z8 coma x | 1 | +0.58 | +0.70 | +0.44 | 21 / 22 / 58 |
| Z11 spherical | 2 | +0.87 | +0.51 | +0.52 | 10 / 14 / 33 |
| Z9 trefoil y | 1 | +0.80 | +0.35 | +0.20 | 28 / 30 / 28 |
| Z10 trefoil x | 1 | +0.85 | +0.27 | +0.02 | 31 / 32 / 42 |
| Z13 2nd astig 45 | 2 | +0.31 | +0.08 | +0.13 | 8 / 11 / 29 |
| Z12 2nd astig 0 | 2 | +0.55 | +0.10 | -0.36 | 5 / 9 / 17 |
| Z17 2nd coma y | 1 | +0.81 | +0.65 | +0.67 | 12 / 19 / 22 |
| Z16 2nd coma x | 1 | +0.81 | +0.60 | +0.75 | 12 / 15 / 30 |
| Z22 2nd spherical | 2 | +0.45 | +0.31 | +0.31 | 11 / 14 / 22 |
| Z15 quadrafoil 45 | 1 | +0.12 | +0.28 | +0.29 | 2 / 10 / 15 |
| Z14 quadrafoil 0 | 1 | +0.32 | -0.46 | -0.44 | 3 / 18 / 8 |
| Z19 2nd trefoil y | 1 | +0.73 | +0.56 | +0.65 | 5 / 11 / 11 |
| Z18 2nd trefoil x | 1 | +0.63 | +0.57 | +0.69 | 7 / 12 / 7 |
| Z23 3rd astig 45 | 2 | +0.83 | -0.16 | -0.11 | 11 / 12 / 14 |
| Z24 3rd astig 0 | 2 | +0.32 | +0.08 | -0.03 | 5 / 6 / 6 |
| Z30 3rd coma x | 1 | +0.62 | +0.74 | +0.72 | 11 / 14 / 15 |
| Z29 3rd coma y | 1 | +0.72 | +0.85 | +0.54 | 13 / 15 / 12 |
| Z37 3rd spherical | 1 | -0.27 | -0.37 | +0.01 | 2 / 7 / 7 |

- The polynomial follows the tile fits where they vary (Z6 0.95, Z11 0.87, Z9 / Z10 0.80 - 0.85, Z16 / Z17 0.81, Z23
  0.83) with a smaller spread for the weak modes (Z14, Z15, Z37: 2 - 3 mλ vs 7 - 18 mλ over the tiles, i.e. the
  per-tile values there are mostly noise, which the joint fit and the prior do not follow).
- For coma (Z7, Z8, Z29) the polynomial agrees **better with FD-DeepLoc's map than the tiles do** (0.85 vs 0.54 for
  Z7): pooling all spots removes the tile noise of the low-order shape. The trefoil-x gradient (Z10) of stage 1 is in
  the polynomial too (r 0.85 with the tiles) and still not in their map (0.27).
- Z5 (astigmatism 45) has twice the spread over the tiles (25 mλ) than the quadratic polynomial (13 mλ): its tile
  pattern has structure beyond order 2 (the aber_map's Z5 also varies most at the corners).

## 5. 2-colour beads, left channel (dense beads and phase diversity)

Script `study/fieldpsf2/beads_field.py` (outputs `~/fieldpsf2_out/beads/` (dense) and `~/fieldpsf2_out/beads_pd/`
(phase diversity): `beads_field.json`, the PSFs as psf.zip), maps `beads_maps.py`. Data and assumptions of
`dense_beads.md` (97 nm px, NA 1.49 start, n_imm 1.518, n_med 1.33, 670 nm, 0.597 e-/ADU, offset 100.6, x < 475,
beads.tif training, beads2.tif held out, every 2nd plane for the dense fit). Each run: the estimator with
`fit_na=True, fit_apod=True` started from the global NA + apod PSF of the phase-diversity study (dense:
`psf_dense_fit_na_apod.json`, PD: `psf_pd_fit_na_apod_long.json`), so the global fit is first refitted (300
iterations, the same as the field fit gets afterwards: the comparison has matched iterations), then the field fit
from it: base Zernikes, polynomial, NA and apod jointly (300 iterations; PD: the Poisson stage, PSF held 40). Field
box = the bounding box of the training beads (dense) / patch centres (PD). Score: the 52 held-out beads2 clusters
(101 beads, the aggregates left out) by `dense_nll` exactly as `study/phase_diversity/common.py` (each cluster's
positions, photons, background, focus offsets and drift refitted with the PSF fixed; every bead at its own field
position for the field model). Orders: 'mixed' = order 2 for Z5, Z6, Z11, order 1 for Z7 - Z10, Z12 - Z17, 0 for
Z18 - Z21 (35 coefficients); 'linear' = all 17 modes order 1 (34); 'quad' = all order 2 (85); 'astig' = only Z5, Z6,
Z11 order 2 (15). Prior sd 0.05 waves.

| method | field orders | held-out deviance / px: global -> field | NLL / px field - global | clusters better | NA global / field | run (s) |
|---|---|---|---|---|---|---|
| dense | mixed (35) | 1.3384 -> **1.2992** | -0.0196 | 44 / 52 | 1.400 / 1.401 | 357 |
| dense | linear (34) | 1.3384 -> 1.3048 | -0.0168 | 44 / 52 | 1.399 / 1.402 | 282 |
| dense | quad (85) | 1.3384 -> 1.2999 | -0.0192 | 41 / 52 | 1.400 / 1.401 | 399 |
| dense | astig (15) | 1.3384 -> 1.3037 | -0.0173 | 44 / 52 | 1.400 / 1.401 | 337 |
| phase diversity | mixed (35) | 1.3363 -> **1.2969** | -0.0197 | 48 / 52 | 1.410 / 1.409 | 348 |
| phase diversity | linear (34) | 1.3363 -> 1.3022 | -0.0170 | 47 / 52 | 1.410 / 1.409 | 417 |

(The global rows reproduce the earlier global NA + apod numbers, 1.3375 dense and 1.3395 phase diversity, within the
refit; run = the whole estimator, both fits and its own held-out scoring, on a shared GPU.)

- **The field model lowers the held-out deviance from 1.34 to 1.30** in both routes (-0.020 NLL per pixel), better in
  44 - 48 of the 52 held-out clusters; for comparison, the NA / apod fit was worth 0.10 (1.44 -> 1.34) and the
  per-patch PD pupils of the phase-diversity study did not improve the held-out score at all (1.340 local vs 1.329
  global). The gain is largest at the left and bottom edges (`figs/beads_heldout_clusters.png`), where the
  aberrations change most.
- Quadratic astigmatism / spherical carry most of it (astig: 15 coefficients, -0.0173); mixed adds the linear terms
  of the other modes (-0.0196); all-quadratic adds nothing (-0.0192). Linear only: -0.0168.
- Dense and phase diversity agree: the same held-out gain and nearly the same polynomial (maps below). NA / apod hardly
  move when the field terms are added (NA 1.400 -> 1.401).
- The estimators' own held-out numbers (all beads2 clusters, aggregates included) go 1.544 -> 1.512 (dense) and the
  PD patch deviance 1.126 -> 1.117; `report['field']['kept']` is True in every run.

**Maps** (`~/fieldpsf2_out/figs/beads_field_maps.png`: the polynomial of dense and PD (mixed) per mode, with the
per-patch PD pupils of the phase-diversity study, both stacks; `json/beads_maps.json`). Correlation of the polynomial
at the patch centres with the per-patch values of beads.tif (training stack) / beads2.tif (an independent check),
and the range over the field:

| mode | per-patch r, stack 1 vs stack 2 | dense: r stack 1 / 2, range mλ | PD: r stack 1 / 2, range mλ |
|---|---|---|---|
| Z5 | +0.81 | +0.90 / +0.88, -56 .. +6 | +0.90 / +0.88, -60 .. +7 |
| Z6 | +0.56 | +0.74 / +0.72, +57 .. +105 | +0.74 / +0.72, +64 .. +107 |
| Z7 | +0.16 | +0.07 / +0.06, -36 .. -32 | +0.13 / +0.05, -40 .. -34 |
| Z8 | -0.12 | +0.44 / +0.00, -10 .. -4 | +0.47 / -0.02, -10 .. -2 |
| Z11 | +0.27 | +0.38 / +0.08, -13 .. +1 | +0.29 / -0.14, -14 .. -4 |
| Z12 | -0.00 | +0.29 / +0.33, -3 .. +9 | +0.25 / +0.26, +1 .. +7 |
| Z13 | +0.60 | +0.65 / +0.66, -12 .. +4 | +0.64 / +0.70, -15 .. +6 |

- The polynomials reproduce the per-patch pattern better than the two stacks reproduce each other (Z5 0.88 - 0.90 vs
  0.81, Z6 0.72 - 0.74 vs 0.56, Z13 0.65 - 0.70 vs 0.60): Z5 from about 0 (top, right) to -56 .. -60 mλ (bottom
  left), Z6 from ~105 (top left) to ~60 mλ (bottom right), as in the regional tables of `dense_beads.md` and
  `phase_diversity.md`. Coma Z7 is flat (range 4 - 6 mλ, no per-patch correlation, as the patches said). For Z8,
  Z11, Z12 the field range (5 - 14 mλ) is near the per-patch noise; the two routes agree on their sign / gradient
  direction except Z11 (dense: top left high; PD: bottom right low), i.e. these are weakly determined.


## 6. Problems, limits, open questions

- **psf.zip spec gaps (format not changed; to decide):**
  1. *Support / extrapolation.* The `zernike_poly` block has no key for the fitted support and the spec says the
     polynomial is evaluated at (x, y) without a rule outside it. The Python reader and writer now clip u, v to
     [-1, 1] (the feats_norm box) and the writer puts the training bounding box there; another reader that follows
     the spec literally extrapolates beyond the box. Proposal: one sentence in PSF_FORMAT.md ("evaluate with u, v
     clipped to [-1, 1]"), or an optional `support` key (box or hull polygon).
  2. *Per-mode order* is implicit (the zero pattern of coef). Readable, but a writer cannot say "order 2 with a
     zero coefficient" vs "order 1"; harmless for evaluation.
  3. *Tile grid of a tiled spline* has no place in the field block; it goes to `provenance['field_tiles']`
     (free-form). It only affects how a reader rebuilds the fast form, not the model.
- **Tile approximation in the in-situ fit.** Spots of a 268 px tile are fitted with the PSF at the tile's mean
  position. The scores show the field changes within such a tile by enough to matter (134 px scoring tiles: -0.45
  per spot over 268 px with the same polynomial), so the fit itself is slightly biased towards a smoother field.
  Smaller estimation tiles cost one build + VJP each per iteration (144 tiles ~4x the 36-tile cost); a first-order
  correction (a spline basis per polynomial term, the stage-1 option K0 + sum p_t dK/dc_t) would remove it.
- **The 6 x 6 tiles are still 1.1 NLL per spot better than the polynomial** on the FD-DeepLoc movie (field 134 px
  vs tile6; 0.8 with every mode quadratic). Candidates: structure beyond order 2 in a few modes (Z5's tile spread is twice the
  polynomial's), the tile approximation above (`figs/heldout_mixed_long.png`: almost all of the gap is one corner, the
  top-right 6 x 6 tile: field -3, tile -18 per spot; elsewhere the two agree within a few per spot), and spot-to-spot differences that are not field position (z / depth
  dependent aberrations, sample regions) that independent tiles absorb.
- **Per-bead focus offsets vs field-dependent defocus / spherical**: the dense fit keeps a free focus offset per bead;
  a field-dependent Z11 trades with them weakly (dense and PD disagree on the small Z11 gradient). A tilt plane plus
  regularised per-bead offsets (dense_beads.md) is still open.
- **NaN at the critical angle** (fixed, commit 1faec38): a VectorialPSF whose NA put a pupil sample exactly at
  NA rho = n_med returned NaN images (met by a dense field fit ending at NA 1.401259). Any NA fit could hit it.
- **Spot memory**: spot_deviance evaluated all ROIs at once (8 GB for 40k 41 px ROIs); now in batches.
- **CPU test time** depends heavily on the machine load: the three heavier CPU tests of test_field.py take 17 - 18 s
  each (whole file 53 s, load ~30 on 20 cores) but took 41 - 73 s at a load of ~40 (the user's jobs). They are kept
  on CPU; mark them needs_gpu if CI turns out slower.
- Not done: FD-DeepLoc bead stacks with field= (stage 1 has per-bead fits; the movie is the in-situ validation); a
  field model for NA / apod (global here; the per-patch NA / apod of the PD study were noise); refine=True (free
  knots) per tile.


## 7. Design for stage 2b: several channels (the 2- and 3-colour data)

Not built; what stage 2a leaves in place for it, and the plan.

**Coordinates.** One emitter (bead) has a position in the reference channel 0, (x0, y0) px. Channel c sees it at
`(x_c, y_c) = M_c^-1(x0, y0)` (`psfzip.Mapping.from_ref`; identity / affine / SIMCODE PolyMap / grid, already read and
written), its PSF there is channel c's field model at (x_c, y_c) in **channel coordinates** (the zernike_poly block of
channel c, feats_norm in channel c's crop, exactly what psf.zip specifies). z is shared (a focus offset per channel
lives in the channel models). Photons: `I_c = intensity_ratio_c I` for one dye seen in several channels (biplane,
ratiometric colour), or free per channel for beads (each bead emits into both colour channels with its own ratio).

**Model classes** (`torch_psf/multichannel.py`):
- `ChannelsPSF([FieldPSF or TiledSplinePSF per channel], mappings=[Mapping per channel])`: `render_channels(pos_c,
  extra, grad)` needs each channel's field position; the per-spot offsets that MultiChannelPSF already takes
  (`offsets [N, C, 2]`) become (ROI corner of channel c + centre) + offset, so a `field_xy [N, C, 2]` argument (or a
  bound copy, as `FieldPSF.bind` here) carries it. `PSFZip.read(path).multichannel()` would then build it from the
  channels' `Channel.psf` (the field models stage 2a returns) plus their mappings, instead of the field-constant
  models.
- ROIs per channel: cut around `from_ref(detection)` in each channel; the fractional part of the mapped position is
  the per-spot offset (fit(..., offsets=) exists). A TiledSplinePSF per channel keeps the localization at spline
  speed; the tile is picked per channel from (x_c, y_c).

**Joint estimation from beads** (the user's 2-colour stack, both channels):
- Cluster detection in channel 0 (or in both, matched through the mapping); a cluster's pixels in channel c are the
  union of its members' boxes at `from_ref(x_b, y_b)`, in channel c's own field stop (`field_stop_mask` per channel).
  Clusters that leave one channel's field stop are fitted in the other channel only (a bead mask per channel).
- Shared per bead: (x0, y0) and the focus offset (bead height); per channel: photons per plane (or bead x plane
  factor), background per cluster and plane, the drift if the channels are separate cameras (else shared), the field
  model (FieldZernikes in channel coordinates; base Zernikes, NA / apod per channel since the paths differ).
- The mapping: start from a registration of the bead detections (affine, then PolyMap order 2-3, as SIMCODE does), and
  either keep it fixed or fit its coefficients jointly (they are a few parameters with a strong signal: every bead in
  both channels). Fitting it jointly matters because a mapping error shows up as a field-dependent lateral offset that
  the per-channel field models cannot absorb (they have no tilt terms).
- Implementation: `fit_dense_beads` already scatters each bead's render box into its cluster's pixel list; the
  channel becomes another index of the flat pixel list (pixels of channel c of cluster k), each bead is rendered once
  per channel with that channel's VectorialPSF and Zernike offsets, so one batched evaluation per channel. Phase
  diversity per channel is independent (no shared object between colour channels unless the object is mapped: a
  second step).
- In situ: the same emitter in C channels (biplane / ratiometric): `fit_field_knots` per channel with shared emitter
  parameters, i.e. the localization step becomes MultiChannelPSF.fit_theta on the C tile splines of the spot, the
  knot gradient per channel and tile, one VJP per channel and tile.

**psf.zip**: nothing new is needed for independent channels with a mapping and a zernike_poly field each (both are
specified); stage 2b writes one zip per setup with C channels, each FieldPSF unwrapped as in stage 2a. The open points
of section 6 (support, tiles) apply per channel.

