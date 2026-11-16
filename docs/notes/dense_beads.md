# PSF from dense bead stacks (overlapping beads)

Status: step 1 (single channel, field-constant PSF), Python only. Code: `torch_psf/estimation/dense_beads.py`,
`DenseBeadEstimator` in `torch_psf/estimators.py`, tests `tests/test_dense_beads.py`, study scripts
`study/dense_beads/`.

## Why

`ZStackEstimator` cuts one ROI per bead and needs beads far apart. Its `find_beads` rule keeps the brightest peak and
skips fainter peaks within `roisize`, but it does not check that the kept bead's ROI is free of neighbours: in a dense
field the ROIs contain parts of other beads, and with a strict isolation rule only a few beads (and a biased
selection: the sparse corners of the field) remain. In the user's 2-colour bead stacks ~70 % of the beads have a
neighbour within the PSF extent.

## Method

```python
from torch_psf import DenseBeadEstimator, Camera
est = DenseBeadEstimator(optics=dict(pixel_um=0.097, NA=1.49, wavelength_um=0.67), camera=Camera(...),
                         region=(0, 0, 475, 400), holdout=1, focus_offset='bead', reject=3.0)
psf = est.estimate(['beads.tif', 'beads2.tif'], dz_um=0.01)      # VectorialPSF; est.report, est.data, est.result
```

1. **Detection** (`find_all_beads`): local maxima (>= `min_dist` = 3 px apart) of the maximum projection over z of the
   blurred planes, minus a local background (grey opening), above `threshold` (6) robust sd.
2. **Grouping** (`find_bead_clusters`): beads whose `patch x patch` boxes overlap form a cluster (connected
   components). `patch` = `psf_extent`: the smallest box whose border stays below 2 % of the brightest in-focus peak at
   every scanned plane (an energy criterion does not work for beads on the coverslip with NA 1.49: the supercritical
   light spreads into a wide faint ring when defocused; at 0.75 um defocus a 41 px box still misses ~20 % of the
   light). Detections fainter than `faint` (0.1) x the median of the brightest 20 % (debris, specks, noise) do not
   take part in the grouping (they would chain everything into one cluster); a faint one inside a member box joins
   that cluster as a modelled member that does not add pixels. A cluster is dropped when a member box leaves the frame
   or the field stop (`field_stop_mask`: Otsu split of the blurred mean image, largest bright component, eroded
   2 px; off when there is no dark border region), has more than `max_members` (8) bright members or a box side above
   `max_box` (3 patch).
3. **Model** (`fit_dense_beads`), per cluster c and plane k on the pixels of the union of its members' boxes:
   `mu = bg_ck(x, y) + sum_b I_bk PSF(x - x_b - drift_s(k), y - y_b - drift_s(k), z_e; focus_k * scale + df_b)`:
   one shared `VectorialPSF` (Zernikes; optionally NA / n_med), per bead x, y and photons per plane (or per bead
   x per plane factor), per cluster and plane a constant or linear (`bg='linear'`) background, per stack a cubic
   lateral drift over the planes, and a focus offset per bead (`'bead'`), per stack (`'stack'`) or as a tilted plane
   per stack (`'tilt'`), optional z-step scale. Every bead is rendered in its own `R = patch + 2 render_pad` box
   (default `render_pad = patch // 3`) around its starting pixel; the box pixels are scattered (index_add) into the
   flat pixel list of the cluster, render-box pixels outside the cluster's pixels go to a sink. So the tails of a
   member on its neighbours' pixels are modelled, and clusters of any shape share one batched PSF evaluation.
   Poisson likelihood, Adam (the PSF fixed for the first `warmup` = 50 iterations, cosine decay of all learning
   rates over the last 30 %), gradients accumulated over chunks of clusters (`max_renders` bead-planes per pass).
   `reject`: at iters / 3, clusters with chi2 > reject x median leave the fit (aggregates, debris).
4. **Held-out check**: clusters with `train=False` (a whole stack: `holdout=1`, or `holdout_fraction`) are refitted
   after the fit with the PSF fixed (own positions, photons, background; drift / focus offsets of their stack) and
   scored (`heldout_nll_per_px`, `heldout_deviance_per_px`). `dense_nll(data, focus, psf)` scores any PSF the same way,
   e.g. the isolated-bead estimate on held-out clusters. The deviance per pixel is ~1 for a model that explains the
   data (Poisson), so it is readable on its own; NLL differences between PSFs on the same pixels are what count.

Defaults: `bg='linear'`, `focus_offset='tilt'`, `intensity='free'`, `drift_order=3`, `iters=300`, `warmup=50`,
`render_pad=patch // 3`. Cost: one VectorialPSF evaluation per bead and plane per iteration (as fit_beads); 92 beads x
75 planes x 300 iterations take ~5 min on a shared RTX 3090.

## Simulation

`study/dense_beads/sim_compare.py` (`run_sim.sh`, `run_sim2.sh`, `sim_summary.py`): 320 x 320 px (0.1 um px, NA 1.49,
670 nm), 51 planes over 1.5 um, 9 Zernikes up to 60 mλ, beads in clusters (neighbours 1-3 um apart in 'mixed' /
'dense', 0.8-1.7 um in 'packed'), log-normal photons, background 40, a coverslip tilt (0.1 um across the field, x) and
a lateral drift (0.4 px linear, -0.3 px quadratic). Every bead is rendered in a 41 px box (tails included). Three seeds
per scenario. Fits: `iso` = ZStackEstimator (roisize = patch, nbeads = 64: every bead its find_beads rule accepts),
`single` = the joint fitter on the strictly isolated training beads (clusters of one), `dense` = DenseBeadEstimator
(a quarter of the clusters held out). Held-out deviance per pixel of each PSF on the held-out clusters minus that of
the true PSF (each cluster refitted with the PSF fixed).

| scenario | method | beads used (mean) | Zernike rms error mλ (mean, per seed) | held-out deviance / px - truth |
|---|---|---|---|---|
| mixed, tilt 0.1 µm, 30000 ph | iso | 34.7 | 1.31 (2.01, 0.95, 0.96) | +0.0271 |
| mixed, tilt 0.1 µm, 30000 ph | single | 15.7 | 0.13 (0.13, 0.13, 0.11) | +0.0001 |
| mixed, tilt 0.1 µm, 30000 ph | dense | 37.0 | 0.11 (0.11, 0.11, 0.11) | +0.0001 |
| dense, tilt 0.1 µm, 30000 ph | iso | 33.0 | 0.80 (0.68, 1.05, 0.69) | +0.0114 |
| dense, tilt 0.1 µm, 30000 ph | single | 11.7 | 0.16 (0.18, 0.17, 0.13) | +0.0004 |
| dense, tilt 0.1 µm, 30000 ph | dense | 35.3 | 0.12 (0.13, 0.12, 0.11) | +0.0002 |
| dense, tilt 0.0 µm, 30000 ph | iso | 33.0 | 0.78 (0.67, 1.05, 0.62) | +0.0113 |
| dense, tilt 0.0 µm, 30000 ph | single | 11.3 | 0.15 (0.16, 0.16, 0.14) | +0.0001 |
| dense, tilt 0.0 µm, 30000 ph | dense | 36.3 | 0.11 (0.10, 0.11, 0.11) | +0.0001 |
| dense, tilt 0.1 µm, 5000 ph | iso | 33.0 | 0.79 (0.64, 1.18, 0.55) | +0.0008 |
| dense, tilt 0.1 µm, 5000 ph | single | 11.0 | 0.32 (0.34, 0.32, 0.29) | +0.0001 |
| dense, tilt 0.1 µm, 5000 ph | dense | 37.0 | 0.19 (0.21, 0.16, 0.21) | +0.0000 |
| packed, tilt 0.1 µm, 30000 ph | iso | 20.0 | 3.09 (1.90, 5.26, 2.10) | +0.1812 |
| packed, tilt 0.1 µm, 30000 ph | single | 2.0 | 0.21 (0.15, 0.31, 0.17) | +0.0008 |
| packed, tilt 0.1 µm, 30000 ph | dense | 37.0 | 0.11 (0.09, 0.14, 0.10) | -0.0001 |
| packed, tilt 0.1 µm, 5000 ph | iso | 20.0 | 2.63 (1.56, 4.63, 1.71) | +0.0088 |
| packed, tilt 0.1 µm, 5000 ph | single | 2.0 | 0.84 (0.77, 1.28, 0.47) | +0.0008 |
| packed, tilt 0.1 µm, 5000 ph | dense | 37.3 | 0.16 (0.11, 0.20, 0.17) | +0.0000 |

(`beads used`: iso = all beads ZStackEstimator picks; single / dense = training beads, a quarter of the
clusters held out.)

- The dense fit recovers the Zernikes to 0.11-0.12 mλ rms at 30000 photons and 0.16-0.19 mλ at 5000 photons per
  bead, and scores as well as the true PSF on held-out clusters (difference <= 0.0002 deviance / px).
- ZStackEstimator is biased by neighbours inside its ROIs: 0.8-1.3 mλ in the mixed / dense layouts, 2.6-3.1 mλ in the
  packed one, the same at 5000 and 30000 photons (bias, not noise), and its held-out deviance is clearly above the
  truth's (+0.01 to +0.18 / px). It has no focus offset per bead, but the coverslip tilt hardly matters here (0.80 vs
  0.78 mλ with and without).
- The strictly isolated beads with the joint fitter are unbiased; with 10-15 bright isolated beads they come within
  15-35 % of the dense fit. When the isolated beads are few and dim (packed layout, 2 isolated beads, 5000 photons) the
  dense fit is 5x better (0.16 vs 0.84 mλ).
- The first version (focus-offset / drift learning rate 2e-3) gave ~0.21 mλ at 30000 photons: the drift and tilt
  terms had not converged (see Problems).
- Figure: `~/densebeads_out/sim_summary.png`.
- Unit test (`tests/test_dense_beads.py`, CPU, 96 x 96 px, 8 beads in a single, two pairs and a triplet, 21 planes):
  dense 0.32 mλ rms (8 beads) vs ZStackEstimator 3.5 mλ (4 beads); the CPU run of the 4 tests takes ~4.5 min.

## Real data: 2-colour astigmatic beads, left channel

Data: `/mnt/data/multicolor/june/astig/2col/beads100nm_10nm_steps/beads.tif` (training) and `beads2.tif` (a
different field, held out), 150 planes at 10 nm, 400 x 950 px, left channel = x < 475 behind a polygonal field stop.
Run: `study/dense_beads/run_real.sh` (`real_fit.py`, `real_variants.py`, figures `fig_real.py`, `fig_detect.py`;
output `~/densebeads_out/real/`). Every 2nd plane is used (75 planes, 20 nm) for speed; patch 21 px (psf_extent of
the start model), render_pad 7, `focus_offset='bead'`, `bg='linear'`, `reject=3`, `max_members=10`, `max_box=84`.

**Assumptions** (not in the metadata): pixel 97 nm (the DECODE configs of the same setup, `june/astig/3col/...` and
`june/9-6/...`, use 97 nm; the SMAP calibration `beads_left_3dcal.mat` says 100 nm, its default; dz 10 nm and the
focus at plane ~74 agree with it); NA 1.49 oil (n_imm 1.518); emission 670 nm (a far-red 2-colour set; no dye or
filter information found); beads in water (n_med 1.33) on the coverslip (emitter_z 0; 50 nm tried as a variant);
camera (`camera_calib.pickle` not opened): sCMOS-like, offset 100.6 ADU and 0.597 e-/ADU (Poisson-equivalent) from a
photon-transfer line of this stack (variance of adjacent-plane differences vs mean, 30-plane blocks, background
pixels; `study/dense_beads/camera_ptc.py`); read noise neglected (~1.4 e- vs a background of 30-60 photons / px).

**Detection / grouping**: beads.tif 244 detections (~120 bright beads, the rest faint debris and specks; the
amplitudes are bimodal), 48 clusters with 105 beads (13 faint members) kept, 19 beads dropped (field stop, frame
edge); 28 clusters are single beads. beads2.tif: 54 clusters, 105 beads. Two training clusters were rejected (chi2 > 3x
median: chi2 6 and 17, a pair and a single bead with debris) and 2 held-out clusters are not scored (chi2 above the
same limit: the large aggregate at (315, 360), chi2 19, and one bead at (185, 345), chi2 4.7): 92 training beads in
46 clusters; 101 held-out beads in 52 clusters. ZStackEstimator (nbeads=8 / 40 per stack) uses 8 / 78 beads over both
stacks (its own held-out score is on the beads it picks in beads2.tif; not used here); the strictly isolated set
(singletons of beads.tif) has 27 beads.

**Zernikes** (mλ rms) and held-out score on the 52 beads2 clusters (Poisson deviance per pixel and NLL per pixel
relative to dense; each cluster refitted with the PSF fixed):

| | ZStackEstimator nbeads=8 | ZStackEstimator nbeads=40 | strictly isolated (joint fitter) | dense (all clusters) | start (no aberrations) |
|---|---|---|---|---|---|
| training beads (beads.tif) | 8 | 40 (78 over both stacks) | 27 | 92 | - |
| Z5 | -13.4 | -10.3 | -20.0 | -10.4 | 0 |
| Z6 | +77.8 | +77.6 | +76.4 | +80.5 | 0 |
| Z7 | -29.9 | -31.3 | -40.2 | -40.6 | 0 |
| Z8 | -9.0 | -2.2 | -8.9 | -5.8 | 0 |
| Z9 | +1.5 | -0.6 | -1.5 | -2.2 | 0 |
| Z10 | -2.7 | -2.8 | -1.8 | -1.8 | 0 |
| Z11 | -2.1 | -5.7 | +0.8 | +1.4 | 0 |
| Z12 | +0.7 | +0.2 | -1.3 | -1.8 | 0 |
| Z13 | -3.1 | -4.2 | -3.2 | -1.9 | 0 |
| Z14 | -2.5 | -3.2 | -4.5 | -4.8 | 0 |
| Z15 | +0.8 | +0.3 | +0.4 | -0.4 | 0 |
| Z16 | +4.4 | +7.8 | +5.8 | +5.5 | 0 |
| Z17 | -2.9 | -2.7 | -8.5 | -9.3 | 0 |
| held-out deviance / px | 1.471 | 1.475 | 1.454 | 1.441 | 2.597 |
| held-out NLL / px - dense | +0.0151 | +0.0172 | +0.0067 | +0.0000 | +0.5779 |
| held-out chi2 median | 1.31 | 1.31 | 1.29 | 1.28 | 2.38 |

(Z18-Z21 are below 2 mλ in all fits.)

- All fits agree on the main terms: astigmatism Z6 ~ +80 mλ, coma Z7 -30..-41 mλ, Z5 -10..-20 mλ; the other fits
  differ from the dense one by 2.7 (strictly isolated) and 3.7 mλ rms (ZStackEstimator). The dense fit is best on the
  held-out clusters, the strictly isolated beads next, ZStackEstimator last (its ROIs hold neighbours: training chi2
  median 3.1 vs 1.5 for the dense fit and 1.4 for the singletons). The differences between the PSF estimates
  (0.013-0.034 deviance per pixel) are small compared with what all of them leave unexplained (deviance 1.44 instead of
  1) and with the field dependence below.
- Figures: `~/densebeads_out/real/detections.png` (detections, kept / dropped clusters, field-stop contour),
  `clusters.png` (data / model / residual of 4 dense clusters at 5 planes), `zernikes.png` (bar chart),
  `regions.png` (regional Zernikes), `deviance.png` (deviance per cluster over the field).

**What the model misses** (residuals, `clusters.png`, and the variants below):

- the defocused PSF of the model has a sharp ring at +-0.74 um that the data do not show (the data are smoother):
  finite bead size, an apodization / lower effective NA, a different wavelength or pixel size, or a z-dependent
  aberration (field-dependent spherical);
- structured residuals at the bead centres around focus (field dependence, below);
- background: the haze varies within a cluster box; a linear model per cluster and plane is the minimum.

**Model variants** (`real_variants.py`; trained on beads.tif, scored by the estimator's own held-out refit on 51
beads2 clusters (one more left out than above, and a constant-learning-rate refit, so the numbers are lower than in
the table above; compare within this table):

| variant | training deviance / px | held-out deviance / px | held-out NLL / px - base | NA | Z6 | Z7 | Z11 |
|---|---|---|---|---|---|---|---|
| as above (bead focus offsets, linear bg, render_pad 7) | 1.697 | 1.393 | +0.0000 | 1.490 | +81 | -41 | +1 |
| + NA fitted | 1.537 | 1.309 | -0.0423 | 1.394 | +82 | -34 | -8 |
| constant bg per cluster-plane | 1.793 | 1.457 | +0.0318 | 1.490 | +81 | -39 | +0 |
| focus_offset='tilt' (plane per stack) | 1.719 | 1.433 | +0.0198 | 1.490 | +81 | -41 | +1 |
| emitter_z 50 nm (bead centre) | 1.612 | 1.350 | -0.0215 | 1.490 | +83 | -41 | +3 |
| render_pad 0 | 1.754 | 1.431 | +0.0186 | 1.490 | +81 | -41 | +1 |

Reading: what lowers the deviance most is less supercritical light in the model: fitting NA (it goes to 1.39; NA, the
wavelength and the pixel size are partly interchangeable here, so read it as "the defocused PSF is less extended than
NA 1.49 / 670 nm / 97 nm predicts") or putting the bead centre 50 nm above the coverslip (the 100 nm bead size). Both
soften the ring at +-0.74 um. Per-bead focus offsets beat a tilt plane per stack (some beads are off the coverslip),
the linear background beats a constant one, and rendering the members' tails (render_pad) helps. The main Zernikes
hardly move (Z6 +81..+83 mλ), except Z7 and Z11 with the NA fit (-34 / -8 mλ), which trade off against the NA. The next
fits should use fit_na (or a bead-size model) before field dependence is added.

**Field dependence**: the clusters of 3 x 3 field regions (both stacks, rejected / aggregate clusters left out) fitted
separately (300 iterations from the global dense PSF), and the global PSF scored on the same clusters (in-sample, so
the local fit always has the advantage of its extra parameters; the size of the gain is what matters):

| region (col, row) | beads | Z5 | Z6 | Z7 | Z8 | Z11 | deviance local / global | rms vs global (mλ) |
|---|---|---|---|---|---|---|---|---|
| (0, 0) | 20 | -6 | +101 | -43 | -7 | +8 | 1.515 / 1.650 | 5.9 |
| (1, 0) | 28 | -5 | +92 | -42 | -5 | +4 | 1.692 / 1.759 | 3.3 |
| (2, 0) | 12 | -4 | +88 | -45 | -7 | -1 | 1.532 / 1.588 | 2.9 |
| (0, 1) | 13 | -29 | +92 | -42 | -9 | +5 | 1.306 / 1.397 | 5.6 |
| (1, 1) | 34 | -12 | +84 | -45 | -6 | +0 | 1.530 / 1.558 | 1.8 |
| (2, 1) | 28 | -1 | +83 | -43 | -3 | -2 | 1.462 / 1.508 | 2.7 |
| (0, 2) | 10 | -39 | +82 | -40 | -8 | +3 | 1.303 / 1.510 | 7.4 |
| (1, 2) | 36 | -22 | +71 | -44 | -7 | -2 | 1.364 / 1.427 | 4.0 |
| (2, 2) | 12 | -5 | +66 | -43 | -4 | -5 | 1.513 / 1.607 | 4.1 |
| global (dense) | 92 | -10 | +81 | -41 | -6 | +1 | | |

The Zernikes change across the field by more than the differences between the estimators (2-3 mλ rms): Z6
astigmatism from +101 (top left) to +66 mλ (bottom right), Z5 from about -5 (top row, right column) to -39 mλ (bottom
left), spherical Z11 and secondary astigmatism Z12 from +8 / +4 mλ (left) to -5 / -4 mλ (right); coma Z7 is the same
everywhere (-40..-45 mλ). Regional fits differ from the global one by 1.8 (centre) to 7.4 mλ rms (bottom left). A
local PSF lowers the deviance per pixel by 0.03 in the centre and by 0.09-0.21 at the left and bottom edges: the
aberrations change mostly towards the bottom-left corner (Z5, Z6) and from left to right (Z11, Z12), consistent with a
tilted / decentred element. The deviance per cluster is higher for larger clusters (`deviance.png`).

## Problems and limits

- **Patch size vs cluster size**: the vectorial PSF of a coverslip bead has a wide supercritical ring when defocused,
  so any finite patch truncates it. psf_extent uses 2 % of the in-focus peak (patch 19-21 px for +-0.75 um); a larger
  patch chains a dense field into a few giant clusters. render_pad models the members' tails on each other's pixels
  (in the simulation it took the deviance from 1.23 to 1.00 and the error from 0.9 to 0.3 mλ); the tails of
  non-members further than a patch away are not modelled.
- **Faint detections**: debris and faint specks must not take part in the grouping: with them, beads.tif gave 22 usable
  clusters with 34 beads (the rest chained into clusters too large to fit), without them 48 clusters with 105 beads;
  they now join a cluster only as non-growing members. Aggregates and debris that are
  bright need the chi2 rejection.
- **Field stop**: clusters with a member box outside the (eroded) field stop are dropped; the field stop is found by
  an Otsu split of the clipped mean image. The field-stop edge is sharp in these data, so the light of beads close to
  it is clipped and cannot be modelled by a PSF without the stop.
- **Background**: per cluster and plane, constant or linear. Big clusters (up to 84 px) would need a smoother spatial
  model (e.g. a coarse grid per plane shared by neighbouring clusters).
- **Per-bead focus offsets** are needed on real data (beads in stack 1 sit ~0.2 um from stack 0's focus centre; a few
  beads are 0.4 um off; a tilt plane per stack is worse on held-out data, see the variants): a field-dependent model must separate a focus offset of a bead from a field-dependent
  defocus / spherical term (a tilt plane per stack plus per-bead offsets regularised to it is a natural option).
- **Convergence**: Adam moves a parameter by about its learning rate per step. With lr 2e-3 for the focus offsets
  (first version) beads 0.2-0.4 um off the stack centre (the held-out stack sits ~0.2 um off; a few beads 0.4 um)
  could not reach their focus in a 150-iteration refit: held-out deviance 2.14 instead of 1.65, and one regional fit
  ended far off. Now lr_focus = lr_drift = 1e-2 (cosine decay at the end), 200 refit iterations; the simulation
  errors dropped from ~0.21 to ~0.11 mλ with it (drift and tilt converge). The intensities per bead and plane could
  be solved in closed form (variable projection, as fit.solve_linear) to speed this up, and a coarse start for the
  per-bead focus (e.g. the plane of the astigmatic focus) would make it robust.
- **Cost**: one PSF evaluation per bead and plane per iteration; fine on a GPU (minutes), slow on a CPU (the CPU
  tests use 32-sample pupils and 21 planes).
- ZStackEstimator's `find_beads` keeps the brightest bead even when a fainter neighbour sits inside its ROI; in a
  dense field it is biased. Using `DenseBeadEstimator` on the singletons only is the clean isolated-bead fit.

## Next step: field dependence and the two channels

- Field-dependent Zernikes: zcoef(x, y) = sum of low-order polynomials (or Zernikes over the field) per coefficient;
  VectorialPSF.forward already takes per-emitter `zcoef_offset` [B, nz], so the joint fit only needs the per-bead
  offsets from a field model with a few coefficients per Zernike (Z5, Z6, Z11 vary most here; quadratic in x, y).
  Regularise towards the field-constant fit; validate with held-out clusters and held-out stacks, and per region.
- Separate per-bead focus offsets (tilted coverslip, beads off the coverslip) from field-dependent defocus /
  spherical: tilt plane per stack + regularised per-bead residual offsets.
- Model the defocused ring mismatch first (fit_na, which helped most here, or the finite bead size via sphere_points
  as in fit_beads, apodization, wavelength) or it will leak into the field terms.
- Multi-channel: the right channel is the same beads through another path; fit both channels with shared bead
  positions (mapped by an affine / polynomial channel transform fitted from the bead pairs) and focus offsets, per
  channel Zernike fields and photons. The cluster machinery carries over per channel; the transform needs bead pairs
  matched across the two field stops, and clusters must be usable in both channels or be fitted per channel.
- Background: a smooth per-plane model over the field (shared by clusters) instead of per-cluster linear terms.
