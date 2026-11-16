# Phase diversity: PSF from bead z-stacks with the object eliminated, NA and pupil apodization fitted

Status: Python only. Code: `torch_psf/estimation/phase_diversity.py` (`fit_phase_diversity`, `PDPatches`,
`gonsalves_metric`, `object_estimate`, `psf_otf`, `VectorialStack`, `CallableStack`, `spline_stack`),
`PhaseDiversityEstimator` in `torch_psf/estimators.py`, `fit_apod` in `ZStackEstimator` / `DenseBeadEstimator`
(`'apod'` in `fit_beads` / `fit_dense_beads`), tests `tests/test_phase_diversity.py`, study scripts
`study/phase_diversity/`, output `~/phasediv_out/`.

## Why

The bead fitters (isolated or dense) model every bead as a point at a known height and need bead detection,
grouping and per-bead terms. On the user's 2-colour bead stack all of them leave a held-out deviance of ~1.44 per
pixel; the main mismatch is a sharp ring in the defocused model PSF that the data lack, and fitting NA (to 1.39)
removes part of it (`docs/notes/dense_beads.md`). Phase diversity uses every pixel of a z-stack of a thin object
without a bead model, and gives one pupil per patch (field dependence) at a cost of one PSF stack per patch.
The missing auto-estimated parameter (NA / pupil amplitude) is fitted here in both routes.

## Method

```python
from torch_psf import PhaseDiversityEstimator, Camera
est = PhaseDiversityEstimator(optics=dict(pixel_um=0.097, NA=1.49, wavelength_um=0.67), camera=Camera(...),
                              region=(0, 0, 475, 400), patch=64, holdout=1, fit_na=True, fit_apod=True,
                              per_patch=True)
psf = est.estimate(['beads.tif', 'beads2.tif'], dz_um=0.01)   # VectorialPSF (shared pupil); est.field: per patch
```

Plane k of a z-stack images the same object O at focus f_k: `I_k = a_k (O * h_k)(x - d_k) + b_k`. Per patch
(N x N px, default 64) and plane: an intensity factor a_k (bleaching; mean log 0), a lateral drift d_k (per stack: a
cubic in the plane index without constant term, or free), a background b_k (constant, optionally a linear gradient),
a focus offset per patch (object height vs the stack centre; tilt), optionally the z-step scale. h_k comes from a
VectorialPSF (Zernikes, NA, apod, n_med, saf, free pupil phase / amplitude maps) rendered at the pixel centres in a
41 px box (oversample 1; the pixel integration is a sinc in the Fourier domain, exact since 2 NA / lambda < 1 / (2
px)), or from any differentiable PSF stack (`CallableStack(fn(focus [n, K]) -> [n, K, R, R], params)`;
`spline_stack(spline)` fits spline knots with the object profiled out).

1. **Least squares, Gonsalves' reduced form** (`gonsalves_metric`): with D_k, H_k the spectra of the windowed
   (Tukey 0.25, times the field-stop mask) background-subtracted data and of the PSF (drift phase ramp, a_k), and
   Poisson plane weights w_k = 1 / sum_x window^2 I_k:
   `E = sum_f [sum_k w_k |D_k|^2 - |sum_k w_k conj(H_k) D_k|^2 / (sum_k w_k |H_k|^2 + eps)]` over the OTF support
   (rfft half spectrum, weights 2 for the doubled columns); `E = min_O sum_k w_k |D_k - H_k O|^2 + eps |O|^2`, so the
   object is gone. Adam on the PSF parameters and nuisance terms; the background is fixed in this stage (it is
   degenerate with the eliminated object's mean).
2. **Poisson refinement** (default, `method='poisson'`): the object on an extended grid (patch + the PSF half width
   on each side: light from outside the patch and the patch edges are modelled, no window; pixels outside the field
   stop are simply not used, which is exact for a field stop in an image plane), `mu_k = a_k crop(O * h_k) + b_k`,
   Poisson NLL. The object is profiled out: started from the unwindowed least-squares object, then Fisher-scoring
   (Newton) steps solved by 6 preconditioned CG iterations (preconditioner: the Fourier-diagonal approximation
   sum_k |H_k|^2 mean(1 / mu_k)), warm-started, one per iteration; the background by EM steps. The PSF and nuisance
   terms follow the gradient at the current object (at the object's optimum this is the profile-likelihood gradient).

Problems met on the way:

- A non-negative pixel object (Richardson-Lucy, `object_solver='rl'`) cannot represent a bead between pixel
  centres (the band-limited interpolation kernel has negative lobes); the PSF absorbs the misfit. Noise-free
  simulation, true PSF as start: RL 0.24-0.29 mλ rms Zernike bias (deviance 0.020-0.026 / px where 0 is exact);
  beads on integer pixels: 0.10 mλ; the free object (Newton) 0.037 mλ (deviance 0.0007). With 30000-photon beads
  (25 beads, 9 patches, 51 planes): RL 0.24-0.34 mλ, Newton 0.07-0.11 mλ.
- The background must not be a plain gradient parameter next to the object: its error is degenerate with an object
  pedestal, Adam wandered (bg 46 instead of 40, deviance 1.4 with the true PSF). EM steps of b_k with the object fix
  it (deviance 1.02 -> 0.98 with the true PSF).
- Fitting NA in the vectorial model: when a pupil sample sits at the critical angle (NA rho = n_med) the derivative
  of sqrt(1 - s^2) is infinite and the Adam state turned NaN (one real-data patch, one simulated seed). Non-finite
  gradients are now zeroed for that step (`safe_step`).
- Per-patch fits started from the global pupil without a phase with the PSF fixed ran into a local minimum on the
  held-out stack, whose beads sit 0.2-0.4 um off the stack centre (Z7 -80..-100 mλ instead of -40): the Zernikes
  moved before the focus offset had converged. The PSF is now held fixed for the first `warmup` iterations also when
  stage 1 is skipped (the field map uses 40).

## Simulation

`study/phase_diversity/sim_compare.py` (`run_sim.sh`, `sim_summary.py`; output `~/phasediv_out/sim/`, table
`sim_table.md`): 320 x 320 px of 0.1 um, 51 planes over 1.5 um, 9 Zernikes up to 60 mλ, truth NA 1.40 with a pupil
amplitude exp(-0.5 rho^2) (the fits start at the nominal NA 1.49, apod 0), beads in clusters ('dense': neighbours 1-3
um apart, 'packed': 0.8-1.7 um), log-normal photons around 30000, background 40, a coverslip tilt (0.1 um across x),
a lateral drift. Three seeds. Phase diversity: 64 px patches with signal (22-25), shared pupil, 200 + 400 iterations;
dense: DenseBeadEstimator, all clusters, 300 iterations. A second simulated stack (other layout and noise) is held
out: every PSF is scored on its clusters by dense_nll. 'nominal': truth NA 1.49, no apod (two seeds): what fitting the
extra parameters costs when they are not needed. Pupil transmission: rms difference of the radial amplitude
(including the NA cut-off) over the pupil coordinate 0 .. 1.49.

| scenario | method | Zernike rms error mλ (mean, per seed) | NA (mean) | pupil transmission rms | held-out deviance / px - truth | seconds |
|---|---|---|---|---|---|---|
| dense | pd | 11.62 (8.20, 8.10, 18.58) | 1.4900 | 0.311 | +3.4012 | 61 |
| dense | pd_na_apod | 0.27 (0.26, 0.29, 0.27) | 1.4084 | 0.053 | +0.0056 | 72 |
| dense | dense | 4.87 (4.90, 5.04, 4.67) | 1.4900 | 0.311 | +2.1262 | 36 |
| dense | dense_na_apod | 0.22 (0.21, 0.25, 0.20) | 1.4076 | 0.059 | +0.0082 | 46 |
| packed | pd | 8.53 (8.56, 8.56, 8.49) | 1.4900 | 0.311 | +2.9350 | 60 |
| packed | pd_na_apod | 0.29 (0.33, 0.33, 0.23) | 1.4076 | 0.049 | +0.0050 | 74 |
| packed | dense | 5.29 (5.07, 5.37, 5.42) | 1.4900 | 0.311 | +2.5059 | 40 |
| packed | dense_na_apod | 0.28 (0.27, 0.31, 0.26) | 1.4081 | 0.057 | +0.0081 | 45 |
| nominal | pd | 0.18 (0.11, 0.25) | 1.4900 | 0.000 | +0.0007 | 32 |
| nominal | pd_na_apod | 0.28 (0.23, 0.33) | 1.4953 | 0.016 | +0.0016 | 42 |
| nominal | dense | 0.11 (0.13, 0.09) | 1.4900 | 0.000 | +0.0001 | 23 |
| nominal | dense_na_apod | 0.32 (0.31, 0.33) | 1.4888 | 0.060 | +0.0004 | 31 |

(seconds: on a shared RTX 3090, other jobs running; the dense-layout seed 2 of pd_na_apod is a rerun after the NaN fix.)

- With the true NA 1.40 and an apodized pupil, both estimators need NA / apod in the fit: without them the
  Zernikes are off by 5-12 mλ rms (the Zernikes try to mimic a smaller pupil) and the held-out deviance is 2-3 per
  px above the truth's. With them: 0.22-0.29 mλ rms, NA 1.408 (+0.008), pupil transmission within 0.05-0.06 rms,
  held-out deviance within 0.005-0.008 of the truth. Phase diversity and the dense fit are equally accurate; phase
  diversity is slightly better on the held-out score (+0.005 vs +0.008), the dense fit slightly better on the
  Zernikes in the dense layout (0.22 vs 0.27 mλ); in the packed layout they are the same.
- The NA bias (+0.008 in both methods) is convergence, not a model error: the NA / apod valley is flat. Noise-free
  data, phase diversity: 400 iterations NA 1.4036, 0.17 mλ; 1500 iterations NA 1.3998, 0.013 mλ, deviance 0.0002;
  started at the truth it stays there (1.4001). Use >= 1000 iterations (or a second-order optimizer) when NA / apod
  matter.
- When NA / apod are not needed (nominal), fitting them costs 0.1-0.2 mλ of Zernike accuracy and nothing measurable
  on the held-out score.
- Zernikes only, correct NA: phase diversity 0.07-0.11 mλ (`quick` runs: 25 beads in 9 patches, 51 planes, 300
  iterations) to 0.18 mλ (study), the dense fit 0.09-0.13 mλ.
- Least squares only (stage 1, windowed): 1.2 mλ in the quick runs vs 0.07-0.11 with the Poisson stage; the window
  and the edge beads are the main error, the least-squares stage is a start, not an estimator.
- Unit tests (`tests/test_phase_diversity.py`): 6 beads (pair 0.9 um, triplet, single) in 64 x 64 px, 21 planes,
  truth NA 1.40, apod exp(-0.6 rho^2): phase diversity from the nominal start recovers the Zernikes to 0.2 mλ rms, NA
  1.408, pupil transmission 0.016 rms; the Zernike-only fit is clearly worse (deviance 1.6 vs 0.94). The metric test
  checks that the reduced form is ~0 at the true PSF for a sparse and for a smooth object, equals the least-squares
  error at the closed-form object, and has its minimum over Z6 at the truth for both objects.

## Real data: 2-colour astigmatic beads, left channel

Same data and assumptions as `docs/notes/dense_beads.md` (97 nm px, NA 1.49 oil, n_imm 1.518, n_med 1.33, 670 nm,
0.597 e-/ADU, offset 100.6 ADU, x < 475, 150 planes at 10 nm). Scripts: `study/phase_diversity/common.py` (data, the
held-out cluster set and its scoring), `real_pd.py` (phase diversity variants), `real_dense.py` (DenseBeadEstimator
with fit_na / fit_apod, settings of `real_variants.py`), `score_blur.py`, `fig_ring.py`, `real_field.py`; output
`~/phasediv_out/real/`.

Training: beads.tif only. Phase diversity: 64 px patches with at least half their pixels inside the field stop and a
peak of 8 robust sd: 44 patches of beads.tif, all 150 planes; per-patch focus offset, cubic drift, per-plane
intensity and constant background per patch, 200 + 300 iterations. Held-out score: the 52 beads2.tif clusters (101
beads) of the dense-beads study (`real_fit.py`, the two aggregate clusters left out), every PSF scored by `dense_nll`
(each cluster's positions, photons, linear background, per-bead focus offsets and the stack drift refitted with the
PSF fixed; every 2nd plane). `common.py` reproduces the dense fit's 1.441 (1.4415).

| PSF | NA | apod | Z6 | Z7 | Z11 | held-out deviance / px | NLL / px vs dense | run (s) |
|---|---|---|---|---|---|---|---|---|
| dense (dense-beads study) | 1.49 | 0 | +80.5 | -40.6 | +1.4 | 1.4415 | 0 | 330 |
| dense, fit_apod | 1.49 | -0.08, -0.21, -0.28 | +90.5 | -40.5 | +1.6 | 1.3632 | -0.0391 | 222 |
| dense, fit_na | 1.394 | 0 | +81.6 | -33.7 | -7.9 | 1.3438 | -0.0488 | 269 |
| dense, fit_na + fit_apod | 1.410 | +0.03, -0.08, -0.13 | +84.4 | -35.2 | -6.1 | **1.3375** | -0.0520 | 276 |
| PD, Zernikes | 1.49 | 0 | +81.0 | -44.0 | -1.8 | 1.4513 | +0.0049 | 277 |
| PD, + apod | 1.49 | -0.24, -0.34, -0.46 | +97.5 | -40.9 | -3.7 | 1.3688 | -0.0363 | 273 |
| PD, + NA | 1.374 | 0 | +80.1 | -34.4 | -10.1 | 1.3498 | -0.0458 | 327 |
| PD, + NA + apod | 1.417 | -0.11, -0.17, -0.21 | +87.8 | -37.0 | -7.4 | 1.3438 | -0.0488 | 280 |
| PD, + NA + apod, 300 + 1000 iterations | 1.417 | +0.04, -0.17, -0.30 | +86.2 | -37.8 | -7.3 | 1.3395 | -0.0510 | 725 |
| PD, + NA + apod + free pupil phase | 1.425 | -0.11, -0.17, -0.22 | +75.9 | -33.3 | -7.2 | 1.3413 | -0.0501 | 398 |
| PD, + NA + apod + z-step scale (0.972) | 1.431 | -0.10, -0.17, -0.21 | +88.0 | -36.6 | -5.8 | 1.3428 | -0.0494 | 216 |
| PD, + NA + apod, beads 50 nm high | 1.427 | -0.11, -0.16, -0.20 | +89.2 | -38.1 | -4.5 | 1.3493 | -0.0466 | 407 |
| PD least squares only (stage 1), + NA + apod | 1.415 | -0.27, -0.31, -0.33 | +83.9 | -34.8 | -9.7 | 1.3969 | -0.0223 | 172 |

(run: estimator wall time on a shared GPU, including the held-out patch refit; NLL differences per pixel are what
counts between PSFs on the same pixels. The bead-50-nm row is scored with emitter_z 0.05 as well.)

- **The missing parameter is mostly NA / the pupil amplitude**: fitting NA (and apod) takes the held-out deviance
  from 1.44 to 1.34 in both routes, the largest single gain found on these data (the regional field dependence gave
  0.03-0.2 in-sample). Fitting apod alone gets 80 % of it (1.363 / 1.369); NA + apod is the best (dense 1.3375,
  phase diversity 1.3395 with enough iterations). Both routes land at NA ~1.41-1.42 with a pupil amplitude falling
  towards the rim when apod is free, at 1.37-1.39 when only NA is free (a hard cut-off has to do the job of a soft
  roll-off). NA, wavelength and pixel size remain partly interchangeable; read it as an effective transmission:
  `radial_amplitude(psf)`.
- **Phase diversity vs the dense fit on the dense fit's own score**: equal within 0.002-0.01 deviance / px (PD is
  slightly behind at 300 iterations, equal at 1000). The score is the bead model's (point beads, PSF fixed): PD did
  not see these clusters as beads at all, and it uses no bead detection, grouping or rejection. A lateral bead blur
  (100 nm stained shell, variance r^2 / 3 = 0.089 px^2, `score_blur.py`) improves both alike (dense 1.3375 -> 1.3199,
  PD 1.3438 -> 1.3238, dense base 1.4415 -> 1.4248, PD base 1.4513 -> 1.4339).
- Least squares alone (Gonsalves' reduced form on windowed patches) is clearly worse (1.397): the Poisson stage with
  the extended object is needed. A free pupil phase map, the z-step scale (fitted 0.972) and a 50 nm bead height add
  nothing measurable once NA and apod are fitted.
- The Zernikes of all fits agree on the main terms (Z6 +80..+90, Z7 -34..-44, Z5 -10..-12 mλ); with NA free,
  spherical Z11 goes to -6..-10 mλ in both routes (it trades with the pupil edge).

**Does fitting apod remove the ring mismatch?** Figures `~/phasediv_out/real/ring_profiles.png` (mean radial profile
of the 28 single-bead held-out clusters at -0.75 .. +0.73 um, data vs the models), `ring_images.png` (one bead: data,
models, residuals) and `deviance_focus.png` (held-out deviance per plane):

- The ring is on the positive-focus side (+0.5 .. +0.73 um, the focal plane above the bead): the
  NA 1.49 model has a sharp ring and a dark gap inside it at +0.49 / +0.73 um that the data do not have; with NA /
  apod fitted the ring is soft, the radial profiles follow the data, and the deviance per plane at +0.3 .. +0.73 um
  drops from 1.30-1.35 to 1.21-1.25 (dense, NA + apod) and 1.17-1.20 (phase diversity, NA + apod). So yes for the
  ring.
- What remains is elsewhere: the planes around -0.5 .. -0.2 um (the astigmatic line focus and the brightest planes)
  go from 1.55-1.72 (NA 1.49) only to 1.43-1.55 with NA / apod, with structured residuals at the bead centre (`ring_images.png`), and at -0.75 ..
  -0.5 um the fitted-NA models are now too peaked in the centre (radial profile: model 0.014-0.017 vs data 0.009 per
  photon at -0.75 um) where the NA 1.49 model matched. A single effective pupil cannot be sharper on one side of
  focus and softer on the other: something focus-asymmetric remains (field-dependent Zernikes within a cluster box,
  the bead's axial extent, a defocus-dependent blur such as residual vibration or the camera, or a refractive-index /
  depth effect beyond n_med 1.33).

**Field map** (`real_field.py`: one pupil per 64 px patch, started from the global PD NA + apod fit, 200 iterations
with the PSF held for the first 40, 6-7 s per patch; both stacks, 41 + 40 patches; figures
`~/phasediv_out/real/field_z/field_map.png` (Zernikes only per patch, NA / apod global), `field_z/field_stacks.png`
(beads.tif vs beads2.tif at the same field position), `field_all/` the same with NA and apod per patch):

| region (col, row), beads.tif | patches | Z5 | Z6 | Z7 | Z8 | Z11 | Z12 | focus offset (um) |
|---|---|---|---|---|---|---|---|---|
| (0, 0) | 1 | -7 | +102 | -35 | -13 | -13 | +8 | -0.08 |
| (1, 0) | 6 | -1 | +98 | -37 | -7 | -3 | +5 | +0.01 |
| (2, 0) | 3 | -3 | +92 | -37 | +2 | -2 | +7 | +0.11 |
| (0, 1) | 4 | -26 | +109 | -39 | -9 | -5 | +9 | -0.15 |
| (1, 1) | 6 | -9 | +91 | -34 | -4 | -5 | +5 | -0.02 |
| (2, 1) | 5 | -1 | +90 | -37 | -5 | -6 | +5 | +0.05 |
| (0, 2) | 3 | -42 | +85 | -37 | -6 | -7 | +3 | -0.21 |
| (1, 2) | 9 | -25 | +82 | -40 | -4 | -8 | +1 | -0.10 |
| (2, 2) | 4 | -7 | +73 | -39 | -2 | -9 | +3 | -0.04 |

(medians of the patches per region; mλ.)

- The same pattern as the dense regional fits (dense notes: Z5 from -5 at the top / right to -39 bottom left, Z6 from
  +101 top left to +66 bottom right, Z7 flat at -40..-45): Z5 -1 / -3 at the top and right to -42 bottom left, Z6
  +102..+109 on the left to +73 bottom right (all Z6 values ~7 mλ higher than in the dense table because NA / apod
  are fitted here), coma Z7 flat. The focus offsets show the coverslip tilt (-0.21 bottom left to +0.11 top right).
- The two stacks (different bead fields, same camera positions) agree for Z5 (r = 0.81, median |difference| 4 mλ,
  field range -39..+6 mλ) and Z6 (r = 0.56, 6 mλ, range 68..115); for Z7, Z8, Z11, Z12 the per-patch noise (median
  |difference| 4-7 mλ) is as large as the field variation, and the per-patch NA / apod (`field_all`) are noise
  (NA 1.38-1.44 without a pattern; stack-to-stack r ~0).
- Held-out (each beads2 cluster scored with the PSF of the beads.tif patch at its position vs the global PSF, same
  grouping): the local PSF wins in 21 of 32 groups (median -0.006 deviance / px) but loses badly in a few (patches
  with an aggregate or a bead off the coverslip: e.g. (128, 128) with Z7 -2 instead of -38), so the pixel-weighted
  total is 1.340 local vs 1.329 global (1.384 with NA / apod per patch). Per-patch pupils need regularisation: a
  smooth field model (low-order polynomials per Zernike over the field, fitted jointly over all patches, NA / apod
  global) and a robust patch weighting are the next step.

Runtime: the global fit (44 patches x 150 planes, 200 + 300 iterations) 3.5-5.5 min on a shared RTX 3090, the same
as the dense fit of 92 beads x 75 planes (4-5.5 min); one PSF stack per patch instead of one PSF per bead and plane.
Per patch 6-7 s (200 iterations). The cost is dominated by the vectorial PSF (K x 41 x 41 px per patch) and the
6 CG steps of the object (2 x 13 batched FFTs of K x 104 x 104 per iteration).

## Limits

- **Least squares vs Poisson**: Gonsalves' reduced form needs a periodic (windowed) patch; the window and the beads
  cut by it bias the PSF (1.2 mλ in simulation; real held-out 1.397 vs 1.344). The Poisson stage (extended free
  object, profiled out by Newton / CG, background by EM) removes both and is what the estimator returns; the
  least-squares stage only provides the start. The object must be free (band-limited), not a non-negative pixel
  image (Richardson-Lucy), or beads between pixels bias the PSF.
- **Thick samples**: the object is a single plane per patch (one focus offset). Beads off the coverslip (here a few
  are 0.4 um off), aggregates and dirt at another height are modelled wrongly and pull the patch's pupil (the
  outlier patches of the field map). The bead fitters have a focus offset per bead; a per-patch object with a few
  height layers, or a robust patch / pixel weighting, would be needed for thicker or less clean samples. The axial
  extent of a 100 nm bead is not modelled (its lateral extent is part of the object); a 50 nm emitter height did not
  help here.
- **Drift**: lateral drift is a phase ramp per plane (cubic per stack or free), so it must be the same for the whole
  patch (or stack); a drift that differs within a patch (sample deformation) is not modelled. Axial drift during the
  scan appears as a z-step scale error (fittable: 0.972 here, no held-out gain; but in the 21-plane test simulation
  with a true scale of 1 the fitted scale was 0.96-0.98 after 80 and 0.983-0.993 after 300 iterations: it converges
  slowly and trades with the PSF, so 0.972 is not a measurement) or not at all if non-linear.
- **Convergence**: NA and apod form a flat valley; Adam needs ~1000 iterations to settle NA to 0.001 (simulation),
  and the free per-patch fits need the PSF held fixed until the focus offset has converged (warmup), or they drift
  into a coma / astigmatism local minimum.
- **Per-patch noise**: a 64 px patch with a few beads gives 4-7 mλ per-patch scatter; per-patch NA / apod are not
  determined. Use a field model rather than independent patches.
- **Scoring**: the held-out score is the bead model's; phase diversity's own deviance (1.17 training / 1.13 held-out
  stack) is lower because the free object also absorbs bead size, debris and aggregates, and is not comparable.
- FD-DeepLoc stacks (optional item) were not run.
