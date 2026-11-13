# CRITIC_INSITU_1: review of the in-situ spline PSF study (branch insitu-study, 1003139)

Reviewer: independent critic, 2026-11-13. Scope: `docs/insitu_spline_psf.md`, `docs/insitu_psf_literature.md`,
`torch_convspline/knotpsf.py`, `torch_convspline/vectorial.py`, `study/insitu/*.py`.
My own scripts are in `study/insitu/critic/` (run from the repo root with the simcode env). Nothing else was changed.
The GPU was shared with another job at 99 % load the whole time, so my absolute timings are 2-4x slower than the
doc's. Only ratios measured inside one script are used below.

Tests I ran:

| script | what it does |
|---|---|
| `critic/grad_speed.py 2000` | route K vs route V gradient at 3 points and for the pupil maps; interpolation bias on noise-free data; an optimised route V; timings, interleaved |
| `critic/vec_sampling.py` | accuracy of the vectorial model itself at pupil 32/64/128/256 and oversampling 3-7 |
| `critic/movie_diag.py OUT.pt` | detected movie patches of scenario A matched to the truth: sites, z distribution, offset from the patch centre against z |
| `critic/movie_fits.py OUT.pt 1.5 3.0` | free pupil fit on the movie patches with the default +-1.5 px offset clamp and with +-3 px |
| `critic/inverse_crime.py` | scenario A Zernike in-situ fit on data rendered with a finer vectorial model (and a 1.35 sample index) |

Severity: **blocker** = a reported conclusion is wrong or unsupported; **major** = a number or claim would change
materially or needs a new experiment; **minor** = wording, presentation, or a small effect.

---

## Findings verified by running

### C-01 (blocker). The "detection-step tilt" of section 3.4 is mostly a clamp in the code

Section 3.4 and Summary 4 say the 28-36 nm y-vs-z tilt of the free pupil on movie patches "comes from the movie
patches" and that "the detection step causes it"; centring or selection, not established which.

Root cause: `KnotPSF` defaults to `xmax=1.5` px (`torch_convspline/knotpsf.py:20`). `evaluate` clamps the emitter
offset to +-1.5 px (`knotpsf.py:61-62`), and `localize` clamps x and y to +-1.45 px (`knotpsf.py:133-134`).
`fit.new_spline` keeps the default (`study/insitu/fit.py:15-16`). The DoG detector centres the patch on the
intensity peak (`study/insitu/common.py:152-187`). With this pupil (astigmatism plus 0.03 waves coma) the peak moves
away from the emitter as z grows. `critic/movie_diag.py` matches every detected patch to the truth:

```
z bin     events  detected  frac   mean dx   mean dy   sd dy
0.4-0.5     2373     1399  0.59  +0.204  -0.394  0.347
0.5-0.6     3452     1263  0.37  +0.338  -0.759  0.405
0.6-0.7     3610      559  0.15  +0.441  -1.230  0.438
0.7-0.8     3169      114  0.04  +0.682  -1.724  0.617
true offset from patch centre: |dy| max 3.11 px; fraction beyond the knot clamp 1.45 px: 0.0270
```

Above z = 0.6 µm a large share of the training emitters sit beyond the clamp, all on the same side (−y). The
localizer cannot reach them, so the PSF has to absorb the shift. Localizing the movie patches with the TRUE PSF
already gives a +14 nm mean y error for z > 0.6. The Zernike model is too stiff to follow, so it stays clean
(1-2 nm, as the doc reports). The free pupil can follow.

Controlled test (`critic/movie_fits.py`): same movie patches, free pupil from the true Zernikes, 200 iterations.
Only the clamp changes. Mean y error per z bin on the 6000 test emitters, |z err| < 150 nm mask:

```
xmax 1.5 px: free pupil  dnll 0.065  y: -1.8 -0.2 +2.1 +6.9 +13.9 +24.3 +38.5   failures 0.053
xmax 3.0 px: free pupil  dnll 0.048  y: -0.6 -0.4 -0.3 +0.6  +1.3  +1.9  +5.0   failures 0.015
```

The tilt drops from 38.5 to 5.0 nm, and the failures drop to the true-PSF level. The author's own new, uncommitted
`run_detect_bias.py` case (a), cropped at the true position, also has no tilt. That fits this explanation (no
offset beyond the clamp) as well as "centring".

Consequences:
- 3.3 claims the free pupil "does not remove the y error that grows with z" and in A "creates such an error".
  Those statements, and the 3.3 table rows for the free pupil, are artefacts of the clamp.
- The "Correction" paragraph and the 3.4 bullets blame the detection step. Wrong in the form stated.
- Summary 4 and section 4 ("Its weak point is bias from the spot-detection step") are wrong in the form stated.
- Scenario B movie numbers (3.2, 3.3), including "free knots ... leave both curves where they were", may be
  contaminated too. The free-knot model can absorb a clamp-induced shift even more easily. The "failure rate goes up
  (7 % -> 14 %), cause not investigated" in 3.2 is a likely symptom.

What remains, 5 nm at the top bin, could be selection: only 4 % of the 0.7-0.8 µm events are detected, and those are
twice as bright (C-07). It is small.

Resolve: make xmax at least the largest peak-to-emitter offset (or re-centre patches after a first localization).
Re-run run_study, run_pupil and run_movie_clean, then rewrite 3.3, 3.4, Summary 4 and section 4. Add a test that
flags localizations hitting the clamp.

### C-02 (major). Two different masks and binnings make the 3.3 and 3.4 tables disagree on the same experiment

- `F.evaluate` drops a test emitter as a "failure" when |z err| > 150 nm **or** |x or y err| > 0.5 px (`fit.py:236`).
  run_study and run_pupil compute the bias curves, and all RMS columns, on that set, with 14 bins.
- run_clean, run_movie_clean and run_pattern use a mask on |z err| only (`run_clean.py:17`, `run_pattern.py:112`),
  with 7 bins.

A biased PSF pushes its high-z emitters past 54 nm in y. The `ok` mask then removes exactly the biased emitters, so
the tilt is truncated and the RMS looks better. Same fit, both masks (`critic/movie_fits.py`, xmax 1.5):

| mask | top-bin y error |
|---|---|
| ok mask | +28.7 nm |
| z-only mask | +38.5 nm |

This is why 3.3 says "A, free phase + amplitude: −2..+29" while 3.4 says "A, movie, all patches, free pupil:
−1..+36" for what is the same experiment.

The same truncation makes the "failures" column count bias, not wrong z branches. The doc concedes this for the
pupil fits. It also applies to "Zernike, beads only", where 33-42 % failures come with x/y mean biases of 12-17 nm
(`out/results.json`). "12-40 % gross failures" in Summary 2 is therefore misleading.

Resolve: one mask everywhere (z-branch errors only). Report wrong-branch rate and bias separately. Report RMS about
the per-bin mean or over all emitters, not after dropping the biased tail.

### C-03 (major). The speed-up is for the gradient alone; per iteration and against a sensible vectorial setting it is smaller

From the doc's own `out/speed.json`:
- Localizing on the spline costs 8.3 µs per emitter (3 iterations, N = 64,000). Both routes pay it every iteration
  (`fit.py:166-167`), and it is as large as the knot gradient (7.4 µs).
- Per iteration, 64/3 setting:
  - at 1e6 emitters: K = 7.4 + 8.3 = 15.7 µs, V = 45 + 8.3 = 53 µs → **3.4x**, not 6.1x;
  - at 1e4 emitters: about 1.3x.
- The measured whole fit (11,593 emitters, 60 iterations) is 47 s vs 25 s = **1.9x**.
- "Cost per emitter drops from 45-390 µs to 7 µs" (Summary 1) leaves out the 8 µs both routes share.

The 50x row uses pupil 256 / oversampling 6. `critic/vec_sampling.py` (scenario A, 200 random emitters, relative L2
against a 256/7 reference):

```
pupil  64 os 3: median 4.1e-03   nll cost @4000 ph 0.0105
pupil 128 os 3: median 2.3e-03
pupil 128 os 4: median 1.6e-03   nll cost 0.0015
pupil 256 os 6: median 2.2e-04
pupil  32 os 3: median 1.5e-02
```

Route K is limited by the spline's own error (0.2 %, section 2.1). The 256/6 model is ten times more accurate than
the spline, so the 256/6 comparison sets an exact model against a 0.2 %-accurate one. The like-for-like comparison
is 128/4 (0.16 %), which gives 11-13x for the gradient and about 6x per iteration at 1e5-1e6 emitters. The 50x figure
should not be in the summary.

Smaller fairness points:
- Route V uses batches of 250 (`fit.py:138`). With batches of 1000 it is 1.37x faster in my interleaved test
  (144 → 105 µs per emitter, shared GPU).
- Precomputing the shared pupil, folding the x/y shift into the MFT matrices and TF32 gave nothing beyond that
  (103-110 µs). The gradient is identical (rel diff 7e-8).
- The route V implementation is otherwise competent: an MFT (chirp-z style) on a cropped grid, complex64. A fused
  custom kernel could probably speed it up further, as the doc says the CuPy kernels would for route K.
- Route K's fixed cost runs the knot build twice per iteration: the no-grad build at `fit.py:166`, then a full
  forward again inside `vjp_vectorial` (`knotpsf.py:109-122`). Building once with grad would cut the fixed cost by
  about 40 % and lower the break-even. Both sides have slack of a similar size.
- GPU timing is synchronised correctly (`bench_speed.py:13-16`).

Resolve:
- Report the per-iteration speed-up (gradient + localization) next to the gradient-only one.
- Pair each vectorial setting with its accuracy, and headline the setting whose error matches the spline's.
- Use route V with batch 1000.
- State that the "conventional" route V already localizes on the spline (a hybrid). A fully vectorial pipeline would
  be slower still, but it was not measured.

### C-04 (major). "Cosine similarity 1.00000" is measured only where it is easiest; near the optimum the routes differ by 4-12 %

`bench_speed.py` compares the routes at the bead-fit Zernikes `zc_s` (`bench_speed.py:8-12`). That is far from the
in-situ optimum, where the gradient is dominated by model mismatch. `critic/grad_speed.py`, 2000 noisy patches,
emitter parameters at the truth:

```
zcoef at bead_fit        : cos 1.000000  rel diff 9.13e-04
zcoef at truth           : cos 0.999440  rel diff 4.02e-02
zcoef at truth+0.01 rand : cos 1.000000  rel diff 1.11e-03
pupil_phase map at truth : cos 0.999189  rel diff 4.35e-02
pupil_amp   map at truth : cos 0.992247  rel diff 1.25e-01
```

So the identity is exact only for the spline-interpolated likelihood. Against the vectorial likelihood it holds up
to interpolation error, which, as section 2.3 says, matters where the gradient is small. Size of the resulting bias
(noise-free data d = mu, true parameters; route V gradient exactly 0, route K gradient not):
- fixed-point shift of the Zernikes: 1.6e-5 waves rms (max 3.3e-5);
- with emitter parameters known, the CRLB per Zernike is 1.4e-4 / 4.4e-5 / 1.4e-5 waves at 1e4 / 1e5 / 1e6 emitters;
- bias/CRLB = 0.11 / 0.36 / **1.15**. These are upper bounds; with emitter parameters free the CRLB is larger.

The bias is physically negligible (λ/60,000). But it reaches the statistical error exactly in the 1e6-emitter regime
where the speed claim lives. For a 6,400-pixel pupil map the relative mismatch is larger (12 % for the amplitude
gradient at the truth).

Resolve:
- Write "identical up to the spline interpolation error; the fixed point moves by about 1e-5 waves" instead of
  "identical".
- Report the gradient comparison at the converged point and for the pupil maps.
- Mention S = 3 or a final polish with route V if more than 1e6 emitters is the goal.

### C-05 (major). Inverse crime: same discretised model for simulation and fit, and scenario B lies exactly inside the free-pupil model class

- Data and fit use the same `VectorialPSF` with pupil 64 / oversampling 3, the same NA, indices, wavelength, pixel
  size, isotropic dipole and Poisson-only camera (`common.py:24`). `critic/vec_sampling.py` shows the 64/3 model is
  itself 0.4 % (median, up to 1.4 %) away from the converged model. That is twice the spline error the study
  optimises.
- Scenario B's "structure outside the Zernike series" is a phase screen and a Gaussian amplitude patch defined on the
  same 64 x 64 pupil grid (`common.py:46-58`) as the free pupil map that is fitted (`fit.py:192`). The free pupil
  therefore contains the B truth exactly.
- Conclusions that rest on this: Zernike in situ "as good as the truth" in A (deficit 0.01); free pupil "represents
  scenario B" (deficit 0.01 on clean patches); "with a pattern the free pupil fit is exact within noise" (3.5).
- Conclusions largely independent of it: the gradient chain rule, the gauge-freedom argument of 2.4, and the speed
  numbers.

Results of `critic/inverse_crime.py` (details in the addendum):
- a finer data discretisation (128/4) leaves the scenario-A Zernike result intact (deficit 0.002);
- a sample index of 1.35 that the fit does not model gives deficit 0.10 and z bias −22..−5 nm.
So the A conclusion is robust to discretisation but not to fixed optical constants. The B / free-pupil conclusions
remain untested off-grid.

Resolve:
- Simulate with a finer, different discretisation (pupil 128-256, oversampling 5+), a B screen not on the fitted grid
  (analytic, or higher-resolution then resampled), and at least one effect outside the model: small NA or index error,
  sCMOS read noise and gain, residual blur, a fixed-dipole fraction, bead size.
- Report deficit and bias against the true PSF of those data.

### C-06 (major). Single seed; several reported differences are within the likely run-to-run spread

No error bars anywhere (admitted in section 6). Evidence that training-set noise is not small:
- My re-run of "A, movie, free pupil", same movie but Zernike start at the truth instead of the in-situ fit, gives
  +38.5 nm at the top bin against the doc's 36 (3.4) and 29 (3.3, other mask).
- The free-pupil-on-clean-B tilt is "−3..+10" in 3.4 but "−3..+15" in the M = 0 control of 3.5, a setting meant to
  be comparable (different N and photons).

Differences I would not interpret without repeats:
- 9.6 vs 9.7 nm, 13.3 vs 13.3, 21.5 vs 21.5 (fine as "equal", but not as "better");
- deficit 0.12 (phase+amp) vs 0.15 (phase only) vs 0.25 (free knots) in B;
- "28 vs 36 nm, changes little" (3.4);
- all min..max ranges of a few nm ("−1..+3", "0..+1", "−2..+2"): a min..max over 7-14 noisy bin means is biased
  away from zero by noise. 1-2 nm ranges are noise.

The per-bin standard error from test noise alone is about 0.5 nm (about 850 emitters per bin, 15 nm rms). The
training-set (PSF estimate) spread is unknown and is the dominant term.

Resolve: 3-5 seeds for movie, beads and test data. Report mean ± sd of deficit, failure rate and the top-bin bias, or
fit a slope dy/dz with its standard error instead of a min..max range.

### C-07 (major). The training z distribution is not what the doc implies; high z is barely represented

`critic/movie_diag.py`, scenario A:
- 11,593 patches from 25,192 on-events, but only **571 distinct sites**; median 19 detections per site, max 64. The
  patches are not 11,600 independent emitters.
- Detection efficiency is 58-69 % for z < 0.5 µm, 37 % at 0.5-0.6, 15 % at 0.6-0.7 and **4 % at 0.7-0.8**.
- The detected emitters at the top are twice as bright (median 8,215 vs 4,034 photons).
- The test set is uniform on 0.05-0.75 µm (`common.py:105`), and the movie on 0-0.8 µm (`common.py:119`).

So the top of every bias curve is scored where training data are scarce and photon-selected. Free models then
mis-fit there (the doc's "few photons constrain the knots" is in part "few emitters"). This also feeds the residual
5 nm in C-01.

Resolve: report the training z histogram. Score the bias curve with per-bin training counts, or restrict the test z
range to where training data exist. Compare against clean patches with the same site count and z histogram, to
separate "few sites, few high-z emitters" from detection effects.

---

## Findings from reading only

### C-08 (major). Bead-vs-sample mismatch: the size of the effect is a free choice, and the bead-spline baseline uses a known-wrong z labelling

- "Zernike, beads only" is evaluated through the in-situ vectorial model with the correct index mismatch
  (`fit.knots_from(..., 'insitu')`). Its failure is entirely the chosen aberration drift `DZ_INSITU` (Noll 5, 7, 8,
  11 by 0.02-0.04 waves, `common.py`). With no drift it would be exact in A. "Beads alone are not enough" is
  therefore true by construction, and its size (50-80 nm z error, 33 %) scales with an arbitrary drift. The
  magnitudes are not justified from the literature, and there is no sweep over drift size.
- "Spline, beads only" labels the bead planes with the paraxial focal-shift factor 1.33/1.518 = 0.876
  (`common.py:23`, `fit.py:34`). For NA 1.49 into water, the effective factor that practitioners use is about 0.7-0.8
  (Li et al. 2018 rescale by 0.75). Its z scale 0.90 and offset 107 nm (`out/results.json`) are largely this
  labelling choice; z_rmse_lin is 30 nm against 76 nm raw. The baseline is weaker than a competent user's.

Resolve:
- Sweep the drift (0, 0.01, 0.02, 0.04 waves) and plot error against drift.
- Give the bead spline a realistic depth-scaling correction (or fit the scale on the SMLM data, a one-parameter fix)
  before calling it a failure.
- Reword Summary 2 as "when the aberrations drift by X between calibration and experiment".

### C-09 (major). The pattern test (3.5) leaks truth and stays inside the model class

- The pattern pitch, phases, depth and orientation are exactly the simulated ones (`run_pattern.py:15-28`), and the
  pattern is registered to the camera grid without error.
- Data are clean patches with uniform ±0.5 px offsets (`run_pattern.py:35-37`), so the C-01 clamp never triggers:
  3.5 never meets the problem it is offered as a cure for.
- 12,000 photons per emitter (3x the movie), 3,000 emitters.
- Scenario B is exactly representable (C-05).
- Stage 0 (`run_pattern.py:124-127`) fits Zernikes on the summed frames first. In A that is already the truth; in B
  the summed-frame Zernike error (≤ 45 nm) is well inside half a pitch (110 nm). So stage 0 does not hide a
  failure here. It does hide the fact that the method needs a near-correct PSF before the pattern can be used, which
  is disclosed (fringe locking from the bead PSF, 125 nm error).
- In real SIMFLUX/ROSE-type data the pattern phase and depth are estimated from the same localizations, which couples
  them to the PSF.

Resolve: estimate pitch, phase and depth jointly. Add realistic pattern errors (phase drift between frames, 1-2 %
pitch error). Run on movie patches once C-01 is fixed. Use a non-representable B. Present 3.5 as "with a perfectly
known pattern" until then.

### C-10 (minor). The likelihood-deficit metric

- `dnll` (`fit.py:248`) is the mean over all test emitters of the local-optimum NLL under the estimated PSF minus
  that under the true PSF, both from the same 5-start optimizer. That is sensible and held-out (test seed 200,
  training seeds 100/300/207: independent).
- By construction it is blind to the three gauge freedoms of 2.4. The doc says so; good.
- It mixes in wrong-branch emitters, whose optimum can be lower under a wrong PSF. Report it on correctly-branched
  emitters as well.
- Its noise level is never given. A per-emitter standard deviation would show whether 0.01 vs 0.06 differ.

### C-11 (minor). Intensity bias not reported

In B, every Zernike-based and free-knot estimate has I_ratio about 0.856 (`out/results.json`). The 60 % amplitude
defect costs 14 % photometry. The tables do not show it, and photometry is the third gauge freedom of 2.4. Add an
I-bias column.

### C-12 (minor). Regularisation and hyperparameters were tuned on the test set

Disclosed in section 6: lam_ref 3000, lam_s 1000 and w_stack 0.1 were picked by looking at the scenario-A test
likelihood. The Adam rates for the pupil maps (`fit.py:182`) give each pixel the same step size whatever its
information. Use a validation set separate from the test set.

### C-13 (minor). Stated limitations to keep visible

- Field-constant PSF and uniform background.
- Free-knot S=2/hz=40 nm knot grid with zpad 5.
- 15 px ROI: a defocused PSF at z = 0.8 with a 1.7 px peak offset is near the patch edge.
- The 1e6 column is extrapolated.

All are disclosed. The headline speed claim needs "field-dependent / many-emitter" settings that are not tested
(section 5 says so; the Summary does not).

### C-14 (major). Novelty claim: defensible only narrowly; reword

What the literature search and reading support:
- **uiPSF** (Liu et al. 2024, Nat Methods) evaluates the pupil-based PSF per emitter, with Fourier phase ramps for
  the shift. Methods: PSF "of a emitter at location xi,yi,zi from the pupil function using Fourier transforms". No
  table plus single backprop, so the claim holds against uiPSF.
- **INSPR / VISPR** sort emitters into z classes, average, and phase-retrieve on the averages. Mathematically the
  knot gradient is a residual stack back-projected onto a z grid, so route K is the exact-likelihood (adjoint)
  version of the same "pool emitters on a z grid, then fit the pupil" idea. That should be said explicitly; it is
  not an unrelated new principle.
- **Droste et al. 2025** (Optica; first author confirmed) precompute OTF patches over the field, a similar
  amortisation over space rather than z.
- Lookup-table localization (Hauser & Jacak 2021, PMC8407837), Li et al. 2018 cspline, and PPG3D tabulate PSFs but
  do not learn the pupil through them.
- Generically, "gradient on a lookup table, then one VJP through the generator" is plain reverse-mode autodiff of
  `L(interp(B(φ)))`. Any autodiff code that tabulates a pupil model and interpolates it would get this for free.
  The novelty is the application and the batching, not the chain rule.
- I found no paper that does exactly this. The search was not exhaustive: Nature/PMC full texts were partly
  unreachable, and LUNAR and DeepSTORM3D-style physical layers were only checked at abstract level; DeepSTORM3D and
  LUNAR evaluate the physical model per emitter, as far as I can tell.

Resolve: "To our knowledge no published in-situ method back-propagates through a tabulated (spline) PSF. Route K is
an exact-likelihood counterpart of the z-binned phase retrieval of INSPR/VISPR." Fix the literature file: Droste is
right; also add Hauser & Jacak.

### C-15 (minor). Small code observations (no effect on reported numbers found)

- `vjp_vectorial` re-runs the build forward (C-03).
- `vectorial.py:338` recomputes the Zernike sum per emitter. That is wasteful for route V, but measured at under 3 %
  (C-03).
- The `bench_speed2` 256/6 run uses nb = 50 for route V (`bench_speed2.py:24`), which handicaps V further at that
  setting.
- `localize` without th0 starts every patch at x = y = 0. For movie patches with a 1.7 px offset this is fine once
  xmax is fixed.
- I checked the knot geometry in `stack_planes` against `evaluate` (u = u0 + m/S on both sides) and found it
  consistent. The Catmull-Rom matrix is the standard one (tau 0.5).

---

## Addendum: C-05 results (critic/inverse_crime.py)

Scenario A, clean patches: 8,000 for training, 4,000 for testing, with new seeds. The fitted model is the study's
(64/3, n 1.33). Zernike in situ, route K, 60 iterations from the true Zernikes.

```
data 64/3 (inverse crime)  Zernike in situ  dnll -0.000  z rms 21.4 (true PSF 21.4)  y err -1.9..+0.2  z err  -3.9..+2.1 nm
data 128/4                 Zernike in situ  dnll  0.002  z rms 21.5 (true PSF 21.6)  y err -2.1..+0.2  z err  -4.1..+2.5 nm
data 128/4, n_med 1.35     Zernike in situ  dnll  0.101  z rms 26.3 (true PSF 20.6)  y err -1.9..+0.8  z err -22.0..-4.9 nm
```

- The pupil discretisation alone does not matter for the Zernike fit in A. That part of the inverse crime is
  harmless, and the A result survives it.
- A modest model error that the fit cannot absorb (sample index 1.35 instead of 1.33) costs 0.10 nats per emitter,
  a 17-nm-wide z-bias band and 28 % worse z precision. Noll 5-21 does not absorb it.
- So "Zernike in situ = truth" rests on the fitted optical constants being exact, not on the discretisation. A
  real-data paper needs n_med (and probably NA) fitted, or a sensitivity table. The free-pupil/B on-grid question of
  C-05 is still untested; the free pupil may well absorb an index error, which would be a point in its favour worth
  showing.

---

## Verdict on the paper claims

**Can stand as written (after small edits)**
- The spline reproduces the vectorial model to 0.2 % at S = 2 (2.1). Add that the 64/3 vectorial model itself is
  0.4 % from converged.
- The gauge-freedom analysis of 2.4, and that free knots keep the tilt of their start on clean patches.
- A shared bead + SMLM spline keeps the bead geometry (Summary 6), as a statement about this simulation.

**Need rewording**
- Summary 1 / 3.1:
  - gradient speed-up vs per-iteration speed-up (about 1.3-3.4x at 64/3, about 6x at 128/4);
  - drop "50x" or tie it to model accuracy;
  - "identical up to interpolation error" with the measured 4 % mismatch at the optimum and the 1e-5-wave
    fixed-point shift.
- Summary 2: "gross failures" → bias plus failures. State that the gap scales with the chosen drift and that the
  bead-spline baseline uses paraxial z labelling.
- Novelty (5, literature file): an exact-likelihood counterpart of z-binned phase retrieval; "to our knowledge".
- 3.5: "with a perfectly known pattern, on clean patches, with a representable pupil".

**Need a new experiment**
- 3.3, 3.4, Summary 4 and section 4 (free pupil tilt, "detection step"): re-run with xmax ≥ 3 px (C-01). Present
  conclusions are artefacts.
- 3.2 scenario B movie rows, especially free knots and the 7 → 14 % failure increase: re-run with the clamp fixed
  and with the z-only mask (C-01, C-02).
- Everything accuracy-related: 3-5 seeds and error bars (C-06); realistic training z coverage (C-07).
- A and B without the inverse crime: finer or different discretisation, B off the fitted grid, at least one effect
  outside the model (C-05).
- Drift sweep for the bead-vs-sample claim, and a depth-scaled bead spline baseline (C-08).
- Pattern with an estimated pattern and on movie patches (C-09).
