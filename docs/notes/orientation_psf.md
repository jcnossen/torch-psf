# Dipole-orientation PSF: model, CRLB study for SIMCODE, fitter, Vortex PSF reproduction

Branch `dipole-orientation` (from master 0947eb3). Code: `torch_psf/orientation.py`, `VectorialPSF.dipole_basis`,
`tests/test_orientation.py`. Study scripts and figures: `study/orientation/` (`crlb_orientation.py`,
`fit_validation.py`, `vortex_repro.py`, `common.py`); numbers in `study/orientation/results/*.json` (not committed).

## 1. Model

* Emitter tensor `M = (1 - g) I/3 + g mu mu^T` (trace 1). `g` = order parameter (g2 of Hulleman et al.): 1 = fixed
  dipole, 0 = free rotator; a uniform wobble cone of half-angle `a` has `g = cos a (1 + cos a) / 2`
  (`cone_to_g`, `g_to_cone`: g = 0.75 -> 37 deg, 0.4 -> 60 deg, 0.2 -> 72 deg).
* `mu = v / |v|` with an unconstrained 3-vector `v`: no pole singularity, `mu` and `-mu` identical.
  `angles(mu)` returns theta in [0, 90] deg (from the optical axis) and phi in [0, 360) for the representative with
  `mu_z >= 0`. The requested phi range [0, 180) would merge two different orientations: (theta, phi) and
  (theta, phi + 180) tilt the same in-plane axis towards +z and -z, and their PSFs are mirror images. `fold=True`
  gives the [0, 180) axis when that is all one wants. (Hulleman et al. use the equivalent half sphere theta in [0, 180),
  phi in [0, 180).)
* Linear structure: at a fixed position the image is linear in `m = (Mxx, Myy, Mzz, Mxy, Mxz, Myz)`:
  `psf = sum_k m_k B_k`. `VectorialPSF.dipole_basis(pos, focus)` computes the six basis images from one field
  evaluation (`forward` was split into `_fields` / `_image`; its results are unchanged, existing tests pass).
  `solve_M` is the closed-form start: Poisson-weighted linear least squares of `(I m, bg)` per spot (or summed over a
  track). `project_M` maps an estimated M to `(mu, g)`: the top eigenvector, and `g = l1 - (l2 + l3)/2` from the
  eigenvalues of the trace-normalised M.
* Excitation: `excitation_tensor(e, eps, k_dir, ck)` (the `K` of `photosel_tensor`) and the per-frame brightness
  factor `3 tr(M K)` (1 for the isotropic emitter). For an ensemble with order g and uniformly distributed axes the
  photoselected tensor is `(1 - gamma) I/3 + gamma K` with `gamma = 0.4 g^2` (`ensemble_gamma`, tested by Monte Carlo).
* Fisher / CRLB (`crlb_orientation`) in the tangent coordinates `(a, b)` of mu along `(e_theta, e_phi)`, so the pole
  is finite: `sd(theta) = sd(a)`, `sd(phi) = sd(b) / sin(theta)`, `sd(inplane) = sd(b)`. Tracks: T frames that share
  x, y, z, I, bg and the orientation, with known per-frame factors h_t and optional K_t (Fisher summed over frames).
* Fit (`fit_orientation`): Poisson MLE of `(x, y, z, I, bg, v, g)` with the existing Levenberg-Marquardt
  (`torch_psf.fit.lm_poisson`). `|v|` is renormalised after each step and g is clamped to [0, 1]. The start is the
  isotropic fit, then `solve_M` + `project_M`, plus a mirror start (`mu_z -> -mu_z`, the other tilt sign). The best
  likelihood wins. Spots `[N, R, R]` or tracks `[N, T, R, R]`. Returns estimates, angles and the CRLB at the estimate.
* `vortex_phase(model, charge)`: spiral phase plate `charge * atan2(p_y, p_x)` written to `model.pupil_phase`.

Tests (`tests/test_orientation.py`, CPU): roundtrips, `sum m_k B_k == forward(dipM=M)` (with blur), ensemble gamma,
`solve_M` exact on noise-free data, the analytic Jacobian vs finite differences (track with K), finite CRLB at the
pole, the Vortex donut, and fits of spots and of a track with K.

## 2. SIMCODE settings and assumptions

* Optics of the two in-situ PSF files (read-only): NA 1.49, n_imm 1.518, n_med 1.33, 700 nm, pixel 108.33 nm.
  `m1809` (`calib_1809_2_psf_v3.json`): Zernikes 5-21 (small, |Z| ~0.02 waves), focus 308 nm, no blur.
  `m25` (`calib_2p5nM_m2_psf_v3.json`): larger astigmatism (Z5 -0.033, Z6 -0.070 waves) and coma (Z11 0.057),
  focus 101 nm, general blur covariance [0.65, 0.36, -0.47] px^2. The aberration-free reference cases use the same
  two foci (`free_f0.10`, `free_f0.31`). ROI 15 x 15, npupil 64, oversample 2.
* Background **10 photons / px / frame** (assumption; SIMCODE's simulator presets use 6.1 and 10.8). bg 30 is
  checked as a sensitivity case. No read noise in the Fisher matrices; the 0.9 e read noise is negligible next to 10
  bg. Photons per frame: 500 / 1000 / 2000, as the intensity of an isotropic emitter at the coverslip. A fixed
  dipole's detected total depends on its orientation and height.
* Excitation, from SIMCODE (`psf3d.e_dir` and `psf_dipole_M_angle` in the `_axis.json` files, read-only). Two pattern
  angles with fringe directions 1.2 deg (~x) and 91.4 deg (~y). s-polarised: the in-plane field e_a is perpendicular
  to k_a (angle 0 excites the y dipole component, angle 1 the x component). 3 phase steps per angle, blocked, so a
  6-frame track has 3 frames per polarisation. The fringe modulation is taken as a known per-frame factor h_t = 1,
  i.e. averaged over the phases.
* **Mapping the in-situ ensemble tensors to a single-dipole K.** SIMCODE fits
  `M_a = (1 - gamma_s) diag((1 - zeta)/2, (1 - zeta)/2, zeta) + gamma_s e_a e_a^T`
  (gamma_s 0.068, zeta 0.356 for 1809). With `photosel_tensor` as written, `M_xx` of angle 0 would be at least
  `(1 - gamma)/3 = 0.311`; the fit gives 0.300. The extra z weight (M_zz 0.3316 > M_xx 0.3002) therefore has to be in K,
  not in the isotropic part. For single dipoles we use
  `K_a = (1 - eps) e_a e_a^T + eps (ck k_a k_a^T + (1 - ck) z z^T)`, i.e. an eps fraction of evanescent p-polarised
  excitation, with ck = |E_x|^2 / |E|^2 of the p wave = 0.1525. This value follows from the TIRF penetration depth:
  82 nm at 642 nm gives kappa^2 = (beta^2 - n^2) = 0.388, beta^2 = 2.157, ck = kappa^2 / (kappa^2 + beta^2).
  With uniformly distributed axes and order g, the ensemble is `(1 - gamma) I/3 + gamma K_a` with `gamma = 0.4 g^2`.
  Solving the diagonal in the (k_a, e_a, z) frame of both angles gives:

  | data | gamma_s, zeta (SIMCODE) | gamma (ours) | eps | sqrt(gamma / 0.4) = rms g |
  |---|---|---|---|---|
  | 1809 | 0.068, 0.356 | 0.120 | 0.38 | 0.55 |
  | 2.5 nM | 0.093, 0.398 | 0.238 | 0.53 | 0.77 |

  This reproduces both fitted per-angle tensors to 1e-7, including the zz entry. An eps of 0.4-0.5 p-polarisation
  is large for a nominally s-polarised setup. The zz excess may also come from something else: an anisotropic
  distribution of dye axes (more upright), or PSF-model error absorbed by M_zz (SAF / height distribution; the 2.5 nM
  PSF also has the strong blur). The ensemble data cannot tell these apart. We use the 1809 K (eps 0.38) as the main
  "in-situ K" and the pure s-polarised `K_a = e_a e_a^T` (eps = 0) as the optimistic bound. Only the difference
  between the angles carries information, `f_1 - f_0 = 3 g (1 - eps (1 + ck)) (mu_x^2 - mu_y^2)`, so eps = 0.38
  costs a factor 0.57 in excitation contrast relative to pure s.
  Under this model the ensemble implies an rms order of **g ~ 0.55** (1809). With SIMCODE's own parametrisation it is
  sqrt(0.068 / 0.4) = 0.41. These are averages: a mix of fixed and mobile dyes gives the same ensemble.

## 3. CRLB / fittability results (`study/orientation/crlb_orientation.py`)

Sweep: theta 0-90 (10 deg steps), phi 0-315 (8 values), z 0 / 50 / 100 / 200 nm, g 1 / 0.7 / 0.4 / 0.3 / 0.2 / 0.1,
500 / 1000 / 2000 photons, bg 10 (and 30). Parameters x, y, z, I, bg, the two tangent angles, g. Figures:
`fig_crlb_vs_theta.png`, `fig_crlb_vs_g.png`, `fig_iso_bias.png`, `fig_corr_z.png`. The whole sweep is in
`results/crlb_sweep.npz` / `crlb_summary.json` (not committed; the script regenerates them in ~40 min on a shared GPU).

**Headline (1000 photons / frame, bg 10, z = 50 nm, theta = 60 deg, median over phi; sd_phi / sd_theta in deg):**

| PSF | case | g = 1 | g = 0.7 | g = 0.4 | g = 0.2 |
|---|---|---|---|---|---|
| aberration-free, f 101 nm | single frame | 9.4 / 5.5 / 0.44 | 13.4 / 8.3 / 0.45 | 23.5 / 15.8 / 0.44 | 47 / 34 / 0.40 |
| 2.5 nM in-situ (f 101) | single frame | 12.2 / 8.7 / 0.29 | 17.8 / 12.1 / 0.29 | 32 / 20 / 0.30 | 67 / 39 / 0.30 |
| aberration-free, f 308 nm | single frame | 8.5 / 5.3 / 0.27 | 12.7 / 7.8 / 0.31 | 23.1 / 14.3 / 0.35 | 47 / 29 / 0.37 |
| 1809 in-situ (f 308) | single frame | 8.1 / 4.9 / 0.25 | 12.1 / 7.3 / 0.28 | 21.9 / 13.0 / 0.31 | 45 / 27 / 0.32 |
| 1809 in-situ | track, 6 frames, no excitation term | 3.3 / 2.0 / 0.10 | 4.9 / 3.0 / 0.11 | 9.0 / 5.3 / 0.13 | 18.4 / 10.8 / 0.13 |
| 1809 in-situ | track, 6 frames, in-situ K | 1.9 / 1.9 / 0.07 | 2.9 / 2.9 / 0.08 | 5.5 / 5.3 / 0.09 | 11.4 / 10.7 / 0.09 |
| 1809 in-situ | track, 6 frames, pure-s K | 1.5 / 1.7 / 0.05 | 2.4 / 2.6 / 0.06 | 4.8 / 5.0 / 0.07 | 10.3 / 10.5 / 0.08 |
| 2.5 nM in-situ | track, 6 frames, no excitation term | 5.0 / 3.6 / 0.12 | 7.3 / 4.9 / 0.12 | 13.1 / 8.3 / 0.12 | 27 / 16 / 0.12 |
| 2.5 nM in-situ | track, 6 frames, in-situ K | 2.3 / 3.1 / 0.07 | 3.4 / 4.3 / 0.07 | 6.4 / 7.1 / 0.07 | 13.2 / 13.3 / 0.08 |

(cells: sd_phi / sd_theta / sd_g.) "Track without excitation term" = the same brightness in every frame (free
intensity), i.e. single frame / sqrt(6). Photon scaling, 1809, g = 1: single frame 14.2 / 8.5 at 500 photons,
4.8 / 3.0 at 2000; track + in-situ K 3.2 / 3.3 at 500, 1.2 / 1.2 at 2000. bg 30 instead of 10 costs ~1.4x
(1809 single, g = 1: 11.6 / 6.9; track + K: 2.6 / 2.7).

Answers to the key questions:

* **Is phi measurable at ~1000 photons, in focus, from one frame?** Only for fixed or nearly fixed dipoles that are
  well out of the plane. At g = 1, theta >= 40 deg, sd_phi ~ 7-13 deg (1809: 12.7 at 40, 7 at 70-90; 2.5 nM:
  12-14). It diverges towards the pole (38 deg at theta = 10). At g = 0.4 it is already 22-32 deg. sd_g from one
  frame is 0.25-0.45 at any g, so one frame cannot tell a fixed dipole from a mobile one. The 1809 PSF, 200-300 nm
  defocused for these emitters, is slightly better than the 2.5 nM PSF near focus, despite the latter's
  astigmatism. Astigmatism helps z (the 2.5 nM sd_z is 26 nm at every theta, against 45-70 nm aberration-free at
  the same focus) but not the angles (2.5 nM sd_phi 12 vs 9.4 aberration-free; its blur costs more than the
  astigmatism gains). Defocus helps the angles more than astigmatism: aberration-free at f = 308 nm is ~10 %
  better than at 101 nm for phi and much better for theta near 0 and 90 deg.
* **theta-z coupling (SAF).** In the joint fit the correlation is moderate: |rho(theta, z)| <= 0.47 (1809) and
  <= 0.53 (2.5 nM). The strongest couplings are lateral: tilt vs x / y (|rho(a, x)| up to 0.89 near theta = 80-90,
  |rho(phi, y)| up to 0.78), the lobe shift of a tilted dipole. Fitting the orientation costs lateral precision.
  At theta = 60, g = 1 the joint sd_x is 14.6 nm (1809) / 19.5 nm (2.5 nM), against 11.3 / 13.9 nm for an isotropic
  emitter (5-parameter CRLB). sd_z is about the same (18 vs 20 nm and 26 vs 25 nm). sd_theta grows with height
  (1809, theta = 60: 4.9 at 0 nm, 8.5 at 200 nm; 2.5 nM: 8.2 -> 12.3, `fig_corr_z.png`), because the
  supercritical light that encodes the out-of-plane component decays.
* **Bias of the isotropic model on fixed dipoles** (`fig_iso_bias.png`; noise-free spots, 1000 photons, bg 10, best
  of a truth start and 6 z starts):

  | PSF, z = 50 nm, g = 1 | z bias vs theta 0 -> 90 deg | lateral bias along the tilt (max) |
  |---|---|---|
  | 1809 (f 308), tilt along x | -78 nm (theta 0), 0 near 35, +73 nm (theta 70) | +65 nm at theta 50 (also -25 nm at theta 0, from coma) |
  | 1809, tilt along y | -78 -> +27 nm | +48 nm at theta 40 |
  | 2.5 nM (f 101), tilt along x | -54 -> +4 nm | -49 nm at theta 40 |
  | 2.5 nM, tilt along y | -54 -> +15 nm | +37 nm at theta 40 |
  | aberration-free, f 101 | -150 nm (theta <= 30; the fit lands on the other side of focus), +22 nm for theta >= 60 | <= 3 nm |
  | aberration-free, f 308 | -83 -> +8, then a jump to +320 nm (other side of focus) for theta >= 60 | +57 nm at theta 40 |

  At g = 0.7 the biases are ~0.6-0.7 times these. The biases scale with the defocus. For the 1809 geometry (emitters
  200-300 nm below the focal plane) a fixed dipole at 40-60 deg is mislocalised laterally by 50-65 nm, 4-5 times its
  CRLB. If a substantial fraction of the SIMCODE dyes were fixed, the isotropic in-situ PSF would show this as
  excess lateral spread and a z spread correlated with brightness.
* **Excitation term.** On a 6-frame track (3 frames per polarisation) the brightness ratio between the angles
  measures `sin^2(theta) cos(2 phi)` and roughly halves sd_phi (1809, g = 1: 3.3 -> 1.9 deg with the in-situ K,
  1.5 deg with pure s) and also helps theta. Its contribution scales with g, so the relative gain is the same at all g.
* **At what g do the angles become meaningless?** Take sd > 20 deg as meaningless (the CRLB itself stops being
  trustworthy there; see the fits in §4). Single frame, 1000 photons: below g ~ 0.4-0.5 (phi) and ~0.3 (theta).
  6-frame track with the excitation term: below g ~ 0.1-0.15 (g = 0.1: 21-27 deg). The cost scales roughly as
  1/g. sd_g itself is almost independent of g (0.25-0.45 single frame, 0.07-0.13 per 6-frame track), so g = 0.2
  and g = 0.6 can be told apart only on tracks.
* **Angle-brightness contrast per track.** `c = (F_1 - F_0) / (F_1 + F_0)` between the two angles. Its CRLB for a
  6-frame track at 1000 photons / frame, bg 10 (per-angle intensities free, shared position) is **0.020-0.022** for
  all four PSFs. The spread of c over uniformly random axes (Monte Carlo):

  | g | sd_c, in-situ K (1809, eps 0.38) | sd_c, pure s | max abs(c), in-situ K |
  |---|---|---|---|
  | 1 | 0.43 | 0.71 | 0.83 |
  | 0.7 | 0.30 | 0.48 | 0.59 |
  | 0.55 | 0.24 | 0.38 | 0.46 |
  | 0.4 | 0.17 | 0.28 | 0.34 |
  | 0.3 | 0.13 | 0.22 | 0.25 |
  | 0.2 | 0.09 | 0.15 | 0.17 |

  SIMCODE sees per-docking-site (0.15-0.2) and per-dye (0.2-0.25) components in the angle contrast. These are
  ~10 times the photon-limited CRLB of one 6-frame track, so they are well above shot noise. They are also exactly
  the size expected from dipole orientation at g ~ 0.4-0.55 (in-situ K) or g ~ 0.25-0.35 (pure s). That is the same
  range as the rms g implied by the ensemble photoselection (0.55 / 0.41, §2). The two observations are mutually
  consistent with partly constrained dyes. The per-frame blinking and brightness fluctuation (SIMCODE simulator
  `fluct` 0.15-0.35 per frame) adds roughly 0.06-0.14 to sd_c per 6-frame track, so the per-dye term needs a few
  tracks per dye, and the per-site term needs averaging over several binding events.

## 4. Fitter validated on simulation (`study/orientation/fit_validation.py`)

300 simulated spots / tracks per row. bg 10, z = 50 nm, x / y uniform within +-0.5 px, Poisson noise.
`fit_orientation` with defaults (isotropic start, linear-M start plus mirror start, 50 LM iterations, z in
[0, 0.4] um). Axis errors in the tangent coordinates of the truth (a: polar, b: in-plane, deg); outliers = axis error
> 30 deg, excluded from sd / bias. Cells: sd / CRLB. Results in `results/fit_validation.json`.

| PSF | theta, phi, g | data | outliers | a | b | g | z (nm) | x (nm) | biases |
|---|---|---|---|---|---|---|---|---|---|
| 1809 | 60, 30, 1 | 1000 ph | 14% | 6.3 / 5.0 | 7.1 / 7.0 | 0.118 / 0.226 | 67 / 20 | 17.4 / 14.5 | a +1.8, b -0.0, g -0.066, z +14 |
| 1809 | 60, 30, 1 | 5000 ph | 0% | 1.6 / 1.6 | 2.2 / 2.2 | 0.037 / 0.065 | 6 / 6 | 4.4 / 4.5 | a +0.2, b -0.1, g -0.023, z -0 |
| 1809 | 60, 30, 1 | 6 x 1000 + K | 0% | 2.0 / 2.0 | 1.4 / 1.4 | 0.053 / 0.086 | 8 / 8 | 5.0 / 5.4 | a +0.2, b -0.4, g -0.031, z +1 |
| 1809 | 80, 120, 0.7 | 1000 ph | 13% | 11.7 / 13.7 | 8.8 / 10.1 | 0.159 / 0.204 | 41 / 15 | 16.1 / 15.9 | a +3.0, b -0.8, g +0.118, z +2 |
| 1809 | 80, 120, 0.7 | 5000 ph | 0% | 4.1 / 4.3 | 3.2 / 3.3 | 0.066 / 0.067 | 5 / 5 | 5.3 / 5.1 | a +0.2, b -0.0, g +0.015, z -0 |
| 1809 | 80, 120, 0.7 | 6 x 1000 + K | 0% | 5.9 / 5.4 | 1.9 / 1.9 | 0.078 / 0.077 | 6 / 6 | 6.5 / 6.4 | a +0.5, b +0.1, g +0.014, z +0 |
| 1809 | 40, 200, 1 | 1000 ph | 44% | 5.2 / 4.3 | 9.9 / 9.8 | 0.135 / 0.234 | 59 / 24 | 17.7 / 11.6 | a -1.5, b -0.7, g -0.097, z +2 |
| 1809 | 40, 200, 1 | 5000 ph | 2% | 1.4 / 1.5 | 2.9 / 3.0 | 0.037 / 0.072 | 6 / 7 | 3.4 / 3.8 | a -0.3, b -0.3, g -0.023, z -1 |
| 1809 | 40, 200, 1 | 6 x 1000 + K | 15% | 1.9 / 1.7 | 3.9 / 3.2 | 0.105 / 0.090 | 23 / 9 | 10.0 / 4.3 | a -0.1, b -0.8, g -0.043, z -1 |
| 1809 | 20, 45, 1 | 1000 ph | 0% | 5.5 / 6.6 | 6.0 / 6.7 | 0.065 / 0.162 | 20 / 23 | 9.0 / 8.0 | a -2.7, b -0.8, g -0.036, z -1 |
| 1809 | 20, 45, 1 | 5000 ph | 0% | 1.9 / 2.2 | 2.2 / 2.4 | 0.029 / 0.053 | 6 / 7 | 2.8 / 2.7 | a -0.7, b -0.4, g -0.020, z -1 |
| 1809 | 20, 45, 1 | 6 x 1000 + K | 0% | 2.0 / 2.6 | 1.6 / 1.6 | 0.035 / 0.066 | 9 / 10 | 3.4 / 3.5 | a -0.7, b -0.1, g -0.024, z -1 |
| 1809 | 60, 30, 0.4 | 1000 ph | 48% | 8.6 / 13.1 | 13.2 / 19.1 | 0.217 / 0.287 | 71 / 23 | 19.6 / 22.1 | a -2.1, b +2.2, g +0.246, z +12 |
| 1809 | 60, 30, 0.4 | 5000 ph | 6% | 4.2 / 4.4 | 6.1 / 6.2 | 0.088 / 0.090 | 7 / 7 | 7.5 / 7.0 | a +0.3, b +0.2, g +0.021, z -0 |
| 1809 | 60, 30, 0.4 | 6 x 1000 + K | 11% | 6.4 / 5.3 | 4.0 / 4.1 | 0.096 / 0.106 | 10 / 9 | 7.2 / 7.5 | a +0.7, b -0.5, g +0.015, z +0 |
| 2.5 nM | 60, 30, 1 | 1000 ph | 5% | 9.2 / 8.7 | 10.0 / 9.9 | 0.125 / 0.310 | 26 / 30 | 17.8 / 20.7 | a -0.3, b -2.3, g -0.066, z -4 |
| 2.5 nM | 60, 30, 1 | 5000 ph | 0% | 2.8 / 2.8 | 3.4 / 3.2 | 0.047 / 0.095 | 8 / 9 | 5.9 / 6.2 | a +0.2, b -0.3, g -0.030, z -1 |
| 2.5 nM | 60, 30, 1 | 6 x 1000 + K | 0% | 3.5 / 3.4 | 2.0 / 2.3 | 0.067 / 0.123 | 11 / 11 | 7.3 / 8.2 | a +0.0, b -0.9, g -0.050, z -1 |
| 2.5 nM | 80, 120, 0.7 | 1000 ph | 21% | 10.0 / 11.6 | 12.2 / 19.6 | 0.166 / 0.353 | 21 / 20 | 14.7 / 14.9 | a +0.4, b -1.0, g +0.156, z -1 |
| 2.5 nM | 80, 120, 0.7 | 5000 ph | 0% | 3.5 / 3.7 | 6.8 / 6.2 | 0.111 / 0.111 | 7 / 7 | 5.4 / 4.9 | a -0.2, b +0.8, g +0.020, z +0 |
| 2.5 nM | 80, 120, 0.7 | 6 x 1000 + K | 0% | 4.3 / 4.4 | 2.6 / 2.3 | 0.085 / 0.085 | 8 / 8 | 5.3 / 5.6 | a -0.1, b -0.2, g +0.006, z +1 |
| 2.5 nM | 40, 200, 1 | 1000 ph | 6% | 7.6 / 7.3 | 8.8 / 9.0 | 0.116 / 0.229 | 31 / 31 | 19.4 / 20.0 | a -0.0, b -2.0, g -0.066, z -3 |
| 2.5 nM | 40, 200, 1 | 5000 ph | 0% | 2.5 / 2.4 | 2.7 / 3.0 | 0.042 / 0.072 | 8 / 9 | 5.5 / 6.3 | a +0.0, b -0.4, g -0.028, z -1 |
| 2.5 nM | 40, 200, 1 | 6 x 1000 + K | 0% | 2.6 / 2.6 | 3.1 / 3.2 | 0.053 / 0.092 | 12 / 11 | 7.5 / 8.0 | a +0.1, b -0.8, g -0.032, z -1 |
| 2.5 nM | 20, 45, 1 | 1000 ph | 2% | 8.4 / 8.9 | 5.9 / 6.1 | 0.083 / 0.191 | 27 / 28 | 16.5 / 16.3 | a -0.7, b -0.7, g -0.043, z +2 |
| 2.5 nM | 20, 45, 1 | 5000 ph | 0% | 2.7 / 3.0 | 2.1 / 2.1 | 0.040 / 0.067 | 8 / 8 | 4.5 / 5.3 | a -0.3, b -0.1, g -0.024, z -0 |
| 2.5 nM | 20, 45, 1 | 6 x 1000 + K | 0% | 3.4 / 3.5 | 1.7 / 1.6 | 0.043 / 0.078 | 12 / 12 | 6.0 / 6.7 | a -0.4, b -0.3, g -0.026, z -0 |
| 2.5 nM | 60, 30, 0.4 | 1000 ph | 43% | 11.5 / 19.8 | 13.5 / 25.9 | 0.218 / 0.315 | 29 / 27 | 18.6 / 19.6 | a -1.0, b +0.4, g +0.260, z +4 |
| 2.5 nM | 60, 30, 0.4 | 5000 ph | 1% | 6.6 / 6.6 | 8.5 / 8.5 | 0.099 / 0.098 | 9 / 8 | 5.6 / 5.9 | a -0.3, b -0.6, g +0.027, z +1 |
| 2.5 nM | 60, 30, 0.4 | 6 x 1000 + K | 2% | 8.1 / 8.0 | 6.8 / 6.3 | 0.114 / 0.127 | 11 / 11 | 8.1 / 7.9 | a +1.2, b -0.9, g +0.026, z -0 |

* **Asymptotic regime (5000 photons single frame, or 6 x 1000 photons with K).** Bias ~0: angles within +-1 deg (one
  +1.2), z within 1 nm. The sd equals the CRLB within ~10 % for the angles, z and x. The exception is g at g = 1,
  where the clamp truncates: sd ~0.5-0.6 of the CRLB with a -0.02 to -0.05 bias. Outliers 0-6 %, except two 1809
  rows (15 % at 40 / 200 / 1 on a track, 11 % at g = 0.4).
* **Single frames at 1000 photons are not asymptotic.** 0-48 % of the fits land in a second solution, mostly the
  mirrored tilt (theta -> 180 - theta) with a shifted x / y / z. For the 1809 PSF the z sd is 2-3.5x the CRLB
  (bimodal z). g is biased (+0.25 at g = 0.4, -0.07 to -0.1 at g = 1), and the angle sd of the inliers is at or below
  the CRLB only because the outliers are removed. The 2.5 nM PSF has fewer outliers (2-6 % at g = 1) thanks to its
  astigmatism.
* Speed (CPU, shared): ~95 s per 300 fits with two starts. GPU batches (500) are much faster when the GPU is free.

## 5. Vortex PSF: reproduction of Hulleman et al., Nat. Commun. 12, 5934 (2021), simulation part

Script `study/orientation/vortex_repro.py`; figures `fig_vortex_images.png`, `fig_vortex_crlb.png`; numbers in
`results/vortex_repro.json`. Their setup: NA 1.45, 597.5 nm, n_med 1.33, pixel 65 nm, ROI 15 x 15, 4000 signal
photons, 10 bg / px, Poisson noise. Their coverslip has n = 1.523 and their immersion 1.518. Our model assumes an
index-matched coverslip, so we use 1.518 for both. The 0.005 difference adds a small depth-dependent spherical
aberration in their model and none in ours. Their g2 is our g. Photons: N is the number over the whole image
plane for that orientation and height, `I = N / S(M, z)`, with S the total of our PSF over a 7.9 um field. The angle
CRLBs do not depend on how the intensity is parametrised.

**(i) Fig. 1b/c images.** Reproduced: a ring for theta = 10 deg and a compact spot for 80 deg with the standard PSF.
With the Vortex PSF: a central spot with a ring at 10 deg, an off-centre lobe at 45 deg, and a ring with a dark
centre whose brightness varies with azimuth at 80 deg. Handedness: their figure images match **charge -1** in our pupil
coordinates (p_x along columns, p_y along rows) when shown with row 0 at the top (MATLAB `imagesc`). Equivalently,
charge +1 with the image y axis mirrored. The CRLBs are the same for both signs (mirror symmetry). Their real data
prefer **charge +1** by a large likelihood margin (section 7), consistent with their code read in the DIPimage
convention, so the figure's display axes differ from the camera axes.
The Vortex peaks are 0.41-0.45 of the standard theta = 80 peak. Their colour scale ends at 0.35 and their theta = 80
Vortex image is saturated, which is consistent. With the standard PSF our ROI holds 64-78 % of N. They quote 44-46 %;
see the photon-scale discrepancy below.

**(ii) Fig. 1e/f CRLB vs theta** (phi = 45, g2 = 1, emitter at the coverslip, in focus):

| curve | paper (read off) | ours, N = 4000 | ours, N = 1700 |
|---|---|---|---|
| standard azimuth, minimum at 90 deg | 4.7 | 2.5 | 4.75 |
| standard azimuth = 15 deg at theta | ~35 | 23 | 37 |
| Vortex azimuth, minimum (theta ~50 / 125) | 4.3 | 2.6 | 4.6 |
| Vortex azimuth at 90 deg (local bump) | 6.3 | 3.1 | 5.4 |
| Vortex azimuth = 15 deg at theta | ~10 | 7 | 13 |
| standard polar, minimum (theta ~52) | 2.4 | 1.46 | 2.6 |
| standard polar at 90 deg (peak) | 8 | 9.0 | 17 |
| Vortex polar at 0 / ~35 / 90 deg | 2.6 / 3.5 / 2.2 | 1.65 / 2.2 / 1.44 | 3.1 / 4.1 / 2.5 |

The shapes reproduce: the standard azimuth CRLB diverges towards the poles, the Vortex keeps the azimuth below
~5 deg over 20-160 deg with a bump at 90, the Vortex polar CRLB is flat, and the standard polar CRLB has minima near
52 / 128 and a peak at 90. The level does not. At N = 4000 our curves are ~0.6 times theirs. They match within
~15 % at N ~ 1700, i.e. our photon count must be ~2.3 times lower to agree. One exception is the standard polar peak
at exactly 90 deg. There, in focus and without aberrations, the derivative with respect to theta is almost degenerate
with an x / y shift, so the value depends on the fine details of geometry and sampling. It matches their 8 deg at
N = 4000, not at 1700. The most likely cause of the factor is the photon normalisation. Their Eq. (1) weights the
fixed and free parts by g2/3 and (1 - g2)/3, so the normalisation of H_fixed and H_free relative to N is not
explicit; their quoted ROI fraction of 0.44-0.46, against our 0.64-0.78, also points to a different "whole-FOV"
reference. A different emitter depth would also change the level (next point).

**(iii) Text numbers, uniform orientations, g2 = 0.75, 4000 / 10.** 1000 instances: x, y uniform within +-1 px, focal
plane uniform within +-65 nm of the emitter. Two z definitions: the emitter height, and the focal-plane position (a
focus / stage scan). Their geometry is not fully specified, so three geometries:

| quantity | paper | coverslip, in focus | 500 nm deep, focus 500 nm | 500 nm deep, best focus (730 nm) |
|---|---|---|---|---|
| Vortex sigma_xy (nm) | 5.6 | 3.9 | **5.6** | 3.9 |
| Vortex sigma_z (nm), emitter / focal plane | 27 | 14.7 / 22.4 | 13.7 / 18.6 | 19.7 / **28.9** |
| Vortex sigma_phi (deg), median / mean over 20-160 | 5.5 | 3.8 / 3.9 | 4.3 / 4.4 | 4.3 / 4.4 |
| Vortex sigma_theta (deg) | 3.1 | 2.3 | **3.1** | 2.2 |
| Vortex sigma_g2 | 0.08 | 0.055 | **0.078** | 0.060 |
| Vortex sigma_z, g2 = 1 (nm), focal plane | 23 | 16.1 | 16.3 | 20.8 |
| Vortex sigma_z, g2 = 0 (nm), focal plane | 49 | 68 | 18.7 | 58 |
| free rotator sigma_xy, Vortex vs standard (nm) | 5.9 vs 4.5 | 4.0 vs 2.8 | 5.0 vs 3.3 | 3.7 vs 2.6 |

At the coverslip and in focus, every number is 1.3-1.5 times smaller than theirs, the same photon-scale factor as in
(ii). At 500 nm depth with the focal plane at the nominal 500 nm (defocused by the index mismatch), sigma_xy,
sigma_theta and sigma_g2 match theirs exactly, but sigma_z and sigma_phi are 20-30 % lower. At the best focus,
sigma_z matches for g2 = 0.75 and 1. The ordering and ratios are reproduced in every geometry: Vortex vs standard,
the g2 dependence (sigma_z grows from g2 = 1 to 0), and the Vortex costing 1.3-1.5 times in sigma_xy for free
rotators. The absolute values agree within the spread caused by the unstated geometry (depth, focus, z definition,
photon normalisation).

**(iv) MLE fits reach the CRLB.** Vortex spots, 4000 / 10, emitter 30 nm above the coverslip, in focus, x / y within
+-0.5 px, 300 spots per orientation (`fit_orientation`, isotropic start + linear-M start + mirror start, 50 LM
iterations). Errors of the axis in the tangent coordinates of the truth (a: polar, b: in-plane):

| theta, phi, g2 | sd a / CRLB (deg) | sd b / CRLB (deg) | sd g2 / CRLB | sd x / CRLB (nm) | sd z / CRLB (nm) | biases |
|---|---|---|---|---|---|---|
| 30, 45, 0.75 | 2.74 / 2.77 | 2.28 / 2.27 | 0.042 / 0.041 | 5.0 / 4.9 | 14.2 / 14.9 | a -0.17, b -0.17 deg, g +0.005 |
| 60, 120, 0.75 | 2.43 / 2.38 | 2.89 / 3.00 | 0.057 / 0.058 | 3.4 / 3.4 | 12.7 / 12.5 | < 0.05 deg, g +0.008, z -1.3 nm |
| 85, 20, 0.75 | 2.18 / 2.01 | 4.32 / 4.05 | 0.065 / 0.067 | 3.3 / 3.4 | 11.0 / 11.0 | < 0.1 deg, g +0.009 |
| 60, 120, 1 | 1.61 / 1.80 | 2.25 / 2.22 | 0.032 / 0.057 | 2.8 / 3.1 | 9.5 / 9.6 | a -0.28 deg, g -0.019 (boundary) |
| 15, 80, 1 | 1.93 / 1.92 | 1.60 / 1.60 | 0.024 / 0.044 | 3.8 / 3.8 | 16.1 / 16.9 | a +0.21 deg, g -0.015 (boundary) |

No outliers (axis error > 30 deg) in 1500 fits. Precision equals the CRLB within the sampling error (+-4 % for 300
spots). The only exception is g2 at the boundary g2 = 1, where the clamp truncates the estimates: sd below the CRLB
and a small negative bias. This confirms their claim that MLE with the Vortex PSF reaches the CRLB. CPU, 300 spots
in ~2 min.

## 6. Go / no-go for step 4 (real SIMCODE data)

**Go, for per-track fits; no-go for per-frame orientation.**

* One frame at ~1000 photons cannot tell fixed from mobile dyes (sd_g 0.25-0.45). For g <~ 0.5 its angles are
  > 20 deg, and the fit is bimodal (mirror tilt, 5-48 % outliers). Per-localisation orientation is not worth it.
* A 6-frame track with the excitation term (3 frames per pattern angle) reaches sd_phi ~2 deg / sd_theta ~2-3 deg /
  sd_g ~0.07 for fixed dipoles, ~5-6 deg / 5-7 deg / 0.09 at g = 0.4 and ~11-13 deg at g = 0.2. The fitter reaches
  these bounds (§4). Real binding events are longer than 6 frames, which helps further. The excitation term halves
  sd_phi and should be in the model. The in-situ K (eps 0.38) costs ~25 % against pure s.
* The ensemble photoselection (rms g ~0.41-0.55) and SIMCODE's per-site / per-dye angle-contrast spread (0.15-0.25)
  point to g ~ 0.3-0.55 on average. That spread is ~10x the photon-limited CRLB of the contrast for one 6-frame
  track (0.02), so **a per-emitter contrast of that size is measurable**, per track and even more after averaging
  binding events per site. Within that g range the full per-track orientation is meaningful (sd ~5-10 deg).
* Risks for step 4: (1) The size of the excitation term depends on how the zz excess of the ensemble tensors is
  split between p-polarised excitation, orientation distribution and model error. Check this by fitting eps per
  dataset, or use pure-s and in-situ K as brackets. (2) Fixed dipoles bias isotropic localisations by up to
  50-65 nm laterally and +-75 nm in z (1809 geometry). A per-track orientation fit gives a cleaner per-localisation
  test of this than the isotropic in-situ PSF does. (3) The fringe modulation per phase step (h_t) must be supplied
  from the pattern estimate. (4) Blinking and per-frame brightness fluctuations add noise to the excitation contrast
  (~0.06-0.14 per 6-frame track at SIMCODE's fluct 0.15-0.35); use per-frame intensities or a fluctuation term.
  (5) The 2.5 nM PSF's large blur costs ~30 % in the angle CRLB, and if the blur is drift / vibration it may not be
  the same for all emitters.

## Figures (`study/orientation/`)

* `fig_crlb_vs_theta.png`: single-frame sd_phi, sd_theta, sd_z vs theta (g = 1, 1000 photons, z = 50 nm) for the
  four PSFs.
* `fig_crlb_vs_g.png`: sd_phi, sd_theta, sd_g vs g (theta = 60): single frame, 6-frame track, track + in-situ K,
  track + pure-s K, for the 1809 and 2.5 nM PSFs.
* `fig_iso_bias.png`: z and lateral bias of the isotropic-model fit of fixed dipoles vs theta.
* `fig_corr_z.png`: |rho(theta, z)| and |rho(phi, x)| vs theta; sd_theta vs emitter height.
* `fig_vortex_images.png`, `fig_vortex_crlb.png`: Vortex reproduction (Fig. 1b/c, 1e/f of Hulleman et al.).
* `fig_pipeline_sim.png`: the pipeline on simulations: z vs focus and the per-slice theta error of the PMMA z-stack,
  and the lambda-DNA Delta phi and g histograms against the truth.
* `fig_real_pmma.png`: real PMMA z-stack, per-plane z / theta / phi / g of six molecules.
* `fig_real_dna.png`: real lambda-DNA, Delta phi to the local axis and g: ours vs their localisations.

## 7. Vortex orientation pipeline: code, simulations, real data (Hulleman et al. 2021)

**Code.**
* `torch_psf/vortex.py`
  * `vortex_psf`: a VectorialPSF with the spiral plate as a fixed `pupil_phase`. `vortex_phase` takes a plate
    centre, the number of quantisation levels and the phase depth.
  * `calibrate_beads`: the ZStackEstimator Zernike fit on top of the plate, one run per charge; the best NLL wins.
    Optionally the plate centre is fitted, alternating with the Zernikes.
  * `fit_dipole_stacks`: plate calibration in situ from z-stacks of fixed single molecules. Per molecule: x, y, a
    focus offset, the axis and g. Per plane: photons, background and a shared drift. Global: the plate centre,
    Zernikes, emitter height and medium index.
* `torch_psf/orientation.py`
  * `OrientationEstimator`: spot detection → `fit_orientation` → a `Dataset` with these columns:
    `theta_deg`, `phi_deg`, `g`, `mu`, `crlb_theta_deg`, `crlb_phi_deg`, `crlb_inplane_deg`, `crlb_g`, `photons`
    (in the ROI), `intensity`, `bg`, `chi2`, `loglik`.
  * `FocusZ`: the model with z = the emitter's position relative to the focal plane, at a fixed emitter height. This
    is needed whenever the emitters sit at an interface with n_med < NA: moving a supercritical emitter below the
    interface is unphysical.
  * `fit_orientation(zgrid=...)`: extra starts on a z grid, because for the Vortex PSF the isotropic fit is a poor
    guide to z.
* Tests: the estimator on a small movie, the charge choice of the bead calibration, and `FocusZ`.

**Simulations** (`study/orientation/vortex_pipeline_sim.py`; their λ-DNA optics: NA 1.47, 597.5 nm, 65 nm pixels,
camera offset 106 ADU and 0.33 e/ADU):
* **Bead calibration** with the plate (charge −1, centre offset (0.03, −0.02), 64 levels, depth 593 / 597.5):
  * Zernikes recovered within 1.2 mλ (rms error 2 mλ of 52 mλ).
  * Charge chosen correctly, ΔNLL 0.84 per pixel.
  * Centre after two rounds: (0.021, −0.016).
* **PMMA z-stack** (11 planes, g = 0.9, 5e3–4e4 photons, bg 20), fitted with the calibrated model:
  * Per-molecule spread over the |z| ≤ 300 nm planes: σ_θ 0.86° (CRLB 1.04°), σ_φ 1.51° (CRLB 1.69°).
  * rms bias 0.5° / 0.9°, no outliers.
  * z slope −0.94, 94 ± 6 nm per 100 nm step. Only 12 of 20 molecules passed detection at threshold 8.
* **λ-DNA-like data** (2884 emitters on lines, dipoles perpendicular to the line with a ±25° tilt out of the plane,
  g ~ 0.8, ~4600 photons, bg 10):

  | quantity | estimated | truth |
  |---|---|---|
  | Δφ to the DNA axis, median / MAD | 90.4° / 6.2° | 90.1° / 5.5° |
  | g, median | 0.815 | 0.813 |
  | axis error, median / 90th percentile | 3.1° / 6.8° | — |

  The lateral error is 7.8 nm rms against a CRLB of 3.4 nm, likely from the residual plate-centre error of the
  calibration.

**Real data** (4TU 10.4121/13031837, `study/orientation/vortex_real.py`; untrusted input, read with tifffile / h5py,
run with `python -I`):
* **Data details.** The bead z-stacks (51 planes, 100 nm) are taken **without** the vortex plate, as in their own
  code (`set_parameters_zstack_bead.m`: doetype none). The plate therefore has to be calibrated in situ on the
  single molecules. Camera settings from their code: beads offset 104.2 ADU, 0.52 e/ADU; single molecules 106 ADU,
  0.33 e/ADU. Background noise gives 0.30 e/ADU for the PMMA stack, consistent. NA 1.47 as in their code.
* **Beads.** Some of the brightest beads are saturated (65535 ADU), so I blanked them. The beads are in focus at
  plane ~22 of 51, not at the stack centre, so I used the 31 planes around it, the central 1024 × 1024 field, the
  bead size as a 40 nm Gaussian blur and drift. Variants were chosen by the held-out stack:
  * Scan sign +1 with a fitted radial apodization wins: held-out NLL −9509, against −9482 without the apodization
    and −9472 with NA fitted instead (NA fit → 1.30).
  * The apodization amplitude at the pupil edge is 0.36 (exp(−0.58 ρ² − 0.29 ρ⁴ − 0.16 ρ⁶)). This is a strong
    high-NA loss, possibly from the 4f relay.
  * χ² stays at ~100 on these very bright beads (~15 000 photons / px at the peak) with constant aberrations. The
    beads cover the whole field, and their aberration map (NAT) is field dependent.
* **Plate (in situ).** 16 bright PMMA molecules × 17 planes, `fit_dipole_stacks`:
  * **Charge +1** in our convention wins clearly. (Their code's `doevortexflip = 1` with x = first array index =
    column, DIPimage convention, is also +1.) The stage sign is +1.
  * The decisive finding: the molecules in the thin PMMA film behave as at a **glass / air interface**. n_med 1.0
    instead of 1.49 lowers the NLL by 1.4 per pixel and the joint-fit χ² from 7.0 to 4.8, and g becomes consistent
    (0.75–0.84 instead of 0.1–0.76). With n_med, the emitter height, the Zernikes and the centre all fitted:
    n_med = 0.955, height 0, centre (−0.026, 0.013) pupil radii, χ² 2.06, g 0.85–0.94.
    (Their simulations and λ-DNA settings use 1.333; their PMMA settings are not in the released code.)
* **PMMA through-focus validation.** 34 molecules; every plane fitted independently with `FocusZ`, z = defocus:

  | quantity | ours | paper |
  |---|---|---|
  | σ_θ / σ_φ per molecule over the \|z\| ≤ 300 nm planes, mean | 2.36° / 2.28° | 1.8° / 2.3° |
  | σ_θ / σ_φ, median | 1.96° / 1.72° | — |
  | CRLB θ / φ | 1.09° / 1.16° | 1.5° / 1.8° |
  | spread / CRLB | ~2.0 | 1.2–1.3 |
  | z vs stage slope | −0.88 ± 0.16 | −0.99 |
  | step per 100 nm | 88 nm | 97 ± 11 nm |
  | g, mean ± sd over molecules | 0.90 ± 0.05 | 0.86 ± 0.03 |
  | Vortex axis vs standard-PSF axis (same molecules, standard stack fitted per defocused plane): Δθ / Δφ | −1.1 ± 5.9° / −1.2 ± 9.1° | −0.2 ± 1.2° / −0.4 ± 1.4° |

  * 4 of 233 planes are outliers. Our molecules are brighter, hence the smaller CRLB.
  * Our Vortex vs standard comparison is noisier because their standard-PSF reference is a full z-stack fit.
  * The precision gap to the CRLB (2× vs their 1.2–1.3×) and the z scale (0.88) point at the remaining model error:
    the joint χ² is 2. Candidates are the field dependence of the aberrations (constant here), the thin-film optics
    (a single medium of n ≈ 1 approximates glass / PMMA film / air) and the emission spectrum (one wavelength).
* **Re-orientation movie** (ATTO 565 on glass in air, 900 ms frames, TIRF, central 1024 × 1024 field, bright spots
  only (threshold 20): 4935 spots, 225 molecules tracked over ≥ 10 of the 23 frames). `FocusZ` model with n_med 1.0,
  emitter at the glass and the in-situ plate:
  * χ² median 1.14: the model fits well.
  * g median 0.88 (paper's three molecules 0.75–0.85).
  * Photons median 13.7k, background 60 / px (paper 3–6 × 10⁴ and 80–91 for their three bright molecules).
  * 80 molecules show at least one axis jump > 20° between frames.
  * On the stationary stretches the frame-to-frame axis change is median 2.3°, rms 5.2°. Two independent estimates
    with the CRLB would give rms √2 × 2.7° = 3.8°, so the single-frame axis precision is ~1.4 × the CRLB, close to the
    paper's "within 30 % of the CRLB".
* **λ-DNA** (`stk_0001`, frames 0–600, the 256 × 256 px crop densest in their localisations, water n 1.333, `FocusZ`
  with the emitter at the coverslip, the in-situ plate): 17 763 spots fitted, 15 899 kept by their χ² filter
  (0.75–3); χ² median 1.55. Δφ is measured to the local DNA axis, taken from the PCA of the localisations within
  3 px, on the 6370 spots where it is defined. Against their localisations (4TU hdf5, same crop, frames 0–600;
  `fig_real_dna.png`):

  | quantity | ours | theirs | paper |
  |---|---|---|---|
  | Δφ to the DNA axis, median / MAD | 97.6° / 29.8° | 89.1° / 33.5° | 82 ± 17° (one strand) |
  | g, median | 0.78 (a pile-up at the g = 1 bound) | 0.75 | peak 0.8 |
  | CRLB median: xy / in-plane / θ / g | 6.0 nm / 3.9° / 3.6° / 0.072 | — | — |

  * Per-localisation matching with their table failed. The spatial densities agree (correlation 0.85 for the same
    600 frames, no flip or transpose), but no frame offset pairs the localisations. Their frame numbering does not
    map onto stk_0001 as I read it; this is open.
  * Their 'photons' is the whole-field N (median 4620). Ours is the signal in the 15 × 15 ROI (median 1780).
  * z is defocus relative to our model focus (median −32 nm). Theirs is a stage-type z (median +58 nm).

**Status of the real-data pipeline.**
* **Works:** bead Zernikes, then the in-situ plate (charge, centre, Zernikes, n_med, height), then per-spot
  orientation with CRLBs on the PMMA stack, the re-orientation movie and λ-DNA. g and Δφ are in the paper's range,
  and the z tracking is linear.
* **Open:**
  1. Precision is 1.4–2 × the CRLB, and the z scale is 0.88 against their 0.99. Next steps: field-dependent Zernikes
     (FieldPSF; their NAT map), a two-layer glass / PMMA / air model instead of n_med ≈ 1, and polychromatic
     emission.
  2. The pile-up at g = 1: a soft bound, or report g as is with its CRLB.
  3. The frame mapping to their λ-DNA hdf5 for a per-localisation comparison.
  4. Throughput: about 17 spots / s on the shared GPU with 6 starts per spot. The full movie (~0.5 M spots) needs
     fewer starts or a spline / lookup model.

## 8. Rebase onto master 4b0ffc3, model refinements, the paper's baseline

**Rebase.** `dipole-orientation` now sits on master 4b0ffc3. The old head is tagged
`archive/dipole-orientation-0947`. `VectorialPSF.forward` of master is split into `_fields()` and `_image()`, with
`pixel_integration` in `_image()`, plus a one-line hook `_emitter_pupil()`.
* Checked against master: `render`, `render(grad=True)`, `forward(dipM=...)` and the `to_spline` knots differ by
  exactly 0, for 'box', 'spectral' and 'gauss_legendre', with and without blur.
* `dipole_basis()` works in every pixel_integration mode (tested).
* Full suite on CPU: 191 passed, 32 skipped (master 177 + 14 orientation tests); `tests/test_layered.py` adds 3 more.

**PMMA precision gap: layered sample, field-dependent Zernikes** (`vortex_real.py` stages `layer` and `field`).
* **(a) Layered sample.** `torch_psf/layered.py` models a coverslip / film / top-medium stack, checked in two limits:
  * without a top interface it reproduces the plain model;
  * as the film thickness goes to 0 it equals an emitter in the top medium, with the z dipole weaker by (n_film / n_top)⁴.

  On the PMMA stacks a glass / PMMA (1.49) / air model drives the fitted film thickness to its lower bound 0 from
  any start (10–120 nm). It is slightly worse than the single effective medium:

  | model (joint fit) | NLL / px | χ² | per-plane σθ / σφ | z slope |
  |---|---|---|---|---|
  | effective medium n 0.94 | −269.12 | 2.05 | 2.35° / 2.27° | −0.875 |
  | layered | −269.06 | 2.11 | 4.38° / 2.38° | −0.84 |

  The effective medium index is kept as a parameter.
* **(b) Field-dependent Zernikes.** `FieldZernikes` (modes 5, 6, 7, 8, 11; orders 2, 2, 1, 1, 1; ridge prior 0.05 λ)
  is used inside `fit_dipole_stacks`, and spots get per-spot `zcoef_offset`. `field.py` was not edited.
  * Training is on the even-numbered molecules; validation is on the odd ones, which were held out.
  * Joint-fit χ² falls from 1.72 to 1.64. The fitted field variation is 30–58 mλ over the field.
  * Held-out per-plane σθ falls from 2.08° to 1.66° and σφ from 2.26° to 1.89°, against a CRLB of 1.06° / 1.12°:
    the ratio to the CRLB goes from 1.9–2.0 to 1.6–1.7. The z slope moves from −0.913 to −0.924.
  * (c) Polychromatic emission: implemented (`orientation.Polychromatic`) but not run, per the change of priority.
* **g at the bound.** The Dataset now has `g` from a fit whose bounds (`g_limits`) can be widened, e.g. (−0.5, 1.5), and
  `g_display`, clamped to [0, 1].
  * Simulation (Vortex PSF, 4000 photons, bg 10, true g = 1): clamped fit bias −0.018 and sd 0.028 against a CRLB of
    0.056; unconstrained fit bias +0.008 and sd 0.054, which matches the CRLB.
  * At true g = 0.5, the unconstrained sd (0.089) is larger than the CRLB (0.058): a few fits wander, and this is not
    resolved.

**Baseline with the paper's processing** (`vortex_baseline.py`; settings read from their MATLAB code):
* **Optics and sampling:** NA 1.47; n_imm 1.518; the 1.523-design / 120 µm working-distance immersion mismatch
  phase W_mis (or 1.5209); 597.5 nm; PSF point-sampled at the pixel centres with a 32 × 32 pupil.
* **Plate and aberrations:** 64-level plate with phase depth 593 nm, charge +1; their NAT aberration map applied per
  spot as fixed `zcoef_offset`. NAT field coordinates: their xn is the image row and yn the column (their array
  layout). Zernike (n, m) maps to Noll 5, 6, 7, 8, 9, 10, 11, 13, 12, 17, 16, 22.
* **Detection:** FastPeakFind (threshold 25, 3 × 3 median, 7 × 7 Gaussian, weighted centroids), 15 px edge, spots with
  a neighbour closer than 5.3 px dropped.
* **Fit:** their moment-based initial values, one start at z = 0, 50 LM iterations, g in [0, 1].
* **Filters:** their outlier rules and χ² between 0.75 and 3.
* **g2 convention:** their g2 weights the free and fixed parts each normalised by its pupil energy. Converted from our
  g as g2 = g n_fix / (g n_fix + (1 − g) n_free); the fits themselves are identical.

| PMMA through focus (their 11 slices) | slope | σθ / CRLB | σφ / CRLB | g2 | χ² |
|---|---|---|---|---|---|
| paper | −0.99 | 1.8° / 1.5° | 2.3° / 1.8° | 0.86 ± 0.03 | — |
| their settings (n_med 1.49, W_mis, NAT, z0 = 0) | −0.07 (z not tracked) | — | — | 0.69 | 2.91 |
| + starts on a z grid | +0.15 | 9.0° / 2.0° | 10.1° / 3.1° | 0.80 | 2.48 |
| + n_med 1.0, z grid | −0.29 | 3.4° / 1.1° | 8.3° / 1.8° | 0.80 | 2.31 |
| + n_med 1.0, z grid, without W_mis | −0.79 | 4.7° / 1.2° | 4.9° / 1.4° | 0.84 ± 0.09 | 2.05 |
| ours: in-situ plate (centre, Zernikes, n_med, apodization), z grid | −0.88 | 2.35° / 1.10° | 2.27° / 1.16° | (g 0.90) | 2.05 |

Causes of the gaps, as far as isolated:
1. **The single z = 0 start** of their initial values does not follow the defocus. Their published slope implies
   their per-slice fits were started differently, or fitted the stack jointly (their code has a 'zstack' mode).
2. **n_med 1.49 vs ≈ 1** (thin film under air).
3. **The immersion-mismatch phase as I implement it from their code** (rms 0.22 λ, mostly Z11 −0.18, Z22 −0.11;
   0.13 λ with 1.5209) consistently makes the fits worse. Their NAT spherical (+56 mλ) alone equals our in-situ
   estimate (+57 mλ). Either I misread how W_mis enters, or their bead calibration absorbed it differently. **Open.**
4. **The plate centre and the remaining Zernikes**, fitted in situ in our pipeline, close most of the rest.

λ-DNA (stk_0001, frames 0–600, 256² crop, their detection: 14 896 spots, against 13 342 in their hdf5 for the same crop
and frames):

| | Δφ to the DNA axis (median / MAD) | g2 peak / median | N median | FWHM across the strand | χ² |
|---|---|---|---|---|---|
| paper (one strand, drift-corrected) | 82° / 17° (87° / 18° in other work) | 0.8 | 4600 | 20 nm | — |
| their hdf5, same crop, our axis estimate | 95.7° / 35.7° | 0.83 / 0.75 | 4620 | — | — |
| our baseline (their settings) | 90.5° / 27.7° | pile-up at 0 / 0.66 | 5320 | 68 nm | 1.83 |
| our baseline without W_mis | 93.9° / 25.1° | pile-up at 0 / 0.62 | 3280 | 40 nm | 1.88 |
| our pipeline (section 7) | 97.6° / 29.8° (radius 3 px axis) | g median 0.78 | — | — | 1.55 |

* Δφ and its MAD are dominated by how the DNA axis is estimated: local PCA within 8 px against their spline along a
  hand-picked strand. On the same crop and the same axis estimate, theirs and ours agree within a few degrees.
* The g2 pile-up at 0 in the baseline, where their hdf5 peaks at 0.83, is the clearest remaining discrepancy. It goes
  with the baseline model's larger χ² (1.8–1.9 against 1.55 for our in-situ calibrated model).
* The strand width (40–68 nm) is without drift correction or a spline fit, so it is not comparable yet.
