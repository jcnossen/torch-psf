# Field-dependent PSF study, stage 1: FD-DeepLoc data, conventions, measurements

Goal: show that torch-psf measures field-dependent aberrations on real data, using the public FD-DeepLoc data set
(Fu et al., Nat. Methods 2023, "Field dependent deep learning enables high-throughput whole-cell 3D super-resolution
imaging"; Zenodo 6547064). It has its own per-pixel aberration map, built from bead stacks. This stage uses the library as it is; scripts are in
`study/fieldpsf/`, outputs (json, figures, logs) in `~/fieldpsf_out/` (not in the repo).

## Summary

- **Conventions** (section 1): aber_map channels 0-20 = 21 Zernikes in their own (n, m) order (Noll 5, 6, 7, 8, 11, 9,
  10, 13, 12, 17, 16, 22, 15, 14, 19, 18, 23, 24, 30, 29, 37), unit-rms normalised, in waves (nm / 680); channels 21-22
  = a fixed Gaussian blur sigma (px). Their pupil x is the image row: their coefficients map onto ours by a signed
  permutation (`common.their_to_ours_matrix`). objStage = -focus. NA 1.35, n = 1.406 (silicone), lambda 680 / 670 nm,
  pixel 108 (calibration) / 110 nm (localisation), Prime 95B sCMOS, offset 100 ADU, gain 0.818 e-/ADU (photon
  transfer). Bead stacks 61 x 100 nm, focus moving up with the frame; movie z = +-3 um around the focal plane.
- **Beads vs FD-DeepLoc** (section 4): 57 beads of 3 stacks, median per-bead rms difference **0.0037 waves**
  (2.5 nm) over 21 modes; per-mode median |diff| <= 0.008 waves; 11 beads differ by 0.01 - 0.12 waves (flat
  likelihood / convergence, both fits). The alternative conventions are clearly worse (0.058 - 0.148 vs 0.034 rms).
  Ours vs the movie's example map at the bead positions: 0.027 waves rms, r 0.66 - 0.90 for the modes that vary.
- **In situ per tile** (section 5): the global in-situ PSF beats the bead PSF by 9.9 NLL per held-out spot; every
  tile PSF beats the global one (4 x 4: mean -14.5 per spot, 16 / 16 tiles; 6 x 6: mean -16.0, 36 / 36), and beats
  FD-DeepLoc's map averaged over the tile (map: -5.5 / -7.6). A control fit with the same spots / iterations from the whole field equals the global PSF (+-0.5), so the tile gain is spatial (> 15 standard errors in every tile). **Field dependence
  matters for this data set**, most at the field edges. Tile Zernikes follow their map's spatial pattern for
  astigmatism, coma, spherical and 2nd / 3rd coma (r 0.44 - 0.75 over 36 tiles); in situ has global offsets of 0.02 -
  0.04 waves from beads and map (coma, trefoil x, sphericals).
- **For the library stage** (sections 6, 8): their 21 modes (Noll 5-21 is not enough); a per-mode polynomial of
  order 1, order 2 for spherical / astigmatism; matched-filter detection and more z starts for engineered PSFs.


## 1. Conventions (sources)

Sources: their calibration files in the bead-stack zip (`*_3dcal.mat`, `*_maps.mat`, `*_aber_map.mat` of stacks 20,
30 and 61; read with scipy / h5py), the Micro-Manager metadata, the FD-DeepLoc code on GitHub
(Li-Lab-SUSTech/FD-DeepLoc, main branch and the May 2022 commits 215eb028b2 / 1be6b7dcb3 of the TOM20 demo), and the
bioRxiv preprint 10.1101/2022.10.14.512179 (same Methods as the paper). File names below are in that repo.

### aber_map [1608, 1608, 23]

| channel (0-based) | content | unit |
|---|---|---|
| 0..20 | 21 Zernike coefficients in their own (n, m) order, see below | waves (nm / lambda, lambda = 680 nm), unit-rms (Noll) normalisation |
| 21, 22 | `psfrescale`: sigma x / y of a Gaussian image blur (5 x 5 kernel, fixed, not fitted in the current code) | px; constant over the field |

- Written by `Field Dependent PSF Modeling/fitFDaberration/calibrate_abr_map.m` (l. 223-241):
  `aber_map(:,:,i) = maps_gauss{i}/lambda` for i = 1..21, `aber_map(:,:,22) = aber_map(:,:,23) = psfrescale`.
- Mode order (`zernikefitBeadstack.m` l. 9, also `Example_SMLM.m` and the Python `zernikeModes`):
  (2,-2) (2,2) (3,-1) (3,1) (4,0) (3,-3) (3,3) (4,-2) (4,2) (5,-1) (5,1) (6,0) (4,-4) (4,4) (5,-3) (5,3) (6,-2) (6,2)
  (7,1) (7,-1) (8,0). Note (7,1) before (7,-1). As Noll indices (m > 0 cosine, m < 0 sine, as in torch-psf):
  **5, 6, 7, 8, 11, 9, 10, 13, 12, 17, 16, 22, 15, 14, 19, 18, 23, 24, 30, 29, 37**. No piston, tip / tilt or defocus;
  no pentafoil (Noll 20, 21), but 2nd / 3rd spherical (22, 37), 3rd astigmatism (23, 24) and 3rd coma (29, 30).
  **Noll 5-21 (the torch-psf default) is not enough**: it lacks 22, 23, 24, 29, 30 and 37 (see section 4 for what
  they are worth).
- Normalisation (`MLE_FitAbberation_Final_FixPara_Regu_sigma2.m` l. 80, kernel l. 49):
  `normfac = sqrt(2 (n + 1) / (1 + (m == 0)))` times Stallinga's un-normalised `get_zernikefunctions` = the Noll
  (rms 1) normalisation, the same as `torch_psf.zernike`. So their coefficient / lambda = our waves rms, up to the
  axis convention below.
- Checked on the data: `aber_map[y - 1, x - 1, :21] == value_P / 680` exactly at the bead positions `pos = (x, y)`
  (1-based, x = column) of the same stack (`*_maps.mat`, `beads_zrnikecoeff`): the map interpolates the per-bead fits
  (natural-neighbour `scatteredInterpolant`, then `imgaussfilt` with sigma `mapsmooth`; the paper: 200 px for the
  tetrapod) after an outlier filter (`zernikecoeffMap.m`). Map index order is [row, col] = [y, x].
- Channels 21 / 22: 0.5 / 0.5 px in `example_aber_map.mat`; 0.045 / 0.037 in the stack-20 map (= the fitted
  `Imgsigmax / Imgsigmay` of that older calibration run), 1.62 / 0.01 (stack 30), 0.015 / 0.008 (stack 61).
- No lateral-shift, photon or background channels.

### Their pupil / axis conventions vs torch-psf

- Pupil phase `exp(+2 pi i W / lambda)` (kernel l. 54), like ours (`exp(+2 pi i W)`, W in waves).
- Position phase (kernel l. 84-86): `exp(i (-x kx - y ky + zemit kz_med + objStage kz_imm))`. Ours
  (`vectorial.py`): `exp(i (-kx x - ky y - kz_imm f + kz_med z_e))`. So **objStage = -(our focus f)**.
- Pupil grid: `[YPupil, XPupil] = meshgrid(XYPupil, XYPupil)` (l. 28): their pupil x runs along the first array
  index, i.e. the image **row**; `phi = atan2(YPupil, XPupil)`. Bead crops are `imstack(pos(2)+r, pos(1)+r, :)`
  (`images2beads_globalfit.m` l. 132), i.e. images are not transposed; the Python simulator swaps x / y explicitly
  (`_PSF_vector_gpu.py` l. 70-73). torch-psf: px = column, py = row, phi = atan2(row, col). Hence
  **phi_FD = pi / 2 - phi_ours** (a transpose), which maps their coefficients onto ours as a signed permutation
  (`common.their_to_ours_matrix`):
  (2,-2) -> +Z5, (2,2) -> -Z6, (3,-1) -> +Z8, (3,1) -> +Z7, (4,0) -> +Z11, (3,-3) -> -Z10, (3,3) -> -Z9,
  (4,-2) -> +Z13, (4,2) -> -Z12, (5,-1) -> +Z16, (5,1) -> +Z17, (6,0) -> +Z22, (4,-4) -> -Z15, (4,4) -> +Z14,
  (5,-3) -> -Z18, (5,3) -> -Z19, (6,-2) -> +Z23, (6,2) -> -Z24, (7,1) -> +Z29, (7,-1) -> +Z30, (8,0) -> +Z37.
  Section 4 confirms this mapping on the beads; the alternatives (no transpose, or a flipped focus axis) are clearly
  worse.
- With n_med = n_imm and a real pupil amplitude, flipping the focus axis is equivalent to c_j -> -(-1)^m c_j
  (conjugating the pupil and mirroring k): bead data alone cannot tell the scan direction, the metadata must. Here
  both agree (next section).

### Optics and camera

| | value | source |
|---|---|---|
| objective | Olympus UPLSAPO 100XS, NA 1.35, silicone immersion | paper Methods |
| n_imm = n_med | 1.406 (silicone oil; imaging buffer RI 1.406) | `*_3dcal.mat` zernikefit, notebook, paper |
| n_cov | 1.525 (calibration) / 1.524 (training); no effect in their model or ours when n_med = n_imm | same |
| wavelength | 680 nm in the bead calibration (`zernikefit.lambda`), 670 nm in the training notebook; filter ET680/40 | `*_3dcal.mat`, `demo1_train.ipynb` |
| pixel | 108 nm in the bead calibration (`pixelSizeX/Y`), **110 nm** in training and in the localisation csv | `*_3dcal.mat`; notebook; our detections match their csv at 110 nm (82 % within 3 px) but not at 108 nm (1 %) |
| camera | Photometrics Prime 95B 25 mm (chip GS144BSI, sCMOS, 1608 x 1608, 11 um px), "1-HDR" gain mode, 16 bit, 100 MHz | MM metadata |
| offset | 100 ADU (`Camera-1-Offset`); photon transfer on the movie: 100.8 | metadata; `gain.py` |
| gain | **0.818 e- / ADU** (photon transfer on the movie, Poisson-equivalent; `gain.py`); FD-DeepLoc training used e_per_adu 0.5, QE 0.95, read noise 1.6 e- | `gain_movie.json`; notebook |
| sCMOS maps | none (constants) | paper / notebook |

### z

- Bead stacks: 61 planes, 100 nm steps (`z-step_um` 0.1; `ZPositionUm` 43.98 -> 49.98 um, increasing with the
  frame), 100 ms exposure. Their fit assigns `objStageStack = zmax:-dz:-zmax` (frame 1 = +2.9 um, 59 planes after
  trimming one plane per end), i.e. our focus f = -objStage **increases with the frame**: `stage_focus(61, 0.1,
  sign=+1)`. This agrees with the piezo moving the objective towards the sample as ZPositionUm grows.
- Movie / csv: `znano` = emitter z relative to the nominal focal plane, positive = further from the coverslip
  (training: `z = U(-1, 1) * 3000` nm, `zemit0 = -refmed / refimm * objStage0 = +3000`, `initial_obj_stage = -3000`).
  With n_med = n_imm only z_e - f matters; we use focus_um = 3.0 and z_range = (0, 6) um, i.e. our height z =
  znano + 3 um. Checked: the template z of our matched-filter detections minus 3 um vs their znano, median -0.02 um
  (p90 |dz| 0.24 um) on 40 frames.
- csv: `xnano = (0.5 + col + dx) * 110` (pixel corner origin); `intensity` in their photons (e_per_adu 0.5 / QE 0.95).
- Movie: 800 frames, 1608 x 1608, median 168 ADU (background ~56 photons / px with our gain), ~304 FD-DeepLoc
  localisations per frame, intensities 1.6k - 8k (their photons), znano -2.95 .. +2.98 um (5-95 %: -1.46 .. +1.20 um).

### How their map was made

`calibrate_abr_map.m`: beads segmented on the maximum projection (neighbours closer than 50 px rejected), an averaged
PSF fitted first (`zernikefitBeadstack`) for start values, then every bead stack fitted on its own
(`zernikefitBeadstack_all`: 21 Zernikes + x, y, z shared over the planes, photons / background per plane, Poisson MLE,
Levenberg-Marquardt, 75 iterations, own CUDA mex; built on Stallinga's vectorial code), outliers removed (a
coefficient more than 2.5 local std from the local mean, or RRSE > 0.2), natural-neighbour interpolation, Gaussian
smoothing (sigma 200 px), division by lambda. The calibration outputs shipped with stacks 20 / 30 / 61 are single-stack
maps (19 / 17 / 22 beads); `example_aber_map.mat` matches none of them (rms 0.018 - 0.029 waves at their beads), so it
was made from another (probably larger, multi-stack) bead set that is not identified in the release.

## 2. Our bead calibration (method)

`beads_fit.py`, per bead stack (stacks 20, 30, 61: the ones that ship their per-bead fits, so the same beads can be
compared): the stack in photons ((ADU - 100) x 0.818), 61 x 61 px ROIs (their ROIxy) around their bead positions, all
61 planes, `stage_focus(61, 0.1, sign=+1)`, `VectorialPSF` with NA 1.35, n_imm = n_med = 1.406, lambda 0.68 um,
pixel 0.108 um (their calibration values), npupil 64, oversample 3, their 21 modes (Noll THEIR_NOLL), and
`torch_psf.estimation.fit_beads` (Poisson, Adam) with per-plane photons / background, a cubic lateral drift and the
bead height free (= the axial offset of the bead from the stack centre; with n_med = n_imm height and focus offset are
the same parameter). No bead size, no image blur, no NA / apodisation fit.

Protocol (found by trial; results in section 4): a joint fit of all beads of the stack from zero aberrations (shared
Zernikes; independent of their numbers), then every bead alone from the joint fit (600 + 300 Adam steps), then again
from the median of all beads and from its 2 nearest beads, keeping the lowest NLL. The first version (400 + 200
steps, joint start only) left 4-5 of 19 beads of stack 20 0.025-0.12 waves rms from their values; refits of those
beads started from their values (`beads_check.py`) reached a lower NLL by 0.04-0.4 per px, and starting from the bead
median with more steps reaches the same point: slow convergence in a flat direction plus a biased start (the joint fit
has one height for all beads and chi2 2.9), not a different optimum.

## 3. In situ per tile (method)

Detection (`detect_tetrapod.py`): torch_psf.detect (difference of Gaussians, peak) is unusable for this PSF: the
peak sits on a lobe, 5 px (in focus) to 20 px (at +-3 um) from the emitter (`detect_check.py`), beyond any spline
offset limit, and a tetrapod gives two peaks. A matched filter instead: every frame (photons, median subtracted) is
correlated (FFT) with zero-mean unit-norm templates of the stack-20 bead PSF at 31 heights (0.2 um steps over +-3 um);
each response is divided by its robust noise (1.4826 MAD over the frame), the maximum over heights is taken, local
maxima in a 41 x 41 window above 7 are detections. On simulated spots (4000 photons, 56 photons / px background) the
centre error is < 1 px (p95 0.6-1.5 px), the template z within 0.1 um, every spot found at all heights, one detection
per spot. On the movie: 162k detections (203 per frame; FD-DeepLoc reports 304 / frame, it also resolves overlapping
spots), 82 % of ours within 3 px of one of theirs, template z - their z median -0.02 um. Isolated = no other detection
within 41 px (Chebyshev) in the frame: 108k ROIs of 41 x 41 px.

In-situ fit (`insitu_tiles.py`): the steps of `InSituEstimator` called through the library functions it uses (its
built-in detection finds lobes): `fit_zernike_knots` (knot route; spline 41 px, oversample 2, dz 0.04 um, xy_limit
3 px, z 0..6 um at focus 3 um), overlap rejection with `spot_deviance` (median + 3 robust scales), a refit, and the
held-out NLL after a full refit of every held-out spot (per-spot differences give the error bars). The first
localisation's multi-start uses 9 z starts instead of 5 (a `functools.partial` around `torch_psf.fit.localize` in the
script; with 5 starts over 6 um, 1.8 % of simulated spots end > 200 nm off in z, with 9 starts 0.2 %). Camera:
gain 0.818 e- / ADU, offset 100, read noise 1.6 e- (their value). Optics as for the beads but pixel 0.110 um (their
localisation pixel; see the bead pixel test). Split: a checkerboard of 32 px squares in absolute coordinates, the same
for the global fit and every tile (no held-out spot of a tile is used by the global fit). Global fit: 12000 training
spots from the whole field, 80 + 60 iterations, started from the median of our stack-20 bead fits; tiles: started from the global fit;
4 x 4: up to 4000 training / 3000 held-out spots per tile, 40 + 30 iterations; 6 x 6 (run in parallel on another GPU):
up to 3000 / 2000 spots, 30 + 22 iterations. Candidates on each tile's
held-out spots: global (in situ, whole field), tile, aber_map (their map averaged over the tile, converted to our
convention, no blur), beads (the median bead PSF).

## 4. Results: bead fits vs FD-DeepLoc

57 beads (stacks 20 / 30 / 61; their bead lists). Outputs: `~/fieldpsf_out/json/beads_s{20,30,61}_theirs.json`,
`beads_compare.json`, `beads_check_s*.json`, figure `~/fieldpsf_out/figs/beads_vs_theirs.png`.

**Convention test** (rms over all beads and modes of ours - theirs converted):

| mapping | rms (waves) |
|---|---|
| transpose (their pupil x = image row), focus up with the frame | **0.034** |
| no transpose | 0.058 |
| transpose + flipped focus axis | 0.142 |
| no transpose + flipped focus axis | 0.148 |

The transposed mapping wins in every per-bead comparison (median per-bead rms 0.0037 waves); the others differ by
signs / swaps in the m = 1, 2, 3 modes. The focus-axis test is exact: the fit with the stage direction reversed gives
the same NLL to 1e-5 per px and coefficients c -> -(-1)^m c to < 1e-4 waves (`beads_s20_signneg.json`), and those
match their values only after the same flip (0.0034 vs 0.138). So the bead data cannot decide the scan direction;
the metadata (ZPositionUm increasing) and their `objStage = -f` convention do, and they agree with each other.

**Per bead, per mode** (ours vs their per-bead fit of the same bead; waves rms; `beads_compare.json`):
median per-bead rms 0.0037 waves (= 2.5 nm), p90 0.043; 46 of 57 beads within 0.01 waves rms over the 21 modes. The
per-mode median |diff| is <= 0.008 waves for every mode (Z7 / Z8 coma 0.007 - 0.008, Z9 / Z10 trefoil 0.006,
everything else <= 0.004). The rms / r columns of the per-mode table in `beads_compare.json` are dominated by the 11
outlier beads (e.g. Z5 rms 0.066, r 0.07; Z13 rms 0.044) and say little.

The 11 outliers (0.011 - 0.123 waves; s20: 0, 3, 4; s30: 1, 15; s61: 0, 1, 3, 4, 18, 20): `beads_check.py` refits
each with our model at their coefficients (fixed) and started from them. In 7 of 11 our solution has the lower NLL
in our model, in 4 theirs; refits started from theirs end 0.011 - 0.12 waves from ours with NLL differences of
0.001 - 0.05 per px. These are beads with a flat likelihood in some directions (dim: 16-26k photons, border or
crowded ROIs), plus slow convergence of the Adam bead fit (refitting only the nuisance parameters at fixed Zernikes
moves the NLL by up to 0.14 per px). Their own fit flags the same beads (highest RSE_avg in `_maps.mat`). Not a
convention problem.

**Ours vs `example_aber_map.mat` at the bead positions** (the map of the movie, built from another bead set): rms
0.027 waves over beads and modes (0.026 after removing per-mode offsets); per-mode median |diff| 0.002 - 0.013, r over
the 57 beads 0.66 - 0.90 for Z5, Z6, Z7, Z8, Z9, Z10, Z13, Z19 (the modes with real field variation), 0.2 - 0.6 for the
weak higher modes. Field means agree to 0.009 waves rms (table in section 5).

**Mode sets** (stack 20, 19 beads, same protocol; `beads_s20_noll5_21.json`): the torch-psf default Noll 5-21
(17 modes) vs their 21 modes: joint fit NLL -37.1624 vs -37.1981 per px (2 dNLL ~ 3e5 over the stack for 4 fewer
parameters), per bead median 2 dNLL +4958 (range -7.5k .. +59k; the negative end is optimiser noise). **Noll 5-21
is not sufficient**; the 6 extra modes (22, 23, 24, 29, 30, 37) are needed. Noll 5-37 (33 modes): joint fit
-37.2825 per px, a further 2 dNLL ~ 7e5 for 12 more parameters (per-bead run stopped for time). At bead SNR (20-55k
photons per plane) the extra modes also absorb unmodelled effects (bead size, blur, apodisation), so this says the
model can use them, not that the pupil has them; in situ (2-8k photons) 21 modes were used throughout.

**Pixel size.** 110 nm vs 108 nm on stack 20: the joint fit prefers 110 (NLL / px -37.1991 vs -37.1981), the per-bead
fits prefer 108 (median 2 dNLL +246 per bead for 110). Not decided by the beads; the bead fits use 108 (their
calibration), the in-situ fits 110 (their localisation).

Bead-fit chi2 per px is 1.4 - 2.9 (bright beads, 20-55k photons per plane): model error at that SNR (no bead size,
no blur, no apodisation / NA fit; their fit reports RRSE ~0.17 on the same beads).

## 5. Results: in situ per tile

Outputs: `~/fieldpsf_out/json/insitu_r41.json` (global + 4 x 4), `insitu_r41_g6.json` (6 x 6),
`insitu_rescore_r41.json` (control), `insitu_report_g{4,6}_r41.json`; figures in `~/fieldpsf_out/figs/`:
`zernike_maps_g{4,6}_g{4,6}_r41.png` (per mode: their map converted to our convention as background, our tile
values as squares, our bead fits as dots, one colour scale per mode), `tiles_vs_map_g*_r41.png` (scatter per mode),
`heldout_g*_r41.png` (held-out NLL maps), `beads_vs_theirs.png`.

**Global in-situ PSF** (12000 training spots over the field, 11 % rejected as overlaps, deviance median 1.05, i.e.
gain and noise model consistent; 6000 held-out spots): better than the median bead PSF by **9.9 +- 0.24 NLL per
held-out spot** (79 % of the spots better). Its Zernikes differ from the bead median by 0.019 waves rms (waves rms per mode):

| mode | in situ (global) | beads (median of 57) | aber_map field mean | in situ - beads |
|---|---|---|---|---|
| Z5 (2,-2) astig 45 | -0.081 | -0.068 | -0.066 | -0.013 |
| Z6 (2,+2) astig 0 | -0.025 | -0.006 | -0.018 | -0.019 |
| Z7 (3,-1) coma y | +0.066 | +0.033 | +0.054 | +0.032 |
| Z8 (3,+1) coma x | +0.034 | -0.006 | -0.030 | +0.040 |
| Z11 (4,+0) spherical | +0.005 | +0.029 | +0.033 | -0.023 |
| Z9 (3,-3) trefoil y | +0.001 | +0.010 | +0.022 | -0.009 |
| Z10 (3,+3) trefoil x | -0.007 | +0.025 | +0.035 | -0.032 |
| Z13 (4,-2) 2nd astig 45 | +0.316 | +0.307 | +0.309 | +0.009 |
| Z12 (4,+2) 2nd astig 0 | -0.004 | +0.006 | +0.000 | -0.010 |
| Z17 (5,-1) 2nd coma y | +0.017 | +0.009 | +0.013 | +0.008 |
| Z16 (5,+1) 2nd coma x | -0.003 | -0.004 | -0.007 | +0.001 |
| Z22 (6,+0) 2nd spherical | -0.018 | +0.008 | +0.007 | -0.025 |
| Z15 (4,-4) quadrafoil 45 | -0.025 | -0.014 | -0.013 | -0.011 |
| Z14 (4,+4) quadrafoil 0 | -0.005 | +0.014 | +0.018 | -0.019 |
| Z19 (5,-3) 2nd trefoil y | -0.009 | -0.001 | +0.002 | -0.007 |
| Z18 (5,+3) 2nd trefoil x | +0.007 | +0.001 | -0.002 | +0.006 |
| Z23 (6,-2) 3rd astig 45 | +0.035 | +0.007 | +0.014 | +0.027 |
| Z24 (6,+2) 3rd astig 0 | +0.003 | -0.003 | -0.004 | +0.006 |
| Z30 (7,+1) 3rd coma x | +0.018 | +0.011 | +0.010 | +0.007 |
| Z29 (7,-1) 3rd coma y | +0.016 | +0.003 | +0.006 | +0.012 |
| Z37 (8,+0) 3rd spherical | -0.000 | +0.006 | +0.004 | -0.006 |

The bead median and the map's field mean agree to 0.009 waves rms; in situ differs from both mainly in coma
(Z7 +0.03, Z8 +0.04), trefoil x (Z10 -0.03), the sphericals (Z11 -0.023, Z22 -0.025) and 3rd astigmatism
(Z23 +0.027): a sample-induced / day-to-day difference that the movie itself supports (the held-out gain above). The
spherical shift is what a sample index different from the assumed n_med = n_imm = 1.406 (or a depth-dependent
aberration) would give; the model has no room for it other than these Zernikes.

**Held-out NLL per tile vs the global in-situ PSF** (per held-out spot; negative = better; +- standard error over
the tile's held-out spots):

| | 4 x 4 (402 px tiles, 16 tiles) | 6 x 6 (268 px, 36 tiles) |
|---|---|---|
| tile fit | mean **-14.5**, range -2.5 .. -26.9; better than global in 16 / 16 tiles | mean **-16.0**; 36 / 36 tiles |
| aber_map tile mean (converted) | mean -5.5 (range +4.7 .. -19.2) | mean -7.6 |
| bead median PSF | mean +9.8 | mean +9.5 |
| tile fit better than aber_map | 16 / 16 | 36 / 36 |
| held-out spots | 46k | 52k |

**Control** (`insitu_rescore.py`, `insitu_rescore_r41.json`): the global PSF refitted exactly like a tile (4000
training spots, 40 + 30 iterations, started from the global fit) but with spots from the whole field, scored with
global, tile and aber_map on a fresh subsample of every tile's held-out spots (2000 per tile; 32k / 52k spots). The
control is as good as the global PSF (mean +0.05 / +0.04 per spot, range -0.48 .. +0.50): extra iterations or spots
buy nothing. Tile minus control: 4 x 4 mean **-14.6** (range -2.3 .. -26.8, 16 / 16 tiles, every tile > 17
standard errors), 6 x 6 mean **-16.1** (-1.9 .. -30.9, 36 / 36 tiles, > 15 standard errors); a median 81 / 86 % of
the held-out spots of a tile are better fitted by its own PSF. aber_map minus control: -5.5 / -7.6. The gain of the
tile fits is field dependence.

The gain is smallest in the centre (tiles (1,1), (2,1): -2.5, -3.4) and largest at the edges and corners
(-18 .. -27), where the aber_map also helps most (-12 .. -19). **Spots per tile** (800 frames only): 4 x 4 tiles had 1.9 - 5.5k isolated training spots (all tiles used, 5 - 17 %
rejected as overlaps) and 2.2 - 3k held-out spots; 6 x 6 tiles 0.56 - 2.3k training and 0.55 - 2k held-out spots, no
tile skipped. Enough for 21 Zernikes per tile (the sparsest 6 x 6 tiles, < 1000 training spots, in corners, still
gain -17 .. -27 per spot), but the per-tile coefficients of the weak modes are noisy at that count (~0.01 waves), which
limits the 6 x 6 comparison more than the 4 x 4 one. The isolated fraction is the limit: 108k isolated of 162k
detections (41 px isolation); FD-DeepLoc itself reports 304 localisations per frame.

**Tile Zernikes vs aber_map tile means** (correlation over tiles of the spatial pattern; mean offsets as above):

| mode | r (4 x 4) | r (6 x 6) | std over tiles, ours / map (6 x 6) |
|---|---|---|---|
| Z5 (2,-2) astig 45 | +0.45 | +0.51 | 0.025 / 0.042 |
| Z6 (2,+2) astig 0 | +0.62 | +0.48 | 0.039 / 0.030 |
| Z7 (3,-1) coma y | +0.79 | +0.54 | 0.027 / 0.055 |
| Z8 (3,+1) coma x | +0.57 | +0.44 | 0.022 / 0.058 |
| Z11 (4,+0) spherical | +0.62 | +0.52 | 0.014 / 0.033 |
| Z9 (3,-3) trefoil y | +0.28 | +0.20 | 0.030 / 0.028 |
| Z10 (3,+3) trefoil x | -0.13 | +0.02 | 0.032 / 0.042 |
| Z13 (4,-2) 2nd astig 45 | -0.04 | +0.13 | 0.011 / 0.029 |
| Z12 (4,+2) 2nd astig 0 | -0.31 | -0.36 | 0.009 / 0.017 |
| Z17 (5,-1) 2nd coma y | +0.77 | +0.67 | 0.019 / 0.022 |
| Z16 (5,+1) 2nd coma x | +0.77 | +0.75 | 0.015 / 0.030 |
| Z22 (6,+0) 2nd spherical | +0.29 | +0.31 | 0.014 / 0.022 |
| Z15 (4,-4) quadrafoil 45 | +0.26 | +0.29 | 0.010 / 0.015 |
| Z14 (4,+4) quadrafoil 0 | -0.56 | -0.44 | 0.018 / 0.008 |
| Z19 (5,-3) 2nd trefoil y | +0.47 | +0.65 | 0.011 / 0.011 |
| Z18 (5,+3) 2nd trefoil x | +0.60 | +0.69 | 0.012 / 0.007 |
| Z23 (6,-2) 3rd astig 45 | -0.15 | -0.11 | 0.012 / 0.014 |
| Z24 (6,+2) 3rd astig 0 | +0.05 | -0.03 | 0.006 / 0.006 |
| Z30 (7,+1) 3rd coma x | +0.76 | +0.72 | 0.014 / 0.015 |
| Z29 (7,-1) 3rd coma y | +0.74 | +0.54 | 0.015 / 0.012 |
| Z37 (8,+0) 3rd spherical | -0.15 | +0.01 | 0.007 / 0.007 |

The spatial patterns agree for the modes with the strongest field dependence: astigmatism (Z5, Z6), coma (Z7, Z8),
primary spherical (Z11), 2nd coma (Z16, Z17), 2nd trefoil (Z18, Z19) and 3rd coma (Z29, Z30), r 0.44 - 0.75 on the 6 x 6
grid. They do not agree for trefoil x (Z10: ours has a strong top-bottom gradient, -0.03 .. +0.05, that the map only
partly shows), quadrafoil 0 (Z14, r -0.44), 2nd astigmatism 0 (Z12) and the 3rd-order astigmatism / spherical
(Z23, Z24, Z37): weak modes, and partly degenerate with the emitter z and each other in situ. Tile vs our bead fits
inside the tile (4 x 4, every tile has beads): similar picture (Z6 r 0.88, Z7 0.68, Z17 0.68, Z29 0.68).

The map's corner values (e.g. Z5, Z8, Z13, Z17 in the bottom-left / top-left corners, up to 0.2 - 0.4 waves from the
field mean) come from extrapolating the natural-neighbour interpolant outside the bead hull; the in-situ tiles do not
show them (figure `zernike_maps_g6_g6_r41.png`). The corner tiles are nevertheless where the map helps most in
held-out NLL, so the corner aberrations are real but smaller than the map says.

## 6. Polynomial order and which modes vary (input for the library stage)

`field_poly.py` (`~/fieldpsf_out/json/field_poly.json`):
- their example map (subsampled): field std 0.006 - 0.061 waves per mode; a quadratic leaves 0.004 - 0.041, order 3
  0.002 - 0.021, order 4 <= 0.007. The high orders mostly model the extrapolated corners.
- our in-situ 6 x 6 tiles (leave-one-out over tiles): a **linear** field term takes Z6 from 0.039 to 0.012, trefoil
  Z9 / Z10 from 0.030 / 0.032 to 0.017, 2nd coma Z16 / Z17 from 0.015 / 0.019 to 0.010 / 0.012, 2nd trefoil Z19 from
  0.011 to 0.009; a **quadratic** adds spherical Z11 (0.015 -> 0.008), Z5 (0.023 -> 0.020) and Z23 (0.010 -> 0.008).
  The other modes are flat within the tile noise (~0.01 waves).
- our bead fits: too few well-determined beads per mode (outliers) for a clean order selection; linear helps
  Z6 - Z10 here too.

## 7. What worked, what did not, open questions

Worked:
- The conventions: with their 21 modes, Noll normalisation, waves = nm / 680, the transpose (their pupil x = image row)
  and focus = -objStage, our bead fits reproduce their per-bead coefficients to a median 0.0037 waves (2.5 nm) rms.
- The bead fit with the existing `fit_beads` (vectorial, Poisson) on a 6 um tetrapod, 61 planes, 61 px ROIs: about
  1.5 s per bead and 900 Adam steps on an idle 3090.
- In situ with the existing knot route (`fit_zernike_knots`, `spot_deviance`, held-out NLL) on 41 px tetrapod ROIs
  over a 6 um z range: the global in-situ PSF beats the bead PSF by ~10 NLL per spot. Per-tile PSFs beat the global
  one in every tile of both grids, by 2.5 - 27 per spot, and the control shows the gain is spatial. Field dependence
  matters for this data set, and the in-situ tiles capture more of it (on held-out spots) than FD-DeepLoc's own map.

Did not work / limits:
- `torch_psf.detect` (DoG peaks) and therefore `InSituEstimator.estimate` as a whole: the peak of a tetrapod spot is
  a lobe, 5 - 20 px from the emitter. Replaced by a study-side matched filter; the in-situ steps were then called
  directly.
- Multi-start z with 5 starts over 6 um: ~2-5 % of spots in a wrong z minimum on the first localisation; 9 starts
  needed (patched in the script).
- Adam bead fits converge slowly. The nuisance parameters (61 per-plane photons and backgrounds, drift, height) and
  flat Zernike directions leave a few beads per stack off the optimum. Restarts from neighbouring beads and more
  steps fix most of them; 11 of 57 beads still differ from theirs by > 0.01 waves (both fits are uncertain there).
- ROI 41 px holds only 60 - 86 % of the tetrapod PSF energy (51 px: 73 - 92 %). Bigger ROIs mean fewer isolated
  spots (25 % of FD-DeepLoc's localisations are isolated at 41 px). The spline (knots c = 25 px half width, 161 z
  planes at 0.04 um) is small; memory was not a problem (< 4 GB per in-situ process).
- Runtime: ~3 - 5 min per 4 x 4 tile (4000 + 3000 spots, 40 + 30 iterations, 9 z starts), ~1.5 - 2 min per 6 x 6 tile, on GPUs shared with other jobs;
  the whole in-situ study ~3 h.

Open questions:
- Why in situ differs from the beads by ~0.02 - 0.04 waves in coma, trefoil x and the sphericals: sample index /
  depth (a fitted n_med or a depth-dependent spherical term would test it), or a DM / alignment change between the
  bead and the cell acquisitions.
- Trefoil x (Z10) has a strong top-bottom gradient in situ that neither the map nor the beads show to that extent.
  Real (sample tilt, coverslip), or a degeneracy with something the tile fit cannot separate?
- Which pixel size (108 / 110 nm) is right; a stage-micrometer calibration or a fit of the pixel size on beads
  with a better optimiser would settle it.
- The image blur channel of their map (0.5 px in the example map) is not modelled here; fitting `VectorialPSF.blur`
  in situ would tell whether it is needed.
- Their example map's bead set is not in the release; our bead comparison uses the three single-stack calibrations.

## 8. What the library stage needs (from this data)

- **Modes**: their 21 (Noll 5-19, 22, 23, 24, 29, 30, 37); Noll 5-21 lacks 22, 23, 24, 29, 30, 37,
  and the tetrapod's 3rd-order modes are not small (Z23 field mean 0.014, Z22 0.007, Z29 / Z30 0.006 - 0.010 waves,
  std over the field 0.006 - 0.016). Make the mode list a parameter everywhere (it is in VectorialPSF); default for
  engineered PSFs: Noll 5-37.
- **Field model**: per mode a polynomial in the field coordinates. Order 1 (3 coefficients) captures most of the
  measurable in-situ variation; order 2 (6) for spherical and astigmatism. Order 2 for every mode = 21 x 6 = 126
  parameters, well determined by 10^4 - 10^5 spots, with a ridge prior towards the global (order-0) fit for the
  weak modes. Their map's higher orders (3 - 4) mostly fit corner extrapolation; a polynomial should not be
  extrapolated beyond the spot hull either (clamp, or regularise the highest order).
- **Strongly varying modes** (field std / range in situ, waves): Z6 astig 0 (0.039, linear), Z10 trefoil x (0.032,
  linear), Z9 trefoil y (0.030), Z7 / Z8 coma (0.027 / 0.022), Z5 astig 45 (0.025), Z17 / Z16 2nd coma
  (0.019 / 0.015), Z14 quadrafoil (0.018), Z11 spherical (0.014, quadratic); the rest <= 0.015.
- **Per-emitter Zernike offsets** already exist in `VectorialPSF.forward(zcoef_offset=...)`; the knot route has one
  spline per PSF, so a field-dependent model needs either per-tile splines (this study), or a spline basis per
  polynomial term (linear in the coefficients for small offsets: K(x, y) ~ K0 + sum_t p_t(x, y) dK/dc_t), or the
  vectorial route (V) for the field terms.
- **Detection for engineered PSFs**: a matched filter over z templates (`detect_tetrapod.py`) belongs in the library
  (`detect`), so that `InSituEstimator` works on tetrapod / large-range PSFs; plus `zstarts` as an estimator option.
- **Bead fitting**: a second-order (LM / Gauss-Newton) step for the nuisance parameters per bead, and multi-start
  for engineered PSFs; per-bead fits as a supported mode (field-dependent calibration from beads = this study's
  `beads_fit.py`).
- **Conventions**: an importer for FD-DeepLoc aber_map / per-bead `*_maps.mat` (`common.their_to_ours_matrix`,
  waves = nm / lambda, map [row, col]) would let their calibrations be used as a start or a prior.
