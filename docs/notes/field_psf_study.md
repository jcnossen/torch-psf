# Field-dependent PSF study, stage 1: FD-DeepLoc data, conventions, measurements

Goal: show that torch-psf measures field-dependent aberrations on real data, using the public FD-DeepLoc data set
(Fu et al., Nat. Methods 2023, "Field dependent deep learning enables high-throughput whole-cell 3D super-resolution
imaging"; Zenodo 6547064). It has its own per-pixel aberration map, built from bead stacks. This stage uses the library as it is; scripts are in
`study/fieldpsf/`, outputs (json, figures, logs) in `~/fieldpsf_out/` (not in the repo).

<!-- RESULTS -->

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
  **Noll 5-21 (the torch-psf default) is not enough**: it lacks 22, 23, 24, 29, 30 and 37 (see section 2 for what
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
  Section 2 confirms this mapping on the beads; the alternatives (no transpose, or a flipped focus axis) are clearly
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
