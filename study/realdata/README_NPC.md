# NPC-A647-3D: 3D localizations (no drift correction)

Data: EPFL SMLM challenge 2016, real 3D data set "NPC-A647-3D" (Ries lab; Nup107-SNAP-AF647 in U2OS, dSTORM,
astigmatism by a cylindrical lens). Movie = `NPC-A647-3D-stack_1.tif` (frames 0-74999) followed by
`NPC-A647-3D-stack_2.tif` (frames 75000-149999); the MicroManager FrameIndex in the TIFF labels runs continuously
0-149999 over the two files, so they are consecutive halves of one acquisition.

Code: `C:\dev\torch-spline-3d\study\realdata\` (server copy `~/torch-spline-3d-insitu/study/realdata/`, snapshot in
`code/` next to this README), not committed. The full NPC run used `loc.py` before a later change that only batches
the spots for GPU memory and writes the CSV incrementally (same algorithm). `test/` holds scratch test runs. Spline library: `torch_convspline/knotpsf.py` (KnotPSF, Catmull-Rom knots) of the user's torch-spline-3d
repository (branch insitu-study).

## File

`NPC-A647-3D_locs.csv` (comma separated, one header line). 1,051,017 localizations, 7.0 per frame.

| column | unit | meaning |
|---|---|---|
| frame | - | 0-based frame index over the whole 150,000-frame movie (stack_1 then stack_2) |
| x, y | camera pixels | full-frame coordinates of the 135 x 148 px movie. **Pixel-centre convention**: x = 0.0 is the centre of the first column, y = 0.0 the centre of the first row (x = column index, y = row index). Add 0.5 for the corner convention (origin at the outer corner of the top-left pixel). x is the fast (column) axis of the TIFF, y the row axis. Not drift corrected. |
| z | nm (stage nm) | height from the bead calibration, see "z" below. Not drift corrected. Fit range -800 ... +800 |
| photons | photoelectrons | emitter signal I; the PSF model is normalised so that it sums to 1 over a 23 x 23 px window at every z, so this is the total signal within +-11.5 px (the 17 x 17 px fit patch holds 99.3 % of it at z = 0). Quantum efficiency not applied (electrons, not photons). |
| bg | photoelectrons / pixel | background per pixel per frame (constant over the 17 x 17 patch) |
| crlb_x, crlb_y | pixels | Cramer-Rao lower bound (std) of x, y, **including the EMCCD excess-noise factor** (Poisson CRLB x sqrt(2)) |
| crlb_z | nm | as above for z |
| crlb_photons | photoelectrons | as above for photons |
| chi2 | - | goodness of fit: mean over the fitted pixels of (d - mu)^2 / (2 mu), d and mu in photoelectrons; the factor 2 is the EMCCD excess noise; expected 1 for a correct model |
| nll_ratio | - | Poisson log-likelihood ratio against the saturated model per pixel, computed on data and model halved (EMCCD); about 0.5 expected |
| flags | bitmask | 1: x or y offset at the fit limit (3 px from the detection peak); 2: z at the fit limit (+-800 nm); 4: patch extends beyond the frame (missing pixels are left out of the likelihood, fit is valid but less precise); 8: another detection inside the 17 x 17 patch (possible overlap bias); 16: photons or bg at their lower clamp. flags = 0 for 86.8 % |
| n_neighbors | - | number of other detections inside the 17 x 17 patch in the same frame |
| peak_x, peak_y | pixels | integer position of the detection peak (patch centre) |

Recommended filter for a clean set: `flags == 0` (912,192 localizations); optionally also `chi2 < 1.5`.

## Camera (Andor iXon Ultra 897, DU897_BV, serial 8172 per the MicroManager metadata)

Metadata of the movie: EM gain 200, pre-amp "Gain 1", 10 MHz readout, 15 ms exposure, frame interval ~15.9 ms
(2388 s for 150,000 frames), ROI "191-239-135-148" (x0 = 191, y0 = 239, width 135, height 148 on the 512 x 512
chip), acquired 2016-12-08. `Andor-CCDTemperature` reads 10 (deg C) in every frame although the set point is -75;
I do not know whether that value is real (dark current would then add to the background, which the fit absorbs).

Conversion used: **g = 10.9 ADU per photoelectron (EM gain included), offset = 295 ADU**,
photoelectrons = (ADU - 295) / 10.9. Obtained from the movie itself (no dark frames exist): per pixel and per block
of 400 frames (16 blocks spread over the movie), mean = temporal median, variance from differences of consecutive
frames (robust, clipped at 3.5 sigma), then var = 2 g (mean - offset) + read^2 fitted over all pixels and blocks
(`camera_noise.json`, `cam_calib.py`):
- line through binned (mean, var): slope 21.77 (median as mean) / 21.81 (trimmed mean) -> g = 10.88 / 10.91;
  offset = -intercept/slope = 292 / 299 ADU (read noise neglected; it would lower the offset by read^2/slope);
  residuals of the line are within +-1 % but curved;
- per-pixel temporal regression over the blocks after frame 4000: slope 21.56-21.62 (g = 10.78-10.81), the intercept
  of this fit is not usable (the temporal mean range is only ~60 ADU);
- including the first blocks the per-pixel temporal slope is 24.4: the first ~2000 frames have many more emitters
  on and their blinking inflates the variance, so those blocks were not trusted.
The offset is the least certain number (+-30 ADU is my honest range); an offset error shifts bg by about +-3
photoelectrons per pixel and leaves photons nearly unchanged.

EMCCD excess noise: the fit is a Poisson MLE on the photoelectron data. Because the score equations are unchanged
when data and model are both halved, this is the same estimate as the "half-counts" Poisson likelihood that has the
correct EMCCD variance 2 mu; the Fisher information is halved, so every CRLB is the Poisson CRLB times sqrt(2), and chi2
uses variance 2 mu. The repeat-precision check below confirms this scaling (ratio 1.00-1.02).

## Pixel size: not stated in any source I could find

Not in the metadata (PixelSizeUm = 0), not in Sage et al. 2019 or its supplement, not in Li et al. 2018, not on
the challenge web pages (the "20 nm/pix" there is the render pixel size). **Empirical value**: I compared these
localizations with the Cspline submission (Ries lab) on the challenge web site,
`https://srm.epfl.ch/srm/Data/challenge-3D-real/NPC-A647-3D-Results/Cspline/Cspline____loca___NPC_driftCorr.csv`
(copy in `../reference_participants/`): per-frame nearest-neighbour matching (27,161 of their 236,332
localizations matched within 100 nm), least squares with a free offset per 2000-frame window (their file is drift
corrected): x_Cspline = **134.9 nm/px** * x, y_Cspline = **142.0 nm/px** * y, rotation 0.008 deg, robust residual 14.8 nm
(with one isotropic scale the residual is 30 nm, so the anisotropy is real in their numbers). This is the scale that
submission used, not an independent calibration; a cylindrical lens can give slightly different magnification in x
and y, but I cannot confirm it. Internal check with the Nup107 ring radius (about 53 nm): circle fits to 2,967
ring-like clusters give a median radius of 0.435 px (IQR 0.38-0.53), the pooled radial density peaks at 0.34 px; with
~138 nm/px that is 60 nm and 47 nm, which brackets 53 nm (both estimates are crude). Treat the pixel size as
**unknown, ~135-142 nm/px if one trusts the Cspline submission**.

## Bead calibration (`calib/`)

Stack `NPC-A647-3D-BEADS-as-stacks.tif`: 41 planes, MicroManager SlicePosition -1.0 ... +1.0 um in 50 nm steps (PI
piezo "PIZStage" as focus device), same Andor camera, full chip, EM gain 4, pre-amp "Gain 2", 1 MHz, 40 ms, acquired
2016-12-01 (one week before the movie). Beads on the coverslip in water (challenge page).

- Bead finding (`beads_find.py`, `beads_found.json`): DoG maxima of the max projection over z, isolation (no other
  candidate brighter than 5 % within 20 px), border margin; 82 isolated candidates. Per plane an elliptical Gaussian
  fit; good = width curves sx(z), sy(z) within 0.35 px rms of the median curve over +-0.6 um after aligning each
  bead on its crossing sx = sy, brightness 0.3-6 x the median, crossing within +-0.5 um: 67 beads.
- Used: the 32 good beads within 100 px of the movie ROI (only 2 lie inside the ROI); after a first fit 2 beads with
  chi2/pixel > 1.5 were dropped (248,247: 2.6; 359,335: 1.7) -> **30 beads**, 1,080 bead frames. The 35 good beads
  further away were held out to check the field dependence.
- Bead model: frame k of bead b: mu = I_bk g(o - x_b, o - y_b, z_k - dz_b) + bg_bk; (x_b, y_b) shared by all planes
  of a bead, so the spline carries the z-dependent lateral shift ("wobble") of the PSF; dz_b = focus offset of the
  bead, initialised at its Gaussian width crossing and refined by a 1D likelihood search (mean refinement held at 0);
  I and bg free per frame. Only frames with |z| <= 0.9 um enter.
- Noise model of the bead stack (EM gain 4: excess factor and read noise unknown): from the residuals of a first
  pass, var = -140 + 0.708 mu (ADU); the fit then uses y = (ADU + a/b) / b, whose variance equals its mean.
- Spline: free Catmull-Rom knots every 0.5 px (S = 2) and every 50 nm in z, calibrated range z = -800 ... +800 nm
  (knots -1000 ... +1000), 23 x 23 px bead patches; Poisson likelihood, alternating bead parameters (LM per frame)
  and knots (L-BFGS), 14 rounds; roughness penalty lambda = 3000 on the second differences in x, y, z (weighted by
  1/|knot|), chosen by leave-one-bead-out on 8 beads (held-out chi2/pixel: lambda 0: 1.178, 300: 1.067, 1000: 1.054,
  3000: 1.053, 1e5: 1.45). A constant added to a knot plane is exactly degenerate with the background; it is fixed by
  setting the mean of the knots on the border ring of the 23 x 23 patch to zero. Each z plane is normalised to sum 1
  over the 23 x 23 patch (x = y = 0).
- z origin: z = 0 is the stage position at which a bead's Gaussian widths are equal (sx = sy), averaged over beads.
- Movie spline: same knots cropped to the 17 x 17 px fit patch with a lateral offset range of +-3 px (`loc.py`).

Calibration checks:
- in-sample chi2/pixel per bead: median 1.03 (0.89-1.47); per 100 nm of z: 0.99-1.13 over +-800 nm (1.21 at -900).
- leave-one-bead-out (all 30 beads, each held out and fitted with a spline from the other 29): held-out chi2/pixel
  median 1.073 (0.91-1.52) vs 1.031 in-sample; held-out frames localized with all parameters free: z error rms
  median 10.8 nm (5.7-17.4 nm per bead), max |z error| median 23 nm (13-42), x / y scatter rms 0.042 / 0.062 px
  (bead photon noise included, bead frames are bright).
- field dependence: the 35 held-out beads further from the ROI give chi2/pixel median 1.12 (1.03-1.43), rising
  slowly with distance from the ROI (~1.05-1.10 at 100-130 px, 1.15-1.23 beyond 200 px). The PSF varies a little
  over the chip; within the ROI it should be close to the calibration (only 2 beads lie inside the ROI).
- figures: `calib/calib_widths.png` (Gaussian widths of spline and bead frames, chi2 vs z, bead map),
  `calib/calib_montage.png` (data / model / residual of one bead).

## z

z is in **stage nm** of the bead stack, relative to the bead focus defined above, **no refractive-index rescaling**.
The beads are on the coverslip and the stage moves the focus; the emitters are in aqueous medium above the coverslip,
so the true height difference is the stage difference times a focal-shift factor (commonly ~0.7-0.8 for an oil
objective imaging into water; Li et al. 2018 multiply by 0.75). The bead PSF is also not identical to the PSF of an
emitter inside the sample (spherical aberration from the index mismatch grows with depth); this is not modelled.
The sign of z relative to "up" is not known from the metadata. The height map below suggests that z decreases where
the nuclear envelope curves away from the coverslip at the rim of the nucleus, i.e. more negative z = deeper into
the sample, but this is an inference. The Cspline submission's z relates to ours as z_Cspline = 0.907 z + const
(robust residual 31 nm); I do not know what scale they applied.

## Localization

`loc.py`: photoelectrons; difference of Gaussians (sigma 1.5 and 5 px, reflect padding), local maxima in 7 x 7 px,
threshold 6 x the robust (MAD) std of the DoG image per frame; 17 x 17 px patches centred on the peak (pixels outside
the frame are masked out); Poisson MLE of (x, y, z, photons, bg) on the spline by Levenberg-Marquardt on the GPU, 3
starting heights (-450, 0, +450 nm, 12 iterations each), the best continued for 20 iterations; x, y limited to +-3
px from the peak, z to +-800 nm. 150,000 frames in 240 s on one RTX 3090.

## Checks (all frames; `checks/checks.json`, PNGs in `checks/`)

- Fits at limits: x/y 1.9 %, z 0.93 %, patch over the edge 5.0 %, neighbour in patch 7.5 %, photons/bg clamp 0 %;
  any flag 13.2 %.
- photons: median 2,516 (5 % / 95 %: 726 / 5,111); bg: median 80.7 photoelectrons/px (68 / 93).
- CRLB (median, flags = 0): x 0.082 px, y 0.080 px, z 32 nm (5-95 %: x 0.044-0.37 px, z 19-97 nm).
- chi2/pixel (expect 1): median 1.008 (all), 0.998 (flags = 0; 5-95 %: 0.86-1.37); flat in z (0.99-1.10 per 100 nm)
  and in photons (0.99-1.03 up to the top decile).
- Repeat precision (same emitter in consecutive frames, mutual nearest neighbour within 1 px, flags = 0; 683,043
  pairs): robust std of the difference / sqrt(2): x 0.084 px vs median CRLB 0.078; y 0.082 vs 0.077; z 31.9 nm vs
  31.5 nm. Difference normalised by the pair's CRLBs: robust std 1.010 (x), 1.018 (y), 0.998 (z); plain std 1.10,
  1.13, 1.22 (heavy tails: wrong pairings, overlaps). Per photon bin the ratio is 0.99-1.10 except > 8,000 photons
  (x 1.46), where systematic PSF error starts to dominate.
- z distribution (flags = 0, photons > 1000): median -30 nm, IQR 257 nm, 0.6 % within 100 nm of the limits; not
  piled up at the limits. It is not one narrow layer: the median-z map (`checks/z_map_time.png`, 4 px bins) shows the
  underside of the nucleus tilted/curved from about +200 nm to -480 nm towards the rim of the nucleus (5-95 % of
  bins). Within single ring-like clusters the z IQR is 66 nm (median over clusters).
- z over time: median of (z - local height map) per 2000 frames drifts from about -15 nm to +19 nm over the movie
  (a rough z-drift estimate, `checks/z_map_time.json`).
- Renders: `checks/render_full.png` (whole field, 0.1 px bins, all frames, flags = 0, photons > 500) and
  `checks/render_zoom_all.png` (20 x 20 px at x 50-70, y 80-100): nuclear-pore rings are clearly visible even without
  drift correction, diameter about 1 px. `render_zoom_win1.png` uses only frames < 10,000.
- Starting heights (frames 0-2999, 52,036 fits): against a fit with 5 starts (-650 ... +650 nm) and 30 + 30
  iterations, 0.8 % of the flags = 0 fits move by more than 50 nm in z and 0.5 % by more than 200 nm (another
  likelihood branch); 0.6 % reach a lower likelihood with 5 starts. Repeating the same 3-start fit with a different
  GPU batching changes z by > 10 nm for 0.14 % (by 1-2 nm for 9 %: float32 and a fixed iteration count).

## Assumed / unknown / possible problems

- Pixel size unknown (see above).
- Offset 295 ADU estimated, +-30 ADU; dark frames absent. Camera temperature reading of 10 deg C unexplained.
- Gain and offset assumed uniform over the chip; EM excess factor assumed exactly 2 (high-gain limit).
- PSF from beads on the coverslip one week earlier; no correction for index mismatch or for aberration changes; only
  2 calibration beads inside the ROI (30 within 100 px). Mild field dependence seen in the held-out beads.
- z in stage nm, no refractive-index factor, sign of "up" not established.
- Not drift corrected (x, y, z). Repeated detections of one emitter in consecutive frames are not merged.
- Detections with a neighbour in the patch are fitted as single emitters (flag 8).
