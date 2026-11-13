# Tubulin-A647-3D: 3D localizations (no drift correction)

Data: EPFL SMLM challenge 2016, real 3D data set "Tubulin-A647-3D" (Ries lab; microtubules, Alexa 647, dSTORM,
astigmatism by a cylindrical lens). Movie = `Tubulin-A647-3D-stacks_1.tif` ... `_4.tif` concatenated in that order,
91,946 frames of 324 x 271 px. Their MicroManager IndexMap gives acquisition frames 22987-45973, 45974-68960,
68961-91946, 91947-114932: the four files are contiguous and **our frame f = MicroManager frame f + 22987**.
(The challenge page lists 112,683 frames; the Cspline submission's frame numbers equal ours + 20737, i.e. their
frame 0 is MicroManager frame 2250; the first 20,737 frames of the challenge movie are not in these four files.)

Code: `C:\dev\torch-spline-3d\study\realdata\` (server copy `~/torch-spline-3d-insitu/study/realdata/`, snapshot in
`code/`), not committed; same pipeline as NPC-A647-3D (see that README for the method in detail). `test/` holds
scratch test runs.

## File

`Tubulin-A647-3D_locs.csv`, 10,112,241 localizations (110 per frame). Columns as for NPC:

| column | unit | meaning |
|---|---|---|
| frame | - | 0-based over the concatenated four files (MicroManager frame - 22987; Cspline frame - 20737) |
| x, y | camera pixels | full-frame coordinates of the 324 x 271 px movie; **pixel-centre convention** (x = column index, y = row index; add 0.5 for the corner convention). Not drift corrected |
| z | nm, **nominal**: 10 nm per bead-stack plane assumed (see "z") | height from the bead calibration, fit range -800 ... +800 nominal nm. Not drift corrected |
| photons | photoelectrons | signal; PSF normalised to sum 1 over 23 x 23 px at every z (the 17 x 17 fit patch holds 99.2 % at z = 0); QE not applied |
| bg | photoelectrons / pixel | background |
| crlb_x, crlb_y | pixels | CRLB including the EMCCD excess factor (Poisson CRLB x sqrt(2)) |
| crlb_z | nominal nm | as above |
| crlb_photons | photoelectrons | as above |
| chi2 | - | mean of (d - mu)^2 / (2 mu) over the fitted pixels; 1 expected |
| nll_ratio | - | log-likelihood ratio per pixel against the saturated model (halved data); ~0.5 expected |
| flags | bitmask | 1 x/y at the 3 px limit; 2 z at +-800; 4 patch over the frame edge (masked); 8 other detection inside the 17 x 17 patch; 16 photons/bg at clamp. flags = 0 for 66.2 % |
| n_neighbors | - | other detections inside the patch |
| peak_x, peak_y | pixels | detection peak (patch centre) |

Recommended clean set: `flags == 0` (6.70 M). This data set is dense (110 detections per frame, 27.6 % have a
neighbour inside the patch), so the flags matter more than for NPC.

## Camera (Photometrics Evolve 512 Delta, serial A14G150010, per the MicroManager metadata)

Metadata: "Port" Multiplication Gain, MultiplierGain 100, Gain 1, readout 10 MHz 16 bit, Offset 400 (camera
setting), 15 ms exposure, frame transfer, ROI "103-230-324-271" (x0 = 103, y0 = 230, 324 x 271 on the 512 x 512
chip), acquired 2017-04-04. CCDTemperature reads -23.3 deg C with set point -70 (not at set point).

Conversion used: **g = 17.0 ADU per photoelectron (EM gain included), offset = 400 ADU**.
- g from the movie (`camera_noise.json`, same method as NPC): spatial line through binned (mean, var): slope 33.96 /
  34.05 -> g = 16.98 / 17.02; per-pixel temporal regressions over 16 blocks: slope 33.7 (pixels whose mean changes by
  > 50 ADU), 33.9 (> 100 ADU), 32.6 (> 200 ADU) -> g = 16.3-17.0. The spatial line is clearly curved (residuals -38 %
  at the darkest pixels, +9 % in the middle), so the numbers are less clean than for NPC.
- offset: the fits give 364-383 ADU, but they extrapolate far. I used 400: the camera's Offset setting is 400 and the
  background of the bead stack (same camera, no metadata on its settings) is 404 ADU with a variance of only ~20 ADU^2
  (read-noise level, i.e. almost no signal), which puts the bias at 400-404. With offset 380 instead, bg would be
  ~1.2 photoelectrons/pixel higher (median bg is 23), photons nearly unchanged.
- EMCCD excess factor 2 assumed, as for NPC.

## Pixel size: not stated in any source I found

Not in the metadata (PixelSize_um 0) nor in the papers / challenge pages. **Empirical**: matched against the Cspline
submission (`https://srm.epfl.ch/srm/Data/challenge-3D-real/Tubulin-A647-3D-Results/Cspline/Cspline____loca___Tubulin_driftCorr.csv`,
285 MB; I fetched two byte ranges, `../reference_participants/`): after finding the frame offset (+20737, from the
correlation of the per-frame localization counts, correlation 0.17 at that lag vs < 0.11 elsewhere; matches peak
sharply at that offset), 50,761 of their 1,226,735 localizations in their frames 62270-79249 match within 100 nm:
x_Cspline = **117.0 nm/px** * x, y_Cspline = **118.2 nm/px** * y, rotation 0.002 deg, robust residual 27 nm. This is the
scale that submission used, not an independent calibration. Different camera and probably a different microscope
than NPC (Andor there, 135 x 142 nm/px by the same method).

## Bead calibration (`calib/`)

Stack `Tubulin-A647-3D-BEADS-as-stacks.tif`: 401 planes of 512 x 512, **no metadata at all** (labels are only file
names 0000.tif ... 0400.tif): the z step and the camera settings of the stack are unknown.

- 8 bright beads in the whole field (DoG amplitude > 100), all in the lower part of the chip; 1 at the border; **7
  used, all inside the movie ROI** (movie coordinates x 74-256, y 121-248; the upper half of the movie field has no
  bead nearby). Bead focus (Gaussian width crossing) at planes ~167-174.
- **z step: assumed 10 nm per plane** (stack = -2.0 ... +2.0 um nominal). Evidence: (1) with 10 nm per plane the
  Gaussian width curves of this PSF (sy minimum ~1.1 px at -200 nm, sx minimum ~1.15 px at +150 nm, ~6 px at +-750 nm)
  are close to those of the NPC calibration whose 50 nm steps are in its metadata (same lab, similar optics); 5 nm
  per plane would make the PSF change twice as fast with z. (2) The Cspline submission's z relates to ours as
  z_Cspline = 0.958 z_ours + const (50,761 matched localizations, robust residual 40 nm); for NPC (known step) the same
  comparison gives 0.907. If the Cspline z convention is the same in both data sets, the step is 10 x 0.958 / 0.907 =
  10.6 nm per plane. Both point to ~10 nm, but it is an assumption: **if the true step is s nm, multiply z and crlb_z
  by s / 10.**
- Noise model of the stack (settings unknown): photon-transfer curve from second differences along z of the bead
  patches (adjacent planes differ mainly by noise at this fine step): var ~ 20-23 ADU^2 at the background (404 ADU)
  rising to ~94 at 786 ADU; linear fit var = -70.6 + 0.211 mu, fit with y = (ADU + a/b) / b. The EM gain of the stack
  was apparently low.
- Spline: as NPC (S = 2, 50 nominal-nm knots, calibrated range +-800 nominal nm, 23 x 23 bead patches, border-ring
  pedestal 0, per-plane normalisation), roughness lambda = 1e5 chosen by leave-one-bead-out over all 7 beads
  (held-out chi2/pixel mean: lambda 1e3: 6.97, 1e4: 5.23, 1e5: 4.47; held-out z rms error 3.8-8.6 / 3.4-9.1 /
  3.0-7.9 nm). No bead was dropped (chi2 cut 10: these beads are ~100x brighter than single molecules, so chi2 is far
  above 1 for small systematic differences).
- z origin: z = 0 where a bead's Gaussian widths are equal, averaged over the 7 beads.

Calibration checks:
- in-sample chi2/pixel per bead 2.05-4.40; per 100 nm of z 2.6-3.5 (4.7 at -900). The residual montage
  (`calib/calib_montage.png`) shows structured residuals of a few % of the peak for single beads: the beads differ
  from each other more than their (very low) shot noise.
- leave-one-bead-out (all 7): held-out chi2/pixel median 4.45 (2.95-6.11) vs 2.97 in-sample; held-out frames
  localized with all parameters free: z error rms median 5.4 nominal nm (3.0-8.0), max |z error| median 11 nm
  (6.7-18.4), x / y scatter rms 0.040 / 0.039 px. So the systematic PSF error is small compared with single-molecule
  precision, but only 7 beads, all from one region, were available; no held-out beads for a field check.
- `calib/calib_widths.png`, `calib/calib_montage.png`; `calib/bead_montage.png`, `calib/bead_profile.png`,
  `calib/bead_mip.png` show the raw stack.

## Localization

As NPC (`loc.py`): DoG (1.5 / 5 px, reflect padding), 7 x 7 local maxima, threshold 6 x robust DoG noise per frame,
17 x 17 px patches with edge masking, 3 starting heights, LM Poisson MLE on the GPU; 91,946 frames in 27 min.

## Checks (`checks/checks.json`, PNGs in `checks/`)

- Fits at limits: x/y 15.8 %, z 1.8 %, edge 2.1 %, neighbour in patch 27.6 %, photons/bg clamp ~0; any flag 33.8 %.
  The x/y-limit fraction is high: most of these are detections between or next to overlapping emitters (dense).
- photons: median 2,482 (5 % / 95 %: 725 / 6,942); bg median 22.9 photoelectrons/px (9.7 / 55.6).
- CRLB (median, flags = 0): x 0.081 px, y 0.081 px, z 22.6 nominal nm.
- chi2/pixel: median 1.58 (all), **1.19 (flags = 0)**; flags = 0 by z: 1.08-1.31 except +300 ... +500 nm, where it
  rises to 1.76-2.01; by photons 1.15-1.21 up to 4,650 photons, 1.73 above. The model fits clearly worse than for NPC.
- Repeat precision (consecutive frames, flags = 0, 3,482,289 pairs): robust std of the difference / sqrt(2): x 0.092
  px vs median CRLB 0.075 (ratio 1.21 normalised), y 0.097 vs 0.077 (1.25), z 27.2 vs 22.2 nominal nm (1.21); plain
  std ratios 1.83 / 1.84 / 2.91. The ratio grows with photons (x: 1.08 below 1,000 photons, 1.19 at 1-2k, 1.32 at
  2-4k, 1.57 at 4-8k, 3.0 above 8k): a systematic, signal-independent error of order 0.05 px and 15 nm per
  localization, from PSF mismatch (beads on glass vs. emitters up to ~1 um deep in the cell, field dependence over a
  324 x 271 px field calibrated by 7 beads in one region) and from wrongly paired neighbours in this dense data.
  **The CRLB columns underestimate the real precision of this data set by ~20 % or more for bright emitters.**
- z distribution (flags = 0, photons > 1000): median +13 nominal nm, IQR 384 nm, 0.7 % within 100 nm of the limits.
  Sharp upper edge at about +250 nm (the coverslip side, presumably: microtubules near the glass); long tail to -650.
  A secondary population at +450 ... +700 nm appears mostly among flagged fits (`checks/z_hist.png`), and the
  flags = 0 fits at +300 ... +500 have high chi2: treat z > +300 with suspicion (possible wrong-branch or overlap fits).
- Height map (`checks/z_map_time.png`, 8 px bins): from about -430 to +380 nm over the cell (5-95 %), lower (more
  negative) in the cell centre, higher at the periphery: consistent with the cell being thicker in the middle if
  more negative z = farther from the coverslip, the same reading as for NPC. Sign still an inference.
- z over time: median (z - height map) changes by about -5 ... +7 nm over the movie (rough z-drift estimate).
- Renders: `checks/render_full.png` (whole field), `checks/render_zoom_all.png` (x 200-220, y 150-170 px, all frames)
  and `render_zoom_win1.png` (first 5,000 frames): microtubules sharp and continuous without drift correction.

## Assumed / unknown / possible problems

- **Bead-stack z step unknown, 10 nm/plane assumed** (z and crlb_z scale with it). Camera settings of the bead stack
  unknown.
- Pixel size unknown (Cspline submission implies 117.0 x 118.2 nm/px).
- Offset 400 ADU taken from the camera setting and the bead-stack background, not from the movie (movie fits give
  364-383); g = 17.0 from the movie, with a curved mean-variance relation.
- Only 7 calibration beads, all in the lower half of the field; field dependence not checked; PSF of beads on glass
  used for emitters inside the cell (no index-mismatch model).
- Dense data: 27.6 % with a neighbour in the patch, fitted as single emitters; precision ~20 % worse than the CRLB.
- z in nominal stage nm, no refractive-index factor, sign of "up" not established.
- Not drift corrected; consecutive-frame repeats not merged. The first 20,737 frames of the challenge movie are not
  in the provided files.
