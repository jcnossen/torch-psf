# Field-dependent PSFs in several channels (stage 2b): mappings, joint bead fits, 2- and 3-colour beads

Status: Python. Code: `torch_psf/mapping.py` (ChannelMap, register_points, channel_rois), the mapped geometry of
`MultiChannelPSF` / `ChannelsPSF` (`torch_psf/multichannel.py`), `torch_psf/estimation/multichannel_beads.py`
(MultiChannelBeads, fit_multichannel_beads, multichannel_nll), `torch_psf/multichannel_estimator.py`
(MultiChannelBeadEstimator); tests `tests/test_mapping.py`, `tests/test_multichannel_beads.py`; study scripts
`study/multichannel/`; outputs `~/mcfield_out/` (not in the repo). Design: `field_psf_lib.md` section 7; stage 2a (one
channel): `field_psf_lib.md`.

Stage 2b implemented and tested. On 2-colour beads the per-channel field models reduce held-out deviance
(independent refit) from **1.3055 / 1.4655 to 1.2664 / 1.4199** (left/right); jointly trained PSFs give essentially
the same result. With shared held-out bead positions the joint affine field model scores **1.2859 / 1.4360**,
and a quadratic mapping gives **1.2820 / 1.4333**. Fixed-map global: 1.3207 / 1.4909; fixed-map field is invalid
(NaNs, rejected), so that ablation is incomplete. Bright-singleton mapping RMS improves 0.2308 -> 0.1903 px
(affine) / 0.1647 px (quadratic). The 3-colour experiment uses two camera channels with dye-dependent ratios;
broad-band beads do not provide three separate PSFs. CPU target tests: 7 passed, 2 skipped; full GPU suite:
135 passed, 9 skipped. psf.zip format unchanged; open issues and validation caveats below.

## 1. Model

```python
from torch_psf import ChannelsPSF, ChannelMap, FieldPSF, MultiChannelBeadEstimator
from torch_psf.mapping import channel_rois
mc = ChannelsPSF([FieldPSF(v0, f0), FieldPSF(v1, f1)], names=['left', 'right'], mappings=[None, map1])
rois, corners = channel_rois([left_frames, right_frames], frame, xy_ref, mc.mappings, 15)
ds = mc.fit(rois, corners=corners)            # positions in the reference (left) channel, all channels jointly
mc = PSFZip.read('psf.zip').multichannel(field=True, mapping=True)           # the same model from a file
```

- **Coordinates** (as psf.zip): every channel has its own crop coordinates (x = column, y = row, pixel centres at
  integers); channel 0 is the reference. An emitter has one position (x0, y0) in channel 0; channel c sees it at
  `from_ref_c(x0, y0)`.
- **ChannelMap** (torch module): the psf.zip mapping types in one class. Order 1 = affine, higher = the SIMCODE PolyMap
  (features u^i v^j, i + j <= order, i outer, u = (x - cx) / sx, v = (y - cy) / sy; one feats_norm for both
  directions). The model direction is reference -> channel (`inv` [nf, 2], px: the coefficients of normalised
  features, so one Adam step size fits all of them); `fwd` (channel -> reference) is refitted on a grid when the block
  is written (its rms error is stored as `roundtrip_px`); `to_ref` adds Newton steps on `from_ref`, so both directions
  agree to ~1e-6 px. `ChannelMap.translation(d)` is the old constant `channel_offsets = d` (tested to give the same
  images and derivatives). An order-1 map is written as an `affine` block, higher orders as `poly`.
- **MultiChannelPSF with mappings** (`set_mappings`, `ChannelsPSF(mappings=)`): the ROI of each channel is cut around
  the mapped position, and with the per-channel ROI corners `corners [N, C, 2]`

      pos_c = from_ref_c(corner_0 + h + (x, y)) - corner_c - h,   h = (P - 1) / 2

  d mu / d (x, y) = sum over the channel coordinates of d g_c / d pos_c times the mapping Jacobian (a rotation, scale
  or mirror changes the derivatives, not only the offsets), so LM fits, Fisher information and CRLB are exact for any
  affine or polynomial map (test: a mirrored and rotated channel, the fit reaches the CRLB). Clamp: x, y moved through
  the inverse Jacobian until every channel's position is inside its ROI; start: the least-squares position that
  centres the channels. Without mappings the old code path runs (the 4Pi and offset tests are unchanged).
- **Field dependence per channel**: a channel model may be a `FieldPSF` (or `TiledSplinePSF`); it is evaluated at its
  ROI centre `corner_c + h` in its own channel coordinates, i.e. with the polynomial (and feats_norm box) of that
  channel's psf.zip field block. `ChannelsPSF.to_tiled(tile)` gives the spline form per channel.

## 2. Joint estimation from multi-channel bead stacks

`MultiChannelBeadEstimator(regions=[(0, 0, 475, 400), (475, 0, 950, 400)], ...)`:

1. **Channels**: every frame split into the channel crops; field stop per channel and stack (`channel_stop_mask`: the
   median over the planes, a grey opening of 15 px, threshold at 35 % between the 1st and 90th percentile; the Otsu
   split of `field_stop_mask` fails on the right half of the 2-colour stack, whose inside has a gradient of 2x).
2. **Detection and registration**: the dense-bead detection per channel (bright / faint split per channel); channel c
   registered to channel 0 without a starting guess (`register_points`: four flip candidates, the peak of the
   histogram of all pairwise differences, mutual nearest neighbours refined with tolerances 4, 2, 1.5 px and an affine
   least-squares fit until the pairs are stable; then all stacks pooled, tolerances 2 and 1.5 px, order
   `mapping_order`).
3. **Beads and clusters** (`build_multichannel_beads`): detections of all channels mapped to the reference channel
   and merged (2 px); a bead is bright when it is bright in any channel; beads whose patch boxes overlap **in any
   channel** form a cluster; a cluster is used in channel c when all its bright members' boxes lie inside channel c's
   frame and field stop (else that channel does not see it; the other may). Per channel one `BeadClusters` (the
   pixel lists of fit_dense_beads), plus the global bead / cluster ids.
4. **Model** (`fit_multichannel_beads`), per channel c, cluster k, plane:
   `mu = bg_ck(x, y) + sum_b I_cbk PSF_c(from_ref_c(X_b + drift_s(k)) - box_cb, z_e; focus_k + df_b + dfc_c;
   Zernikes_c + field_c(x_cb, y_cb))`. Shared: the bead positions X_b (reference channel), the focus offsets df_b
   (bead height), the drift (one camera: the sample moves, the drift is mapped into every channel). Per channel:
   photons per bead and plane (or `intensity='ratio'`: shared photons times a per-bead ratio), linear background per
   cluster and plane, the VectorialPSF (Zernikes, NA and apod per channel, or `share=('NA', 'apod')`), the field
   polynomial (FieldZernikes in channel coordinates, its ridge prior), a focus offset dfc_c (chromatic focal shift;
   channel 0 = 0). The mappings of channels 1.. are fitted jointly (`fit_mapping`, all / affine / shift coefficients)
   from the registration. Poisson likelihood summed over the channels, Adam with the fit_dense_beads schedule (PSF
   held for the warmup, cosine decay), chi2 rejection per channel (a cluster leaves the fit when it is an outlier in
   any channel). Held-out clusters are refitted with PSFs, mappings and channel focus fixed (shared positions) and
   scored per channel. `joint=False`: every channel fitted alone (`data.channel(c)`, the map fixed, so the positions
   are free per channel), and the mapping refitted to the fitted bead positions of both channels.
5. **Output**: a ChannelsPSF (per channel the VectorialPSF with `focus_um` = the channel focus offset, or its
   FieldPSF; the fitted mappings; the median intensity ratio of the fitted photons); `est.save_zip(path)` writes one
   psf.zip with every channel's model, field block, mapping block, crop, flip and intensity ratio;
   `PSFZip.read(path).multichannel(field=True, mapping=True)` gives the same model back (identical renders, test).

## 3. Simulation (tests)

`tests/test_multichannel_beads.py` (GPU, 15 s each): two channels of 128 x 128 px, 21 planes over 1.2 um, 18 clusters
(23 beads) in two stacks, 20000 photons per bead (0.8 in channel 1), background 20; channel 1 is the reference seen
through an affine map (1.5 deg rotation, scales 1.004 / 0.998, shift (3.2, -2.7) px), with its own Zernikes (Z5 -20 /
+30, Z6 80 / 60, Z7, Z8, Z11 mλ), its own linear Z5 / Z6 field (20 - 40 mλ over the field) and a focus offset of 80 nm.
Registration of the max-projection detections: 0.48 px rms, the map off by up to 0.21 px over the field. Joint fit
(150 iterations from zero aberrations, stack 0 trained, stack 1 held out): the map within **0.011 px** of the truth
everywhere (test: < 0.05 px and < 1/4 of the registration error), the Zernikes including the field within **0.3 mλ
rms** (test: < 4 mλ everywhere), channel focus 80.1 nm (test: +-5 nm), held-out deviance 1.008 / 0.996 per px. The
estimator test runs the same through `MultiChannelBeadEstimator` on side-by-side frames and checks the zip round trip;
with `joint=False` the mapping refitted to the fitted bead positions is also within 0.009 px. CPU tests
(`test_mapping.py`, the data / registration test): the mapped fit of a mirrored and rotated channel reaches the CRLB,
`ChannelMap.translation` equals `channel_offsets`, the affine / poly blocks round-trip through the numpy reader.

## 4. 2-colour beads: independent and joint fits, per-channel field models

Scripts `study/multichannel/beads2col.py`, `score2col.py`, `fig2col.py`. Completed 2026-11-16;
outputs `~/mcfield_out/2col/`. Both scoring jobs already running at handover were allowed to finish without a
restart (main: 2319.45 s; fixed-map comparison: 1596.08 s, shared GPUs).

**Data and protocol.** The two 400 x 475 px camera halves of `beads.tif` (training) and `beads2.tif` (held out),
97 nm pixels, assumed 670 nm, NA 1.49 initially, n_imm 1.518, n_med 1.33, gain 0.597 e-/ADU, offset 100.6 ADU;
150 planes at 10 nm, every second plane used (75 planes at 20 nm). There are 219 merged beads in 103 clusters
across both stacks (216 / 210 beads visible per channel); the held-out stack has 109 beads in 54 clusters,
54 / 53 usable clusters per channel. The field stops retain 78.6 / 75.3 % of the channel crops, averaged over the two
stacks. Fits use patch 21, render pad 7, linear backgrounds, free per-channel photons, per-bead focus offsets and
shared drift; base Z5-Z21, NA and apodization fitted for 300 iterations, then another 300 with the mixed field
polynomial (35 coefficients per channel; Z5, Z6, Z11 quadratic, Z7-Z10 and Z12-Z17 linear, prior sd 0.05 waves).
`joint_shared` ties NA and apodization between channels; `joint_poly2` fits an order-2 mapping;
`joint_fixmap` fixes the detection-registration affine map during both training and scoring.

Every held-out cluster's nuisance parameters are refitted for 200 iterations with the PSFs and mappings fixed.
Two scoring protocols are reported: **independent refit** permits separate bead positions/focus per channel;
**joint refit** shares them and fixes the trained channel focus (for independently trained models only, channel
focus is estimated in this scoring refit). Aggregates are selected once from independent/global: chi2 > 3 times
the channel median in either channel, cluster ids 0, 1, 11. The same exclusions apply to every row, leaving
51 left / 50 right clusters. The independent baselines repeated in the fixed-map scoring job agree within
3e-8 deviance/px and select the identical exclusions. All-cluster scores and paired NLL differences are retained
in `score2col.json` and `score2col_fixmap.json`.

Held-out Poisson deviance per pixel (lower is better):

| training / mapping | PSF | independent refit left | independent refit right | joint refit left | joint refit right |
|---|---|---:|---:|---:|---:|
| independent / affine refit to positions | global | 1.3055 | 1.4655 | 1.3235 | 1.4820 |
| independent / affine refit to positions | field | 1.2664 | 1.4199 | 1.2858 | 1.4369 |
| joint / fitted affine | global | 1.3062 | 1.4649 | 1.3248 | 1.4826 |
| joint / fitted affine | field | 1.2673 | 1.4190 | 1.2859 | 1.4360 |
| joint, shared NA/apod / fitted affine | global | 1.3064 | 1.4678 | 1.3271 | 1.4868 |
| joint, shared NA/apod / fitted affine | field | 1.2655 | 1.4219 | 1.2862 | 1.4402 |
| joint / fitted quadratic | global | 1.3060 | 1.4648 | 1.3210 | 1.4795 |
| joint / fitted quadratic | field | 1.2669 | 1.4191 | 1.2820 | 1.4333 |
| joint / fixed registration affine | global | 1.3006 | 1.4707 | 1.3207 | 1.4909 |
| joint / fixed registration affine | field | invalid (NaN) | invalid (NaN) | invalid (NaN) | invalid (NaN) |

- The field improves both channels consistently: independent training/scoring gives 1.3055 -> 1.2664 left and
  1.4655 -> 1.4199 right (-0.01957 / -0.02279 NLL/px, better in 43/51 and 44/50 clusters). Joint training with
  the affine map gives 1.3248 -> 1.2859 and 1.4826 -> 1.4360 under the joint refit. Independently and jointly
  trained PSFs score almost identically under a common refit protocol; sharing positions adds about 0.018-0.020
  deviance/px, so it does not remove the residual mismatch between channels.
- The quadratic mapping gives a small further improvement under joint scoring (field 1.2820 / 1.4333 versus
  affine 1.2859 / 1.4360). Shared NA/apodization adds no gain (field 1.2862 / 1.4402); separate effective optics
  per channel remain appropriate (joint field NA 1.3983 / 1.4326 at the assumed common wavelength).
- Fixing the registration map gives mixed global changes: left 1.3207 versus fitted-map 1.3248, right 1.4909
  versus 1.4826. The fixed-map **field fit is invalid**: NA, apodization, focus and likelihood became NaN;
  `report.json` has `field.kept=false`, `chosen=global`. Its saved field zip was scored by the pre-existing job
  and also gives NaNs. There is no valid fixed-map field ablation; this numerical failure is an open issue,
  not evidence for or against fitting the map. The cause has not been established and no replacement run was made.
- These absolute left-channel numbers differ from stage 2a's 1.3384 -> 1.2992 because the joint detection/cluster
  construction and aggregate mask differ (51 clean left clusters here, 52 there). The field gain is the same
  approximately 0.039 deviance/px; compare methods within this matched table.

**Mapping.** Detection registration: no flip, 183 paired detections, RMS 0.5767 px; centre shift
(+0.5774, -6.3031) px, rotation 1.03771 deg, scales 1.001643 / 1.001207. The jointly fitted field affine is
`right = A @ left + b` in each half's crop coordinates:

```text
A = [[1.0013239834, -0.0171636683],
     [0.0190130824,  1.0014968437]]
b = [3.6987877894, -10.9706760708] px
```

Its shift at (237.5, 200) is (+0.5805, -6.1557) px, rotation 1.03482 deg, scales 1.001504 / 1.001644;
right-channel focus -77.68 nm (global -76.80 nm), photon ratio right/left 0.8298. The mapping changes by up to
0.28 px in either coordinate over the crop relative to registration. Residuals below are radial distances
`map(left) - right` using the **independently fitted field-model bead positions**, in pixels:

| mapping | bright beads (n=178): RMS / median | bright singletons (n=53): RMS / median | faint beads (n=29): RMS / median |
|---|---:|---:|---:|
| registration / fixed map | 0.3589 / 0.2005 | 0.2308 / 0.1951 | 1.2048 / 0.7449 |
| independent field affine | 0.3195 / 0.1268 | 0.2061 / 0.1375 | 1.1514 / 0.6684 |
| joint global affine | 0.3231 / 0.1174 | 0.1914 / 0.1015 | 1.1740 / 0.6837 |
| joint field affine | 0.3226 / 0.1150 | 0.1903 / 0.1064 | 1.1739 / 0.6854 |
| joint field quadratic | 0.3182 / 0.0827 | 0.1647 / 0.0752 | 1.1867 / 0.6798 |

For bright singletons the fitted affine reduces RMS from 22.39 to 18.46 nm and median from 18.92 to 10.32 nm;
quadratic gives 15.97 / 7.29 nm. The much larger faint-bead residuals and outliers in dense clusters dominate
all-bead RMS, so these are consistency residuals, not a ground-truth localization error. JSON also contains the
saved-fit training/non-training split; its non-training group includes rejected training clusters as well as the
held-out stack. Registration uses detections pooled from both stacks, and `_combine` refits the independent
models' map to fitted positions from both stacks. Thus the independent-map scores/residuals are **not a strict
train-only validation of the mapping**. A mapping validation that holds out all registration and position pairs
from the second stack remains open; no held-out PSF coefficients are fitted in these scoring refits.

**Per-channel field maps.** `~/mcfield_out/2col/figs/field_maps.png` compares independent and joint fits with a
shared colour scale for each channel/mode pair. Ranges are base plus field, mλ, within that channel's training-stack
field stop on a 5 px grid; correlation is between the field offsets over the full crop grid (clipped to each
field's fitted box):

| mode | joint left range mλ | joint right range mλ | independent vs joint r: left / right |
|---|---:|---:|---:|
| Z5 | -50.8 .. 4.3 | -39.9 .. 10.9 | 0.9999 / 0.9999 |
| Z6 | 57.1 .. 104.4 | 63.3 .. 109.7 | 0.9997 / 0.9990 |
| Z7 | -35.0 .. -32.2 | -30.7 .. -26.6 | 0.9924 / 0.9999 |
| Z8 | -10.0 .. -3.6 | -10.8 .. -2.5 | 1.0000 / 0.9993 |
| Z11 | -15.7 .. 0.1 | -18.0 .. -10.4 | 0.9892 / 0.9486 |
| Z12 | -2.4 .. 8.5 | -4.3 .. 5.5 | 0.9988 / 0.9980 |
| Z13 | -12.0 .. 2.7 | -8.0 .. 0.6 | 0.9998 / 0.9997 |

Independent and joint field patterns agree closely; the largest range is in Z5/Z6 (about 47-55 mλ), while coma
Z7 changes by only 3-4 mλ. The weaker right Z11 pattern is less stable (r=0.9486). Maps are evaluated in each
channel's own coordinates. `~/mcfield_out/2col/field_maps.json` stores ranges, means and correlations;
`~/mcfield_out/2col/figs/mapping.png` shows the bright-bead residuals and the fitted-affine minus registration
map (arrows amplified x100 / x200, y increasing downwards).


## 5. 3-colour beads: how 3 colours map onto 2 camera halves

Scripts `study/multichannel/look3col.py` (detections, registration, box photometry), `beads3col.py` (the estimator,
both halves; `--mode joint | indep`, `--zb` bead-centre height), `analyze3col.py` (per-bead ratios, PSFs, mapping),
`compare_smlm_maps.py` (against the SMLM analyses' affine maps). Outputs `~/mcfield_out/3col/`. Data
`june/astig/3col/beads/beads1.tif` (training) and `beads2.tif` (held out): 180 planes, 400 x 950 px, assumed 10 nm
steps (the SMAP calibrations of the same stacks say dz 10 nm) and the 2-colour optics / camera.

**What the data show.**

- Both halves see the **same beads**: the registration finds no flip, a shift of (+7.3, -73.2) px (right minus left at
  the centre of the half), a rotation of 1.04 deg and scales 1.000 - 1.003 (1.04 deg / 1.001 - 1.002 in the 2-colour
  stacks, whose shift is (+0.6, -6.2) px: the splitter was re-aligned between the two experiments, its rotation is the
  same). With the 73 px vertical shift the top 73 rows of the left half have no partner on the right (those beads are
  fitted in the left channel only).
- There is no field stop in these stacks: each half is lit by a round illumination spot (`field_stop=False`).
- **Every bead appears in both halves with one ratio**: photons right / left per bead (joint fit, all planes) = 0.81
  median, 0.72 - 0.92 over the 12 beads seen in both halves (CV 7 %; box photometry of the 14 brightest detections:
  0.70 - 0.87); no groups of ratios. These are broad-band beads: the bead stack does not separate colours.
- The colours are separated **ratiometrically** in the SMLM data of the same setup (`3col/*/ratios*.hdf5`,
  `ratio-histogram.png`, `merged-decode`, per-structure files `mito`, `tub`, `clath`): every emitter is localized in
  both halves (two 200 px ROIs, `info.yaml`), the halves are merged with an affine map (`affine_mapping.txt`) and the
  dye is assigned from the intensity ratio of the two halves: one dichroic splits each dye's emission between the
  halves at a dye-specific ratio. The histograms show it: `tirf/4` (mito + tubulin) two peaks at ratio ~1.1 and ~2.0,
  `hilo/6` (clathrin) one peak at right / left ~0.25. So **3 colours = 3 ratio classes on the same 2 channels**; the
  PSF model is the same per half for all dyes (up to the wavelength dependence within a half, not resolved here), and
  the colour is an emitter property (its ratio), not a channel.
- The bead mapping agrees with the SMLM analyses' affine maps: expressed in their ROI coordinates, `tirf/4`, `hilo/2`,
  `hilo/3`, `hilo/5` agree with the bead map to 0.55 - 1.0 px over the 200 px ROI (their 2 x 2 part is the transpose
  of ours: a row / column convention), `hilo/1`, `8`, `9` to 1.0 - 1.7 px in (y, x) order; `hilo/10` differs by 5 px,
  `hilo/6`, `hilo/11` and `exp3D-hilo/2` have scales of 1.3 - 1.9 (not valid maps). The 2-colour SMLM ROIs
  (`2col/3`, `2col/4` `info.yaml`: (554, 37) and (76, 46)) are placed so that the bead map takes the left ROI onto
  the right one to within (-0.03, -0.57) px; `mapped.yaml` there is only the metadata of a mapped localization file
  (no mapping). The old MATLAB calibrations (`left_3dcal.mat`, `right_3dcal.mat`, also `beads_left_3dcal.mat` in the
  2-colour folder) are SMAP cspline calibrations of each half separately (100 nm pixels assumed, dz 10 nm, no channel
  transformation: `cspline.global.isglobal` 0, `transformation` empty).

**Fits** (`beads3col.py`, NA / apod fitted, no field model: 15 beads, 12 in both halves; detection `rel_threshold`
0.02, no chi2 rejection: chi2 grows with the photons and the brightest beads would be rejected):

| run | left: deviance / px train / held-out | right | NA left / right | Z6 mλ | Z7 mλ | Z11 mλ | channel focus right (nm) | ratio right / left | mapping: rotation deg, scale x / y |
|---|---|---|---|---|---|---|---|---|---|
| indep | 11.53 / 12.53 | 8.92 / 8.58 | 1.374 / 1.390 | 82 / 88 | -53 / -36 | 56 / 55 | - | 0.774 | 1.051, 0.9997 / 0.9997 |
| joint | 11.74 / 12.75 | 8.99 / 8.52 | 1.368 / 1.400 | 83 / 88 | -48 / -42 | 53 / 60 | -37 | 0.809 | 1.040, 0.9994 / 0.9998 |
| joint_zb0.1 | 12.37 / 13.66 | 9.41 / 9.04 | 1.365 / 1.409 | 84 / 90 | -48 / -44 | 57 / 67 | -21 | 0.837 | 1.041, 0.9994 / 0.9998 |

## 6. psf.zip

Unchanged. A multi-channel result is written as one zip with C independent channels: each the VectorialPSF at the
reference point (`focus_um` = the channel focus offset, so all channels share the emitter z, as the spec says) plus
its `zernike_poly` field block in channel coordinates (`feats_norm` = the box of that channel's training beads), the
`mapping` (`affine` for order 1: the channel -> reference matrix; `poly` for higher orders, with fwd refitted as the
inverse of inv and the extra key `roundtrip_px`), `crop` (the half of the camera frame), `flip`, `intensity_ratio` (the
median photon ratio of the beads). `PSFZip.read(p).multichannel(field=True, mapping=True)` rebuilds the model; renders
are identical (test).

Gaps found (not changed; to decide):

1. **Ratiometric colours** (section 5): `intensity_ratio` is one number per channel ("expected photons in this channel
   = ratio x I"), but with several dyes in one acquisition the ratio is a property of the dye. There is no key for a
   table of species ratios (e.g. `species: [{name, intensity_ratio: [C]}]`); `per_channel_params` could declare `I`
   free per channel (the spec allows the key, the writer always writes `["bg"]`), or the table goes to `acquisition`
   (free-form) for now. The bead ratio written now (0.81 right / left for the 3-colour beads) is the beads' ratio, not
   a dye's.
2. **Support of a poly mapping**: the field block has its fitted box (feats_norm + the clip rule), a `poly` mapping
   does not: it is evaluated (and extrapolated) anywhere; an order-2 map fitted on the beads may be unreliable at the
   frame corners. An optional `support` box, or the rule that `feats_norm` is the fitted region, would say where it
   holds.
3. **fwd / inv of a poly mapping** are two independent polynomials in the spec; a writer that fits one direction
   (here inv, the model direction) can only approximate the other (`roundtrip_px` 3e-4 px for the order-2 map here;
   an extra key, ignored by readers). The spec could state which direction is authoritative.
4. Per-channel wavelength: each channel's model carries its own optics (the spec allows it); here both use the
   assumed 670 nm and the NA fit absorbs the difference (left 1.398, right 1.433), so the stored NA is effective.

## 7. Validation and remaining numerical issue

Validation on qnano1, 2026-11-16, Python `~/miniconda3/envs/simcode-py311/bin/python`, this worktree on
`multichannel-field`. CPU command used `CUDA_VISIBLE_DEVICES=` and `PYTHONPATH=$PWD`:
`python -m pytest -q tests/test_mapping.py tests/test_multichannel_beads.py --durations=0`.
**7 passed, 2 skipped, 1 warning in 3.85 s**; `/usr/bin/time -p`: real 6.59 s, user 14.59 s, sys 1.34 s.

| CPU test | call time (s) |
|---|---:|
| test_channel_map_affine_poly_and_blocks | 0.06 |
| test_register_points_flip_rotation_outliers | 0.23 |
| test_translation_mapping_equals_channel_offsets | 0.01 |
| test_mapped_jacobian_and_fit_reach_crlb | 0.67 |
| test_channel_rois_and_field_channels | 0.01 |
| test_psfzip_round_trip_mapped_field_channels | 2.07 |
| test_data_registration_and_masks | 0.39 |

No additional `@needs_gpu` markers were needed: every executed CPU test is below 30 s. The two existing markers
in `test_multichannel_beads.py` remain on `test_joint_fit_recovers_mapping_aberrations_fields` and
`test_estimator_two_channels_zip` (they explicitly use CUDA); both were exercised in the GPU suite, taking
5.87 s and 9.97 s respectively.

Full suite, least-memory-loaded GPU 0 selected after `nvidia-smi`:
`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m pytest -q tests/ --durations=20`.
**135 passed, 9 skipped, 19 warnings in 403.18 s**; real 407.02 s, user 682.68 s, sys 29.09 s.
Warnings: one tensor-to-scalar warning and 18 TorchScript deprecation warnings. Logs:
`~/mcfield_out/logs/tests_cpu.log`, `tests_gpu.log`. Figures were generated successfully by
`python -I study/multichannel/fig2col.py` and visually inspected; log `~/mcfield_out/logs/fig2col.log`.

The fixed-map field NaN failure in section 4 remains unresolved (the estimator selected the valid global model).
A stable fixed-map field fit is needed before drawing a full fitted-versus-fixed-mapping field comparison.
The mapping validation caveat in section 4 and psf.zip questions in section 6 also remain open. No merge, push,
format change, or changes to other worktrees were made.
