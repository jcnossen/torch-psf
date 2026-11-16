# Field-dependent PSFs in several channels (stage 2b): mappings, joint bead fits, 2- and 3-colour beads

Status: Python. Code: `torch_psf/mapping.py` (ChannelMap, register_points, channel_rois), the mapped geometry of
`MultiChannelPSF` / `ChannelsPSF` (`torch_psf/multichannel.py`), `torch_psf/estimation/multichannel_beads.py`
(MultiChannelBeads, fit_multichannel_beads, multichannel_nll), `torch_psf/multichannel_estimator.py`
(MultiChannelBeadEstimator); tests `tests/test_mapping.py`, `tests/test_multichannel_beads.py`; study scripts
`study/multichannel/`; outputs `~/mcfield_out/` (not in the repo). Design: `field_psf_lib.md` section 7; stage 2a (one
channel): `field_psf_lib.md`.

RESULTS_SUMMARY

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

TWOCOLOUR

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

PROBLEMS
