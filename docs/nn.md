# torch_psf.nn: neural-network localization

`torch_psf.nn` is a neural-network localizer for SMLM movies. It is the detection network of SIMCODE v1 in its
plain-SMLM form ("V1Net"). The design follows DECODE (Speiser et al., *Deep learning enables fast and dense
single-molecule localization with high accuracy*, Nature Methods 18, 1082-1090, 2021), and the original network
design, the Gaussian-mixture loss and the 3 x 3 spatial integration are DECODE's. The SIMCODE version adds an
optional full x, y, z covariance per localization, an offset-aware merge of neighbouring detections, a training-scene
simulator with blinking and filaments, and calibrations of BatchNorm and of the sigmas. The package does not import
SIMCODE.

**Results.** On Tubulin-A647-3D (SMLM challenge 2016) at equal localization count, the full-covariance network
(`cov='full_xyz'`) gives FRC resolutions of 16.3 / 17.3 / 18.1 nm at K = 5 / 20 / 52 localizations per frame. DECODE
gives 17.6 / 17.7 / 18.8 nm. **These are SIMCODE-side results** (the network trained with SIMCODE's trainer). They
still have to be reproduced with `torch_psf.nn` on qnano1 (recipe below). What this repository shows is that the port
computes the same things: see [Equivalence with SIMCODE](#equivalence-with-simcode).

## The network (`SMLMNet`, model.py)

The input is a window of T = 3 consecutive frames t - 1, t, t + 1 in photoelectrons / px. The output is per-pixel
maps for frame t, `[B, C, H, W]`:

| channels | meaning |
|---|---|
| 0 | p: detection probability (sigmoid) |
| 1 .. n | mu of x, y (px, image coordinates: pixel centres at integers, x = column), z (um, the z of the PSF), bg (photoelectrons / px in frame t), photons (photoelectrons of the emitter in frame t) |
| n+1 .. 2n | sigma of the same parameters (marginal SDs) |
| 2n+1 .. 2n+3 | with `cov='full_xyz'`: partial correlations rho_xy, rho_xz, rho_yz given x (0.95 tanh) |

Here n = 5, or 4 with `z=False`. The mean offsets are tanh x `xyz_scale` (1 px, 1 px, 1 um) plus the pixel
coordinate. bg and photons are tanh x `max_bg` / `max_photons`. The sigmas are max(sigmoid, 0) + 1e-3 (DECODE's
floor) x the same scales. `xyz_cholesky(sigma, rho)` gives the Cholesky factor of the x, y, z covariance.

Architecture:

1. A frame U-Net with shared weights runs on every frame (1 -> 48 channels; levels 48, 96).
2. The 3 x 48 channels are concatenated.
3. A window U-Net (levels 48, 96, 192 -> 256 channels) processes them. Both U-Nets use a 1 x 1 stem, two 3 x 3
   conv + BatchNorm + ELU blocks per level, stride-2 2 x 2 convolutions down and nearest-neighbour upsampling with
   skips.
4. A head of a 3 x 3 conv (32 channels per output) and a grouped 1 x 1 conv (one group per output channel) gives the
   outputs. With full_xyz a zero-initialised 1 x 1 conv gives the correlations.

The Tubulin network has 2.43 M parameters. The input is normalised as (frames - 3) x 0.01. Frame sizes must be
multiples of 4; `localize_movie` pads.

## The loss (`GMMLoss`, loss.py)

DECODE's loss: every pixel k is one mixture component with weight p_k / sum p. Its mean is mu_k. Its covariance is
diagonal, or with full_xyz the x, y, z covariance of the Cholesky factor times independent bg and photon terms. The
loss of an image is -sum_j log sum_k w_k N(theta_j; mu_k, Sigma_k) over its targets, plus a count term
-0.01 log N(n_targets; sum p, sum p (1 - p)). The training loss is the mean over images.

## The training scenes (`Scenes`, scenes.py)

This is the SIMCODE plain-SMLM simulator (`simcode.plain.simulate.PlainSim`) with the labels of its live training
stream (`examples/baselines/decode_replay.Live`, `simcode.v1net.train.packet_to_batch`). It uses torch_psf PSFs
(`PSF.load` of the psf.zip, resized to a 23 x 23 render support) and a `torch.Generator`, so a seed gives the same
scenes. A batch is 16 fields of 40 x 40 px and 3 frames. It contains:

- **Blinking emitters.** Events arrive per field by a Poisson process (density log-uniform 3e-4 .. 1.5e-2 / px^2).
  Each event is on for an exponential time (mean on-time log-uniform 0.7 .. 5 frames), with the exact on-fraction
  per frame.
- **Re-blinks** (15 %) after a gap, **close pairs** (10 %, 0.3 .. 3 px apart) and **filaments** (in half of the
  fields).
- **z** uniform in -0.4 .. 0.4 um.
- **Photons.** A log-uniform median per field (180 .. 2700 photoelectrons per frame), log-normal per emitter, a 10 %
  jitter per frame, and a PSF blur jitter (0 .. 0.3 px).
- **Background.** Log-uniform per field (5.3 .. 93), with smooth structure and an AR(1) frame fluctuation.
- **EMCCD camera.** Poisson -> Gamma (EM register) -> read noise -> offset, ADU rounding, then a per-field gain
  error and offset error. The frames are converted back to photoelectrons with the nominal gain and offset.

**Labels.** An emitter is labelled when its rounded position is in the supervised square, its on-fraction in the
centre frame is >= min_on (0) and its centre-frame photons are >= label_min (100). The target columns are x, y, z,
the centre-frame background at its pixel and the centre-frame photons.

## Training (`train_model`, train.py)

```bash
python -m torch_psf nn train config.json [--out RUN] [--steps N]
torchrun --standalone --nproc_per_node 2 -m torch_psf nn train config.json     # optional, data parallel
```

There is one JSON config per run. `train.DEFAULTS` gives its keys, and the module docstring explains each one.

| key | contents |
|---|---|
| run | out, seed, steps, batch, size, halo, device, deterministic |
| optimiser | Lion lr, weight_decay, warmup, decay_every, decay_gamma |
| loss | count_loss_weight |
| logging | val_batches, log_every, save_every |
| model | the network options |
| scenes | psf, pixel_nm, pixel_nm_y, camera, label_min, min_on, and the simulator `ranges` (scenes.RANGES) |

The optimiser is Lion with a linear warm-up from lr / 100 over 512 steps, then the lr is halved every 40000 steps.
Every step uses a fresh scene batch.

**Seeds.** The stream uses the seed. The network initialisation uses seed + 1 and the validation batches seed + 7.
After a resume at step s, the stream restarts at seed + 1000 s. With `deterministic` (the default) the same config
gives the same weights, on the CPU and the GPU.

**Outputs** in the run directory:

- `config.json` (resolved, with the torch settings)
- `train.log` (one JSON line per log interval: losses, lr, speed and validation metrics: recall, precision, rmse,
  z-scores, and the 1-sigma coverage of the whitened errors)
- `last.ckpt`, `weights.step<N>.pt`
- at the end `weights.pt` + `model.json`, so the run directory is a model directory

Under torchrun, each rank simulates its own stream (seed + rank x 1000003). BatchNorm becomes SyncBatchNorm, the
batch is split, and rank 0 logs and saves. A single process runs the plain path.

**Converting a SIMCODE run.** `nn convert --run run.json --args args.json --psf local.psf.zip -o config.json` turns
a SIMCODE run.json and the stage args.json it reads its scenes from into such a config
(`convert_simcode_config`). It checks that the SIMCODE run is one that this package can repeat.

## Calibration (calibrate.py)

```bash
python -m torch_psf nn calibrate MODEL_DIR --bn --sigma -o NEW_MODEL_DIR    # uses MODEL_DIR/config.json
```

- **BatchNorm** (`bn_recalibrate`). The running statistics are reset and re-estimated as a cumulative average over
  256 fresh scene batches (seed 91170001, 4 batches per forward). The weights are unchanged, and this is checked.
- **Sigmas** (`fit_calibration`). Two sets of 150 scene batches are simulated, a fit set (seed 777001) and a check set
  (seed 777002). The detections (rule 'nms' raw_th 0.05, p >= 0.5, as SIMCODE's plain fit) are matched to the labels:
  Hungarian, with sqrt((dxy / 250 nm)^2 + (dz / 500 nm)^2) <= 1, and true z within the training range. The fit then
  depends on the model:
  - **full_xyz**: per photon bin (edges 0, 200, 400, 700, 1000, 1500, 2500 photoelectrons) a lower-triangular A with
    L -> L A. `rms` is the second moment of the whitened errors. `quantile` (robust, the default) scales A so that
    68.27 % of the squared Mahalanobis radii are below the chi2(3) quantile.
  - **diagonal models**: per bin and axis the 68.27 % quantile of |error / sigma|.

`calibration.json` holds the edges, `xyz_matrix` or `factors`, the method, and the check-set coverages before and
after. `localize_movie` applies it to the merged localizations.

## Localization (localize.py)

```bash
python -m torch_psf nn localize movie.tif --model MODEL_DIR --camera model -o locs.hdf5
# or --gain 0.0652 --offset 394.1 (e- / ADU incl. EM gain, ADU), or --camera calib.json; nothing: frames in photoelectrons
```

Each frame t is the centre of its window. Beyond the ends of the movie, the first or last frame is repeated. ADU are
converted to photoelectrons as (ADU - offset) x gain. Unlike `Camera.to_photons`, this does not divide by the EMCCD
excess-noise factor, because the network is trained on photoelectrons. `--camera model` uses the camera that the
training scenes were simulated for.

The maps are decoded by the offset-aware merge (`merge.merge`; SIMCODE's default rule):

1. Every pixel with p > 0.05 is a candidate with its own predicted position.
2. Candidates within 2 px whose positions agree (lateral gate sqrt(0.6^2 + s_i^2 + s_j^2), z gate likewise with
   0.3 um) form one emitter. The emitters are formed greedily in decreasing p.
3. The emitter's p is the summed p (clamped to 1). Its position, photons and bg are p-weighted means. Its
   covariance is the mean covariance plus the scatter of the members.
4. Rows with p >= 0.2 are kept. Choose the final threshold on `score`.

`--rule nms` gives DECODE's 3 x 3 rule.

The result is a torch_psf `Dataset` ([DATASET_FORMAT.md](DATASET_FORMAT.md)):

| field | contents |
|---|---|
| `pos` | x, y, z in px, with z in px of `pixel_nm` (the x pixel; `pixel_nm_y` for non-square pixels) |
| `crlb` | the calibrated sigmas, px |
| `photons`, `bg` | photoelectrons |
| `score` | the merged p |
| `sigma` | the network's sigmas |
| `s_photons` | the photon sigma |
| `cov` (full_xyz) | the calibrated covariances xy, xz, yz, px^2 |
| `rho` (full_xyz) | the network's correlations |

`meta` has `calibrated` and `photon_unit`.

## The model directory (files.py)

| file | contents |
|---|---|
| `weights.pt` | the state_dict |
| `model.json` | format `torch-psf nn model` v1: `architecture` (window, z, cov, widths, batch_norm), `scaling` (input_scale, input_offset, max_photons, max_bg, xyz_scale), `pixel_nm`, `pixel_nm_y`, `psf` (files + sha256), `camera` (torch_psf Camera dict of the training camera), `units`, `training`, `source`, `calibration` |
| `calibration.json` | optional |

`nn convert --run run.json --args args.json --psf tub3d_bead.psf.zip --weights v1net.pt --sigma-scale
sigma_scale_quantile.json -o MODEL_DIR` converts a SIMCODE V1Net (v1net.pt + v1net.json + its sigma_scale json). The
state_dict keys are renamed (unet_shared -> frame_unet, unet_combiner -> window_unet, output_heads_layer_main ->
head, cov_head -> corr_head, inc / encoders / decoders / outc -> stem / down / up / out), and the tensors are
unchanged.

## Equivalence with SIMCODE

`tests/test_nn_simcode.py` compares the port with SIMCODE at v1-dev 95ed17c. It needs `TORCH_PSF_NN_REF` (the
Tubulin reference files) and `TORCH_PSF_NN_SIMCODE` (the source, e.g. `C:/dev/simcode-dev@refs/tmp/ddp_stack`). It
checks:

- **Network.** The converted Tubulin weights give bit-identical outputs (CPU, fp32) on random and simulated windows.
  Random diagonal, 2D and full_xyz networks give the same.
- **Decode.** The merge (diagonal, full_xyz, precision weights) and nms give identical rows.
- **Loss.** The diagonal and full_xyz losses give the same value and gradients.
- **Calibration.** The rms and quantile `xyz_matrix` fits agree to 1e-12, and the application agrees.
- **Scenes.** Per-batch statistics match SIMCODE's `Live` on the Tubulin args.json: density, centre-frame emitters,
  labels, photons, on-times, background and frame statistics, all within 2 standard errors over 500 batches. The
  scenes are not bit-identical, because the random streams differ.
- **Training.** A smoke training of the full Tubulin network on the GPU: the loss decreases.

## Intentional differences from SIMCODE

1. **Random numbers.** They come from a `torch.Generator` (torch_psf's convention) instead of the global torch RNG,
   so the distributions are the same but the scenes are not bit-identical. The network initialisation runs under
   seed + 1 instead of the global RNG state that the scene setup left behind.
2. **EMCCD gain.** The Gamma register is sampled with Marsaglia-Tsang from the generator instead of
   `torch._standard_gamma` (which takes no generator). It samples the same distribution.
3. **Deterministic mode.** `deterministic` is on by default (`torch.use_deterministic_algorithms`, cuDNN
   deterministic). SIMCODE's V1Net trainer did not set it. Filament cumsums move to the CPU in that mode on CUDA, as
   in SIMCODE's repeatable helpers.
4. **halo > 0.** The loss sees only the supervised square, with coordinates shifted. SIMCODE's V1Net trainer fed the
   whole frame with shifted targets. The Tubulin run uses halo 0, where both are the same.
5. **Output layout.** `[B, C, H, W]` without SIMCODE's singleton frame axis. Parameter, option and state_dict names
   are torch_psf names (no aliases). The photon column is `photons`.
6. **Lion.** Lion is implemented here with lion-pytorch's update (tested identical), so it is not a dependency.
7. **Calibration fit.** The calibration fit matches with the model's pixel sizes. SIMCODE's plain fit hard-coded
   117 x 127 nm, which are the Tubulin values. The application runs in float64 (SIMCODE: float32).
8. **Movie localization** (edge frames repeated, padding to a multiple of 4, the Dataset table) is new. At this
   SIMCODE revision there was no plain-SMLM movie pipeline for this network.
9. **Logging.** Validation metrics go to train.log, with optional TensorBoard scalars. There are no image or CRLB
   panels. The checkpoints are `weights.step<N>.pt` and `last.ckpt`.

Not ported, all SIM-specific or not used by the plain network:

- V1NetI / V1NetP2 (intensity networks, pass 2)
- per-frame background maps (bg_frames > 1)
- SIM scenes (Sim6)
- the merge_soft / merge2 rules
- crowding-binned sigma factors
- the pattern / illumination / joint-solve code

## Retraining Tubulin-A647-3D with torch_psf.nn (qnano1)

```bash
python -m torch_psf nn convert --run run.json --args args.json --psf /mnt/data/dmdsf/simcode2_out/tubulin_0fea4ac_joint/psf/tub3d_bead.psf.zip -o tub_nn.json
# set "out" in tub_nn.json (or pass --out), then:
CUDA_VISIBLE_DEVICES=0 python -m torch_psf nn train tub_nn.json --out /mnt/data/<run>/tub_nn
python -m torch_psf nn calibrate /mnt/data/<run>/tub_nn --sigma --batches 400 -o /mnt/data/<run>/tub_nn_cal
python -m torch_psf nn localize Tubulin-A647-3D.tif --model /mnt/data/<run>/tub_nn_cal --camera model -o locs.hdf5
```
