# Auto PSF, step 1: band limit from the power spectrum, and which PSF route a movie needs

Status: first Python study ("figure out the processes") for a "drop in any movie, get the PSF" tool. Microscopy only.
Code: `torch_psf/auto/` (`spectrum.py`, `classify.py`, `analyze.py`, `synth.py`), tests `tests/test_auto.py` (CPU,
~12 s), study scripts `study/auto/` (`datasets.py`, `run_real.py`, `run_synth.py`, `summarize.py`, `plots.py`).
Outputs (not in the repo): `~/autopsf_out/` (`real_table.md`, `synth_table.md`, `json/`, `figs/`, `synth/`, `cache/`).

## API

```python
from torch_psf.auto import analyze_movie, band_limit, power_spectrum, fit_band_limit, movie_statistics, classify_movie

r = analyze_movie('movie.tif', pixel_um=0.108, wavelength_um=0.67)   # path, array [T, H, W] or a movie reader
print(r.summary())             # band limit + mode + reason + advice
r.band                         # BandLimit: status, message, fc_px +- fc_px_err (stat / sys), f_detect_px (lower bound),
                               #   na_over_lambda(_err), NA(_err), na_over_lambda_lower / NA_lower, floor, floor_slope,
                               #   band_snr, dchi2, anisotropy {fc per sector, ellipse, contrast_m2, ...}, model(f)
r.spectrum                     # Spectrum: f, p, se, blocks, count, p2d, sectors, tile, ...
r.mode                         # ModeResult: mode (smlm | sofi | zstack | static | no_signal), confidence, probs,
                               #   reason (plain language), key (the numbers behind it), stats, advice
r.to_dict()                    # json-able

bl, spec = band_limit(frames, pixel_um=..., wavelength_um=...)        # spectrum only
st = movie_statistics(frames, band=bl); m = classify_movie(st)        # classifier only
from torch_psf.auto import synthetic_movie                            # FFT-imaged test movies with an exact f_c
```

Frame choice (`analyze.choose_frames`): spectrum from 8 blocks spread over the movie after the first 5 % (dense dSTORM
start), block length grown so that ~64 M pixels are used (small frames need more frames; up to 4096); classifier on one
contiguous block of 400 frames at 40 % of the movie (all frames of shorter movies).

## 1. NA from the power spectrum

Incoherent imaging: the OTF is zero beyond f_c = 2 NA / lambda whatever the aberrations; camera noise is white. So the
ring-averaged power spectrum = band-limited signal + flat floor, and the spectrum gives f_c in cycles / px; with the
pixel size NA / lambda = f_c / 2 (1 / um), with lambda also NA.

Method (`spectrum.py`): tiles (side min(H, W, 256), up to 16 per frame) minus their mean, 2D Hann window, |FFT|^2 /
sum w^2 (white noise of variance s^2 -> s^2), averaged per block of frames; rings of 1 / tile cycles / px; lines kx = 0,
ky = 0 masked (row / column noise, EMCCD smear), sharp peaks > 4 x the ring median masked (SIMFLUX patterns). Fit over
f >= 0.3 f_c: floor + exp(poly2(f)) * OTF0(f / f_c)^2 (OTF0 = aberration-free circular pupil; the polynomial takes the
object spectrum, defocus, aberrations, pixel integration), weighted by the block standard error, from a grid of starts.
Uncertainty: bootstrap over blocks (100) + rms over 5 fit variants (fit start 0.2 / 0.3 / 0.45 f_c, degree 2 / 3).
Model-free lower bound f_detect: end of the first run (from low f) of rings where the excess over the floor is > 4 sigma
(5-ring running mean; sigma = noise-only ring error p / sqrt(N) plus 1 % of the floor for imperfectly white cameras).

What can be estimated is stated by `status` (in this order): `no_signal` (nothing significant, or only a weak non-white
structure without an edge: dark frames, flat light, fixed-pattern noise), `weak` (only low frequencies; lower bound),
`edge_unresolved` (no edge identifiable: > 20 % of bootstrap / variant fits run off the spectrum, signal beyond the
fitted edge, or the edge far beyond the last significant ring, f_detect < 0.85 f_c; lower bound only), `undersampled`
(signal beyond Nyquist; a value from the diagonal corners if an edge shows there, else a lower bound), `no_floor` (rings
beyond 1.1 f_c not flat), `weak` (relative error > 10 %), `ok`.

### What the measured cutoff is: the *effective* NA

Noise-free vectorial OTF^2 (VectorialPSF, 600 nm, 0.1 um px, relative to f ~ 0.01):

| case | 0.5 f_c | 0.8 f_c | 0.9 f_c | 0.95 f_c |
|---|---|---|---|---|
| NA 1.2, index matched, in focus | 9e-2 | 3e-3 | 2e-4 | 2e-5 |
| NA 1.49, index matched, in focus | 6e-2 | 1e-3 | 8e-5 | 3e-6 |
| NA 1.49, water (n_med 1.33), emitter at the coverslip | 3e-2 | 2e-4 | 9e-8 | 2e-8 |
| NA 1.49, water, emitter 1 um deep, in focus | 6e-3 | 4e-6 | 5e-10 | 5e-12 |

- The last ~10 % below f_c carry 1e-4 .. 1e-6 of the low-frequency power: what is "seen" depends on the SNR; the OTF0^2
  edge model extrapolates the last part.
- **Water samples under an NA 1.49 oil objective: the supercritical-angle band (2 n_med / lambda .. 2 NA / lambda) is
  invisible** even for emitters at the coverslip (1e-7 level in the vectorial model). Effective NA = min(NA, n_med) at
  best: 1.33 in buffer, ~1.40 in 50 % glycerol / mounting media.
- Depth (spherical aberration from the index mismatch), defocus spread and engineered 3D PSFs (tetrapod) lower the
  visible cutoff further. Apodization (pupil transmission falling at the rim) does too.

### Synthetic checks (`run_synth.py`; full tables in `~/autopsf_out/synth_table.md`)

Vectorial SMLM movies, 400 frames 128 x 128, 0.1 um px, 600 nm, 600 sites, photons 1000 / 5000:
- in-focus 2D and astigmatic PSFs, index matched, NA 1.0 / 1.2 / 1.4, 5000 photons: fit / true = 0.997 .. 1.04, status
  ok (errors 0.3-2 %). 1000 photons: 0.94 .. 1.02 or `edge_unresolved` (lower bound) at NA 1.4.
- NA 1.49 in water at the coverslip, 5000 photons: fit / (2 n_med / lambda) = 1.02 (in focus), 0.995 (astig, weak):
  the estimator sees n_med, not the objective NA.
- 1 um deep in water: 0.87 of the n_med cutoff (ok at 5000 photons; `edge_unresolved` at 1000).
- **Tetrapod: 0.74-0.88 of the true cutoff, often with status ok** (sometimes `edge_unresolved`; at 1 um depth and 1000
  photons 0.37 "ok"). This is the main failure: with a PSF whose OTF is small near the edge the fitted edge sits where
  the signal disappears and nothing in the spectrum alone shows that it is too low. Treat the cutoff of 3D-engineered
  PSFs as a lower bound.
- Sampling (scalar FFT-imaged movies, `synth.py`): f_c 0.40 / 0.48 -> 0.398 / 0.487 (ok); 0.56 (beyond Nyquist) ->
  0.566 from the spectrum corners (`undersampled`); 0.75 (beyond the corners) -> lower bound 0.67 (`undersampled`);
  150 photons x 50 frames -> `weak`, lower bound 0.29; 50 photons x 20 frames and pure noise -> `no_signal`.

### Real data (`run_real.py`; sources of the optics per data set in `study/auto/datasets.py`)

Expected f_c: 2 NA / lambda; and 2 min(NA, n_med) / lambda where the sample is aqueous. "?" = not documented.

| data set | px (um) | NA | lambda (um) | expected f_c (c/px) | fitted f_c (c/px) | f_detect | status | NA/lambda (1/um) | NA_eff | true mode | estimated (conf.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EPFL SOFI 2D Flip565 | 0.108 | 1.27 (water obj.) | ~0.60 | 0.457 | 0.398 +- 0.011 | 0.390 | ok | 1.84 +- 0.05 | 1.11 +- 0.03 | sofi | sofi (0.49) |
| FD-DeepLoc tetrapod movie | 0.110 | 1.35 sil. | 0.67 | 0.443; 0.437 | > 0.250 | 0.250 | edge_unresolved | > 1.14 | > 0.76 | smlm | smlm (0.62) |
| FD-DeepLoc beads, stack 20 | 0.108 | 1.35 | 0.68 | 0.429 | 0.346 +- 0.017 | 0.310 | no_floor | 1.60 +- 0.08 | 1.09 +- 0.05 | zstack | zstack (0.78) |
| user 2-col beads, left | 0.097 | 1.49 (fit 1.40) | 0.67 | 0.431 | > 0.438 | 0.438 | edge_unresolved | > 2.26 | > 1.51 | zstack | zstack (0.67) |
| user 2-col SMLM (2col/3, roi0) | 0.097 | 1.49 | 0.67 | 0.431; 0.385 | > 0.390 | 0.390 | edge_unresolved | > 2.01 | > 1.35 | smlm | smlm (0.76) |
| DNA-PAINT SIMFLUX 1809 | 0.10833 | 1.49 | 0.70 | 0.461; 0.412 | 0.379 +- 0.007 | 0.329 | no_floor | 1.75 +- 0.03 | 1.22 +- 0.02 | smlm | smlm (0.55) |
| DNA-PAINT 20 nM (dense) | 0.10833 | 1.49 | 0.70 | 0.461; 0.412 | 0.380 +- 0.004 | 0.346 | ok | 1.75 +- 0.02 | 1.23 +- 0.01 | smlm or sofi | sofi (0.40) |
| DNA-PAINT bead stack | 0.10833 | 1.49 | 0.70 | 0.461; 0.412 | 0.407 +- 0.033 | 0.354 | no_floor | 1.88 +- 0.15 | 1.32 +- 0.11 | zstack | zstack (0.72) |
| EPFL AO astig Flip565 movie | 0.105 | 1.49 | ~0.60 | 0.521; 0.466 | 0.351 +- 0.018 | 0.318 | no_floor | 1.67 +- 0.08 | 1.00 +- 0.05 | smlm | smlm (0.73) |
| EPFL AO astig bead stack | 0.105 | 1.49 | ~0.60 | 0.521; 0.466 | 0.426 +- 0.008 | 0.402 | ok | 2.03 +- 0.04 | 1.22 +- 0.02 | zstack | zstack (0.83) |
| EPFL Tubulin-A647-3D movie | ~0.117 | ? | 0.67 | - | > 0.462 | 0.462 | undersampled | > 1.97 | > 1.32 | smlm | smlm (0.85) |
| EPFL Tubulin-A647-3D beads | ~0.117 | ? | 0.67 | - | 0.451 +- 0.023 | 0.422 | no_floor | 1.93 +- 0.10 | 1.29 +- 0.07 | zstack | zstack (0.86) |
| EPFL NPC-A647-3D movie | ~0.138 | ? | 0.67 | - | > 0.415 | 0.415 | edge_unresolved | > 1.50 | > 1.01 | smlm | smlm (0.85) |
| EPFL NPC-A647-3D beads | ~0.138 | ? | 0.67 | - | 0.491 +- 0.011 | 0.458 | ok | 1.78 +- 0.04 | 1.19 +- 0.03 | zstack | zstack (0.83) |
| DECODE tubules | 0.127 x 0.117 | ? | 0.67 | - | > 0.508 | 0.508 | undersampled | > 2.08 | > 1.40 | smlm | smlm (0.77) |
| DECODE beads Pos0 | as above | ? | 0.67 | - | > 0.622 | 0.622 | undersampled | > 2.55 | > 1.71 | zstack | zstack (0.86) |
| dark (offset) frames | - | - | - | - | - | - | no_signal | - | - | no_signal | no_signal (0.88) |
| flat-light gain frames | - | - | - | - | - | - | no_signal | - | - | no_signal | no_signal (1.00) |
| live ER, mEmerald (eSRRF Set03) | ? | ? | ~0.51 | - | 0.619 +- 0.010 | 0.621 | undersampled | - | - | static (slow) | static (0.78) |

Reading:
- **Mode: 19 / 19 correct** (the 20 nM DNA-PAINT counts either way). Confidences 0.4-1.0; the low ones are honest
  borderline cases (EPFL SOFI is dense but kurtotic, DNA-PAINT SIMFLUX is sparse but long-lived and pattern-modulated).
- **Band limit: one clean "ok and close to nominal" case is rare in real data.** EPFL SOFI: a sharp, well-defined edge
  (excess drops from 0.3 to 0.01 of the floor within 0.03 c/px) at NA_eff 1.11 for the assumed 600 nm (nominal 1.27,
  a water objective into 50 % glycerol; 1.27 would need lambda 0.69 um). Either the emission is redder than assumed or
  the effective NA is lower (rim apodization / index mismatch). The DNA-PAINT sets (TIRF, emitters at ~0.1 um) give
  NA_eff 1.22-1.23 +- 0.02 at 700 nm, i.e. 92 % of n_med = 1.33, consistent with the synthetic "water, coverslip"
  case; their bead stack 1.32 +- 0.11. The AO bead stack gives 1.22 (lambda uncertain); the AO SMLM movie, imaged up to
  10 um deep in PBS, only 1.00: depth aberrations kill the high frequencies.
- **3D movies** (FD-DeepLoc tetrapod, user astigmatic SMLM, EPFL NPC): only lower bounds; the excess decays smoothly
  (nearly exponentially) into the floor. For the user 2-colour setup the beads give f_detect 0.438 c/px, which with
  97 nm / 670 nm means NA > 1.51 > 1.49: the assumed pixel or wavelength is off by >= 1.5 %, or the floor near Nyquist
  is not perfectly white at the 1 % level (the bound uses a 1 % floor systematic). Worth checking the pixel size.
- **Unknown optics**: the spectrum then constrains NA x px / lambda. DECODE (127 x 117 nm) is undersampled: signal to
  >= 0.51 c/px, so NA_eff > 1.40 at 670 nm. EPFL Tubulin-A647-3D reaches Nyquist (NA_eff > 1.32 at 117 nm). EPFL NPC
  beads: f_c 0.491 c/px; with the Cspline-implied 138 nm that is NA_eff 1.19, and with NA_eff <= 1.33 (water) the pixel
  must be <= 0.124 um, so the 135-142 nm implied by the Cspline submission looks too large (or the beads were not in
  water); a cross-check worth doing with a known target.
- **Failure cases and why**: (1) engineered 3D PSFs and depth: lower bound / underestimate; (2) real camera floors are
  not white at the 1-30 % level near the corners (sCMOS FD-DeepLoc, EMCCD iXon AO / NPC, the Evolve: rising floor;
  status `no_floor`); (3) bead z-stacks: the block bootstrap mixes plane-to-plane changes into the "noise" (se), so the
  fit weights are poor; (4) undersampled data give only bounds; (5) dark frames carry a weak fixed-pattern spectrum
  (row/column offsets) that a naive edge fit would call "signal" (`no_signal` now requires a band SNR > 0.1 for
  undersampled / unresolved spectra).

### What else the spectrum tells (aberrations)

- The support is a disc whatever the aberrations; an elliptical support in pixel frequencies would mean anisotropic
  pixel scales. The per-sector cutoff fit is too noisy for that: ellipse ratios 1.00-1.23 scatter on data whose pixels
  are square; DECODE (127 x 117 nm, a 1.085 ratio) is undersampled, so it could not be tested. Not reliable yet.
- The OTF *contrast* is anisotropic with astigmatism, but **in the time-averaged spectrum of an SMLM movie it cancels
  when the emitters are spread symmetrically around the astigmatic focus**: synthetic astigmatic SMLM, z symmetric: m2 =
  0.005 +- 0.017 (no astigmatism: 0.005 +- 0.004); emitters only below focus: m2 = 0.33. Real: the user astigmatic
  movie m2 = 0.28 (its z distribution is one-sided relative to the astigmatic midpoint), EPFL Tubulin / NPC 0.10,
  DNA-PAINT 20 nM 0.22 (an aberration or a structured sample: line-like objects also give m2 > 0; not separable).
- **Per-frame anisotropy spread** (`aniso_frame_rms`) does see balanced astigmatism: 0.025 without vs 0.225 with
  astigmatism (synthetic, same density); it also grows with fewer emitters per frame (interference of a few spots), so
  it needs a density-matched reference.
- **z-stacks**: the signed anisotropy along the stack (`astig_swing`, sign flip through focus) is a clear astigmatism
  detector: 1.1-1.25 on the bead stacks of the astigmatic setups (DECODE, EPFL Tubulin, AO, user beads), 0.76 EPFL NPC,
  0.56 on the DNA-PAINT stack; 0.18-0.19 on the non-astigmatic synthetic stack and the tetrapod beads (whose
  anisotropy rotates instead of flipping). On SMLM movies the same number stays 0.1-0.4 (random spots per frame).
- The floor: white-noise variance per pixel (ADU^2); with the camera offset, floor / (mean - offset) = gain x excess
  noise factor (not used yet).

## 2. Mode classifier (`classify.py`)

On a band-passed copy of the frames (difference of Gaussians with sigma = 0.42 / f_c px, the Airy-core width implied by
the spectrum, and 3 x that), with the white-noise variance in the filtered image from the spectrum floor:
dynamic and static band-limited power over the noise (`dyn_snr`, `static_snr`), lag autocorrelation of the fluctuations
(`rho[L]`, noise-free for L >= 1), per-pixel normalised 4th cumulant (`kurtosis`; blinking of few emitters -> large),
spots (local maxima > 5 sigma): per PSF area (pi (1.22 / f_c)^2), isolated fraction (no neighbour within 2 Airy radii),
persistence at lag L within 1.5 px and within 2 Airy radii (chance removed), per-frame sharpness (high-band fraction of
the in-band power) with its smooth trend (`trend_r2`, `trend_range`, peak position), the signed per-frame anisotropy.

Soft rules (log-logistic steps, multiplied down a small tree; confidence = probability of the winner):
signal (band SNR / 0.15, dyn SNR / 0.3, static SNR / 1) -> zstack (smooth sharpness trend R2 > 0.5 with range > 0.3,
spots stay within 2 Airy radii, not heavy-tailed) -> dynamic (dyn SNR > 0.08 and spots renewed at long lags) ->
sparse (geometric mean of: isolated > 0.45, < 0.15 spots per PSF area, kurtosis > 1) = smlm, else sofi unless slow
(rho at >= 20 frames / rho at 1 > 0.5 and kurtosis < 0.3: slowly moving, counted static).

Synthetic: 11 / 11 correct (scalar SMLM sparse and 0.07 / PSF, two SOFI settings, beads through focus with and without
astigmatism, static filaments bright and dim, noise, vectorial astigmatic and tetrapod SMLM). Real: 19 / 19 (table
above). Overview figure: `~/autopsf_out/figs/classifier_stats.png` (the separating statistics per data set).

Thresholds were set by hand on these data (risk of overfitting; 19 real sets): the SMLM / SOFI boundary is the least
certain (density per PSF area 0.1-0.2 with 30-60 % isolated is genuinely in between, e.g. dense DNA-PAINT).

## Figures

Per data set `~/autopsf_out/figs/<name>.png`: radial spectrum with fit, floor, fitted f_c +- error, f_detect and the
expected cutoffs; excess over the floor (log) with the standard error; 2D spectrum with the cutoff circles; per-frame
sharpness / anisotropy; lag correlation and spot persistence; the numbers and the decision. Synthetic: `synth_*.png`.

## Suggestions for the web "auto" page

1. One drop zone; read only what is needed (first / spread blocks of frames, tiles of <= 256 px) so a multi-GB movie
   gives an answer in seconds. Avoid parsing every TIFF page of multi-GB Andor / MM files (this study needed a strided
   reader: parsing 24k IFDs over a busy disk took > 30 min); interleaved multi-channel frames need a channel picker.
2. Show the radial spectrum (excess over floor, log) with the fitted cutoff band and f_detect, and the 2D spectrum with
   circles; let the user type pixel size / wavelength / NA and draw the expected 2 NA / lambda and 2 n_med / lambda
   circles live: the visual check is more convincing than the number. Say "effective NA" and the status message
   verbatim ("lower bound", "pixel too coarse", "not a fluorescence image").
3. If pixel and NA are given but not lambda (or lambda and NA but not pixel), show what the spectrum implies for the
   missing one, with a warning that effective NA < objective NA for aqueous samples.
4. Mode card: mode, confidence bar, the plain-language reason, 3-4 key numbers with tiny sparkline / bar visuals
   (persistence vs lag, sharpness vs frame) and a "route" button (SMLM -> detect & in-situ fit; z-stack -> bead fit;
   SOFI -> cumulant route; static -> NA only). Allow overriding the mode.
5. All of it is FFTs, Gaussian blurs and reductions: straightforward WebGPU (the existing detect / camera kernels cover
   most); the fit is a 5-parameter least squares on ~150 rings (CPU in the worker is fine).

## Open points / next steps

- A PSF-aware edge model (defocus spread, or the averaged |OTF(z)|^2 of a fitted pupil) instead of OTF0^2 x exp(poly),
  to remove the 3D-PSF bias; or flag 3D PSFs from spot size vs the diffraction width.
- Floor whiteness: estimate the camera's noise spectrum from dark / flat frames (or from frame differences of static
  regions) and divide it out before the fit.
- z-stacks: per-plane spectra, cutoff from the sharpest planes, and noise-only weights.
- Use the floor with the photon-transfer gain (torch_psf.camera.photon_transfer) for absolute SNR numbers.
- Thresholds of the classifier on more data (phone / widefield / confocal later); an explicit "bead time series
  (static beads)" and "drift" category.
