# In-situ PSF estimation in SMLM: prior work

Web survey of 2026-11-13 (by a search agent; Nature and PMC pages were not reachable, so preprints and GitHub were
read instead). Entries marked [unverified] could not be confirmed from a source. Check all of it before citing.

| Method | PSF representation | Algorithm; optics model per emitter? | Data; how z and shifts are fixed | Reported cost | Stated limits |
|---|---|---|---|---|---|
| INSPR (Xu 2020) [preprint](https://www.biorxiv.org/content/10.1101/727354v1.full), [code](https://github.com/HuanglabPurdue/INSPR) | pupil, 21 Zernikes, scalar with OTF rescaling | assign spots to z classes by cross-correlation, average, phase retrieval, 6-10 rounds; model on the averages only | blinking data; biplane, or astigmatism with known orientation | <300 spots at high SBR, >2100 at low; runtime [unverified] | depth < 20 µm; higher modes ~41 % error |
| VISPR (Yang 2024) [preprint](https://www.biorxiv.org/content/10.1101/2023.11.03.565592v1.full), DOI 10.1364/PRJ.520469 | vectorial pupil, 21 Zernikes | same loop, then maximum-likelihood phase retrieval; model on the averages | blinking data; astigmatism prior | 500-1000 simulated molecules; no runtime | fails if all molecules at one z; no public code |
| uiPSF (Liu 2024) [preprint](https://www.biorxiv.org/content/10.1101/2023.10.26.564064v1.full), [code](https://github.com/ries-lab/uiPSF), DOI 10.1038/s41592-024-02282-x | voxel (beads only), pixel pupil, or Zernike; scalar or vectorial; blur, apodization, bead size, field maps, 4Pi | TensorFlow autodiff + L-BFGS, joint over PSF and emitters (per-emitter model evaluation inferred, not stated) | beads or blinking; localize, pick several thousand emitters across z, re-learn, 2-6 rounds; needs initial Zernikes | RTX 3080: 40 s (voxel, 20 beads); in situ 1 channel 4.7 min, 2 channels 13.1, 4Pi 35, field-dependent 49.7 | needs a wide z range; Zernike model blurry for Tetrapod |
| Vectorfit / NAT (Droste 2025, Optica) [preprint](https://www.biorxiv.org/content/10.1101/2024.12.11.627909.full) | vectorial, field-dependent Zernike surfaces | alternating maximum likelihood, emitters / aberrations (random 20 % subset); OTF precomputed on 20 x 20 field patches | blinking only; sign ambiguity by convention | 1,000-5,000 emitters, 30-45 s on 12 CPU cores; 3D fitting ~30x slower than spline | local maxima; residual attributed to higher-order and amplitude aberrations |
| LUNAR (2025) [preprint](https://www.biorxiv.org/content/10.1101/2025.02.16.638552v1.full) | vectorial, 21 Zernikes + Transformer localizer | self-supervised, alternating physics / network | dense raw frames | no time given | "computationally intensive" |
| Li cspline (2018) [preprint](https://www.biorxiv.org/content/10.1101/172643v2.full) | cubic spline, 64 coefficients per voxel | align and average bead stacks; no optics model | beads on coverslip; z scaled by 0.75 afterwards | 1.6e5 fits/s (GTX 1070) | depth-induced distortion; bead blur |
| ZOLA-3D (2018) [code](https://github.com/imodpasteur/ZOLA-3D) | 28-45 Zernikes with index mismatch | maximum-likelihood phase retrieval on beads | beads | ~2 h per reconstruction (K40) | |
| VIPR (Ferdman 2020) [preprint](https://www.biorxiv.org/content/10.1101/2020.01.02.892661.full) | pixel-wise pupil phase, vectorial | analytic gradients + Adam | bead stack | 31.5 s (RTX 2070, 300 iterations) | needs known optical parameters |
| FD-DeepLoc (Fu 2023) [preprint](https://www.biorxiv.org/content/10.1101/2022.10.14.512179v1.full) | vectorial, Zernike maps over the field | vectorial fit per bead, then a network | thousands of beads | <1 s per bead stack | no depth correction |
| PPG3D (2023) [preprint](https://www.biorxiv.org/content/10.1101/2023.07.30.551150.full) | PCA of calibrated PSFs | table interpolation | bead grid | ~100x faster than VIPR as generator | no physical model |

Findings:

* No published method was found that collects the gradient on an interpolation table and back-propagates once
  through the pupil model. Nearest: INSPR / VISPR (model on z-binned averages), Droste (precomputed OTF patches).
* No published spline or voxel PSF learned from blinking data was found. uiPSF states that voxel learning works
  for bead data only.
* Effects named as outside a Zernike vectorial model: discontinuous phase masks (uiPSF, VIPR), unmodulated
  light (PPG3D), pupil amplitude and higher-order aberrations (Droste; uiPSF apodization), residual blur (uiPSF),
  fixed dipoles (uiPSF future work; VIPR), bead size (uiPSF, Li), stage-scan versus in-sample PSF with
  supercritical light ([Siemons 2022](https://doi.org/10.1364/OE.461750): axial bias 140-250 nm), 4Pi /
  light-sheet / multi-channel (uiPSF), field dependence (FD-DeepLoc, Droste, uiPSF, PPG3D).
* Speed: Droste reports uiPSF's field-dependent mode at "more than half an hour on a GPU"; uiPSF recommends the
  vectorial model only "if computational resources permit".
* Only skimmed: LUNAR, and the 2026 preprints DECODE-Plex and 4Pi-BRAINSPOT.
