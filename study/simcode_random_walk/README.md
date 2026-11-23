# SIMCODE random-walk bead benchmark

Each movie contains 100 frames of 400 × 400 camera pixels, two illumination axes with three phases, 200 nm pitch, 40 nm beads, and a 400 nm peak-to-peak axial random walk. The signed Noll 5–11 coefficients vary by seed. Python uses the browser generator's distributions and optics with an independent NumPy random stream. The camera planes are rendered with 3 × 3 subpixel stamps and six quadrature nodes per bead. Only the TIFF and acquisition parameters go to the browser estimator.

From this directory, generate a dense dataset with `python simcode_bead_movie.py dense`. Use `python simcode_bead_movie.py sparse --beads 1200 --seeds 17 83 519` for the lower density. Fit each TIFF using both the baseline (TorchPSF commit f351c3f) and updated browser app, saving each bead-fit.json as SEED-baseline.json or SEED-updated.json alongside the movie. Use the simcode Python environment with TorchPSF installed. The scoring scripts use CUDA.

`python score_fit.py dense` refines the baseline estimates in Python without access to truth during the fit. `python compare_web.py dense sparse` evaluates the browser results independently against withheld truth. Metrics are signed coefficient RMS across seven modes in milliwaves, absolute per-frame focus RMSE in nm, and relative L2 intensity error over all 100 noiseless 33 × 33 point PSFs. The focus is relative to the bead plane. The intensity metric includes errors from both the fitted pupil and the fitted focus.

| Beads | Seed | Coefficients (mλ), before → after | Focus (nm), before → after | PSF intensity error, before → after |
|---:|---:|---:|---:|---:|
| 12,000 | 17 | 0.55 → 0.48 | 17.8 → 8.8 | 3.05% → 1.52% |
| 12,000 | 83 | 0.56 → 0.44 | 36.9 → 4.8 | 6.80% → 1.16% |
| 12,000 | 244 | 0.58 → 0.44 | 34.3 → 5.4 | 6.63% → 1.13% |
| 12,000 | 519 | 0.96 → 0.85 | 87.4 → 4.8 | 15.20% → 1.18% |
| 1,200 | 17 | 1.17 → 1.23 | 30.0 → 25.9 | 5.48% → 4.48% |
| 1,200 | 83 | 0.48 → 0.39 | 52.9 → 8.3 | 9.15% → 1.46% |
| 1,200 | 519 | 0.52 → 0.60 | 47.8 → 10.6 | 8.36% → 2.16% |

The Python refinement reached dense PSF errors of 1.09–1.42% and sparse errors of 1.58–3.64%. Sparse seed 17 motivated the low-signal confidence check. It is part of development data. The simulator and fitter share the same vectorial optical model. Accuracy on experimental movies requires separate validation.
