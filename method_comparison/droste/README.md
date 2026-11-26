# Droste NAT and field-dependent Zernikes

[Read the comparison PDF](droste_field_comparison.pdf). Pages 1–13 cover the field mapping, Fisher information, paired simulations, optimizer audit and native field parity. The remaining pages cover the experimental 1809 comparison.

The simulations use a joint calibration optimizer to isolate the effect of the field constraints. The 1809 experiment uses the alternating `torch_psf.estimation.fit_nat` estimator. It starts with the v4 optical geometry and a zero Zernike pupil on each illumination axis.

## Reproduce the report

Install this checkout of torch-psf. From this directory run `python compare.py` to rebuild the PDF from the adjacent results. To regenerate the numerical simulations run `python simulate.py`, `python matched_precision.py` and `python audit.py` first. The native field fixture can be generated separately with `vectorfit-py==0.3.4` and `python vectorfit_reference.py`.

With access to the 1809 acquisition, run the following commands, replacing the three input filenames and detection-cache directory with their local paths.

```sh
python sample_1809.py movie.tif detection_cache sample.npz
python fit_1809.py sample.npz camera_maps.npz v4_axis.json results_1809.json
python check_spline_1809.py v4_axis.json results_1809.json spline_1809.json
python polish_1809.py sample.npz camera_maps.npz v4_axis.json results_1809.json polish_1809.json
python gradient_check.py v4_axis.json results_1809.json gradient_check.json
python compare.py
```

`results_1809.json` includes both fitted 13-vectors, their field normalization and the inner optimizer status for each outer iteration. To reconstruct an axis, load the matching v4 optical configuration, zero its Zernikes, set its axial range to 0–0.4 µm, and call `nat_field(base, gamma, norm)`.
