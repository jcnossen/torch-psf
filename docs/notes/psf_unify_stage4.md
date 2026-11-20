# PSF unification, stage 4

The baseline is `f4d4ff9` on `psf-unify`. The zip writer takes the maximum version declared by the
models, compositions and sources it serializes. Standalone JSON uses the same feature levels and keeps
the 0.1.0 header for baseline single-model files.

| Minimum zip version | Features |
|---|---|
| 0.2.0 | Gaussian models, box optics, scalar Catmull-Rom pixel splines and phase grids, channel geometry and nearest pixel tiles |
| 0.3.0 | Non-box vectorial pixel integration and independent B-spline density tables, including per-angle tables |
| 0.4.0 | Non-box 4Pi optics, integrated phase grids, dipole component tables, integrated or bilinearly blended tiles |
| 0.5.0 | Catmull-Rom density, B-spline pixel and per-angle pixel grids in the explicit `spline_table` block |

Python accepts zip minors 0.2 through 0.5 and standalone minors 0.1 through 0.5. The web reader accepts
zip minors 0.2 and 0.3. In particular, a 0.4 reader assumes fixed interpolation rules when it rebuilds
a phase table or a field tile. A new minor prevents that assumption from changing the pixels.

A Catmull-Rom density grid stores `basis="catmull_rom"`, `integrated=True` and density samples in
`coefs`. A B-spline pixel grid stores `basis="cubic_bspline"`, `integrated=False` and prefiltered
pixel coefficients in `knots`. Each scalar volume is `[nz,n,n]`. The phase, angle, dipole and field
layouts are specified in the [format reference](../PSF_FORMAT.md#model-types-channelscmodel-sourcemodel).

The plain builder default remains `('catmull', 'pixel', 2)` in `torch_psf.spline.DEFAULT_SPLINE`.
The round-trip tests change that constant to each density kernel at two knots per pixel, build a
table, change the default again, and compare the loaded coefficients, renders and derivatives exactly
for both JSON/NPZ and zip. Explicit sampling profiles for phase, field and calibration builds retain
their previous settings.

## Numerical and archive checks

On Windows with CPU float32 and one PyTorch thread, all 205 unrounded arrays matched `f4d4ff9` byte
for byte. Each render case includes the forward result, the value from the gradient path, and all
returned derivatives. Orientation includes the images, Jacobian and fitted parameters. SOFI includes
the images and Jacobian.

| Case | Arrays compared | Result |
|---|---:|---|
| `box` | 5 | Exact |
| `box_dipole` | 5 | Exact |
| `box_blur` | 5 | Exact |
| `box_blur_dipole` | 5 | Exact |
| `box_angle` | 5 | Exact |
| `spectral` | 5 | Exact |
| `spectral_dipole` | 5 | Exact |
| `spectral_blur` | 5 | Exact |
| `spectral_blur_dipole` | 5 | Exact |
| `spectral_angle` | 5 | Exact |
| `spectral_angle_mean` | 5 | Exact |
| `gauss_legendre` | 5 | Exact |
| `gauss_legendre_dipole` | 5 | Exact |
| `gauss_legendre_angle` | 5 | Exact |
| `exact_prototype` | 5 | Exact |
| `focus_z` | 5 | Exact |
| `polychromatic` | 5 | Exact |
| `gauss2d` | 5 | Exact |
| `gauss3d` | 5 | Exact |
| `cubic` | 5 | Exact |
| `cubic_dipole` | 5 | Exact |
| `integrated` | 5 | Exact |
| `integrated_angle` | 5 | Exact |
| `integrated_angle_mean` | 5 | Exact |
| `integrated_dipole` | 5 | Exact |
| `integrated_angle_dipole` | 5 | Exact |
| `pixel_catmull` | 5 | Exact |
| `pixel_catmull_conv` | 5 | Exact |
| `pixel_bspline` | 5 | Exact |
| `pixel_bspline_conv` | 5 | Exact |
| `fourpi_box` | 6 | Exact |
| `phase_cubic` | 6 | Exact |
| `fourpi_spectral` | 6 | Exact |
| `phase_integrated` | 6 | Exact |
| `phase_dipole` | 6 | Exact |
| `channels` | 5 | Exact |
| `field` | 5 | Exact |
| `field_bound` | 5 | Exact |
| `tiles` | 5 | Exact |
| `orientation` | 3 | Exact |
| `sofi` | 2 | Exact |

All 13 archive manifests and complete SHA-256 hashes matched the stage 3 pins on the pinned writer
stack (Windows, Python 3.12.2, torch 2.5.1, NumPy 1.26.4, matplotlib 3.8.0, tifffile 2024.2.12,
Pillow 10.2.0, zlib 1.2.13). Loading these archives in both the stage 3 and stage 4 readers produced
69 byte-identical forward, value and derivative arrays across the 13 cases.

| Archive | Version | Verified SHA-256 |
|---|---|---|
| `box` | 0.2.0 | `420c099653da00424c0dae4223ce0e230f78b692c3b146e615759f0369ae8962` |
| `channels` | 0.2.0 | `4a463309d37ed7a8f06b62650e5ba56bb6348d1c84b46ec65f93e93763310902` |
| `cubic` | 0.2.0 | `17c25cdfc1cadb699ea50cba046d37c371fb32c3f7b59058296985198dff02fe` |
| `field` | 0.2.0 | `aebb80b6e4daf06b2f236ee5a3ebe523083123290ca5383ca241b28ec5037a07` |
| `fourpi_box` | 0.2.0 | `d587b14320b17fa7f44a77b9e1eb1ea46a87367bcde6d251f52aae45049cf56e` |
| `gauss2d` | 0.2.0 | `c380c7458659c4e5e7b4767541c82a1519c72631503b4840207b71f391f575ef` |
| `gauss3d` | 0.2.0 | `463830bae7e18c4e94ec8a4a773d8b614abcd1ce8103c57c95098425c12c0610` |
| `integrated` | 0.3.0 | `661e35644b556eb596e43ee4ff7c95f7686ded7cdf05bb67a2958fc375f68d6d` |
| `integrated_angle_dipole` | 0.4.0 | `751a71226188756967c20ba0bdb607f07349d68ddff16b5b621fbbe9e4203c51` |
| `phase_cubic` | 0.2.0 | `a33cd81af328cb8c7727c79ae97e115435a2ba04ec2c241355695c5bb49e4c03` |
| `phase_dipole` | 0.4.0 | `cc291208ad853d8093e8f953b0a8ff01cdf3dab09631d41b40fac722317841a9` |
| `phase_integrated` | 0.4.0 | `98ff4354b750131d90837e8cedcbcf9196f57868c566e5e3757f981b777ad075` |
| `tiles` | 0.2.0 | `2d91900b74c0cfc430c6cf2ad3b24dff89da01e59eabb33c88d8ff4278217fa0` |

The private `NPC_f_dxy15` bead-fit state was available locally. On CPU, the optical and phase-spline
renders and gradients at two off-centre, nonzero-phase positions matched both `4b0ffc3` and
`f4d4ff9` byte for byte. The spline coefficients also matched at ROI 15, two knots per pixel,
z range ±0.04 µm, dz 0.02 µm, zpad 3 and xy limit 1 px. The full-size rebuild remains for the server.
The m1809/m25 v4 calibration files and `check_v4.py` were unavailable locally. The synthetic
per-angle calibration pins passed.

The 0.5 phase tests cover stored float32 tables and a same-device rebuild. Rebuild recomputes the
optical source on the available device. The unit tests' CPU-generated coefficients were
bit-identical on CPU rebuild. Rebuilding those test grids on CUDA
differed from CPU by 5.62e-7 of peak for Catmull-Rom density and 3.56e-6 for B-spline pixel.

## Local tests

- Strict pins passed all 55 tests, including the 13 complete zip hashes.
- All 20 stage 4 file-version tests passed across focused runs.
- Six existing regression tests passed across `test_integrated_spline.py`, `test_spline_tables.py`,
  `test_psfzip.py`, `test_composition.py` and `test_dipole_channels.py`. These cover version selection
  with sources, old JSON dispatch, invalid basis/arrays, angle defaults, bilinear tiles and mixed channels.
- Two tests from `web/test/test_psfzip.ts` passed under Node 22 with locally generated Python fixtures.
  These cover the 0.3 CPU renders and the refusal cases for versions, spline bases, coefficients and
  integration modes. Browser/WebGPU execution remains for the reviewer.

## Server verification

Run the strict pin file with `TORCH_PSF_PINS_STRICT=1` on the recording stack. For the full Python
suite run `python -m pytest tests`. The following files cover the stage 4 changes directly.

- `tests/test_unify_pins.py`, `tests/test_format_versions.py`, `tests/test_psfzip.py`, `tests/test_io.py`
- `tests/test_vectorial.py`, `tests/test_pixint.py`, `tests/test_spline.py`, `tests/test_spline_tables.py`, `tests/test_integrated_spline.py`
- `tests/test_fourpi.py`, `tests/test_fourpi_pixint.py`, `tests/test_dipole_channels.py`, `tests/test_field.py`, `tests/test_composition.py`, `tests/test_multichannel.py`
- `web/test/test_psfzip.ts` via `python web/test/run_tests.py --only psfzip`, followed by `tests/test_web_psfzip.py` and `tests/test_viewer.py`

Run the real m1809/m25 v4 read/build comparisons and the full-size NPC grid from design sections 5.6
and 6 on the server. This stage includes no benchmarks.

## Changed files

- Model and file code under `torch_psf/` includes `psf.py`, `vectorial.py`, `spline.py`, `fourpi.py`, `field.py`, `multichannel.py`, `psfzip.py` and `io.py`.
- Python tests under `tests/` include `test_format_versions.py` and `test_composition.py`.
- Web reader and tests in `web/src/io/psfzip.ts` and `web/test/test_psfzip.ts`.
- `README.md`, `docs/PSF_FORMAT.md` and this report.
