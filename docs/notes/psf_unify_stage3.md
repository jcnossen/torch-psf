# PSF unification, stage 3

The baseline is `7ffb5e7` on `psf-unify`. This stage changes the Python interface for rendering, optical
composition and spline indexing. The numerical comparison used CPU float32 with one PyTorch thread in the
`smlm` environment. It compares the arrays before the seven-significant-digit rounding used in the JSON pins.

## SIMCODE migration

| Previous call or name | Stage 3 |
|---|---|
| Model-specific render and dipole basis arguments | `render(pos, grad=False, *, dipM=None, phi=None, xy=None, focus=None, angle=None)`. `dipole_basis` has the same signature. |
| `FourPiPSF.forward(..., M=..., phi_extra=...)` | `forward(..., dipM=..., phi=...)`. Use the same keywords with `render`. |
| `FourPiPSF(..., zmode='stage')` or `from_state(..., zmode=...)` | Wrap the optical model in `ZMode(model, 'stage')`. `from_state(state)` interprets the stored axial mode automatically. The arm depths still belong to `FourPiPSF`. |
| `torch_psf.orientation.FocusZ(model, height=h)` | `torch_psf.ZMode(model, 'stage', height=h)`. |
| `emitter_height_pos_fn(f)`, `stage_scan_pos_fn(f, s, emitter_z=h)` | Tabulate `ZMode(model, 'emitter', focus=f)` or `ZMode(model, 'stage', focus=f, scale=s, height=h)`. The low-level knot VJP accepts the wrapper's `coordinates` method. |
| `torch_psf.orientation.Polychromatic` | `torch_psf.Polychromatic(model, wavelengths_um, weights)`. Tabulation stores the weighted response once per output component. |
| `field_xy=` | `xy=` in camera pixels. For a mapped channel stack supply ROI centres as `xy[N,C,2]` in render. Fit continues to take ROI corners. |
| `render(..., offsets=..., corners=...)` | `render_geometry(..., offsets=..., corners=...)` for explicit acquisition geometry. Fixed model offsets apply automatically in `render`. |
| Spline `render(..., knots=..., weights=...)` | `render_tables(..., knots=..., weights=...)` for explicit coefficient adjoints and custom table weights. |
| Optical `resized(P, oversample=..., pixel_integration=...)` | `resampled(P, oversample=..., pixel_integration=...)`. All models expose `resized(roisize)`, `to(device=None, dtype=None)` and `to_spline(roisize=None, z_range=None, **grid)`. |
| Orientation fitter with externally evaluated field Zernikes | Compose `FieldPSF` and pass `xy=` to the fitter. The fitter also accepts measured per-spot `zcoef_offset` for studies supplying coefficients directly. |
| Private `field._Bound` | `field_model.bind(xy)` returns a `Bound` model. Tiled fits use the same fixed coordinates and batching as exact field fits. |

The old keyword aliases and orientation wrapper imports are removed. The stage-2 SIMCODE calls for
`to_spline(per_angle_models, kernel='bspline', value='density', **grid)`, `PSF.load(path, angle=k)`,
`convert_calibration` and `save_zip` retain their meaning.

## Accuracy and reference checks

All 55 strict pin tests passed. Of 205 unrounded arrays, 204 were byte-identical. The remaining difference
was in a grouped convolution over two nearest-tile images. The table below gives the result for each pin.
`Exact` includes the forward image, the value returned with gradients and every derivative present for that
case. The orientation row covers images, Jacobians and fitted parameters. The SOFI row covers images and
Jacobians.

| Pin | Compared with 7ffb5e7 |
|---|---|
| `box` | Exact |
| `box_dipole` | Exact |
| `box_blur` | Exact |
| `box_blur_dipole` | Exact |
| `box_angle` | Exact |
| `spectral` | Exact |
| `spectral_dipole` | Exact |
| `spectral_blur` | Exact |
| `spectral_blur_dipole` | Exact |
| `spectral_angle` | Exact |
| `spectral_angle_mean` | Exact |
| `gauss_legendre` | Exact |
| `gauss_legendre_dipole` | Exact |
| `gauss_legendre_angle` | Exact |
| `exact_prototype` | Exact |
| `focus_z` | Exact |
| `polychromatic` | Exact |
| `gauss2d` | Exact |
| `gauss3d` | Exact |
| `cubic` | Exact |
| `cubic_dipole` | Exact |
| `integrated` | Exact |
| `integrated_angle` | Exact |
| `integrated_angle_mean` | Exact |
| `integrated_dipole` | Exact |
| `integrated_angle_dipole` | Exact |
| `pixel_catmull` | Exact |
| `pixel_catmull_conv` | Exact |
| `pixel_bspline` | Exact |
| `pixel_bspline_conv` | Exact |
| `fourpi_box` | Exact |
| `phase_cubic` | Exact |
| `fourpi_spectral` | Exact |
| `phase_integrated` | Exact |
| `phase_dipole` | Exact |
| `channels` | Exact |
| `field` | Exact |
| `field_bound` | Exact |
| `tiles` | `forward` differs by 1.3220414e-7 of peak. Gradient-path value and all derivatives exact. |
| `orientation` | Exact |
| `sofi` | Exact |

All 17 orientation tests passed across short runs, including the composed-Vortex image, dipole basis,
noise-free fit and seeded-noise fit checks against the original reference. The reference data and tilt
convention are unchanged.

The small field comparison used 10 px tiles. At the four-tile corner `xy=(10,10)`, RMS error relative to
the exact image peak was 0.06589 for nearest and 0.00631 for bilinear. At `(12,9)` the respective errors
were 0.00123 and 0.00545. The accuracy assertion is specific to the corner. The tests also check centres,
edge clamping, continuity, derivatives at fixed xy, dipole tensors, indexed channel calls and file reading.
All 12 composition tests passed, including batched ROI fits compared with individually bound field positions.

## Archive bytes

All 13 archive manifests matched. Nine complete archives matched their original SHA-256 hashes on the pinned
writer stack. These were `box`, `gauss2d`, `gauss3d`, `fourpi_box`, `phase_cubic`, `phase_integrated`,
`phase_dipole`, `channels` and `field`.

For `cubic`, `integrated`, `integrated_angle_dipole` and `tiles`, the only changed member was
`fit_report.pdf`. It now prints the actual `SplinePSF` repr. Every corresponding JSON, coefficient, TIFF
and PNG member matched byte for byte. `tests/test_unify_pins.py` explicitly pins the four new archive hashes.
The original pin JSON is unchanged. For nearest tiles the on-disk metadata is unchanged. A bilinear file
adds `blend="bilinear"` to its field tile metadata and requests format 0.4.0.

## Server verification

Run the full suite with `python -m pytest tests` in the reviewer environment. For a focused server run use
the following files under `tests/`.

- Composition and references — `test_unify_pins.py`, `test_composition.py`, `test_orientation.py`, `test_dipole_channels.py`.
- Optics, tables and channels — `test_vectorial.py`, `test_pixint.py`, `test_spline.py`, `test_spline_tables.py`, `test_integrated_spline.py`, `test_fourpi.py`, `test_fourpi_pixint.py`, `test_multichannel.py`, `test_mapping.py`, `test_field.py`.
- Estimation and fitting — `test_fit.py`, `test_insitu.py`, `test_dense_beads.py`, `test_phase_diversity.py`, `test_multichannel_beads.py`, `test_sofi.py`, `test_pipeline.py`, `test_downstream_api.py`.
- File and viewer interoperability — `test_psfzip.py`, `test_io.py`, `test_viewer.py`, `test_web_psfzip.py`.

Run the SIMCODE checks in design sections 5.6 and 6 with the private m1809 / m25 calibrations and
NPC_f_dxy15 state. The real v4 calibration pair and `check_v4.py` were absent from the local worktree and
the searched `C:/dev` and `C:/data` directories. The synthetic per-angle pins passed. The full-size spline
rebuilds and GPU suite remain for the server. No benchmarks were run.

## Changed files

- `docs/` — `API.md`, `DESIGN.md`, `PSF_FORMAT.md`.
- `docs/notes/` — `field_psf_lib.md`, `orientation_psf.md`, `psf_unify_stage3.md`.
- `examples/` — `quickstart.py`.
- `study/insitu/` — `bench_conventional.py`, `bench_split.py`, `bench_variants.py`.
- `study/orientation/` — `composed_reference.py`, `vortex_baseline.py`, `vortex_crosscheck.py`, `vortex_exact.py`, `vortex_real.py`.
- `study/psf_unify/` — `bench.py`.
- `study/realdata/` — `beads_fit.py`, `insitu_fit.py`.
- `tests/` — `make_unify_pins.py`, `make_web_reference.py`, `make_web_smlm_reference.py`, `test_composition.py`, `test_dipole_channels.py`, `test_fourpi.py`, `test_fourpi_pixint.py`, `test_mapping.py`, `test_multichannel.py`, `test_multichannel_beads.py`, `test_orientation.py`, `test_pixint.py`, `test_spline.py`, `test_spline_tables.py`, `test_unify_pins.py`.
- `torch_psf/` — `__init__.py`, `composition.py`, `estimators.py`, `field.py`, `fit.py`, `fourpi.py`, `gaussian.py`, `integrate.py`, `multichannel.py`, `orientation.py`, `pixint.py`, `psf.py`, `psfreport.py`, `psfzip.py`, `simulate.py`, `spline.py`, `vectorial.py`, `vortex.py`.
- `torch_psf/estimation/` — `dense_beads.py`, `field_knots.py`, `heldout.py`, `knots.py`, `phase_diversity.py`, `tracks.py`.
- `torch_psf/sofi/` — `model.py`.
