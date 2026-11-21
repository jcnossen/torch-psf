# Dataset format

`torch_psf.Dataset` holds the localizations of one acquisition: a dictionary of fields (torch tensors, CPU by
default), each with its own dtype and shape and the same first dimension n, plus metadata.
This document specifies the fields and the HDF5 file written by `Dataset.save('name.hdf5')` and read by
`Dataset.load`. Every program that writes the format with all fields can be read by every other one.

A selection is not stored: it is computed when needed (filters). Other estimates of the same localizations (for
example a detector position, or an earlier pass) are separate datasets linked by `id`, not extra fields.

## Fields

Core fields (always present after construction; `id` and `n_frames` get defaults):

| field | dtype | shape | unit | meaning |
|---|---|---|---|---|
| `id` | int64 | [n] | | localization id; the same id in related datasets links the rows (default 0 .. n-1) |
| `frame` | int32 | [n] | frames | first frame of the localization |
| `n_frames` | int32 | [n] | frames | frames it covers: 1 for single-frame fits, the window length of multi-frame methods, the on-time of linked blinks (default 1). Last frame = frame + n_frames - 1 |
| `pos` | float64 | [n, D] | px | x, y (and z): pixel centres at integer coordinates, x = column, y = row; z relative to the focal plane of the PSF, in px of the lateral pixel size (`metadata.pixel_nm`): one unit for all axes |
| `crlb` | float32 | [n, D] | px | precision (one standard deviation) per axis: a Cramér-Rao bound or a calibrated uncertainty |
| `photons` | float32 | [n] | photons | total signal of the localization |
| `bg` | float32 | [n] | photons / px / frame | background |

Standard optional fields:

| field | dtype | shape | meaning |
|---|---|---|---|
| `chi2` | float32 | [n] | goodness of fit (mean squared normalized residual per pixel) |
| `score` | float32 | [n] | detection score (method-specific scale) |
| `group` | int64 | [n] | localizations of one emitter / binding event linked across frames |
| `cov` | float32 | [n, 3] | 3D only: covariances (xy, xz, yz) of the position estimate, px^2 |
| `loglik`, `crlb_photons`, `crlb_bg`, `at_limit` | | [n] | written by `PSF.fit` / `PSF.localize` |

Any other field (any dtype, shape [n, ...], e.g. per-frame intensities [n, frames, channels]) can be added with
`add_fields`; all fields are carried through indexing, filters, crop, concatenate, save and load.

## Metadata

A free dictionary of JSON-serializable values. Common keys: `pixel_nm` (camera pixel size in the sample, nm; for non-square pixels the x size, also the
unit of z), `pixel_nm_y` (only for non-square pixels: the y size, nm),
`image_size` [h, w] (px), `n_frames` (frames of the movie), `n_dims`,
`method`, `source`, `region` / `origin` (set by crop), `drift` (the estimation record for the correction applied to these positions).

## Drift

A `tp.Drift` holds a float64 curve [F, D'] in px per movie frame and its estimation record.
For DME v2, the record includes the method, entry point, package path and version, parameters, DME
diagnostics, and per-axis peak-to-peak and RMS displacement in pixels.
`drift = ds.estimate_drift()` estimates it with DME v2. `corrected = ds.drift_corrected(drift)`
subtracts the curve from the first D' position axes at each centre frame (`frame + n_frames // 2`,
clipped to the curve's frame range). Keep `ds` for the raw positions. The corrected dataset carries
`meta['drift'] = drift.info`. Correcting it again raises unless you pass `force=True`, which subtracts
again and replaces the record. Concatenation requires all inputs to have matching correction records
or all to lack a record.

Use `drift.save(path)` and `tp.Drift.load(path)` for a standalone NPZ archive containing `curve` and
JSON-encoded `info`.

## HDF5 container

| path | content |
|---|---|
| file attributes | `format` = 'torch-psf dataset', `format_version` = 2 |
| `/localizations/<field>` | one dataset per field, with its dtype and shape |
| `/metadata` | a group whose attributes are the metadata; strings and numbers as they are, other values (lists, dicts) as the string `'json:' + JSON` |

Version 1 files may also contain a `/drift` dataset. The reader preserves the stored positions.
If `applied=True`, its JSON `info` becomes the correction record. If `applied=False`, the reader
warns and omits the record. In either case, `tp.Drift.load(path)` retrieves the curve and info.

## Exports

`save_picasso(path)` writes a Picasso hdf5 ('locs' table: lpx / lpy = precision, sx / sy = PSF width; Picasso units: x, y, lpx, lpy in px, z and lpz in nm, so 3D needs pixel_nm) and its
.yaml; `load_picasso` reads one. `save_csv(path)` / `load_csv(path, pixel_nm)` use the ThunderSTORM conventions
(positions and uncertainties in nm, frames counted from 1, the centre of the first pixel at 0.5 px). `save_smlm(path)` /
`load_smlm(path, pixel_nm)` read and write ShareLoc.XYZ .smlm files (a zip of manifest.json and one binary table, format
0.2) with the same conventions; tables are float32 by default (`dtype='float64'` for a lossless round trip), and the
manifest's `torch_psf` block records the pixel size, conventions, frame count and image size, so a file written here
loads without `pixel_nm`.
