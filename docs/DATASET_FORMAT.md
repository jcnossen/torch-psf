# Dataset format

`torch_psf.Dataset` holds the localizations of one acquisition: a dictionary of fields (torch tensors, CPU by
default), each with its own dtype and shape and the same first dimension n, plus metadata and an optional drift.
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

A free dictionary of JSON-serializable values. Common keys: `pixel_nm` (camera pixel size in the sample, nm),
`image_size` [h, w] (px), `n_frames` (frames of the movie), `n_dims`,
`method`, `source`, `region` / `origin` (set by crop), `drift_info`.

## Drift

Optional array [F, D'] (px per movie frame, D' <= D). The drift of a localization is `drift[frame + n_frames // 2]`
(its centre frame); `apply_drift` subtracts it from `pos`, `undo_drift` adds it back.

## HDF5 container

| path | content |
|---|---|
| file attributes | `format` = 'torch-psf dataset', `format_version` = 1 |
| `/localizations/<field>` | one dataset per field, with its dtype and shape |
| `/metadata` | a group whose attributes are the metadata; strings and numbers as they are, other values (lists, dicts) as the string `'json:' + JSON` |
| `/drift` | optional dataset [F, D'], attributes `applied` (bool) and `info` (JSON) |

## Exports

`save_picasso(path)` writes a Picasso hdf5 ('locs' table: lpx / lpy = precision, sx / sy = PSF width) and its
.yaml; `load_picasso` reads one. `save_csv(path)` / `load_csv(path, pixel_nm)` use the ThunderSTORM conventions
(positions and uncertainties in nm, frames counted from 1, the centre of the first pixel at 0.5 px).
