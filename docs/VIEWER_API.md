# PSF viewer API (`python -m torch_psf view`)

`torch_psf/viewer.py` runs a standard-library HTTP server on `http://127.0.0.1:<port>/` (default port 8000). It
serves the bundled front end (`torch_psf/viewer_static/`, `viewer.html` at `/`) and the API below from the same
origin, so no CORS is needed. Without a bundled front end, `/` returns a short HTML page that explains how to build it
(`scripts/build_viewer.sh`); the API works either way.

`api_version`: **1**. Every response carries it: in the JSON body, or in the JSON header of a binary response. The
page should check it and refuse to work with a different major version.

## Conventions

- Units: lengths in um (`pixel_um`, `wavelength_um`, `focus_um`, `z_um`, `z_range`, `dx_um`, `dz_um`, `z0_um`), lateral
  positions of the PSF images in px, aberrations in waves rms (Noll-indexed Zernikes), intensities in photons.
- z = emitter height above the coverslip (um); `focus_um` = focal plane relative to the coverslip.
- Image axes: x = column (fastest index), y = row. Volumes are stored z, y, x (x fastest).

## Parameters (the request body of /api/volume and /api/render)

A JSON object. Missing fields take the server's current defaults (`GET /api/default`).

| field | type | unit | meaning |
|---|---|---|---|
| `model` | `"vectorial"` \| `"spline"` \| `"gaussian"` | | `spline` = SplinePSF built from the vectorial parameters (oversample 2, 20 nm), or the spline loaded with /api/load |
| `optics` | object | | `pixel_um`, `NA`, `wavelength_um`, `n_imm`, `n_med` (partial objects are merged with the defaults) |
| `focus_um` | number | um | focal plane position |
| `z_um` | number | um | emitter height of the pixel images (/api/render) |
| `z_range` | [lo, hi] | um | z range of the volume and of the CRLB curve |
| `zernike` | object | waves rms | `{"<noll>": value}`, Noll 5 ... 22; absent = 0 |
| `roisize` | odd integer 5 ... 81 | px | size of the pixel images; the volume covers roisize + 2 px |
| `photons`, `bg` | number | photons, photons / px | for the CRLB curve |
| `sigma_px` | number | px | width of the Gaussian model |
| `levels` | [number] | fraction of the peak | isosurface levels (default [0.5, 0.1, 0.01]) |
| `meshes` | bool | | compute isosurface meshes on the server (needs scikit-image; else `meshes` is empty and `mesh_note` says why) |

## Endpoints

### `GET /api/default` -> JSON
The current parameters (the fields above, plus `file`: the name of a PSF file given on the command line or loaded,
`presets`: names for /api/preset), and `api_version`.

### `POST /api/load` `{ "path": "<file on the server>" }` -> JSON
Reads a PSF file (torch-psf json, or a plain calibration json) and returns the parameters to show it: for a
vectorial model `model`, `optics`, `focus_um`, `zernike`, `z_range`; for a spline `model: "spline"`, `z_range`, `z_um`,
`roisize` (the server keeps the knots; later `model: "spline"` requests use them); for a Gaussian model `model`,
`sigma_px`, `roisize`. Plus `file` and `api_version`. Errors: 404 `not_found`, 422 `unreadable_psf`.

### `POST /api/preset` `{ "name": "astigmatic" | "tetrapod" }` -> JSON
The parameters of a preset (and forgets a loaded spline). `tetrapod`: Z6 0.2, Z12 -0.2 waves, focus 1 um, z 0 ... 2 um,
roisize 25.

### `POST /api/volume` `{parameters}` -> binary
Header fields: `kind: "volume"`, `model`, `shape: [nz, ny, nx]`, `axis_order: "z,y,x"`, `dx_um` (xy voxel size = pixel
size / 2), `dz_um`, `z0_um` (z of plane 0), `origin` (x = y = 0 at the volume centre: voxel (i, j) of a plane is at
x = (i - (nx - 1) / 2) dx_um, y = (j - (ny - 1) / 2) dx_um), `vmax` (volume maximum), `levels` (the levels that have a
mesh), and `meshes`: a list of `{level, n_vertices, n_faces, vertices, normals, faces}`, where the last three are the
names of arrays.
Arrays, in this order:
1. `volume` float32 [nz, ny, nx]: the normalized PSF (pixel-integrated, per px^2 of the camera pixel), sampled every
   dx_um laterally (2x oversampled) and dz_um in z.
2. For every mesh i, from the highest level down: `mesh<i>_vertices` float32 [n_vertices, 3] (x, y, z in um, same
   frame as above, z absolute), `mesh<i>_normals` float32 [n_vertices, 3] (unit normals x, y, z, from the
   volume gradient in physical units), `mesh<i>_faces` uint32 [n_faces, 3] (vertex indices, triangles).

### `POST /api/render` `{parameters}` -> binary
Header fields: `kind: "render"`, `model`, `z_um`, `roisize`, `pixel_um`, `sum` (PSF sum over the ROI), `peak`,
`photons`, `bg`.
Arrays, in this order: `psf`, `d_dx`, `d_dy`, `d_dz`: float32 [roisize, roisize] (axes y, x): the normalized PSF of an
emitter at x = y = 0 (ROI centre), z = z_um, and its derivatives per px (x, y) and per um (z); then `crlb_z_um`,
`crlb_x_nm`, `crlb_y_nm`, `crlb_z_nm`: float32 [41]: the Cramer-Rao lower bound (std, nm) of x, y, z at 41 heights
evenly over z_range, for `photons` and `bg` (Poisson noise). For the 2D Gaussian model `crlb_z_nm` is 0.

## Binary response format (`Content-Type: application/octet-stream`)

```
offset 0      uint32 little endian: N = length of the JSON header in bytes
offset 4      N bytes UTF-8 JSON header, padded with spaces so that 4 + N is a multiple of 4
offset 4 + N  the arrays, back to back, little endian, C order (last axis fastest)
```
The header holds `api_version`, the endpoint-specific fields above, and `arrays`: a list (in storage order) of
`{name, dtype ("float32" | "uint32"), shape, axes, offset, nbytes}`, where `offset` counts bytes from the start of
the array section (4 + N) and is a multiple of 4, so `new Float32Array(buffer, 4 + N + offset, nbytes / 4)` views an
array without a copy. Reference implementation: `torch_psf.viewer.pack` / `unpack`.

## Errors

HTTP status 4xx / 5xx with `Content-Type: application/json` and body `{ "error": "<code>", "detail": "<text>",
"api_version": 1 }`. Codes: `bad_request` (400: invalid JSON or parameters, e.g. an even roisize), `not_found` (404:
file or preset), `unknown_endpoint` (404), `unreadable_psf` (422), `internal` (500: the exception text in `detail`).

## Fixtures for offline front-end tests

`python -m torch_psf.viewer --dump-fixtures DIR` writes one response of every endpoint: `default.json`,
`request_tetrapod.json` (the parameters used), `tetrapod_psf.json` (a PSF file) and `load_tetrapod.json` (its
/api/load response), `volume_tetrapod.bin`, `render_tetrapod.bin`, `volume_astigmatic.bin`, `error_example.json`.
These are generated, not tracked.

## Packaging

`scripts/build_viewer.sh` runs `npm --prefix web run build:viewer` (the viewer entry of the web/ project, output
`web/dist-viewer/`, base `./`) and copies the result to `torch_psf/viewer_static/` (git-ignored, included as package
data by pyproject.toml). Run it before building a wheel.
