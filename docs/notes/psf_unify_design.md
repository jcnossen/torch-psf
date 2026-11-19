# torch-psf: map of the PSF classes and a unified design (draft for discussion)

Base: master 6afbd1f, plus branch dipole-orientation (d092fcf, qnano1; not merged). Nothing implemented yet.

## 1. What there is today

| Class | File | Kind | type_name | What it is |
|---|---|---|---|---|
| PSF | psf.py:29 | base | – | render(pos, grad), fit/localize/crlb, save/load, registry |
| VectorialPSF | vectorial.py:168 | optical | vectorial | pupil model (Zernikes, Fresnel, SAF), dipole tensor, blur, pixel integration box / spectral / gauss_legendre, per-angle terms |
| ExactVectorialPSF | pixint.py:105 | optical (prototype) | – | wraps VectorialPSF with other pixel integration (spectral / gauss / box / point); copies of density, spectral, grid |
| GaussianPSF, AstigmaticGaussianPSF | gaussian.py | analytic | gaussian, astig_gaussian | erf-integrated Gaussian |
| FourPiPSF | fourpi.py:109 | optical, multichannel | fourpi | two VectorialPSF arms, coherent sum per channel, 4Pi phase; copies the pupil / blur / integration steps |
| CubicSplinePSF | spline.py:58 | table | spline | Catmull-Rom of pixel values at sub-pixel shifts, conv2d 4x4x4 |
| PixelSplinePSF | pixint.py:239 | table (prototype) | – | catmull / bspline kernel, integrate or not, sep / conv backend; second evaluator |
| IntegratedSplinePSF | pixint.py:373 | table | integrated_spline | PixelSplinePSF fixed at bspline + integrate + sep; per-angle coefs; what SIMCODE uses |
| DipoleSpline | _dipole.py:21 | mixin | – | 6 dipole component tables on a spline |
| FourPiSplinePSF | fourpi.py:598 | table, multichannel | fourpi_spline | 3 x C inner splines (inc, cr, ci), analytic phase |
| MultiChannelPSF | multichannel.py:55 | base | – | [N, C, P, P], channel offsets / ratios / mappings, multichannel fit |
| ChannelsPSF | multichannel.py:544 | composition | (zip channels) | list of independent channel models |
| FieldPSF | field.py:287 | composition | (zip field block) | VectorialPSF + FieldZernikes (exact per position) |
| TiledSplinePSF | field.py:416 | composition | (zip field block) | one spline per field tile, nearest tile |
| _Bound | field.py:217 | wrapper | – | binds per-emitter field xy for PSF.fit |
| branch: LayeredVectorialPSF | layered.py:30 | optical | vectorial_layered | emitter in a film; via VectorialPSF._emitter_pupil hook |
| branch: FocusZ, Polychromatic | orientation.py | wrappers (duck-typed) | – | stage z-mode; wavelength average |
| outside: sofi GridPSF | sofi/model.py | wrapper | – | needs VectorialPSF internals |

Main duplications (details with file:line in the survey):
- Two spline evaluators (CubicSplinePSF, PixelSplinePSF), three knot-assembly routines, the Catmull-Rom basis defined twice.
- The dipole-tensor contraction in ~5 places, the M <-> 6-vector convention 3 times.
- Pixel integration / spectral / band-limit / oversample checks in VectorialPSF, ExactVectorialPSF and FourPiPSF; FourPiPSF re-implements the pupil and image steps.
- The forward-mode jvp render loop written ~7 times (Vectorial, Exact, Field, FourPi, branch basis_images, FocusZ, Polychromatic).
- The same concept under different names: M / dipM, phi / phi_extra, xy / field_xy / corners, focus present or not; `resized`, `to`, `to_spline`, `dipole_basis` with different signatures.
- z-mode three ways (FourPiPSF.zmode, spline pos_fn, FocusZ). Format-version rules guessed by psfzip from keys.

## 2. The observation that unifies most of it

Every model here is a sum of fixed basis images with weights that depend on a few per-emitter "linear" parameters:

    image_c(x, y, z; extras) = sum_k  w_ck(extras) * B_ck(x, y, z)

- dipole tensor: k = 6 components, w = m6(M)            (exact: the image is linear in M)
- 4Pi phase: k = (inc, cr, ci), w = (1, cos Phi, -sin Phi)  (exact)
- per pattern angle: k = angle, w = one-hot(angle)
- field tiles with bilinear blending: k = 4 tiles, w = bilinear weights of xy
- isotropic single-channel PSF: k = 1, w = 1

and B depends only on (x, y, z) (and, for optical models, on the learnable optics). So one interface covers the
optical models and the tables: `basis(pos, ...) -> [N, C, K, P, P]` plus `weights(extras) -> [N, C, K]`, render = the
contraction, d/d extras = analytic through the weights, d/d (x, y, z) = the basis gradient.

## 3. Proposed layers

**A. Optical models (continuous, nn.Module, learnable, slow, differentiable in everything)**
- `PupilModel` = today's VectorialPSF split in stages (as the branch already starts: `_emitter_pupil`, `_fields`, `_image`):
  pupil -> image fields E [N, 6, n, n] -> dipole contraction -> one shared `PixelIntegrator`
  ('box' | 'spectral' | 'gauss_legendre' | 'point') with the blur inside it.
- Subclasses only replace a stage: LayeredVectorialPSF (`_emitter_pupil`), FourPiPSF (two arms + coherent combine,
  reusing the stages instead of copying them), Polychromatic (a wrapper or a stage).
- ExactVectorialPSF goes away (its modes become PixelIntegrator modes; 'point' added).
- One forward-mode-gradient helper used by all optical models.
- Gaussian models stay a small separate analytic family.

**B. Tables (one class)**
- `SplinePSF`: knots [C, K, nz, n, n], one evaluator (sep and conv backends), Catmull-Rom kernel by default (your
  rule; B-spline only where a benchmark shows it is needed, and kept readable for existing files).
- `value`: 'pixel' (Catmull-Rom of exact pixel values at sub-pixel shifts = today's CubicSplinePSF) or 'density'
  (point density, exact pixel integral of the spline = integrated Catmull-Rom). Same knot grid and evaluator.
- Built by one `to_spline(model, grid, value, components)` that samples the model's basis; replaces
  knots_from_model / knots_from_vectorial / point_knots / build_planes / FourPiPSF.to_spline.
- Replaces CubicSplinePSF, PixelSplinePSF, IntegratedSplinePSF, DipoleSpline and FourPiSplinePSF's inner luts.

**C. Compositions (wrappers that take any A or B model)**
- `Channels` (today's ChannelsPSF/MultiChannelPSF): offsets, ratios, mappings, multichannel fit. FourPi becomes "a
  coupled channel model" (its channels share components), not a special path in psfzip.
- `Field`: per-position model from a base + FieldZernikes, either exact (optical base) or tiled (table per tile,
  nearest or bilinear = 4 components with weights). FieldPSF and TiledSplinePSF become two modes of one wrapper.
- `ZMode` wrapper (emitter height vs stage focus) replacing FourPiPSF.zmode, spline pos_fn and FocusZ.
- `Bound` (fixed per-emitter xy) for the generic fitter.

**One call signature everywhere**
`render(pos, grad=False, *, dipM=None, phi=None, xy=None, focus=None, angle=None)` -> [N, P, P] or [N, C, P, P];
a model that does not support an argument raises clearly. `basis(...)` / `weights(...)` as above;
`resized(roisize)`, `to(device, dtype)`, `to_spline(**grid)` with one signature. Renames without aliases
(M -> dipM, phi_extra -> phi, field_xy -> xy).

**Files**
Each class declares the minimum format version its features need (instead of psfzip inferring it from keys), and
the readers keep reading every existing block (spline, integrated_spline incl. B-spline coefficients, fourpi_spline,
phase_spline, field) so old files and SIMCODE's models keep working.

## 4. Different demands, and how each is served
- In-situ / bead PSF estimation: optical models (A), differentiable in the optics; tables only as a fast inner loop
  with a vjp back to the optics (today's vjp_vectorial).
- SIMCODE / fast localization: tables (B), analytic gradients, conv or sep backend.
- Orientation (branch): basis / weights of A or B; the fitter stays in orientation.py.
- 4Pi: A (two arms) and B (3 phase components x C channels), same interface.
- Field dependence: C over A (exact) or B (tiles).
- SOFI GridPSF, web viewer, psfzip: consumers of the uniform interface and of the file blocks.

## 5. Decisions (user, 2026-11-19)
1. Kernels: both Catmull-Rom and cubic B-spline stay supported (B-spline files must stay readable; SIMCODE uses them).
   Plain Catmull-Rom of pixel values (old CubicSplinePSF) is NOT the default for new tables.
2. Default for new tables: integrated ('density') Catmull-Rom at 2 knots/px, UNLESS the integrated B-spline at
   2 knots/px is faster, then that (it is already the more accurate one at S = 2: 2.5e-4 vs 2.1e-3 of the peak on
   m1809). Decided by the stage-0 timing report.
3. API: renames and removals without aliases are OK on this branch; every existing file must stay readable.
   SIMCODE is updated on its own branch afterwards; both are merged into master together.
4. Wrappers by indexing, not copying:
   - tables: all tables of a composite model in ONE knot tensor [T, nz, nv, nu] (T = channels x components x tiles x
     angles x wavelengths, flattened) plus a small integer map (channel, component, tile, angle) -> table; the
     evaluator's per-spot knot-block gather gets a table offset (table * nz nv nu); one batched gather + conv / einsum
     for all (spot, table) pairs, the weights applied by an einsum over the components; backward flows into the one
     knot tensor. Channels, 4Pi phase (1, cos, -sin), dipole (m6(M)), bilinear tiles (4 tiles + weights) and angles
     are all such components.
   - optical models: wrappers hold references to the same modules (shared parameters): field = base + per-spot
     Zernike offsets; channels that share optics = one batched call with per-spot focus / offset; 4Pi = two arm
     modules, coherent combination.
5. Branch dipole-orientation (torch-psf-sofi) is merged into psf-unify first. Its experimental parts: FocusZ (to be
   replaced by the z-mode wrapper), Polychromatic (a special case: stays a standalone wrapper in orientation.py and
   does not shape the class design; adapted only to the new signatures), LayeredVectorialPSF (experimental). orientation.py stays its own module (fitter, estimator); the refactor adapts it
   to the new interface and keeps its tests and the (xx, yy, zz, xy, xz, yz) / factor-2 basis convention.
6. Order: stage 0 benchmark report + output-pinning tests; 1 optical stages + one PixelIntegrator; 2 one SplinePSF
   with the table-indexed evaluator; 3 compositions + one render signature; 4 per-class format versions; 5 benchmark
   report again. Each a Codex task (gpt-6.2-astra, high reasoning effort), reviewed and tested before the next.
