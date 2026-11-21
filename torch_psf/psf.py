"""PSF: the base class of every PSF model.

A model supplies render(pos, grad) (the normalized PSF on a roisize x roisize ROI, and its x / y / z derivatives) and
its geometry (roisize, z_range, xy_limit); everything else lives here, once: the expected image, Fisher information
and Cramer-Rao bound, the batched Poisson maximum-likelihood fit of ROIs, localization of a whole movie, files, and
the conversion to a cubic-spline model.

Parameter vector of an emitter: theta = (x px, y px, z um, photons, bg photons / px), x, y relative to the ROI centre
(pixel centres at integers, x = column, y = row). 2D models (z_range None) keep z fixed (any value).

Files: serializable models register their psf.json type names with PSF. PSF.load dispatches through this registry. Importing torch_psf (also implicitly, e.g. `import torch_psf.psf`, which runs the package __init__ first)
registers all of them.
"""
import json
import os
import numpy as np
import torch
from torch_psf.camera import Camera
from torch_psf.fit import fisher, crlb, localize, at_limit
from torch_psf.movie import open_movie
from torch_psf.spots import Spots, extract_spots
from torch_psf.dataset import Dataset, _plain
from torch_psf.optics import pixel_meta, pixel_size, pixel_x

FORMAT = 'torch-psf'
READ_VERSIONS = ((0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7))
PIXEL_PAIR_VERSION = (0, 6)     # non-square pixels: pixel_um stored as [x, y]


def pixel_version(version, *pixel_sizes):
    """version, raised to PIXEL_PAIR_VERSION when one of the pixel sizes is a non-square pair."""
    return max(version, PIXEL_PAIR_VERSION) if any(isinstance(p, (tuple, list)) for p in pixel_sizes) else version


def version_string(version):
    major, minor = version
    return f'{major}.{minor}.0'


def check_version(version, supported, container):
    parts = str(version).split('.')
    if len(parts) in (2, 3) and all(p.isascii() and p.isdecimal() for p in parts):
        if (int(parts[0]), int(parts[1])) in supported:
            return
    versions = ', '.join(f'{major}.{minor}.x' for major, minor in supported)
    raise ValueError(f'{container} version {version}: this reader supports {versions}')


def render_with_grad(fn, pos, grad=False, **extra):
    """Render and optionally return JVPs in x, y, z and each named extra (in order).
    For channel positions [N, C, 3], a position tangent shifts all channels together.
    """
    if not grad:
        return fn(pos, **extra)

    def evaluate(p, *values):
        return fn(p, **dict(zip(extra, values)))

    # Forward AD cannot attach tangents to expanded (overlapping) storage, e.g. a scalar 4Pi phase.
    inputs = tuple(arg.contiguous() for arg in (pos, *extra.values()))
    derivatives = []
    for axis in range(3 + len(extra)):
        tangents = tuple(torch.zeros_like(arg) for arg in inputs)
        if axis < 3:
            tangents[0][..., axis] = 1
        else:
            tangents[axis - 2].fill_(1)
        value, derivative = torch.func.jvp(evaluate, inputs, tangents)
        derivatives.append(derivative)
    return (value, *derivatives)


def render_options(dipM, phi, xy, focus, angle):
    return {k: v for k, v in dict(dipM=dipM, phi=phi, xy=xy, focus=focus, angle=angle).items() if v is not None}


class PSF:
    """Abstract PSF model. Subclasses set self.roisize (odd, px), self.z_range ((lo, hi) um, or None for 2D),
    self.dev (torch.device) and implement render(pos, grad=False); they may override xy_limit, z_limits, resized and
    to."""
    roisize: int
    type_name = None         # psf.json `type` of the subclass
    _registry = {}           # type_name -> class
    z_range = None
    provenance = None        # dict written to / read from the PSF file (estimators put their report here)
    pixel_um = None          # lateral pixel size (um); needed to express z in px when fitting a 3D model
    dev = torch.device('cpu')

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get('type_name')
        if isinstance(name, str):
            PSF._registry[name] = cls

    # ------------------------------------------------------------------ the model
    def render(self, pos, grad=False, *, dipM=None, phi=None, xy=None, focus=None, angle=None):
        """Normalized images and optional position derivatives. xy is fixed during a fit."""
        return self._render(pos, grad, **render_options(dipM, phi, xy, focus, angle))

    def dipole_basis(self, pos, grad=False, *, dipM=None, phi=None, xy=None, focus=None, angle=None):
        """Six response images in xx, yy, zz, xy, xz, yz order."""
        return self._dipole_basis(pos, grad, **render_options(dipM, phi, xy, focus, angle))

    def _dipole_basis(self, pos, grad):
        raise ValueError(f'{type(self).__name__} has no dipole response')

    def basis(self, pos, grad=False, **options):
        """Linear response images [N,C,K,P,P]."""
        if options.pop('dipM', None) is not None:
            images = self.dipole_basis(pos, grad, **options)
            def channel(v):
                return v[:, None] if v.ndim == 4 else v
            return tuple(channel(v) for v in images) if grad else channel(images)
        images = self.render(pos, grad, **options)
        def axes(v):
            return v[:, None, None] if v.ndim == 3 else v[:, :, None]
        return tuple(axes(v) for v in images) if grad else axes(images)

    def weights(self, pos, *, dipM=None, **options):
        channels = getattr(self, 'n_channels', 1)
        if dipM is not None:
            from torch_psf._dipole import M_to_m6
            return M_to_m6(torch.as_tensor(dipM).to(pos)).reshape(-1, 1, 6).expand(len(pos), channels, 6)
        return pos.new_ones(len(pos), channels, 1)

    def _render(self, pos, grad=False):
        """pos [N, 3] = (x px, y px, z um) relative to the ROI centre -> normalized PSF [N, P, P] (sums to ~1 over the
        image plane for an in-focus emitter); grad=True: (psf, d/dx per px, d/dy per px, d/dz per um)."""
        raise NotImplementedError

    @property
    def n_dims(self):
        """3 if the model has a z axis (z_range set), else 2."""
        return 2 if self.z_range is None else 3

    @property
    def xy_limit(self):
        """Largest |x|, |y| (px) from the ROI centre that the fitter allows."""
        return self.roisize / 2 - 1.0

    @property
    def z_limits(self):
        """z interval (um) the fitter clamps to (default z_range)."""
        return self.z_range

    @property
    def device(self):
        return self.dev

    def to(self, device=None, dtype=None):
        """Move the model to another device in place and return self (torch.nn.Module semantics; a copy of another
        ROI size is resized())."""
        if isinstance(self, torch.nn.Module):
            if dtype is None:
                return torch.nn.Module.to(self, device)
            return torch.nn.Module.to(self, device, dtype)
        raise NotImplementedError

    def resized(self, roisize):
        """The same model on another ROI size (needed by to_spline)."""
        raise NotImplementedError

    def __call__(self, pos, intensity=None, bg=None):
        """intensity * render(pos) + bg (intensity, bg: [N] or None)."""
        out = self.render(pos)
        if intensity is not None:
            out = out * intensity[:, None, None]
        if bg is not None:
            out = out + bg[:, None, None]
        return out

    # ------------------------------------------------------------------ statistics
    def expected(self, theta):
        """Expected image mu = photons * render(x, y, z) + bg, [N, P, P], for theta [N, 5]."""
        theta = torch.as_tensor(theta, dtype=torch.float32, device=self.dev)
        return theta[:, 3, None, None] * self.render(theta[:, :3]) + theta[:, 4, None, None]

    def fisher(self, theta, var=None):
        """Poisson Fisher information [N, p, p] at theta [N, 5] (p = 5, or 4 without z for 2D models: x, y, photons,
        bg). var: sCMOS read-noise variance (e^2; scalar or [N, P, P])."""
        params = (0, 1, 2, 3, 4) if self.n_dims == 3 else (0, 1, 3, 4)
        return fisher(self, torch.as_tensor(theta, dtype=torch.float32, device=self.dev), var, None, params)

    def crlb(self, theta, var=None):
        """Cramer-Rao lower bound (standard deviation) [N, 5] of (x px, y px, z um, photons, bg) at theta [N, 5]; the z
        column is NaN for 2D models."""
        return crlb(self, torch.as_tensor(theta, dtype=torch.float32, device=self.dev), var)

    # ------------------------------------------------------------------ fitting
    def fit(self, rois, camera=None, theta0=None, corners=None, frames=None, var=None, iters=30, zstarts=5,
            batch=4000, pixel_um=None):
        """Maximum-likelihood fit of ROIs -> Dataset (torch_psf.dataset).

        rois: [N, P, P] in photons, or Spots (torch_psf.spots; then corners, frames and the read-noise variance come
        with it). camera: Camera (only its read noise is used here: ROIs are already in photons; with per-pixel maps
        the corners are needed). theta0: [N, 5] starting values (default: multi-start in z, see
        torch_psf.fit.localize). corners [N, 2] = (x0, y0) of the ROIs in the image: positions are then image
        coordinates (else relative to the ROI centre). frames [N]: frame numbers.
        Dataset fields: id, frame, n_frames (1), pos [N, D] px (z converted from um to px with the lateral pixel size:
        pixel_um, default the model's pixel_um; a 3D model without a pixel size raises ValueError),
        crlb [N, D] (same units), photons, bg, crlb_photons, crlb_bg, loglik (= -NLL, Poisson, constant dropped),
        chi2 (mean of (d - mu)^2 / (mu + var) per pixel), at_limit (x or y at the offset limit)."""
        if isinstance(rois, Spots):
            corners = rois.corner if corners is None else corners
            frames = rois.frame if frames is None else frames
            var = rois.var if var is None else var
            rois = rois.rois
        rois = torch.as_tensor(rois, dtype=torch.float32, device=self.dev)
        N, P = len(rois), rois.shape[-1]
        if var is None and camera is not None:
            var = camera.variance(rois, None if corners is None else torch.as_tensor(corners, device=self.dev))
        if torch.is_tensor(var) and var.dim() == 0:
            var = None if float(var) == 0 else var
        h = (P - 1) / 2
        out = {k: [] for k in ('theta', 'crlb', 'nll', 'chi2', 'lim', 'xy')}
        for s in range(0, N, batch):
            d = rois[s:s + batch]
            v = var[s:s + batch] if (torch.is_tensor(var) and var.dim() > 0) else var
            t0 = None if theta0 is None else torch.as_tensor(theta0[s:s + batch], dtype=torch.float32, device=self.dev)
            th, nll = localize(self, d, t0, var=v, iters=iters, zstarts=zstarts)
            mu = self.expected(th)
            vv = 0.0 if v is None else v
            out['chi2'].append(((d - mu) ** 2 / (mu + vv).clamp_min(1e-6)).flatten(1).mean(1))
            xy = th[:, :2].double()
            if corners is not None:
                xy = xy + torch.as_tensor(corners[s:s + batch], device=self.dev).double() + h
            out['theta'].append(th); out['crlb'].append(crlb(self, th, var=v)); out['nll'].append(nll)
            out['lim'].append(at_limit(th, self)); out['xy'].append(xy)
        cat = lambda k, shape: torch.cat(out[k]).cpu() if out[k] else torch.zeros(shape)
        px = pixel_size(pixel_um if pixel_um is not None else getattr(self, 'pixel_um', None), floats=True)
        if self.n_dims == 3 and not px:
            raise ValueError(f'{type(self).__name__} has no pixel size: z is stored in px of the lateral pixel size, so '
                             f'pass pixel_um=... or set psf.pixel_um')
        th, cr = cat('theta', (0, 5)), cat('crlb', (0, 5))
        meta = dict(method=f'torch_psf {type(self).__name__}.fit', n_dims=self.n_dims)
        if out['chi2']:
            meta['chi2_median'] = float(cat('chi2', (0,)).median())      # ~1 if the camera gain / noise model is right
        if px:
            meta.update(pixel_meta(px))
        pos = [cat('xy', (0, 2))]
        crp = [cr[:, :2]]
        if self.n_dims == 3:
            pos.append(th[:, 2:3].double() / pixel_x(px))        # z in px of the x pixel size
            crp.append(cr[:, 2:3] / pixel_x(px))
        fr = (torch.as_tensor(frames).cpu() if frames is not None else torch.zeros(len(th), dtype=torch.int32))
        return Dataset(dict(frame=fr, pos=torch.cat(pos, 1), crlb=torch.cat(crp, 1), photons=th[:, 3], bg=th[:, 4],
                            crlb_photons=cr[:, 3], crlb_bg=cr[:, 4], loglik=-cat('nll', (0,)), chi2=cat('chi2', (0,)),
                            at_limit=cat('lim', (0,)).bool()), meta)

    def localize(self, movie, camera=None, frames=None, threshold=6.0, chunk=500, block=5000, progress=None,
                 stop=None, pixel_um=None, iters=30, zstarts=5, batch=4000, **detect_kw):
        """Localize a movie: detection -> spot ROIs (roisize = self.roisize) -> fit, in blocks of `block` frames.

        movie: path (.tif / .npy), array [T, H, W] or reader (torch_psf.movie). camera: Camera or calibration json path
        (None: the movie is in photons). frames: (f0, f1). threshold: x noise of the band-pass filtered frame.
        progress(frames_done, frames_total, n_locs) (after every chunk), stop() -> bool: callbacks. iters, zstarts,
        batch: passed to fit (zstarts=1 for near-2D data is ~5x faster). detect_kw: detect_spots options (sigma, nms,
        noise, isolation, median). Returns a Dataset (image coordinates; ids 0..n-1; meta image_size, n_frames,
        chi2_median)."""
        if camera is not None and not isinstance(camera, Camera):
            camera = Camera.from_json(camera)
        mv = open_movie(movie)
        f0, f1 = (0, mv.n) if frames is None else frames
        out = []
        for b0 in range(f0, f1, block):
            b1 = min(b0 + block, f1)
            nloc = sum(len(o) for o in out)
            prog = None if progress is None else (lambda done, tot, n, b0=b0, nloc=nloc:
                                                  progress(b0 - f0 + done, f1 - f0, nloc))
            sp = extract_spots(mv, self.roisize, camera, (b0, b1), chunk, self.dev, threshold=threshold,
                               progress=prog, **detect_kw)
            if len(sp):
                out.append(self.fit(sp, pixel_um=pixel_um, iters=iters, zstarts=zstarts, batch=batch))
            if progress is not None:
                progress(b1 - f0, f1 - f0, sum(len(o) for o in out))
            if stop is not None and stop():
                break
        if out:
            ds = Dataset.concatenate(out)
            ds['id'] = torch.arange(len(ds), dtype=torch.int64)
            if len(ds):
                ds.meta['chi2_median'] = float(ds['chi2'].median())
        else:
            ds = self.fit(torch.zeros(0, self.roisize, self.roisize), pixel_um=pixel_um)
        ds.meta.update(image_size=[int(v) for v in mv.shape[-2:]], n_frames=int(f1),
                       method=f'torch_psf {type(self).__name__}.localize')
        return ds

    # ------------------------------------------------------------------ files
    def format_version(self, source=None):
        """Minimum (major, minor) for the serialized features, including any stored source."""
        if self.type_name is None:
            raise NotImplementedError(f'{type(self).__name__} has no model file block')
        return pixel_version((0, 2), getattr(self, 'pixel_um', None))

    def file_extensions(self):
        """Extension entries (torch_psf.extensions.entry) stored with this model; none for the built-in models.
        A model extension (a subclass or an attached module from another package) returns its own here."""
        return []

    def json_format_version(self, source=None):
        """Minimum version in the standalone layout, which predates the 0.2 channel container."""
        version = self.format_version(source)
        return (0, 1) if version == (0, 2) else version

    def _file_dict(self, source=None):
        """(json dict without format / version / type, {name: array} for the .npz) of this model; subclasses."""
        raise NotImplementedError

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        """The model from a parsed PSF json d and its arrays; subclasses."""
        raise NotImplementedError

    def save(self, path, camera=None, provenance=None, source=None):
        """Write the model to a PSF json (format 'torch-psf': type, optics, model parameters; see the subclass) plus an
        .npz next to it when there are arrays (pupil maps, knots). source: for a spline, the vectorial model it was
        built from (its parameters are stored too). camera: optional Camera. provenance: dict stored with the model
        (default: self.provenance, which the estimators set to their settings and report). Returns the json dict."""
        from torch_psf import extensions
        d, arrays = self._file_dict(source)
        d = extensions.with_entries(self, d)
        version = self.json_format_version(source)
        if extensions.used(d):
            version = max(version, extensions.EXTENSIONS_VERSION)
        d = dict(format=FORMAT, version=version_string(version), type=self.type_name, **d)
        if camera is not None:
            d['camera'] = camera.to_dict()
        d['provenance'] = _plain(provenance if provenance is not None else (self.provenance or {}))
        path = str(path)
        if arrays:
            npz = os.path.splitext(path)[0] + '.npz'
            np.savez(npz, **arrays)
            d['arrays'] = os.path.basename(npz)
        with open(path, 'w') as f:
            json.dump(d, f, indent=1)
        return d

    @staticmethod
    def read_file(path):
        """(parsed json, arrays or None, type name) of a PSF json, a psf.zip (single-channel or coupled; the 0.1.0
        dict of its model, torch_psf.psfzip.legacy) or a plain calibration json (path or parsed dict; array files
        of a dict are resolved against the current directory)."""
        from torch_psf import psfzip             # lazy: psfzip imports this module
        if psfzip.is_psf_zip(path):
            return psfzip.legacy(*psfzip.read_zip(path))
        if isinstance(path, dict):
            d, base = path, '.'
        else:
            path = str(path)
            with open(path) as f:
                d = json.load(f)
            base = os.path.dirname(path)
        if d.get('format') == FORMAT:
            check_version(d.get('version', '0.1.0'), ((0, 1), *READ_VERSIONS), 'psf.json')
            typ = d.get('type', 'vectorial')
        else:                    # plain calibration json: vectorial, or a 2D Gaussian with psf_gauss_sigma_um
            typ = 'gaussian' if d.get('psf_gauss_sigma_um') is not None else 'vectorial'
        arrays = dict(np.load(os.path.join(base, d['arrays']))) if d.get('arrays') else None
        return d, arrays, typ

    @staticmethod
    def _has_angle_terms(psf):
        """True if the model has per-pattern-angle terms that PSF.load(angle=a) selects: VectorialPSF psf_dipole_M_angle /
        psf_blur_angle_px2, SplinePSF coefs_angle."""
        return bool(getattr(psf, 'n_angles', 0)) or any(getattr(psf, k, None) is not None
                                                        for k in ('dipM_angle', 'blur_angle'))

    @staticmethod
    def load(path, roisize=None, device=None, angle=None, npupil=None, oversample=None):
        """Read a single-model PSF json, a psf.zip or a plain calibration json. For one independent channel use
        torch_psf.io.load(path, channel=c). The path may also be the parsed dict (array files are then resolved against the
        current directory). roisize: for vectorial / Gaussian models (default: the stored one, or 15 for a
        calibration json). angle: for calibrations with per-pattern-angle PSF terms: 0 / 1 = that angle, None = the
        average. npupil / oversample override the stored pupil sampling of a vectorial model. A calibration with
        psf_gauss_sigma_um gives a GaussianPSF (sigma = psf_gauss_sigma_um / pixel size). A psf.zip channel with a
        zernike_poly field block gives the field-dependent model (torch_psf.field: FieldPSF of a vectorial channel,
        TiledSplinePSF of a spline channel with its vectorial source)."""
        d, arrays, typ = PSF.read_file(path)
        if typ not in PSF._registry:
            raise ValueError(f'unknown PSF type {typ!r} (known: {sorted(PSF._registry)})')
        psf = PSF._registry[typ]._from_file(d, arrays, roisize=roisize, device=device, angle=angle, npupil=npupil,
                                            oversample=oversample)
        from torch_psf import extensions
        psf = extensions.apply(psf, d.get('extensions'), f'PSF model ({typ})')
        if angle is not None and not PSF._has_angle_terms(psf):
            unused = (d.get('provenance') or {}).get('unused_model_keys') or {}
            hint = (' (its per-angle terms are only in provenance.unused_model_keys: the file was converted by a '
                    'torch-psf that did not keep them; convert the calibration json again)') \
                if 'psf_dipole_M_angle' in unused or 'psf_blur_angle_px2' in unused else ''
            raise ValueError(f'angle={angle!r}, but this PSF model ({typ}) has no per-angle terms{hint}')
        if d.get('provenance'):
            psf.provenance = d['provenance']
        fd = d.get('_field')
        if fd:                      # a psf.zip channel with a zernike_poly field block: the field-dependent model
            from torch_psf.psfzip import field_model
            src = None if typ == 'vectorial' else PSF._registry['vectorial']._from_file(
                d, arrays, roisize=d.get('roisize', 15), device=device, npupil=npupil, oversample=oversample)
            psf = field_model(psf, src, fd['block'], None, fd.get('crop'), fd.get('tiles'))
        return psf

    # ------------------------------------------------------------------ cubic spline
    def to_spline(self, roisize=None, z_range=None, **grid):
        """Sample the optical model on a regular spline grid (see torch_psf.to_spline)."""
        from torch_psf.spline import _tabulate
        # Historical method calls without a kernel keep the file model's arithmetic.
        grid.setdefault('backend', None)
        return _tabulate(self, z_range, roisize=roisize, **grid)
