"""PSF: the base class of every PSF model.

A model supplies render(pos, grad) (the normalized PSF on a roisize x roisize ROI, and its x / y / z derivatives) and
its geometry (roisize, z_range, xy_limit); everything else lives here, once: the expected image, Fisher information
and Cramer-Rao bound, the batched Poisson maximum-likelihood fit of ROIs, localization of a whole movie, files, and
the conversion to a cubic-spline model.

Parameter vector of an emitter: theta = (x px, y px, z um, photons, bg photons / px), x, y relative to the ROI centre
(pixel centres at integers, x = column, y = row). 2D models (z_range None) keep z fixed (any value).
"""
import numpy as np
import torch


class PSF:
    """Abstract PSF model. Subclasses set self.roisize (odd, px), self.z_range ((lo, hi) um, or None for 2D),
    self.dev (torch.device) and implement render(pos, grad=False); they may override xy_limit, z_limits, resized and
    to."""
    roisize: int
    z_range = None
    pixel_um = None          # lateral pixel size (um); needed to express z in px when fitting a 3D model
    dev = torch.device('cpu')

    # ------------------------------------------------------------------ the model
    def render(self, pos, grad=False):
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

    def to(self, device):
        """The model on another device (a new object or self)."""
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
        from torch_psf.fit import fisher
        params = (0, 1, 2, 3, 4) if self.n_dims == 3 else (0, 1, 3, 4)
        return fisher(self, torch.as_tensor(theta, dtype=torch.float32, device=self.dev), var, None, params)

    def crlb(self, theta, var=None):
        """Cramer-Rao lower bound (standard deviation) [N, 5] of (x px, y px, z um, photons, bg) at theta [N, 5]; the z
        column is NaN for 2D models."""
        from torch_psf.fit import crlb
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
        from torch_psf.fit import localize, crlb, at_limit
        from torch_psf.spots import Spots
        from torch_psf.dataset import Dataset
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
        px = pixel_um if pixel_um is not None else getattr(self, 'pixel_um', None)
        if self.n_dims == 3 and not px:
            raise ValueError(f'{type(self).__name__} has no pixel size: z is stored in px of the lateral pixel size, so '
                             f'pass pixel_um=... or set psf.pixel_um')
        th, cr = cat('theta', (0, 5)), cat('crlb', (0, 5))
        meta = dict(method=f'torch_psf {type(self).__name__}.fit', n_dims=self.n_dims)
        if px:
            meta['pixel_nm'] = 1000.0 * float(px)
        pos = [cat('xy', (0, 2))]
        crp = [cr[:, :2]]
        if self.n_dims == 3:
            pos.append(th[:, 2:3].double() / px)
            crp.append(cr[:, 2:3] / px)
        fr = (torch.as_tensor(frames).cpu() if frames is not None else torch.zeros(len(th), dtype=torch.int32))
        return Dataset(dict(frame=fr, pos=torch.cat(pos, 1), crlb=torch.cat(crp, 1), photons=th[:, 3], bg=th[:, 4],
                            crlb_photons=cr[:, 3], crlb_bg=cr[:, 4], loglik=-cat('nll', (0,)), chi2=cat('chi2', (0,)),
                            at_limit=cat('lim', (0,)).bool()), meta)

    def localize(self, movie, camera=None, frames=None, threshold=6.0, chunk=500, block=5000, progress=None,
                 stop=None, pixel_um=None, **detect_kw):
        """Localize a movie: detection -> spot ROIs (roisize = self.roisize) -> fit, in blocks of `block` frames.

        movie: path (.tif / .npy), array [T, H, W] or reader (torch_psf.movie). camera: Camera or calibration json path
        (None: the movie is in photons). frames: (f0, f1). threshold: x noise of the band-pass filtered frame.
        progress(frames_done, frames_total, n_locs), stop() -> bool: callbacks. detect_kw: detect_spots options
        (sigma, nms, noise, isolation, median). Returns a Dataset (image coordinates; ids 0..n-1; meta image_size,
        n_frames)."""
        from torch_psf.movie import open_movie
        from torch_psf.spots import extract_spots
        from torch_psf.config import Camera
        from torch_psf.dataset import Dataset
        if camera is not None and not isinstance(camera, Camera):
            from torch_psf.io import load_camera
            camera = load_camera(camera)
        mv = open_movie(movie)
        f0, f1 = (0, mv.n) if frames is None else frames
        out = []
        for b0 in range(f0, f1, block):
            b1 = min(b0 + block, f1)
            sp = extract_spots(mv, self.roisize, camera, (b0, b1), chunk, self.dev, threshold=threshold, **detect_kw)
            if len(sp):
                out.append(self.fit(sp, pixel_um=pixel_um))
            if progress is not None:
                progress(b1 - f0, f1 - f0, sum(len(o) for o in out))
            if stop is not None and stop():
                break
        if out:
            ds = Dataset.concatenate(out)
            ds['id'] = torch.arange(len(ds), dtype=torch.int64)
        else:
            ds = self.fit(torch.zeros(0, self.roisize, self.roisize), pixel_um=pixel_um)
        ds.meta.update(image_size=[int(v) for v in mv.shape[-2:]], n_frames=int(f1),
                       method=f'torch_psf {type(self).__name__}.localize')
        return ds

    # ------------------------------------------------------------------ files
    def save(self, path, camera=None, provenance=None, source=None):
        """Write the model to a PSF json (plus an .npz next to it for arrays); see torch_psf.io."""
        from torch_psf.io import save
        return save(path, self, source=source, camera=camera, provenance=provenance)

    @staticmethod
    def load(path, roisize=None, device=None, angle=None, npupil=None, oversample=None):
        """Read a PSF json (type 'vectorial' | 'spline' | 'gaussian' | 'astig_gaussian') or a plain calibration json and
        return the model of that type; path may also be the parsed dict (array files are then resolved against the
        current directory). See torch_psf.io.load."""
        from torch_psf.io import load
        return load(path, roisize=roisize, device=device, angle=angle, npupil=npupil, oversample=oversample)

    # ------------------------------------------------------------------ cubic spline
    def to_spline(self, roisize=None, z_range=None, oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0, chunk=8):
        """Tabulate the model as a CubicSplinePSF for roisize x roisize ROIs over z_range (um; default the model's
        z range): knots every 1/oversample px and dz_um um, offsets up to xy_limit px. The model is rendered on the
        larger ROI the knots need, at oversample x oversample sub-pixel shifts."""
        from torch_psf.spline import CubicSplinePSF
        if self.n_dims == 2 and z_range is None:
            raise ValueError('a 2D model has no z range: pass z_range (the spline is then constant in z)')
        zr = self.z_range if z_range is None else z_range
        spl = CubicSplinePSF(roisize or self.roisize, zr, oversample=oversample, dz_um=dz_um, zpad=zpad,
                                 xy_limit=xy_limit, device=self.dev, pixel_um=self.pixel_um)
        spl.knots = spl.knots_from_model(self if self.roisize == spl.Pl else self.resized(spl.Pl), chunk=chunk)
        return spl
