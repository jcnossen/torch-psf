"""PSF estimators: from bead z-stacks (ZStackEstimator; DenseBeadEstimator for dense, overlapping beads) and in situ
from an SMLM movie (InSituEstimator).

Both take the starting point the same way (a PSF object or file, or Optics), an optional Camera, progress / stop
hooks, and return the estimated PSF itself; the numbers of the run are on the estimator: estimator.report (dict) and
estimator.history (list of per-iteration records).

    est = InSituEstimator(start='beads.json', camera=camera, refine=True)
    psf = est.estimate('movie.tif')
    print(est.report['heldout'], est.report['z_scale'])
"""
import copy
import dataclasses
import time
import numpy as np
import tifffile
import torch
from scipy.ndimage import gaussian_filter, maximum_filter
from torch_psf.optics import Optics, pixel_x, resolve_device
from torch_psf.camera import Camera
from torch_psf.fit import poisson_nll
from torch_psf.psf import PSF
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import SplinePSF
from torch_psf.composition import ZMode
from torch_psf.gaussian import AstigmaticGaussianPSF
from torch_psf.movie import open_movie
from torch_psf.spots import extract_spots
from torch_psf.estimation import (fit_beads, stage_focus, fit_zernike_knots, fit_pupil_knots, fit_free_knots,
                              split_checkerboard, compare_heldout, spot_deviance, z_scale, FREE_KNOT_WARNING)
from torch_psf.fit import localize


def _camera(camera):
    if camera is None or isinstance(camera, Camera):
        return camera
    if isinstance(camera, dict):
        return Camera.from_dict(camera)
    return Camera.from_json(camera)


def _start_vectorial(start, optics, roisize, device, zernike_indices=tuple(range(5, 22)), npupil=64, oversample=3):
    """Starting VectorialPSF: start (VectorialPSF or PSF file), else optics (Optics / dict / None = defaults)."""
    if start is not None and not isinstance(start, PSF):
        start = PSF.load(start, roisize=roisize, device=device)
    if start is not None:
        if not isinstance(start, VectorialPSF):
            raise TypeError(f'the start must be a vectorial PSF, not {type(start).__name__}')
        return start.resized(roisize) if start.roisize != roisize else start
    o = optics if isinstance(optics, Optics) else Optics.from_dict(optics or {})
    return VectorialPSF.from_optics(o, roisize, zernike_indices=zernike_indices, npupil=npupil, oversample=oversample,
                                    device=device)


def find_beads(stack, roisize, nmax=32, min_sep=None, sigma=2.0, border=None):
    """Bright isolated beads in a z-stack [nk, H, W] (any units): maxima of the blurred maximum projection, at least
    min_sep (default roisize) px from each other and roisize / 2 + 1 px from the border, brightest first.
    Returns corners [n, 2] = (x0, y0) of roisize x roisize ROIs centred on them (numpy int)."""
    mp = gaussian_filter(np.asarray(stack, np.float32).max(0), sigma)
    mp = mp - np.median(mp)
    h = roisize // 2
    sep = roisize if min_sep is None else min_sep
    pk = (mp == maximum_filter(mp, size=2 * h + 1)) & (mp > 0.1 * mp.max())
    y, x = np.nonzero(pk)
    b = h + 1 if border is None else border
    ok = (x >= b) & (y >= b) & (x < mp.shape[1] - b) & (y < mp.shape[0] - b)
    x, y = x[ok], y[ok]
    o = np.argsort(-mp[y, x])
    x, y = x[o], y[o]
    keep = []
    for i in range(len(x)):
        dd = np.max(np.abs(np.stack([x - x[i], y - y[i]], 1)), 1)
        if keep and (dd[keep] < sep).any():
            continue
        keep.append(i)
        if len(keep) >= nmax:
            break
    return np.stack([x[keep] - h, y[keep] - h], 1).astype(int)


class Estimator:
    """Base of the estimators: device, progress(stage, iteration, value, seconds) and stop() -> bool hooks, and the
    report / history of the last estimate()."""

    def __init__(self, device=None, progress=None, stop=None):
        self.dev = resolve_device(device)
        self.progress, self.stop = progress, stop
        self.report, self.history = {}, []
        self._t0 = time.time()

    def _log(self, stage):
        """a log(it, value, seconds, ...) callback for the low-level fits that records the history and calls
        progress"""
        def log(it, value, seconds, *rest):
            self.history.append(dict(stage=stage, it=it, value=float(value), seconds=float(seconds)))
            if self.progress is not None:
                self.progress(stage, it, float(value), float(seconds))
        return log

    def _stopped(self):
        return self.stop is not None and bool(self.stop())

    def _reset(self):
        self.report, self.history = {}, []
        self._t0 = time.time()


class ZStackEstimator(Estimator):
    """PSF from bead z-stacks.

    model: 'vectorial' (Zernike fit of a VectorialPSF), 'spline' (a SplinePSF whose knots are fitted to the
    stacks, regularised towards the vectorial fit with weight `regularize`), or 'astig_gauss' (the astigmatic
    Gaussian calibration curves). start (VectorialPSF or PSF file) or optics: the starting vectorial model.
    camera: Camera / calibration json / None (stacks in photons). roisize (px), nbeads (per stack, the brightest
    isolated ones), sign (scan direction), iters, holdout (index of a stack kept out of the fit; its beads are refitted
    (own position / photons / bg, PSF fixed) and their NLL per pixel is reported), drift / drift_order (lateral drift
    of each stack over the planes, a polynomial of this degree; default cubic), fit_dz_scale (fit the z-step scale),
    fit_aspect (vectorial / spline: fit the pixel aspect py / px of non-square camera pixels at a fixed geometric-mean
    pixel size; report pixel_aspect, pixel_um; use it with fit_height, see estimation.fit_beads),
    emitter_z_um (starting bead centre height; default bead_diameter_um / 2, else 0), fit_height (fit it),
    bead_diameter_um (finite bead size; with fit_height the diameter is tied to 2 x the fitted height: a bead lies on
    the coverslip), fit_na / fit_n_med / fit_apod (fit NA / the medium index / the radial pupil amplitude apod),
    intensity ('free' or 'bead_x_plane': per-plane factors, e.g. bleaching); see estimation.fit_beads.
    Spline grid: oversample, spline_dz_um, xy_limit.

    estimate(stacks, dz_um) -> PSF. Plane k of a stack is at focus sign dz_um (k - (nk - 1) / 2) um (the stack centre
    is focus 0, so centre the stacks on the beads); the beads sit on the coverslip. The vectorial model's focus_um is
    0; the spline and the astigmatic Gaussian have z = -focus (the defocus of the bead, in stage um, no refractive-index
    rescaling) over the scanned range."""

    def __init__(self, model='vectorial', start=None, optics=None, camera=None, roisize=21, nbeads=8, sign=1.0,
                 iters=300, holdout=None, drift=True, drift_order=3, fit_dz_scale=False, fit_height=False,
                 emitter_z_um=None, bead_diameter_um=None, fit_na=False, fit_n_med=False, fit_apod=False,
                 intensity='free', regularize=3000.0, oversample=2, spline_dz_um=None, xy_limit=3.0,
                 zernike_indices=tuple(range(5, 22)), npupil=64, oversample_pupil=3, device=None, progress=None,
                 stop=None, fit_aspect=False):
        super().__init__(device, progress, stop)
        if model not in ('vectorial', 'spline', 'astig_gauss'):
            raise ValueError("model: 'vectorial', 'spline' or 'astig_gauss'")
        self.model, self.start, self.optics, self.camera = model, start, optics, _camera(camera)
        self.roisize, self.nbeads, self.sign, self.iters = roisize, nbeads, sign, iters
        self.holdout, self.drift, self.regularize = holdout, drift, regularize
        self.fit_dz_scale, self.fit_height, self.bead_diameter_um, self.intensity = (fit_dz_scale, fit_height,
                                                                                     bead_diameter_um, intensity)
        self.drift_order, self.fit_na, self.fit_n_med, self.fit_apod = drift_order, fit_na, fit_n_med, fit_apod
        self.fit_aspect = fit_aspect
        # starting bead (centre) height: given, else the bead radius (a bead lies on the coverslip), else 0
        self.emitter_z_um = (float(emitter_z_um) if emitter_z_um is not None else
                             0.5 * float(bead_diameter_um) if bead_diameter_um else 0.0)
        self.oversample, self.spline_dz_um, self.xy_limit = oversample, spline_dz_um, xy_limit
        self.zernike_indices, self.npupil, self.oversample_pupil = zernike_indices, npupil, oversample_pupil

    def _rois(self, stacks):
        rois, sid = [], []
        ii = torch.arange(self.roisize, device=self.dev)
        for i, s in enumerate(stacks):
            a = tifffile.imread(s) if isinstance(s, str) else np.asarray(s)
            a = torch.as_tensor(np.asarray(a, np.float32), device=self.dev)
            if self.camera is not None:
                a = self.camera.to_photons(a)
            for x0, y0 in find_beads(a.cpu().numpy(), self.roisize, nmax=self.nbeads):
                rois.append(a[:, y0 + ii[:, None], x0 + ii[None, :]])
                sid.append(i)
        if not rois:
            raise ValueError('no beads found')
        return torch.stack(rois), torch.tensor(sid, device=self.dev)

    def estimate(self, stacks, dz_um):
        """stacks: list of [nk, H, W] arrays or TIFF paths (one bead field each, the same nk). Returns the PSF (its
        provenance holds the settings and the report)."""
        psf = self._estimate(stacks, dz_um)
        psf.provenance = dict(estimator='ZStackEstimator', model=self.model, dz_um=float(dz_um), report=self.report)
        return psf

    def _estimate(self, stacks, dz_um):
        self._reset()
        if isinstance(stacks, (str, np.ndarray)) or torch.is_tensor(stacks):
            stacks = [stacks]
        d, sid = self._rois(stacks)
        nb, nk, P = d.shape[0], d.shape[1], d.shape[-1]
        focus = stage_focus(nk, dz_um, self.sign).to(self.dev)
        w = None if self.holdout is None else (sid != self.holdout).float()
        self.report.update(model=self.model, n_stacks=len(stacks), n_beads=nb, n_planes=nk, dz_um=dz_um,
                           sign=self.sign)
        if self.model == 'astig_gauss':
            psf = self._fit_astig(d, focus, w)
            self.report['seconds'] = time.time() - self._t0
            return psf
        vec = _start_vectorial(self.start, self.optics, P, self.dev, self.zernike_indices, self.npupil,
                               self.oversample_pupil)
        fit = (('zcoef',) + (('NA',) if self.fit_na else ()) + (('n_med',) if self.fit_n_med else ())
               + (('apod',) if self.fit_apod else ()))
        r = fit_beads(d, focus, vec, iters=self.iters, fit=fit, stack_id=sid, drift=self.drift,
                      drift_order=self.drift_order, weights=w, emitter_z_um=self.emitter_z_um,
                      fit_dz_scale=self.fit_dz_scale, fit_height=self.fit_height,
                      bead_diameter_um=self.bead_diameter_um, intensity=self.intensity,
                      log=self._log('vectorial'), stop=self.stop, fit_aspect=self.fit_aspect)
        psf = r['psf']
        psf.focus_um = 0.0
        if self.fit_na or self.fit_n_med:
            self.report.update(NA=float(psf.NA), n_med=float(psf.n_med))
        if self.fit_apod:
            self.report['apod'] = psf.apod.detach().cpu().tolist()
        for k in ('nll_per_px', 'heldout_nll_per_px', 'dz_scale', 'height_um', 'chi2_median', 'heldout_chi2_median',
                  'pixel_aspect', 'pixel_um'):
            if k in r['report']:
                self.report[k] = r['report'][k]
        self.data = dict(rois=d, stack_id=sid, focus_um=focus, xy=r['xy'], photons=r['photons'], bg=r['bg'],
                         model=r['model'], chi2=r['chi2'])
        if self.model == 'vectorial':
            self.report['seconds'] = time.time() - self._t0
            return psf
        return self._fit_spline(psf, d, focus, sid)

    def _fit_spline(self, vec, d, focus, sid):
        """knots fitted to the bead stacks (z = -focus), starting from and regularised towards the vectorial fit"""
        zlo, zhi = float(-focus.max()), float(-focus.min())
        dz = self.spline_dz_um or max(abs(float(focus[1] - focus[0])), 0.02)
        spl = ZMode(vec, focus=0.).to_spline(self.roisize, (zlo + dz, zhi - dz), oversample=self.oversample, dz_um=dz, zpad=2,
                            xy_limit=self.xy_limit, kernel='catmull', integrated=False)   # the knot route refines pixel-value knots
        K0 = spl.knots[0].clone()
        keep = sid != self.holdout if self.holdout is not None else torch.ones_like(sid, dtype=torch.bool)
        z_nom = -focus
        inr = (z_nom >= spl.z_limits[0]) & (z_nom <= spl.z_limits[1])
        stack = dict(d=d[keep][:, inr], z_nom=z_nom[inr])
        r = fit_free_knots(spl, K0, stack=stack, ref=K0 if self.regularize > 0 else None, lam_ref=self.regularize,
                           outer=8, log=self._log('spline'), stop=self.stop)
        self.report.update(spline=dict(z_range=list(spl.z_range), oversample=spl.S, dz_um=spl.hz,
                                       change_vs_vectorial=float((r['knots'] - K0).norm() / K0.norm())))
        self.report['seconds'] = time.time() - self._t0
        return r['psf']

    def _fit_astig(self, d, focus, w):
        """Poisson fit of the astigmatic Gaussian curves (sx0, sy0, gamma, d) with per-bead xy and per-plane
        photons / bg; z = -focus."""
        nb, nk, P = d.shape[0], d.shape[1], d.shape[-1]
        o = Optics.from_dict(self.optics) if isinstance(self.optics, dict) else (self.optics or Optics())
        s0 = 0.21 * o.wavelength_um / o.NA / pixel_x(o.pixel_um) * 1.1
        z = (-focus)[None].expand(nb, -1).reshape(-1)
        lsx = torch.tensor([np.log(s0), np.log(s0)], device=self.dev, requires_grad=True)
        gam = torch.tensor(0.15, device=self.dev, requires_grad=True)
        ld = torch.tensor(np.log(0.4), device=self.dev, requires_grad=True)
        xy = torch.zeros(nb, 2, device=self.dev, requires_grad=True)
        bg0 = d.flatten(2).quantile(0.1, 2)
        logI = torch.log((d.flatten(2).sum(2) - bg0 * P ** 2).clamp_min(100)).requires_grad_(True)
        bg = bg0.clone().requires_grad_(True)
        opt = torch.optim.Adam([dict(params=[lsx, gam, ld], lr=0.01), dict(params=[xy], lr=0.01),
                                dict(params=[logI], lr=0.01), dict(params=[bg], lr=0.3)])
        dd = d.reshape(-1, P, P)
        wt = None if w is None else w.repeat_interleave(nk)
        zr = (float(z.min()), float(z.max()))
        log = self._log('astig_gauss')
        for it in range(self.iters):
            opt.zero_grad()
            psf = AstigmaticGaussianPSF(P, (1.0, 1.0), 0.0, 1.0, zr, device=self.dev)
            psf.sigma = (torch.exp(lsx[0]), torch.exp(lsx[1]))
            psf.gamma, psf.d = gam, torch.exp(ld)
            pos = torch.cat([xy[:, None].expand(-1, nk, -1).reshape(-1, 2), z[:, None]], 1)
            mu = torch.exp(logI).reshape(-1, 1, 1) * psf.render(pos) + bg.reshape(-1, 1, 1)
            l = poisson_nll(dd, mu)
            l = (l * wt).sum() if wt is not None else l.sum()
            l.backward()
            opt.step()
            log(it, float(l), time.time() - self._t0)
            if self._stopped():
                break
        out = AstigmaticGaussianPSF(P, (float(torch.exp(lsx[0])), float(torch.exp(lsx[1]))), float(gam),
                                    float(torch.exp(ld)), zr, device=self.dev, pixel_um=o.pixel_um)
        self.report.update(sigma_px=list(out.sigma), gamma_um=out.gamma, d_um=out.d,
                           nll_per_px=float(l.detach()) / dd.numel())
        return out


class DenseBeadEstimator(Estimator):
    """PSF from bead z-stacks with dense, overlapping beads (vectorial model; torch_psf.estimation.dense_beads): all
    beads are detected in the maximum projection over z, beads whose patch x patch boxes overlap are grouped into
    clusters, and one shared VectorialPSF is fitted to all clusters jointly (per bead position and photons per plane,
    per cluster and plane a background, per stack a lateral drift and a focus offset). Same starting point (start /
    optics), camera and progress / stop hooks as ZStackEstimator; same z convention (plane k at focus
    sign dz_um (k - (nk - 1) / 2), the beads on the coverslip at emitter_z_um).

    patch: grouping box (odd px; None = psf_extent of the start model over the scanned range, rel=extent_rel, at most
    max_patch); render_pad: extra px per side in which every bead is rendered (default patch // 3: the tails of a
    member on its neighbours' pixels are modelled); region (x0, y0, x1, y1): crop of every stack (e.g. one channel of
    a split camera frame); field_stop: 'auto' (detect a dark field-stop region; clusters with a member box outside it
    are dropped), False, or a bool mask [H, W] (of the region); threshold (robust sd), min_dist (px), faint (relative
    amplitude below which a detection does not take part in the grouping but joins a cluster it lies in), max_members,
    max_box (px): detection and grouping (estimation.find_all_beads / find_bead_clusters); holdout: index of a stack
    kept out of the fit, or holdout_fraction: a random fraction of the clusters (seed holdout_seed); held-out clusters
    are refitted with the PSF fixed and scored (report heldout_nll_per_px, heldout_deviance_per_px); plane_step: use
    every n-th plane; reject: drop clusters whose chi2 is above reject x the median (debris, unresolved aggregates).
    drift / drift_order / focus_offset / bg / intensity / fit_dz_scale / fit_na / fit_n_med / fit_apod (the radial
    pupil amplitude apod) / emitter_z_um / warmup / max_renders: see estimation.fit_dense_beads.

    field: field-dependent Zernikes (a dict, torch_psf.field.field_from_spec: modes, order, prior_sd, iters): after
    the global fit, a second fit from it with every bead at the Zernikes base + poly(x, y) of its position (camera px)
    and the polynomial fitted jointly with the base Zernikes (and NA / apod / n_med when fitted); the held-out clusters
    are scored with both (report heldout_* = global, report['field'] = the field fit's numbers) and the field model
    (torch_psf.field.FieldPSF) is returned when it is better on them (or when nothing is held out).

    estimate(stacks, dz_um) -> VectorialPSF (roisize patch, focus_um 0) or FieldPSF. report: n_detected, n_clusters,
    n_beads, cluster_sizes ({size: count}), dropped ({reason: clusters}), patch, render_pad, nll_per_px,
    deviance_per_px, chi2_median, heldout_*, field, chosen. data (BeadClusters), result (the fit_dense_beads dict;
    result['xy_frame']: bead positions in the full frame), result_field (the field fit) and detections (per stack:
    xy, amp in region px, field-stop mask) for figures."""

    def __init__(self, start=None, optics=None, camera=None, patch=None, render_pad=None, sign=1.0, iters=300,
                 warmup=50, holdout=None, holdout_fraction=0.0, holdout_seed=0, region=None, field_stop='auto',
                 threshold=6.0, min_dist=3, faint=0.1, max_members=8, max_box=None, drift=True, drift_order=3,
                 focus_offset='tilt', bg='linear', intensity='free', fit_dz_scale=False, fit_na=False,
                 fit_n_med=False, fit_apod=False, emitter_z_um=0.0, plane_step=1, reject=None, extent_rel=0.02,
                 max_patch=31,
                 zernike_indices=tuple(range(5, 22)), npupil=64, oversample_pupil=3, max_renders=3000, field=None,
                 device=None, progress=None, stop=None):
        super().__init__(device, progress, stop)
        self.start, self.optics, self.camera = start, optics, _camera(camera)
        self.field = field
        self.patch, self.render_pad, self.sign, self.iters, self.warmup = patch, render_pad, sign, iters, warmup
        self.holdout, self.holdout_fraction, self.holdout_seed = holdout, holdout_fraction, holdout_seed
        self.region, self.field_stop, self.threshold, self.min_dist = region, field_stop, threshold, min_dist
        self.max_members, self.max_box, self.faint = max_members, max_box, faint
        self.fit_kw = dict(drift=drift, drift_order=drift_order, focus_offset=focus_offset, bg=bg, intensity=intensity,
                           fit_dz_scale=fit_dz_scale, emitter_z_um=float(emitter_z_um), max_renders=max_renders)
        self.fit_na, self.fit_n_med, self.plane_step, self.reject = fit_na, fit_n_med, plane_step, reject
        self.fit_apod = fit_apod
        self.extent_rel, self.max_patch = extent_rel, max_patch
        self.zernike_indices, self.npupil, self.oversample_pupil = zernike_indices, npupil, oversample_pupil

    @staticmethod
    def fit_movie(movie, groups=None, *, optics=None, radius=7, calibration_stack=None, dz_um=None):
        """Fit overlapping beads at a single fixed focus. Supply a TYX movie in photoelectrons.
        Optional groups are condition IDs, one per exposure. Returns a DenseMovieFit with
        the in-plane PSF, positions, per-frame bead photons, drift and background. This
        movie entry point also accepts a calibration ZYX stack in photoelectrons and its signed
        step dz_um in micrometres. The stack fits the shared pupil before movie XYZ drift.
        optics supplies the bead diameter and the nominal movie focus in micrometres.
        """
        if calibration_stack is not None:
            from os import PathLike, fspath
            from torch_psf.estimation.crowded_movie import fit_crowded_movie
            if optics is None or 'diameter_um' not in optics or 'focus_um' not in optics:
                raise ValueError('stack-calibrated movies require optics with diameter_um and focus_um')
            if dz_um is None or not np.isfinite(dz_um) or dz_um == 0:
                raise ValueError('supply the signed calibration stack spacing dz_um')
            stack = tifffile.imread(fspath(calibration_stack)) if isinstance(calibration_stack, (str, PathLike)) else calibration_stack
            if stack.ndim != 3 or len(stack) < 3:
                raise ValueError('expected a ZYX calibration stack with at least three planes')
            calibration = DenseBeadEstimator(optics=optics, patch=2*radius+1, sign=np.sign(dz_um),
                emitter_z_um=optics['diameter_um']/2, zernike_indices=tuple(range(5,12)),
                field_stop=False, npupil=32, oversample_pupil=2)
            psf = calibration.estimate(stack, abs(dz_um))
            settings = dict(optics, calibrated_psf=psf, focus_span_um=abs(dz_um)*(len(stack)-1)/2)
            frames = movie.detach().cpu().numpy() if torch.is_tensor(movie) else np.asarray(movie)
            if frames.ndim != 3 or not len(frames) or not np.isfinite(frames).all():
                raise ValueError('expected a finite TYX movie')
            return fit_crowded_movie(frames, groups, settings, radius)
        from torch_psf.estimation.bead_movie import fit_bead_movie
        return fit_bead_movie(movie, groups, optics=optics, radius=radius)

    def _stacks(self, stacks):
        if isinstance(stacks, (str, np.ndarray)) or torch.is_tensor(stacks):
            stacks = [stacks]
        out = []
        for s in stacks:
            a = tifffile.imread(s) if isinstance(s, str) else s
            a = (a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a, np.float32))).to(self.dev).float()
            if self.camera is not None:
                a = self.camera.to_photons(a)
            if self.region is not None:
                x0, y0, x1, y1 = self.region
                a = a[:, y0:y1, x0:x1]
            out.append(a[::self.plane_step])
        return out

    def estimate(self, stacks, dz_um):
        """stacks: list of [nk, H, W] arrays / tensors or TIFF paths (the same nk). Returns the VectorialPSF (its
        provenance holds the settings and the report)."""
        from torch_psf.estimation.dense_beads import (field_stop_mask, find_all_beads, find_bead_clusters,
                                                      psf_extent, BeadClusters, fit_dense_beads)
        self._reset()
        st = self._stacks(stacks)
        nk_full = (st[0].shape[0] - 1) * self.plane_step + 1
        focus = stage_focus(nk_full, dz_um, self.sign).to(self.dev)[::self.plane_step]
        vec = _start_vectorial(self.start, self.optics, 21, self.dev, self.zernike_indices, self.npupil,
                               self.oversample_pupil)
        P = self.patch or psf_extent(vec, focus, self.fit_kw['emitter_z_um'],
                                     rel=self.extent_rel, pmax=self.max_patch)
        pad = P // 3 if self.render_pad is None else int(self.render_pad)
        clusters, self.detections, dropped, ndet = [], [], {}, 0
        for i, a in enumerate(st):
            an = a.cpu().numpy()
            if isinstance(self.field_stop, np.ndarray):
                mask = self.field_stop
            elif self.field_stop == 'auto':
                mask = field_stop_mask(an)
            else:
                mask = None
            xy, amp = find_all_beads(an, threshold=self.threshold, min_dist=self.min_dist)
            kept, drop = find_bead_clusters(xy, amp, P, an.shape[1:], mask=mask, max_members=self.max_members,
                                            max_box=self.max_box, faint=self.faint)
            for c in kept:
                c['stack'] = i
            clusters += kept
            for c in drop:
                dropped[c['why']] = dropped.get(c['why'], 0) + 1
            ndet += len(xy)
            self.detections.append(dict(xy=xy, amp=amp, mask=mask, kept=kept, dropped=drop))
        if not clusters:
            raise ValueError('no usable bead clusters found')
        data = BeadClusters.from_stacks(st, clusters, P, self.dev, render_pad=pad)
        train = np.ones(data.n_clusters, bool)
        if self.holdout is not None:
            train = data.cl_stack.cpu().numpy() != self.holdout
        elif self.holdout_fraction > 0:
            rng = np.random.default_rng(self.holdout_seed)
            nho = int(round(self.holdout_fraction * data.n_clusters))
            train[rng.permutation(data.n_clusters)[:nho]] = False
        if not train.any():
            raise ValueError('no training clusters left')
        sizes = data.cluster_sizes()
        self.report.update(n_stacks=len(st), n_planes=len(focus), dz_um=dz_um * self.plane_step, sign=self.sign,
                           n_detected=ndet, n_clusters_found=data.n_clusters, n_beads_found=data.n_beads,
                           n_faint_members=int(sum(np.sum(c['faint']) for c in clusters)),
                           cluster_sizes={int(k): int((sizes == k).sum()) for k in np.unique(sizes)}, dropped=dropped,
                           patch=P, render_pad=pad)
        fit = (('zcoef',) + (('NA',) if self.fit_na else ()) + (('n_med',) if self.fit_n_med else ())
               + (('apod',) if self.fit_apod else ()))
        r = fit_dense_beads(data, focus, vec, iters=self.iters, fit=fit, train=train, warmup=self.warmup,
                            reject=self.reject, log=self._log('dense_beads'), stop=self.stop, **self.fit_kw)
        psf = r['psf'].resized(P)
        psf.focus_um = 0.0
        for k, v in r['report'].items():
            if k not in ('method', 'patch'):
                self.report[k] = v
        if self.fit_na or self.fit_n_med:
            self.report.update(NA=float(psf.NA), n_med=float(psf.n_med))
        if self.fit_apod:
            self.report['apod'] = psf.apod.detach().cpu().tolist()
        off = torch.tensor(self.region[:2] if self.region is not None else (0, 0), device=self.dev).float()
        r['xy_frame'] = r['xy'] + off
        self.data, self.result, self.focus_um = data, r, focus
        self.result_field = None
        if self.field and not self._stopped():
            from torch_psf.field import field_from_spec, FieldPSF
            fxy = data.xy_image + off                                        # bead positions, camera px
            btr = torch.as_tensor(r['train'], device=self.dev)[data.bead_cluster]
            fz, fkw = field_from_spec(self.field, psf.zidx, fxy[btr])
            it_f = int(self.field.get('iters', self.iters)) if isinstance(self.field, dict) else self.iters
            rf = fit_dense_beads(data, focus, r['psf'], iters=it_f, fit=fit + ('field',), train=r['train'],
                                 warmup=self.warmup, reject=None, field=fz, xy=fxy,
                                 exclude=r['report'].get('rejected_clusters') or None,
                                 log=self._log('dense_beads field'), stop=self.stop, **self.fit_kw)
            fpsf = FieldPSF(rf['psf'].resized(P), rf['field'])
            fpsf.base.focus_um = 0.0
            rr = rf['report']
            self.report['field'] = {k: rr[k] for k in ('nll_per_px', 'deviance_per_px', 'chi2_median', 'heldout_nll_per_px',
                                                       'heldout_deviance_per_px', 'field', 'seconds') if k in rr}
            self.report['field'].update(block=fpsf.field_block(), feats_norm=list(fz.norm))
            if self.fit_na or self.fit_n_med:
                self.report['field'].update(NA=float(fpsf.base.NA), n_med=float(fpsf.base.n_med))
            if self.fit_apod:
                self.report['field']['apod'] = fpsf.base.apod.detach().cpu().tolist()
            better = ('heldout_nll_per_px' not in rr or 'heldout_nll_per_px' not in self.report
                      or rr['heldout_nll_per_px'] < self.report['heldout_nll_per_px'])
            self.report['field']['kept'] = bool(better)
            rf['xy_frame'] = rf['xy'] + off
            self.result_field = rf
            if better:
                psf = fpsf
        self.report['chosen'] = type(psf).__name__
        self.report['seconds'] = time.time() - self._t0
        psf.provenance = dict(estimator='DenseBeadEstimator', dz_um=float(dz_um), report=self.report)
        return psf


class InSituEstimator(Estimator):
    """In-situ PSF from an SMLM movie: detection, spot ROIs sampled over the whole frame range, a checkerboard split
    into training / held-out spots, the vectorial Zernike fit by the knot route (torch_psf.estimation.fit_zernike_knots;
    method 'pupil' adds free pupil maps), rejection of spots the single-emitter model does not explain (overlapping
    emitters; see Overlap rejection), an optional scan of the focus and / or the wavelength by held-out NLL, an optional
    cubic-spline refinement (refine=True: free knots regularised towards the vectorial spline, kept only if the
    held-out NLL improves), the held-out comparison and a chi2 check of the camera model.

    start (VectorialPSF or PSF file, e.g. a bead fit) or optics: the starting model (zernike_indices, npupil,
    oversample_pupil: pupil model of a start built from optics); focus_um: the focal plane during the acquisition
    (default: the start's focus_um). camera: Camera / calibration json / None (movie in photons). roisize (px),
    z_range (um): emitter heights. nspots / nheldout: training / held-out spots (tile px checkerboard).
    Spots: the movie is read in chunks of `chunk` frames in a bit-reversed order over the requested range
    (torch_psf.spots.extract_spots(sample='spread')) until 2 (nspots + nheldout) spots are found, so they come from the
    whole range, not from its start.
    Overlap rejection (reject_overlap): Poisson deviance per pixel of each training spot after the first fit; rejected
    above median + deviance_k x max(1.4826 MAD, median sqrt(2 / P^2)), robust to a camera gain error or EMCCD
    statistics, which scale all deviances. A warning is added when more than warn_rejected of the spots are rejected.
    Focus / wavelength (for 2D data the focus is weakly determined): scan_focus_um and / or scan_wavelength_um = lists of
    values; each combination is fitted (scan_iters iterations) and the one with the lowest held-out NLL is used.
    Spline grid: oversample, dz_um, xy_limit (>= 3 px recommended). Detection: threshold, isolation, detect_kw;
    detector: another detection function (e.g. torch_psf.matched_filter.MatchedFilterDetector for engineered PSFs,
    whose band-pass peaks sit on a lobe). zstarts: z starts of every multi-start localization (default 5; ~9 for a
    6 um tetrapod).
    Field dependence (field: a dict, see torch_psf.field.field_from_spec: modes, order, prior_sd, tile, iters, lr,
    score_tile): after the global fit (and the overlap rejection) the base Zernikes and a polynomial of the field
    position per mode are fitted jointly (estimation.fit_field_knots, per field tile), started from the global fit;
    the field model (its tiled spline, score_tile px tiles) is a held-out candidate ('field') and is returned (as a
    torch_psf.field.FieldPSF) when it wins. Held-out spots are scored at their own field positions.

    estimate(movie, frames=None) -> PSF (VectorialPSF, or the refined SplinePSF when refine=True and it
    wins on the held-out spots); its provenance holds the settings and the report. report: n_spots, n_train, n_test,
    frames_requested, frames_read, frame_span, deviance (median, scale, threshold, rejected_fraction), n_rejected,
    at_limit, z_scale / z_offset_um (fit vs start), scan, heldout (NLL per pixel of start / vectorial / refined),
    chosen, heldout_chi2_median (held-out spots fitted with the chosen PSF: ~1 if the camera gain and noise model are
    right), field (with field=: the fit report and coefficients), warnings. data: spots, train, test, vectorial (the
    fitted VectorialPSF), vectorial_spline (its spline), field_psf / field_tiled (with field=),
    psf (the returned PSF), spline (the spline of the returned PSF, ready for localization); with reject_overlap also
    the first-fit overlap test of the training spots: deviance [n_train] (numpy; rejected where >= the threshold),
    deviance_theta [n_train, 5] (their fits) and deviance_psf (the first-fit spline)."""

    def __init__(self, start=None, optics=None, camera=None, focus_um=None, roisize=15, z_range=(0.0, 0.8),
                 nspots=8000, nheldout=4000, method='zernike', iters=80, reject_overlap=True, deviance_k=3.0,
                 warn_rejected=0.25, refine=False, refine_lam_ref=3000.0,
                 refine_lam_s=1000.0, refine_outer=12, oversample=2, dz_um=0.04, xy_limit=3.0, threshold=6.0,
                 isolation=None, tile=16, chunk=50, zernike_indices=tuple(range(5, 22)), npupil=64,
                 oversample_pupil=3, scan_focus_um=None, scan_wavelength_um=None, scan_iters=None, detector=None,
                 zstarts=5, field=None, device=None, progress=None, stop=None, **detect_kw):
        super().__init__(device, progress, stop)
        self.start, self.optics, self.camera, self.focus_um = start, optics, _camera(camera), focus_um
        self.detector, self.zstarts, self.field = detector, int(zstarts), field
        self.roisize, self.z_range, self.nspots, self.nheldout = roisize, tuple(z_range), nspots, nheldout
        self.method, self.iters, self.reject_overlap, self.deviance_k = method, iters, reject_overlap, deviance_k
        self.warn_rejected = warn_rejected
        self.refine, self.refine_lam_ref, self.refine_lam_s, self.refine_outer = (refine, refine_lam_ref,
                                                                                   refine_lam_s, refine_outer)
        self.grid = dict(oversample=oversample, dz_um=dz_um, zpad=5, xy_limit=xy_limit, kernel='catmull', integrated=False)  # knot-route tables
        self.threshold, self.isolation, self.tile, self.detect_kw = threshold, isolation, tile, detect_kw
        self.chunk = chunk
        self.zernike_indices, self.npupil, self.oversample_pupil = tuple(zernike_indices), npupil, oversample_pupil
        self.scan_focus_um, self.scan_wavelength_um, self.scan_iters = scan_focus_um, scan_wavelength_um, scan_iters

    def _fit(self, spline, rois, vec, iters, stage, var=None):
        fn = fit_zernike_knots if self.method == 'zernike' else fit_pupil_knots
        if self.method not in ('zernike', 'pupil'):
            raise ValueError(self.method)
        loc = None
        if self.zstarts != 5:
            loc = lambda spl, d, th: localize(spl, d, th, var=var, iters=30 if th is None else 3, zstarts=self.zstarts)
        return fn(spline, rois, vec, iters=iters, log=self._log(stage), stop=self.stop, var=var, localizer=loc)

    def _variant(self, vec0, focus_um, wavelength_um):
        """the start model at another focus / wavelength (the Zernikes of the start are kept)"""
        if wavelength_um is None or float(wavelength_um) == float(vec0.optics.wavelength_um):
            v = copy.deepcopy(vec0)
        else:
            o = dataclasses.replace(vec0.optics, wavelength_um=float(wavelength_um))
            v = VectorialPSF.from_optics(o, vec0.roisize, zernike_indices=tuple(vec0.zidx), npupil=vec0.npupil,
                                         oversample=vec0.os, device=self.dev, focus_um=vec0.focus_um,
                                         z_range=vec0.z_range, pixel_integration=vec0.pixel_integration)
            with torch.no_grad():
                v.zcoef.copy_(vec0.zcoef.detach())
        v.focus_um = float(vec0.focus_um if focus_um is None else focus_um)
        v.z_range = self.z_range
        return v

    def _deviance_threshold(self, dv):
        P2 = self.roisize ** 2
        med = float(np.median(dv))
        mad = 1.4826 * float(np.median(np.abs(dv - med)))
        scale = max(mad, med * float(np.sqrt(2.0 / P2)))
        return med + self.deviance_k * scale, med, scale

    def estimate(self, movie, frames=None):
        self._reset()
        R, zr = self.roisize, self.z_range
        vec0 = _start_vectorial(self.start, self.optics, R, self.dev, self.zernike_indices, self.npupil,
                                self.oversample_pupil)
        if self.focus_um is not None:
            vec0.focus_um = float(self.focus_um)
        vec0.z_range = zr
        mv = open_movie(movie)
        f0, f1 = (0, mv.n) if frames is None else (int(frames[0]), int(frames[1]))
        sp = extract_spots(mv, R, self.camera, (f0, f1), chunk=self.chunk, device=self.dev, sample='spread',
                           max_spots=2 * (self.nspots + self.nheldout), threshold=self.threshold,
                           isolation=self.isolation, detector=self.detector, **self.detect_kw)
        warnings = []
        if len(sp) == 0:
            raise ValueError(f'no spots detected in frames {f0}..{f1}')
        h = (R - 1) / 2
        par = split_checkerboard(sp.corner[:, 0] + h, sp.corner[:, 1] + h, self.tile)
        tr = np.flatnonzero(par == 0)[:self.nspots]
        te = np.flatnonzero(par == 1)[:self.nheldout]
        rois = sp.rois[torch.as_tensor(tr, device=self.dev)]
        rois_te = sp.rois[torch.as_tensor(te, device=self.dev)]
        # read-noise variance of the ROIs (photons^2; Camera.variance: scalar, or per ROI with per-pixel maps)
        v_all = sp.var if (sp.var is not None and float(sp.var.abs().max()) > 0) else None
        sub = lambda idx: v_all if (v_all is None or v_all.dim() == 0) else v_all[torch.as_tensor(idx, device=v_all.device)]
        var, var_te = sub(tr), sub(te)
        fr = sp.frame.cpu().numpy()
        self.report.update(method=self.method, n_spots=len(sp), n_train=len(tr), n_test=len(te),
                           frames_requested=[f0, f1], frames_read=sp.frames_read,
                           frame_span=[int(fr.min()), int(fr.max()) + 1], n_chunks=len(sp.frame_ranges or []),
                           spots_per_frame=len(sp) / max(sp.frames_read, 1), warnings=warnings)
        if len(tr) < self.nspots:
            warnings.append(f'only {len(tr)} training spots (nspots {self.nspots}): the whole range was read')
        spline = SplinePSF(R, zr, device=self.dev, **self.grid)

        zs = self.zstarts

        def heldout(v):
            return compare_heldout({'m': v.to_spline(R, zr, **self.grid)}, rois_te, var_te, zstarts=zs)['m']['nll_per_px']

        # focus / wavelength scan (held-out NLL)
        fs = list(self.scan_focus_um) if self.scan_focus_um is not None else [None]
        ws = list(self.scan_wavelength_um) if self.scan_wavelength_um is not None else [None]
        if (len(fs) > 1 or len(ws) > 1 or fs[0] is not None or ws[0] is not None) and len(te):
            scan = []
            for fz in fs:
                for wl in ws:
                    if self._stopped():
                        break
                    v = self._variant(vec0, fz, wl)
                    r = self._fit(spline, rois, v, self.scan_iters or max(self.iters // 2, 1),
                                  f'scan focus {v.focus_um:g} wavelength {v.optics.wavelength_um:g}', var)
                    scan.append(dict(focus_um=float(v.focus_um), wavelength_um=float(v.optics.wavelength_um),
                                     heldout_nll_per_px=heldout(r['psf'])))
            best = min(scan, key=lambda q: q['heldout_nll_per_px'])
            self.report['scan'] = dict(results=scan, best=best)
            vec0 = self._variant(vec0, best['focus_um'], best['wavelength_um'])
            for k, vals in (('focus_um', fs), ('wavelength_um', ws)):
                if len(vals) > 2 and best[k] in (min(vals), max(vals)):
                    warnings.append(f'the best {k} ({best[k]:g}) is at the edge of the scanned values')

        r = self._fit(spline, rois, vec0, self.iters, 'vectorial', var)
        vec = r['psf']
        n_rej = 0
        first = None
        tr_kept = tr
        if self.reject_overlap and not self._stopped():
            s1 = vec.to_spline(R, zr, **self.grid)
            dv, th1 = spot_deviance(s1, rois, var, zstarts=zs)
            first = dict(deviance=dv, deviance_theta=th1.detach(), deviance_psf=s1)
            thr, med, scale = self._deviance_threshold(dv)
            keep = dv < thr
            n_rej = int((~keep).sum())
            frac = n_rej / max(len(dv), 1)
            self.report['deviance'] = dict(rule='median', k=self.deviance_k, median=med, scale=scale,
                                           threshold=float(thr), rejected_fraction=frac,
                                           quantiles=dict(zip(('q05', 'q25', 'q50', 'q75', 'q95'),
                                                              np.quantile(dv, [.05, .25, .5, .75, .95]).tolist())))
            if frac > self.warn_rejected:
                warnings.append(f'{100 * frac:.0f} % of the training spots were rejected as overlaps (median deviance '
                                f'{med:.2f} per px): dense data (use isolation=...) or a wrong camera model (gain, '
                                f'excess_noise)')
            if med > 1.3:
                warnings.append(f'median deviance per px {med:.2f} (1 for Poisson statistics): check the camera gain '
                                f'and excess_noise (EMCCD: 2)')
            kt = torch.as_tensor(keep, device=self.dev)
            rois = rois[kt]
            tr_kept = tr[keep]
            var = var if (var is None or var.dim() == 0) else var[kt]
            r = self._fit(spline, rois, vec, max(self.iters * 3 // 4, 1), 'vectorial (overlaps rejected)', var)
            vec = r['psf']
        s0 = vec0.to_spline(R, zr, **self.grid)
        s_vec = vec.to_spline(R, zr, **self.grid)
        sc, off = z_scale(s_vec, s0)
        self.report.update(n_rejected=n_rej, at_limit=r['report']['at_limit'], z_scale=sc, z_offset_um=off,
                           focus_um=float(vec.focus_um), wavelength_um=float(vec.optics.wavelength_um))
        cands = {'start': s0, 'vectorial': s_vec}
        result, result_spline = vec, s_vec
        xy_te = None
        fres = {}
        if self.field and not self._stopped():
            from torch_psf.field import field_from_spec
            from torch_psf.estimation import fit_field_knots
            spec = dict(self.field)
            xy_all = sp.corner.float() + h
            xy_tr = xy_all[torch.as_tensor(tr_kept, device=xy_all.device)]
            xy_te = xy_all[torch.as_tensor(te, device=xy_all.device)]
            fz, fkw = field_from_spec(spec, vec.zidx, xy_tr)
            rf = fit_field_knots(SplinePSF(R, zr, device=self.dev, **self.grid), rois, xy_tr, vec, fz,
                                 tile=fkw['tile'], iters=fkw['iters'], lr=fkw['lr'], zstarts=zs, var=var,
                                 log=self._log('field'), stop=self.stop)
            fres = dict(psf=rf['psf'], tiled=rf['psf'].to_tiled(tile=fkw['score_tile'], roisize=R, z_range=zr,
                                                                **self.grid))
            cands['field'] = fres['tiled']
            self.report['field'] = dict(rf['report'], modes=fz.noll, orders=fz.orders, feats_norm=list(fz.norm),
                                        block=rf['psf'].field_block(), score_tile=fkw['score_tile'])
        if self.refine and not self._stopped():
            s_ref = vec.to_spline(R, zr, **self.grid)          # a separate grid object: the fit overwrites its knots
            K_vec = s_ref.knots[0].clone()
            rr = fit_free_knots(s_ref, K_vec, rois=rois, var=var, ref=K_vec, lam_ref=self.refine_lam_ref,
                                lam_s=self.refine_lam_s, outer=self.refine_outer, log=self._log('refine'),
                                stop=self.stop)
            cands['refined'] = rr['psf']
            sc_r, off_r = z_scale(rr['psf'], s_vec)
            self.report['refine'] = dict(z_scale_vs_vectorial=sc_r, z_offset_um_vs_vectorial=off_r,
                                         at_limit=rr['report']['at_limit'])
            warnings.append(FREE_KNOT_WARNING)
        if len(te):
            ho = compare_heldout(cands, rois_te, var_te, zstarts=zs, xy=xy_te)
            self.report['heldout'] = ho
            best = ho['vectorial']['nll_per_px']
            if 'refined' in cands:
                better = ho['refined']['nll_per_px'] < ho['vectorial']['nll_per_px']
                self.report['refine']['kept'] = bool(better)
                if better:
                    result = result_spline = cands['refined']
                    best = ho['refined']['nll_per_px']
            if 'field' in cands:
                better = ho['field']['nll_per_px'] < best
                self.report['field']['kept'] = bool(better)
                if better:
                    result, result_spline = fres['psf'], fres['tiled']
            ds = result_spline.fit(rois_te, var=var_te, pixel_um=vec.pixel_um, zstarts=zs,
                                   corners=sp.corner[torch.as_tensor(te, device=sp.corner.device)])
            c2 = float(ds['chi2'].median())
            self.report['heldout_chi2_median'] = c2
            if not 0.7 < c2 < 1.4:
                warnings.append(f'held-out median chi2 per px {c2:.2f} (about 1 expected): check the camera gain / '
                                f'offset / excess_noise')
        elif fres:
            result, result_spline = fres['psf'], fres['tiled']
        self.report['chosen'] = type(result).__name__
        self.report['seconds'] = time.time() - self._t0
        result.provenance = dict(estimator='InSituEstimator', settings=self._settings(), report=self.report)
        self.data = dict(spots=sp, train=tr, test=te, vectorial=vec, vectorial_spline=s_vec, psf=result,
                         spline=result_spline, **(first or {}))
        if fres:
            self.data.update(field_psf=fres['psf'], field_tiled=fres['tiled'])
        return result

    def _settings(self):
        keep = ('focus_um', 'roisize', 'z_range', 'nspots', 'nheldout', 'method', 'iters', 'reject_overlap',
                'deviance_k', 'refine', 'threshold', 'isolation', 'tile', 'chunk',
                'zernike_indices', 'npupil', 'scan_focus_um', 'scan_wavelength_um', 'scan_iters', 'zstarts', 'field')
        d = {k: getattr(self, k) for k in keep}
        d['camera'] = None if self.camera is None else self.camera.to_dict()
        d['detect_kw'] = dict(self.detect_kw)
        return d


class PhaseDiversityEstimator(Estimator):
    """PSF from z-stacks of a thin object (beads on the coverslip at any density) by phase diversity
    (torch_psf.estimation.phase_diversity): every plane images the same object through the PSF at a known focus, the
    object is eliminated (stage 1: Gonsalves' reduced least-squares form on windowed patches) or profiled out
    (stage 2, method='poisson': Poisson likelihood with an extended free object solved by Newton / CG), and the
    VectorialPSF parameters (Zernikes; fit_na, fit_apod (radial pupil amplitude), fit_n_med, fit_pupil (a free pupil
    phase map, L2 penalty pupil_l2)) are fitted by autograd. No bead detection, grouping or bead model: overlapping
    beads, aggregates and debris are part of the object; beads off the coverslip (other heights) are not modelled.

    Same starting point (start / optics), camera, region and field stop handling as DenseBeadEstimator; plane k at
    focus sign dz_um (k - (nk - 1) / 2) (+ a focus offset per patch). patch (px) / step: the patch grid (patches with
    at least min_valid of their pixels inside the field stop and a peak of min_signal robust sd); a shared pupil over
    all patches (holdout: index of a stack kept out of the fit; its patches are refitted with the PSF fixed and
    scored: report heldout_deviance_per_px), then with per_patch=True one pupil per patch (started from the shared
    one, patch_ls_iters / patch_iters iterations): self.field (list of per-patch dicts: centre, Zernikes, NA, apod,
    focus offset, deviance) and self.patch_psfs. focus_offset / drift / drift_order / intensity / bg / fit_dz_scale /
    emitter_z_um / render_size / oversample / window / eps / ls_iters / iters / cg_iters / max_planes: see
    estimation.fit_phase_diversity.

    field: field-dependent Zernikes (a dict, torch_psf.field.field_from_spec: modes, order, prior_sd, iters): after
    the shared pupil, a joint fit (from it, Poisson stage, the PSF held for field_warmup iterations) of the base and
    a polynomial of the patch centre (camera px) per mode, over all training patches; the held-out patches are scored
    with both (report heldout_deviance_per_px = shared pupil, report['field'] = the field fit's numbers) and the field
    model (torch_psf.field.FieldPSF) is returned when it is better (or when nothing is held out).

    estimate(stacks, dz_um) -> VectorialPSF (roisize `roisize`, focus_um 0) or FieldPSF; report, patches (PDPatches),
    result, result_field."""

    def __init__(self, start=None, optics=None, camera=None, patch=64, step=None, region=None, field_stop='auto',
                 min_valid=0.5, min_signal=8.0, sign=1.0, method='poisson', fit_na=False, fit_apod=False,
                 fit_n_med=False, fit_pupil=False, pupil_l2=1.0, ls_iters=200, iters=300, per_patch=False,
                 patch_ls_iters=0, patch_iters=200, holdout=None, plane_step=1, focus_offset='patch', drift='poly',
                 drift_order=3, intensity='plane', bg='const', fit_dz_scale=False, emitter_z_um=0.0, render_size=41,
                 oversample=1, window=0.25, eps=1e-4, cg_iters=6, roisize=21, zernike_indices=tuple(range(5, 22)),
                 npupil=64, oversample_pupil=3, max_planes=600, field=None, field_warmup=40, device=None, progress=None,
                 stop=None):
        super().__init__(device, progress, stop)
        self.start, self.optics, self.camera = start, optics, _camera(camera)
        self.field, self.field_warmup = field, field_warmup
        self.patch, self.step, self.region, self.field_stop = patch, step, region, field_stop
        self.min_valid, self.min_signal, self.sign, self.method = min_valid, min_signal, sign, method
        self.fit = (('zcoef',) + (('NA',) if fit_na else ()) + (('apod',) if fit_apod else ())
                    + (('n_med',) if fit_n_med else ()) + (('pupil_phase',) if fit_pupil else ()))
        self.ls_iters, self.iters, self.per_patch = ls_iters, iters, per_patch
        self.patch_ls_iters, self.patch_iters, self.holdout, self.plane_step = (patch_ls_iters, patch_iters, holdout,
                                                                                 plane_step)
        self.fit_kw = dict(focus_offset=focus_offset, drift=drift, drift_order=drift_order, intensity=intensity, bg=bg,
                           fit_dz_scale=fit_dz_scale, emitter_z_um=float(emitter_z_um), render_size=render_size,
                           oversample=oversample, window=window, eps=eps, cg_iters=cg_iters, max_planes=max_planes,
                           pupil_l2=pupil_l2 if fit_pupil else 0.0)
        self.roisize, self.zernike_indices, self.npupil, self.oversample_pupil = (roisize, zernike_indices, npupil,
                                                                                  oversample_pupil)

    _stacks = DenseBeadEstimator._stacks

    def estimate(self, stacks, dz_um):
        from torch_psf.estimation.dense_beads import field_stop_mask
        from torch_psf.estimation.phase_diversity import PDPatches, fit_phase_diversity
        self._reset()
        st = self._stacks(stacks)
        nk_full = (st[0].shape[0] - 1) * self.plane_step + 1
        focus = stage_focus(nk_full, dz_um, self.sign).to(self.dev)[::self.plane_step]
        vec = _start_vectorial(self.start, self.optics, self.roisize, self.dev, self.zernike_indices, self.npupil,
                               self.oversample_pupil)
        masks = []
        for a in st:
            if isinstance(self.field_stop, np.ndarray):
                masks.append(self.field_stop)
            elif self.field_stop == 'auto':
                masks.append(field_stop_mask(a.cpu().numpy()))
            else:
                masks.append(None)
        pt = PDPatches.from_stacks(st, self.patch, self.step, masks=masks, min_valid=self.min_valid,
                                   min_signal=self.min_signal)
        train = np.ones(pt.n, bool) if self.holdout is None else pt.stack.cpu().numpy() != self.holdout
        if not train.any():
            raise ValueError('no training patches')
        self.report.update(n_stacks=len(st), n_planes=len(focus), dz_um=dz_um * self.plane_step, sign=self.sign,
                           n_patches=int(train.sum()), patch=self.patch, fit=list(self.fit))
        r = fit_phase_diversity(pt.subset(np.nonzero(train)[0]), focus, vec, fit=self.fit, method=self.method,
                                ls_iters=self.ls_iters, iters=self.iters, log=self._log('phase_diversity'),
                                stop=self.stop, **self.fit_kw)
        psf = r['psf']
        psf.focus_um = 0.0
        for k, v in r['report'].items():
            if k not in ('n_patches', 'patch'):
                self.report[k] = v
        if (~train).any():
            rh = fit_phase_diversity(pt.subset(np.nonzero(~train)[0]), focus, psf, fit=(), method='poisson',
                                     ls_iters=0, iters=max(self.iters // 2, 1), **self.fit_kw)
            self.report.update(heldout_deviance_per_px=rh['report']['deviance_per_px'],
                               heldout_nll_per_px=rh['report']['nll_per_px'], heldout_n_patches=int((~train).sum()))
            self.heldout_result = rh
        self.result_field = None
        fpsf = None
        if self.field and not self._stopped():
            from torch_psf.field import field_from_spec, FieldPSF
            off = np.asarray(self.region[:2] if self.region is not None else (0, 0), float)
            cen = pt.centers + off                                           # patch centres, camera px
            itr, iho = np.nonzero(train)[0], np.nonzero(~train)[0]
            fz, fkw = field_from_spec(self.field, psf.zidx, cen[itr])
            it_f = int(self.field.get('iters', self.iters)) if isinstance(self.field, dict) else self.iters
            fk = dict(self.fit_kw)
            rf = fit_phase_diversity(pt.subset(itr), focus, psf, fit=self.fit + ('field',), method='poisson',
                                     ls_iters=0, iters=it_f, warmup=self.field_warmup, field=fz, xy=cen[itr],
                                     log=self._log('phase_diversity field'), stop=self.stop, **fk)
            fpsf = FieldPSF(rf['psf'], rf['field'])
            fpsf.base.focus_um = 0.0
            rr = rf['report']
            self.report['field'] = dict(deviance_per_px=rr.get('deviance_per_px'), nll_per_px=rr.get('nll_per_px'),
                                        field=rr.get('field'), NA=rr.get('NA'), apod=rr.get('apod'),
                                        block=fpsf.field_block(), feats_norm=list(fz.norm), seconds=rr.get('seconds'))
            better = True
            if len(iho):
                rhf = fit_phase_diversity(pt.subset(iho), focus, fpsf, fit=(), method='poisson', ls_iters=0,
                                          iters=max(self.iters // 2, 1), xy=cen[iho], **fk)
                self.report['field'].update(heldout_deviance_per_px=rhf['report']['deviance_per_px'],
                                            heldout_nll_per_px=rhf['report']['nll_per_px'])
                better = rhf['report']['nll_per_px'] < self.report['heldout_nll_per_px']
            self.report['field']['kept'] = bool(better)
            self.result_field = rf
        self.field, self.patch_psfs = [], []
        if self.per_patch:
            for i in np.nonzero(train)[0]:
                if self._stopped():
                    break
                ri = fit_phase_diversity(pt.subset([i]), focus, psf, fit=self.fit, method=self.method,
                                         ls_iters=self.patch_ls_iters, iters=self.patch_iters, **self.fit_kw)
                p = ri['psf']
                p.focus_um = 0.0
                self.patch_psfs.append(p)
                off = np.asarray(self.region[:2] if self.region is not None else (0, 0), float)
                self.field.append(dict(index=int(i), stack=int(pt.stack[i]), corner=(pt.corner[i] + off).tolist(),
                                       center=(pt.centers[i] + off).tolist(), zernike_noll=list(p.zidx),
                                       zcoef=p.zcoef.detach().cpu().tolist(), NA=float(p.NA),
                                       apod=p.apod.detach().cpu().tolist(),
                                       focus_offset_um=float(ri['focus_offset'][0]),
                                       deviance_per_px=float(ri['report'].get('deviance_per_px', float('nan'))),
                                       valid_fraction=float(pt.valid[i].mean())))
            self.report['n_field_patches'] = len(self.field)
        self.patches, self.result, self.focus_um, self.train = pt, r, focus, train
        if fpsf is not None and self.report['field']['kept']:
            psf = fpsf
        self.report['chosen'] = type(psf).__name__
        self.report['seconds'] = time.time() - self._t0
        psf.provenance = dict(estimator='PhaseDiversityEstimator', dz_um=float(dz_um), report=self.report)
        return psf
