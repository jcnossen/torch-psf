"""From the movie mode to a PSF-estimation route: which estimator, which inputs it still needs, and what to tell the
user. plan_route(result) is the dispatch table; make_estimator(route, **kw) builds the estimator (lazy import, so
torch_psf.auto stays a light layer).

Routes (mode -> estimator):
  smlm      InSituEstimator (spots of the movie; needs pixel size, wavelength, NA, refractive indices)
  sofi      SOFIEstimator (cumulants of the fluctuations; on the sofi-psf branch until merged)
  zstack    ZStackEstimator (isolated beads), DenseBeadEstimator (overlapping beads) or PhaseDiversityEstimator
            (an extended thin object); all need the focus step between frames
  static    none: the band limit (effective NA / lambda) only; a blind PSF from one focus is ambiguous
  moving    none yet: register the frames on the measured global shifts first; then the band limit, and phase
            diversity if the focus also changed
  scene     none: not a microscope image (the camera's own blur, not an NA)
  no_signal none
"""
import dataclasses
import importlib
import numpy as np


@dataclasses.dataclass
class Route:
    mode: str
    estimator: str = None              # class name, None = no PSF estimator for this mode (yet)
    module: str = 'torch_psf.estimators'
    needs: list = dataclasses.field(default_factory=list)       # inputs the estimator needs and the movie lacks
    hints: dict = dataclasses.field(default_factory=dict)       # numbers from the analysis useful as start values
    steps: list = dataclasses.field(default_factory=list)       # what to do, in order, in words
    note: str = ''

    def to_dict(self):
        return dataclasses.asdict(self)

    def summary(self):
        s = f'route {self.mode}: ' + (self.estimator or 'no PSF estimator')
        if self.needs:
            s += f' (needs {", ".join(self.needs)})'
        for i, t in enumerate(self.steps):
            s += f'\n  {i + 1}. {t}'
        if self.note:
            s += f'\n  note: {self.note}'
        return s


def _optics_needs(known):
    return [k for k in ('pixel_um', 'wavelength_um', 'NA') if known.get(k) is None]


def plan_route(result, pixel_um=None, wavelength_um=None, NA=None, dz_um=None, meta=None):
    """Route for an AutoResult (analyze_movie / analyze_frames). The keyword arguments are what the caller knows
    about the acquisition; missing ones are listed in route.needs, with the analysis' own estimates in route.hints
    where it has one (e.g. the effective NA from the band limit). meta: e.g. the PSF grab meta.json (per-frame
    focusDistance gives the focus steps of a phone focus sweep)."""
    m, b = result.mode, result.band
    k = m.key
    known = dict(pixel_um=pixel_um, wavelength_um=wavelength_um, NA=NA)
    hints = dict(fc_px=b.fc_px, band_status=b.status)
    if getattr(b, 'NA', None) is not None and np.isfinite(b.NA):
        hints['NA_eff'] = b.NA                 # effective NA (<= n_medium for water samples; lower bound for 3D PSFs)
    mode = m.mode
    if mode == 'smlm':
        return Route(mode, 'InSituEstimator', needs=_optics_needs(known), hints=hints, steps=[
            'detect spots and cut ROIs (InSituEstimator does this)',
            'fit a vectorial PSF to the spots in situ; start from a bead calibration if there is one',
            'check the held-out NLL and the chi2 of the camera model in the report'])
    if mode == 'sofi':
        return Route(mode, 'SOFIEstimator', needs=_optics_needs(known), hints=hints, steps=[
            'accumulate the spatial cross-cumulants of the fluctuations',
            'fit the PSF to the cumulants (SOFIEstimator)'],
            note='SOFIEstimator is on the sofi-psf branch until it is merged into master')
    if mode == 'zstack':
        if dz_um is None and meta:
            dz_um = _focus_step_from_meta(meta)
        needs = _optics_needs(known) + ([] if dz_um is not None else ['dz_um'])
        hints.update(peak_pos=k.get('peak_pos'), astig_swing=k.get('astig_swing'))
        iso, dens = k.get('isolated'), k.get('spots_per_psf')
        if iso is not None and np.isfinite(iso) and iso > 0.6 and (dens is None or dens < 0.05):
            est, why = 'ZStackEstimator', 'isolated beads'
        elif iso is not None and np.isfinite(iso) and iso > 0.2:
            est, why = 'DenseBeadEstimator', 'overlapping beads'
        else:
            est, why = 'PhaseDiversityEstimator', 'an extended thin object (no isolated spots)'
        return Route(mode, est, needs=needs, hints=dict(hints, dz_um=dz_um), steps=[
            f'{why}: {est}',
            'the focus step between frames (dz_um) and the scan direction (sign) set the z axis'],
            note='a phone or eyepiece focus sweep has no calibrated focus step: dz_um must come from the stage or '
                 'be fitted (fit_dz_scale)' if dz_um is None else '')
    if mode == 'static':
        return Route(mode, None, hints=hints, steps=[
            'report the band limit (effective NA / lambda, NA with the wavelength)',
            'to get the aberrations: record a focus series (z-stack) or blinking emitters'],
            note='one image at one focus: object and PSF trade off; even and odd aberrations are not separable')
    if mode == 'moving':
        return Route(mode, None, hints=dict(hints, motion_step_px=k.get('motion_step_px'),
                                            motion_path_px=k.get('motion_path_px')), steps=[
            'register the frames on the global shifts (motion statistics)',
            'band limit on the registered frames',
            'if the focus also changed: phase diversity on the registered stack (focus steps unknown: fit them)'],
            note='registration and the moving-sample PSF fit are not implemented yet')
    if mode == 'scene':
        return Route(mode, None, hints=hints, steps=[
            'not a microscope image: no NA to estimate',
            "the camera's blur kernel / MTF could be estimated from edges (not implemented)"])
    return Route(mode, None, steps=['no usable signal: check the input'])


def _focus_step_from_meta(meta):
    """Median focus step between frames from a PSF grab meta.json (frames[i].settings.focusDistance, in diopters
    for most phone cameras: not a microscope focus step, so only returned if the recording says it is in um)."""
    fr = meta.get('frames') or []
    unit = (meta.get('recording') or {}).get('focus_unit')
    if unit != 'um':
        return None
    fd = np.array([f.get('settings', {}).get('focusDistance', np.nan) for f in fr], float)
    d = np.diff(fd[np.isfinite(fd)])
    return float(np.median(d)) if len(d) else None


def make_estimator(route, **kw):
    """The estimator object of a route (kw: its constructor arguments, e.g. optics=..., camera=...). Raises
    ValueError for routes without an estimator, ImportError if the estimator is not available in this install."""
    if route.estimator is None:
        raise ValueError(f'mode {route.mode!r} has no PSF estimator: ' + '; '.join(route.steps))
    mod = importlib.import_module(route.module)
    if not hasattr(mod, route.estimator):
        raise ImportError(f'{route.estimator} is not in {route.module} (this install); {route.note}')
    return getattr(mod, route.estimator)(**kw)
