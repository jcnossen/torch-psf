"""The result type of every PSF estimator."""
import dataclasses
import json
import math
import numpy as np
import torch


def _jsonable(v):
    if torch.is_tensor(v):
        return v.detach().cpu().tolist() if v.numel() <= 64 else f'<tensor {tuple(v.shape)}>'
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else f'<array {v.shape}>'
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


@dataclasses.dataclass
class PSFEstimate:
    """Result of a PSF estimator.

    psf: the estimated PSF model (VectorialPSF, or ConvSplinePSF for free knots).
    report: dict of summary numbers (always: method, seconds, at_limit = fraction of fits at the lateral offset limit;
        in situ also z_scale = z axis of the estimate relative to the starting PSF, held-out results when computed,
        and warnings).
    history: per-iteration records (dicts).
    data: method-specific arrays (per-emitter theta, knots, pupil maps, bead positions, ...).
    """
    psf: object
    report: dict
    history: list = dataclasses.field(default_factory=list)
    data: dict = dataclasses.field(default_factory=dict)

    def save(self, path, camera=None, source=None):
        """Write the PSF (torch_psf.io.save_psf) with the report as provenance."""
        from torch_psf.io import save_psf
        return save_psf(path, self.psf, source=source, camera=camera, provenance=dict(report=_jsonable(self.report)))

    def __repr__(self):
        r = ', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in self.report.items()
                      if isinstance(v, (int, float, str)))
        return f'PSFEstimate({type(self.psf).__name__}; {r})'


@torch.no_grad()
def z_scale(psf_new, psf_ref, z_range=None, n=33, photons=1e5, bg=1.0):
    """z axis of psf_new relative to psf_ref: noise-free bright emitters at x = y = 0 and n heights in z_range rendered
    with psf_new are localized with psf_ref; returns the slope d z_ref / d z_new of a straight line through the inner
    80 % of the range (1 = same z scale) and the offset (um) at the range centre."""
    zr = z_range or psf_new.z_range
    z = torch.linspace(zr[0], zr[1], n, device=psf_ref.dev if hasattr(psf_ref, 'dev') else None)
    pos = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], 1)
    mu = photons * psf_new.render(pos) + bg
    from torch_psf.fit import localize
    th, _ = localize(psf_ref, mu, None)
    k = slice(int(0.1 * n), n - int(0.1 * n))
    A = torch.stack([torch.ones_like(z[k]), z[k] - z[k].mean()], 1).double()
    co = torch.linalg.lstsq(A, th[k, 2:3].double()).solution[:, 0]
    return float(co[1]), float(co[0] - z[k].mean())


FREE_KNOT_WARNING = ('free knots can improve the held-out likelihood and the repeat precision while warping the z '
                     'axis (blinking data do not fix the z scale; see docs/paper, section 4.7): check report["z_scale"] '
                     'against a reference')
