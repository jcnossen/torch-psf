"""Held-out checks of a PSF estimate: split the spots by position into a checkerboard of tiles (train on one parity,
test on the other: the same emitters / structures never appear in both), and compare PSF models by the mean
negative log-likelihood per pixel of the test spots after a full refit of each spot."""
import numpy as np
import torch
from torch_psf.fit import localize, at_limit


def split_checkerboard(x, y, tile=16):
    """Parity (0 / 1) of the tile (tile x tile px) that holds each position (image px; numpy or torch)."""
    x, y = np.asarray(x.cpu() if torch.is_tensor(x) else x), np.asarray(y.cpu() if torch.is_tensor(y) else y)
    return ((x // tile).astype(int) + (y // tile).astype(int)) % 2


def _localize(psf, rois, theta0, var, zstarts, xy):
    if xy is not None and hasattr(psf, 'bind'):           # a field-dependent model (torch_psf.field)
        from torch_psf.estimation.field_knots import localize_field
        return localize_field(psf, rois, xy, var=var, theta0=theta0, zstarts=zstarts)
    return localize(psf, rois, theta0, var=var, zstarts=zstarts)


@torch.no_grad()
def heldout_nll(psf, rois, var=None, theta0=None, zstarts=5, xy=None):
    """Mean NLL per pixel of the patches rois [N, P, P] after localizing each with psf (Poisson, constant dropped);
    also the fraction at the offset limit and the NLL per spot (nll [N] numpy). Lower is better; differences between
    PSFs are what matters. zstarts: z starts of the multi-start localization. xy [N, 2]: field positions of the ROIs
    (camera px) for a field-dependent model (torch_psf.field.FieldPSF / TiledSplinePSF; ignored by other models)."""
    th, nll = _localize(psf, rois, theta0, var, zstarts, xy)
    lim = at_limit(th, psf).float().mean() if hasattr(psf, 'xy_limit') else torch.zeros(())
    return dict(nll_per_px=float(nll.mean()) / rois[0].numel(), at_limit=float(lim), theta=th,
                nll=nll.double().cpu().numpy())


def compare_heldout(psfs, rois, var=None, zstarts=5, xy=None):
    """{name: heldout_nll(psf)} for a dict of PSF models, plus the difference to the first one (nll_per_px_vs_first)
    and, per spot, the mean NLL difference to the first one with its standard error (dnll_per_spot_vs_first,
    dnll_se) and the fraction of spots that it fits better (frac_better). xy: field positions for field-dependent
    models (heldout_nll)."""
    out = {k: heldout_nll(p, rois, var, zstarts=zstarts, xy=xy) for k, p in psfs.items()}
    v0 = next(iter(out.values()))
    first, n0 = v0['nll_per_px'], v0['nll']
    for v in out.values():
        v['nll_per_px_vs_first'] = v['nll_per_px'] - first
        dd = v['nll'] - n0
        v['dnll_per_spot_vs_first'] = float(dd.mean()) if len(dd) else 0.0
        v['dnll_se'] = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
        v['frac_better'] = float((dd < 0).mean()) if len(dd) else 0.0
    for v in out.values():
        v.pop('theta'); v.pop('nll')
    return out


@torch.no_grad()
def spot_deviance(psf, rois, var=None, theta=None, zstarts=5, xy=None, batch=4000):
    """Poisson deviance per pixel of every ROI under psf after its MLE fit (theta: fitted parameters, else fitted
    here): 2 sum(d log(d / mu) - (d - mu)) / P^2, about 1 for a spot the model explains. Overlapping emitters and
    other spots the single-emitter model does not explain give larger values. Returns (deviance [N] numpy, theta).
    zstarts / xy: as heldout_nll (a field-dependent model needs xy)."""
    if theta is None:
        theta, _ = _localize(psf, rois, None, var, zstarts, xy)
    field = xy is not None and hasattr(psf, 'bind')
    if field:
        from torch_psf.estimation.field_knots import expected_field
        xy = torch.as_tensor(xy, dtype=torch.float32, device=rois.device).reshape(-1, 2)
    out = []
    for s in range(0, len(rois), batch):                   # in batches: [N, P, P] images of a large set
        th = theta[s:s + batch]
        mu = expected_field(psf, th, xy[s:s + batch]) if field else psf.expected(th)
        d = rois[s:s + batch]
        v = var[s:s + batch] if (torch.is_tensor(var) and var.dim() > 0) else var
        if v is not None:
            d, mu = d + v, mu + v
        dv = 2 * (torch.where(d > 0, d * torch.log(d.clamp_min(1e-9) / mu.clamp_min(1e-6)), torch.zeros_like(d))
                  - (d - mu))
        out.append((dv.flatten(1).sum(1) / d[0].numel()).cpu().numpy())
    return (np.concatenate(out) if out else np.zeros(0)), theta
