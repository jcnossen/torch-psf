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


@torch.no_grad()
def heldout_nll(psf, rois, var=None, theta0=None):
    """Mean NLL per pixel of the patches rois [N, P, P] after localizing each with psf (Poisson, constant dropped);
    also the fraction at the offset limit. Lower is better; differences between PSFs are what matters."""
    th, nll = localize(psf, rois, theta0, var=var)
    return dict(nll_per_px=float(nll.mean()) / rois[0].numel(), at_limit=float(at_limit(th, psf).float().mean()),
                theta=th)


def compare_heldout(psfs, rois, var=None):
    """{name: heldout_nll(psf)} for a dict of PSF models, plus the difference to the first one (nll_per_px_vs_first)."""
    out = {k: heldout_nll(p, rois, var) for k, p in psfs.items()}
    first = next(iter(out.values()))['nll_per_px']
    for v in out.values():
        v['nll_per_px_vs_first'] = v['nll_per_px'] - first
        v.pop('theta')
    return out
