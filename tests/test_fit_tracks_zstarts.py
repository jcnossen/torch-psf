"""fit_tracks(zstarts=n): per-sample start heights. With samples on both sides of the focus of an astigmatic PSF, every
sample starts within one grid step of its height on the correct side (the default starts all at zmax / 2), and the
default (zstarts=None) is unchanged."""
import torch

from conftest import DEV
from torch_psf import VectorialPSF
from torch_psf.estimation import fit_tracks
from torch_psf.spots import Tracks


def _tracks(B=20, T=4, R=11):
    vec = VectorialPSF(R, npupil=32, oversample=2, device=DEV, wavelength_um=0.7,
                       pixel_integration='midpoint').set_zernike({6: 0.12})
    g = torch.Generator(device=DEV).manual_seed(1)
    xy = 0.6 * (torch.rand(B, 2, generator=g, device=DEV) - 0.5)
    z = torch.where(torch.arange(B, device=DEV) % 2 == 0, torch.tensor(0.07, device=DEV), torch.tensor(0.33, device=DEV))
    with torch.no_grad():
        img = vec(torch.cat([xy, z[:, None]], 1), torch.full((B,), 0.2, device=DEV))
    rois = torch.poisson(5000.0 * img[:, None].expand(-1, T, -1, -1) + 5.0, generator=g)
    tr = Tracks(rois, torch.ones(B, T, dtype=torch.bool, device=DEV), torch.arange(T, device=DEV)[None].expand(B, -1),
                torch.zeros(B, 2, dtype=torch.long, device=DEV), xy + (R - 1) / 2)
    return vec, tr, z


def test_zstarts_start_heights():
    vec, tr, z = _tracks()
    kw = dict(readnoise_e=1.0, zmax_um=0.4, iters=1, warmup=1, focus_um=0.2, fix_zernike=True, fix_focus=True,
              lr_xyz=1e-9)                                        # one negligible step: the start heights
    single = fit_tracks(tr, vec, **kw)['z']
    multi = fit_tracks(tr, vec, zstarts=5, **kw)['z']
    torch.testing.assert_close(single, torch.full_like(single, 0.2), atol=1e-6, rtol=0)
    assert float((multi - z).abs().max()) <= 0.4 / 5 + 1e-6      # within one grid step of the true height
    assert bool(((multi - 0.2).sign() == (z - 0.2).sign()).all())   # on the correct side of the focus


def test_zstarts_converges():
    vec, tr, z = _tracks()
    kw = dict(readnoise_e=1.0, zmax_um=0.4, iters=150, warmup=150, focus_um=0.2, fix_zernike=True, fix_focus=True)
    assert float((fit_tracks(tr, vec, zstarts=5, **kw)['z'] - z).abs().max()) < 0.03
