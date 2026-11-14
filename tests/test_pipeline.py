"""Entry points: localize a simulated movie; bead estimation from simulated stacks; the command line."""
import json
import numpy as np
import tifffile
import torch
from conftest import DEV, needs_gpu
from torch_psf import GaussianPSF, VectorialPSF, localize_movie, estimate_psf_beads, save_psf
from torch_psf.simulate import simulate_movie


def test_localize_movie():
    psf = GaussianPSF(11, sigma_px=1.3, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(1)
    movie, truth = simulate_movie(psf.render, 11, T=100, H=64, W=64, nsites=30, p_on=0.05, mean_on=2.0, bg=10.0,
                                  photons=3000.0, photons_logsd=0.1, z_range=(0.0, 0.0), margin=8, generator=g, device=DEV)
    L = localize_movie(movie.numpy(), psf, device=DEV, isolation=6)
    assert len(L) > 0.5 * len(truth)
    tr = truth.cpu().numpy()
    err = []
    for r in L:
        t = tr[tr[:, 0] == r['frame']]
        dd = np.hypot(t[:, 1] - r['x'], t[:, 2] - r['y'])
        if len(dd) and dd.min() < 1:
            err.append(dd.min())
    assert len(err) > 0.9 * len(L) and np.median(err) < 3 * np.median(L['crlb_x'])


@needs_gpu
def test_estimate_psf_beads(tmp_path):
    vec = VectorialPSF(21, npupil=48, oversample=2, device=DEV).set_zernike({6: 0.08, 11: 0.03})
    nk, H = 31, 64
    focus = 0.03 * (torch.arange(nk, device=DEV) - (nk - 1) / 2)
    stack = torch.full((nk, H, H), 20.0, device=DEV)
    for cx, cy in ((20, 22), (44, 40)):
        img = vec.forward_batched(torch.tensor([[0.2, -0.1, 0.0]] * nk, device=DEV), focus)
        stack[:, cy - 10:cy + 11, cx - 10:cx + 11] += 30000 * img
    stack = torch.poisson(stack).cpu().numpy().astype(np.uint16)
    tifffile.imwrite(tmp_path / 'beads.tif', stack)
    est = estimate_psf_beads([str(tmp_path / 'beads.tif')], dz_um=0.03, optics=dict(pixel_um=0.10833, NA=1.49,
                             wavelength_um=0.68), roisize=21, npupil=48, oversample=2, iters=250, device=DEV)
    err = (est.psf.zcoef.detach() - vec.zcoef.detach()).pow(2).mean().sqrt()
    assert est.report['n_beads'] == 2 and float(err) < 0.01, float(err)
    est.save(tmp_path / 'psf.json')
    assert json.load(open(tmp_path / 'psf.json'))['type'] == 'vectorial'


def test_cli_localize(tmp_path):
    from torch_psf.__main__ import main
    psf = GaussianPSF(11, sigma_px=1.3, device=DEV)
    save_psf(tmp_path / 'g.json', psf)
    g = torch.Generator(device=DEV).manual_seed(2)
    movie, truth = simulate_movie(psf.render, 11, T=50, H=48, W=48, nsites=10, p_on=0.1, mean_on=2.0, bg=10.0,
                                  photons=3000.0, z_range=(0.0, 0.0), margin=8, generator=g, device=DEV)
    tifffile.imwrite(tmp_path / 'm.tif', (movie.numpy() / 0.5 + 100).astype(np.uint16), photometric='minisblack')
    assert main(['localize', str(tmp_path / 'm.tif'), '--psf', str(tmp_path / 'g.json'), '--gain', '0.5', '--offset',
                 '100', '-o', str(tmp_path / 'locs.csv'), '--device', DEV]) == 0
    L = np.genfromtxt(tmp_path / 'locs.csv', delimiter=',', names=True)
    assert len(L) > 0.5 * len(truth) and abs(np.median(L['photons']) / 3000 - 1) < 0.15
