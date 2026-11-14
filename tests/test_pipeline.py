"""Entry points: localize a simulated movie; bead estimation from simulated stacks; the command line."""
import json
import numpy as np
import tifffile
import torch
from conftest import DEV, needs_gpu
from torch_psf import GaussianPSF, VectorialPSF, ZStackEstimator
from torch_psf.simulate import simulate_movie


def test_localize_movie():
    psf = GaussianPSF(11, sigma_px=1.3, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(1)
    movie, truth = simulate_movie(psf.render, 11, T=100, H=64, W=64, nsites=30, p_on=0.05, mean_on=2.0, bg=10.0,
                                  photons=3000.0, photons_logsd=0.1, z_range=(0.0, 0.0), margin=8, generator=g, device=DEV)
    L = psf.localize(movie.numpy(), isolation=6)
    assert len(L) > 0.5 * len(truth)
    tr = truth.cpu().numpy()
    err = []
    for r in L:
        t = tr[tr[:, 0] == r['frame']]
        dd = np.hypot(t[:, 1] - r['x'], t[:, 2] - r['y'])
        if len(dd) and dd.min() < 1:
            err.append(dd.min())
    assert len(err) > 0.9 * len(L) and np.median(err) < 3 * np.median(L['crlb_x'])


def bead_tif(tmp_path, vec, nk=31, H=64, dz=0.03):
    focus = dz * (torch.arange(nk, device=DEV) - (nk - 1) / 2)
    stack = torch.full((nk, H, H), 20.0, device=DEV)
    for cx, cy in ((20, 22), (44, 40)):
        img = vec.forward_batched(torch.tensor([[0.2, -0.1, 0.0]] * nk, device=DEV), focus)
        stack[:, cy - 10:cy + 11, cx - 10:cx + 11] += 30000 * img
    stack = torch.poisson(stack).cpu().numpy().astype(np.uint16)
    tifffile.imwrite(tmp_path / 'beads.tif', stack)
    return str(tmp_path / 'beads.tif')


@needs_gpu
def test_estimate_psf_beads(tmp_path):
    vec = VectorialPSF(21, npupil=48, oversample=2, device=DEV).set_zernike({6: 0.08, 11: 0.03})
    bead_tif(tmp_path, vec)
    est = ZStackEstimator('vectorial', optics=dict(pixel_um=0.10833, NA=1.49, wavelength_um=0.68), roisize=21,
                          npupil=48, oversample_pupil=2, iters=250, device=DEV)
    psf = est.estimate([str(tmp_path / 'beads.tif')], dz_um=0.03)
    err = (psf.zcoef.detach() - vec.zcoef.detach()).pow(2).mean().sqrt()
    assert est.report['n_beads'] == 2 and float(err) < 0.01, float(err)
    psf.save(tmp_path / 'psf.json')
    assert json.load(open(tmp_path / 'psf.json'))['type'] == 'vectorial'


def test_cli_localize(tmp_path):
    from torch_psf.__main__ import main
    psf = GaussianPSF(11, sigma_px=1.3, device=DEV)
    psf.save(tmp_path / 'g.json')
    g = torch.Generator(device=DEV).manual_seed(2)
    movie, truth = simulate_movie(psf.render, 11, T=50, H=48, W=48, nsites=10, p_on=0.1, mean_on=2.0, bg=10.0,
                                  photons=3000.0, z_range=(0.0, 0.0), margin=8, generator=g, device=DEV)
    tifffile.imwrite(tmp_path / 'm.tif', (movie.numpy() / 0.5 + 100).astype(np.uint16), photometric='minisblack')
    assert main(['localize', str(tmp_path / 'm.tif'), '--psf', str(tmp_path / 'g.json'), '--gain', '0.5', '--offset',
                 '100', '-o', str(tmp_path / 'locs.csv'), '--device', DEV]) == 0
    L = np.genfromtxt(tmp_path / 'locs.csv', delimiter=',', names=True)
    assert len(L) > 0.5 * len(truth) and abs(np.median(L['photons']) / 3000 - 1) < 0.15


@needs_gpu
def test_zstack_spline_and_astig_gauss(tmp_path):
    from torch_psf import ConvCubicSplinePSF, AstigmaticGaussianPSF
    vec = VectorialPSF(21, npupil=48, oversample=2, device=DEV).set_zernike({6: 0.12})
    f = bead_tif(tmp_path, vec)
    opt = dict(pixel_um=0.10833, NA=1.49, wavelength_um=0.68)
    est = ZStackEstimator('spline', optics=opt, roisize=21, npupil=48, oversample_pupil=2, iters=150, device=DEV)
    spl = est.estimate([f], dz_um=0.03)
    assert isinstance(spl, ConvCubicSplinePSF) and est.report['spline']['change_vs_vectorial'] < 0.5
    est = ZStackEstimator('astig_gauss', optics=opt, roisize=21, iters=400, device=DEV)
    ag = est.estimate([f], dz_um=0.03)
    assert isinstance(ag, AstigmaticGaussianPSF) and abs(ag.gamma) > 0.05      # astigmatism -> separated foci
    th = torch.tensor([[0.0, 0.0, 0.1, 5000.0, 10.0]], device=DEV)
    assert torch.isfinite(ag.crlb(th)).all() and torch.isfinite(spl.crlb(th)).all()


@needs_gpu
def test_insitu_estimator_with_refinement():
    from torch_psf import InSituEstimator
    from torch_psf.simulate import simulate_reference_movie
    r = simulate_reference_movie('astigmatic', T=1500, H=96, W=96, emitters_per_frame=6, photons=4000, seed=1,
                                 device=DEV)
    start = VectorialPSF(15, npupil=48, oversample=2, device=DEV, focus_um=0.4).set_zernike({6: 0.08})
    est = InSituEstimator(start=start, camera=r['camera'], z_range=(0.0, 0.8), iters=60, refine=True,
                          refine_outer=4, isolation=12, device=DEV)
    psf = est.estimate(r['movie_adu'])
    rep = est.report
    assert rep['chosen'] in ('VectorialPSF', 'ConvCubicSplinePSF') and 'refine' in rep and rep['warnings']
    assert rep['heldout']['vectorial']['nll_per_px'] < rep['heldout']['start']['nll_per_px']
    vec = r['psf']
    fitted = est.data['spline']
    assert abs(rep['z_scale'] - 1) < 0.5 and fitted.n_dims == 3 and psf is not None and vec is not None
