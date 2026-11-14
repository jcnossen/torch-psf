"""The MLE fitter reaches the Cramer-Rao lower bound on simulated isolated emitters (2D and astigmatic 3D)."""
import torch
from conftest import DEV
from torch_psf import GaussianPSF, VectorialPSF, CubicSplinePSF
from torch_psf.fit import localize, crlb
from torch_psf.fit import lm_poisson, model_and_jacobian, fisher
from torch_psf.simulate import simulate_emitters


def check_crlb(psf, n=3000, photons=2000.0, bg=10.0, z_range=(0.0, 0.0), dims=(0, 1, 3, 4), tol=0.12, var=None):
    g = torch.Generator(device=DEV).manual_seed(5)
    sim = simulate_emitters(psf.render, n, bg=bg, photons=photons, photons_logsd=0.0, z_range=z_range, generator=g,
                            device=DEV)
    d = sim['d']
    if var is not None:      # sCMOS: add read noise, shift as in the model
        d = d + var ** 0.5 * torch.randn(d.shape, generator=g, device=DEV)
    th, _ = localize(psf, d, var=var)
    th0 = sim['th']
    cr = crlb(psf, th0, var=var)
    for k in dims:
        sd = float((th[:, k] - th0[:, k]).std())
        cb = float(cr[:, k].pow(2).mean().sqrt())      # rms of the per-emitter bounds (the CRLB varies with z)
        assert abs(sd / cb - 1) < tol, (k, sd, cb)
        assert abs(float((th[:, k] - th0[:, k]).mean())) < 0.2 * cb, (k, 'bias')


def test_crlb_2d_gaussian():
    check_crlb(GaussianPSF(11, sigma_px=1.3, device=DEV))


def test_crlb_2d_gaussian_scmos():
    check_crlb(GaussianPSF(11, sigma_px=1.3, device=DEV), var=4.0)


def test_crlb_3d_astigmatic_spline():
    vec = VectorialPSF(15, npupil=48, oversample=2, device=DEV, focus_um=0.4, z_range=(0.0, 0.8))
    vec.set_zernike({6: 0.1})
    spl = vec.to_spline(15, dz_um=0.04)
    check_crlb(spl, photons=3000.0, z_range=(0.1, 0.7), dims=(0, 1, 2, 3), tol=0.15)


def test_lm_poisson_generic_and_fisher():
    psf = GaussianPSF(9, sigma_px=1.2, device=DEV)
    th0 = torch.tensor([[0.3, -0.2, 0.0, 1000.0, 5.0]], device=DEV)
    mu, _ = model_and_jacobian(psf, th0)
    start = th0.clone(); start[:, :2] = 0; start[:, 3] = 800
    th, nll = lm_poisson(lambda t: model_and_jacobian(psf, t), mu, start, fixed=(2,))
    assert torch.allclose(th, th0, atol=1e-3 * 1000)
    assert float((th[:, :2] - th0[:, :2]).abs().max()) < 1e-3
    F = fisher(psf, th0, params=(0, 1, 3, 4))
    assert F.shape == (1, 4, 4) and bool((torch.linalg.eigvalsh(F.double()) > 0).all())
