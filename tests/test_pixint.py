"""torch_psf.pixint (prototype): exact pixel integration of the vectorial PSF and of a cubic spline."""
import math
import pytest
import torch
from conftest import DEV, needs_gpu
from torch_psf import VectorialPSF, CubicSplinePSF
from torch_psf.pixint import (ExactVectorialPSF, PixelSplinePSF, build_pixel_spline, bspline_prefilter, _kernel,
                              _dkernel, _qkernel)


def _vec(dev='cpu', R=9, blur=None):
    v = VectorialPSF(R, npupil=32, oversample=2, device=dev, focus_um=0.2, z_range=(0.0, 0.3), wavelength_um=0.7)
    v.set_zernike({6: 0.05, 11: 0.03})
    v.blur = blur
    return v


@pytest.mark.parametrize('kind', ['catmull', 'bspline'])
def test_kernels(kind):
    s = torch.linspace(-3, 3, 6001, dtype=torch.float64)
    K, dK, Q = _kernel(s, kind), _dkernel(s, kind), _qkernel(s, kind)
    assert float(Q[0]) == 0.0 and abs(float(Q[-1]) - 1) < 1e-12
    assert float((torch.gradient(Q, spacing=(s,))[0] - K).abs().max()) < 1e-5
    assert float((torch.gradient(K, spacing=(s,))[0] - dK).abs().max()) < 1e-3
    # partition of unity of the integer translates
    t = torch.rand(50, dtype=torch.float64)
    assert float((sum(_kernel(t - k, kind) for k in range(-2, 4)) - 1).abs().max()) < 1e-12


def test_bspline_prefilter_interpolates():
    x = torch.randn(12, 15, 15, dtype=torch.float64)
    c = bspline_prefilter(x)
    # B-spline value at the knots = (c[k-1] + 4 c[k] + c[k+1]) / 6 per axis (interior)
    f = lambda a, d: (a.narrow(d, 0, a.shape[d] - 2) + 4 * a.narrow(d, 1, a.shape[d] - 2) + a.narrow(d, 2, a.shape[d] - 2)) / 6
    v = f(f(f(c, 0), 1), 2)
    assert float((v - x[1:-1, 1:-1, 1:-1]).abs().max()) < 1e-10


def test_exact_vectorial_agree():
    """spectral (uniform grid + exact sinc) and Gauss-Legendre agree; the midpoint rule converges to them as 1 / s^2"""
    vec = _vec()
    pos = torch.tensor([[0.1, -0.3, 0.05], [0.45, 0.2, 0.2]])
    with torch.no_grad():
        a = ExactVectorialPSF(vec, 'spectral', os=3, pad=4).render(pos)
        b = ExactVectorialPSF(vec, 'gauss', os=6).render(pos)
        pk = a.max()
        assert float((a - b).abs().max() / pk) < 2e-6
        e = [float((ExactVectorialPSF(vec, 'box', os=s).render(pos) - a).abs().max() / pk) for s in (2, 4)]
        assert 3.0 < e[0] / e[1] < 5.0                      # second order
        # 'box' reproduces VectorialPSF
        assert float((ExactVectorialPSF(vec, 'box', os=2).render(pos) - vec.render(pos)).abs().max()) < 1e-7


def test_exact_vectorial_blur():
    vec = _vec(blur=torch.tensor([0.4, 0.2, -0.2]))
    pos = torch.tensor([[0.1, -0.3, 0.05]])
    with torch.no_grad():
        a = ExactVectorialPSF(vec, 'spectral', os=2, pad=3).render(pos)
        b = ExactVectorialPSF(vec, 'spectral', os=4, pad=6).render(pos)
        c = ExactVectorialPSF(vec, 'box', os=8).render(pos)    # midpoint rule, error ~ 1 / 64 of s = 1
    assert float((a - b).abs().max() / b.max()) < 2e-5
    assert float((c - b).abs().max() / b.max()) < 2e-3
    with pytest.raises(ValueError):
        ExactVectorialPSF(vec, 'gauss', os=3)


def _gauss_spline(S, kernel, integrate, sigma=1.1, R=9):
    geo = CubicSplinePSF(R, (0.0, 0.2), oversample=S, dz_um=0.05, zpad=2, xy_limit=2.0, device='cpu')
    u = geo.u0 + torch.arange(geo.nu, dtype=torch.float64) / S
    if integrate:       # density per px^2
        g1 = torch.exp(-u ** 2 / (2 * sigma ** 2)) / math.sqrt(2 * math.pi * sigma ** 2)
    else:               # pixel integral of a pixel centred at -u: knot value g(u) = int over [u - 1/2, u + 1/2]
        Phi = lambda t: 0.5 * (1 + torch.erf(t / (sigma * math.sqrt(2))))
        g1 = Phi(u + 0.5) - Phi(u - 0.5)
    plane = g1[:, None] * g1[None, :]
    samples = plane[None].expand(geo.nz, -1, -1).float().contiguous()
    return PixelSplinePSF(geo, samples, kernel=kernel, integrate=integrate)


def _gauss_pixels(pos, sigma=1.1, R=9):
    o = torch.arange(R, dtype=torch.float64) - (R - 1) / 2
    Phi = lambda t: 0.5 * (1 + torch.erf(t / (sigma * math.sqrt(2))))
    ax = lambda c: Phi(o[None] + 0.5 - c[:, None]) - Phi(o[None] - 0.5 - c[:, None])
    return (ax(pos[:, 1].double())[:, :, None] * ax(pos[:, 0].double())[:, None, :]).float()


@pytest.mark.parametrize('S', [2, 3, 4])
def test_integrated_spline_gaussian(S):
    """integrated spline of a sampled Gaussian density vs the analytic pixel integral (erf)"""
    pos = torch.rand(40, 3) * torch.tensor([1.0, 1.0, 0.2]) - torch.tensor([0.5, 0.5, 0.0])
    ref = _gauss_pixels(pos)
    tol = {2: 3e-3, 3: 6e-4, 4: 2e-4}[S]
    for kernel, f in (('catmull', 1.0), ('bspline', 0.25)):
        spl = _gauss_spline(S, kernel, True)
        err = float((spl.render(pos) - ref).abs().max() / ref.max())
        assert err < tol * f * (4 if kernel == 'catmull' else 1), (kernel, err)


def test_point_spline_matches_cubicsplinepsf():
    spl = _gauss_spline(3, 'catmull', False)
    cs = CubicSplinePSF(9, (0.0, 0.2), oversample=3, dz_um=0.05, zpad=2, xy_limit=spl.xmax, device='cpu',
                        knots=spl.samples, knot_u0_px=spl.geo.u0, knot_z0_um=spl.geo.z0)
    pos = torch.rand(30, 3) * torch.tensor([1.0, 1.0, 0.2]) - torch.tensor([0.5, 0.5, 0.0])
    a, b = spl.render(pos, grad=True), cs.render(pos, grad=True)
    for x, y in zip(a, b):
        assert float((x - y).abs().max()) < 1e-5          # d/dz: float32 cancellation times 1 / hz


@pytest.mark.parametrize('integrate', [True, False])
def test_derivatives_and_backends(integrate):
    spl = _gauss_spline(3, 'bspline', integrate)
    spl.knots = spl.knots * (1 + 0.3 * torch.rand_like(spl.knots))       # make it z dependent
    pos = torch.tensor([[0.13, -0.27, 0.11], [-0.4, 0.2, 0.07], [0.0, 0.0, 0.1]])
    out = spl.render(pos, grad=True)
    for k in range(3):
        e = 1e-3
        p1, p2 = pos.clone(), pos.clone()
        p1[:, k] += e; p2[:, k] -= e
        fd = (spl.render(p1) - spl.render(p2)) / (2 * e)
        assert float((fd - out[k + 1]).abs().max() / out[k + 1].abs().max()) < 5e-3
    spl.backend = 'conv'
    for x, y in zip(out, spl.render(pos, grad=True)):
        assert float((x - y).abs().max() / y.abs().max()) < 1e-5
    # linear in the knots: autograd gradient
    K = spl.knots.clone().requires_grad_(True)
    (spl.render(pos, knots=K) ** 2).sum().backward()
    assert K.grad is not None and float(K.grad.abs().sum()) > 0


@needs_gpu
def test_vectorial_integrated_spline_accuracy():
    """integrated B-spline (S = 3) of the vectorial point density vs the exact pixel integral"""
    torch.backends.cudnn.allow_tf32 = False
    vec = _vec(DEV, R=11)
    spl = build_pixel_spline(vec, 11, (0.0, 0.3), 3, dz_um=0.01, kernel='bspline', integrate=True)
    ref = ExactVectorialPSF(vec, 'spectral', os=3, pad=4)
    pos = torch.rand(100, 3, device=DEV) * torch.tensor([1.0, 1.0, 0.3], device=DEV) - torch.tensor([0.5, 0.5, 0], device=DEV)
    with torch.no_grad():
        a, b = spl.render(pos), ref.render(pos)
    assert float((a - b).abs().max() / b.max()) < 2e-4
