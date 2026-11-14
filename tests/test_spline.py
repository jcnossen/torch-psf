"""CubicSplinePSF: legacy reproduction, accuracy against the vectorial model, derivatives, knot gradient, z-stacks."""
import os
import numpy as np
import torch
from conftest import DEV, legacy_reference
from torch_psf import VectorialPSF, CubicSplinePSF, GaussianPSF, AstigmaticGaussianPSF


def test_reproduces_legacy_spline():
    r = {k: torch.tensor(v) for k, v in np.load(legacy_reference()).items()}
    s = CubicSplinePSF(13, (0.0, 0.8), oversample=2, dz_um=0.05, zpad=2, xy_limit=3.0, device='cpu')
    big = VectorialPSF(s.Pl, pixel_um=0.108, wavelength_um=0.68, npupil=48, oversample=2, device='cpu', focus_um=0.4)
    with torch.no_grad():
        big.zcoef.copy_(r['zcoef'])
    s.knots = s.knots_from_vectorial(big)
    out = s.render(r['pos'], grad=True)
    for a, k in zip(out, ('spline_img', 'spline_dx', 'spline_dy', 'spline_dz')):
        assert torch.allclose(a, r[k], rtol=0, atol=1e-6 * float(r[k].abs().max())), k


def test_spline_matches_vectorial():
    vec = VectorialPSF(15, npupil=64, oversample=3, device=DEV, focus_um=0.4, z_range=(0.0, 0.8))
    vec.set_zernike({6: 0.1, 11: 0.03})
    spl = vec.to_spline(15, oversample=3, dz_um=0.02)
    g = torch.Generator(device=DEV).manual_seed(1)
    pos = torch.stack([torch.rand(50, generator=g, device=DEV) - 0.5, torch.rand(50, generator=g, device=DEV) - 0.5,
                       0.05 + 0.7 * torch.rand(50, generator=g, device=DEV)], 1)
    with torch.no_grad():
        a, b = spl.render(pos), vec.render(pos)
    err = ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)).median()
    assert float(err) < 3e-3


def test_spline_derivatives():
    vec = VectorialPSF(13, npupil=48, oversample=2, device=DEV, focus_um=0.3, z_range=(0.0, 0.6))
    vec.set_zernike({6: 0.1})
    spl = vec.to_spline(13, dz_um=0.03)
    pos = torch.tensor([[0.13, -0.27, 0.31], [-0.4, 0.2, 0.12]], device=DEV)
    out = spl.render(pos, grad=True)
    for k in range(3):
        e = 1e-3
        p1, p2 = pos.clone(), pos.clone()
        p1[:, k] += e; p2[:, k] -= e
        fd = (spl.render(p1) - spl.render(p2)) / (2 * e)
        assert float((fd - out[k + 1]).abs().max() / out[k + 1].abs().max()) < 5e-3


def test_knot_gradient_matches_reference_indexing():
    spl = CubicSplinePSF(11, (0.0, 0.5), device=DEV)
    spl.knots = torch.rand_like(spl.knots)
    pos = torch.rand(20, 3, device=DEV) * torch.tensor([1.0, 1.0, 0.5], device=DEV) - torch.tensor([0.5, 0.5, 0], device=DEV)
    w = torch.randn(20, 11, 11, device=DEV)
    K = spl.knots.clone().requires_grad_(True)
    (spl.render(pos, knots=K) * w).sum().backward()
    # the same with the knot block read by advanced indexing (a different backward kernel)
    K2 = spl.knots.clone().requires_grad_(True)
    orig = torch.Tensor.index_select
    try:
        torch.Tensor.index_select = lambda self, dim, idx: self[idx]
        (spl.render(pos, knots=K2) * w).sum().backward()
    finally:
        torch.Tensor.index_select = orig
    assert torch.allclose(K.grad, K2.grad, atol=1e-5)


def test_from_zstack_reproduces_the_stack_at_grid_points():
    g = AstigmaticGaussianPSF(21, sigma_px=1.5, gamma_um=0.2, d_um=0.4, z_range=(-0.5, 0.5), device='cpu')
    zs = torch.linspace(-0.5, 0.5, 21)
    stack = g.render(torch.stack([torch.zeros(21), torch.zeros(21), zs], 1))
    spl = CubicSplinePSF.from_zstack(stack, z0_um=-0.5, dz_um=0.05, roisize=11)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.1]])
    a = spl.render(pos)
    c = (21 - 11) // 2
    assert torch.allclose(a[0], stack[10, c:c + 11, c:c + 11], atol=1e-6)
    assert torch.allclose(a[1], stack[12, c + 1:c + 12, c - 1:c + 10], atol=1e-6)


def test_even_roi_and_wrapped_knots():
    vec = VectorialPSF(15, npupil=48, oversample=2, device=DEV, focus_um=0.3, z_range=(0.0, 0.6)).set_zernike({6: 0.1})
    odd = vec.to_spline(15, dz_um=0.03)
    even = vec.to_spline(16, dz_um=0.03)
    pos = torch.tensor([[0.0, 0.0, 0.3], [0.1, -0.2, 0.2]], device=DEV)
    a = odd.render(pos)
    # the 16 px ROI is centred between pixels: an offset of +0.5 px in the even ROI = the same emitter in the odd one
    b = even.render(pos + torch.tensor([0.5, 0.5, 0.0], device=DEV))
    assert torch.allclose(b[:, 1:16, 1:16], a, atol=1e-6)
    # wrap the stored knots with an explicit origin and z origin: identical images
    w = CubicSplinePSF(15, odd.z_range, oversample=odd.S, dz_um=odd.hz, knots=odd.knots, knot_u0_px=odd.u0,
                           knot_z0_um=odd.z0, device=DEV)
    assert torch.equal(w.render(pos), a) and w.xmax == odd.xmax


def test_clamped_derivative_outside_the_z_range():
    vec = VectorialPSF(13, npupil=32, oversample=2, device=DEV, focus_um=0.3, z_range=(0.0, 0.6)).set_zernike({6: 0.1})
    spl = vec.to_spline(13, dz_um=0.03, zpad=0)
    lo = spl.z_limits[0]
    out = spl.render(torch.tensor([[0.0, 0.0, lo - 0.2], [0.0, 0.0, lo]], device=DEV), grad=True)
    assert torch.allclose(out[0][0], out[0][1]) and torch.allclose(out[3][0], out[3][1])     # value and d/dz clamped
    assert float(out[3][0].abs().max()) > 0


def test_to_spline_of_gaussians():
    g2 = GaussianPSF(11, sigma_px=1.3, device=DEV)
    s2 = g2.to_spline(11, z_range=(0.0, 0.2))
    pos = torch.tensor([[0.2, -0.3, 0.1]], device=DEV)
    err = lambda s, g: float((s.render(pos) - g.render(pos)).abs().max() / g.render(pos).max())
    assert err(s2, g2) < 0.02
    ga = AstigmaticGaussianPSF(11, sigma_px=1.3, gamma_um=0.2, d_um=0.4, z_range=(-0.4, 0.4), device=DEV)
    sa = ga.to_spline(11, dz_um=0.02)
    assert err(sa, ga) < 0.02
