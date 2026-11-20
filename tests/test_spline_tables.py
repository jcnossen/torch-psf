"""Indexed spline contraction, coefficient adjoints and legacy multi-response blocks."""
import pytest
import torch
import torch.nn.functional as F

from torch_psf import PSF, SplinePSF
from torch_psf.spline import _kernel, _qkernel


def scalar_reference(s, volume, pos):
    """One legacy scalar-table convolution. Slice each block independently of the indexed evaluator."""
    values = []
    for x, y, z in pos:
        t = torch.stack(((z - s.z0) / s.hz, s.t0 - y * s.S, s.t0 - x * s.S))
        iz, iy, ix = (t.floor().long() - 1).tolist()
        fz, fy, fx = t - t.floor()
        def axis(fr, taps, integral):
            shift = fr + 1 - torch.arange(taps)
            if integral:
                return (_qkernel(shift + s.S, s.kernel) - _qkernel(shift, s.kernel)) / s.S
            return _kernel(shift, s.kernel)
        wz = axis(fz, 4, False)
        wy = axis(fy, s.taps, s.integrated)
        wx = axis(fx, s.taps, s.integrated)
        filt = torch.einsum('z,y,x->zyx', wz, wy, wx)
        block = volume[iz:iz + 4, iy:iy + s.Lw, ix:ix + s.Lw]
        values.append(F.conv2d(block[None], filt[None], stride=s.S)[0, 0])
    return torch.stack(values)


@pytest.mark.parametrize('kernel,integrated', [('catmull', False), ('catmull', True),
                                           ('bspline', False), ('bspline', True)])
def test_indexed_bank_and_adjoint(kernel, integrated):
    # Reordered and repeated entries exercise the table offset and scatter-add into a shared volume.
    ids = torch.tensor([[2, 0], [1, 2]])
    knots = torch.randn(3, 8, 26, 26, dtype=torch.float64, requires_grad=True)
    s = SplinePSF(5, (0, .15), 2, .05, 2, device='cpu', dtype=knots.dtype, knots=knots,
                  axes=('tile', 'angle'), spline_index=ids, kernel=kernel, integrated=integrated)
    assert s.knots is knots
    pos = torch.tensor([[.13, -.17, .073], [-.25, .41, .031]], dtype=knots.dtype, requires_grad=True)
    a, b = torch.randn(2, 2, 2, dtype=knots.dtype)
    a.requires_grad_()
    kw = dict(weights={'tile': a, 'angle': b})
    images = s.render_splines(pos, grad=True, **kw)
    scalar = [[scalar_reference(s, bank, pos) for bank in row] for row in knots[ids]]
    ref = sum(im * (a[:, i] * b[:, j])[:, None, None]
              for i, row in enumerate(scalar) for j, im in enumerate(row))
    torch.testing.assert_close(images[0], ref)
    sensitivity = torch.randn_like(ref)
    ref_k, ref_pos = torch.autograd.grad((ref * sensitivity).sum(), (knots, pos))
    got_k, got_w = torch.autograd.grad((images[0] * sensitivity).sum(), (knots, a))
    torch.testing.assert_close(got_k, ref_k)
    for i, di in enumerate(images[1:]):
        torch.testing.assert_close((di * sensitivity).sum((1, 2)), ref_pos[:, i])
    delta = torch.randn_like(a)
    eps = 1e-5
    def loss(w):
        return (s.render_splines(pos, weights={'tile': w, 'angle': b}) * sensitivity).sum()
    fd = (loss(a + eps * delta) - loss(a - eps * delta)) / (2 * eps)
    torch.testing.assert_close(fd, (got_w * delta).sum())
    s.backend = 'conv'
    for got, expected in zip(s.render_splines(pos, grad=True, **kw), images):
        torch.testing.assert_close(got, expected)


def test_stored_angle_defaults_and_dipole_tables(tmp_path):
    # The 0.3 layout can store calibrated fixed responses alongside six independent basis volumes per angle.
    g = SplinePSF(5, (0., .15), 2, .05, 2, 1., kernel='bspline', integrated=True, device='cpu')
    meta = g.to_dict()
    meta.update(n_angles=2, dipole_components=True)
    shape = (g.nz, g.nv, g.nu)
    fixed = torch.randn(2, *shape)
    dipole = torch.randn(2, 6, *shape)
    tables = SplinePSF._from_file({'integrated_spline': meta}, dict(coefs_angle=fixed, dipole_coefs_angle=dipole),
                                  device='cpu')
    assert tables.knots.shape == (14, *shape)
    path = tmp_path / 'old.json'
    tables.save(path)
    pos = torch.tensor([[.2, -.11, .067]])
    M = torch.tensor([[.2, .12, -.1], [.12, .5, .05], [-.1, .05, .3]])
    m6 = M[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    for k in (None, 0, 1):
        s = PSF.load(path, angle=k, device='cpu')
        d = dipole.mean(0) if k is None else dipole[k]
        f = fixed.mean(0) if k is None else fixed[k]
        torch.testing.assert_close(s.render(pos), scalar_reference(g, f, pos))
        torch.testing.assert_close(s.render(pos, dipM=M), sum(w * scalar_reference(g, v, pos) for w, v in zip(m6, d)))
        assert s.resized(3).knots is s.knots


@pytest.mark.parametrize('kernel,integrated', [('catmull', False), ('bspline', True)])
def test_stage_scan_knot_vjp(kernel, integrated):
    from torch_psf import VectorialPSF
    from torch_psf.composition import ZMode

    spline = SplinePSF(5, (0., .06), 2, .02, 2, 1., device='cpu', kernel=kernel, integrated=integrated)
    source = VectorialPSF(spline.Pl, npupil=16, oversample=2, device='cpu')
    scan = ZMode(source, focus=.11, scale=.9, height=.03).coordinates
    knots = spline.sample(source, scan, chunk=2, differentiable=True)
    adjoint = torch.randn_like(knots)
    reference, = torch.autograd.grad(knots, source.zcoef, adjoint)
    got = spline.vjp_vectorial(source, scan, adjoint[None], chunk=2)['zcoef']
    torch.testing.assert_close(got, reference, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('kernel,integrated,backend', [('catmull', False, 'sep'), ('bspline', True, 'conv')])
def test_explicit_backend_round_trip(tmp_path, kernel, integrated, backend):
    s = SplinePSF(5, (0., .15), 2, .05, 2, 1., device='cpu', kernel=kernel, integrated=integrated, backend=backend)
    s.knots.normal_()
    path = tmp_path / 'table.json'
    s.save(path)
    loaded = PSF.load(path, device='cpu')
    assert loaded.backend == backend
    pos = torch.tensor([[.21, -.32, .09], [.041, .19, .07]])
    for got, expected in zip(loaded.render(pos, grad=True), s.render(pos, grad=True)):
        assert torch.equal(got, expected)


def test_phase_and_dipole_weight_gradients():
    from torch_psf import FourPiSplinePSF

    volumes = torch.randn(3, 2, 6, 8, 26, 26)
    phase_view = FourPiSplinePSF(5, (0., .15), volumes, oversample=2, dz_um=.05, zpad=2, device='cpu')
    phase_view.to(dtype=torch.float64)
    pos = torch.tensor([[.25, -.13, .071], [-.23, .31, .031]], dtype=torch.float64)
    phase = torch.tensor([.43, -.31], requires_grad=True)
    M = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    phase_view.knots.requires_grad_()
    integrated, *_, dphi = phase_view.render(pos, phi=phase, dipM=M, grad=True)
    probe = torch.randn_like(integrated)
    dM, phase_grad, adjoint = torch.autograd.grad((integrated * probe).sum(), (M, phase, phase_view.knots))
    torch.testing.assert_close((dphi * probe).sum((1, 2, 3)), phase_grad.double(), atol=2e-6, rtol=2e-6)
    direction = torch.randn_like(M)
    eps = 1e-5
    plus = phase_view.render(pos, phi=phase, dipM=M + eps * direction)
    minus = phase_view.render(pos, phi=phase, dipM=M - eps * direction)
    torch.testing.assert_close(((plus - minus) * probe).sum() / (2 * eps), (dM * direction).sum())
    assert torch.all(adjoint.flatten(1).abs().sum(1) > 0)


def test_default_table_rule():
    """Without kernel / integrated: optical models get the integrated B-spline at 2 knots/px, pixel-only models
    (Gaussians) Catmull-Rom pixel values at 2 knots/px."""
    from torch_psf import AstigmaticGaussianPSF, VectorialPSF
    from torch_psf.spline import DEFAULT_PIXEL_SPLINE, DEFAULT_SPLINE
    assert DEFAULT_SPLINE == ('bspline', True, 2) and DEFAULT_PIXEL_SPLINE == ('catmull', False, 2)
    vec = VectorialPSF(9, npupil=16, oversample=2, z_range=(0.0, 0.1), device='cpu')
    s = vec.to_spline(9, dz_um=0.05)
    assert (s.kernel, s.integrated, s.S) == DEFAULT_SPLINE
    g = AstigmaticGaussianPSF(9, (1.1, 1.3), pixel_um=0.108, device='cpu')
    sg = g.to_spline(9, dz_um=0.05)
    assert (sg.kernel, sg.integrated, sg.S) == DEFAULT_PIXEL_SPLINE
