"""Re-tabulating a spline on another knot grid from its own interpolant: equal to a direct build from the optical
model up to rounding when the new knots are a subset of the old ones."""
import pytest
import torch

from torch_psf import FourPiPSF, PSF, SplinePSF, VectorialPSF, to_spline

GRID = dict(dz_um=.01, xy_limit=1., zpad=4)
POS = torch.rand(300, 3, generator=torch.Generator().manual_seed(3)) * torch.tensor([1.6, 1.6, .1]) \
    + torch.tensor([-.8, -.8, 0.])


def optics(**kw):
    return VectorialPSF(7, npupil=16, oversample=2, z_range=(0., .10), device='cpu', focus_um=.05, **kw) \
        .set_zernike({6: .07})


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize('kernel,value', [('bspline', 'density'), ('catmull', 'density'),
                                          ('catmull', 'pixel'), ('bspline', 'pixel')])
def test_subset_grid_matches_a_direct_build(kernel, value):
    v = optics()
    s4 = to_spline(v, oversample=4, kernel=kernel, value=value, **GRID)
    s2 = to_spline(v, oversample=2, kernel=kernel, value=value, **GRID)
    r = s4.to_spline(oversample=2)
    assert torch.equal(to_spline(s4, oversample=2).knots, r.knots)
    assert r.knots.shape == s2.knots.shape and (r.kernel, r.value, r.xmax) == (s2.kernel, s2.value, s2.xmax)
    assert rel(r.knots, s2.knots) < 1e-6
    for a, b in zip(r.render(POS, grad=True), s2.render(POS, grad=True)):
        assert rel(a, b) < 1e-5
    assert torch.equal(s4.retabulated().knots, s4.knots) or rel(s4.retabulated().knots, s4.knots) < 1e-6


def test_off_grid_resampling_adds_only_interpolation_error():
    v = optics()
    s4 = to_spline(v, oversample=4, kernel='bspline', value='density', **GRID)
    for change in (dict(oversample=3), dict(dz_um=.02), dict(oversample=2, dz_um=.015, zpad=2)):
        r = s4.retabulated(**change)
        direct = to_spline(v, oversample=r.S, kernel='bspline', value='density', dz_um=r.hz, zpad=r.zpad,
                           xy_limit=GRID['xy_limit'])
        assert r.knots.shape == direct.knots.shape and r.zpad == change.get('zpad', 4 * .01 / r.hz // 1)
        # the extra error is the old spline's interpolation error at the new knots: small against the
        # 2.7e-4 difference between the S=2 and S=4 tables themselves
        assert rel(r.render(POS), direct.render(POS)) < 2e-4, change


def test_kernel_change_and_refusals():
    v = optics()
    s4 = to_spline(v, oversample=4, kernel='bspline', value='density', **GRID)
    cr = to_spline(v, oversample=2, kernel='catmull', value='density', **GRID)
    assert rel(s4.retabulated(oversample=2, kernel='catmull').knots, cr.knots) < 1e-6
    with pytest.raises(ValueError, match='optical model'):
        s4.retabulated(value='pixel')
    with pytest.raises(ValueError, match='beyond the stored spline'):
        s4.retabulated(z_range=(-.2, .1))


@pytest.mark.parametrize('components', [None, 'dipole'])
def test_per_angle_tables(components):
    models = [optics(), optics()]
    for m, M in zip(models, torch.diag_embed(torch.tensor([[.6, .3, .1], [.2, .5, .3]]))):
        m.dipM = M
    G = dict(kernel='bspline', value='density', components=components, **GRID)
    s4 = to_spline(models, oversample=4, **G)
    s2 = to_spline(models, oversample=2, **G)
    assert s4.n_angles == 2
    r = s4.retabulated(oversample=2)
    assert r.axes == s2.axes and torch.equal(r.table_map, s2.table_map)
    if components:
        torch.testing.assert_close(r.component_weights, s2.component_weights)
    assert rel(r.knots, s2.knots) < 1e-6
    angle = torch.tensor([1, 0] * 150)
    assert rel(r.render(POS, angle=angle), s2.render(POS, angle=angle)) < 1e-5


def test_fourpi_component_tables():
    base = FourPiPSF(7, npupil=16, device='cpu', pol=(0, 1), z_range=(0., .1))
    base.top.set_zernike({6: .05})
    G = dict(kernel='bspline', value='density', components='dipole', **GRID)
    s4 = base.to_spline(oversample=4, **G)
    s2 = base.to_spline(oversample=2, **G)
    r = s4.retabulated(oversample=2)
    assert r.table.axes == s2.table.axes and r.S == 2
    assert rel(r.knots, s2.knots) < 1e-6
    phase = torch.rand(len(POS)) * 6.28
    assert rel(r.render(POS, phi=phase), s2.render(POS, phi=phase)) < 1e-5


def test_loaded_psfzip(tmp_path):
    v = optics()
    s4 = to_spline(v, oversample=4, kernel='bspline', value='density', **GRID)
    path = tmp_path / 'm.psf.zip'
    s4.save(path, source=v)
    loaded = PSF.load(path, device='cpu')
    assert isinstance(loaded, SplinePSF) and torch.equal(loaded.knots, s4.knots)
    src = SplinePSF.load_source(path, device='cpu')
    direct = src.to_spline(oversample=2, kernel='bspline', value='density', **GRID)
    r = loaded.to_spline(oversample=2)
    assert rel(r.knots, direct.knots) < 1e-5
    out = tmp_path / 'm2.psf.zip'
    r.save(out, source=src)
    torch.testing.assert_close(PSF.load(out, device='cpu').render(POS), r.render(POS), rtol=0, atol=0)
