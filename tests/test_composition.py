"""Small checks of channel and field indexing and the optical compositions."""
from unittest.mock import patch

import pytest
import torch

from torch_psf import (ChannelsPSF, FieldPSF, FieldZernikes, FourPiPSF, Polychromatic, PSF,
                       VectorialPSF, ZMode, psfzip)

GRID = dict(oversample=2, dz_um=.02, xy_limit=1., zpad=2)
POS = torch.tensor([[.18, -.21, .032], [-.27, .11, .064]])
M = torch.diag_embed(torch.tensor([[.6, .3, .1], [.1, .2, .7]]))


def optics():
    return VectorialPSF(7, npupil=16, oversample=2, z_range=(0., .10), device='cpu', focus_um=.05)


def field():
    z = FieldZernikes([6], 1, norm=(10., 10., 10., 10.), coef=[[.035, .048]], device='cpu')
    return FieldPSF(optics().set_zernike({6: .07}), z)


@pytest.mark.parametrize('components', [None, 'dipole'])
def test_indexed_channels(components):
    v = optics()
    models = [v.to_spline(**GRID, components=components),
              v.set_zernike({6: .07}).to_spline(**GRID, components=components)]
    psf = ChannelsPSF(models, [[.17, -.3], [-.21, .14]], [.65, .35])
    pc = psf.channel_positions(POS)
    dipM = M + .02 if components else None
    reference = [m.render(pc[:, i], grad=True, dipM=dipM) for i, m in enumerate(models)]
    table = psf.indexed.table
    assert table.axes[0] == 'channel'
    table.knots.requires_grad_()
    with patch.object(table, 'evaluate_tables', wraps=table.evaluate_tables) as evaluate:
        actual = psf.render(POS, grad=True, dipM=dipM)
        evaluate.assert_called_once()
    for part, expected in zip(actual, zip(*reference)):
        torch.testing.assert_close(part, torch.stack(expected, 1) * psf.intensity_ratio[None, :, None, None])
    actual[0].sum().backward()
    assert torch.all(table.knots.grad.flatten(1).abs().sum(1) > 0)


def test_shared_optics_batch_with_focus():
    m = optics()
    psf = ChannelsPSF([m, m], [[.2, -.1], [-.1, .2]])
    focus = torch.tensor([[.05, -.08], [.01, -.11]])
    with patch.object(m, 'render', wraps=m.render) as render:
        images = psf.render(POS, focus=focus, dipM=M)
        render.assert_called_once()
    pc = psf.channel_positions(POS)
    for i in range(2):
        torch.testing.assert_close(images[:, i], m.render(pc[:, i], focus=focus[:, i], dipM=M))


def test_bilinear_field(tmp_path, monkeypatch):
    exact = field()
    m = exact.to_tiled(tile=10., blend='bilinear', components='dipole', **GRID)
    nearest = exact.to_tiled(tile=10., **GRID)
    xy = torch.tensor([[10., 10.], [12., 9.]])
    ids, weights = m.tile_weights(torch.tensor([[-100., -2.], [100., 200.], [10., 10.], [12., 9.]]))
    torch.testing.assert_close(weights.sum(1), torch.ones(4))
    assert ids[0, 0] == 0 and weights[0, 0] == 1
    assert ids[1].eq(3).all()
    for k, center in enumerate(m.centers):
        out = m.render(POS, xy=torch.tensor(center).expand(2, 2), dipM=M)
        torch.testing.assert_close(out, m.spline(k).render(POS, dipM=M))
    border = torch.tensor([[9.9999, 8.], [10.0001, 8.]])
    sides = m.render(POS[:1].expand(2, 3), xy=border)
    torch.testing.assert_close(sides[0], sides[1], rtol=0., atol=2e-6)
    target = exact.render(POS, xy=xy)
    assert (m.render(POS, xy=xy)[0]-target[0]).norm() < (nearest.render(POS, xy=xy)[0]-target[0]).norm()
    got = m.bind(xy).render(POS, True, dipM=M)
    basis = m.dipole_basis(POS, xy=xy)
    torch.testing.assert_close(got[0], (basis[:, :3] * M.diagonal(dim1=1, dim2=2)[:, :, None, None]).sum(1))
    for axis in range(3):
        step = torch.zeros_like(POS)
        eps = 1e-4 if axis == 2 else .001
        step[:, axis] = eps
        fd = (m.render(POS+step, xy=xy, dipM=M) - m.render(POS-step, xy=xy, dipM=M)) / (2*eps)
        torch.testing.assert_close(fd, got[axis+1], rtol=.015, atol=.0002)
    torch.testing.assert_close(m.at(*xy[0]).render(POS[:1], dipM=M[0]), got[0][:1])
    path = tmp_path / 'field.zip'
    psfzip.save_zip(path, m, zstack_planes=3)
    assert psfzip.read_zip(path)[0]['version'] == '0.4.0'
    loaded = PSF.load(path, device='cpu')
    assert loaded.blend == 'bilinear'
    torch.testing.assert_close(loaded.render(POS, xy=xy, dipM=M), got[0])
    # Also exercise the version feature in isolation from dipole components.
    assert exact.to_tiled(tile=10., blend='bilinear', **GRID).format_version() == (0, 4)
    monkeypatch.setattr(psfzip, 'READ_VERSIONS', ((0, 2), (0, 3)))
    with pytest.raises(ValueError, match='0.4.0'):
        PSF.load(path)


def test_spectrum_and_zmode():
    base = optics().set_zernike({6: .06})
    base.pupil_phase.copy_(base.phi * .12)
    spectrum = Polychromatic(base, [.64, .72], [.3, .7])
    m = ZMode(spectrum, height=.013, scale=.9)
    original = base.pupil_phase.clone()
    expected = 0.
    for wl, w in zip([.64, .72], [.3, .7]):
        mono = base.resized(base.roisize)
        mono.wavelength_um = wl
        scale = base.wavelength_um/wl
        mono.set_zernike(base.zcoef.detach()*scale)
        mono.pupil_phase.mul_(scale)
        p = POS.clone()
        p[:, 2] = .013
        expected += w * mono.render(p, focus=base.focus_um - POS[:, 2]/.9, dipM=M)
    torch.testing.assert_close(m.render(POS, dipM=M), expected)
    torch.testing.assert_close(ZMode(base, 'emitter').render(POS), base.render(POS))
    torch.testing.assert_close(original, base.pupil_phase, atol=0., rtol=0.)
    table = m.to_spline(**GRID, value='density', components='dipole')
    assert table.knots.shape[0] == 6
    torch.testing.assert_close(table.render(POS, dipM=M), expected, atol=.0015, rtol=.01)


def test_coupled_spectrum():
    base = FourPiPSF(7, npupil=16, device='cpu', pol=(0, 1), z_range=(0., .1), depth_um=(.05, .2))
    base.top.set_zernike({6: .05})
    mono = ZMode(base, 'stage')
    spectrum = Polychromatic(mono, [.64, .72], [.3, .7])
    phase = torch.tensor([.23, -.6])
    images = spectrum.render(POS, phi=phase, dipM=M)
    table = spectrum.to_spline(**GRID, components='dipole')
    torch.testing.assert_close(table.render(POS, phi=phase, dipM=M), images, rtol=.015, atol=.0003)
    assert table.table.axes == ('phase', 'channel', 'dipole')


def test_joint_channel_tile_bank():
    f = field()
    models = [f.to_tiled(tile=10, blend=blend, **GRID) for blend in ('nearest', 'bilinear')]
    channels = ChannelsPSF(models, [[0., .25], [-.3, .1]])
    xy = torch.tensor([[[2., 4.], [7., 3.]], [[16., 19.], [17., 13.]]])
    pc = channels.channel_positions(POS)
    expected = [m.render(pc[:, i], True, xy=xy[:, i]) for i, m in enumerate(models)]
    channels.render(POS, xy=xy)
    table = channels._tile_bank.table
    with patch.object(table, 'evaluate_tables', wraps=table.evaluate_tables) as evaluate:
        actual = channels.render(POS, True, xy=xy)
        evaluate.assert_called_once()
    for v, rows in zip(actual, zip(*expected)):
        torch.testing.assert_close(v, torch.stack(rows, 1))


def test_angles_and_dipoles_by_index():
    a, b = optics(), optics().set_zernike({6: .1})
    from torch_psf import to_spline
    s = to_spline([a, b], value='density', components='dipole', **GRID)
    channels = ChannelsPSF([s, s], channel_offsets=[[.1, .2], [.2, -.3]])
    angle = torch.tensor([1, 0])
    pc = channels.channel_positions(POS)
    for dipM in (None, M):
        images = channels.render(POS, angle=angle, dipM=dipM)
        for k in range(2):
            for n in range(2):
                m = None if dipM is None else dipM[n]
                ref = s.render(pc[n:n+1, k], angle=int(angle[n]), dipM=m)
                torch.testing.assert_close(images[n:n+1, k], ref)


def test_shared_field_and_axial_optics():
    base = optics()
    f = FieldPSF(base, field().field)
    models = [ZMode(f, height=.02, focus=.07), ZMode(f, 'emitter', focus=.1)]
    channels = ChannelsPSF(models)
    xy = torch.tensor([[8., 12.], [13., 7.]])
    result = channels.render(POS, True, xy=xy, dipM=M)
    expected = [m.render(POS, True, xy=xy, dipM=M) for m in models]
    for v, rows in zip(result, zip(*expected)):
        torch.testing.assert_close(v, torch.stack(rows, 1))


def test_linear_basis_and_weights():
    v = optics()
    f = field()
    spectrum = ZMode(FieldPSF(Polychromatic(v, [.65, .72]), f.field), height=.02)
    xy = torch.tensor([[7., 9.], [12., 15.]])
    models = [(spectrum, dict(xy=xy)), (spectrum, dict(xy=xy, dipM=M)),
              (f.to_tiled(tile=10., blend='bilinear', **GRID), dict(xy=xy))]
    coherent = FourPiPSF(7, npupil=16, pol=(0, 1), device='cpu')
    phase = dict(phi=torch.tensor([.32, -.73]))
    models += [(coherent, phase), (coherent.to_spline(z_range=(0., .1), **GRID), phase)]
    for model, kw in models:
        basis = model.basis(POS, **kw)
        weights = model.weights(POS, **kw)
        result = (basis * weights[..., None, None]).sum(2)
        expected = model.render(POS, **kw)
        if expected.ndim == 3:
            expected = expected[:, None]
        torch.testing.assert_close(result, expected, atol=2e-7, rtol=1e-5)


def test_coherent_stage_matches_calibrated_arm_phases():
    base = FourPiPSF(7, npupil=16, pol=(0, 1), depth_um=(.1, .25), device='cpu')
    def stage_pupil(arm, pos, pixels):
        depth = base.depth[0 if arm is base.top else 1]
        return arm._pupil(pos, 0., pixels.u, height=depth.clamp_min(0), stage_um=pos[:, 2])
    with patch.object(base, '_pupil_at_depth', side_effect=stage_pupil):
        expected = base.render(POS, True, phi=.3)
    for v, ref in zip(ZMode(base, 'stage').render(POS, True, phi=.3), expected):
        torch.testing.assert_close(v, ref, rtol=0, atol=0)


def test_blended_fit_keeps_roi_field_coordinates():
    tiled = field().to_tiled(tile=10, blend='bilinear', **GRID)
    xy = torch.tensor([[9., 7.], [11., 12.]])
    theta = torch.cat((POS, torch.tensor([[4000., 7.], [4500., 9.]])), 1)
    images = tiled.bind(xy).expected(theta).detach()
    start = theta.clone()
    start[:, 0] += .05
    ds = tiled.fit(images, xy=xy, theta0=start, batch=1, iters=8)
    fitted = ds['pos'].to(POS) * POS.new_tensor([1., 1., tiled.pixel_um])
    torch.testing.assert_close(fitted, POS, atol=1e-3, rtol=0)
    for i, centre in enumerate(xy):
        fixed = tiled.bind(centre[None]).fit(images[i:i+1], theta0=start[i:i+1], iters=8)
        torch.testing.assert_close(ds['pos'][i], fixed['pos'][0], atol=0, rtol=0)
