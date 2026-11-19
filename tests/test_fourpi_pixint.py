import math

import pytest
import torch

from conftest import DEV
from torch_psf import FourPiPSF, IntegratedSplinePSF, PSF, io, psfzip
from torch_psf.integrate import PixelIntegrator


@torch.no_grad()
@pytest.mark.parametrize('pixel_um', [0.018, 0.10833])
def test_fourpi_midpoint_convergence(pixel_um):
    kw = dict(roisize=39 if pixel_um < .1 else 13, pixel_um=pixel_um, npupil=24, device=DEV, NA=(1.2, 1.35),
              channel_offsets=[[0, 0], [.21, -.19], [0, 0], [-.1, .2]])
    pos = torch.tensor([[.13, -.34, -.12], [-.31, .25, .02], [.17, .04, .15]], device=DEV)
    phase = torch.tensor([.37, 1.6, 2.1], device=DEV)
    exact = FourPiPSF(**kw, oversample=3 if pixel_um < .1 else 2, pixel_integration='spectral')(pos, phi_extra=phase)
    high = FourPiPSF(**kw, oversample=16)(pos, phi_extra=phase)
    if pixel_um > .1:
        # Remove the O(1/os^2) midpoint error, still visible at os=16 for typical camera pixels.
        low = FourPiPSF(**kw, oversample=8)(pos, phi_extra=phase)
        high = (4 * high - low) / 3
    peak = exact.amax((-2, -1))
    assert torch.all((exact - high).abs().amax((-2, -1)) / peak < 1e-5)


def test_fourpi_stored_exact_and_scalar_phase():
    model = FourPiPSF(7, npupil=16, device='cpu', pixel_integration='spectral')
    state = model.state()
    state['meta']['pixel_integration'] = 'exact'
    loaded = FourPiPSF.from_state(state, device='cpu')
    pos = torch.tensor([[.12, -.22, .07], [-.2, .3, .04]])
    actual = loaded.render(pos, phi=.4, grad=True)
    expected = model.render(pos, phi=torch.full((2,), .4), grad=True)
    assert all(torch.equal(v, w) for v, w in zip(actual, expected))
    with pytest.raises(ValueError, match='pixel_integration'):
        model.pixel_integration = 'exact'


@torch.no_grad()
def test_fourpi_box_legacy_pixels(monkeypatch):
    model = FourPiPSF(9, npupil=24, device=DEV)
    pos = torch.tensor([[.32, -.15, .09]], device=DEV)
    expected = model(pos)

    def legacy(grid, intensity):
        b, c, n, _ = intensity.shape
        bv = model.blur_px2
        f = torch.fft.fftfreq(n, d=1 / model.os, device=intensity.device)
        fr = torch.fft.rfftfreq(n, d=1 / model.os, device=intensity.device)
        transfer = torch.exp(-2 * math.pi ** 2 * (bv * fr[None, :] ** 2 + bv * f[:, None] ** 2))
        blurred = torch.fft.irfft2(torch.fft.rfft2(intensity) * transfer, s=(n, n))
        border = 3 * model.os
        blurred = blurred[:, :, border:-border, border:-border]
        return blurred.reshape(b, c, 9, model.os, 9, model.os).mean((3, 5))

    monkeypatch.setattr(PixelIntegrator, '__call__', legacy)
    assert torch.equal(model(pos), expected)
    assert 'pixel_integration' not in model.state()['meta']


@torch.no_grad()
def test_fourpi_integrated_round_trip(tmp_path, monkeypatch):
    model = FourPiPSF(9, npupil=24, device=DEV, pixel_integration='spectral', z_range=(-.025, .025),
                      channel_offsets=[[0, 0], [.21, -.19], [0, 0], [-.1, .2]])
    pos = torch.tensor([[.23, -.17, -.0023], [-.14, .38, .012]], device=DEV)
    phase = torch.tensor([.61, -1.51], device=DEV)
    spline = model.to_spline(oversample=5, dz_um=.005, xy_limit=1, zpad=6, chunk=12)
    assert isinstance(spline.luts[0][0], IntegratedSplinePSF)
    ref = model.render(pos, phi=phase)
    assert torch.all((spline.render(pos, phi=phase) - ref).abs().amax((-2, -1)) / ref.amax((-2, -1)) < 2e-5)
    for psf in (model, spline):
        json_path = tmp_path / (psf.type_name + '.json')
        psf.save(json_path)
        assert torch.equal(PSF.load(json_path, device=DEV).render(pos, phi=phase), psf.render(pos, phi=phase))
        path = tmp_path / (psf.type_name + '.zip')
        meta = io.save_zip(path, psf, zstack_planes=3)
        assert meta['version'] == '0.4.0'
        if psf is spline:
            assert meta['channels'][0]['model']['integrated_spline']['value'] == 'pixel_integral'
        loaded = PSF.load(path, device=DEV)
        assert torch.equal(loaded.render(pos, phi=phase), psf.render(pos, phi=phase))
        with monkeypatch.context() as m:
            m.setattr(psfzip, 'READ_VERSIONS', ((0, 2), (0, 3)))
            with pytest.raises(ValueError, match='version 0.4.0'):
                PSF.load(path)


@torch.no_grad()
@pytest.mark.parametrize('kind', ['integrated', 'cubic'])
@pytest.mark.parametrize('storage', ['float32', 'rebuild'])
def test_phase_spline_dipole_from_box_source(tmp_path, monkeypatch, kind, storage):
    model = FourPiPSF(7, npupil=16, oversample=1, z_range=(-.02, .02), device=DEV)
    model.dipM = torch.diag(torch.tensor([.5, .2, .3], device=DEV))
    restored = FourPiPSF.from_state(model.state(), device=DEV)
    torch.testing.assert_close(restored.dipM, model.dipM)
    spline = model.to_spline(kind=kind, oversample=2, dz_um=.01, xy_limit=.5)
    assert spline.zpad == (6 if kind == 'integrated' else 3)
    path = tmp_path / 'rebuild.zip'
    meta = io.save_zip(path, spline, zstack_planes=3, spline_knots=storage)
    assert meta['version'] == ('0.4.0' if kind == 'integrated' else '0.2.0')
    assert 'pixel_integration' not in meta['source']['model']['fourpi']['meta']
    loaded = PSF.load(path, device=DEV)
    torch.testing.assert_close(loaded.knots, spline.knots, atol=1e-7, rtol=1e-5)
    pos = torch.tensor([[.13, -.17, .007]], device=DEV)
    ref = spline.render(pos, phi=.54)
    assert (loaded.render(pos, phi=.54) - ref).abs().max() <= 1e-6 * ref.max()
    if kind == 'integrated':
        monkeypatch.setattr(psfzip, 'READ_VERSIONS', ((0, 2), (0, 3)))
        with pytest.raises(ValueError, match='version 0.4.0'):
            PSF.load(path)


@torch.no_grad()
def test_integrated_default_padding_near_axial_limits():
    model = FourPiPSF(7, npupil=16, pol=(0, 1), device='cpu', pixel_integration='spectral', z_range=(-.05, .05))
    z = torch.tensor([-.05, -.047, -.045, .045, .047, .05])
    pos = torch.stack([torch.full_like(z, .17), torch.full_like(z, -.21), z], 1)
    ref = model.render(pos, phi=.73)
    def error(spline):
        return ((spline.render(pos, phi=.73) - ref).abs().amax((-1, -2)) / ref.amax((-1, -2))).max()
    default = error(model.to_spline(dz_um=.01, xy_limit=.6))
    assert default < 1e-4
    assert default < .2 * error(model.to_spline(dz_um=.01, xy_limit=.6, zpad=3))
