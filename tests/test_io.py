"""PSF files: save / load round trips and the plain calibration json."""
import json
import torch
from conftest import DEV
from torch_psf import VectorialPSF, SplinePSF, GaussianPSF, AstigmaticGaussianPSF, PSF
from torch_psf.io import load_camera, load_spline_source

POS = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.1, 0.05]])


def test_vectorial_round_trip(tmp_path):
    v = VectorialPSF(13, npupil=48, oversample=2, device=DEV, focus_um=0.3)
    v.set_zernike({6: 0.1, 11: 0.02})
    with torch.no_grad():
        v.pupil_phase.copy_(0.03 * v.mask * v.px)
    v.blur = torch.tensor([0.1, 0.05, 0.01], device=DEV)
    v.dipM = torch.diag(torch.tensor([0.3, 0.3, 0.4], device=DEV))
    v.save(tmp_path / 'v.json')
    w = PSF.load(tmp_path / 'v.json', device=DEV)
    p = POS.to(DEV)
    assert torch.allclose(v.render(p), w.render(p), atol=1e-7)
    d = json.load(open(tmp_path / 'v.json'))
    assert d['format'] == 'torch-psf' and d['optics']['pixel_um'] == v.pixel_um and d['pixelsize_um'] == v.pixel_um


def test_spline_round_trip(tmp_path):
    v = VectorialPSF(13, npupil=48, oversample=2, device=DEV, focus_um=0.3, z_range=(0.0, 0.6))
    v.set_zernike({6: 0.1})
    s = v.to_spline(11)
    s.save(tmp_path / 's.json', source=v)
    t = PSF.load(tmp_path / 's.json', device=DEV)
    p = POS.to(DEV)
    assert isinstance(t, SplinePSF) and torch.equal(s.render(p), t.render(p))
    src = load_spline_source(tmp_path / 's.json', device=DEV)
    assert torch.allclose(src.zcoef, v.zcoef)


def test_gaussian_round_trip(tmp_path):
    g = AstigmaticGaussianPSF(11, sigma_px=(1.2, 1.4), gamma_um=0.2, d_um=0.4, device=DEV)
    g.save(tmp_path / 'g.json')
    h = PSF.load(tmp_path / 'g.json', device=DEV)
    p = POS.to(DEV)
    assert torch.allclose(g.render(p), h.render(p))


def calibration_json(path, **extra):
    """A plain calibration json (top-level optics, Zernikes, focus, camera block with per-pixel maps)."""
    import numpy as np
    d = dict(pixelsize_um=0.10833, wavelength_um=0.7, NA=1.49, n_imm=1.518, n_med=1.33, apod=[0.0, 0.0, 0.0],
             zernike_noll=list(range(5, 22)), zernike_waves=[0.01 * ((-1) ** j) for j in range(17)], focus_um=0.11,
             camera=dict(gain_e_per_adu=0.56, offset_adu=99.3, readnoise_e=0.9, maps='maps.npz'), **extra)
    json.dump(d, open(path, 'w'))
    np.savez(path.parent / 'maps.npz', offset_adu=np.full((64, 64), 100.0, np.float32),
             readnoise_e=np.full((64, 64), 1.5, np.float32))
    return d


def test_calibration_json(tmp_path):
    d = calibration_json(tmp_path / 'calib.json', psf_blur_px2=[0.2, 0.0, -0.01],
                         psf_dipole_M_angle=[[[0.3, 0, 0], [0, 0.38, 0], [0, 0, 0.32]],
                                             [[0.38, 0, 0], [0, 0.3, 0], [0, 0, 0.32]]])
    v = PSF.load(tmp_path / 'calib.json', roisize=11, device=DEV)
    assert abs(v.focus_um - 0.11) < 1e-9 and v.pixel_um == 0.10833 and v.blur is not None
    assert torch.allclose(v.dipM, torch.diag(torch.tensor([0.34, 0.34, 0.32], device=DEV)))
    v1 = PSF.load(tmp_path / 'calib.json', roisize=11, device=DEV, angle=1)
    assert float(v1.dipM[0, 0]) == float(torch.tensor(0.38))
    cam = load_camera(tmp_path / 'calib.json')
    assert cam.offset_adu.shape == (64, 64) and cam.gain_e_per_adu == 0.56
    json.dump(dict(d, psf_gauss_sigma_um=0.14), open(tmp_path / 'g.json', 'w'))
    assert isinstance(PSF.load(tmp_path / 'g.json', device=DEV), GaussianPSF)


def test_load_from_dict(tmp_path):
    d = calibration_json(tmp_path / 'calib.json')
    v = PSF.load(json.load(open(tmp_path / 'calib.json')), roisize=11, device=DEV)
    assert isinstance(v, VectorialPSF) and abs(v.focus_um - 0.11) < 1e-9
