"""The interface a downstream localization pipeline builds on (kept stable; no data or code of that pipeline here):
movie reader interface, Camera with per-pixel maps and ROI corners, PSF.load of a plain calibration json, the generic
Levenberg-Marquardt / Fisher functions, the closed-form linear solve and the track fit with a per-group PSF hook
(e.g. photoselection per illumination-pattern angle)."""
import numpy as np
import tifffile
import torch
from conftest import DEV
from test_io import calibration_json
from torch_psf import Camera, VectorialPSF, PSF, open_movie
from torch_psf.io import load_camera
from torch_psf.movie import TiffMovie, ArrayMovie
from torch_psf.fit import lm_poisson, fisher, model_and_jacobian, solve_linear
from torch_psf.spots import Tracks
from torch_psf.estimation import fit_tracks, heldout_tracks, paired_nll
from torch_psf.vectorial import photosel_tensor


def test_reader_interface(tmp_path):
    a = (np.arange(7 * 8 * 9) % 1000).reshape(7, 8, 9).astype(np.uint16)
    tifffile.imwrite(tmp_path / 'm.tif', a[:4], photometric='minisblack')
    tifffile.imwrite(tmp_path / 'm_X1.tif', a[4:], photometric='minisblack')           # multi-part movie
    for mv in (TiffMovie(str(tmp_path / 'm.tif')), ArrayMovie(a), open_movie(str(tmp_path / 'm.tif')), open_movie(a)):
        assert mv.n == 7 and tuple(mv.shape) == (8, 9)
        r = mv.read(3, 6, 1, 2, 4)
        assert r.dtype == np.float32 and np.array_equal(r, a[3:6, 1:5, 2:6])


def test_camera_maps_and_roi_corners(tmp_path):
    calibration_json(tmp_path / 'calib.json')
    cam = load_camera(tmp_path / 'calib.json')
    cam.offset_adu = cam.offset_adu + np.arange(64, dtype=np.float32)[None, :]       # column-dependent offset
    adu = torch.full((2, 5, 5), 300.0, device=DEV)
    corners = torch.tensor([[3, 4], [10, 0]], device=DEV)
    ph = cam.to_photons(adu, corners)
    expect = (300.0 - (100.0 + torch.arange(5, device=DEV)[None, None, :] + torch.tensor([3, 10], device=DEV)[:, None, None])) * 0.56
    assert torch.allclose(ph, expect.float())
    assert torch.allclose(cam.variance(adu, corners), torch.full_like(adu, 1.5 ** 2))
    full = cam.to_photons(torch.full((1, 64, 64), 300.0, device=DEV))
    assert torch.allclose(full[0, 4:9, 3:8], ph[0])
    c2 = Camera.from_dict(dict(gain_e_per_adu=2.0, offset_adu=100.0, readnoise_e=1.0, type='scmos', tirf_depth_um=1))
    assert c2.gain_e_per_adu == 2.0


def test_calibration_json_psf(tmp_path):
    calibration_json(tmp_path / 'calib.json')
    v = PSF.load(tmp_path / 'calib.json', roisize=11, device=DEV)
    assert isinstance(v, VectorialPSF) and v.zidx == list(range(5, 22))
    with torch.no_grad():
        img = v(torch.zeros(2, 3, device=DEV), torch.full((2,), v.focus_um, device=DEV))
    assert img.shape == (2, 11, 11) and float(img.sum()) > 1.0


def test_generic_lm_and_fisher():
    """A user model: 2 parameters (amplitude, background) of a fixed pattern, analytic Jacobian [N, p, M]."""
    pat = torch.linspace(0, 1, 50, device=DEV)

    def model(t):
        mu = t[:, 0, None] * pat[None] + t[:, 1, None]
        return mu, torch.stack([pat[None].expand(len(t), -1), torch.ones_like(mu)], 1)

    th0 = torch.tensor([[200.0, 10.0], [50.0, 3.0]], device=DEV)
    d = model(th0)[0]
    th, nll = lm_poisson(model, d, torch.tensor([[100.0, 5.0], [100.0, 5.0]], device=DEV), var=2.0)
    assert torch.allclose(th, th0, rtol=1e-3)
    from torch_psf import GaussianPSF
    F = fisher(GaussianPSF(9, device=DEV), torch.tensor([[0.0, 0.0, 0.0, 1000.0, 5.0]], device=DEV), var=4.0,
               params=(0, 1, 3, 4))
    assert F.shape == (1, 4, 4)


def test_solve_linear():
    B, T, P = 3, 4, 25
    psf = torch.rand(B, P, device=DEV)
    I = torch.rand(B, T, device=DEV) * 1000
    bg = torch.rand(B, 2, device=DEV) * 10
    group = torch.tensor([[0, 0, 1, 1]] * B, device=DEV)
    data = I[..., None] * psf[:, None] + torch.gather(bg, 1, group)[..., None]
    I2, bg2 = solve_linear(psf, data, torch.ones(B, T, dtype=torch.bool, device=DEV), group, 1.0)
    assert torch.allclose(I2, I, rtol=1e-2, atol=0.5) and torch.allclose(bg2, bg, rtol=1e-2, atol=0.05)


def test_fit_tracks_with_group_hook():
    R, B, T = 11, 40, 6
    vec = VectorialPSF(R, npupil=32, oversample=2, device=DEV)
    vec.set_zernike({6: 0.05})
    gamma_true = 0.3
    M = [photosel_tensor(torch.tensor(gamma_true, device=DEV), e) for e in ((1.0, 0.0), (0.0, 1.0))]
    g = torch.Generator(device=DEV).manual_seed(0)
    xy = torch.rand(B, 2, generator=g, device=DEV) - 0.5
    z = 0.1 * torch.rand(B, generator=g, device=DEV)
    groups = torch.tensor([[0, 0, 0, 1, 1, 1]] * B, device=DEV)
    with torch.no_grad():
        pos = torch.cat([xy, z[:, None]], 1)
        imgs = torch.stack([vec(pos, torch.zeros(B, device=DEV), dipM=M[k]) for k in range(2)], 1)
        mu = 3000 * torch.gather(imgs, 1, groups[..., None, None].expand(-1, -1, R, R)) + 10
    rois = torch.poisson(mu, generator=g)
    tr = Tracks(rois, torch.ones(B, T, dtype=torch.bool, device=DEV), torch.arange(T, device=DEV)[None].expand(B, -1),
                torch.zeros(B, 2, dtype=torch.long, device=DEV), xy + (R - 1) / 2)
    gam = torch.tensor(0.05, device=DEV, requires_grad=True)
    hook = lambda k: dict(dipM=photosel_tensor(gam, ((1.0, 0.0), (0.0, 1.0))[k]))
    est = fit_tracks(tr, vec, readnoise_e=0.0, zmax_um=0.2, iters=150, warmup=20, groups=groups, ngroups=2,
                     group_psf_kwargs=hook, extra_params=[gam], lr_extra=0.02)
    assert abs(float(gam.detach()) - gamma_true) < 0.1, float(gam.detach())
    assert 0.7 < est['report']['resid_var'] < 1.3
    assert est['I'].shape == (B, T)


def test_fit_tracks_fixed_zernikes():
    R, B, T = 11, 20, 4
    vec = VectorialPSF(R, npupil=32, oversample=2, device=DEV).set_zernike({6: 0.05})
    g = torch.Generator(device=DEV).manual_seed(1)
    xy = torch.rand(B, 2, generator=g, device=DEV) - 0.5
    with torch.no_grad():
        img = vec(torch.cat([xy, torch.full((B, 1), 0.05, device=DEV)], 1), torch.zeros(B, device=DEV))
    rois = torch.poisson(2000 * img[:, None].expand(-1, T, -1, -1) + 10, generator=g)
    tr = Tracks(rois, torch.ones(B, T, dtype=torch.bool, device=DEV), torch.arange(T, device=DEV)[None].expand(B, -1),
                torch.zeros(B, 2, dtype=torch.long, device=DEV), xy + (R - 1) / 2)
    r = fit_tracks(tr, vec, readnoise_e=0.0, iters=30, warmup=5, fix_zernike=True, fix_focus=True)
    assert torch.equal(r['psf'].zcoef.detach(), vec.zcoef.detach()) and r['report']['focus_um'] == 0.0
    assert r['report']['nll_per_px'] <= r['report']['nll_per_px_before_last_step'] + 1e-6


def test_heldout_tracks_frozen_psf_paired():
    """frozen-PSF refit of track samples (per-group PSF, background per group, z multi-start): the true PSF beats an
    aberrated one on the paired per-sample NLL; chi2 about 1; the spline refit gives about the same NLL"""
    R, B, T = 11, 60, 6
    vec = VectorialPSF(R, npupil=32, oversample=2, device=DEV).set_zernike({6: 0.05})
    M = [photosel_tensor(torch.tensor(0.3, device=DEV), e) for e in ((1.0, 0.0), (0.0, 1.0))]
    g = torch.Generator(device=DEV).manual_seed(2)
    xy = torch.rand(B, 2, generator=g, device=DEV) - 0.5
    z = 0.02 + 0.12 * torch.rand(B, generator=g, device=DEV)
    groups = torch.tensor([[0, 0, 0, 1, 1, 1]] * B, device=DEV)
    with torch.no_grad():
        pos = torch.cat([xy, z[:, None]], 1)
        imgs = torch.stack([vec(pos, torch.zeros(B, device=DEV), dipM=M[k]) for k in range(2)], 1)
        mu = 3000 * torch.gather(imgs, 1, groups[..., None, None].expand(-1, -1, R, R)) + 10 + 5 * groups[..., None, None]
    rois = torch.poisson(mu, generator=g)
    tr = Tracks(rois, torch.ones(B, T, dtype=torch.bool, device=DEV), torch.arange(T, device=DEV)[None].expand(B, -1),
                torch.zeros(B, 2, dtype=torch.long, device=DEV), xy + (R - 1) / 2)

    def per_group(v):
        out = []
        for k in range(2):
            m = v.resized(R)
            m.dipM = M[k]
            out.append(m)
        return out
    wrong = VectorialPSF(R, npupil=32, oversample=2, device=DEV).set_zernike({6: -0.05, 8: 0.05})
    kw = dict(readnoise_e=0.0, zmax_um=0.2, groups=groups, ngroups=2, z_starts_um=(0.03, 0.12), iters=80)
    ht, hw = heldout_tracks(tr, per_group(vec), **kw), heldout_tracks(tr, per_group(wrong), **kw)
    assert ht['nll'].shape == (B,) and ht['I'].shape == (B, T) and ht['bg'].shape == (B, 2)
    assert 0.8 < ht['chi2_per_px'] < 1.2 and hw['chi2_per_px'] > ht['chi2_per_px']
    p = paired_nll(ht['nll'], hw['nll'])
    assert p['mean'] > 0 and p['z'] > 3, p
    assert np.median(np.abs(ht['z'] - z.cpu().numpy())) < 0.03
    spl = [m.to_spline(z_range=(0.0, 0.2), oversample=4, dz_um=0.01, zpad=3, xy_limit=2.0) for m in per_group(vec)]
    hs = heldout_tracks(tr, per_group(vec), refit_psfs=spl, **kw)
    assert abs(hs['nll_per_px'] - ht['nll_per_px']) < 2e-3
