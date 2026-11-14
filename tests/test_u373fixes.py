"""Fixes from a real-data test (ImageJ TIFF > 4 GB, sampling over the frame range, robust overlap rejection, camera
excess noise and photon transfer, spline resizing, provenance, FRC region / mask, ShareLoc .smlm)."""
import json
import os
import zipfile
import numpy as np
import pytest
import tifffile
import torch
from conftest import DEV
import torch_psf as tp
from torch_psf.movie import TiffMovie
from torch_psf.spots import extract_spots, spread_order
from torch_psf.simulate import simulate_movie
from torch_psf.estimators import InSituEstimator


def test_imagej_single_ifd(tmp_path):
    """ImageJ files > 4 GB hold one IFD and the frames contiguously (tifffile truncate=True writes the same layout)."""
    a = np.random.default_rng(0).integers(0, 1000, (40, 24, 20)).astype(np.uint16)
    p = tmp_path / 'movie.tif'
    tifffile.imwrite(p, a, imagej=True, truncate=True)
    with tifffile.TiffFile(p) as t:
        assert len(t.pages) == 1                       # the layout that made the earlier reader see 1 frame
    m = TiffMovie(p)
    assert m.n == 40 and tuple(m.shape) == (24, 20)
    np.testing.assert_array_equal(m.read(5, 17), a[5:17].astype(np.float32))
    np.testing.assert_array_equal(m.read(3, 6, 2, 4, 8), a[3:6, 2:10, 4:12].astype(np.float32))


def test_multipart_names(tmp_path):
    """movie.tif + movie_X1.tif; a '.tif' elsewhere in the path is not touched."""
    d = tmp_path / 'run.tif.d'
    d.mkdir()
    a = np.arange(7 * 8 * 8, dtype=np.uint16).reshape(7, 8, 8)
    tifffile.imwrite(d / 'movie.tif', a[:4], photometric='minisblack')
    tifffile.imwrite(d / 'movie_X1.tif', a[4:], photometric='minisblack')
    m = TiffMovie(d / 'movie.tif')
    assert m.n == 7 and len(m.parts) == 2
    np.testing.assert_array_equal(m.read(2, 6), a[2:6].astype(np.float32))


def test_spread_sampling():
    assert sorted(spread_order(11)) == list(range(11)) and spread_order(8)[:4] == [0, 4, 2, 6]
    psf = tp.GaussianPSF(11, sigma_px=1.3, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(2)
    movie, _ = simulate_movie(psf.render, 11, T=400, H=48, W=48, nsites=40, p_on=0.05, mean_on=2.0, bg=10.0,
                              photons=3000.0, photons_logsd=0.1, z_range=(0.0, 0.0), margin=8, generator=g, device=DEV)
    seq = extract_spots(movie.numpy(), 11, chunk=10, max_spots=150, device=DEV)
    spr = extract_spots(movie.numpy(), 11, chunk=10, max_spots=150, device=DEV, sample="spread")
    assert len(spr) == 150 and spr.frames_read < 400
    assert int(seq.frame.max()) < 100                  # sequential: from the start only
    assert int(spr.frame.min()) < 40 and int(spr.frame.max()) > 300       # spread over the range
    assert bool((spr.frame[1:] >= spr.frame[:-1]).all())


def test_deviance_rule():
    est = InSituEstimator(roisize=15, device=DEV)
    rng = np.random.default_rng(0)
    dv = np.concatenate([1.2 * (1 + 0.094 * rng.standard_normal(900)), 3 + rng.random(100)])   # gain error + overlaps
    thr, med, scale = est._deviance_threshold(dv)
    assert abs(med - 1.2) < 0.05 and 1.4 < thr < 1.75
    assert 0.08 < (dv >= thr).mean() < 0.13            # ~ the overlaps only
    est.deviance_rule = 'poisson'
    thr_p, _, _ = est._deviance_threshold(dv)
    assert abs(thr_p - (1 + 3 * np.sqrt(2 / 225))) < 1e-9 and (dv >= thr_p).mean() > 0.2


def test_camera_excess_noise():
    c = tp.Camera(gain_e_per_adu=4.0, offset_adu=100.0, readnoise_e=2.0, excess_noise=2.0, type='emccd')
    x = torch.full((2, 3, 3), 150.0)
    assert torch.allclose(c.to_photons(x), torch.full((2, 3, 3), 100.0))
    assert float(c.variance(x)) == pytest.approx(1.0)
    assert tp.Camera.from_dict(c.to_dict()).excess_noise == 2.0
    assert tp.Camera().excess_noise == 1.0


@pytest.mark.parametrize('F2', [1.0, 2.0])
def test_photon_transfer(F2):
    """background-only movie with spots: gain e- / ADU 0.5 (x F2 for excess noise), offset 100"""
    g = torch.Generator().manual_seed(3)
    T, H, W = 400, 64, 64
    bg = 20.0 + 30.0 * torch.rand(1, H, W, generator=g)               # photons, varying over the field
    psf = tp.GaussianPSF(11, sigma_px=1.3, device='cpu')
    movie, _ = simulate_movie(psf.render, 11, T=T, H=H, W=W, nsites=30, p_on=0.05, mean_on=2.0, bg=0.0,
                              photons=2000.0, photons_logsd=0.1, z_range=(0.0, 0.0), margin=8, generator=g,
                              device='cpu')
    e = torch.poisson(bg.expand(T, H, W)) + movie.float()             # photoelectrons
    if F2 > 1:                                                         # EM register: variance x F2 at the same mean
        e = e + torch.randn(e.shape, generator=g) * torch.sqrt((F2 - 1) * e.clamp_min(0))
    gain = 0.5 * F2
    adu = (e / gain + 100.0).numpy()
    cam, rep = tp.photon_transfer(adu, nblocks=4, block=100, excess_noise=F2, device='cpu')
    assert cam.offset_adu == pytest.approx(100.0, abs=3.0)
    assert cam.gain_e_per_adu == pytest.approx(gain, rel=0.08)
    assert rep['poisson_equivalent_gain'] == pytest.approx(0.5, rel=0.08) and cam.excess_noise == F2


def test_spline_resized_and_provenance(tmp_path):
    a = tp.AstigmaticGaussianPSF(15, 1.2, 0.3, 0.4, (-0.4, 0.4), device=DEV, pixel_um=0.1)
    s = a.to_spline(15, dz_um=0.05)
    s11 = s.to_spline(11)
    assert s11.roisize == 11 and s11.knots is s.knots and s11.xy_limit == s.xy_limit
    pos = torch.tensor([[0.3, -0.4, 0.1], [-1.0, 0.5, -0.2]], device=DEV)
    torch.testing.assert_close(s11.render(pos), s.render(pos)[:, 2:13, 2:13])
    assert s.resized(21).xy_limit < s.xy_limit
    s.provenance = dict(estimator='test', report=dict(n=3, v=np.float32(1.5), t=(1, 2)))
    s.save(tmp_path / 's.json')
    r = tp.PSF.load(tmp_path / 's.json', device=DEV)
    assert r.provenance == dict(estimator='test', report=dict(n=3, v=1.5, t=[1, 2]))
    assert json.load(open(tmp_path / 's.json'))['provenance']['estimator'] == 'test'
    s.save(tmp_path / 't.json', provenance=dict(a=1))
    assert tp.PSF.load(tmp_path / 't.json').provenance == dict(a=1)


def test_localize_options_and_progress():
    psf = tp.GaussianPSF(11, sigma_px=1.3, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(1)
    movie, _ = simulate_movie(psf.render, 11, T=120, H=48, W=48, nsites=20, p_on=0.05, mean_on=2.0, bg=10.0,
                              photons=3000.0, photons_logsd=0.1, z_range=(0.0, 0.0), margin=8, generator=g, device=DEV)
    calls = []
    L = psf.localize(movie.numpy(), chunk=20, block=60, iters=10, zstarts=1, isolation=6,
                     progress=lambda d, t, n: calls.append((d, t, n)))
    assert len(L) > 0 and 'chi2_median' in L.meta
    assert len(calls) >= 6 and calls[-1][0] == 120 and [c[0] for c in calls] == sorted(c[0] for c in calls)


def test_detect_mad_frame():
    x = torch.randn(4, 64, 64) * torch.tensor([1.0, 1.0, 3.0, 3.0])[:, None, None] + 10
    x[0, 30, 30] += 60
    x[2, 20, 20] += 60
    p = tp.detect_spots(x, threshold=8, noise='mad_frame', sigma=(1.0, 4.0), border=10)
    assert set(p['frame'].tolist()) <= {0, 2}
    assert p['score'].shape == p['frame'].shape


def test_frc_region_mask():
    rng = np.random.default_rng(0)
    n = 20000
    x = np.concatenate([rng.uniform(0, 40, n), rng.uniform(60, 100, n)])
    pos = np.stack([x, rng.uniform(0, 40, 2 * n)], 1) + 0.02 * rng.standard_normal((2 * n, 2))
    L = tp.Dataset(dict(pos=pos, frame=rng.integers(0, 3000, 2 * n)), dict(pixel_nm=100.0))
    a = L.frc(region=(-10, -10, 50, 50))
    b = L.frc(mask=L['pos'][:, 0] < 50)
    assert a == b
    assert np.isfinite(L.frc()) or np.isnan(L.frc())


def test_load_smlm(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    cols = dict(id=np.arange(1, n + 1), frame=rng.integers(1, 100, n), x=rng.uniform(0, 5000, n),
                y=rng.uniform(0, 5000, n), intensity_photon_=rng.uniform(100, 900, n), offset_photon_=np.full(n, 5.0),
                uncertainty_nm_=rng.uniform(5, 20, n), chi2=rng.uniform(0, 2, n))
    hdr = list(cols)
    dt = np.dtype([(h, 'float32') for h in hdr])
    a = np.zeros(n, dt)
    for h in hdr:
        a[h] = cols[h]
    p = tmp_path / 'data.smlm'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('manifest.json', json.dumps(dict(
            format_version='0.2', formats={'t': dict(type='table', mode='binary', headers=hdr,
                                                     dtype=['float32'] * len(hdr))},
            files=[dict(name='t.bin', format='t', rows=n, type='table')])))
        z.writestr('t.bin', a.tobytes())
    L = tp.Dataset.load(str(p), pixel_nm=100.0)
    assert len(L) == n and L.n_dims == 2 and os.path.basename(L.meta['source']) == 'data.smlm'
    np.testing.assert_allclose(L['pos'][:, 0].numpy(), a['x'] / 100.0 - 0.5, rtol=1e-6)
    np.testing.assert_array_equal(L['frame'].numpy(), a['frame'].astype(int) - 1)
    np.testing.assert_allclose(L['crlb'][:, 1].numpy(), a['uncertainty_nm_'] / 100.0, rtol=1e-6)
    np.testing.assert_allclose(L['photons'].numpy(), a['intensity_photon_'], rtol=1e-6)
    assert 'chi2' in L.keys() and 'bg' in L.keys()
