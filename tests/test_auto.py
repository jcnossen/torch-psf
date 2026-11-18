"""torch_psf.auto: band limit from the power spectrum and the movie mode classifier, on small synthetic movies with
an exactly known cutoff (CPU, a few seconds each)."""
import numpy as np
import pytest

from torch_psf.auto import (synthetic_movie, power_spectrum, fit_band_limit, otf_circ, movie_statistics,
                            classify_movie, analyze_movie, pupil_otf, scene_movie, moving_movie, global_motion,
                            motion_summary, plan_route, make_estimator, read_psfgrab)


def test_otf_circ():
    assert otf_circ(0.0) == pytest.approx(1.0)
    assert otf_circ(1.0) == pytest.approx(0.0, abs=1e-12)
    assert otf_circ(1.5) == 0.0
    # the scalar pupil OTF of synth has the same support and shape
    N, px, NA, lam = 128, 0.1, 1.0, 0.6
    o = np.abs(pupil_otf(N, px, NA, lam))
    fc = 2 * NA / lam * px
    k = int(0.5 * fc * N)
    assert o[0, k] == pytest.approx(otf_circ(k / N / fc), abs=0.02)
    assert np.abs(o[0, int(fc * N) + 2:N // 2]).max() < 1e-6


def test_band_limit_smlm():
    mov, info = synthetic_movie('smlm', T=120, N=96, NA=1.1, density=0.03, duty=0.02, seed=1)
    spec = power_spectrum(mov, device='cpu')
    bl = fit_band_limit(spec, pixel_um=0.1, wavelength_um=0.6, n_boot=20)
    assert bl.status == 'ok'
    assert abs(bl.fc_px - info['fc_px']) < max(3 * bl.fc_px_err, 0.01)
    assert bl.NA == pytest.approx(1.1, abs=0.05)
    assert bl.f_detect_px <= info['fc_px'] + 0.01
    # white noise: floor = variance of the noise in ADU (Poisson bg 20 photons, gain 0.5 e/ADU -> 80 ADU^2 + read)
    assert bl.floor == pytest.approx(20 / 0.5 ** 2 + 3.0 ** 2, rel=0.1)


def test_band_limit_undersampled_and_noise():
    mov, _ = synthetic_movie('static', T=16, N=96, NA=1.45, wavelength_um=0.5, pixel_um=0.13, photons=4000)
    bl = fit_band_limit(power_spectrum(mov, device='cpu'), n_boot=0)    # f_c = 0.754 cycles/px > corners
    assert bl.status == 'undersampled'
    mov, _ = synthetic_movie('noise', T=32, N=64)
    bl = fit_band_limit(power_spectrum(mov, device='cpu'), n_boot=0)
    assert bl.status == 'no_signal'


@pytest.mark.parametrize('mode,kw', [
    ('smlm', dict(T=150, density=0.02, duty=0.01)),
    ('sofi', dict(T=150, density=0.3, duty=0.3)),
    ('zstack', dict(T=61, density=0.002, photons=20000)),
    ('static', dict(T=60)),
    ('noise', dict(T=60)),
])
def test_classifier_synthetic(mode, kw):
    mov, _ = synthetic_movie(mode, N=96, seed=2, **kw)
    r = analyze_movie(mov, pixel_um=0.1, wavelength_um=0.6, n_boot=0, device='cpu')
    expected = 'no_signal' if mode == 'noise' else mode
    assert r.mode.mode == expected, (r.mode.probs, r.mode.reason)
    assert r.mode.confidence > 0.5
    assert isinstance(r.mode.reason, str) and len(r.mode.reason) > 20
    r.to_dict()


def test_zstack_astigmatism_visible():
    kw = dict(T=61, N=96, density=0.002, photons=20000, seed=3)
    a, _ = synthetic_movie('zstack', astig_waves=0.15, **kw)
    s = movie_statistics(a, device='cpu')
    assert classify_movie(s).mode == 'zstack'
    assert s['astig_swing'] > 0.35


@pytest.mark.parametrize('expected,make', [
    ('scene', lambda: scene_movie(T=30, N=256, seed=1)),                       # handheld, large shifts
    ('scene', lambda: scene_movie(T=30, N=256, step_px=(0.0, 0.3), zoom=0.0, seed=3)),   # still, autofocus hunting
    ('scene', lambda: scene_movie(T=20, N=256, blur=(1.5, 3.0), seed=6)),       # blurry: slope ~ -3 .. -4 like real clips
    ('moving', lambda: moving_movie(T=40, seed=1)),                            # fluorescent cells, stage panning
    ('moving', lambda: moving_movie(T=40, brightfield=True, photons=3000, seed=2)),
    ('static', lambda: moving_movie(T=40, brightfield=True, photons=3000, pan_px=0, swim_px=0.0,
                                    focus_range=(0.3, 0.3), seed=3)),
])
def test_classifier_scene_and_moving(expected, make):
    mov, _ = make()
    r = analyze_movie(mov, n_boot=0, device='cpu')
    assert r.mode.mode == expected, (r.mode.probs, r.mode.reason)
    assert r.mode.confidence > 0.5


def test_global_motion_shift_and_sparse_spots():
    import torch
    mov, info = scene_movie(T=12, N=128, step_px=(4.0, 8.0), zoom=0.0, blur=(0.6, 0.6), seed=4)
    g = global_motion(torch.from_numpy(mov))
    p = np.asarray(info['path'])
    steps = g['pairs'][:, 1] == g['pairs'][:, 0] + 1
    # camera moved by +d -> the image content moves by -d
    np.testing.assert_allclose(g['shift'][steps], -np.diff(p, axis=0)[g['pairs'][steps, 0]], atol=0.6)
    ms = motion_summary(g)
    assert ms['motion_conf'] > 0.8 and ms['motion_moving'] == 1.0
    # sparse blinking spots: no consistent global motion
    mov, _ = synthetic_movie('smlm', T=80, N=96, density=0.02, duty=0.01, seed=2)
    ms = motion_summary(global_motion(torch.from_numpy(mov)))
    assert not (ms['motion_conf'] > 0.5 and ms['motion_moving'] > 0.5), ms


def test_routes():
    mov, _ = synthetic_movie('smlm', T=150, N=96, density=0.02, duty=0.01, seed=2)
    r = analyze_movie(mov, pixel_um=0.1, wavelength_um=0.6, n_boot=0, device='cpu')
    rt = r.route(pixel_um=0.1, wavelength_um=0.6)
    assert rt.estimator == 'InSituEstimator' and rt.needs == ['NA'] and 'NA_eff' in rt.hints
    assert 'InSituEstimator' in rt.summary()
    mov, _ = synthetic_movie('zstack', T=61, N=96, density=0.002, photons=20000, seed=2)
    rt = analyze_movie(mov, n_boot=0, device='cpu').route()
    assert rt.estimator in ('ZStackEstimator', 'DenseBeadEstimator') and 'dz_um' in rt.needs
    mov, _ = scene_movie(T=20, N=192, seed=5)
    rt = analyze_movie(mov, n_boot=0, device='cpu').route()
    assert rt.mode == 'scene' and rt.estimator is None
    with pytest.raises(ValueError):
        make_estimator(rt)


def test_read_psfgrab(tmp_path):
    import json, zipfile, tifffile, io
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (3, 3, 20, 24), dtype=np.uint8)
    buf = io.BytesIO(); tifffile.imwrite(buf, a, imagej=True, metadata={'axes': 'TCYX'})
    meta = dict(format='psfgrab/1', frames=[dict(t_ms=72 * i, settings={}) for i in range(3)])
    p = tmp_path / 'rec.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('frames.tif', buf.getvalue()); z.writestr('meta.json', json.dumps(meta))
    f, m = read_psfgrab(p)
    assert f.shape == (3, 20, 24) and f.dtype == np.float32 and m['format'] == 'psfgrab/1'
    np.testing.assert_allclose(f[1], 0.2126 * a[1, 0] + 0.7152 * a[1, 1] + 0.0722 * a[1, 2], rtol=1e-5)
    raw, _ = read_psfgrab(p, gray=False)
    assert raw.shape == a.shape and raw.dtype == np.uint8


def test_large_frames_memory():
    """Full-HD phone frames: the classifier crops / chunks instead of allocating ~T x H x W x kernel taps."""
    try:                                   # peak resident memory in GB: resource on Linux / macOS, psutil on Windows
        import resource
        peak_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # ru_maxrss in kB (Linux)
    except ImportError:
        psutil = pytest.importorskip('psutil')
        peak_gb = lambda: psutil.Process().memory_info().peak_wset / 1e9
    mov = np.random.default_rng(0).normal(100, 5, (24, 1920, 1080)).astype(np.float32)
    r0 = peak_gb()
    st = movie_statistics(mov, device='cpu')
    grow_gb = peak_gb() - r0
    assert st['crop'][2:] == [1024, 1024]
    assert grow_gb < 3.0, grow_gb
