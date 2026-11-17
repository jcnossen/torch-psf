"""torch_psf.auto: band limit from the power spectrum and the movie mode classifier, on small synthetic movies with
an exactly known cutoff (CPU, a few seconds each)."""
import numpy as np
import pytest

from torch_psf.auto import (synthetic_movie, power_spectrum, fit_band_limit, otf_circ, movie_statistics,
                            classify_movie, analyze_movie, pupil_otf)


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
