"""Cheap synthetic movies with an exactly known band limit, for tests and calibration of the auto analysis.

Imaging is done in Fourier space: frame = IFFT(FFT(object) x OTF), with the OTF of a scalar circular pupil of radius
NA / lambda (cycles / um), optionally defocused (high-NA defocus phase 2 pi z sqrt((n / lambda)^2 - k^2)) and with
primary astigmatism; so the support of every frame's spectrum is exactly f_c = 2 NA / lambda. The object is a set of
point emitters on the pixel grid (or a static filament image). Poisson noise on (signal + background), then
ADU = photons / gain + offset + Gaussian read noise. Small and fast (CPU), deterministic for a given seed.

synthetic_movie(mode, ...) with mode one of 'smlm', 'sofi', 'zstack', 'static', 'noise'; returns (movie [T, H, W]
float32 ADU, info dict with the truth: fc_px, NA, pixel_um, wavelength_um, mode)."""
import math
import numpy as np


def pupil_otf(N, pixel_um, NA, wavelength_um, z_um=0.0, n=1.518, astig_waves=0.0):
    """OTF [N, N] (unshifted FFT layout, OTF[0, 0] = 1) of a scalar circular pupil, defocus z_um (um in a medium of
    index n) and primary astigmatism (Noll 6 shape rho^2 cos 2 phi, peak-to-valley-ish amplitude astig_waves)."""
    k = np.fft.fftfreq(N) / pixel_um                                  # cycles / um
    KY, KX = np.meshgrid(k, k, indexing='ij')
    K = np.hypot(KX, KY)
    rad = NA / wavelength_um
    inside = K < rad
    ph = np.zeros_like(K)
    if z_um:
        ph += 2 * np.pi * z_um * np.sqrt(np.maximum((n / wavelength_um) ** 2 - K ** 2, 0))
    if astig_waves:
        rho2 = (K / rad) ** 2
        ph += 2 * np.pi * astig_waves * rho2 * np.cos(2 * np.arctan2(KY, KX))
    P = inside * np.exp(1j * ph)
    h = np.abs(np.fft.ifft2(P)) ** 2
    otf = np.fft.fft2(h)
    return otf / otf[0, 0]


def _filaments(N, rng, n_fil=5, width_px=0.0):
    img = np.zeros((N, N))
    for _ in range(n_fil):
        x, y = rng.uniform(0, N, 2)
        a = rng.uniform(0, 2 * np.pi)
        for _ in range(6 * N):
            a += 0.08 * rng.standard_normal()
            x, y = x + 0.5 * math.cos(a), y + 0.5 * math.sin(a)
            if not (0 <= x < N and 0 <= y < N):
                break
            img[int(y), int(x)] += 1
    return img


def synthetic_movie(mode='smlm', T=200, N=128, pixel_um=0.1, NA=1.2, wavelength_um=0.6, photons=2000.0, bg=20.0,
                    density=0.02, duty=0.01, mean_on=3.0, z_range=(-0.2, 0.2), nz=5, astig_waves=0.0,
                    gain=0.5, offset=100.0, readnoise=1.5, focus_range=(-1.0, 1.0), seed=0):
    """Synthetic movie with a known band limit (see the module docstring).

    smlm   : sites (density per px^2 of all sites), two-state blinking, stationary on-fraction `duty`, mean on-time
             mean_on frames, z uniform over nz layers in z_range (um); photons per emitter per frame
    sofi   : the same with dense sites and a high duty cycle (use e.g. density 0.3, duty 0.3)
    zstack : beads (density per px^2), all frames on, focus swept over focus_range (um) across the T frames
    static : a filament image, the same in every frame (only the noise changes)
    noise  : background only
    """
    rng = np.random.default_rng(seed)
    fc = 2 * NA / wavelength_um * pixel_um
    info = dict(mode=mode, fc_px=fc, NA=NA, pixel_um=pixel_um, wavelength_um=wavelength_um, T=T, N=N)
    sig = np.zeros((T, N, N))
    if mode in ('smlm', 'sofi'):
        ns = max(1, int(density * N * N))
        ys, xs = rng.integers(0, N, ns), rng.integers(0, N, ns)
        zl = np.linspace(*z_range, nz) if nz > 1 else np.array([0.5 * sum(z_range)])
        zi = rng.integers(0, len(zl), ns)
        otfs = [pupil_otf(N, pixel_um, NA, wavelength_um, z, astig_waves=astig_waves) for z in zl]
        p_off = 1 / mean_on
        p_on = duty / (1 - duty) * p_off
        st = rng.random(ns) < duty
        for t in range(T):
            u = rng.random(ns)
            st = np.where(st, u > p_off, u < p_on)
            I = photons * np.exp(0.3 * rng.standard_normal(ns)) * st
            F = 0
            for j, otf in enumerate(otfs):
                o = np.zeros((N, N))
                m = zi == j
                np.add.at(o, (ys[m], xs[m]), I[m])
                F = F + np.fft.fft2(o) * otf
            sig[t] = np.real(np.fft.ifft2(F))
    elif mode == 'zstack':
        ns = max(1, int(density * N * N))
        ys, xs = rng.integers(8, N - 8, ns), rng.integers(8, N - 8, ns)
        o = np.zeros((N, N)); np.add.at(o, (ys, xs), photons)
        Fo = np.fft.fft2(o)
        for t, f in enumerate(np.linspace(*focus_range, T)):
            sig[t] = np.real(np.fft.ifft2(Fo * pupil_otf(N, pixel_um, NA, wavelength_um, -f, astig_waves=astig_waves)))
        info['focus_um'] = np.linspace(*focus_range, T).tolist()
    elif mode == 'static':
        o = _filaments(N, rng) * photons * 0.05
        img = np.real(np.fft.ifft2(np.fft.fft2(o) * pupil_otf(N, pixel_um, NA, wavelength_um, 0.0,
                                                              astig_waves=astig_waves)))
        sig[:] = img
    elif mode != 'noise':
        raise ValueError(mode)
    mu = np.maximum(sig, 0) + bg
    ph = rng.poisson(mu).astype(np.float64)
    adu = ph / gain + offset + readnoise / gain * rng.standard_normal(ph.shape)
    return adu.astype(np.float32), info
