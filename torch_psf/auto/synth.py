"""Cheap synthetic movies with an exactly known band limit, for tests and calibration of the auto analysis.

Imaging is done in Fourier space: frame = IFFT(FFT(object) x OTF), with the OTF of a scalar circular pupil of radius
NA / lambda (cycles / um), optionally defocused (high-NA defocus phase 2 pi z sqrt((n / lambda)^2 - k^2)) and with
primary astigmatism; so the support of every frame's spectrum is exactly f_c = 2 NA / lambda. The object is a set of
point emitters on the pixel grid (or a static filament image). Poisson noise on (signal + background), then
ADU = photons / gain + offset + Gaussian read noise. Small and fast (CPU), deterministic for a given seed.

synthetic_movie(mode, ...) with mode one of 'smlm', 'sofi', 'zstack', 'static', 'noise'; returns (movie [T, H, W]
float32 ADU, info dict with the truth: fc_px, NA, pixel_um, wavelength_um, mode).

Two more generators, for the non-standard routes:
  scene_movie   a handheld camera filming the world (not a microscope): a "dead leaves" scene (occluding disks of
                power-law sizes, the classic model of natural-image statistics, P(f) ~ f^-2), camera shake and
                pans of many px per frame, small zoom changes, autofocus hunting (blur that changes over frames),
                gamma, sharpening, 8-bit quantisation with clipping
  moving_movie  a microscope sample that moves: rod-shaped cells (fluorescent, or dark on a bright field), stage
                panning, cells swimming, slow refocusing; exact band limit as above (incoherent imaging; the
                bright-field variant is the same image with inverted contrast, not a partially coherent model)"""
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


def _dead_leaves(M, rng, n=None, rmin=1.5, rmax=None):
    """Dead-leaves image [M, M] in [0, 1]: disks with radius density ~ r^-3 (scale invariant), random grey levels,
    later disks occlude earlier ones; plus a little texture inside the disks."""
    rmax = rmax or M / 4
    n = n or int(12 * (M / rmin) ** 2 / np.log(rmax / rmin) / 50)
    u = rng.random(n)
    r = 1.0 / np.sqrt(u / rmin ** 2 + (1 - u) / rmax ** 2)           # p(r) ~ r^-3 on [rmin, rmax]
    img = np.full((M, M), 0.5)
    yy, xx = np.mgrid[0:M, 0:M]
    for k in np.argsort(-r):                                           # big first, small (front) last
        cy, cx = rng.uniform(-rmax, M + rmax, 2)
        R = r[k]
        y0, y1 = int(max(0, cy - R)), int(min(M, cy + R + 1))
        x0, x1 = int(max(0, cx - R)), int(min(M, cx + R + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        m = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2 < R * R
        img[y0:y1, x0:x1][m] = rng.uniform(0.05, 0.95)
    return np.clip(img + 0.02 * rng.standard_normal((M, M)), 0, 1)


def _gauss_blur_fft(img, sigma):
    if sigma <= 0:
        return img
    ky = np.fft.fftfreq(img.shape[0])[:, None]; kx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (kx ** 2 + ky ** 2))))


def scene_movie(T=40, N=160, step_px=(3.0, 40.0), zoom=0.02, blur=(0.5, 2.0), noise=2.0, seed=0):
    """Handheld camera movie of a dead-leaves scene (see the module docstring): [T, N, N] float32 in 0..255
    (8-bit levels) and an info dict."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(seed)
    M = 4 * N
    world = _dead_leaves(M, rng)
    pos = np.array([M / 2, M / 2]); scale = 1.0
    s_blur = blur[0] + (blur[1] - blur[0]) * (0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, T) + rng.uniform(0, 6)))
    yy, xx = np.mgrid[0:N, 0:N] - N / 2
    out = np.zeros((T, N, N), np.float32); path = []
    for t in range(T):
        a = rng.uniform(0, 2 * np.pi)
        pos = pos + rng.uniform(*step_px) * np.array([np.sin(a), np.cos(a)])
        pos = np.clip(pos, N, M - N)
        scale *= np.exp(zoom * rng.standard_normal())
        rot = 0.01 * rng.standard_normal()
        c, s_ = np.cos(rot), np.sin(rot)
        Y = pos[0] + scale * (c * yy - s_ * xx); X = pos[1] + scale * (s_ * yy + c * xx)
        f = map_coordinates(world, [Y, X], order=1, mode='reflect')
        f = _gauss_blur_fft(f, s_blur[t])                              # lens + demosaic + autofocus hunting
        f = f + 0.6 * (f - _gauss_blur_fft(f, 1.5))                    # in-camera sharpening
        f = 255 * np.clip(f, 0, 1) ** (1 / 2.2)
        f = f + noise * rng.standard_normal(f.shape)
        out[t] = np.clip(np.round(f), 0, 255); path.append(pos.copy())
    return out, dict(mode='scene', T=T, N=N, path=np.asarray(path).tolist(), blur_px=s_blur.tolist())


def moving_movie(T=60, N=128, pixel_um=0.1, NA=0.9, wavelength_um=0.55, brightfield=False, pan_px=3.0,
                 swim_px=1.0, focus_range=(0.0, 1.0), n_cells=60, photons=300.0, bg=10.0, seed=0):
    """Microscopy movie of rod-shaped cells with stage panning (random walk, pan_px per frame), swimming
    (swim_px per frame per cell) and a slow focus change over focus_range (um); exact band limit 2 NA / lambda.
    [T, N, N] float32 (photon counts with Poisson noise) and an info dict."""
    rng = np.random.default_rng(seed)
    G = N + 64                                                         # canvas with margins for panning
    ang = rng.uniform(0, np.pi, n_cells)
    Lc = rng.uniform(1.5, 4.0, n_cells) / pixel_um
    stage = np.array([2 * G - G / 2, 2 * G - G / 2], float)
    cy, cx = rng.uniform(G, 3 * G, n_cells), rng.uniform(G, 3 * G, n_cells)  # 1/4 of them in view at the start
    out = np.zeros((T, N, N), np.float32)
    focus = np.linspace(*focus_range, T)
    fc = 2 * NA / wavelength_um * pixel_um
    for t in range(T):
        a = rng.uniform(0, 2 * np.pi)
        stage += pan_px * rng.standard_normal(2)
        cy += swim_px * np.sin(ang) * rng.uniform(0.5, 1.5, n_cells)
        cx += swim_px * np.cos(ang) * rng.uniform(0.5, 1.5, n_cells)
        o = np.zeros((G, G))
        for k in range(n_cells):
            s = np.linspace(-Lc[k] / 2, Lc[k] / 2, max(2, int(2 * Lc[k])))
            for wdt in np.linspace(-0.35, 0.35, 5) / pixel_um:          # ~0.7 um wide rods
                y = cy[k] + s * np.sin(ang[k]) + wdt * np.cos(ang[k]) - stage[0]
                x = cx[k] + s * np.cos(ang[k]) - wdt * np.sin(ang[k]) - stage[1]
                ok = (y >= 0) & (y < G - 1) & (x >= 0) & (x < G - 1)
                yi, xi = np.floor(y[ok]).astype(int), np.floor(x[ok]).astype(int)
                fy, fx = y[ok] - yi, x[ok] - xi
                for dy, dx, wgt in ((0, 0, (1 - fy) * (1 - fx)), (1, 0, fy * (1 - fx)), (0, 1, (1 - fy) * fx),
                                    (1, 1, fy * fx)):
                    np.add.at(o, (yi + dy, xi + dx), wgt)
        img = np.real(np.fft.ifft2(np.fft.fft2(o) * pupil_otf(G, pixel_um, NA, wavelength_um, focus[t])))
        img = img[32:32 + N, 32:32 + N] / 5.0
        mu = bg + photons * img if not brightfield else photons * np.clip(1 - 0.5 * img, 0.05, None)
        out[t] = rng.poisson(np.maximum(mu, 0))
    return out, dict(mode='moving', fc_px=fc, NA=NA, pixel_um=pixel_um, wavelength_um=wavelength_um, T=T, N=N,
                     brightfield=brightfield, focus_um=focus.tolist())
