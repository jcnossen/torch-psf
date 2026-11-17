"""The optical band limit of a fluorescence microscope from the power spectrum of its images.

Incoherent imaging passes no spatial frequency above f_c = 2 NA / lambda (cycles per um): the OTF of the detection
optics is zero outside a disc of that radius, whatever the aberrations (they lower the OTF inside the disc, they do
not change its support). Camera noise (shot noise, read noise, sCMOS pixel noise) is white: it adds a flat floor to
the power spectrum at all frequencies. So the radially averaged power spectrum of almost any fluorescence image (or
movie) is a band-limited signal on a flat floor, and the frequency where the signal meets the floor is f_c.

The spectrum gives f_c in cycles per pixel only. With the pixel size it gives NA / lambda (= f_c / 2, cycles per um),
with the wavelength also the NA. What this NA is: the *effective* detection NA, i.e. the radius of the part of the
pupil that carries light of the imaged emitters:
  - emitters in a medium of index n_med < NA (cells in water, n 1.33, under an NA 1.49 oil objective) send light
    above n_med only by near-field coupling (supercritical-angle fluorescence), which decays within ~100-200 nm of
    the coverslip; deeper emitters fill the pupil only up to n_med, so f_c = 2 min(NA, n_med) / lambda;
  - apodization (pupil transmission falling towards the rim) lowers the measured cutoff a little (the OTF near the
    edge is too weak to see);
  - with defocused or strongly aberrated emitters (3D PSFs, tetrapods) the OTF near the edge is small and the
    estimate needs more signal.

Pipeline (power_spectrum, then fit_band_limit; band_limit runs both):
  1. frames are cut into square tiles, each tile minus its mean is multiplied by a 2D Hann window, |FFT|^2 / sum(w^2)
     (normalised so that white noise of variance s^2 per pixel gives a flat spectrum s^2); the spectra are averaged
     over tiles and over frames, in blocks of frames (for the bootstrap);
  2. the 2D spectrum is averaged in rings (radial profile, cycles / px); the lines kx = 0 and ky = 0 are left out
     (row / column noise of sCMOS cameras, EMCCD smear, tile-edge leakage), as are isolated sharp peaks (the
     illumination patterns of SIM / SIMFLUX, periodic camera noise);
  3. the profile is fitted with floor + exp(poly(f)) * OTF0(f / f_c)^2 (OTF0: the incoherent OTF of an aberration-free
     circular pupil; the polynomial absorbs the object spectrum, defocus, aberrations and pixel integration; the hard
     edge of OTF0 at f_c fixes the cutoff), weighted by the block-to-block standard error;
  4. uncertainty: bootstrap over the frame blocks (statistical) plus the spread over fit variants (fit range and
     polynomial degree: systematic); a model-free lower bound f_detect (the highest frequency where the excess over
     the floor is still significant) is reported next to the fit;
  5. what can and cannot be estimated is stated in BandLimit.status / .message: 'ok', 'weak' (signal too weak for a
     precise cutoff), 'undersampled' (f_c at or beyond Nyquist, 0.5 cycles / px: the pixel is too coarse; only a lower
     bound), 'edge_unresolved' (the signal decays smoothly into the floor, no edge identifiable: only a lower bound),
     'no_signal' (no band-limited excess over the noise floor), 'no_floor' (the floor region beyond the cutoff is not
     flat).
Also from the spectrum: the noise floor (ADU^2 per px; with the camera offset: the noise variance per ADU of signal),
the in-band signal-to-noise ratio, the flatness of the floor (correlated pixel noise, interpolated images), and the
anisotropy: the cutoff per direction (an elliptical support = anisotropic pixel scale, e.g. a cylindrical lens) and
the contrast anisotropy inside the band (astigmatism makes the OTF *contrast* anisotropic, not its support).
"""
import dataclasses
import math
import numpy as np
import torch
from scipy.optimize import least_squares

from torch_psf.optics import resolve_device


# ---------------------------------------------------------------------------------------------------------- spectrum

def otf_circ(nu):
    """Incoherent OTF of an aberration-free circular pupil at normalised frequency nu = f / f_c (0 outside)."""
    nu = np.clip(np.asarray(nu, np.float64), 0.0, 1.0)
    return (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu * nu))


def hann2d(S, device=None):
    w = torch.hann_window(S, periodic=False, dtype=torch.float64, device=device).float()
    return w[:, None] * w[None, :]


def tile_origins(H, W, S, max_tiles=16):
    """Top-left corners of up to max_tiles non-overlapping S x S tiles, spread evenly over an H x W frame."""
    ny, nx = max(1, H // S), max(1, W // S)
    ys = np.linspace(0, H - S, ny).round().astype(int)
    xs = np.linspace(0, W - S, nx).round().astype(int)
    o = [(y, x) for y in ys for x in xs]
    if len(o) > max_tiles:
        o = [o[i] for i in np.linspace(0, len(o) - 1, max_tiles).round().astype(int)]
    return o


@dataclasses.dataclass
class Spectrum:
    """Averaged power spectrum of frames. Frequencies in cycles / px; power normalised so that white noise of
    variance s^2 per pixel gives s^2 (ADU^2 for ADU data)."""
    f: np.ndarray            # [nb] ring centres
    p: np.ndarray            # [nb] mean power per ring
    se: np.ndarray           # [nb] standard error of p (from the block-to-block spread)
    blocks: np.ndarray       # [B, nb] ring profiles per block of frames (bootstrap units)
    count: np.ndarray        # [nb] number of 2D frequency samples per ring
    p2d: np.ndarray          # [S, S] mean 2D spectrum, fftshifted (DC at [S // 2, S // 2])
    mask2d: np.ndarray       # [S, S] bool, samples used (fftshifted)
    sectors: np.ndarray      # [nsec, nb] ring profiles per angular sector (angle of (fx, fy) mod pi)
    sector_blocks: np.ndarray  # [B, nsec, nb]
    sector_angles: np.ndarray  # [nsec] sector centres (rad, 0 = along x / columns)
    tile: int
    n_frames: int
    n_tiles: int
    mean_level: float        # mean value of the data (ADU)
    unit: str = 'blocks'     # what the bootstrap resamples: 'blocks' (of frames) or 'tiles'


@torch.no_grad()
def power_spectrum(frames, tile=None, max_tiles=16, nblocks=16, chunk=64, mask_axes=1, n_sectors=8, peak_mask=4.0,
                   bin_width=None, device=None):
    """Average power spectrum of frames [T, H, W] (numpy or torch, any real dtype; camera ADU or photons).

    tile: tile side (px; default min(H, W, 256), even); tiles: up to max_tiles per frame,
    non-overlapping, spread over the frame. nblocks: consecutive groups of frames whose spectra are kept apart for the
    standard errors and the bootstrap (with fewer frames than 4 blocks, the tiles are the bootstrap units). mask_axes:
    half-width (frequency samples) of the left-out lines kx = 0 / ky = 0. peak_mask: 2D samples above peak_mask x the
    ring median (of the mean spectrum, after a 3 x 3 median-free check) are left out as sharp peaks (None: keep all).
    Returns a Spectrum."""
    dev = resolve_device(device)
    x = frames if torch.is_tensor(frames) else torch.from_numpy(np.ascontiguousarray(frames))
    if x.dim() == 2:
        x = x[None]
    T, H, W = x.shape
    if tile is None:
        tile = min(H, W, 256) // 2 * 2
    S = int(tile)
    if S < 16:
        raise ValueError(f'frames too small for a spectrum ({H} x {W})')
    orig = tile_origins(H, W, S, max_tiles)
    w = hann2d(S, dev)
    wn = float((w ** 2).sum())
    single = T < 4
    nb_ = len(orig) if single else min(nblocks, T)
    edges = np.linspace(0, T, nb_ + 1).round().astype(int) if not single else None
    acc = torch.zeros(nb_, S, S, dtype=torch.float64, device=dev)
    nacc = np.zeros(nb_)
    msum = 0.0
    for t0 in range(0, T, chunk):
        c = x[t0:t0 + chunk].to(dev).float()
        msum += float(c.double().sum())
        tl = torch.stack([c[:, y:y + S, xx:xx + S] for y, xx in orig], 1)       # [n, k, S, S]
        tl = tl - tl.mean((-1, -2), keepdim=True)
        P = torch.fft.fft2(tl * w).abs() ** 2 / wn                             # [n, k, S, S]
        if single:
            acc += P.sum(0).double(); nacc += P.shape[0]
        else:
            blk = np.searchsorted(edges, np.arange(t0, t0 + c.shape[0]), side='right') - 1
            for b in np.unique(blk):
                m = torch.from_numpy(blk == b).to(dev)
                acc[b] += P[m].sum((0, 1)).double(); nacc[b] += int(m.sum()) * len(orig)
    acc = acc / torch.from_numpy(np.maximum(nacc, 1)).to(dev)[:, None, None]
    acc = torch.fft.fftshift(acc, dim=(-2, -1)).cpu().numpy()
    fr = (np.arange(S) - S // 2) / S
    FY, FX = np.meshgrid(fr, fr, indexing='ij')
    R = np.hypot(FX, FY)
    TH = np.mod(np.arctan2(FY, FX), np.pi)
    mask = (np.abs(FX) * S > mask_axes - 0.5 + 1e-9) & (np.abs(FY) * S > mask_axes - 0.5 + 1e-9) \
        if mask_axes else R > 0
    bw = bin_width or max(1.0 / S, 0.004)
    rb = np.floor(R / bw).astype(int)
    nbins = int(np.floor(np.sqrt(0.5) / bw))
    mask &= rb < nbins
    mean2d = acc.mean(0)
    if peak_mask:
        med = np.zeros(nbins)
        for b in range(nbins):
            m = mask & (rb == b)
            med[b] = np.median(mean2d[m]) if m.any() else np.inf
        mask &= ~(mean2d > peak_mask * med[np.minimum(rb, nbins - 1)])
    cnt = np.bincount(rb[mask], minlength=nbins)[:nbins].astype(float)
    def ring(a2):
        return np.bincount(rb[mask], weights=a2[mask], minlength=nbins)[:nbins] / np.maximum(cnt, 1)
    blocks = np.stack([ring(a) for a in acc])
    sec = np.minimum((TH / (np.pi / n_sectors) + 0.5).astype(int) % n_sectors, n_sectors - 1)
    def sring(a2, s):
        m = mask & (sec == s)
        c = np.bincount(rb[m], minlength=nbins)[:nbins]
        return np.bincount(rb[m], weights=a2[m], minlength=nbins)[:nbins] / np.maximum(c, 1)
    sblocks = np.stack([[sring(a, s) for s in range(n_sectors)] for a in acc])
    f = (np.arange(nbins) + 0.5) * bw
    ok = cnt >= 4
    B = len(blocks)
    p = blocks.mean(0)
    se = blocks.std(0, ddof=1) / np.sqrt(B) if B > 1 else p / np.sqrt(np.maximum(cnt, 1))
    return Spectrum(f=f[ok], p=p[ok], se=se[ok], blocks=blocks[:, ok], count=cnt[ok], p2d=mean2d, mask2d=mask,
                    sectors=sblocks.mean(0)[:, ok], sector_blocks=sblocks[:, :, ok],
                    sector_angles=np.arange(n_sectors) * np.pi / n_sectors, tile=S, n_frames=T, n_tiles=len(orig),
                    mean_level=msum / (T * H * W), unit='tiles' if single else 'blocks')


# --------------------------------------------------------------------------------------------------------- the fit

def _model(theta, f, deg):
    a = theta[:deg + 1]
    fc, b = np.exp(theta[deg + 1]), np.exp(theta[deg + 2])
    u = f / 0.5
    s = np.polyval(a[::-1], u)
    return b + np.exp(np.clip(s, -60, 60)) * otf_circ(f / fc) ** 2


def _fit(f, p, se, deg, fc0, b0, theta0=None, fc_max=0.95):
    """Weighted least-squares fit of floor + exp(poly) * OTF0(f / f_c)^2; returns (theta, chi2)."""
    if theta0 is None:
        exc = np.maximum(p - b0, 1e-6 * b0)
        nu = np.clip(f / fc0, 0, 0.95)
        m = f < 0.9 * fc0
        if m.sum() < deg + 2:
            m = f <= np.sort(f)[deg + 2]
        y = np.log(exc[m]) - 2 * np.log(np.maximum(otf_circ(nu[m]), 1e-6))
        a = np.polyfit(f[m] / 0.5, y, deg)[::-1]
        theta0 = np.concatenate([a, [math.log(fc0), math.log(max(b0, 1e-12))]])
    fn = lambda th: (_model(th, f, deg) - p) / se
    lo = np.full(deg + 3, -np.inf); hi = np.full(deg + 3, np.inf)
    lo[deg + 1], hi[deg + 1] = math.log(0.03), math.log(fc_max)
    th0 = np.clip(theta0, lo + 1e-9, hi - 1e-9)
    try:
        r = least_squares(fn, th0, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=400)
    except Exception:
        return theta0, np.inf
    return r.x, float((r.fun ** 2).sum())


@dataclasses.dataclass
class BandLimit:
    """Result of fit_band_limit. Frequencies: fc_px in cycles / px; fc_per_um = 2 NA / lambda in cycles / um;
    na_over_lambda = NA / lambda (1 / um); NA (when lambda is given). *_err: 1-sigma total (statistical from the
    bootstrap and systematic from the fit variants, in quadrature); *_lo / *_hi: the 16 / 84 % bootstrap range."""
    status: str
    message: str
    fc_px: float
    fc_px_err: float
    fc_px_stat: float
    fc_px_sys: float
    fc_px_lo: float
    fc_px_hi: float
    f_detect_px: float               # model-free lower bound (last significant ring)
    floor: float                     # noise floor (data units^2 per px)
    floor_slope: float               # relative change of the floor per cycle / px over the floor region
    band_snr: float                  # mean (p - floor) / floor over 0.1 .. 0.5 f_c
    dchi2: float                     # chi2(flat floor) - chi2(fit): significance of the band-limited excess
    hf_contrast: float               # signal factor at 0.75 f_c / at 0.25 f_c, relative to an ideal pupil's OTF^2
    chi2_red: float
    fit_range: tuple
    theta: np.ndarray
    deg: int
    pixel_um: float = None
    wavelength_um: float = None
    fc_per_um: float = None
    fc_per_um_err: float = None
    na_over_lambda: float = None
    na_over_lambda_err: float = None
    NA: float = None
    NA_err: float = None
    na_over_lambda_lower: float = None   # from f_detect (model-free lower bound)
    NA_lower: float = None
    boot: np.ndarray = None
    variants: dict = None
    anisotropy: dict = None

    def model(self, f):
        return _model(self.theta, np.asarray(f, np.float64), self.deg)

    def summary(self):
        if self.status in ('edge_unresolved', 'no_signal') or (self.status == 'weak' and self.dchi2 < 50) or \
                not np.isfinite(self.fc_px_err):
            s = f'f_c > {self.f_detect_px:.4f} cycles/px (lower bound) [{self.status}]'
            if self.na_over_lambda_lower is not None:
                s += f'; NA/lambda > {self.na_over_lambda_lower:.3f} /um'
            if self.NA_lower is not None:
                s += f'; NA > {self.NA_lower:.3f}'
            return s
        s = f'f_c = {self.fc_px:.4f} +- {self.fc_px_err:.4f} cycles/px [{self.status}]'
        if self.na_over_lambda is not None:
            s += f'; NA/lambda = {self.na_over_lambda:.3f} +- {self.na_over_lambda_err:.3f} /um'
        if self.NA is not None:
            s += f'; NA = {self.NA:.3f} +- {self.NA_err:.3f}'
        return s

    def to_dict(self):
        d = {k: v for k, v in dataclasses.asdict(self).items() if k not in ('theta', 'boot')}
        d['theta'] = [float(v) for v in self.theta]
        return _jsonable(d)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _best_fit(f, p, se, deg, b0, fc_grid, fc_max=0.95):
    best = None
    for fc0 in fc_grid:
        th, c2 = _fit(f, p, se, deg, min(fc0, 0.97 * fc_max), b0, fc_max=fc_max)
        if best is None or c2 < best[1]:
            best = (th, c2)
    return best


def _floor_guess(f, p):
    hi = f > 0.6
    if hi.sum() >= 5:
        return float(np.median(p[hi]))
    return float(np.min(np.convolve(p, np.ones(5) / 5, 'valid')))


def fit_band_limit(spec, pixel_um=None, wavelength_um=None, deg=2, n_boot=100, f_lo=0.02, seed=0,
                   anisotropy=True, snr_min=0.005, rel_err_ok=0.05, dchi2_min=50.0, detect_frac=0.85,
                   floor_sys=0.01):
    """Fit the cutoff of a Spectrum. pixel_um / wavelength_um (optional) convert f_c to NA / lambda and NA.

    Two passes: (1) over f >= f_lo from a grid of start values of f_c, (2) over f >= 0.3 f_c of pass 1 (the low
    frequencies carry the object spectrum, which the polynomial need not describe). Bootstrap: n_boot resamplings of
    the blocks. Fit variants (systematic error, rms deviation from the main fit): lower fit limit 0.2 / 0.3 / 0.45 f_c with
    degree 2, 0.2 / 0.3 f_c with degree 3.
    Status, in this order: 'no_signal' (no significant excess starting below 0.1 cycles / px, or band SNR < snr_min);
    'weak' (a flat floor fits almost as well: chi2 difference < dchi2_min; lower bound f_detect only);
    'edge_unresolved' (> 20 % of the bootstrap / variant fits run to the end of the spectrum, or significant signal
    beyond the fitted edge, or the edge far beyond the last significant ring, f_detect < detect_frac f_c (in-focus
    simulations give f_detect / f_c = 0.88 .. 0.99), or f_c > 0.5 without visible signal there; lower bound f_detect
    only); 'undersampled'
    (signal beyond Nyquist: f_c from the diagonal corners of the spectrum, aliased along the axes); 'no_floor' (the
    rings beyond 1.1 f_c are not flat); 'weak' (relative error > 2 rel_err_ok); else 'ok'."""
    f, p, se = spec.f, spec.p, np.maximum(spec.se, 1e-12 * np.abs(spec.p).max())
    m0 = f >= f_lo
    b0 = _floor_guess(f, p)
    fmax_fit = float(f.max())
    grid = np.arange(0.1, fmax_fit, 0.05)
    th1, _ = _best_fit(f[m0], p[m0], se[m0], deg, b0, grid, fc_max=fmax_fit)
    fc1 = math.exp(th1[deg + 1])

    def fit_range(fc, frac, d, th0=None, pp=None):
        m = f >= max(f_lo, frac * fc)
        pp = p if pp is None else pp
        if th0 is None:
            return _best_fit(f[m], pp[m], se[m], d, b0, [fc * k for k in (0.85, 1.0, 1.15)], fc_max=fmax_fit)
        return _fit(f[m], pp[m], se[m], d, fc, b0, th0, fc_max=fmax_fit)

    th, c2 = fit_range(fc1, 0.3, deg)
    fc = math.exp(th[deg + 1]); b = math.exp(th[deg + 2])
    mfit = f >= max(f_lo, 0.3 * fc)
    dof = max(int(mfit.sum()) - (deg + 3), 1)
    chi2_red = c2 / dof

    # model-free lower bound: end of the first run of rings (from low frequencies up) where a 5-ring running mean of
    # the excess over the floor is > 4 sigma (the run ends at 5 consecutive rings below)
    # noise-only standard error of a ring (an averaged periodogram sample has std = mean): the block spread would
    # also count real frame-to-frame changes of the signal (z-stacks), which are not noise
    # also count real frame-to-frame changes of the signal (z-stacks), which are not noise; plus a systematic
    # floor_sys (fraction of the floor) for the imperfect whiteness of real camera noise (~1-3 % in the data seen)
    se_det = np.hypot(p / np.sqrt(np.maximum(spec.count * spec.n_frames * spec.n_tiles, 1)), floor_sys * b)
    z = (p - b) / np.maximum(se_det, 1e-30)
    zs = np.convolve(z, np.ones(5), 'same') / np.sqrt(5)
    above = zs > 4
    f_detect = 0.0
    for i in range(len(f)):
        if above[i]:
            f_detect = float(f[i])
        elif (f_detect > 0 and not above[i:i + 5].any()) or (f_detect == 0 and f[i] > 0.1):
            break                                   # the run must start below 0.1 cycles / px

    # significance of the band-limited excess: chi2 of a flat floor minus chi2 of the fit, over the fit range
    wf = 1 / se[mfit] ** 2
    flat = float((p[mfit] * wf).sum() / wf.sum())
    dchi2 = float((((p[mfit] - flat) / se[mfit]) ** 2).sum() - c2)

    band = (f > 0.1 * fc) & (f < 0.5 * fc)
    band_snr = float(np.mean((p[band] - b) / b)) if band.any() else 0.0
    zmax = float(zs[f < fc].max()) if (f < fc).any() else 0.0

    # bootstrap over blocks
    rng = np.random.default_rng(seed)
    B = spec.blocks.shape[0]
    boot = []
    if B >= 4 and n_boot:
        for _ in range(n_boot):
            pb = spec.blocks[rng.integers(0, B, B)].mean(0)
            tb, _ = fit_range(fc, 0.3, deg, th0=th, pp=pb)
            boot.append(math.exp(tb[deg + 1]))
    boot = np.array(boot)
    if len(boot) > 3:
        fc_stat = float(np.std(boot, ddof=1))
        lo_, hi_ = np.percentile(boot, [16, 84])
    else:
        fc_stat, lo_, hi_ = float('nan'), float('nan'), float('nan')

    variants = {}
    for frac, d in ((0.2, 2), (0.3, 2), (0.45, 2), (0.2, 3), (0.3, 3)):
        tv, _ = fit_range(fc, frac, d)
        variants[f'flo{frac:g}_deg{d}'] = math.exp(tv[d + 1])
    # identifiability: fits that run to the upper end of the spectrum mean that a smooth decay into the floor
    # explains the data as well as an edge (the edge is hidden in the noise)
    at_cap = lambda v: v > 0.97 * fmax_fit
    all_fc = np.concatenate([boot, list(variants.values()), [fc]])
    frac_cap = float(np.mean([at_cap(v) for v in all_fc]))
    vv = np.array([v for v in variants.values() if not at_cap(v)])
    fc_sys = float(np.sqrt(np.mean((vv - fc) ** 2))) if len(vv) > 2 else float('nan')
    fc_err = float(np.hypot(fc_stat if np.isfinite(fc_stat) else 0.0, fc_sys if np.isfinite(fc_sys) else 0.0))
    if not (np.isfinite(fc_stat) or np.isfinite(fc_sys)):
        fc_err = float('nan')

    # floor flatness: line through the rings above 1.08 f_c
    fl = f > 1.1 * fc
    floor_slope_z = 0.0
    if fl.sum() >= 8:
        A = np.stack([np.ones(fl.sum()), f[fl] - f[fl].mean()], 1)
        wts = 1 / se[fl]
        coef, *_ = np.linalg.lstsq(A * wts[:, None], p[fl] * wts, rcond=None)
        cov = np.linalg.pinv((A * wts[:, None]).T @ (A * wts[:, None]))
        floor_slope = float(coef[1] / coef[0])
        floor_slope_z = float(coef[1] / math.sqrt(max(cov[1, 1], 1e-300)))
    else:
        floor_slope = float('nan')

    fmax = float(f.max())
    # high-frequency contrast relative to an ideal pupil: the fitted signal factor exp(poly) at 0.75 f_c over 0.25 f_c
    # (diagnostic only; ~0.2-0.6 for point emitters with in-focus / astigmatic PSFs incl. pixel integration)
    a_ = th[:deg + 1][::-1]
    hf_contrast = float(np.exp(np.clip(np.polyval(a_, 0.75 * fc / 0.5) - np.polyval(a_, 0.25 * fc / 0.5), -50, 50)))
    # the fitted edge must sit where the signal is seen to end: significant signal beyond it, or an edge far out in
    # the noise (f_detect < detect_frac f_c: the fit extrapolates a smooth decay), both mean it is not identified
    fc_conflict = (f_detect > fc + max(2 * fc_err if np.isfinite(fc_err) else 0, 0.01)) or f_detect < detect_frac * fc
    if f_detect == 0.0 or (dchi2 < dchi2_min and f_detect < 0.05) or (band_snr < snr_min and zmax < 6):
        status, msg = 'no_signal', ('no band-limited signal above the noise floor: the images carry no detectable '
                                    'structure (dark frames, flat illumination or far too few photons)')
    elif band_snr < 0.1 and (frac_cap > 0.2 or fc_conflict or fc > 0.5):
        status = 'no_signal'
        msg = (f'no band-limited signal: only a weak, non-white structure (band SNR {band_snr:.3g}) without a cutoff '
               f'edge: fixed-pattern or correlated camera noise, flat illumination; not an optical image')
    elif dchi2 < dchi2_min:
        status = 'weak'
        msg = (f'band-limited signal only at low frequencies (up to {f_detect:.3f} cycles/px); far too weak for '
               f'the cutoff: lower bound f_c > {f_detect:.3f} cycles/px')
    elif frac_cap > 0.2 or fc_conflict or (fc > 0.5 and f_detect < 0.45):
        status = 'edge_unresolved' if f_detect < 0.45 else 'undersampled'
        msg = (f'no cutoff edge identified ({100 * frac_cap:.0f} % of the bootstrap / variant fits find no edge, or '
               f'the fitted edge disagrees with the last significant ring): only a lower bound f_c > '
               f'{f_detect:.3f} cycles/px; ' +
               ('the signal decays into the noise floor before the edge shows (defocused or strongly aberrated '
                'PSFs, e.g. a bead stack or a 3D PSF, or too little signal)' if f_detect < 0.45 else
                'the signal reaches Nyquist (0.5): the pixel may be too coarse, or the floor is not white'))
        fc_err = float('nan')
    elif fc > 0.5 or fl.sum() < 8 or fc > fmax - 0.04:
        status = 'undersampled'
        msg = (f'the signal reaches {f_detect:.3f} cycles/px, beyond Nyquist (0.5): the pixel is too coarse for '
               f'Nyquist sampling; f_c = {fc:.3f} comes from the diagonal corners of the spectrum only (aliased along '
               f'the axes), treat it as approximate; at f_c > 0.7 only the lower bound {f_detect:.3f} remains')
    elif not np.isfinite(floor_slope) or (abs(floor_slope) * (fmax - 1.1 * fc) > 0.02 and abs(floor_slope_z) > 5):
        status = 'no_floor'
        msg = (f'the spectrum beyond the cutoff is not flat (relative slope {floor_slope:+.2f} per cycle/px): '
               'correlated pixel noise, an interpolated / filtered image or signal beyond the fitted cutoff; '
               'treat f_c as uncertain')
    elif not np.isfinite(fc_err) or fc_err > 2 * rel_err_ok * fc:
        status = 'weak'
        msg = (f'signal too weak for a precise cutoff (band SNR {band_snr:.3g}, relative error '
               f'{fc_err / fc:.1%}); the model-free lower bound is {f_detect:.3f} cycles/px')
    else:
        status, msg = 'ok', f'cutoff found, relative error {fc_err / fc:.1%}, band SNR {band_snr:.3g}'

    res = BandLimit(status=status, message=msg, fc_px=fc, fc_px_err=fc_err, fc_px_stat=fc_stat, fc_px_sys=fc_sys,
                    fc_px_lo=float(lo_), fc_px_hi=float(hi_), f_detect_px=f_detect, floor=b, floor_slope=floor_slope,
                    band_snr=band_snr, dchi2=dchi2, hf_contrast=hf_contrast, chi2_red=chi2_red,
                    fit_range=(float(max(f_lo, 0.3 * fc)), fmax), theta=th, deg=deg, boot=boot, variants=variants)
    if pixel_um:
        res.pixel_um = float(pixel_um)
        res.fc_per_um = fc / pixel_um
        res.fc_per_um_err = fc_err / pixel_um
        res.na_over_lambda = res.fc_per_um / 2
        res.na_over_lambda_err = res.fc_per_um_err / 2
        if wavelength_um:
            res.wavelength_um = float(wavelength_um)
            res.NA = res.na_over_lambda * wavelength_um
            res.NA_err = res.na_over_lambda_err * wavelength_um
            res.NA_lower = f_detect / pixel_um / 2 * wavelength_um
        res.na_over_lambda_lower = f_detect / pixel_um / 2
    if status == 'edge_unresolved':
        res.fc_px_err = float('nan')
    if anisotropy and status != 'no_signal':
        # without an identified edge, the sectors use f_detect / 0.9 as the reference cutoff
        ref = res if np.isfinite(res.fc_px_err) else dataclasses.replace(res, fc_px=max(f_detect, 0.05) / 0.9)
        res.anisotropy = sector_anisotropy(spec, ref)
    return res


def sector_anisotropy(spec, res):
    """Direction dependence of the spectrum (sectors of spec): the cutoff per direction (floor fixed at the global
    fit), an ellipse fitted to it (1 / f_c(theta)^2 = a + b cos 2 theta + c sin 2 theta: axis ratio and angle of the
    long axis; an elliptical support in pixel frequencies = different pixel scales along two axes), and the contrast
    modulation of the in-band excess power over 0.25 .. 0.75 f_c (m2 = |sum E(theta) e^{2 i theta}| / sum E: 0 for an
    isotropic OTF; astigmatism, a line-shaped object or drift give m2 > 0), with its bootstrap standard error over
    blocks."""
    deg = res.deg
    b = res.floor
    fc = res.fc_px
    f = spec.f
    nsec = len(spec.sector_angles)
    fcs, errs = [], []
    m = f >= 0.3 * fc
    for s in range(nsec):
        ps = spec.sectors[s]
        ses = spec.sector_blocks[:, s].std(0, ddof=1) / np.sqrt(len(spec.sector_blocks)) \
            if len(spec.sector_blocks) > 1 else ps / np.sqrt(spec.count / nsec)
        ses = np.maximum(ses, 1e-12 * np.abs(ps).max())
        th0 = res.theta.copy()
        fn = lambda t: (_model(np.concatenate([t[:deg + 2], [math.log(b)]]), f[m], deg) - ps[m]) / ses[m]
        try:
            r = least_squares(fn, th0[:deg + 2], method='trf', x_scale='jac', max_nfev=300)
            fcs.append(math.exp(r.x[deg + 1]))
            J = r.jac
            cov = np.linalg.pinv(J.T @ J) * max(1.0, (r.fun ** 2).sum() / max(len(r.fun) - deg - 2, 1))
            errs.append(math.exp(r.x[deg + 1]) * math.sqrt(max(cov[deg + 1, deg + 1], 0)))
        except Exception:
            fcs.append(float('nan')); errs.append(float('nan'))
    fcs, errs = np.array(fcs), np.array(errs)
    ang = spec.sector_angles
    ok = np.isfinite(fcs) & np.isfinite(errs) & (errs > 0) & (np.abs(fcs / fc - 1) < 0.3)
    out = dict(angles_deg=np.degrees(ang).tolist(), fc_px=fcs.tolist(), fc_px_err=errs.tolist())
    if ok.sum() >= 4:
        y = 1 / fcs[ok] ** 2
        wy = 1 / (2 * errs[ok] / fcs[ok] ** 3)
        A = np.stack([np.ones(ok.sum()), np.cos(2 * ang[ok]), np.sin(2 * ang[ok])], 1)
        cf = np.linalg.lstsq(A * wy[:, None], y * wy, rcond=None)[0]
        a0, bc, bs = cf
        amp = math.hypot(bc, bs)
        fmax_, fmin_ = 1 / math.sqrt(max(a0 - amp, 1e-12)), 1 / math.sqrt(a0 + amp)
        # 1/fc^2 smallest along the long axis: angle where cos(2(theta - phi)) = -1
        phi = 0.5 * math.atan2(bs, bc) + math.pi / 2
        if fmax_ / fmin_ < 1.5 and a0 > amp:
            out.update(ellipse_ratio=fmax_ / fmin_, ellipse_long_axis_deg=math.degrees(phi % math.pi),
                       fc_long_px=fmax_, fc_short_px=fmin_)
    band = (f > 0.25 * fc) & (f < 0.75 * fc)
    def m2(sec):
        E = np.maximum((sec[:, band] - b).sum(1), 0)
        z = (E * np.exp(2j * ang)).sum() / max(E.sum(), 1e-30)
        return z
    z = m2(spec.sectors)
    zb = np.array([m2(sb) for sb in spec.sector_blocks]) if len(spec.sector_blocks) > 1 else np.array([z])
    out.update(contrast_m2=float(abs(z)), contrast_angle_deg=float(np.degrees(np.angle(z) / 2) % 180),
               contrast_m2_se=float(np.std(np.abs(zb - z)) / math.sqrt(max(len(zb) - 1, 1)) * math.sqrt(2)),
               contrast_m2_blocks=np.abs(zb).tolist())
    return out


def band_limit(frames, pixel_um=None, wavelength_um=None, spectrum_kw=None, **fit_kw):
    """power_spectrum(frames, **spectrum_kw) then fit_band_limit; returns (BandLimit, Spectrum)."""
    spec = power_spectrum(frames, **(spectrum_kw or {}))
    return fit_band_limit(spec, pixel_um=pixel_um, wavelength_um=wavelength_um, **fit_kw), spec


def frame_blocks(n, nblocks=8, block_len=64, start_frac=0.0):
    """Frame indices of nblocks contiguous blocks of block_len frames spread over frames [start_frac n, n) (all
    frames if they fit)."""
    s0 = int(start_frac * n)
    if (n - s0) <= nblocks * block_len:
        return np.arange(s0, n)
    starts = np.linspace(s0, n - block_len, nblocks).round().astype(int)
    return np.concatenate([np.arange(a, a + block_len) for a in starts])


def read_frames(movie, idx, y0=0, x0=0, H=None, W=None):
    """Frames idx of a movie reader (torch_psf.movie interface) as float32 [len(idx), H, W]; contiguous runs are read
    in one call. Crop: rows y0 .. y0 + H, columns x0 .. x0 + W."""
    idx = np.asarray(idx)
    runs = np.split(idx, np.nonzero(np.diff(idx) != 1)[0] + 1)
    out = []
    for r in runs:
        a = movie.read(int(r[0]), int(r[-1]) + 1)
        out.append(a[:, y0:None if H is None else y0 + H, x0:None if W is None else x0 + W])
    return np.concatenate(out).astype(np.float32, copy=False)
