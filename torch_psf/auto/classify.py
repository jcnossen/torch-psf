"""Which PSF-estimation route fits a movie: a rule-based classifier on simple, explainable statistics.

Modes:
  smlm       sparse, isolated spots that blink (appear and disappear between frames): in-situ PSF from the spots
  sofi       strong temporal fluctuations everywhere, spatially correlated over about a PSF, spots overlap:
             PSF from the fluctuation cumulants (SOFI route)
  zstack     the sharpness changes systematically with the frame (beads / a thin object going through focus):
             bead z-stack or phase-diversity route
  static     a static or slowly varying scene: only the band limit (NA / lambda) from the spectrum is safe, plus a
             blind estimate with its ambiguities
  no_signal  no band-limited signal: dark frames, flat illumination, not a fluorescence image, far too few photons
  scene      not a microscope image: a camera filming the world (natural-scene statistics: power spectrum ~ f^-2,
             structure everywhere, no dark background, no band edge below Nyquist); only the camera's own
             resolution (MTF) can be measured, not an NA
  moving     a microscope sample or stage that moves (panning, swimming cells): consistent global shifts between
             frames; register the frames first, then the band limit, and a blind / phase-diversity estimate if the
             focus also changes

Statistics (movie_statistics; all on a band-passed copy B of the frames, a difference of Gaussians matched to the PSF
size implied by the cutoff f_c, so that slow background and white noise are both suppressed):
  band_snr          in-band excess power over the noise floor (spectrum, torch_psf.auto.spectrum)
  noise_var         variance of the white noise in B (from the spectrum floor and the filter's power gain)
  dyn_snr           temporal variance of B beyond the noise / noise (fluctuating, band-limited signal)
  static_snr        power of the temporal mean of B beyond the noise / noise (static, band-limited signal)
  dyn_fraction      dynamic / (dynamic + static) band-limited power
  rho[L]            lag-L autocorrelation of the fluctuations of B (noise-free for L >= 1)
  kurtosis          median normalised 4th cumulant k4 / k2^2 of the fluctuations over the most active pixels
                    (on/off blinking of few emitters: large; many overlapping emitters: small; slow motion: ~0)
  spots_per_psf     detected spots per frame per PSF area (pi r_airy^2), r_airy = 1.22 / f_c px
  isolated          fraction of spots without another spot within 2 r_airy in the same frame
  persist[L]        fraction of spots with a spot within 1.5 px at frame t + L, chance level removed (0 = new
                    spots every L frames, 1 = the same spots); persist_wide[L]: the same within 2 r_airy
  sharpness         per frame: high-band fraction (0.35 .. 0.8 f_c of 0.05 .. 0.8 f_c) of the floor-subtracted frame spectrum; its smooth trend
                    over the frames (trend_r2: variance fraction explained by a running mean; trend_range: relative
                    range of the trend; peak_pos: position of its maximum, 0..1)
  astig_swing       z-stacks: range of the signed in-band contrast anisotropy (cos 2 theta) over the frames, i.e.
                    spots elongated one way above focus and the other way below: astigmatism
  aniso_frame_rms   robust (median-based) rms frame-to-frame spread of the complex in-band anisotropy (SMLM: astigmatic emitters at
                    different z make single frames anisotropic even when the average cancels; also raised by few,
                    interfering emitters per frame, so only comparable at similar densities)
  motion_*          global frame-to-frame translation (torch_psf.auto.motion.global_motion / motion_summary):
                    motion_conf (fraction of steps with a strong, unique, self-consistent correlation peak),
                    motion_step_px, motion_moving (fraction of those that moved), motion_path_px
  spectral_slope, dark_frac, flat_frac   natural-scene cues (torch_psf.auto.motion.scene_statistics)
Large frames: the motion and scene statistics use the whole frame (downsampled) and a central 512 px crop; all the
others a central crop of at most max_side x max_side px (memory; the statistics are per area or per pixel).
classify_movie turns them into the mode, a confidence (the winning class probability from soft rules), the key
numbers and a plain-language reason.
"""
import dataclasses
import math
import numpy as np
import torch
import torch.nn.functional as Fn

from torch_psf.optics import resolve_device
from torch_psf.detect import blur
from torch_psf.auto.spectrum import hann2d, power_spectrum, fit_band_limit, _jsonable
from torch_psf.auto.motion import global_motion, motion_summary, scene_statistics


def _soft(x, thr, width=0.35):
    """Smooth step in log(x): 0.5 at x = thr, ~0.9 at x = thr * exp(2.2 width)."""
    if x is None or not np.isfinite(x):
        return 0.5
    x = max(float(x), 1e-12)
    return 1.0 / (1.0 + math.exp(-(math.log(x) - math.log(thr)) / width))


@torch.no_grad()
def sharpness_series(x, fc, floor, chunk=64):
    """Per frame [T]: in-band power (floor subtracted) in a low band (0.05 .. 0.35 f_c) and a high band (0.35 .. 0.8
    f_c), the complex contrast anisotropy sum E e^{2 i theta} / sum E over 0.25 .. 0.75 f_c, from the windowed
    spectrum of the central square of each frame, and the number of 2D samples in the two bands. x: [T, H, W]."""
    T, H, W = x.shape
    S = min(H, W, 256)
    y0, x0 = (H - S) // 2, (W - S) // 2
    w = hann2d(S, x.device)
    wn = float((w ** 2).sum())
    fr = torch.fft.fftfreq(S, device=x.device)
    FY, FX = torch.meshgrid(fr, fr, indexing='ij')
    R = torch.hypot(FX, FY)
    ax = (FX.abs() > 0.5 / S) & (FY.abs() > 0.5 / S)
    lo = ax & (R > 0.05 * fc) & (R < 0.35 * fc)
    hi = ax & (R >= 0.35 * fc) & (R < 0.8 * fc)
    mid = ax & (R > 0.25 * fc) & (R < 0.75 * fc)
    e2 = torch.exp(2j * torch.atan2(FY, FX))
    L, Hh, A = [], [], []
    for t0 in range(0, T, chunk):
        c = x[t0:t0 + chunk, y0:y0 + S, x0:x0 + S]
        c = c - c.mean((-1, -2), keepdim=True)
        P = torch.fft.fft2(c * w).abs() ** 2 / wn - floor
        L.append(P[:, lo].sum(1)); Hh.append(P[:, hi].sum(1))
        Pm = P[:, mid]
        A.append((Pm * e2[mid]).sum(1) / Pm.sum(1).clamp_min(1e-12))
    return (torch.cat(L).cpu().numpy(), torch.cat(Hh).cpu().numpy(), torch.cat(A).cpu().numpy(),
            int((lo | hi).sum()))


def _running_mean(s, w):
    w = max(3, int(w) | 1)
    if len(s) < w:
        return np.full_like(s, s.mean())
    pad = np.pad(s, w // 2, mode='reflect')
    return np.convolve(pad, np.ones(w) / w, 'valid')


@torch.no_grad()
def movie_statistics(frames, band=None, spectrum=None, device=None, max_spot_frames=300, lags=(1, 2, 5, 10, 20, 50),
                     threshold=5.0, max_side=1024):
    """Statistics of a contiguous block of frames [T, H, W] (numpy or torch; ADU or photons). band: a BandLimit of
    these (or similar) frames (computed when None). Returns a dict (see the module docstring)."""
    dev = resolve_device(device)
    x = frames if torch.is_tensor(frames) else torch.from_numpy(np.ascontiguousarray(frames))
    x = x.to(dev).float()
    if x.dim() == 2:
        x = x[None]
    T, H, W = x.shape
    if band is None:
        spectrum = power_spectrum(x, device=dev)
        band = fit_band_limit(spectrum, n_boot=0, anisotropy=False)
    fc = band.fc_px if band.status != 'no_signal' and np.isfinite(band.fc_px) else 0.4
    fc = float(np.clip(fc, 0.12, 0.6))
    floor = float(band.floor)
    st = dict(T=T, H=H, W=W, mean_level=float(x.mean()), fc_px=band.fc_px, band_status=band.status,
              band_snr=band.band_snr, floor=floor)

    # band-pass matched to the PSF size: Gaussian sigma of an Airy core ~ 0.42 / f_c px
    sp = float(np.clip(0.42 / fc, 0.7, 4.0))
    st['psf_sigma_px'] = sp
    st.update(motion_summary(global_motion(x, fc=fc), psf_sigma_px=sp))
    st.update(scene_statistics(x, floor=floor, fc=band.fc_px))
    if H > max_side or W > max_side:                                        # central crop for the rest
        y0, x0 = max(0, (H - max_side) // 2), max(0, (W - max_side) // 2)
        x = x[:, y0:y0 + max_side, x0:x0 + max_side]
        T, H, W = x.shape
        st['crop'] = [y0, x0, H, W]
    s1, s2 = sp, 3.0 * sp
    ck = max(1, min(128, int(4e6 // (H * W))))                       # ~4 M px per chunk: conv2d memory ~ px x taps
    B = torch.cat([blur(x[i:i + ck], s1) - blur(x[i:i + ck], s2) for i in range(0, T, ck)])
    m = int(math.ceil(3 * s2))
    if H > 2 * m + 8 and W > 2 * m + 8:
        B = B[:, m:H - m, m:W - m]
    Hb, Wb = B.shape[1:]
    # white-noise variance in B: floor x mean |H(f)|^2 of the DoG
    fy = torch.fft.fftfreq(Hb, device=dev); fx = torch.fft.fftfreq(Wb, device=dev)
    F2 = fy[:, None] ** 2 + fx[None, :] ** 2
    Hf = torch.exp(-2 * math.pi ** 2 * s1 ** 2 * F2) - torch.exp(-2 * math.pi ** 2 * s2 ** 2 * F2)
    nv = floor * float((Hf ** 2).mean())
    st['noise_var'] = nv

    mB = B.mean(0)
    dB = B - mB
    v = (dB ** 2).mean(0) * T / max(T - 1, 1)                                  # per-pixel temporal variance
    D = float(v.mean()) - nv
    S = float((mB ** 2).mean()) - nv / T
    st['dyn_snr'] = D / nv
    st['static_snr'] = S / nv
    st['dyn_fraction'] = max(D, 0) / max(max(D, 0) + max(S, 0), 1e-30)
    rho = {}
    for L in lags:
        if L < T // 2:
            rho[L] = float((dB[:-L] * dB[L:]).mean()) / D if D > 0.05 * nv else float('nan')
    st['rho'] = rho

    # cumulants over the most active pixels
    k2 = v - nv
    k4 = (dB ** 4).mean(0) - 3 * (dB ** 2).mean(0) ** 2
    act = k2 > max(float(torch.quantile(k2.flatten()[::max(1, k2.numel() // 200000)], 0.9)), 1e-30)
    act &= k2 > 3 * nv * math.sqrt(2.0 / T) + 1e-30                             # clearly above the noise
    st['active_pixels'] = float(act.float().mean())
    st['kurtosis'] = float(torch.median(k4[act] / k2[act] ** 2)) if act.sum() > 20 else float('nan')
    k3 = (dB ** 3).mean(0)
    st['skewness'] = float(torch.median(k3[act] / k2[act] ** 1.5)) if act.sum() > 20 else float('nan')

    # spots in up to max_spot_frames frames (contiguous)
    Ts = min(T, max_spot_frames)
    Bs = B[:Ts]
    thr = threshold * math.sqrt(max(nv, 1e-30))
    k = 2 * int(round(1.5 * sp)) + 1
    mx = Fn.max_pool2d(Bs[:, None], k, 1, k // 2)[:, 0]
    pk = (Bs == mx) & (Bs > thr)
    n_t = pk.flatten(1).sum(1).float()
    r_airy = 1.22 / fc
    area = math.pi * r_airy ** 2
    st['spots_per_frame'] = float(n_t.mean())
    st['spots_per_psf'] = float(n_t.mean()) * area / (Hb * Wb)
    st['spot_frames_frac'] = float((n_t > 0).float().mean())
    t, yy, xx = torch.nonzero(pk, as_tuple=True)
    if len(t) > 0:
        R = int(math.ceil(2 * r_airy))
        cnt = Fn.conv2d(pk[:, None].float(), torch.ones(1, 1, 2 * R + 1, 2 * R + 1, device=dev), padding=R)[:, 0]
        st['isolated'] = float((cnt[t, yy, xx] < 1.5).float().mean())
        occ = Fn.max_pool2d(pk[:, None].float(), 3, 1, 1)[:, 0] > 0           # within 1 px (Chebyshev), ~1.5 px
        dens = float(n_t.mean()) / (Hb * Wb)
        chance = 1 - math.exp(-dens * 9)
        pers = {}
        for L in (1, 2, 5, 10, 20, 50, 100):
            if L < Ts // 2:
                sel = t < Ts - L
                if sel.sum() > 10:
                    p = float(occ[t[sel] + L, yy[sel], xx[sel]].float().mean())
                    pers[L] = (p - chance) / max(1 - chance, 1e-9)
        st['persist'] = pers
        st['persist_chance'] = chance
        # wide persistence: a spot within 2 r_airy at lag L (spots that change shape, e.g. beads through focus)
        Rw = int(math.ceil(2 * r_airy))
        occw = Fn.max_pool2d(pk[:, None].float(), 2 * Rw + 1, 1, Rw)[:, 0] > 0
        chw = 1 - math.exp(-dens * (2 * Rw + 1) ** 2)
        pw = {}
        for L in (10, 20, 50, 100):
            if L < Ts // 2:
                sel = t < Ts - L
                if sel.sum() > 10:
                    p = float(occw[t[sel] + L, yy[sel], xx[sel]].float().mean())
                    pw[L] = (p - chw) / max(1 - chw, 1e-9)
        st['persist_wide'] = pw
    else:
        st['isolated'] = float('nan'); st['persist'] = {}; st['persist_chance'] = 0.0; st['persist_wide'] = {}

    # sharpness per frame
    lo, hi, an, nsamp = sharpness_series(x, fc, floor)
    tot = lo + hi
    good = tot > 5 * floor * math.sqrt(nsamp)                                  # in-band power clearly > noise
    s = np.where(good, np.clip(hi, 0, None) / np.maximum(tot, 1e-30), np.nan)  # high-band fraction, 0..1
    st['inband_power'] = tot.tolist() if T <= 1000 else tot[::max(1, T // 1000)].tolist()
    if good.sum() >= 5:
        sv = np.where(good, s, np.nanmedian(s))
        sm = _running_mean(sv, max(3, T // 15))
        var = np.var(sv)
        st['trend_r2'] = float(1 - np.var(sv - sm) / var) if var > 0 else 0.0
        st['trend_range'] = float((sm.max() - sm.min()) / max(abs(np.median(sm)), 1e-3))
        st['peak_pos'] = float(np.argmax(sm) / max(T - 1, 1))
        st['sharpness'] = sv.tolist() if T <= 1000 else sv[::max(1, T // 1000)].tolist()
        st['sharpness_smooth'] = sm.tolist() if T <= 1000 else sm[::max(1, T // 1000)].tolist()
        # signed contrast anisotropy along its dominant direction, smoothed
        a = np.where(good & (np.abs(an) <= 1), an, 0)
        phi = np.angle(np.sum(a * np.abs(a)))
        sa = _running_mean(np.real(a * np.exp(-1j * phi)), max(3, T // 15))
        st['astig_swing'] = float(sa.max() - sa.min())
        ag = an[good & (np.abs(an) <= 1)]
        st['aniso_frame_rms'] = float(np.sqrt(np.median(np.abs(ag - np.median(ag.real) - 1j * np.median(ag.imag)) ** 2))) \
            if len(ag) > 5 else float('nan')                                       # robust frame-to-frame spread
        st['astig_angle_deg'] = float(np.degrees(phi / 2) % 180)
        st['astig_signed'] = sa.tolist() if T <= 1000 else sa[::max(1, T // 1000)].tolist()
        tm = _running_mean(tot, max(3, T // 15))
        st['power_trend'] = float((tm[-1] - tm[0]) / max(abs(np.median(tm)), 1e-30))
    else:
        st.update(trend_r2=0.0, trend_range=0.0, peak_pos=float('nan'), astig_swing=0.0, power_trend=0.0)
    return st


@dataclasses.dataclass
class ModeResult:
    mode: str
    confidence: float
    probs: dict
    reason: str
    key: dict
    stats: dict
    advice: str = ''

    def to_dict(self):
        return _jsonable(dataclasses.asdict(self))


ADVICE = dict(
    smlm='SMLM route: detect isolated spots, fit an in-situ PSF (InSituEstimator) or a bead calibration if available.',
    sofi='SOFI route: dense fluctuations; estimate the PSF from the spatial cross-cumulants of the fluctuations.',
    zstack='z-stack route: beads / thin object through focus; fit a vectorial PSF to the stack (ZStackEstimator, '
           'DenseBeadEstimator, or phase diversity for an extended object).',
    static='static scene: only the band limit (NA / lambda) is reliable; a blind PSF estimate is ambiguous (object '
           'and PSF trade off; aberrations of even and odd symmetry are not separable without diversity).',
    scene='not a microscope image (a camera filming the world): there is no NA to estimate; the camera\'s own '
          'resolution (MTF / blur kernel) can be estimated from edges, or point the camera at a microscope.',
    moving='moving sample or stage: register the frames (global shifts are known from the motion statistics), then '
           'estimate the band limit on the registered frames; if the focus also changes, use phase diversity on the '
           'registered stack.',
    no_signal='no usable signal: check that the movie holds fluorescence images (not dark / flat frames) and enough '
              'photons.')


def classify_movie(stats):
    """Mode, confidence and reason from movie_statistics. Soft rules (probabilities of the decisions, multiplied
    down a small tree):
      signal   : band SNR, dynamic or static SNR
      zstack   : a smooth sharpness trend that explains the frame-to-frame variation (trend_r2), with a large range,
                 the same spots throughout (long-lag persistence)
      dynamic  : band-limited fluctuations well above the noise, which are not the same spots coming back (blinking)
                 or fluctuations that decorrelate (rho at 20 frames)
      sparse   : spots isolated, few per PSF area, heavy-tailed pixel fluctuations (kurtosis)
      slow     : (dense branch) fluctuations correlated over >= 20 frames relative to 1 frame, not heavy-tailed:
                 a slowly varying scene, counted as static
    """
    s = stats
    g = lambda k, d=float('nan'): s.get(k, d) if s.get(k) is not None else d
    pers = s.get('persist', {}) or {}
    rho = s.get('rho', {}) or {}
    p_long = max([v for L, v in pers.items() if int(L) >= 20] or [pers.get(10, float('nan'))] or [float('nan')])
    p_long = p_long if p_long is not None else float('nan')
    rho_far = [v for L, v in rho.items() if int(L) >= 20]
    rho_far = rho_far[0] if rho_far else rho.get(10, float('nan'))

    pw = s.get('persist_wide', {}) or {}
    pw_long = max(pw.values()) if pw else float('nan')
    kurt = g('kurtosis')
    sig = max(g('band_snr', 0) / 0.15, g('dyn_snr', 0) / 0.3, g('static_snr', 0) / 1.0)
    p_sig = _soft(sig, 1.0, 0.3)
    # z-stack: a smooth sharpness trend with a large range, real fluctuations, the spots stay in place (within
    # 2 Airy radii: they change shape through focus), no blinking statistics (heavy tails)
    p_z = _soft(g('trend_r2', 0), 0.5, 0.25) * _soft(g('trend_range', 0), 0.3, 0.4) * \
        _soft(g('dyn_snr', 0), 0.05, 0.4) * _soft(max(pw_long, 1e-3) if np.isfinite(pw_long) else 0.5, 0.3, 0.5) * \
        _soft(6.0 / max(kurt, 0.01) if np.isfinite(kurt) else 1.0, 1.0, 0.5)
    # dynamic (blinking / fluctuating): band-limited fluctuations, and new spots at long lags
    p_dyn = _soft(g('dyn_snr', 0), 0.08, 0.35) * \
        _soft(max(1 - p_long, 1e-3) if np.isfinite(p_long) else 0.5, 0.4, 0.4)
    # sparse: heavy-tailed pixel fluctuations, few spots per PSF area, isolated spots (geometric mean: a tetrapod
    # spot has several maxima, so isolation alone must not veto)
    p_sparse = (_soft(g('isolated', 0.0), 0.45, 0.3) * _soft(0.15 / max(g('spots_per_psf', 1.0), 1e-6), 1.0, 0.4) *
                _soft(kurt if np.isfinite(kurt) and kurt > 0 else 1e-3, 1.0, 0.5)) ** (1 / 3)
    # slowly varying (live cells, drift): dense fluctuations that stay correlated over many frames and are not
    # heavy-tailed (no on/off switching)
    r1 = rho.get(1, float('nan'))
    slow_ratio = rho_far / r1 if np.isfinite(rho_far) and np.isfinite(r1) and r1 > 0.05 else 0.0
    p_slow = _soft(max(slow_ratio, 1e-3), 0.5, 0.3) * _soft(0.3 / max(kurt, 0.01) if np.isfinite(kurt) else 1.0, 1.0, 0.5)
    # natural scene (not a microscope): scale-invariant spectrum, structure everywhere, no dark background, no band
    # edge below Nyquist (geometric mean of the cues, so one borderline cue does not veto)
    fcp = g('fc_px', 0.5)
    edge_free = 1.0 if s.get('band_status') == 'undersampled' else _soft(max(fcp, 1e-3) if np.isfinite(fcp) else
                                                                          0.5, 0.45, 0.1)
    slope = -g('spectral_slope', 0)
    slope = slope if np.isfinite(slope) else 0.0
    # featureless background is the decisive cue (phone clips of rooms: <= 1 % of the field; microscopy: >= 12 % in
    # the 19 real sets, the empty space between cells / spots); the others are combined by a geometric mean.
    # The slope has no upper limit: real phone clips (motion blur, ISP denoising) measure -3.3 .. -4.1.
    p_flat = _soft(0.06 / max(g('flat_frac', 1), 1e-4), 1.0, 0.3)
    p_scene = p_flat * (_soft(max(slope, 1e-3), 1.6, 0.12) * _soft(0.25 / max(g('dark_frac', 1), 1e-4), 1.0, 0.4) *
                        edge_free) ** (1 / 3)
    # global motion of a microscope sample / stage: most steps confident, most of them moving, a path of several
    # PSF widths
    mc, mm = g('motion_conf', 0), g('motion_moving', 0)
    p_mot = _soft(max(mc, 1e-3), 0.5, 0.25) * _soft(max(mm if np.isfinite(mm) else 0, 1e-3), 0.5, 0.25) * \
        _soft(max(g('motion_path_px', 0), 1e-3) / (3 * g('psf_sigma_px', 1.0)), 1.0, 0.4)
    rest = p_sig * (1 - p_scene) * (1 - p_mot)
    P = dict(no_signal=1 - p_sig,
             scene=p_sig * p_scene,
             moving=p_sig * (1 - p_scene) * p_mot,
             zstack=rest * p_z,
             smlm=rest * (1 - p_z) * p_dyn * p_sparse,
             sofi=rest * (1 - p_z) * p_dyn * (1 - p_sparse) * (1 - p_slow),
             static=rest * (1 - p_z) * ((1 - p_dyn) + p_dyn * (1 - p_sparse) * p_slow))
    mode = max(P, key=P.get)
    key = dict(band_snr=g('band_snr'), fc_px=g('fc_px'), dyn_snr=g('dyn_snr'), static_snr=g('static_snr'),
               dyn_fraction=g('dyn_fraction'), rho_1=rho.get(1), rho_far=rho_far, kurtosis=kurt,
               spots_per_psf=g('spots_per_psf'), isolated=g('isolated'), persist_1=pers.get(1), persist_long=p_long,
               persist_wide_long=pw_long,
               trend_r2=g('trend_r2'), trend_range=g('trend_range'), peak_pos=g('peak_pos'),
               astig_swing=g('astig_swing'), aniso_frame_rms=g('aniso_frame_rms'), slow_ratio=slow_ratio,
               motion_conf=mc, motion_moving=mm, motion_step_px=g('motion_step_px'),
               motion_path_px=g('motion_path_px'), spectral_slope=g('spectral_slope'), dark_frac=g('dark_frac'),
               flat_frac=g('flat_frac'))
    f = lambda v, n=2: 'n/a' if v is None or not np.isfinite(v) else f'{v:.{n}g}'
    if mode == 'no_signal':
        reason = (f'No band-limited signal: in-band power is {f(g("band_snr"))} x the noise floor, fluctuations '
                  f'{f(g("dyn_snr"))} x and static structure {f(g("static_snr"))} x the noise in the PSF band.')
    elif mode == 'scene':
        reason = (f'Not a microscope image: the power spectrum falls like a natural scene (slope '
                  f'{f(g("spectral_slope"))} in log-log, a photograph of the world is ~ -2; fluorescence is flatter), '
                  f'structure everywhere ({f(100 * g("flat_frac", 0), 2)} % of the field featureless, '
                  f'{f(100 * g("dark_frac", 0), 2)} % dark background) and no band edge below Nyquist '
                  f'(band: {s.get("band_status")}).')
        if np.isfinite(mm) and mc > 0.3:
            reason += (f' The camera moves (median step {f(g("motion_step_px"))} px, path '
                       f'{f(g("motion_path_px"), 3)} px).')
    elif mode == 'moving':
        reason = (f'The field moves as a whole between frames: {f(100 * mc, 3)} % of the frame steps have a strong, '
                  f'unique and self-consistent correlation peak, {f(100 * mm, 3)} % of those moved (median step '
                  f'{f(g("motion_step_px"))} px, path {f(g("motion_path_px"), 3)} px): stage panning or drift of a '
                  f'microscope sample.')
    elif mode == 'zstack':
        reason = (f'The sharpness changes systematically over the frames (a smooth trend explains '
                  f'{f(100 * g("trend_r2"), 3)} % of its variation, range {f(100 * g("trend_range"), 3)} %, '
                  f'peak at {f(100 * g("peak_pos"), 3)} % of the stack) while the same spots stay '
                  f'(within 2 Airy radii at long lag: {f(pw_long)}; no blinking, kurtosis {f(kurt)}): a focus sweep.')
        if g("astig_swing", 0) > 0.35:
            reason += (f' The spots change elongation through focus (anisotropy swing {f(g("astig_swing"))} at '
                       f'{f(g("astig_angle_deg"), 3)} deg): astigmatic.')
    elif mode in ('smlm', 'sofi'):
        reason = (f'Band-limited fluctuations {f(g("dyn_snr"))} x the noise; spots come and go (persistence after '
                  f'1 frame {f(pers.get(1))}, at long lag {f(p_long)}). ')
        if mode == 'smlm':
            reason += (f'Sparse: {f(g("spots_per_psf"))} spots per PSF area per frame, {f(100 * g("isolated", 0), 3)} % '
                       f'isolated, heavy-tailed pixel fluctuations (kurtosis {f(kurt)}): single-molecule blinking.')
        else:
            reason += (f'Dense: {f(g("spots_per_psf"))} spots per PSF area, only {f(100 * g("isolated", 0), 3)} % '
                       f'isolated, kurtosis {f(kurt)}: overlapping emitters fluctuating everywhere (SOFI-like).')
    elif p_dyn * (1 - p_sparse) * p_slow > (1 - p_dyn):
        reason = (f'Dense band-limited fluctuations ({f(g("dyn_snr"))} x the noise) that stay correlated over many '
                  f'frames (rho at >= 20 frames / rho at 1 frame = {f(slow_ratio)}) and are not heavy-tailed '
                  f'(kurtosis {f(kurt)}): a slowly moving or changing structure (live sample, drift), not blinking.')
    else:
        reason = (f'Signal present (band SNR {f(g("band_snr"))}) but fluctuations are weak or slow '
                  f'(dynamic SNR {f(g("dyn_snr"))}, lag correlation at >= 20 frames {f(rho_far)}, long-lag spot '
                  f'persistence {f(p_long)}) and the sharpness has no trend: static or slowly varying.')
    return ModeResult(mode=mode, confidence=float(P[mode]), probs={k: float(v) for k, v in P.items()},
                      reason=reason, key=key, stats=stats, advice=ADVICE[mode])
