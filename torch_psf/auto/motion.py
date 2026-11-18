"""Global motion and natural-scene statistics of a movie, for the classifier (torch_psf.auto.classify).

global_motion: frame-to-frame translation by phase correlation on downsampled frames (low-pass whitened cross-power
spectrum, sub-pixel peak). A confident peak at a non-zero shift means the camera or the stage moved; SMLM blinking,
SOFI fluctuations and focus sweeps keep the peak at zero (or give no confident peak at all).

scene_statistics: two cues that separate a photograph of the world from a microscope image:
  spectral_slope  slope of log P(f) vs log f over 0.02 .. 0.15 cycles/px of the mean-subtracted frames. Natural
                  scenes are close to scale invariant (occlusion of objects of all sizes, "dead leaves"): P ~ f^-2,
                  slope about -2 (-1.6 .. -2.6). Fluorescence spots are white up to the OTF roll-off (slope ~ 0 at
                  low f), filaments ~ -1; bright-field cells on an empty background are flatter than -1.5 at these
                  scales unless the field is crowded.
  dark_frac       fraction of pixels close to the dark level (fluorescence: most of the field is background)
  flat_frac       fraction of the field without structure (local band-pass energy at the noise level)
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fn

from torch_psf.auto.spectrum import hann2d


def _downsample(x, max_side):
    T, H, W = x.shape
    d = max(1, int(math.ceil(max(H, W) / max_side)))
    if d > 1:
        x = Fn.avg_pool2d(x[:, None], d, d)[:, 0]
    return x, d


def _peak_subpixel(r):
    """Integer peak of r [h, w] (FFT layout) refined by a parabola per axis; returns (dy, dx, value)."""
    h, w = r.shape
    i = int(torch.argmax(r))
    py, px = divmod(i, w)
    v = float(r[py, px])

    def par(a, b, c):
        den = a - 2 * b + c
        return 0.0 if abs(den) < 1e-12 else 0.5 * (a - c) / den
    dy = par(float(r[(py - 1) % h, px]), v, float(r[(py + 1) % h, px]))
    dx = par(float(r[py, (px - 1) % w]), v, float(r[py, (px + 1) % w]))
    sy = py + dy if py + dy < h / 2 else py + dy - h
    sx = px + dx if px + dx < w / 2 else px + dx - w
    return sy, sx, v


@torch.no_grad()
def global_motion(x, max_side=256, max_pairs=200, f_max=0.3, alpha=0.5, fc=None):
    """Frame-to-frame shifts of x [T, H, W] (torch) by (partially whitened) phase correlation on frames downsampled
    to <= max_side px: cross-power spectrum / |cross-power|^alpha below f_max cycles / (downsampled) px (alpha 1 is
    classic phase correlation, sharp but noisy on low-contrast frames; 0.5 is more robust). Returns dict: shift [P, 2]
    (dy, dx in original px, frame b relative to frame a, for the pairs [P, 2] = (a, b): (t, t+1) for up to max_pairs
    frames t, and (t, t+2), (t+1, t+2) for the closure test of motion_summary), peak [P] (correlation peak height),
    z [P] (the peak in robust standard deviations of the correlation surface; the maximum of an uncorrelated
    surface is ~4.5), ratio [P] (peak over the highest correlation outside a few peak widths, both above the median;
    ~1 when many correspondences compete), the downsampling factor and the centre crop size. fc: the band limit in
    cycles / original px (sets the peak width), if known."""
    T = x.shape[0]
    xs, d = _downsample(x.float(), max_side)
    h, w = xs.shape[1:]
    S = min(h, w)                                            # central square: one window for both axes
    y0, x0 = (h - S) // 2, (w - S) // 2
    xs = xs[:, y0:y0 + S, x0:x0 + S]
    win = hann2d(S, xs.device)
    fr = torch.fft.fftfreq(S, device=xs.device)
    R = torch.hypot(fr[:, None], fr[None, :])
    lp = (R < f_max).float() * (R > 1.5 / S).float()
    fcd = min(0.5, fc * d) if fc is not None and np.isfinite(fc) and fc > 0 else 0.5
    excl = int(min(S // 6, max(3, math.ceil(2.0 / fcd))))       # a few widths of the correlation peak
    out = dict(shift=[], peak=[], z=[], ratio=[], pairs=[], down=d, side=S * d)
    if T < 2:
        out.update(shift=np.zeros((0, 2)), peak=np.zeros(0), z=np.zeros(0), ratio=np.zeros(0), pairs=np.zeros((0, 2), int))
        return out
    ts = np.arange(0, T - 1)
    if len(ts) > max_pairs:
        ts = ts[np.linspace(0, len(ts) - 1, max_pairs).round().astype(int)]
    pairs = [(int(t), int(t) + 1) for t in ts] + [(int(t), int(t) + 2) for t in ts if t + 2 < T] + \
        [(int(t) + 1, int(t) + 2) for t in ts if t + 2 < T]
    pairs = list(dict.fromkeys(pairs))
    need = np.unique(np.asarray(pairs).flatten())
    F = {}
    for i in range(0, len(need), 64):
        idx = need[i:i + 64]
        c = xs[torch.as_tensor(idx, device=xs.device)]
        c = c - c.mean((-1, -2), keepdim=True)
        Fi = torch.fft.fft2(c * win)
        for k, t in enumerate(idx):
            F[int(t)] = Fi[k]
    for a, b in pairs:
        cp = F[b] * torch.conj(F[a])
        cp = cp / cp.abs().clamp_min(1e-12) ** alpha * lp             # alpha 1: phase correlation, 0.5: partial
        r = torch.fft.ifft2(cp).real
        sy, sx, v = _peak_subpixel(r)
        rs = r.flatten()[::max(1, r.numel() // 20000)]
        med = float(rs.median())
        mad = max(float((rs - med).abs().median()) * 1.4826, 1e-30)
        # second peak outside the main one (radius excl px, cyclic): a unique correspondence vs many equal ones
        # (sparse spots: every pair of spots in the two frames gives a peak of about the same height)
        py, px = int(round(sy)) % S, int(round(sx)) % S
        r2 = torch.roll(r, (-py + excl, -px + excl), (0, 1)).clone()
        r2[:2 * excl + 1, :2 * excl + 1] = -float('inf')
        v2 = float(r2.max())
        out['shift'].append((sy * d, sx * d)); out['peak'].append(v); out['pairs'].append((a, b))
        out['z'].append((v - med) / mad); out['ratio'].append((v - med) / max(v2 - med, 1e-30))
    out['shift'] = np.asarray(out['shift']); out['peak'] = np.asarray(out['peak']); out['z'] = np.asarray(out['z'])
    out['ratio'] = np.asarray(out['ratio'])
    out['pairs'] = np.asarray(out['pairs'])
    return out


def motion_summary(m, psf_sigma_px=1.0, z_thr=8.0, ratio_thr=1.5):
    """Scalar statistics of global_motion. A step t -> t+1 is confident when its peak is strong (z > z_thr) and
    unique (ratio > ratio_thr), and it closes: shift(t, t+1) + shift(t+1, t+2) = shift(t, t+2) within max(1 px, 0.5 PSF sigma, a downsampled px), with
    all three peaks strong. Sparse spots (SMLM) give many competing peaks at random shifts, which do not close.
    Returns the fraction of confident steps (motion_conf), their median shift (motion_step_px), the fraction of
    them that moved by more than max(0.5 px, 0.3 PSF sigma, half a downsampled pixel) (motion_moving), the range of
    the accumulated path of the confident steps (motion_path_px) and the median peak z (motion_z)."""
    sh, z, pr = m['shift'], m['z'], m['pairs']
    nan = float('nan')
    if len(z) == 0:
        return dict(motion_pairs=0, motion_conf=nan, motion_step_px=nan, motion_moving=nan, motion_path_px=nan,
                    motion_z=nan)
    lut = {(int(a), int(b)): i for i, (a, b) in enumerate(pr)}
    tol = max(1.0, 0.5 * psf_sigma_px, float(m['down']))
    steps, ok = [], []
    for (a, b), i in lut.items():
        if b != a + 1:
            continue
        j, k = lut.get((a + 1, a + 2)), lut.get((a, a + 2))
        good = (z[i] > z_thr) & (m['ratio'][i] > ratio_thr)
        if j is not None and k is not None:
            good &= (z[j] > z_thr) & (z[k] > z_thr) & (np.hypot(*(sh[i] + sh[j] - sh[k])) < tol)
        else:
            good = False
        steps.append(i); ok.append(bool(good))
    steps, ok = np.asarray(steps), np.asarray(ok)
    s1 = sh[steps]
    mag = np.hypot(s1[:, 0], s1[:, 1])
    move_thr = max(0.5, 0.3 * psf_sigma_px, 0.5 * m['down'])
    path = np.cumsum(np.where(ok[:, None], s1, 0.0), 0)
    return dict(motion_pairs=int(len(steps)), motion_conf=float(ok.mean()),
                motion_step_px=float(np.median(mag[ok])) if ok.any() else nan,
                motion_moving=float((mag[ok] > move_thr).mean()) if ok.any() else nan,
                motion_path_px=float(np.hypot(*(path.max(0) - path.min(0)))), motion_z=float(np.median(z[steps])),
                motion_thr_px=move_thr)


@torch.no_grad()
def scene_statistics(x, floor=None, fc=None, nframes=16, S=512, f_lo=0.02, f_hi=0.15):
    """spectral_slope, dark_frac and flat_frac (see the module docstring) of up to nframes frames of x [T, H, W]
    (central S x S crop); floor: the white-noise level of the spectrum (per-pixel variance), else estimated; fc: the
    band limit (cycles / px): the slope is fitted below min(f_hi, 0.35 fc), where the OTF roll-off is still small."""
    T, H, W = x.shape
    idx = np.unique(np.linspace(0, T - 1, min(T, nframes)).round().astype(int))
    S = min(S, H, W)
    y0, x0 = (H - S) // 2, (W - S) // 2
    c = x[torch.as_tensor(idx, device=x.device), y0:y0 + S, x0:x0 + S].float()
    # spectral slope: radial mean of |FFT|^2 of the windowed, mean-subtracted crops, fitted in log-log
    w = hann2d(S, c.device)
    P = (torch.fft.fft2((c - c.mean((-1, -2), keepdim=True)) * w).abs() ** 2).mean(0)
    fr = torch.fft.fftfreq(S, device=c.device)
    R = torch.hypot(fr[:, None], fr[None, :])
    if fc is not None and np.isfinite(fc) and fc > 0:
        f_hi = max(min(f_hi, 0.35 * fc), 2.5 * f_lo)
    edges = np.geomspace(f_lo, f_hi, 13)
    lf, lp = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (R >= a) & (R < b)
        if m.sum() > 4:
            lf.append(math.log(math.sqrt(a * b))); lp.append(math.log(float(P[m].mean()) + 1e-30))
    slope = float(np.polyfit(lf, lp, 1)[0]) if len(lf) >= 4 else float('nan')
    # dark fraction: pixels within 15 % of the (1 .. 99.5 percentile) range above the dark level, per frame
    lo = torch.quantile(c.flatten(1)[:, ::7], 0.01, dim=1)[:, None, None]
    hi = torch.quantile(c.flatten(1)[:, ::7], 0.995, dim=1)[:, None, None]
    dark = float(((c - lo) < 0.15 * (hi - lo).clamp_min(1e-12)).float().mean())
    # flat fraction: local energy of a band-pass (DoG 1 / 3 px) at the noise level, smoothed over ~8 px
    from torch_psf.detect import blur
    B = blur(c, 1.0) - blur(c, 3.0)
    E = blur(B ** 2, 4.0)
    if floor is not None and np.isfinite(floor) and floor > 0:          # white noise through the DoG
        F2 = R ** 2
        Hf = torch.exp(-2 * math.pi ** 2 * F2) - torch.exp(-2 * math.pi ** 2 * 9.0 * F2)
        noise_var = float(floor) * float((Hf ** 2).mean())
    else:
        noise_var = float(torch.quantile(E.flatten()[::13], 0.05))
    flat = float((E < 3.0 * noise_var).float().mean())
    return dict(spectral_slope=slope, dark_frac=dark, flat_frac=flat)
