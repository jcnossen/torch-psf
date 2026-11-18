"""The camera of an acquisition (Camera: ADU -> photons, read-noise variance) and its calibration from the SMLM movie
itself (photon_transfer: offset and gain from background pixels; white_photon_transfer: the gain from the white,
beyond-OTF part of the background noise; maps_from_movie: per-pixel offset and read-noise maps of
an sCMOS from a sparse movie, no dark frames).

A pipeline configuration with a `camera` block of the same field names maps 1:1 onto Camera:
    Camera.from_dict(cfg['camera']).

    camera, report = photon_transfer('movie.tif', excess_noise=2.0)     # EMCCD
    print(report['per_block'], camera)
"""
import dataclasses
import json
import os
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as Fn
from torch_psf.optics import resolve_device
from torch_psf.detect import detect_spots
from torch_psf.movie import open_movie
from torch_psf.spots import spread_order

Number = Union[float, np.ndarray, torch.Tensor]


def _crop(m, corners, P, frame_shape=None):
    """Per-ROI crops [n, P, P] of a per-pixel map m [H, W] at corners [n, 2] = (x0, y0) (torch)."""
    ii = torch.arange(P, device=m.device)
    yy = (corners[:, 1, None, None] + ii[None, :, None]).clamp(0, m.shape[0] - 1)
    xx = (corners[:, 0, None, None] + ii[None, None, :]).clamp(0, m.shape[1] - 1)
    return m[yy, xx]


@dataclasses.dataclass
class Camera:
    """Camera conversion. gain_e_per_adu (e- per ADU), offset_adu (ADU), readnoise_e (e- rms): each a scalar or a
    per-pixel map [H, W] (numpy or torch). maps: optional npz path with per-pixel 'offset_adu' / 'readnoise_e' maps,
    loaded by load_maps(). type: 'scmos' or 'emccd' (a label only).

    excess_noise: the noise factor F^2 of the camera (1 = Poisson; an EMCCD has F^2 ~ 2: the EM gain doubles the
    variance of the photoelectron count). to_photons divides by it, so the fits, CRLBs and deviances work with
    Poisson-equivalent ('effective') photons, var = mean, and the photon / background numbers they report are
    effective photons = photoelectrons / F^2. For an EMCCD give the true gain (photoelectrons per ADU, including the
    EM gain) with excess_noise=2, or equivalently the Poisson-equivalent gain (gain / 2, = 1 / the photon-transfer
    slope) with excess_noise=1. photon_transfer (below) measures it from a movie.

    Kept for files (psf.zip): extra = the keys of the camera dict that are not fields (notes etc.), scalars = the
    scalar offset / read noise / gain given before load_maps() replaced them by maps, map_extra = the other entries of
    the maps npz."""
    gain_e_per_adu: Number = 1.0
    offset_adu: Number = 0.0
    readnoise_e: Number = 0.0
    type: str = 'scmos'
    maps: Optional[str] = None
    excess_noise: float = 1.0
    extra: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)
    scalars: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)
    map_extra: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)

    _KEPT = ('extra', 'scalars', 'map_extra')

    @classmethod
    def from_dict(cls, d, base_dir=None, load_maps=True):
        """From a Camera dict or the camera block of a config / calibration json. A relative 'maps' path is resolved
        against base_dir (the directory of the calib file)."""
        names = {f.name for f in dataclasses.fields(cls)} - set(cls._KEPT)
        c = cls(**{k: v for k, v in d.items() if k in names and v is not None})
        c.extra = {k: v for k, v in d.items() if k not in names and k not in cls._KEPT}
        if c.maps and base_dir is not None and not os.path.isabs(c.maps):
            c.maps = os.path.join(base_dir, c.maps)
        if load_maps and c.maps:
            c.load_maps()
        return c

    @classmethod
    def from_json(cls, path, load_maps=True):
        """Camera block of a calibration / torch-psf json (relative map paths resolved against the json's directory), or
        None when the file has none."""
        with open(path) as f:
            c = json.load(f).get('camera')
        if not c:
            return None
        return cls.from_dict(c, base_dir=os.path.dirname(os.path.abspath(str(path))), load_maps=load_maps)

    def to_dict(self):
        out = {}
        for f in dataclasses.fields(self):
            if f.name in self._KEPT:
                continue
            v = getattr(self, f.name)
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                v = '<map>' if np.ndim(v) else float(v)   # maps live in the npz
            out[f.name] = v
        for k, v in (self.extra or {}).items():
            out.setdefault(k, v)
        return out

    def load_maps(self, path=None):
        """Replace scalar offset / read noise by the per-pixel maps of the npz (keys offset_adu, readnoise_e)."""
        p = path or self.maps
        M = np.load(p)
        for k in ('offset_adu', 'readnoise_e', 'gain_e_per_adu'):
            v = getattr(self, k)
            if k in M and np.ndim(v) == 0 and not (torch.is_tensor(v) and v.dim()):
                self.scalars.setdefault(k, float(v))
        self.map_extra = {k: M[k] for k in M.files if k not in ('offset_adu', 'readnoise_e')}
        if 'offset_adu' in M:
            self.offset_adu = M['offset_adu'].astype(np.float32)
        if 'readnoise_e' in M:
            self.readnoise_e = M['readnoise_e'].astype(np.float32)
        self.maps = p
        return self

    def _t(self, v, like, corners=None, P=None):
        t = torch.as_tensor(v, dtype=torch.float32, device=like.device)
        if t.dim() == 2 and corners is not None:
            return _crop(t, corners.long(), P)
        return t

    def to_photons(self, adu, corners=None):
        """(adu - offset) * gain for frames [T, H, W] (full frames: maps apply directly) or ROIs [n, P, P] with their
        corners [n, 2] = (x0, y0) (maps are cropped)."""
        adu = torch.as_tensor(adu)
        x = adu.float()
        P = x.shape[-1]
        ph = (x - self._t(self.offset_adu, x, corners, P)) * self._t(self.gain_e_per_adu, x, corners, P)
        return ph if self.excess_noise == 1 else ph / float(self.excess_noise)

    def variance(self, like, corners=None):
        """Read-noise variance in e^2 for frames or ROIs shaped like `like` (scalar tensor if the read noise is a
        scalar). Pass it as `var` to localize()."""
        P = like.shape[-1]
        rn = self._t(self.readnoise_e, like, corners, P)
        return rn ** 2 if self.excess_noise == 1 else rn ** 2 / float(self.excess_noise) ** 2


@torch.no_grad()
def photon_transfer(movie, frames=None, nblocks=8, block=400, mask_radius=4, threshold=4.0, offset_adu=None,
                    excess_noise=1.0, nbins=24, device=None, type=None):
    """Offset and gain of the camera from background pixels of a movie (ADU).

    nblocks blocks of `block` consecutive frames, spread over `frames` (f0, f1; default the whole movie). Per block and
    pixel: the temporal mean and the variance from consecutive-frame differences (var(x[t+1] - x[t]) / 2, which
    removes slow trends such as bleaching), using only samples away from spots: not within mask_radius px of a
    detection (detect_spots at `threshold`, in this frame or the one before) and inside a 4-sigma clip. The pixels are
    binned by mean (lowest 80 %), and a line var = k (mean - offset) (+ c when offset_adu is given) is fitted through
    the binned medians; the block results are combined by their median.

    For a camera with excess noise factor F^2 (EMCCD ~2) var_adu = F^2 mean_e / g^2 with g in e- / ADU, so
    g = F^2 / k: pass excess_noise=2 for an EMCCD (the returned Camera then has excess_noise=2 and the same
    Poisson-equivalent photons as gain 1 / k with excess_noise 1). offset_adu: a known offset (dark frames); the
    intercept then gives the read noise. Without it the read noise is taken as 0 (right for an EMCCD; for an sCMOS
    measure the offset on dark frames). Dim emitters below the detection threshold still add a little variance:
    sparse frames (late in the movie) give the best estimate.

    Returns (Camera, report dict: per_block slopes / offsets / kept fraction, slope, offset_adu, gain_e_per_adu)."""
    mv = open_movie(movie)
    dev = resolve_device(device)
    f0, f1 = (0, mv.n) if frames is None else frames
    block = min(block, f1 - f0)
    nstart = max((f1 - f0) // block, 1)
    starts = sorted(f0 + block * i for i in spread_order(nstart)[:nblocks])
    R = int(mask_radius)
    per = []
    for b0 in starts:
        x = torch.as_tensor(mv.read(b0, b0 + block), device=dev).float()
        T, H, W = x.shape
        p = detect_spots(x, threshold=threshold)
        occ = torch.zeros(T, H, W, device=dev)
        occ[p['frame'], p['y'], p['x']] = 1
        occ = Fn.max_pool2d(occ[:, None], 2 * R + 1, 1, R)[:, 0] > 0
        keep = ~occ
        keep[1:] &= ~occ[:-1]                                  # a blink may last into the next frame
        for _ in range(4):
            n = keep.sum(0).clamp_min(1)
            mu = (x * keep).sum(0) / n
            sd = (((x - mu) ** 2) * keep).sum(0).div(n).sqrt()
            keep = keep & ((x - mu).abs() < 4 * sd)
        d = x[1:] - x[:-1]
        kd = keep[1:] & keep[:-1]
        n = kd.sum(0)
        dm = (d * kd).sum(0) / n.clamp_min(1)
        var = (((d - dm) ** 2) * kd).sum(0) / (n - 1).clamp_min(1) / 2
        ok = (n > max(T // 4, 10)).cpu().numpy().ravel()
        mb, vb = mu.cpu().numpy().ravel()[ok], var.cpu().numpy().ravel()[ok]
        if len(mb) < 4 * nbins:
            continue
        s = mb < np.percentile(mb, 80)
        e = np.quantile(mb[s], np.linspace(0.01, 0.99, nbins + 1))
        xm, xv = [], []
        for i in range(nbins):
            sel = (mb >= e[i]) & (mb < e[i + 1])
            if sel.sum() >= 3:
                xm.append(np.median(mb[sel])); xv.append(np.median(vb[sel]))
        xm, xv = np.array(xm), np.array(xv)
        if offset_adu is None:
            k, c = np.polyfit(xm, xv, 1)
            off, c0 = -c / k, 0.0
        else:
            k, c = np.polyfit(xm - offset_adu, xv, 1)
            off, c0 = float(offset_adu), max(float(c), 0.0)
        per.append(dict(frames=[int(b0), int(b0 + block)], slope=float(k), offset_adu=float(off),
                        intercept_adu2=float(c0), median_mean_adu=float(np.median(mb)),
                        kept_fraction=float(kd.float().mean()), n_px=int(ok.sum())))
    if not per:
        raise ValueError('photon_transfer: not enough background pixels')
    k = float(np.median([b['slope'] for b in per]))
    off = float(np.median([b['offset_adu'] for b in per]))
    c0 = float(np.median([b['intercept_adu2'] for b in per]))
    g = float(excess_noise) / k
    cam = Camera(gain_e_per_adu=g, offset_adu=off, readnoise_e=float(np.sqrt(c0)) * g, excess_noise=float(excess_noise),
                 type=type or ('emccd' if excess_noise > 1 else 'scmos'))
    rep = dict(slope_adu=k, offset_adu=off, gain_e_per_adu=g, poisson_equivalent_gain=1.0 / k,
               excess_noise=float(excess_noise), readnoise_e=cam.readnoise_e, per_block=per,
               slope_spread=float(np.std([b['slope'] for b in per]) / k) if len(per) > 1 else 0.0)
    return cam, rep


def _shift_of(ref_var, var):
    """sensor-ROI shift (dy, dx) with ref[y + dy, x + dx] = this[y, x]: peak of the cross-correlation of the high-pass log
    temporal-variance maps, and the peak correlation"""
    from scipy import ndimage as ndi
    hp = lambda v: np.clip(np.log(v) - ndi.median_filter(np.log(v), 5), -1, 3)
    A, B = hp(ref_var), hp(var)
    F = np.fft.ifft2(np.fft.fft2(A - A.mean()) * np.conj(np.fft.fft2(B - B.mean()))).real / (A.std() * B.std() * A.size)
    iy, ix = np.unravel_index(F.argmax(), F.shape)
    dy = iy if iy < F.shape[0] // 2 else iy - F.shape[0]
    dx = ix if ix < F.shape[1] // 2 else ix - F.shape[1]
    return (int(dy), int(dx)), float(F.max())


def maps_from_movie(movie, spots, gain_e_per_adu, offset_adu, nframes=20000, lag=6, mask_radius=3.0, readnoise_floor_e=0.9,
                readnoise_ref=None, readnoise_shift='auto', log=None):
    """Per-pixel sCMOS offset and read-noise maps from a sparse SMLM movie (ADU), without dark frames.

    - offset: global offset_adu + the per-pixel deviation = temporal median of the pixel over the frames away from
      emitters (spots: not within mask_radius px of a spot in its frames frame - 1 .. frame + win) minus its spatially
      smoothed version (median filter 9 px, then Gaussian sigma 2 px: the smooth part is background).
    - read noise: robust variance (1.4826 MAD) of the differences of frames `lag` apart (e.g. one illumination-pattern
      cycle, which removes the pattern and static structure; the MAD rejects blinking), / 2, minus the shot noise of the
      pixel's level (median - offset map) x gain, floored at readnoise_floor_e.
    - readnoise_ref: optional dict of reference maps of the same camera (readnoise_e, var_adu, offset_adu, e.g. a
      previous result of this function) recorded at another sensor ROI position: the read noise is taken from it,
      shifted by readnoise_shift (dy, dx), ref[y + dy, x + dx] = this[y, x], or 'auto' (cross-correlation of the
      high-pass log temporal-variance maps); uncovered pixels get the own estimate high-passed (median filter 7 px).

    movie: path / Movie / array [T, H, W]; the first nframes frames are used. spots: dict of arrays x, y (px, pixel
    centres at integers), frame and optionally win (frames after `frame` that the spot covers, default 0).
    Returns (Camera with the maps as offset_adu / readnoise_e (gain and the scalar offset in `scalars`, the other
    maps in map_extra: var_adu, low5, readnoise_shift, readnoise_covered), report dict)."""
    from scipy import ndimage as ndi
    say = log or (lambda *a: None)
    mv = open_movie(movie)
    n = int(min(nframes, mv.n))
    mov = np.asarray(mv.read(0, n), dtype=np.float32)
    gain, off0 = float(gain_e_per_adu), float(offset_adu)
    low = np.percentile(mov, 5, axis=0)
    x, y = np.asarray(spots['x']), np.asarray(spots['y'])
    fr = np.asarray(spots['frame'])
    win = np.asarray(spots['win']) if 'win' in spots else np.zeros(len(x), int)
    k = (fr < n) & np.isfinite(x) & np.isfinite(y)
    x, y, fr, win = x[k], y[k], fr[k], win[k]
    say(f'{len(x)} localizations used for the mask')
    H, W = mov.shape[1:]
    busy = np.zeros(mov.shape, bool)
    r = int(mask_radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= mask_radius ** 2
    dy_, dx_ = yy[disk], xx[disk]
    cx, cy = np.round(x).astype(int), np.round(y).astype(int)
    for df in range(-1, int(win.max(initial=0)) + 1):
        f = fr.astype(int) + df
        ok = (f >= 0) & (f < n) & (df <= win)
        for oy, ox in zip(dy_, dx_):
            y2, x2 = cy[ok] + oy, cx[ok] + ox
            good = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
            busy[f[ok][good], y2[good], x2[good]] = True
    say(f'masked fraction of pixel-frames: {busy.mean():.3f}')
    m = mov.copy()
    m[busy] = np.nan
    med = np.nanmedian(m, axis=0)
    del m
    med = np.where(np.isfinite(med), med, np.median(mov, axis=0))
    offset_dev = med - ndi.gaussian_filter(ndi.median_filter(med, size=9), 2)
    dens = np.zeros((H, W))
    np.add.at(dens, (np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)), 1)
    dens = ndi.gaussian_filter(dens, 1.0)
    corr = float(np.corrcoef(offset_dev.ravel(), dens.ravel())[0, 1])
    say(f'corr(offset_dev, loc density) = {corr:.3f}  (target < 0.05)')
    d = mov[lag:] - mov[:-lag]
    mad = np.median(np.abs(d - np.median(d, 0)), 0) * 1.4826
    var_adu = mad ** 2 / 2
    level_e = np.clip((np.median(mov, 0) - (off0 + offset_dev)) * gain, 0, None)
    rn2_e = var_adu * gain ** 2 - level_e
    rn_e = np.sqrt(np.clip(rn2_e, readnoise_floor_e ** 2, None))
    shift, cov = (0, 0), np.ones((H, W), bool)
    rep = dict(corr_offset_density=corr)
    if readnoise_ref is not None:
        rn_ref = np.asarray(readnoise_ref['readnoise_e'], dtype=np.float64)
        if isinstance(readnoise_shift, str) and readnoise_shift == 'auto':
            shift, pk = _shift_of(np.asarray(readnoise_ref['var_adu']), var_adu)
            say(f'sensor-ROI shift vs reference (high-pass log variance cross-correlation): dy {shift[0]}, dx {shift[1]}, '
                f'peak corr {pk:.3f}')
            rep['shift_peak_corr'] = pk
        else:
            shift = tuple(int(v) for v in readnoise_shift)
        y3, x3 = np.mgrid[:H, :W]
        ys, xs = y3 + shift[0], x3 + shift[1]
        cov = (ys >= 0) & (ys < rn_ref.shape[0]) & (xs >= 0) & (xs < rn_ref.shape[1])
        rn_sh = np.full((H, W), np.nan)
        rn_sh[cov] = rn_ref[ys[cov], xs[cov]]
        # own estimate, high-passed (emitter signal variance is smooth on the pixel scale; camera excess is per pixel)
        rn_hp = np.sqrt(readnoise_floor_e ** 2 + np.clip(rn2_e - ndi.median_filter(rn2_e, 7), 0, None))
        say(f'read-noise map from the reference shifted by ({shift[0]},{shift[1]}): {cov.mean()*100:.1f}% of pixels covered; '
            f'uncovered -> high-passed own estimate. Check on the covered pixels: corr(own raw, ref) '
            f'{np.corrcoef(rn_e[cov], rn_sh[cov])[0, 1]:.3f}, corr(own high-passed, ref) {np.corrcoef(rn_hp[cov], rn_sh[cov])[0, 1]:.3f}; '
            f'median own raw {np.median(rn_e):.2f} / hp {np.median(rn_hp):.2f} / ref {np.median(rn_sh[cov]):.2f} e; '
            f'ref>3e px: hp median {np.median(rn_hp[cov][rn_sh[cov] > 3]):.2f} vs ref {np.median(rn_sh[cov][rn_sh[cov] > 3]):.2f}')
        rn_e = np.where(cov, rn_sh, rn_hp)
        if 'offset_adu' in readnoise_ref:
            od_ref = np.full((H, W), np.nan)
            od_ref[cov] = np.asarray(readnoise_ref['offset_adu'])[ys[cov], xs[cov]] - off0
            say(f'offset deviation vs the shifted reference map: corr {np.corrcoef(offset_dev[cov], od_ref[cov])[0, 1]:.3f}, '
                f'rms difference {np.std(offset_dev[cov] - od_ref[cov]):.2f} ADU (own sd {offset_dev.std():.2f}, ref sd {np.nanstd(od_ref):.2f})')
    rep.update(offset_dev_p1=float(np.percentile(offset_dev, 1)), offset_dev_p99=float(np.percentile(offset_dev, 99)),
               offset_dev_min=float(offset_dev.min()), rn_median=float(np.median(rn_e)), rn_p99=float(np.percentile(rn_e, 99)),
               rn_max=float(rn_e.max()), frac_rn_gt_2x=float(np.mean(rn_e > 2 * np.median(rn_e))), nframes=n,
               masked_frac=float(busy.mean()), readnoise_shift=list(shift))
    cam = Camera(gain_e_per_adu=gain, offset_adu=(off0 + offset_dev).astype(np.float32), readnoise_e=rn_e.astype(np.float32))
    cam.scalars = dict(offset_adu=off0, gain_e_per_adu=gain)
    cam.map_extra = dict(low5=low, var_adu=var_adu, readnoise_shift=np.array(shift), readnoise_covered=cov)
    return cam, rep


def _occupied(x, threshold, R):
    """pixel-frames within R px of a detection in this, the previous or the next frame (x [T, H, W]; the detection runs
    on x minus its median: the zero-padded band-pass of detect_spots turns a large offset into edge responses that
    inflate its noise estimate)"""
    p = detect_spots(x - x.median(), threshold=threshold)
    occ = torch.zeros_like(x)
    occ[p['frame'], p['y'], p['x']] = 1
    occ = Fn.max_pool2d(occ[:, None], 2 * R + 1, 1, R)[:, 0] > 0
    T, H, W = x.shape
    occ = Fn.max_pool1d(occ.permute(1, 2, 0).reshape(-1, 1, T).float(), 3, 1, 1)
    return occ.reshape(H, W, T).permute(2, 0, 1) > 0


@torch.no_grad()
def white_photon_transfer(movie, frames=None, nblocks=25, block=600, lag=1, patch=16, step=8, f_cut=0.5, threshold=4.0,
                          mask_radius=4, fpn_frames=2000, offset_adu=None, readnoise_e=None, excess_noise=1.0, nbins=20,
                          device=None, type=None):
    """Gain (and offset or read noise) of the camera from the spatially white part of the background noise of a movie
    (ADU).

    Shot noise and read noise are independent per pixel (white); intensity fluctuations of the sample (sub-threshold
    blinks, diffusing fluorophores, haze, a patterned illumination) are imaged through the PSF and so band-limited by
    the OTF cutoff 2 NA / lambda (in cycles / px: 2 NA pixel / lambda, e.g. 0.46 for NA 1.49, 0.70 um, 108 nm px).
    Per patch x patch window (every `step` px) of the differences of frames `lag` apart (lag = the period of a
    patterned illumination cancels static and periodic structure) without a detection (detect_spots at `threshold`,
    mask_radius px, frames t - 1 .. t + 1) in either frame: the mean power of the Hann-windowed patch above f_cut
    cycles / px (set f_cut above the OTF cutoff) = the per-pixel noise variance, against the patch mean level. The
    level is taken after removing the per-pixel offset fixed pattern (masked temporal median of the first fpn_frames
    frames minus its median-9 / Gaussian-2 smoothed version; fpn_frames=0: none), which would otherwise dilute the slope. A line
    var = k level + c through nbins quantile bins gives the gain g = excess_noise / k (e- / ADU). nblocks blocks of
    `block` frames spread over `frames` (f0, f1; default the whole movie); the spread of the per-block slopes is
    reported.

    Offset and read noise are not separately identifiable without dark frames: from the mean level L and variance V
    of the patches, V = k (L - offset) + (readnoise / g)^2. offset_adu given: the read noise follows; readnoise_e given:
    the offset follows; neither: the read noise is taken as 0 (as photon_transfer). Returns (Camera, report dict:
    slope_adu, gain_e_per_adu, offset_adu, readnoise_e, level_mean_adu, var_mean_adu2, level_span_adu, r2, bins
    [[level, var, n]], per_block slopes, slope_spread, fpn_sd_adu)."""
    from scipy import ndimage as ndi
    mv = open_movie(movie)
    dev = resolve_device(device)
    f0, f1 = (0, mv.n) if frames is None else (int(frames[0]), int(min(frames[1], mv.n)))
    block = min(block, f1 - f0)
    P = int(patch)
    f = torch.fft.fftfreq(P, device=dev)
    hi = (f[:, None] ** 2 + f[None, :] ** 2).sqrt() > f_cut
    w = torch.hann_window(P, periodic=False, device=dev)
    W2 = w[:, None] * w[None, :]
    norm = (W2 ** 2).sum()
    # offset fixed pattern
    if fpn_frames:
        x = torch.as_tensor(mv.read(f0, min(f0 + fpn_frames, f1)), device=dev).float()
        occ = _occupied(x, threshold, int(mask_radius))
        med = torch.nanmedian(torch.where(occ, torch.full_like(x, float('nan')), x), 0).values.cpu().numpy()
        med = np.where(np.isfinite(med), med, x.median(0).values.cpu().numpy())
        fpn = torch.as_tensor(med - ndi.gaussian_filter(ndi.median_filter(med, size=9), 2), device=dev).float()
    else:
        fpn = torch.zeros(mv.shape[-2:], device=dev)
    lev, var, blk = [], [], []
    for i, b0 in enumerate(np.linspace(f0, f1 - block, nblocks).astype(int) if nblocks > 1 else [f0]):
        x = torch.as_tensor(mv.read(int(b0), int(b0) + block), device=dev).float()
        occ = _occupied(x, threshold, int(mask_radius))
        T, H, W = x.shape
        for y0 in range(0, H - P + 1, step):
            for x0 in range(0, W - P + 1, step):
                o = occ[:, y0:y0 + P, x0:x0 + P].flatten(1).any(1)
                t = torch.nonzero(~o[:-lag] & ~o[lag:])[:, 0]
                if len(t) == 0:
                    continue
                a, b = x[t, y0:y0 + P, x0:x0 + P], x[t + lag, y0:y0 + P, x0:x0 + P]
                d = b - a
                d = d - d.mean((1, 2), keepdim=True)
                pw = torch.fft.fft2(d * W2).abs() ** 2 / norm / 2
                var.append(pw[:, hi].mean(1))
                lev.append((0.5 * (a + b)).mean((1, 2)) - fpn[y0:y0 + P, x0:x0 + P].mean())
                blk.append(torch.full((len(t),), i, device=dev))
    if not lev:
        raise ValueError('white_photon_transfer: no background patches')
    lev, var, blk = (torch.cat(v).cpu().numpy() for v in (lev, var, blk))

    def line(L, V):
        q = np.quantile(L, np.linspace(0.02, 0.98, nbins + 1))
        rows = [[float(L[s].mean()), float(V[s].mean()), int(s.sum())] for s in ((L >= q[j]) & (L < q[j + 1]) for j in range(nbins)) if s.sum()]
        A = np.array(rows)
        k, c = np.polyfit(A[:, 0], A[:, 1], 1)
        pred = k * A[:, 0] + c
        r2 = 1 - ((A[:, 1] - pred) ** 2).sum() / max(((A[:, 1] - A[:, 1].mean()) ** 2).sum(), 1e-12)
        return float(k), float(c), float(r2), rows

    k, c, r2, rows = line(lev, var)
    per = [line(lev[blk == i], var[blk == i])[0] for i in np.unique(blk) if (blk == i).sum() >= 20 * nbins]
    g = float(excess_noise) / k
    Lm, Vm = float(lev.mean()), float(var.mean())
    if offset_adu is not None:
        off = float(offset_adu)
        rn = float(np.sqrt(max(Vm - k * (Lm - off), 0.0))) * g
    elif readnoise_e is not None:
        rn = float(readnoise_e)
        off = Lm - (Vm - (rn / g) ** 2) / k
    else:
        rn, off = 0.0, Lm - Vm / k
    cam = Camera(gain_e_per_adu=g, offset_adu=off, readnoise_e=rn, excess_noise=float(excess_noise),
                 type=type or ('emccd' if excess_noise > 1 else 'scmos'))
    rep = dict(slope_adu=k, intercept_adu2=c, gain_e_per_adu=g, offset_adu=off, readnoise_e=rn, level_mean_adu=Lm,
               var_mean_adu2=Vm, level_span_adu=float(rows[-1][0] - rows[0][0]), r2=r2, bins=rows, per_block=per,
               slope_spread=float(np.std(per) / k) if len(per) > 1 else 0.0, fpn_sd_adu=float(fpn.std()),
               n_patches=int(len(lev)), lag=int(lag), f_cut=float(f_cut), excess_noise=float(excess_noise))
    return cam, rep
