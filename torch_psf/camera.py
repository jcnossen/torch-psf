"""The camera of an acquisition (Camera: ADU -> photons, read-noise variance) and its calibration from the SMLM movie
itself (photon_transfer: offset and gain from background pixels).

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
    slope) with excess_noise=1. photon_transfer (below) measures it from a movie."""
    gain_e_per_adu: Number = 1.0
    offset_adu: Number = 0.0
    readnoise_e: Number = 0.0
    type: str = 'scmos'
    maps: Optional[str] = None
    excess_noise: float = 1.0

    @classmethod
    def from_dict(cls, d, base_dir=None, load_maps=True):
        """From a Camera dict or the camera block of a config / calibration json. A relative 'maps' path is resolved
        against base_dir (the directory of the calib file)."""
        names = {f.name for f in dataclasses.fields(cls)}
        c = cls(**{k: v for k, v in d.items() if k in names and v is not None})
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
            v = getattr(self, f.name)
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                v = '<map>' if np.ndim(v) else float(v)   # maps live in the npz
            out[f.name] = v
        return out

    def load_maps(self, path=None):
        """Replace scalar offset / read noise by the per-pixel maps of the npz (keys offset_adu, readnoise_e)."""
        p = path or self.maps
        M = np.load(p)
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
