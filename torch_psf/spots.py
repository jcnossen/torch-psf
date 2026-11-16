"""Spot ROIs from movies: extraction around detections (Spots), streaming detection over a movie reader
(extract_spots), and the Tracks container (track samples over consecutive frames).

ROI convention: roisize P odd, corner (x0, y0) = the detection pixel minus (P - 1) / 2, so the detection is the ROI
centre; positions fitted relative to the ROI centre convert to image coordinates as x = x0 + (P - 1) / 2 + x_fit."""
import dataclasses
from typing import Optional
import numpy as np
import torch
from torch_psf.detect import detect_spots
from torch_psf.movie import open_movie
from torch_psf.optics import resolve_device


@dataclasses.dataclass
class Spots:
    """rois [n, P, P] (photons), frame [n] (long), corner [n, 2] = (x0, y0) (long), score [n], var: read-noise
    variance ROIs [n, P, P] (e^2) or a scalar / None."""
    rois: torch.Tensor
    frame: torch.Tensor
    corner: torch.Tensor
    score: Optional[torch.Tensor] = None
    var: Optional[torch.Tensor] = None
    frames_read: int = 0
    frame_ranges: Optional[list] = None      # the (t0, t1) chunks read, in frame order


    def __len__(self):
        return len(self.rois)

    @property
    def roisize(self):
        return self.rois.shape[-1]

    def subset(self, idx):
        v = self.var if (self.var is None or self.var.dim() == 0) else self.var[idx]
        return Spots(self.rois[idx], self.frame[idx], self.corner[idx],
                     None if self.score is None else self.score[idx], v, self.frames_read, self.frame_ranges)

    def to(self, device):
        mv = lambda a: None if a is None else a.to(device)
        return Spots(self.rois.to(device), self.frame.to(device), self.corner.to(device), mv(self.score),
                     mv(self.var), self.frames_read, self.frame_ranges)

def extract_rois(frames, t, y, x, roisize):
    """ROIs [n, P, P] of frames [T, H, W] centred on (t, y, x) (indices; must be at least (P - 1) / 2 from the
    border)."""
    h = (roisize - 1) // 2
    ii = torch.arange(roisize, device=frames.device)
    yy = (y[:, None, None] - h + ii[None, :, None]).expand(-1, roisize, roisize)
    xx = (x[:, None, None] - h + ii[None, None, :]).expand(-1, roisize, roisize)
    return frames[t[:, None, None], yy, xx]


def spread_order(n):
    """0..n-1 in bit-reversed (van der Corput) order: every prefix is spread evenly over the range."""
    if n <= 1:
        return list(range(n))
    bits = int(np.ceil(np.log2(n)))
    rev = [int(format(i, f'0{bits}b')[::-1], 2) for i in range(1 << bits)]
    return [r for r in rev if r < n]


@torch.no_grad()
def extract_spots(movie, roisize, camera=None, frames=None, chunk=500, device=None, max_spots=None, stop=None,
                  progress=None, sample='sequential', detector=None, **detect_kw):
    """Detect spots on a movie (any reader of torch_psf.movie, a path or an array) and extract their ROIs.

    camera: Camera (ADU -> photons; None: the movie is already in photons). frames: (f0, f1) range (default all).
    chunk: frames per GPU chunk. max_spots: stop when this many spots are collected. sample: order in which the chunks
    are read: 'sequential' (from the start of the range) or 'spread' (bit-reversed chunk order, so that the chunks read
    before max_spots is reached are spread evenly over the whole range; spots are returned in frame order).
    Spots.frames_read = frames read, Spots.frame_ranges = the chunks read. stop(): optional callable that
    returns True to stop early (use what we have). progress(frames_done, frames_total, n_spots): optional callback.
    detect_kw: arguments of detect_spots (threshold, sigma, nms, noise, isolation, median); border defaults to
    (P - 1) / 2 + 2 px. detector: another detection function with the interface of detect_spots (frames [T, H, W]
    photons, **detect_kw -> dict(frame, y, x, score)), e.g. torch_psf.matched_filter.MatchedFilterDetector for
    engineered PSFs. Returns Spots (on `device`)."""
    mv = open_movie(movie)
    device = resolve_device(device)
    f0, f1 = (0, mv.n) if frames is None else frames
    h = (roisize - 1) // 2
    detect_kw.setdefault('border', h + 2)
    rois, fr, cor, sc, var = [], [], [], [], []
    starts = list(range(f0, f1, chunk))
    if sample == 'spread':
        starts = [starts[i] for i in spread_order(len(starts))]
    elif sample != 'sequential':
        raise ValueError(f"sample must be 'sequential' or 'spread', not {sample!r}")
    n, nread, ranges = 0, 0, []
    for t0 in starts:
        t1 = min(t0 + chunk, f1)
        x = torch.as_tensor(mv.read(t0, t1)).to(device).float()
        if camera is not None:
            x = camera.to_photons(x)
        p = (detect_spots if detector is None else detector)(x, **detect_kw)
        t, y, xx = p['frame'], p['y'], p['x']
        rois.append(extract_rois(x, t, y, xx, roisize))
        fr.append(t + t0)
        c = torch.stack([xx - h, y - h], 1)
        cor.append(c)
        sc.append(p['score'])
        if camera is not None and np.ndim(camera.readnoise_e) == 2:
            var.append(camera.variance(rois[-1], c))
        n += len(t)
        nread += t1 - t0
        ranges.append((t0, t1))
        if progress is not None:
            progress(nread, f1 - f0, n)
        if (max_spots is not None and n >= max_spots) or (stop is not None and stop()):
            break
    cat = lambda L: torch.cat(L) if L else None
    v = cat(var)
    if v is None and camera is not None:
        v = torch.as_tensor(float(np.asarray(camera.readnoise_e)) ** 2, device=device)
    if not rois:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return Spots(torch.zeros(0, roisize, roisize, device=device), z, torch.zeros(0, 2, dtype=torch.long,
                     device=device), torch.zeros(0, device=device), v, 0, [])
    s = Spots(torch.cat(rois), torch.cat(fr), torch.cat(cor), torch.cat(sc), v, nread, sorted(ranges))
    if max_spots is not None and len(s) > max_spots:
        s = s.subset(slice(0, max_spots))       # drops the end of the last chunk read
    if sample == 'spread' and len(s):
        s = s.subset(torch.argsort(s.frame, stable=True))
    return s


# ---------------------------------------------------------------------------------------------------- tracks
@dataclasses.dataclass
class Tracks:
    """Track samples of up to T frames: rois [B, T, R, R] (photons), mask [B, T] (frame present), frames [B, T]
    (-1 = absent), corner [B, 2] (x0, y0), xy_init [B, 2] (track mean position relative to the corner, px)."""
    rois: torch.Tensor
    mask: torch.Tensor
    frames: torch.Tensor
    corner: torch.Tensor
    xy_init: torch.Tensor

    def __len__(self):
        return len(self.rois)

    def subset(self, idx):
        return Tracks(self.rois[idx], self.mask[idx], self.frames[idx], self.corner[idx], self.xy_init[idx])
