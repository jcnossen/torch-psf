"""Spot ROIs from movies: extraction around detections (Spots), streaming detection over a movie reader
(extract_spots), and linking of localizations into isolated tracks over consecutive frames (link_tracks,
extract_tracks).

ROI convention: roisize P odd, corner (x0, y0) = the detection pixel minus (P - 1) / 2, so the detection is the ROI
centre; positions fitted relative to the ROI centre convert to image coordinates as x = x0 + (P - 1) / 2 + x_fit."""
import dataclasses
from typing import Optional
import numpy as np
import torch
from scipy.spatial import cKDTree
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

    def image_xy(self, th):
        """ROI-relative fitted x, y (theta[:, :2]) -> image coordinates (px)."""
        h = (self.roisize - 1) / 2
        return self.corner.to(th.dtype) + h + th[:, :2]


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
                  progress=None, sample='sequential', **detect_kw):
    """Detect spots on a movie (any reader of torch_psf.movie, a path or an array) and extract their ROIs.

    camera: Camera (ADU -> photons; None: the movie is already in photons). frames: (f0, f1) range (default all).
    chunk: frames per GPU chunk. max_spots: stop when this many spots are collected. sample: order in which the chunks
    are read: 'sequential' (from the start of the range) or 'spread' (bit-reversed chunk order, so that the chunks read
    before max_spots is reached are spread evenly over the whole range; spots are returned in frame order).
    Spots.frames_read = frames read, Spots.frame_ranges = the chunks read. stop(): optional callable that
    returns True to stop early (use what we have). progress(frames_done, frames_total, n_spots): optional callback.
    detect_kw: arguments of detect_spots (threshold, sigma, nms, noise, isolation, median); border defaults to
    (P - 1) / 2 + 2 px. Returns Spots (on `device`)."""
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
        p = detect_spots(x, **detect_kw)
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
def link_tracks(frame, x, y, max_dist=0.8, max_gap=1, min_len=6, isolation=7.0):
    """Link localizations (numpy arrays frame, x, y; image px) over consecutive frames (gaps up to max_gap frames) into
    tracks; keep tracks spanning >= min_len frames with no other localization within `isolation` px of the track mean
    in the track frames +- 1. Returns a list of index arrays (into the inputs, in frame order)."""
    f = np.asarray(frame).astype(np.int64)
    xy = np.stack([np.asarray(x), np.asarray(y)], 1).astype(np.float64)
    o = np.argsort(f, kind='stable')
    f, xy = f[o], xy[o]
    nf = int(f.max()) + 1 if len(f) else 0
    starts = np.searchsorted(f, np.arange(nf + 2))
    trees = {}

    def tree(fr):
        if fr not in trees:
            trees[fr] = cKDTree(xy[starts[fr]:starts[fr + 1]]) if starts[fr + 1] > starts[fr] else None
        return trees[fr]

    nxt = -np.ones(len(f), int)
    for fr in range(nf - 1):
        a_ = slice(starts[fr], starts[fr + 1])
        if a_.stop == a_.start:
            continue
        for gap in range(1, max_gap + 2):
            t = tree(fr + gap) if fr + gap < nf else None
            if t is None:
                continue
            need = np.arange(a_.start, a_.stop)[nxt[a_] < 0]
            if len(need) == 0:
                break
            d, j = t.query(xy[need], distance_upper_bound=max_dist)
            ok = np.isfinite(d)
            nxt[need[ok]] = j[ok] + starts[fr + gap]
    has_prev = np.zeros(len(f), bool)
    has_prev[nxt[nxt >= 0]] = True
    tracks = []
    for s in np.nonzero(~has_prev)[0]:
        idx = [s]
        while nxt[idx[-1]] >= 0:
            idx.append(nxt[idx[-1]])
        if f[idx[-1]] - f[idx[0]] + 1 >= min_len:
            tracks.append(np.array(idx))
    keep = []
    for t in tracks:
        c = xy[t].mean(0)
        ok = True
        st = set(t.tolist())
        for fr in range(max(f[t[0]] - 1, 0), min(f[t[-1]] + 1, nf - 1) + 1):
            tr = tree(fr)
            if tr is None:
                continue
            if any((k + starts[fr]) not in st for k in tr.query_ball_point(c, isolation)):
                ok = False
                break
        if ok:
            keep.append(o[t])
    return keep


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


def extract_tracks(movie, tracks, frame, x, y, roisize=11, T=12, min_len=6, camera=None, device=None):
    """Cut each track (index arrays of link_tracks into frame / x / y) into samples of T contiguous frames (from the
    first localization; frames where the emitter was missed are included) and read their ROIs (photons)."""
    mv = open_movie(movie)
    device = resolve_device(device)
    H, W = mv.shape[-2:]
    R = roisize
    frame = np.asarray(frame).astype(np.int64)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    samples = []
    for t in tracks:
        cx_, cy_ = x[t].mean(), y[t].mean()
        cx, cy = int(round(cx_)) - R // 2, int(round(cy_)) - R // 2
        if cx < 0 or cy < 0 or cx + R > W or cy + R > H:
            continue
        f0, f1 = frame[t].min(), frame[t].max()
        for s0 in range(f0, f1 + 1, T):
            n = min(T, f1 + 1 - s0)
            if n >= min_len:
                samples.append((s0, n, cx, cy, cx_ - cx, cy_ - cy))
    B = len(samples)
    rois = np.zeros((B, T, R, R), np.float32)
    mask = np.zeros((B, T), bool)
    frames = -np.ones((B, T), np.int64)
    need = sorted({s0 + k for s0, n, *_ in samples for k in range(n)})
    cache = {}
    for fr in need:   # reads in runs of consecutive frames
        if fr not in cache:
            r1 = fr
            while r1 + 1 in need and r1 + 1 - fr < 256:
                r1 += 1
            blk = mv.read(fr, r1 + 1)
            for k in range(fr, r1 + 1):
                cache[k] = blk[k - fr]
    for i, (s0, n, cx, cy, *_) in enumerate(samples):
        for k in range(n):
            rois[i, k] = cache[s0 + k][cy:cy + R, cx:cx + R]
            mask[i, k] = True
            frames[i, k] = s0 + k
    smp = np.array([s[2:] for s in samples], np.float64).reshape(-1, 4)
    r = torch.tensor(rois, device=device)
    corner = torch.tensor(smp[:, :2], device=device, dtype=torch.long)
    if camera is not None:
        r = camera.to_photons(r.reshape(B * T, R, R), corner.repeat_interleave(T, 0)).reshape(B, T, R, R)
    return Tracks(r, torch.tensor(mask, device=device), torch.tensor(frames, device=device), corner,
                  torch.tensor(smp[:, 2:], device=device, dtype=torch.float32))
