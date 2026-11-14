"""Spot detection on movie frames (photons): band-pass filter (difference of Gaussians, or one Gaussian minus the
temporal median), local maxima above a threshold times the noise, optional isolation and border rules.

Coordinates: x = column, y = row, pixel centres at integers."""
import torch
import torch.nn.functional as Fn


def gaussian_kernel(sigma, device, truncate=4.0):
    x = torch.arange(-int(truncate * sigma), int(truncate * sigma) + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma * sigma))
    return k / k.sum()


def blur(x, sigma):
    """Separable Gaussian blur of frames [T, H, W] (zero padding)."""
    k = gaussian_kernel(sigma, x.device)
    x = Fn.conv2d(x[:, None], k[None, None, :, None], padding=(len(k) // 2, 0))
    return Fn.conv2d(x, k[None, None, None, :], padding=(0, len(k) // 2))[:, 0]


def filter_frames(frames, sigma=(2.0, 6.0), median=False):
    """Band-pass filtered frames: blur(sigma0) - blur(sigma1) (difference of Gaussians), or with median=True
    blur(sigma0) of the difference to the temporal median of the frames."""
    if median:
        return blur(frames - frames.median(0).values, sigma[0])
    return blur(frames, sigma[0]) - blur(frames, sigma[1])


@torch.no_grad()
def detect_spots(frames, threshold=6.0, sigma=(2.0, 6.0), nms=9, noise='mad', border=0, isolation=None, median=False):
    """Local maxima of the band-pass filtered frames [T, H, W] (photons, on any device).

    threshold: in units of the noise of the filtered image: 'std' (per frame standard deviation), 'mad' (robust,
    1.4826 x median absolute deviation over the whole chunk) or 'mad_frame' (the same per frame: for movies whose
    density or background changes within a chunk, e.g. dense blinking early in a dSTORM movie, where one noise level
    for the chunk raises the threshold of the sparse frames). nms: size of the max-pool window (px). border: peaks
    closer than this to the frame edge are dropped. isolation: if given (px), peaks with another peak within this
    Chebyshev distance in the same frame are dropped (both of them).
    Returns dict(frame, y, x, score) of long / float tensors (frame relative to frames[0]); score = filtered value /
    noise."""
    T, H, W = frames.shape
    f = filter_frames(frames, sigma, median)
    if noise == 'std':
        sig = f.flatten(1).std(1)[:, None, None]
    elif noise == 'mad':
        sig = 1.4826 * (f - f.median()).abs().median()
    elif noise == 'mad_frame':
        ff = f.flatten(1)
        sig = 1.4826 * (ff - ff.median(1, keepdim=True).values).abs().median(1).values[:, None, None]
    else:
        raise ValueError(noise)
    mx = Fn.max_pool2d(f[:, None], nms, 1, nms // 2)[:, 0]
    pk = (f == mx) & (f > threshold * sig)
    t, y, x = torch.nonzero(pk, as_tuple=True)
    ok = (x >= border) & (x < W - border) & (y >= border) & (y < H - border)
    if isolation is not None:
        r = int(isolation)
        cnt = Fn.conv2d(pk[:, None].float(), torch.ones(1, 1, 2 * r + 1, 2 * r + 1, device=f.device), padding=r)[:, 0]
        ok = ok & (cnt[t, y, x] < 1.5)
    t, y, x = t[ok], y[ok], x[ok]
    s = sig.expand(T, 1, 1)[t, 0, 0] if torch.is_tensor(sig) and sig.dim() == 3 else sig
    return dict(frame=t, y=y, x=x, score=f[t, y, x] / s)
