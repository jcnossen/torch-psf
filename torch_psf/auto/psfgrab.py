"""Reader for PSF grab recordings (psfgrab.jcnossen.nl): a zip with frames.tif (ImageJ hyperstack [N, 3, H, W] uint8
RGB, or [N, H, W] gray), meta.json (format psfgrab/1: recording settings and per-frame camera settings, e.g.
focusDistance for focus sweeps) and optionally still.jpg (not read here)."""
import io
import json
import zipfile

import numpy as np
import tifffile

# Rec. 709 luma weights (phone sensors are sRGB-ish); the classifier and the spectrum work on one channel
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def read_psfgrab(path, gray=True, max_bytes=4 << 30):
    """(frames, meta) of a PSF grab zip. frames: float32 [N, H, W] luminance (gray=True) or uint8 [N, 3, H, W] / [N, H, W]
    as stored (gray=False). meta: the parsed meta.json ({} if absent). The data are untrusted: only frames.tif and
    meta.json are read, members larger than max_bytes are refused."""
    with zipfile.ZipFile(path) as z:
        names = {i.filename: i for i in z.infolist()}
        if 'frames.tif' not in names:
            raise ValueError(f'{path}: no frames.tif in the zip')
        for n in ('frames.tif', 'meta.json'):
            if n in names and names[n].file_size > max_bytes:
                raise ValueError(f'{path}: {n} is {names[n].file_size} bytes (> {max_bytes})')
        meta = json.loads(z.read('meta.json').decode('utf-8')) if 'meta.json' in names else {}
        a = tifffile.imread(io.BytesIO(z.read('frames.tif')))
    if a.ndim == 2:
        a = a[None]
    rgb = a.ndim == 4
    if rgb and a.shape[1] != 3:
        raise ValueError(f'{path}: frames.tif has shape {a.shape}, expected [N, 3, H, W] or [N, H, W]')
    if a.ndim not in (3, 4):
        raise ValueError(f'{path}: frames.tif has shape {a.shape}')
    if not gray:
        return a, meta
    if rgb:
        f = np.empty((a.shape[0],) + a.shape[2:], np.float32)
        for i in range(a.shape[0]):                                   # frame by frame: no [N, 3, H, W] float copy
            f[i] = np.tensordot(LUMA, a[i].astype(np.float32), 1)
        return f, meta
    return a.astype(np.float32), meta
