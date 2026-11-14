"""Movie readers with one interface (other readers, e.g. for other camera formats, can implement the same):

  reader.n          number of frames
  reader.shape      shape of one frame (H, W)
  reader.parts      the source files
  reader.read(f0, f1, y0=0, x0=0, S=None) -> float32 [f1 - f0, ...] camera ADU (absolute frame index);
                    (y0, x0, S) = a square crop of the last two axes (S None = the whole frame)

  TiffMovie(path)      multi-part TIFF: movie.tif + movie_X1.tif + ... (Andor SOLIS); the frame index runs over the parts
  ArrayMovie(a)        an in-memory or memory-mapped stack [n, H, W] (a numpy array or a .npy path)
  open_movie(path)     by extension (.tif / .tiff, .npy); any object with this interface can be used instead
"""
import os
import numpy as np
import tifffile


def _part_paths(path):
    """movie.tif + movie_X1.tif + ... (Andor SOLIS multi-part files): the suffix goes before the extension only."""
    base, ext = os.path.splitext(path)
    parts = [path]
    for i in range(1, 100):
        p = f'{base}_X{i}{ext}'
        if not os.path.exists(p):
            break
        parts.append(p)
    return parts


class _TiffPart:
    """One TIFF file as a stack [n, H, W].

    ImageJ files (including ImageJ files > 4 GB, which hold one IFD for all frames) and other single-IFD stacks: the
    frame count and the memory map come from the first image series. Multi-page files (e.g. Andor SOLIS: one page per
    frame, metadata between the frames): one frame per page; when every page is one uncompressed block and the blocks
    are evenly spaced, the frames are read through one strided memory map, else page by page."""
    def __init__(self, path):
        self.tif = tifffile.TiffFile(path)
        pages = self.tif.pages
        p0 = pages[0]
        self.shape = tuple(p0.shape[-2:])
        self.mm = None
        if self.tif.is_imagej or len(pages) == 1:
            series = self.tif.series[0]
            shape = tuple(series.shape)
            self.shape = shape[-2:]
            self.n = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
            try:
                mm = tifffile.memmap(path, mode='r', series=0)
                if mm.size == self.n * self.shape[0] * self.shape[1]:
                    self.mm = mm.reshape(self.n, *self.shape)
            except (ValueError, TypeError, OSError):
                self.mm = None
            if self.mm is None and len(pages) < self.n:
                self.mm = series.asarray().reshape(self.n, *self.shape)   # not memory-mappable: one read
        else:
            self.n = len(pages)
            self.mm = self._strided(path, pages, p0)

    def _strided(self, path, pages, p0):
        try:
            if p0.compression != 1 or len(p0.shape) != 2 or getattr(p0, 'samplesperpixel', 1) != 1:
                return None
            dt = np.dtype(self.tif.byteorder + p0.dtype.str[1:])
            nbytes = self.shape[0] * self.shape[1] * dt.itemsize
            offs = np.empty(self.n, np.int64)
            for i in range(self.n):
                pg = pages[i]
                o, c = pg.dataoffsets, pg.databytecounts
                if tuple(pg.shape[-2:]) != self.shape or sum(c) != nbytes or \
                        any(o[k + 1] != o[k] + c[k] for k in range(len(o) - 1)):
                    return None
                offs[i] = o[0]
            stride = int(offs[1] - offs[0]) if self.n > 1 else nbytes
            if stride < nbytes or (self.n > 1 and not np.all(np.diff(offs) == stride)):
                return None
            raw = np.memmap(path, np.uint8, 'r', offset=int(offs[0]), shape=((self.n - 1) * stride + nbytes,))
            frames = np.lib.stride_tricks.as_strided(raw, shape=(self.n, nbytes), strides=(stride, 1), writeable=False)
            return frames.view(dt).reshape(self.n, *self.shape)
        except Exception:
            return None

    def read(self, i0, i1, ys, xs):
        if self.mm is not None:
            return self.mm[i0:i1, ys, xs]
        if i1 <= i0:
            return np.zeros((0,) + self.shape)[:, ys, xs]
        return np.stack([self.tif.pages[i].asarray()[ys, xs] for i in range(i0, i1)])

    def close(self):
        self.mm = None
        self.tif.close()


class TiffMovie:
    """Absolute-frame access to movie.tif + movie_X1.tif + ... (also a single TIFF). Each part is memory-mapped when
    its frames are contiguous (contiguous reads, no per-page decoding), else read page by page."""
    def __init__(self, path):
        path = str(path)
        self.parts = _part_paths(path)
        self.tifs = [_TiffPart(p) for p in self.parts]
        self.bounds = np.cumsum([0] + [t.n for t in self.tifs])
        self.n = int(self.bounds[-1])
        self.shape = self.tifs[0].shape

    def read(self, f0, f1, y0=0, x0=0, S=None):
        ys, xs = (slice(y0, y0 + S), slice(x0, x0 + S)) if S is not None else (slice(y0, None), slice(x0, None))
        f0, f1 = max(int(f0), 0), min(int(f1), self.n)
        out = []
        for j, t in enumerate(self.tifs):
            a, b = max(f0, int(self.bounds[j])), min(f1, int(self.bounds[j + 1]))
            if b > a:
                out.append(np.asarray(t.read(a - int(self.bounds[j]), b - int(self.bounds[j]), ys, xs), np.float32))
        if not out:
            H, W = self.shape
            return np.zeros((0, S, S) if S is not None else (0, H - y0, W - x0), np.float32)
        return out[0] if len(out) == 1 else np.concatenate(out)

    def close(self):
        for t in self.tifs:
            t.close()


class ArrayMovie:
    """Movie-like stack [n, H, W]: a numpy / torch array, or the path of a .npy file (memory-mapped)."""
    def __init__(self, a):
        if isinstance(a, (str, os.PathLike)):
            self.parts = [str(a)]; a = np.load(a, mmap_mode='r')
        else:
            self.parts = ['<array>']
            if hasattr(a, 'numpy'):
                a = a.cpu().numpy()
        self.a = a; self.n = len(a); self.shape = a.shape[1:]

    def read(self, f0, f1, y0=0, x0=0, S=None):
        a = self.a[f0:f1]
        a = a[..., y0:y0 + S, x0:x0 + S] if S is not None else (a[..., y0:, x0:] if (y0 or x0) else a)
        return np.asarray(a).astype(np.float32)


def open_movie(src):
    """Reader for a path (.tif / .tiff -> TiffMovie, .npy -> ArrayMovie), an array (ArrayMovie) or an object that
    already has the reader interface (returned as is)."""
    if hasattr(src, 'read') and hasattr(src, 'n'):
        return src
    if not isinstance(src, (str, os.PathLike)):
        return ArrayMovie(src)
    ext = os.path.splitext(str(src))[1].lower()
    if ext in ('.tif', '.tiff'):
        return TiffMovie(src)
    if ext == '.npy':
        return ArrayMovie(src)
    raise ValueError(f'no movie reader for {src!r} (known: .tif, .tiff, .npy)')
