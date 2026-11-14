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


class TiffMovie:
    """Absolute-frame access to movie.tif + movie_X1.tif + ... (also a single TIFF)."""
    def __init__(self, path):
        import tifffile
        path = str(path)
        self.parts = [path] + [path.replace('.tif', f'_X{i}.tif') for i in range(1, 10)
                               if os.path.exists(path.replace('.tif', f'_X{i}.tif'))]
        self.tifs = [tifffile.TiffFile(p) for p in self.parts]
        self.bounds = np.cumsum([0] + [len(t.pages) for t in self.tifs])
        self.n = int(self.bounds[-1])
        self.shape = self.tifs[0].pages[0].shape

    def read(self, f0, f1, y0=0, x0=0, S=None):
        H, W = self.shape
        out = np.empty((f1 - f0, S, S) if S is not None else (f1 - f0, H - y0, W - x0), np.float32)
        ys, xs = (slice(y0, y0 + S), slice(x0, x0 + S)) if S is not None else (slice(y0, None), slice(x0, None))
        for i, fi in enumerate(range(f0, f1)):
            j = int(np.searchsorted(self.bounds, fi, side='right') - 1)
            out[i] = self.tifs[j].pages[int(fi - self.bounds[j])].asarray()[ys, xs]
        return out

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
