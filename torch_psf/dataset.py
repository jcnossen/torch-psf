"""Dataset: the localizations of one acquisition, as a dictionary of torch tensors (CPU by default) that all have the
same first dimension n, plus metadata and an optional drift.

    ds = psf.localize('movie.tif', camera)          # PSF.fit / PSF.localize return a Dataset
    ds['pos'], ds.photons, len(ds)                  # fields by key or attribute
    good = ds[ds.crlb.max(1).values < 0.2]          # masks / slices / index tensors select rows of every field
    ds.filter_photons_(500)                         # in-place variants end with _
    ds.save('locs.hdf5'); ds = Dataset.load('locs.hdf5')
    img = ds.render(pixel_nm=10); res = ds.frc()

Fields (every field has its own dtype and shape [n, ...]):
  id        [n] int64        localization id (links rows of related datasets of one run)
  frame     [n] int32        first frame of the localization
  n_frames  [n] int32        frames it covers (1 for single-frame fits; the window of multi-frame methods; the on-time
                             of linked blinks); last frame = end_frame = frame + n_frames - 1
  pos       [n, D] float64   position in px: x, y (and z), pixel centres at integer coordinates, x = column. z is in px of
                             the lateral pixel size (meta['pixel_nm']; one unit for all axes), relative to the focal
                             plane of the PSF (pos_nm converts)
  crlb      [n, D] float32   precision (one sigma) per axis, px
  photons   [n] float32      total signal (photons)
  bg        [n] float32      background (photons / px / frame)
  optional: chi2, score, group, cov ([n, 3] = xy, xz, yz covariances, px^2), and any other field (add_fields); all are
  carried through indexing, filters, crop, concatenate, save and load.
Metadata (dict meta): pixel_nm, image_size [h, w], n_frames (of the movie), method, n_dims, drift (the record of the
drift estimate), ... (free; JSON-serializable).
Drift: [F, D] px per movie frame; the drift of a localization is drift[frame + n_frames // 2] (its centre frame).

Files: .hdf5 / .h5 (this class's format: docs/DATASET_FORMAT.md), Picasso hdf5 export / import (save_picasso /
load_picasso), ThunderSTORM csv (save_csv / load_csv: nm, frames from 1, first pixel centre at 0.5 px).
Dependencies: torch, numpy; h5py for hdf5; scipy for render(sigma_nm > 0); the `dme` package
(github.com/jcnossen/drift-estimation, version >= 2) for estimate_drift.
"""
import json
import os
import numpy as np
import torch

FORMAT, FORMAT_VERSION = 'torch-psf dataset', 1
CORE = ('id', 'frame', 'n_frames', 'pos', 'crlb', 'photons', 'bg')
_DTYPES = dict(id=torch.int64, frame=torch.int32, n_frames=torch.int32, pos=torch.float64, crlb=torch.float32,
               photons=torch.float32, bg=torch.float32, chi2=torch.float32, score=torch.float32, group=torch.int64,
               cov=torch.float32)
THRESHOLD_FRC = 1 / 7


def _tensor(v, dtype=None):
    if torch.is_tensor(v):
        t = v.detach()
    else:
        a = np.asarray(v)
        t = torch.from_numpy(a.copy() if not a.flags.writeable or not a.flags.c_contiguous else a)
    return t.to(dtype) if dtype is not None else t


def _jsonable(o):
    if torch.is_tensor(o):
        return o.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _plain(o):
    """JSON-safe copy (str keys, lists for arrays / tuples, str for anything else)"""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.ndarray, np.generic)) or torch.is_tensor(o):
        return _plain(o.tolist())
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return str(o)


class Dataset:
    """Localizations: fields (dict name -> tensor [n, ...]), meta (dict), drift ([F, D] px or None), drift_applied."""

    def __init__(self, fields, meta=None, drift=None, drift_applied=False):
        """fields: dict with at least pos [n, D] and frame [n]; id defaults to 0..n-1, n_frames to 1. Known fields are cast
        to their dtypes; arrays (numpy / lists) are converted to tensors."""
        f = {}
        for k, v in dict(fields).items():
            f[k] = _tensor(v, _DTYPES.get(k))
        for k in ('pos', 'frame'):
            if k not in f:
                raise KeyError(f'Dataset needs the field {k!r}')
        if f['pos'].dim() != 2:
            raise ValueError('pos must be [n, D]')
        n = len(f['pos'])
        f.setdefault('id', torch.arange(n, dtype=torch.int64))
        f.setdefault('n_frames', torch.ones(n, dtype=torch.int32))
        for k, v in f.items():
            if v.dim() == 0 or len(v) != n:
                raise ValueError(f'field {k!r}: shape {tuple(v.shape)}, expected {n} rows')
        object.__setattr__(self, 'fields', f)
        self.meta = dict(meta or {})
        self.meta.setdefault('n_dims', int(f['pos'].shape[1]))
        self.drift = None if drift is None else _tensor(drift, torch.float64)
        self.drift_applied = bool(drift_applied)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(cls, pos, frame, crlb=None, photons=None, bg=None, n_frames=None, meta=None, **fields):
        """From arrays: pos [n, D] px, frame [n]; crlb [n, D], photons, bg, n_frames and any further fields by keyword."""
        F = dict(pos=pos, frame=frame)
        for k, v in dict(crlb=crlb, photons=photons, bg=bg, n_frames=n_frames, **fields).items():
            if v is not None:
                F[k] = v
        return cls(F, meta)

    # ------------------------------------------------------------------ fields, rows
    def __len__(self):
        return len(self.fields['pos'])

    def __getattr__(self, name):
        f = self.__dict__.get('fields')
        if f is not None and name in f:
            return f[name]
        raise AttributeError(f'{type(self).__name__} has no attribute or field {name!r}')

    def __contains__(self, name):
        return name in self.fields

    def keys(self):
        return list(self.fields)

    def __getitem__(self, idx):
        """ds['name'] -> the field; ds[mask] / ds[slice] / ds[indices] / ds[int] -> a new Dataset with those rows (metadata
        and drift carried along)."""
        if isinstance(idx, str):
            return self.fields[idx]
        if isinstance(idx, (int, np.integer)):
            idx = [int(idx)]
        if not isinstance(idx, slice):
            idx = _tensor(idx)
            if idx.dtype == torch.bool and len(idx) != len(self):
                raise IndexError(f'mask of {len(idx)} rows for {len(self)} localizations')
        return type(self)({k: v[idx] for k, v in self.fields.items()}, dict(self.meta), self.drift, self.drift_applied)

    def __setitem__(self, name, value):
        """ds['name'] = tensor / array: add or replace a field (first dimension n)."""
        if not isinstance(name, str):
            raise TypeError('only fields can be assigned: ds[name] = array')
        v = _tensor(value, _DTYPES.get(name))
        if v.dim() == 0 or len(v) != len(self):
            raise ValueError(f'field {name!r}: shape {tuple(v.shape)}, expected {len(self)} rows')
        self.fields[name] = v

    def add_fields(self, **fields):
        """Add or replace fields: ds.add_fields(intensities=tensor [n, frames, channels], ...). Returns self."""
        for k, v in fields.items():
            self[k] = v
        return self

    def copy(self):
        return type(self)({k: v.clone() for k, v in self.fields.items()}, json.loads(json.dumps(self.meta, default=_jsonable)),
                          None if self.drift is None else self.drift.clone(), self.drift_applied)

    def to(self, device):
        """The fields on another device (a new object; the drift moves along)."""
        return type(self)({k: v.to(device) for k, v in self.fields.items()}, dict(self.meta),
                          None if self.drift is None else self.drift.to(device), self.drift_applied)

    def numpy(self):
        """dict of numpy arrays (views of the CPU tensors)."""
        return {k: v.cpu().numpy() for k, v in self.fields.items()}

    @property
    def n_dims(self):
        return int(self.fields['pos'].shape[1])

    @property
    def end_frame(self):
        """Last frame of each localization: frame + n_frames - 1."""
        return self.fields['frame'] + self.fields['n_frames'] - 1

    @property
    def center_frame(self):
        """Frame at the centre of each localization: frame + n_frames // 2 (the time used for drift)."""
        return self.fields['frame'] + self.fields['n_frames'] // 2

    @property
    def pixel_nm(self):
        """Camera pixel size in the sample (nm); None if unknown."""
        v = self.meta.get('pixel_nm')
        return None if v is None else float(v)

    @property
    def pos_nm(self):
        """pos in nm (needs meta['pixel_nm'])."""
        if self.pixel_nm is None:
            raise ValueError("meta['pixel_nm'] is not set")
        return self.fields['pos'] * self.pixel_nm

    @property
    def num_frames(self):
        """Movie frames: meta['n_frames'] if known, else the last end_frame + 1."""
        if self.meta.get('n_frames'):
            return int(self.meta['n_frames'])
        return int(self.end_frame.max()) + 1 if len(self) else 0

    def indices_per_frame(self):
        """List over frames: the row indices whose `frame` is that frame."""
        f = self.fields['frame'].long()
        n = max(self.num_frames, int(f.max()) + 1 if len(f) else 0)
        o = torch.argsort(f, stable=True)
        cnt = torch.bincount(f, minlength=n)
        return list(torch.split(o, cnt.tolist())) if n else []

    # ------------------------------------------------------------------ filters, crop, concatenate
    def _take(self, other):
        object.__setattr__(self, 'fields', other.fields)
        self.meta = other.meta
        return self

    def filter(self, mask):
        """self[mask] (a new object)."""
        return self[mask]

    def filter_(self, mask):
        """filter in place; returns self."""
        return self._take(self[mask])

    def filter_finite(self):
        m = torch.isfinite(self.fields['pos']).all(1)
        if 'crlb' in self.fields:
            m &= torch.isfinite(self.fields['crlb']).all(1)
        return self[m]

    def filter_photons(self, min_photons=None, max_photons=None):
        p = self.fields['photons']
        m = torch.ones(len(self), dtype=torch.bool, device=p.device)
        if min_photons is not None:
            m &= p >= min_photons
        if max_photons is not None:
            m &= p <= max_photons
        return self[m]

    def filter_photons_(self, min_photons=None, max_photons=None):
        return self._take(self.filter_photons(min_photons, max_photons))

    def filter_crlb(self, max_px=None, max_nm=None, axes=(0, 1)):
        """Rows with the largest lateral (axes) precision <= max_px (px) and / or <= max_nm (nm)."""
        s = self.fields['crlb'][:, list(axes)].max(1).values.double()
        m = torch.ones(len(self), dtype=torch.bool, device=s.device)
        if max_px is not None:
            m &= s <= max_px
        if max_nm is not None:
            m &= s * self.pixel_nm <= max_nm
        return self[m]

    def filter_crlb_(self, max_px=None, max_nm=None, axes=(0, 1)):
        return self._take(self.filter_crlb(max_px, max_nm, axes))

    def filter_frames(self, start=0, stop=None, whole=True):
        """Rows in frames [start, stop): frame >= start and (whole) end_frame < stop, else frame < stop."""
        f = self.fields['frame']
        m = f >= start
        if stop is not None:
            m &= (self.end_frame if whole else f) < stop
        return self[m]

    def filter_frames_(self, start=0, stop=None, whole=True):
        return self._take(self.filter_frames(start, stop, whole))

    def crop(self, x0, y0, x1, y1, shift=False):
        """Rows with x0 <= x < x1 and y0 <= y < y1 (px); meta['region'] records it. shift: move the origin to (x0, y0)."""
        x, y = self.fields['pos'][:, 0], self.fields['pos'][:, 1]
        d = self[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)]
        d.meta['region'] = [float(x0), float(y0), float(x1), float(y1)]
        if shift:
            p = d.fields['pos'].clone()
            p[:, 0] -= x0
            p[:, 1] -= y0
            d.fields['pos'] = p
            o = d.meta.get('origin', [0.0, 0.0])
            d.meta['origin'] = [o[0] + float(x0), o[1] + float(y0)]
            d.meta['region'] = [0.0, 0.0, float(x1 - x0), float(y1 - y0)]
        return d

    def crop_(self, x0, y0, x1, y1, shift=False):
        return self._take(self.crop(x0, y0, x1, y1, shift))

    def crop_from_edge(self, dist):
        """Rows at least dist px inside the image (needs meta['image_size'] = [h, w])."""
        h, w = self.meta['image_size']
        return self.crop(dist, dist, w - 1 - dist, h - 1 - dist)

    @classmethod
    def concatenate(cls, items, set_group=False):
        """Rows of several datasets: the fields they all have, metadata of the first (n_frames = the largest);
        set_group: field `group` = index of the source. Ids are kept (they may repeat)."""
        items = list(items)
        if not items:
            raise ValueError('nothing to concatenate')
        if len({d.drift_applied for d in items}) > 1:
            raise ValueError('some datasets are drift corrected and some are not')
        keys = [k for k in items[0].fields if all(k in d.fields for d in items)]
        F = {k: torch.cat([d.fields[k] for d in items]) for k in keys}
        if set_group:
            F['group'] = torch.cat([torch.full((len(d),), i, dtype=torch.int64) for i, d in enumerate(items)])
        meta = dict(items[0].meta)
        if any(d.meta.get('n_frames') for d in items):
            meta['n_frames'] = max(d.num_frames for d in items)
        d0 = items[0].drift
        same = all((d.drift is None and d0 is None) or (d.drift is not None and d0 is not None
                                                          and torch.equal(d.drift, d0)) for d in items)
        return cls(F, meta, d0 if same else None, items[0].drift_applied)

    def __add__(self, other):
        return type(self).concatenate([self, other])

    # ------------------------------------------------------------------ drift
    def drift_at(self):
        """Drift [n, D'] (px) at each localization's centre frame (clipped to the drift's frame range)."""
        if self.drift is None:
            raise ValueError('no drift: estimate_drift() or set_drift() first')
        t = self.center_frame.long().clamp(0, len(self.drift) - 1)
        return self.drift[t]

    def set_drift(self, drift, info=None):
        """Store a drift [F, D'] (px per movie frame, D' <= D) without applying it. Returns self."""
        if self.drift_applied:
            raise ValueError('positions are drift corrected: undo_drift() first')
        self.drift = _tensor(drift, torch.float64)
        if info is not None:
            self.meta['drift'] = info
        return self

    def apply_drift(self, drift=None):
        """pos -= drift[centre frame] (the stored drift, or `drift`). Returns self."""
        if self.drift_applied:
            raise ValueError('positions are already drift corrected')
        if drift is not None:
            self.drift = _tensor(drift, torch.float64)
        d = self.drift_at()
        p = self.fields['pos'].clone()
        p[:, :d.shape[1]] -= d.to(p)
        self.fields['pos'] = p
        self.drift_applied = True
        return self

    def undo_drift(self):
        """Add the applied drift back. Returns self."""
        if not self.drift_applied:
            raise ValueError('no drift correction has been applied')
        d = self.drift_at()
        p = self.fields['pos'].clone()
        p[:, :d.shape[1]] += d.to(p)
        self.fields['pos'] = p
        self.drift_applied = False
        return self

    def estimate_drift(self, n_frames=None, rows=None, dims=2, apply=False, device=None, return_info=False):
        """Drift with DME v2 (package `dme`, github.com/jcnossen/drift-estimation, version >= 2; entry point
        dme.dme_prior.dme_prior_estimate). The package is found as follows: a checkout named by the environment
        variable DME_PATH (the directory that holds dme/) if set, else the installed package
        (pip install git+https://github.com/jcnossen/drift-estimation).
        Rows: `rows` (mask) or all, always only rows with finite positions and 0 < crlb < 1 px. Time = centre frame.
        dims: 2 (lateral) or 3. Stores the drift [F, dims] px in self.drift and the run record in meta['drift']
        (method, entry point, version, package path, parameters: n_frames / dims / n_rows / time, the choices DME made
        (its info dict), and per-axis results: peak-to-peak and rms drift in px); returns the drift (and the record with
        return_info). apply=True also subtracts it."""
        import importlib
        import sys
        if self.drift_applied:
            raise ValueError('positions are already drift corrected: undo_drift() first')
        p = os.environ.get('DME_PATH')
        if p:
            p = os.path.abspath(os.path.expanduser(p))
            if not os.path.isfile(os.path.join(p, 'dme', '__init__.py')):
                raise RuntimeError(f'DME_PATH={p} is not a checkout of github.com/jcnossen/drift-estimation')
            if p not in sys.path:
                sys.path.insert(0, p)
        dme = sys.modules.get('dme') or importlib.import_module('dme')
        version = str(getattr(dme, '__version__', '1'))
        if int(version.split('.')[0]) < 2:
            raise RuntimeError(f'the dme package at {getattr(dme, "__file__", "?")} is version {version}; DME v2 is '
                               f'needed: set DME_PATH to a checkout of github.com/jcnossen/drift-estimation or pip install it')
        from dme.dme_prior import dme_prior_estimate
        from dme.dme_gpu import default_device
        F = int(n_frames or self.num_frames)
        pos = self.fields['pos'][:, :dims].cpu().numpy().astype(np.float32)
        crlb = self.fields['crlb'][:, :dims].cpu().numpy().astype(np.float32)
        t = self.center_frame.cpu().numpy().astype(np.int32)
        ok = (np.isfinite(pos).all(1) & np.isfinite(crlb).all(1) & (crlb.max(1) < 1) & (crlb.min(1) > 0)
              & (t >= 0) & (t < F))
        if rows is not None:
            ok &= _tensor(rows).cpu().numpy().astype(bool)
        d, info = dme_prior_estimate(pos[ok], t[ok], crlb[ok], numframes=F, device=default_device(device),
                                     verbose=False, return_info=True)
        d = np.asarray(d)[:F, :dims]
        self.drift = torch.as_tensor(d, dtype=torch.float64)
        rec = dict(method='DME v2', entry_point='dme.dme_prior.dme_prior_estimate', version=version,
                   package=os.path.dirname(os.path.abspath(dme.__file__)),
                   parameters=dict(n_frames=F, dims=dims, n_rows=int(ok.sum()), time='frame + n_frames // 2',
                                   rows='finite, 0 < crlb < 1 px' + (', user mask' if rows is not None else '')),
                   dme_info=_plain(info),
                   result=dict(axes='xyz'[:dims], ptp_px=np.ptp(d, 0).round(5).tolist(),
                               rms_px=np.sqrt(((d - d.mean(0)) ** 2).mean(0)).round(5).tolist()))
        self.meta['drift'] = rec
        if apply:
            self.apply_drift()
        return (self.drift, rec) if return_info else self.drift

    # ------------------------------------------------------------------ render, FRC
    def region(self):
        """(x0, y0, x1, y1) px: meta['region'] if set, else the image (-0.5 .. size - 0.5) if known, else the bounding
        box."""
        if self.meta.get('region'):
            return tuple(self.meta['region'])
        if self.meta.get('image_size') and not self.meta.get('origin'):
            h, w = self.meta['image_size']
            return (-0.5, -0.5, w - 0.5, h - 0.5)
        x, y = self.fields['pos'][:, 0], self.fields['pos'][:, 1]
        return (float(x.min().floor()), float(y.min().floor()), float(x.max().ceil()), float(y.max().ceil()))

    def render(self, pixel_nm=10.0, region=None, sigma_nm=0.0, weights=None, return_extent=False):
        """2D histogram [ny, nx] (row = y) with pixels of pixel_nm nm (needs meta['pixel_nm']; else pixel_nm is in px).
        region (x0, y0, x1, y1) px (default region()); sigma_nm > 0: Gaussian blur (scipy); weights: field name or array.
        return_extent: also (x0, x1, y0, y1) px of the image edges."""
        x0, y0, x1, y1 = self.region() if region is None else region
        s = (self.pixel_nm or 1.0) / pixel_nm
        nx = max(int(np.ceil((x1 - x0) * s - 1e-9)), 1)
        ny = max(int(np.ceil((y1 - y0) * s - 1e-9)), 1)
        w = self.fields[weights] if isinstance(weights, str) else weights
        p = self.fields['pos'].cpu().numpy()
        img = np.histogram2d(p[:, 1], p[:, 0], bins=(ny, nx), range=[[y0, y0 + ny / s], [x0, x0 + nx / s]],
                             weights=None if w is None else _tensor(w).cpu().numpy())[0].astype(np.float32)
        if sigma_nm > 0:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma_nm / pixel_nm)
        return (img, (x0, x0 + nx / s, y0, y0 + ny / s)) if return_extent else img

    def frc(self, block=300, pixel_nm=5.0, return_curve=False):
        """Fourier ring correlation resolution (nm) between the localizations of even and odd blocks of `block` frames:
        histograms on a square grid of pixel_nm (covering both halves), FRC per integer ring, 5-ring moving average,
        resolution at the first ring (>= 3) below 1/7 (nan if none). Needs meta['pixel_nm']. return_curve: also the ring
        frequencies (1/nm) and the curve."""
        px = self.pixel_nm
        if px is None:
            raise ValueError("meta['pixel_nm'] is not set")
        p = self.fields['pos'].cpu().numpy()
        f = self.fields['frame'].cpu().numpy()
        h = (f // block) % 2 == 0
        xa, ya, xb, yb = p[h, 0], p[h, 1], p[~h, 0], p[~h, 1]
        s = px / pixel_nm
        lo = np.floor(min(xa.min(), ya.min(), xb.min(), yb.min()))
        hi = np.ceil(max(xa.max(), ya.max(), xb.max(), yb.max()))
        n = int((hi - lo) * s)
        n += n % 2
        rng = [[lo, lo + n / s], [lo, lo + n / s]]
        hist = lambda x, y: np.histogram2d(y, x, bins=n, range=rng)[0]
        A, B = np.fft.fftshift(np.fft.fft2(hist(xa, ya))), np.fft.fftshift(np.fft.fft2(hist(xb, yb)))
        yy, xx = np.indices(A.shape) - n // 2
        rr = np.hypot(xx, yy).astype(int)
        num = np.bincount(rr.ravel(), (A * B.conj()).real.ravel())
        da = np.bincount(rr.ravel(), (np.abs(A) ** 2).ravel())
        db = np.bincount(rr.ravel(), (np.abs(B) ** 2).ravel())
        c = np.convolve(num / np.sqrt(da * db + 1e-12), np.ones(5) / 5, 'same')
        below = np.nonzero(c[3:n // 2] < THRESHOLD_FRC)[0]
        res = float('nan') if len(below) == 0 else float(n * pixel_nm / (below[0] + 3))
        if return_curve:
            return res, np.arange(n // 2) / (n * pixel_nm), c[:n // 2]
        return res

    # ------------------------------------------------------------------ summary
    def info(self):
        R = dict(n=len(self), n_dims=self.n_dims, n_frames=self.num_frames, fields=self.keys(),
                 pixel_nm=self.pixel_nm, drift='none' if self.drift is None else
                 ('applied' if self.drift_applied else 'stored, not applied'))
        for k in ('image_size', 'method', 'region', 'source'):
            if self.meta.get(k) is not None:
                R[k] = self.meta[k]
        if len(self) and 'crlb' in self.fields:
            c = self.fields['crlb'][:, :2].max(1).values.double()
            R['median_crlb_px'] = float(c.median())
        if len(self) and 'photons' in self.fields:
            R['median_photons'] = float(self.fields['photons'].median())
        return R

    def __repr__(self):
        I = self.info()
        s = f'Dataset: {I["n"]} localizations, {I["n_dims"]}D, {I["n_frames"]} frames'
        if I['pixel_nm']:
            s += f', pixel {I["pixel_nm"]:g} nm'
        if 'median_crlb_px' in I:
            s += f', median precision {I["median_crlb_px"]:.3f} px'
        if 'median_photons' in I:
            s += f', median photons {I["median_photons"]:.0f}'
        return s + f', drift {I["drift"]}, fields {" ".join(I["fields"])}'

    # ------------------------------------------------------------------ files
    def save(self, path, **kw):
        """By extension: .hdf5 / .h5 (this format), .csv (ThunderSTORM). Picasso: save_picasso. Returns the path."""
        ext = os.path.splitext(str(path))[1].lower()
        if ext in ('.hdf5', '.h5'):
            return self.save_hdf5(path, **kw)
        if ext == '.csv':
            return self.save_csv(path, **kw)
        raise ValueError(f'unknown file type {ext!r} (.hdf5, .h5, .csv; Picasso: save_picasso)')

    @classmethod
    def load(cls, path, **kw):
        """By extension: .hdf5 / .h5 (this format, or a Picasso file: one with a 'locs' table), .csv (ThunderSTORM)."""
        ext = os.path.splitext(str(path))[1].lower()
        if ext in ('.hdf5', '.h5'):
            import h5py
            with h5py.File(path, 'r') as f:
                picasso = 'locs' in f and 'localizations' not in f
            return cls.load_picasso(path, **kw) if picasso else cls.load_hdf5(path)
        if ext == '.csv':
            return cls.load_csv(path, **kw)
        raise ValueError(f'unknown file type {ext!r}')

    def save_hdf5(self, path):
        """docs/DATASET_FORMAT.md: /localizations/<field> (one dataset per field, its dtype and shape), /metadata
        (attributes; dicts / lists as JSON strings), /drift (optional, attributes applied, info). Returns the path."""
        import h5py
        tmp = str(path) + '.tmp'
        with h5py.File(tmp, 'w') as f:
            f.attrs['format'] = FORMAT
            f.attrs['format_version'] = FORMAT_VERSION
            g = f.create_group('localizations')
            for k, v in self.fields.items():
                a = v.cpu().numpy()
                g.create_dataset(k, data=a, compression='gzip' if a.size > 4096 else None)
            m = f.create_group('metadata')
            for k, v in self.meta.items():
                if isinstance(v, (str, int, float, bool)) and v is not None:
                    m.attrs[k] = v
                else:
                    m.attrs[k] = 'json:' + json.dumps(v, default=_jsonable)
            if self.drift is not None:
                d = f.create_dataset('drift', data=self.drift.cpu().numpy())
                d.attrs['applied'] = self.drift_applied
                d.attrs['info'] = json.dumps(self.meta.get('drift', {}), default=_jsonable)
        os.replace(tmp, path)
        return path

    @classmethod
    def load_hdf5(cls, path):
        import h5py
        with h5py.File(path, 'r') as f:
            F = {k: torch.from_numpy(np.asarray(f['localizations'][k][...])) for k in f['localizations']}
            meta = {}
            for k, v in f['metadata'].attrs.items():
                if isinstance(v, bytes):
                    v = v.decode()
                if isinstance(v, str) and v.startswith('json:'):
                    v = json.loads(v[5:])
                elif isinstance(v, np.generic):
                    v = v.item()
                meta[k] = v
            drift, applied = None, False
            if 'drift' in f:
                drift = torch.from_numpy(f['drift'][...])
                applied = bool(f['drift'].attrs.get('applied', False))
        meta.setdefault('source', os.path.abspath(str(path)))
        return cls(F, meta, drift, applied)

    def save_picasso(self, path, psf_sigma_px=1.3):
        """Picasso hdf5 (dataset 'locs': frame u4, x, y, photons, sx, sy (PSF width: field psf_sx / psf_sy, else
        psf_sigma_px), bg, lpx, lpy (precision = crlb), ellipticity, net_gradient, z / lpz in 3D (z in px)) and the .yaml
        Picasso needs. Other 1D fields are added under their names. Returns the path."""
        import h5py
        n, F = len(self), self.numpy()
        z0 = np.zeros(n, np.float32)
        psx = F.get('psf_sx', np.full(n, psf_sigma_px, np.float32))
        psy = F.get('psf_sy', psx)
        cols = [('frame', '<u4', F['frame']), ('x', '<f4', F['pos'][:, 0]), ('y', '<f4', F['pos'][:, 1]),
                ('photons', '<f4', F.get('photons', z0)), ('sx', '<f4', psx), ('sy', '<f4', psy),
                ('bg', '<f4', F.get('bg', z0)), ('lpx', '<f4', F['crlb'][:, 0] if 'crlb' in F else z0),
                ('lpy', '<f4', F['crlb'][:, 1] if 'crlb' in F else z0), ('ellipticity', '<f4', z0),
                ('net_gradient', '<f4', z0)]
        if self.n_dims == 3:
            cols += [('z', '<f4', F['pos'][:, 2]), ('lpz', '<f4', F['crlb'][:, 2] if 'crlb' in F else z0)]
        used = {'frame', 'pos', 'crlb', 'photons', 'bg', 'psf_sx', 'psf_sy'}
        for k, v in F.items():
            if k not in used and v.ndim == 1:
                cols.append((k, v.dtype.str if v.dtype != bool else '|u1', v))
        A = np.zeros(n, dtype=[(k, t) for k, t, _ in cols])
        for k, _, v in cols:
            A[k] = v
        with h5py.File(path, 'w') as f:
            f.create_dataset('locs', data=A)
        h, w = self.meta.get('image_size') or (int(np.ceil(F['pos'][:, 1].max())) + 1 if n else 0,
                                               int(np.ceil(F['pos'][:, 0].max())) + 1 if n else 0)
        Y = {'Byte Order': '<', 'Camera': 'unknown', 'Data Type': 'uint16', 'File': os.path.abspath(str(path)),
             'Frames': self.num_frames, 'Height': int(h), 'Width': int(w), 'Generated by': 'torch_psf.Dataset'}
        if self.pixel_nm:
            Y['Pixelsize'] = self.pixel_nm
        with open(os.path.splitext(str(path))[0] + '.yaml', 'w') as f:
            f.write(''.join(f'{k}: {json.dumps(v) if isinstance(v, str) else v}\n' for k, v in Y.items()))
        return path

    @classmethod
    def load_picasso(cls, path, pixel_nm=None):
        """A Picasso hdf5 ('locs'): x, y (z) -> pos, lpx, lpy (lpz) -> crlb, sx, sy -> psf_sx, psf_sy, other fields by
        name; image size, frames, pixel size from the .yaml next to it."""
        import h5py
        with h5py.File(path, 'r') as f:
            L = f['locs'][:]
        names = L.dtype.names
        D = 3 if 'z' in names else 2
        pos = np.stack([L[k].astype(np.float64) for k in ('x', 'y', 'z')[:D]], 1)
        F = dict(pos=pos, frame=L['frame'].astype(np.int32))
        if 'lpx' in names:
            F['crlb'] = np.stack([L[k] for k in ('lpx', 'lpy', 'lpz')[:D] if k in names], 1).astype(np.float32)
        ren = dict(sx='psf_sx', sy='psf_sy')
        for k in names:
            if k in ('x', 'y', 'z', 'lpx', 'lpy', 'lpz', 'frame') or k in ('ellipticity', 'net_gradient') and not L[k].any():
                continue
            F[ren.get(k, k)] = np.ascontiguousarray(L[k])
        meta = dict(source=os.path.abspath(str(path)), format='picasso hdf5')
        Y = _read_yaml(os.path.splitext(str(path))[0] + '.yaml')
        if 'Height' in Y and 'Width' in Y:
            meta['image_size'] = [int(Y['Height']), int(Y['Width'])]
        if 'Frames' in Y:
            meta['n_frames'] = int(Y['Frames'])
        px = pixel_nm or Y.get('Pixelsize')
        if px:
            meta['pixel_nm'] = float(px)
        return cls(F, meta)

    def save_csv(self, path, extra=True, frame_base=1, offset_px=0.5):
        """ThunderSTORM csv: "id","frame","x [nm]","y [nm]",("z [nm]",)"intensity [photon]","offset [photon]",
        "uncertainty [nm]","uncertainty_x [nm]","uncertainty_y [nm]" (and "uncertainty_z [nm]"); x [nm] = (x + offset_px)
        pixel_nm, frame + frame_base. extra: the other 1D fields under their names. Needs meta['pixel_nm']."""
        px = self.pixel_nm
        if px is None:
            raise ValueError("meta['pixel_nm'] is not set")
        F, n = self.numpy(), len(self)
        z0 = np.zeros(n)
        cr = F['crlb'].astype(np.float64) if 'crlb' in F else np.zeros((n, self.n_dims))
        C = [('id', F['id'], '%d'), ('frame', F['frame'].astype(np.int64) + frame_base, '%d'),
             ('x [nm]', (F['pos'][:, 0] + offset_px) * px, '%.4f'), ('y [nm]', (F['pos'][:, 1] + offset_px) * px, '%.4f')]
        if self.n_dims == 3:
            C.append(('z [nm]', F['pos'][:, 2] * px, '%.4f'))
        C += [('intensity [photon]', F.get('photons', z0), '%.8g'), ('offset [photon]', F.get('bg', z0), '%.8g'),
              ('uncertainty [nm]', np.sqrt(0.5 * (cr[:, 0] ** 2 + cr[:, 1] ** 2)) * px, '%.5f'),
              ('uncertainty_x [nm]', cr[:, 0] * px, '%.5f'), ('uncertainty_y [nm]', cr[:, 1] * px, '%.5f')]
        if self.n_dims == 3:
            C.append(('uncertainty_z [nm]', cr[:, 2] * px, '%.5f'))
        if extra:
            for k, v in F.items():
                if k not in CORE and v.ndim == 1:
                    C.append((k, v, '%d' if v.dtype.kind in 'iub' else '%.9g'))
        M = np.column_stack([np.asarray(v, np.float64) for _, v, _ in C]) if n else np.zeros((0, len(C)))
        np.savetxt(path, M, delimiter=',', header=','.join(f'"{k}"' for k, _, _ in C), comments='',
                   fmt=[f for _, _, f in C])
        return path

    @classmethod
    def load_csv(cls, path, pixel_nm, frame_base=1, offset_px=0.5):
        """A ThunderSTORM csv (positions in nm, frames from 1, first pixel centre at 0.5 px) -> Dataset in px."""
        with open(path) as f:
            hdr = [h.strip().strip('"') for h in f.readline().strip().split(',')]
        M = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        R = {h: M[:, i] for i, h in enumerate(hdr)}
        g = lambda *names: next((R.pop(n) for n in names if n in R), None)
        x, y, z = g('x [nm]'), g('y [nm]'), g('z [nm]')
        pos = [x / pixel_nm - offset_px, y / pixel_nm - offset_px] + ([z / pixel_nm] if z is not None else [])
        F = dict(pos=np.stack(pos, 1), frame=np.round(g('frame')).astype(np.int32) - frame_base)
        ux, uy, u, uz = g('uncertainty_x [nm]'), g('uncertainty_y [nm]'), g('uncertainty [nm]', 'uncertainty_xy [nm]'), \
            g('uncertainty_z [nm]')
        if ux is None and u is not None:
            ux = uy = u
        if ux is not None:
            F['crlb'] = np.stack([ux, uy] + ([uz] if (z is not None and uz is not None) else
                                             ([np.zeros_like(ux)] if z is not None else [])), 1) / pixel_nm
        for src, dst in (('intensity [photon]', 'photons'), ('offset [photon]', 'bg'), ('chi2', 'chi2')):
            v = g(src)
            if v is not None:
                F[dst] = v
        idv = g('id')
        if idv is not None:
            F['id'] = idv.astype(np.int64)
        for k, v in R.items():
            F[k.split(' [')[0]] = v
        return cls(F, dict(source=os.path.abspath(str(path)), format='thunderstorm csv', pixel_nm=float(pixel_nm)))


def _read_yaml(path):
    """top-level `key: value` lines of a (Picasso) .yaml -> dict; {} if there is no file"""
    Y = {}
    try:
        with open(path) as f:
            for line in f:
                if line[:1] in (' ', '\t', '-', '#', '\n') or ':' not in line:
                    continue
                k, v = line.split(':', 1)
                v = v.strip()
                if k in Y or not v:
                    continue
                try:
                    Y[k] = json.loads(v)
                except ValueError:
                    Y[k] = v.strip('\'"')
    except OSError:
        pass
    return Y
