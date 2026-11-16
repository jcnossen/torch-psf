"""Field-dependent PSFs (single channel): Zernike coefficients as low-order polynomials of the field position.

    field = FieldZernikes.from_positions([5, 6, 7, 8, 11], order={5: 2, 6: 2, 11: 2, 7: 1, 8: 1}, xy=spot_xy)
    psf = FieldPSF(vectorial, field)                  # the vectorial model with Zernikes base + field(x, y)
    psf.render(pos, xy=field_xy)                      # every emitter at its own field position (camera px)
    psf.at(x, y)                                      # a VectorialPSF at one field point
    ds = psf.fit(rois, corners=corners)               # localization: each ROI with the PSF at its centre
    tiled = psf.to_tiled(tile=128, roisize=15)        # fast form: one CubicSplinePSF per tile, picked by position
    io.save_zip('psf.zip', psf); PSF.load('psf.zip')  # the psf.zip 'zernike_poly' field block (docs/PSF_FORMAT.md)

Field model (FieldZernikes): for each Noll index j of `noll` with order o_j (0, 1 or 2)

    W_j(x, y) = W_j(ref) + sum_{1 <= i + k <= o_j} c_{j,ik} (u^i v^k - u_ref^i v_ref^k),
    u = clip((x - cx) / sx, -1, 1), v = clip((y - cy) / sy, -1, 1)

with the PolyMap features and normalisation of psf.zip (torch_psf.psfzip.poly_features: u^i v^k for i + k <= order,
i the outer loop; feats_norm = (cx, cy, sx, sy)) and x = column, y = row in camera (channel crop) px. The value at the
reference point (default the feats_norm centre, u = v = 0) is the base model's zcoef, so the base VectorialPSF is the
model at the reference point and the field terms are per-emitter offsets (VectorialPSF.forward(zcoef_offset=)).
Modes of the base model that are not in `noll` are constant over the field.

No extrapolation: u and v are clipped to [-1, 1], i.e. the polynomial is held constant outside the feats_norm box
(the clip is per coordinate: the bounding box, not the convex hull of the data). FieldZernikes.from_positions sets
feats_norm to the bounding box of the training positions, so the clip box is the fitted support. The same clip is
applied by the psf.zip reader (psfzip.FieldZernikePoly); a file written by SIMCODE (feats_norm = the crop centre and
half size) is then clipped to the crop, which changes nothing inside the image.

Ridge prior: penalty() = 0.5 sum (c / prior_sd)^2 over the non-constant coefficients (waves at u or v = 1, i.e. at the
edge of the support): a Gaussian prior towards the field-constant model, in NLL units.

File form (psf.zip channel `field`, unchanged format): {"type": "zernike_poly", "order": max o_j, "zernike_noll": noll,
"coef": [k, nf] (column 0 = the value at u = v = 0; columns of features above a mode's order are 0), "feats_norm",
"reference_point"}. Per-mode orders are not a key of the block: they are the zero pattern of coef (from_block infers
them).

The tiled spline form (TiledSplinePSF): the field box split into tiles of `tile` px; per tile a CubicSplinePSF built
from the Zernikes at the tile centre (lazily, on first use); a spot uses the tile that holds its field position.
"""
import math
import numpy as np
import torch
from torch_psf.psf import PSF
from torch_psf.dataset import Dataset


def feature_powers(order):
    """[(i, k)] powers of the PolyMap features u^i v^k for i + k <= order, in file order (i outer, k inner)."""
    return [(i, k) for i in range(order + 1) for k in range(order + 1 - i)]


def n_features(order):
    return (order + 1) * (order + 2) // 2


def bbox_norm(xy, margin=0.0):
    """feats_norm (cx, cy, sx, sy) of the bounding box of positions xy [N, 2] (px), enlarged by margin px per side;
    sx, sy >= 1."""
    xy = np.asarray(xy.detach().cpu() if torch.is_tensor(xy) else xy, np.float64).reshape(-1, 2)
    lo, hi = xy.min(0) - margin, xy.max(0) + margin
    c, s = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1.0)
    return (float(c[0]), float(c[1]), float(s[0]), float(s[1]))


def _orders(noll, order):
    if isinstance(order, dict):
        return [int(order.get(j, order.get(str(j), 0))) for j in noll]
    if isinstance(order, (list, tuple, np.ndarray)):
        if len(order) != len(noll):
            raise ValueError('order: one per mode')
        return [int(o) for o in order]
    return [int(order)] * len(noll)


class FieldZernikes(torch.nn.Module):
    """Zernike coefficients of the modes `noll` as polynomials of the field position (module docstring).

    noll: Noll indices with a field term; order: int (every mode), list (per mode) or dict {noll: order} (0 / 1 / 2;
    0 = constant, the mode then has no field term); norm: feats_norm (cx, cy, sx, sy) px; ref: reference point (x, y)
    px (default (cx, cy)); prior_sd: sd (waves) of the ridge prior on the non-constant coefficients; clamp: clip u, v
    to [-1, 1] (default; no extrapolation beyond the feats_norm box).
    block_order: the feature set (default the highest per-mode order). Parameter coef [k, nf - 1] (waves; the
    non-constant features of the block order, masked by the per-mode order)."""

    def __init__(self, noll, order=1, norm=(0.0, 0.0, 1.0, 1.0), ref=None, prior_sd=0.05, clamp=True, coef=None,
                 device=None, block_order=None):
        super().__init__()
        self.noll = [int(j) for j in noll]
        self.orders = _orders(self.noll, order)
        if any(o < 0 for o in self.orders):
            raise ValueError('orders must be >= 0')
        self.order = max([0] + self.orders + [int(block_order or 0)])     # the feature set (psf.zip block order)
        self.norm = tuple(float(v) for v in norm)
        self.ref = (self.norm[0], self.norm[1]) if ref is None else (float(ref[0]), float(ref[1]))
        self.prior_sd = float(prior_sd)
        self.clamp = bool(clamp)
        pw = feature_powers(self.order)
        dev = torch.device(device) if device is not None else torch.device('cpu')
        deg = torch.tensor([i + k for i, k in pw[1:]], dtype=torch.long)
        mask = (deg[None, :] <= torch.tensor(self.orders, dtype=torch.long)[:, None]).float()
        self.register_buffer('mask', mask.to(dev))
        self.register_buffer('powers', torch.tensor(pw, dtype=torch.float32, device=dev))
        c = torch.zeros(len(self.noll), len(pw) - 1, device=dev)
        if coef is not None:
            c = torch.as_tensor(np.asarray(coef, np.float32), device=dev).reshape(len(self.noll), len(pw) - 1).clone()
        self.coef = torch.nn.Parameter(c * self.mask)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_positions(cls, noll, order, xy, margin=0.0, **kw):
        """feats_norm = the bounding box of the positions xy [N, 2] (camera px; e.g. the training spots or beads):
        the support of the fit, beyond which the polynomial is held constant."""
        return cls(noll, order, norm=bbox_norm(xy, margin), **kw)

    @property
    def device(self):
        return self.coef.device

    @property
    def n_params(self):
        return int(self.mask.sum())

    def extra_repr(self):
        return (f'noll={self.noll}, orders={self.orders}, norm={tuple(round(v, 2) for v in self.norm)}, '
                f'prior_sd={self.prior_sd}, clamp={self.clamp}')

    # ------------------------------------------------------------------ evaluation
    def uv(self, xy):
        """normalised (clipped) coordinates [N, 2] of positions xy [N, 2] px"""
        xy = torch.as_tensor(xy, dtype=torch.float32, device=self.device).reshape(-1, 2)
        cx, cy, sx, sy = self.norm
        uv = torch.stack([(xy[:, 0] - cx) / sx, (xy[:, 1] - cy) / sy], 1)
        return uv.clamp(-1.0, 1.0) if self.clamp else uv

    def features(self, xy):
        """PolyMap features [N, nf] at xy [N, 2] px (u, v clipped)"""
        uv = self.uv(xy)
        p = self.powers
        return uv[:, None, 0] ** p[None, :, 0] * uv[:, None, 1] ** p[None, :, 1]

    def forward(self, xy):
        """non-constant part [N, k] (waves) of the field polynomial at xy [N, 2] px (differentiable in coef)"""
        return self.features(xy)[:, 1:] @ (self.coef * self.mask).T

    def relative(self, xy):
        """W(xy) - W(ref) [N, k] (waves): the offset from the model at the reference point"""
        r = torch.tensor([self.ref], dtype=torch.float32, device=self.device)
        return self(xy) - self(r)

    def offsets(self, xy, zidx):
        """per-emitter Zernike offsets [N, len(zidx)] for a VectorialPSF with Noll indices zidx (its zcoef is the value
        at the reference point); modes of zidx without a field term get 0. Every field mode must be in zidx."""
        rel = self.relative(xy)
        zidx = list(zidx)
        miss = [j for j in self.noll if j not in zidx]
        if miss:
            raise ValueError(f'field modes {miss} are not modes of the model {zidx}')
        out = torch.zeros(rel.shape[0], len(zidx), device=rel.device, dtype=rel.dtype)
        cols = torch.tensor([zidx.index(j) for j in self.noll], device=rel.device, dtype=torch.long)
        return out.index_copy(1, cols, rel) if len(self.noll) else out

    def penalty(self):
        """ridge prior 0.5 sum (coef / prior_sd)^2 (NLL units; 0 when prior_sd is 0 or inf)"""
        if not self.prior_sd or not math.isfinite(self.prior_sd):
            return self.coef.sum() * 0.0
        return 0.5 * ((self.coef * self.mask / self.prior_sd) ** 2).sum()

    def project_(self):
        """zero the coefficients above each mode's order (after an optimiser step)"""
        with torch.no_grad():
            self.coef.mul_(self.mask)
        return self

    @torch.no_grad()
    def maps(self, x, y):
        """W(x, y) - W(ref) as numpy [..., k] (waves) on arrays x, y of one shape (e.g. a meshgrid)"""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        r = self.relative(torch.as_tensor(np.stack([x.ravel(), y.ravel()], 1), dtype=torch.float32))
        return r.cpu().numpy().reshape(x.shape + (len(self.noll),))

    def copy(self):
        return FieldZernikes(self.noll, self.orders, self.norm, self.ref, self.prior_sd, self.clamp,
                             self.coef.detach().cpu().numpy(), self.device, self.order)

    # ------------------------------------------------------------------ psf.zip block
    def to_block(self, waves_at_ref):
        """The psf.zip 'zernike_poly' block. waves_at_ref: {noll: waves} of the model at the reference point (the
        channel model's zernike_waves); coef[:, 0] is chosen so that the polynomial equals them there."""
        nf = n_features(self.order)
        c = (self.coef * self.mask).detach().cpu().double().numpy()
        r = self.features(torch.tensor([self.ref]))[0, 1:].detach().cpu().double().numpy()
        full = np.zeros((len(self.noll), nf))
        full[:, 1:] = c
        for k, j in enumerate(self.noll):
            full[k, 0] = float(waves_at_ref[j]) - float(c[k] @ r)
        blk = dict(type='zernike_poly', order=int(self.order), zernike_noll=list(self.noll),
                   coef=[[float(v) for v in row] for row in full],
                   feats_norm=dict(cx=self.norm[0], cy=self.norm[1], sx=self.norm[2], sy=self.norm[3]))
        if (abs(self.ref[0] - self.norm[0]) > 1e-9) or (abs(self.ref[1] - self.norm[1]) > 1e-9):
            blk['reference_point'] = dict(x=self.ref[0], y=self.ref[1])
        return blk

    @classmethod
    def from_block(cls, d, arrays=None, crop=None, device=None, prior_sd=0.05, clamp=True):
        """(FieldZernikes, const [k] numpy = coef[:, 0]) of a psf.zip 'zernike_poly' block; the per-mode order is
        the highest degree with a non-zero coefficient."""
        from torch_psf.psfzip import FieldZernikePoly
        f = FieldZernikePoly(d, arrays, crop)
        pw = feature_powers(f.order)
        deg = np.array([i + k for i, k in pw])
        orders = [int(deg[np.abs(f.coef[k]) > 0].max()) if (np.abs(f.coef[k, 1:]) > 0).any() else 0
                  for k in range(len(f.noll))]
        fz = cls(f.noll, orders, f.norm, f.ref, prior_sd, clamp, f.coef[:, 1:], device, block_order=f.order)
        return fz, f.coef[:, 0].copy()


# ====================================================================================================== models
class _Bound(PSF):
    """A field-dependent model with fixed per-emitter field positions xy [N, 2]: render(pos [N, 3]) of exactly
    those N emitters (used by FieldPSF.fit / TiledSplinePSF; localize must not split the batch)."""

    def __init__(self, parent, xy):
        self.parent, self.xy = parent, xy
        self.roisize, self.pixel_um, self.dev = parent.roisize, parent.pixel_um, parent.dev

    @property
    def z_range(self):
        return self.parent.z_range

    @property
    def z_limits(self):
        return self.parent.z_limits

    @property
    def xy_limit(self):
        return self.parent.xy_limit

    def render(self, pos, grad=False):
        if len(pos) != len(self.xy):
            raise ValueError(f'bound field model: {len(pos)} emitters for {len(self.xy)} field positions')
        return self.parent.render(pos, grad=grad, xy=self.xy)


def _field_positions(rois, corners, xy, dev):
    """field positions [N, 2] of ROIs: xy, else the ROI centres corners + (P - 1) / 2"""
    if xy is not None:
        return torch.as_tensor(xy, dtype=torch.float32, device=dev).reshape(-1, 2)
    if corners is None:
        raise ValueError('a field-dependent PSF needs the ROI corners (or xy=field positions) to fit')
    h = (rois.shape[-1] - 1) / 2
    return torch.as_tensor(corners, device=dev).float().reshape(-1, 2) + h


def _take(a, idx, N):
    """rows idx of a per-ROI array / tensor (length N); scalars and None unchanged"""
    if a is None:
        return None
    if torch.is_tensor(a):
        return a if (a.dim() == 0 or len(a) != N) else a[torch.as_tensor(idx, device=a.device)]
    a = np.asarray(a)
    return a if (a.ndim == 0 or len(a) != N) else a[idx]


def _fit_groups(models, groups, rois, corners, frames, var, theta0, kw):
    """PSF.fit of every group (key, ROI indices) with the model models(key, idx); the Datasets reassembled in the
    input order."""
    from torch_psf.spots import Spots
    if isinstance(rois, Spots):
        corners = rois.corner if corners is None else corners
        frames = rois.frame if frames is None else frames
        var = rois.var if var is None else var
        rois = rois.rois
    N = len(rois)
    out, order = [], []
    for g, idx in groups:
        out.append(PSF.fit(models(g, idx), _take(rois, idx, N), theta0=_take(theta0, idx, N),
                           corners=_take(corners, idx, N), frames=_take(frames, idx, N), var=_take(var, idx, N), **kw))
        order.append(np.asarray(idx))
    if not out:
        return None
    ds = Dataset.concatenate(out)
    inv = np.argsort(np.concatenate(order), kind='stable')
    ds = ds[torch.as_tensor(inv)]
    ds['id'] = torch.arange(len(ds), dtype=torch.int64)
    return ds


class FieldPSF(PSF):
    """A VectorialPSF `base` with field-dependent Zernikes `field` (FieldZernikes): the base is the model at the
    reference point, an emitter at field position (x, y) (camera px) has zcoef = base.zcoef + field.offsets(x, y).

    render(pos, grad=False, xy=None): xy [N, 2] field positions (None: the reference point); at(x, y): a VectorialPSF
    at one field point; zernikes_at(xy) -> [N, nz] waves; fit(rois, corners=..., xy=None): each ROI with the PSF at
    its centre (corners + (P - 1) / 2) or at xy; to_tiled(tile, ...): the tiled spline form; field_block(): the
    psf.zip field block (io.save_zip writes it; PSF.load returns a FieldPSF for a vectorial channel with it)."""

    def __init__(self, base, field):
        if not hasattr(base, 'zidx'):
            raise TypeError(f'FieldPSF needs a VectorialPSF, not {type(base).__name__}')
        self.base, self.field = base, field.to(base.dev)
        self.field.offsets(torch.zeros(1, 2), base.zidx)          # checks that the modes are in the model
        self.roisize, self.pixel_um = base.roisize, base.pixel_um
        self.provenance = getattr(base, 'provenance', None)
        self.spline_grid = None      # set by the reader for a spline channel: the stored spline grid (to_tiled default)

    @property
    def dev(self):
        return self.base.dev

    @property
    def z_range(self):
        return self.base.z_range

    @property
    def focus_um(self):
        return self.base.focus_um

    @property
    def xy_limit(self):
        return self.base.xy_limit

    def __repr__(self):
        return f'FieldPSF({self.base!r}, {self.field!r})'

    # ------------------------------------------------------------------ evaluation
    def offsets(self, xy):
        return self.field.offsets(xy, self.base.zidx)

    def zernikes_at(self, xy):
        """[N, nz] Zernike coefficients (waves, the base's Noll order) at field positions xy [N, 2]"""
        with torch.no_grad():
            return (self.base.zcoef[None] + self.offsets(xy)).cpu().numpy()

    def at(self, x, y):
        """a VectorialPSF (copy) at the field point (x, y)"""
        m = self.base.resized(self.base.roisize)
        with torch.no_grad():
            m.zcoef.copy_(self.base.zcoef + self.offsets([[float(x), float(y)]])[0])
        m.provenance = self.provenance
        return m

    def render(self, pos, grad=False, xy=None, focus=None):
        """pos [N, 3] (x, y px from the ROI centre, z um) -> [N, P, P] at the base focus (or `focus`); xy [N, 2]:
        field positions (None = the reference point). grad: also d/dx, d/dy, d/dz (forward-mode AD)."""
        f = self.base.focus_um if focus is None else focus
        off = None if xy is None else self.offsets(xy)
        fn = lambda p: self.base(p, f, zcoef_offset=off)
        if not grad:
            return fn(pos)
        out, val = [], None
        for k in range(3):
            t = torch.zeros_like(pos)
            t[:, k] = 1
            v, d = torch.func.jvp(fn, (pos,), (t,))
            val = v if val is None else val
            out.append(d)
        return val, out[0], out[1], out[2]

    def bind(self, xy):
        """a PSF that renders exactly len(xy) emitters at the field positions xy [N, 2]"""
        return _Bound(self, torch.as_tensor(xy, dtype=torch.float32, device=self.dev).reshape(-1, 2))

    def fit(self, rois, camera=None, theta0=None, corners=None, frames=None, var=None, iters=30, zstarts=5,
            batch=1500, pixel_um=None, xy=None):
        """PSF.fit with each ROI at its own field position (xy [N, 2], default the ROI centres corners + (P - 1) / 2;
        corners also come with Spots). Returns a Dataset in the input order."""
        from torch_psf.spots import Spots
        if not isinstance(rois, Spots):
            rois = torch.as_tensor(rois, dtype=torch.float32, device=self.dev)
        r = rois.rois if isinstance(rois, Spots) else rois
        cor = rois.corner if (isinstance(rois, Spots) and corners is None) else corners
        fxy = _field_positions(r, cor, xy, self.dev)
        N = len(r)
        groups = [(s, np.arange(s, min(s + batch, N))) for s in range(0, N, batch)]
        kw = dict(camera=camera, iters=iters, zstarts=zstarts, batch=batch, pixel_um=pixel_um or self.pixel_um)
        ds = _fit_groups(lambda g, idx: self.bind(fxy[torch.as_tensor(idx, device=fxy.device)]), groups, rois, corners,
                         frames, var, theta0, kw)
        if ds is None:
            return PSF.fit(self.base, torch.zeros(0, self.roisize, self.roisize, device=self.dev), pixel_um=pixel_um)
        ds.meta['method'] = 'torch_psf FieldPSF.fit'
        return ds

    # ------------------------------------------------------------------ copies, conversion
    def resized(self, roisize):
        p = FieldPSF(self.base.resized(roisize), self.field)
        p.spline_grid, p.provenance = self.spline_grid, self.provenance
        return p

    def to(self, device):
        self.base.to(device)
        self.field.to(device)
        return self

    def to_spline(self, *a, **kw):
        """the CubicSplinePSF of the model at the reference point (VectorialPSF.to_spline); to_tiled for the field
        dependence"""
        return self.base.to_spline(*a, **kw)

    def to_tiled(self, tile=128, extent=None, roisize=None, z_range=None, lazy=True, **grid):
        """TiledSplinePSF: tiles of `tile` px over extent (x0, y0, x1, y1) px (default the field box
        cx -+ sx, cy -+ sy), each a CubicSplinePSF of the model at the tile centre (grid: oversample, dz_um, zpad,
        xy_limit of to_spline; default the stored spline grid of a file, else the to_spline defaults)."""
        g = dict(self.spline_grid or {})
        g.update(grid)
        r0, z0 = g.pop('roisize', None), g.pop('z_range', None)
        return TiledSplinePSF(self, tile=tile, extent=extent, roisize=roisize or r0 or self.roisize,
                              z_range=z_range or z0 or self.z_range, lazy=lazy, **g)

    def field_block(self):
        return self.field.to_block(dict(zip(self.base.zidx, self.base.zcoef.detach().cpu().double().tolist())))

    def file_model(self):
        """(model written to the channel, source or None) for io.save_zip"""
        return self.base, None


class TiledSplinePSF(PSF):
    """The tiled spline form of a FieldPSF: the extent (x0, y0, x1, y1) px split into tiles of `tile` px, per tile a
    CubicSplinePSF (roisize, z_range, grid: oversample, dz_um, zpad, xy_limit) of the vectorial model at the tile
    centre, built on first use (lazy) or at once. A spot uses the tile that holds its field position (positions
    outside the extent use the nearest tile).

    render(pos, grad=False, xy=None) (xy None: the reference-point spline), tile_index(xy), spline(t), fit(rois,
    corners=..., xy=None): every ROI fitted with its tile's spline (input order kept)."""

    def __init__(self, field_psf, tile=128, extent=None, roisize=None, z_range=None, lazy=True, **grid):
        self.source = field_psf
        f = field_psf.field
        if extent is None:
            cx, cy, sx, sy = f.norm
            extent = (cx - sx, cy - sy, cx + sx, cy + sy)
        self.extent = tuple(float(v) for v in extent)
        self.tile = float(tile)
        x0, y0, x1, y1 = self.extent
        self.nx = max(1, int(math.ceil((x1 - x0) / self.tile - 1e-9)))
        self.ny = max(1, int(math.ceil((y1 - y0) / self.tile - 1e-9)))
        self.roisize = int(roisize or field_psf.roisize)
        self._zr = tuple(z_range or field_psf.z_range)
        self.grid = dict(grid)
        self.pixel_um = field_psf.pixel_um
        self.provenance = field_psf.provenance
        self._splines = {}
        self._ref = None
        if not lazy:
            for t in range(self.n_tiles):
                self.spline(t)

    @property
    def dev(self):
        return self.source.dev

    @property
    def n_tiles(self):
        return self.nx * self.ny

    @property
    def centers(self):
        """tile centres [T, 2] px (row-major: t = iy nx + ix)"""
        x0, y0, _, _ = self.extent
        ix, iy = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        return np.stack([x0 + (ix.ravel() + 0.5) * self.tile, y0 + (iy.ravel() + 0.5) * self.tile], 1)

    def tile_index(self, xy):
        """tile [N] (long) of field positions xy [N, 2] px"""
        xy = torch.as_tensor(xy, dtype=torch.float32).reshape(-1, 2)
        x0, y0, _, _ = self.extent
        ix = torch.floor((xy[:, 0] - x0) / self.tile).long().clamp(0, self.nx - 1)
        iy = torch.floor((xy[:, 1] - y0) / self.tile).long().clamp(0, self.ny - 1)
        return iy * self.nx + ix

    def _build(self, vec):
        return vec.to_spline(self.roisize, self._zr, **self.grid)

    def spline(self, t):
        """the CubicSplinePSF of tile t"""
        if t not in self._splines:
            c = self.centers[t]
            self._splines[t] = self._build(self.source.at(c[0], c[1]))
        return self._splines[t]

    def reference_spline(self):
        """the spline of the model at the reference point (the channel model written to psf.zip)"""
        if self._ref is None:
            self._ref = self._build(self.source.base)
        return self._ref

    @property
    def z_range(self):
        return self._zr

    @property
    def z_limits(self):
        return self.reference_spline().z_limits

    @property
    def xy_limit(self):
        return self.reference_spline().xy_limit

    def __repr__(self):
        return (f'TiledSplinePSF({self.nx} x {self.ny} tiles of {self.tile:g} px over {self.extent}, roisize '
                f'{self.roisize}, {len(self._splines)} built)')

    def render(self, pos, grad=False, xy=None):
        if xy is None:
            return self.reference_spline().render(pos, grad)
        ti = self.tile_index(xy).to(pos.device)
        res = None
        for t in torch.unique(ti).tolist():
            m = ti == t
            r = self.spline(t).render(pos[m], grad)
            r = r if grad else (r,)
            if res is None:
                res = [torch.zeros((len(pos),) + v.shape[1:], device=v.device, dtype=v.dtype) for v in r]
            for a, v in zip(res, r):
                a[m] = v
        if res is None:
            return self.reference_spline().render(pos, grad)
        return tuple(res) if grad else res[0]

    def bind(self, xy):
        return _Bound(self, torch.as_tensor(xy, dtype=torch.float32, device=self.dev).reshape(-1, 2))

    def fit(self, rois, camera=None, theta0=None, corners=None, frames=None, var=None, iters=30, zstarts=5,
            batch=4000, pixel_um=None, xy=None):
        """PSF.fit of every ROI with the spline of its tile (field position xy, default the ROI centre
        corners + (P - 1) / 2). Returns a Dataset in the input order."""
        from torch_psf.spots import Spots
        if not isinstance(rois, Spots):
            rois = torch.as_tensor(rois, dtype=torch.float32, device=self.dev)
        r = rois.rois if isinstance(rois, Spots) else rois
        cor = rois.corner if (isinstance(rois, Spots) and corners is None) else corners
        ti = self.tile_index(_field_positions(r, cor, xy, self.dev)).cpu().numpy()
        groups = [(int(t), np.flatnonzero(ti == t)) for t in np.unique(ti)]
        kw = dict(camera=camera, iters=iters, zstarts=zstarts, batch=batch, pixel_um=pixel_um or self.pixel_um)
        ds = _fit_groups(lambda g, idx: self.spline(g), groups, rois, corners, frames, var, theta0, kw)
        if ds is None:
            return PSF.fit(self.reference_spline(), torch.zeros(0, self.roisize, self.roisize, device=self.dev),
                           pixel_um=pixel_um or self.pixel_um)
        ds.meta['method'] = 'torch_psf TiledSplinePSF.fit'
        return ds

    def resized(self, roisize):
        t = TiledSplinePSF(self.source, self.tile, self.extent, roisize, self._zr, True, **self.grid)
        t._splines = {k: s.resized(roisize) for k, s in self._splines.items()}
        return t

    def to(self, device):
        self.source.to(device)
        for s in self._splines.values():
            s.to(device)
        if self._ref is not None:
            self._ref.to(device)
        return self

    def to_spline(self, roisize=None, z_range=None, **kw):
        return self.reference_spline().to_spline(roisize, z_range, **kw)

    def field_block(self):
        return self.source.field_block()

    def file_model(self):
        return self.reference_spline(), self.source.base

    def tiles_info(self):
        return dict(tile_px=self.tile, extent=list(self.extent), roisize=self.roisize)


FIELD_SPEC = dict(modes=None, order=1, prior_sd=0.05, margin=0.0, tile=256.0, score_tile=None, iters=60, lr=2e-3,
                  lr_field=None)


def field_from_spec(spec, zidx, xy, device=None):
    """(FieldZernikes, settings) of an estimator's field= option. spec: True / an int (the order of every mode) / a
    dict with keys modes (Noll indices with a field term; default every mode of the model zidx), order (int, list or
    {noll: order}; default 1), prior_sd (waves, 0.05), margin (px added to the bounding box of xy), and the estimation
    settings tile (px; in situ), score_tile (px of the tiled spline used for scoring and localization; default tile /
    2), iters, lr, lr_field. xy [N, 2]: the training positions (the support of the polynomial)."""
    if spec is True:
        spec = {}
    elif isinstance(spec, int) and not isinstance(spec, bool):
        spec = dict(order=spec)
    s = dict(FIELD_SPEC)
    unknown = set(spec) - set(s)
    if unknown:
        raise ValueError(f'unknown field settings {sorted(unknown)} (known: {sorted(s)})')
    s.update(spec)
    modes = list(zidx) if s['modes'] is None else [int(j) for j in s['modes']]
    fz = FieldZernikes.from_positions(modes, s['order'], xy, margin=s['margin'], prior_sd=s['prior_sd'],
                                      device=device)
    if s['score_tile'] is None:
        s['score_tile'] = float(s['tile']) / 2
    return fz, s


def field_psf_from_file(model, source, block, arrays=None, crop=None, tiles=None):
    """The field-dependent model of a psf.zip channel with a 'zernike_poly' field block: FieldPSF of the vectorial
    model (or of the vectorial source of a spline channel); for a spline channel a TiledSplinePSF on the stored
    spline grid (tiles: provenance['field_tiles'] {tile_px, extent} of the writer, else 128 px tiles over the field
    box). The block's polynomial at the reference point should equal the model's zernike_waves (the writer warns);
    the model's values are used."""
    vec = model if hasattr(model, 'zidx') else source
    if vec is None:
        raise ValueError('a zernike_poly field block needs a vectorial model or a vectorial source')
    fz, _ = FieldZernikes.from_block(block, arrays, crop, device=vec.dev)
    fp = FieldPSF(vec, fz)
    if hasattr(model, 'zidx'):
        return fp
    g = model.to_dict()
    fp.spline_grid = dict(oversample=g['oversample'], dz_um=g['dz_um'], zpad=g['zpad'], xy_limit=g['xy_limit'],
                          roisize=g['roisize'], z_range=tuple(g['z_range']))
    t = dict(tiles or {})
    tp = fp.to_tiled(tile=t.get('tile_px', 128), extent=t.get('extent'))
    tp._ref = model
    return tp


def unwrap(psf):
    """(model, source, field block, tiles info) for writing: a FieldPSF / TiledSplinePSF -> its file model, else
    (psf, None, None, None)"""
    if hasattr(psf, 'field_block') and hasattr(psf, 'file_model'):
        m, s = psf.file_model()
        return m, s, psf.field_block(), (psf.tiles_info() if hasattr(psf, 'tiles_info') else None)
    return psf, None, None, None
