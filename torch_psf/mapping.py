"""Channel mappings as differentiable torch models, bead registration between channels, and per-channel ROIs.

A multi-channel setup (biplane, multicolour, dual view on one or several cameras) sees one emitter in C channels. The
reference channel 0 holds the emitter position (x0, y0); channel c sees it at

    (x_c, y_c) = from_ref_c(x0, y0)

in channel c's crop coordinates (x = column, y = row, pixel centres at integers; docs/PSF_FORMAT.md). ChannelMap is
that map as a torch module, with the psf.zip mapping block types:

    identity   from_ref(p) = p
    affine     an order-1 polynomial (the psf.zip 'affine' block holds its inverse, channel -> reference, as a 2 x 3
               matrix); ChannelMap.translation(d): from_ref(p) = p + d, the constant `channel_offsets` of a
               MultiChannelPSF as a mapping
    poly       the SIMCODE PolyMap (psf.zip 'poly' block): from_ref = feats(x0, y0) @ inv, to_ref = feats(x, y) @ fwd,
               features u^i v^j (i + j <= order, i outer), u = (x - cx) / sx, v = (y - cy) / sy, one feats_norm for
               both directions (the channels' crops have the same size)

The model direction is from_ref (reference -> channel: the models render an emitter at its reference position in every
channel), parameter `inv` [nf, 2] in px (the coefficients of normalised features, so one Adam step size fits all of
them). to_ref is the inverse: exact for order 1, else the polynomial `fwd` (refit_fwd(): least squares on a grid through
from_ref) refined by Newton steps on from_ref, so to_ref(from_ref(p)) = p to ~1e-6 px either way. jacobian(p) [N, 2, 2] =
d from_ref / d p.

    m = ChannelMap.fit_points(ref_xy, ch_xy, order=2, norm=bbox)    # least squares
    m, pairs, info = register_points(ref_xy, ch_xy, frame_shape)     # bead matching incl. flips, then affine / poly
    m.to_block(1); ChannelMap.from_block(block, 1)                   # psf.zip mapping block (format unchanged)
    rois, corners = channel_rois(images, frame, xy_ref, maps, 15)    # [N, C, P, P] and corners [N, C, 2]
"""
import math
import numpy as np
import torch


def feature_powers(order):
    """[(i, j)] of the PolyMap features u^i v^j, i + j <= order, i outer (psf.zip / SIMCODE order)"""
    return [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]


def n_features(order):
    return (order + 1) * (order + 2) // 2


def _t(v, dev=None, dtype=torch.float32):
    return torch.as_tensor(np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, np.float64), dtype=dtype,
                           device=dev)


def frame_norm(w, h):
    """feats_norm (cx, cy, sx, sy) of a w x h crop as SIMCODE's PolyMap.from_dict(size=(w, h)): (w / 2, h / 2, w / 2,
    h / 2)"""
    return (w / 2.0, h / 2.0, w / 2.0, h / 2.0)


class ChannelMap(torch.nn.Module):
    """from_ref / to_ref between channel `index` and the reference channel 0 (module docstring).

    ChannelMap(order=1, norm=(cx, cy, sx, sy), inv=None [nf, 2] px (default identity), fwd=None, index=1,
    identity=False, fit='all'): `fit` selects the trainable coefficients when inv is fitted ('all', 'shift': the
    constant column only, 'affine': up to order 1); others have zero gradient (mask). Float64 coefficients (the maps
    are evaluated in float64 and returned in the input dtype: px positions of 500 px need more than float32 for
    1e-5 px round trips)."""

    def __init__(self, order=1, norm=(0.0, 0.0, 1.0, 1.0), inv=None, fwd=None, index=1, identity=False, fit='all',
                 device=None):
        super().__init__()
        self.order, self.index, self.identity = int(order), int(index), bool(identity)
        self.norm = tuple(float(v) for v in norm)
        pw = feature_powers(self.order)
        nf = len(pw)
        dev = torch.device(device) if device is not None else torch.device('cpu')
        self.register_buffer('powers', torch.tensor(pw, dtype=torch.float64, device=dev))
        if inv is None:
            inv = self.identity_coef(self.order, self.norm)
        self.inv = torch.nn.Parameter(_t(inv, dev, torch.float64).reshape(nf, 2).clone())
        self.register_buffer('fwd', _t(fwd, dev, torch.float64).reshape(nf, 2).clone() if fwd is not None
                             else self.identity_coef(self.order, self.norm).to(dev))
        self.set_fit(fit)
        self._fwd_stale = fwd is None and self.order > 1

    # ------------------------------------------------------------------ construction
    @staticmethod
    def identity_coef(order, norm):
        """coefficients [nf, 2] of the identity map for these features: x = cx + sx u, y = cy + sy v"""
        cx, cy, sx, sy = norm
        pw = feature_powers(order)
        c = torch.zeros(len(pw), 2, dtype=torch.float64)
        c[pw.index((0, 0))] = torch.tensor([cx, cy], dtype=torch.float64)
        if order >= 1:
            c[pw.index((1, 0)), 0] = sx
            c[pw.index((0, 1)), 1] = sy
        return c

    @classmethod
    def identity_map(cls, index=0, norm=(0.0, 0.0, 1.0, 1.0), device=None):
        return cls(1, norm, index=index, identity=True, device=device)

    @classmethod
    def translation(cls, d, index=1, norm=(0.0, 0.0, 1.0, 1.0), device=None):
        """from_ref(p) = p + d (d = (dx, dy) px): MultiChannelPSF's constant channel offset as a mapping"""
        c = cls.identity_coef(1, norm)
        c[0] += torch.as_tensor(np.asarray(d, np.float64).reshape(2))
        return cls(1, norm, inv=c, index=index, fit='shift', device=device)

    @classmethod
    def from_affine(cls, A, index=1, norm=(0.0, 0.0, 1.0, 1.0), device=None):
        """from_ref(p) = A [p, 1] (A [2, 3], reference -> channel)"""
        A = np.asarray(A, np.float64).reshape(2, 3)
        cx, cy, sx, sy = norm
        inv = np.zeros((3, 2))                      # features 1, v, u (order 1: (0,0), (0,1), (1,0))
        inv[0] = A[:, 0] * cx + A[:, 1] * cy + A[:, 2]
        inv[1] = A[:, 1] * sy
        inv[2] = A[:, 0] * sx
        return cls(1, norm, inv=inv, index=index, device=device)

    @classmethod
    def fit_points(cls, ref_xy, ch_xy, order=1, norm=None, index=1, weights=None, device=None):
        """least-squares map ref_xy [n, 2] -> ch_xy [n, 2] (norm: default the bounding box of ref_xy); fwd fitted the
        other way on the same points (then refit_fwd() on a grid makes it the inverse of inv)"""
        r, c = np.asarray(ref_xy, np.float64).reshape(-1, 2), np.asarray(ch_xy, np.float64).reshape(-1, 2)
        if norm is None:
            lo, hi = r.min(0), r.max(0)
            norm = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, max((hi[0] - lo[0]) / 2, 1.0), max((hi[1] - lo[1]) / 2, 1.0))
        w = np.ones(len(r)) if weights is None else np.asarray(weights, np.float64)
        Fr, Fc = _features_np(r, order, norm), _features_np(c, order, norm)
        sw = np.sqrt(w)[:, None]
        inv = np.linalg.lstsq(Fr * sw, c * sw, rcond=None)[0]
        fwd = np.linalg.lstsq(Fc * sw, r * sw, rcond=None)[0]
        m = cls(order, norm, inv=inv, fwd=fwd, index=index, device=device)
        m._fwd_stale = order > 1
        return m

    @classmethod
    def from_block(cls, block, index, crop=None, arrays=None, device=None):
        """ChannelMap of a psf.zip mapping block (identity, affine, poly; a grid has no polynomial form: ValueError)"""
        from torch_psf.psfzip import Mapping
        mp = block if isinstance(block, Mapping) else Mapping(block, index, arrays, crop)
        if mp.type == 'identity':
            norm = frame_norm(crop['w'], crop['h']) if crop and crop.get('w') else (0.0, 0.0, 1.0, 1.0)
            return cls.identity_map(index, norm, device)
        if mp.type == 'affine':
            norm = frame_norm(crop['w'], crop['h']) if crop and crop.get('w') else (0.0, 0.0, 1.0, 1.0)
            return cls.from_affine(mp.Ainv, index, norm, device)
        if mp.type == 'poly':
            m = cls(mp.order, mp.norm, inv=mp.ci, fwd=mp.cf, index=index, device=device)
            m._fwd_stale = False
            return m
        raise ValueError(f'a {mp.type!r} mapping has no polynomial form (ChannelMap: identity / affine / poly)')

    def set_fit(self, fit='all'):
        """trainable coefficients of inv: 'all', 'affine' (order <= 1) or 'shift' (the constant column); the mask
        multiplies the gradient"""
        pw = feature_powers(self.order)
        deg = torch.tensor([i + j for i, j in pw], dtype=torch.float64)
        lim = {'all': self.order, 'affine': 1, 'shift': 0}[fit]
        self.fit_mode = fit
        self.register_buffer('mask', (deg <= lim).to(torch.float64)[:, None].expand(-1, 2).clone().to(self.inv.device))
        if self.inv.requires_grad and getattr(self, '_hook', None) is None:
            self._hook = self.inv.register_hook(lambda g: g * self.mask)
        return self

    @property
    def device(self):
        return self.inv.device

    def copy(self):
        m = ChannelMap(self.order, self.norm, self.inv.detach().clone(), self.fwd.clone(), self.index, self.identity,
                       self.fit_mode, self.device)
        m._fwd_stale = self._fwd_stale
        return m

    def extra_repr(self):
        kind = 'identity' if self.identity else ('affine' if self.order == 1 else f'poly order {self.order}')
        return f'index={self.index}, {kind}, norm={tuple(round(v, 2) for v in self.norm)}'

    # ------------------------------------------------------------------ evaluation
    def _feats(self, xy, d=False):
        cx, cy, sx, sy = self.norm
        u, v = (xy[:, 0] - cx) / sx, (xy[:, 1] - cy) / sy
        p = self.powers
        i, j = p[:, 0], p[:, 1]
        F = u[:, None] ** i * v[:, None] ** j
        if not d:
            return F
        du = torch.where(i > 0, i * u[:, None] ** (i - 1).clamp_min(0), torch.zeros_like(F)) * v[:, None] ** j / sx
        dv = torch.where(j > 0, j * v[:, None] ** (j - 1).clamp_min(0), torch.zeros_like(F)) * u[:, None] ** i / sy
        return F, du, dv

    def from_ref(self, xy):
        """reference -> channel coordinates, xy [N, 2] (tensor or array; differentiable in xy and inv)"""
        if not torch.is_tensor(xy):
            return self.from_ref(_t(xy, self.device, torch.float64)).detach().cpu().numpy()
        if self.identity:
            return xy
        x = xy.reshape(-1, 2).to(torch.float64)
        return (self._feats(x) @ self.inv).to(xy.dtype).reshape(xy.shape)

    def jacobian(self, xy):
        """d from_ref / d xy [N, 2, 2] (row: output x / y, column: input x / y)"""
        x = torch.as_tensor(xy).reshape(-1, 2).to(device=self.device, dtype=torch.float64)
        if self.identity:
            return torch.eye(2, dtype=x.dtype, device=x.device).expand(len(x), 2, 2)
        _, du, dv = self._feats(x, d=True)
        return torch.stack([du @ self.inv, dv @ self.inv], 2)                 # [N, 2 (out), 2 (in)]

    def linear(self):
        """(A [2, 2], b [2]) of the order-1 part at the norm centre: from_ref(p) ~ A p + b"""
        c = torch.tensor([[self.norm[0], self.norm[1]]], dtype=torch.float64, device=self.device)
        with torch.no_grad():
            A = self.jacobian(c)[0]
            b = self.from_ref(c)[0] - A @ c[0]
        return A.cpu().numpy(), b.cpu().numpy()

    @torch.no_grad()
    def to_ref(self, xy, newton=4):
        """channel -> reference coordinates (the inverse of from_ref: exact for order 1, else fwd + Newton steps)"""
        if not torch.is_tensor(xy):
            return self.to_ref(_t(xy, self.device, torch.float64), newton).cpu().numpy()
        if self.identity:
            return xy
        x = xy.reshape(-1, 2).to(device=self.device, dtype=torch.float64)
        if self.order == 1:
            A, b = (torch.as_tensor(v, dtype=torch.float64, device=self.device) for v in self.linear())
            p = (x - b) @ torch.linalg.inv(A).T
        else:
            p = self._feats(x) @ self.fwd if not self._fwd_stale else self._affine_inverse(x)
            for _ in range(int(newton)):
                r = self.from_ref(p) - x
                p = p - torch.linalg.solve(self.jacobian(p), r[..., None])[..., 0]
        return p.to(xy.dtype).reshape(xy.shape)

    def _affine_inverse(self, x):
        A, b = (torch.as_tensor(v, dtype=torch.float64, device=self.device) for v in self.linear())
        return (x - b) @ torch.linalg.inv(A).T

    @torch.no_grad()
    def refit_fwd(self, extent=None, n=41):
        """fit the fwd polynomial (channel -> reference) as the inverse of from_ref on an n x n grid over extent (x0,
        y0, x1, y1) in channel coordinates (default the norm box enlarged by 10 %); returns the rms round-trip error
        (px) of the polynomial alone on that grid"""
        if self.identity:
            return 0.0
        cx, cy, sx, sy = self.norm
        if extent is None:
            extent = (cx - 1.1 * sx, cy - 1.1 * sy, cx + 1.1 * sx, cy + 1.1 * sy)
        gx, gy = np.meshgrid(np.linspace(extent[0], extent[2], n), np.linspace(extent[1], extent[3], n))
        ch = torch.as_tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float64, device=self.device)
        self._fwd_stale = True
        ref = self.to_ref(ch, newton=8)
        F = self._feats(ch)
        self.fwd.copy_(torch.linalg.lstsq(F, ref).solution)
        self._fwd_stale = False
        return float(((F @ self.fwd - ref) ** 2).sum(1).mean().sqrt())

    def flip(self):
        """'none' / 'lr' / 'ud' / 'both' from the signs of the linear part (the mirror of the channel's optics
        relative to channel 0; psf.zip `flip`)"""
        A, _ = self.linear()
        lr, ud = A[0, 0] < 0, A[1, 1] < 0
        return {(False, False): 'none', (True, False): 'lr', (False, True): 'ud', (True, True): 'both'}[(lr, ud)]

    def describe(self):
        """dict: shift (px at the norm centre), rotation (deg), scale x / y, shear, determinant, flip"""
        A, b = self.linear()
        cx, cy = self.norm[:2]
        c = A @ np.array([cx, cy]) + b - np.array([cx, cy])
        sgn = np.diag([np.sign(A[0, 0]) or 1.0, np.sign(A[1, 1]) or 1.0])
        B = A @ sgn
        rot = math.degrees(math.atan2(B[1, 0] - B[0, 1], B[0, 0] + B[1, 1]))
        return dict(shift_px=c.tolist(), rotation_deg=rot, scale=[float(np.hypot(*A[:, 0])), float(np.hypot(*A[:, 1]))],
                    det=float(np.linalg.det(A)), flip=self.flip(), A=A.tolist(), b=b.tolist())

    # ------------------------------------------------------------------ psf.zip block
    def to_block(self, index=None, extent=None, kind=None, **meta):
        """psf.zip mapping block of this channel: identity, 'affine' for order 1 (matrix channel -> reference), else
        'poly' (SIMCODE PolyMap dict: fwd / inv keyed by the channel index, feats_norm; fwd refitted as the inverse of
        inv first; `roundtrip_px` = its rms error). kind='poly' writes an order-1 map as a poly block. meta: extra keys
        (kept by readers)."""
        idx = self.index if index is None else int(index)
        if self.identity:
            return dict(type='identity')
        if self.order == 1 and kind != 'poly':
            A, b = self.linear()
            Ai = np.linalg.inv(A)
            M = np.concatenate([Ai, (-Ai @ b)[:, None]], 1)
            return dict(type='affine', matrix=M.tolist(), **meta)
        rt = self.refit_fwd(extent) if self.order > 1 else 0.0
        if self.order == 1:
            self.refit_fwd(extent)
        cx, cy, sx, sy = self.norm
        return dict(type='poly', order=self.order, fwd={str(idx): self.fwd.cpu().numpy().tolist()},
                    inv={str(idx): self.inv.detach().cpu().numpy().tolist()},
                    feats_norm=dict(cx=cx, cy=cy, sx=sx, sy=sy), roundtrip_px=rt, **meta)


def _features_np(xy, order, norm):
    cx, cy, sx, sy = norm
    u, v = (xy[:, 0] - cx) / sx, (xy[:, 1] - cy) / sy
    return np.stack([u ** i * v ** j for i, j in feature_powers(order)], 1)


# ====================================================================================================== registration
def _flip_xy(xy, flip, shape):
    q = np.array(xy, np.float64, copy=True)
    H, W = shape
    if flip in ('lr', 'both'):
        q[:, 0] = W - 1 - q[:, 0]
    if flip in ('ud', 'both'):
        q[:, 1] = H - 1 - q[:, 1]
    return q


def match_pairs(a, b, max_dist):
    """mutual nearest neighbours between point sets a [n, 2] and b [m, 2] closer than max_dist: (ia, ib)"""
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    da, ia = cKDTree(b).query(a)
    db, ib = cKDTree(a).query(b)
    ok = (da < max_dist) & (ib[ia] == np.arange(len(a)))
    return np.nonzero(ok)[0], ia[ok]


def register_points(ref_xy, ch_xy, frame_shape=None, flips=('none', 'lr', 'ud', 'both'), max_shift=None, bin_px=2.0,
                    tol_px=(4.0, 2.0, 1.5), order=1, norm=None, index=1, ref_amp=None, ch_amp=None, max_points=None):
    """Register bead detections of channel c (ch_xy [m, 2], channel crop px) to those of the reference channel
    (ref_xy [n, 2]) without a starting guess: for each flip candidate (of the channel crop, frame_shape (H, W)) the
    shift is the peak of the histogram of all pairwise differences (bin_px bins); the best flip / shift is refined by
    mutual-nearest-neighbour matching and a least-squares affine map, repeated with the tolerances tol_px, then a
    polynomial of `order` on the final pairs. ref_amp / ch_amp with max_points: only the brightest points take part
    in the histogram. Returns (ChannelMap (from_ref: reference -> channel), (i_ref, i_ch) the matched pairs, info:
    flip, histogram peak, n_pairs, rms_px of the final fit, residuals [n_pairs, 2])."""
    r, c = np.asarray(ref_xy, np.float64).reshape(-1, 2), np.asarray(ch_xy, np.float64).reshape(-1, 2)
    if len(r) < 3 or len(c) < 3:
        raise ValueError('register_points needs at least 3 points per channel')
    shape = frame_shape or (int(max(r[:, 1].max(), c[:, 1].max())) + 1, int(max(r[:, 0].max(), c[:, 0].max())) + 1)
    rs, cs = r, c
    if max_points is not None:
        if ref_amp is not None:
            rs = r[np.argsort(-np.asarray(ref_amp))[:max_points]]
        if ch_amp is not None:
            cs = c[np.argsort(-np.asarray(ch_amp))[:max_points]]
    lim = max_shift or max(shape)
    edges = np.arange(-lim, lim + bin_px, bin_px)
    best = None
    for fl in flips:
        q = _flip_xy(cs, fl, shape)
        d = (q[None] - rs[:, None]).reshape(-1, 2)
        Hh, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=[edges, edges])
        Hs = Hh + np.roll(Hh, 1, 0) + np.roll(Hh, 1, 1) + np.roll(np.roll(Hh, 1, 0), 1, 1)     # 2 x 2 bin sums
        i, j = np.unravel_index(Hs.argmax(), Hs.shape)
        score = Hs[i, j]
        if best is None or score > best[0]:
            best = (score, fl, np.array([ex[i], ey[j]]))
    score, fl, shift = best
    # flipped channel q = F(c) ~ ref + shift  ->  channel = F^-1(ref + shift)
    H, W = shape
    sx, sy = (-1.0 if fl in ('lr', 'both') else 1.0), (-1.0 if fl in ('ud', 'both') else 1.0)
    ox = (W - 1) if sx < 0 else 0.0
    oy = (H - 1) if sy < 0 else 0.0
    A = np.array([[sx, 0.0, sx * shift[0] + ox], [0.0, sy, sy * shift[1] + oy]])
    if norm is None:
        norm = frame_norm(W, H)
    m = ChannelMap.from_affine(A, index, norm)
    ir = ic = np.zeros(0, int)
    for tol in tol_px:
        prev = None
        for _ in range(10):                    # refit until the pair set is stable (the matches grow outwards)
            ir, ic = match_pairs(m.from_ref(r), c, tol)
            if len(ir) < 3:
                raise ValueError(f'register_points: only {len(ir)} pairs within {tol} px (flip {fl}, shift {shift})')
            m = ChannelMap.fit_points(r[ir], c[ic], 1, norm, index)
            key = (tuple(ir), tuple(ic))
            if key == prev:
                break
            prev = key
    if order > 1:
        m = ChannelMap.fit_points(r[ir], c[ic], order, norm, index)
        ir, ic = match_pairs(m.from_ref(r), c, tol_px[-1])
        m = ChannelMap.fit_points(r[ir], c[ic], order, norm, index)
    res = m.from_ref(r[ir]) - c[ic]
    info = dict(flip=fl, histogram_peak=float(score), coarse_shift=shift.tolist(), n_pairs=int(len(ir)),
                rms_px=float(np.sqrt((res ** 2).sum(1).mean())), residuals=res, order=int(order))
    return m, (ir, ic), info


# ====================================================================================================== ROIs
def channel_rois(images, frame, xy_ref, maps, roisize):
    """ROIs of every channel around the mapped emitter positions. images: per channel a [F, H_c, W_c] tensor (photons;
    channel crop coordinates), frame [N] (long), xy_ref [N, 2] reference-channel positions, maps: per channel a
    ChannelMap (or None = identity). The ROI of channel c is centred on the rounded mapped position and moved inside
    the image. Returns (rois [N, C, P, P], corners [N, C, 2] (x0, y0) channel px; MultiChannelPSF.fit(rois,
    corners=corners) of a model with these mappings gives reference positions)."""
    P = int(roisize)
    h = P // 2
    xy_ref = torch.as_tensor(xy_ref, dtype=torch.float64)
    fr = torch.as_tensor(frame, dtype=torch.long)
    rois, cors = [], []
    for c, im in enumerate(images):
        m = maps[c] if maps is not None else None
        xy = xy_ref if m is None else m.from_ref(xy_ref.to(m.device)).cpu()
        H, W = im.shape[-2:]
        x0 = (torch.round(xy[:, 0]).long() - h).clamp(0, W - P)
        y0 = (torch.round(xy[:, 1]).long() - h).clamp(0, H - P)
        ar = torch.arange(P)
        yy = (y0[:, None] + ar[None]).to(im.device)
        xx = (x0[:, None] + ar[None]).to(im.device)
        f = fr.to(im.device)
        rois.append(im[f[:, None, None], yy[:, :, None], xx[:, None, :]])
        cors.append(torch.stack([x0, y0], 1))
    return torch.stack(rois, 1), torch.stack(cors, 1).float()
