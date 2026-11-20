"""Table-indexed tensor-product spline PSFs. Coordinates are x/y in pixels and z in micrometres."""
import copy
import math
import torch
from torch_psf.optics import resolve_device
from torch_psf.psf import PSF
from torch_psf._dipole import M_to_m6, combine_tables, default_weights, unit_tensors
from torch_psf.integrate import PixelIntegrator, spectral_min_oversample


DEFAULT_SPLINE = ('catmull', 'pixel', 2)  # kernel, value, knots/px for newly built plain tables

# Sampling presets for an explicitly requested value or a phase table: kernel, S, dz, zpad, xy_limit, chunk.
_GRID_PROFILES = {
    'pixel': ('catmull', 2, .04, 2, 3.0, 8),
    'density': ('bspline', 4, .01, 6, 2.0, 4),
    'phase_pixel': ('catmull', 4, .01, 3, None, 128),
    'phase_density': ('bspline', 4, .01, 6, None, 128),
}


def spline_kind(kind, pixel_integration):
    if kind is None:
        return 'cubic' if pixel_integration == 'box' else 'integrated'
    if kind not in ('cubic', 'integrated'):
        raise ValueError("kind must be 'cubic' or 'integrated'")
    return kind


def geometry_matrix(device, tau: float = 0.5, dtype=torch.float32):
    """Catmull-Rom basis: per-axis weights w = [1, t, t^2, t^3] @ G."""
    return torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [-tau, 0.0, tau, 0.0],
        [2 * tau, tau - 3, 3 - 2 * tau, -tau],
        [-tau, 2 - tau, tau - 2, tau]
    ], device=device, dtype=dtype)


def _kernel(s, kind):
    """Cubic kernel K(s), support [-2, 2]: Catmull-Rom (interpolating, C1) or the cubic B-spline (C2)."""
    a = s.abs()
    if kind == 'catmull':
        k1 = 1.5 * a ** 3 - 2.5 * a ** 2 + 1
        k2 = -0.5 * a ** 3 + 2.5 * a ** 2 - 4 * a + 2
    else:
        k1 = 2 / 3 - a ** 2 + 0.5 * a ** 3
        k2 = (2 - a) ** 3 / 6
    return torch.where(a <= 1, k1, torch.where(a < 2, k2, torch.zeros_like(a)))


def _dkernel(s, kind):
    a = s.abs()
    if kind == 'catmull':
        k1 = 4.5 * a ** 2 - 5 * a
        k2 = -1.5 * a ** 2 + 5 * a - 4
    else:
        k1 = -2 * a + 1.5 * a ** 2
        k2 = -0.5 * (2 - a) ** 2
    return torch.sign(s) * torch.where(a <= 1, k1, torch.where(a < 2, k2, torch.zeros_like(a)))


def _qkernel(s, kind):
    """Antiderivative Q(s) = int_-2^s K: 0 for s <= -2, 1 for s >= 2 (piecewise quartic)."""
    a = s.abs().clamp(max=2.0)
    if kind == 'catmull':
        p1 = 0.375 * a ** 4 - (5 / 6) * a ** 3 + a
        P1 = 0.375 - 5 / 6 + 1
        F = lambda x: -0.125 * x ** 4 + (5 / 6) * x ** 3 - 2 * x ** 2 + 2 * x
        p2 = P1 + F(a) - F(torch.ones_like(a))
    else:
        p1 = 2 * a / 3 - a ** 3 / 3 + a ** 4 / 8
        P1 = 2 / 3 - 1 / 3 + 1 / 8
        p2 = P1 + 1 / 24 - (2 - a) ** 4 / 24
    return 0.5 + torch.sign(s) * torch.where(a <= 1, p1, p2)


def bspline_prefilter(samples, dims=(-3, -2, -1)):
    """Cubic B-spline coefficients that interpolate `samples` along the given dims (mirror boundary): c = M^-1 s with
    M = tridiag(1/6, 4/6, 1/6), solved with a dense inverse per axis (float64)."""
    c = samples.double()
    for d in dims:
        n = c.shape[d]
        M = torch.zeros(n, n, dtype=torch.float64, device=c.device)
        i = torch.arange(n, device=c.device)
        M[i, i] = 4 / 6
        M[i[1:], i[:-1]] = 1 / 6
        M[i[:-1], i[1:]] = 1 / 6
        M[0, 1] = M[-1, -2] = 2 / 6            # mirror: c[-1] = c[1]
        Mi = torch.linalg.inv(M)
        c = torch.movedim(torch.tensordot(Mi, torch.movedim(c, d, 0), dims=([1], [0])), 0, d)
    return c.to(samples.dtype).contiguous()          # contiguous: the knot-block reads of render are 1.5x slower otherwise


class SplinePSF(PSF):
    """One bank of coefficient volumes on a shared grid.

    knots [T,nz,nv,nu] are samples for Catmull-Rom and prefiltered coefficients
    for cubic B-splines. table_map indexes that bank along the named axes.
    Pixel tables interpolate measured pixel values. Density tables integrate
    the interpolant over each pixel. The separable and grouped-convolution
    backends share coordinate, weight and gather code.
    """
    BASIS = 'cubic_bspline'
    FILE_TYPES = ('spline_table', 'integrated_spline', 'spline')

    def __init__(self, roisize, z_range, oversample=2, dz_um=.04, zpad=2, xy_limit=3.0, device=None,
                 dtype=torch.float32, knots=None, knot_u0_px=None, knot_z0_um=None, pixel_um=None, *,
                 kernel='catmull', value='pixel', backend='sep', axes=None, table_map=None,
                 component_weights=None, angle=None):
        if kernel not in ('catmull', 'bspline') or value not in ('pixel', 'density') or backend not in ('sep', 'conv'):
            raise ValueError('unknown spline kernel, value or backend')
        self.kernel, self.value, self.backend = kernel, value, backend
        self.chunk_elems = 2 ** 24
        self.pixel_um = None if pixel_um is None else float(pixel_um)
        zlo, zhi = float(z_range[0]), float(z_range[1])
        S, hz, xmax = int(oversample), float(dz_um), float(xy_limit)
        self.roisize, self.S, self.hz = roisize, S, hz
        self.h = (roisize - 1) / 2
        self.dev, self.dtype = resolve_device(device), dtype
        self.zlo, self.zhi, self.zpad = zlo, zhi, zpad
        if knots is None:
            self.c = int(math.ceil(self.h + xmax)) + 2            # half width (px) of the stored PSF
            self.nz = int(round((zhi - zlo) / hz)) + 1 + 2 * zpad
        else:
            nz_, nv_, nu_ = knots.shape[-3:]
            assert nv_ == nu_ and nu_ % S == 0 and (nu_ // S) % 2 == 1, 'knots [nz, n, n], n / oversample odd'
            self.c = (nu_ // S - 1) // 2
            self.nz = nz_
        self.Pl = 2 * self.c + 1
        self.z0 = zlo - zpad * hz if knot_z0_um is None else float(knot_z0_um)
        self.nv = self.nu = self.Pl * S
        self.u0 = -(self.c * S + S - 1) / S if knot_u0_px is None else float(knot_u0_px)   # u (px) of knot 0
        shape = (self.nz, self.nv, self.nu)
        if knots is None:
            self.knots = torch.zeros((1, *shape), device=self.dev, dtype=dtype)
        else:
            bank = torch.as_tensor(knots, device=self.dev, dtype=dtype)
            self.knots = bank if bank.ndim == 4 else bank.reshape(-1, *shape)
        self.axes = tuple(axes) if axes is not None else ('component',)
        if table_map is None:
            table_map = torch.arange(len(self.knots))
        self.table_map = torch.as_tensor(table_map, device=self.dev, dtype=torch.long)
        if len(self.axes) != self.table_map.ndim:
            raise ValueError('table_map needs one dimension per named axis')
        if (len(set(self.axes)) != len(self.axes)
                or self.table_map.numel() == 0 or self.table_map.min() < 0 or self.table_map.max() >= len(self.knots)):
            raise ValueError('table_map must index the knot bank along the named axes')
        if 'dipole' in self.axes:
            if self.table_map.shape[self.axes.index('dipole')] != 6:
                raise ValueError('the dipole axis needs six volumes')
            if component_weights is None:
                component_weights = default_weights()
                if self.n_angles:
                    component_weights = component_weights.expand(self.n_angles, 6)
        self.component_weights = (None if component_weights is None else
                                  torch.as_tensor(component_weights, device=self.dev, dtype=dtype))
        self._ordered = torch.equal(self.table_map.flatten(), torch.arange(len(self.knots), device=self.dev))
        self.angle = self._angle(angle)
        self._roi(roisize, xmax)

    @property
    def knots(self):
        return self._bank

    @knots.setter
    def knots(self, tables):
        if tables.ndim == 3:
            tables = tables.unsqueeze(0)
        if tables.ndim != 4 or tables.shape[1:] != (self.nz, self.nv, self.nu):
            raise ValueError('knots must contain volumes on the spline grid')
        self._bank = tables

    def _roi(self, roisize, xy_limit):
        self.roisize = int(roisize)
        self.h = (self.roisize - 1) / 2
        S = self.S
        self.taps = S + 4 if self.value == 'density' else 4
        self.Lw = (self.roisize - 1) * S + self.taps
        self.t0 = (-self.h - self.u0) * S
        if self.value == 'density':
            self.t0 -= S / 2
            lim = (self.nu - self.Lw - self.t0) / S - 1e-4
        else:
            lim = (self.nu - 2 - self.t0 - (self.roisize - 1) * S) / S - 1e-4
        self.xmax = max(0., min(xy_limit, (self.t0 - 1) / S, lim))
        r = lambda n: torch.arange(n, device=self.dev)
        self.pat = (r(4)[:, None, None] * self.nv * self.nu + r(self.Lw)[None, :, None] * self.nu
                    + r(self.Lw)[None, None, :]).reshape(-1)
        self.G = geometry_matrix(self.dev, dtype=self.dtype)
        self.e = torch.arange(4, device=self.dev)

    @property
    def zs(self):
        """z (um) of the knot planes [nz] (float64)."""
        return self.z0 + self.hz * torch.arange(self.nz, device=self.dev, dtype=torch.float64)

    @property
    def z_range(self):
        return (self.zlo, self.zhi)

    @property
    def z_limits(self):
        """z interval (um) on which the spline is defined (the fitter clamps z to it)."""
        if self.value == 'density':
            return self.z_range
        return (self.z0 + self.hz, self.z0 + (self.nz - 2) * self.hz)

    @property
    def xy_limit(self):
        return self.xmax

    def __repr__(self):
        return (f'SplinePSF(roisize={self.roisize}, kernel={self.kernel!r}, value={self.value!r}, '
                f'axes={self.axes}, knots={tuple(self.knots.shape)})')

    @property
    def n_angles(self):
        return self.table_map.shape[self.axes.index('angle')] if 'angle' in self.axes else 0

    def _angle(self, angle):
        from torch_psf.vectorial import _angle_index
        if angle is not None and not self.n_angles:
            raise ValueError('this spline has no per-angle terms')
        return None if angle is None else _angle_index(angle, self.n_angles)

    def for_angle(self, angle):
        new = copy.copy(self)
        new.angle = self._angle(angle)
        return new

    def resized(self, roisize):
        new = copy.copy(self)
        limit = float('inf') if self.value == 'density' else self.xmax
        new._roi(roisize, limit)
        return new

    def to(self, device=None, dtype=None):
        self.knots = self.knots.to(device=device, dtype=dtype)
        self.dev, self.dtype = self.knots.device, self.knots.dtype
        self.table_map = self.table_map.to(self.dev)
        if self.component_weights is not None:
            self.component_weights = self.component_weights.to(self.knots)
        self._roi(self.roisize, self.xmax)
        return self

    def _axis(self, t, deriv):
        p = t[..., None] ** self.e
        if deriv:
            p = self.e * t[..., None] ** (self.e - 1).clamp_min(0)
        return p @ self.G

    def _lat_weights(self, fr, deriv):
        """[N, T] lateral weights (or their d/dt) for the fractional knot offset fr [N]."""
        if self.value == 'pixel':
            return self._z_weights(fr, deriv)
        m = torch.arange(self.taps, device=fr.device, dtype=fr.dtype)
        k = self.kernel
        s = fr[:, None] + 1 - m[None]
        S = self.S
        if deriv:
            return (_kernel(s + S, k) - _kernel(s, k)) / S
        return (_qkernel(s + S, k) - _qkernel(s, k)) / S

    def _weights(self, fr, deriv):
        """(z, y, x) weights [N, taps] (or their d/dt) for the fractional knot offsets fr [N, 3]. Catmull-Rom pixel
        tables evaluate the three axes in ONE matmul ([N, 3, 4] @ [4, 4]), as the earlier CubicSplinePSF did: cuBLAS
        picks its kernel by the batch shape, so per-axis products differ in the last bit for small N on CUDA."""
        if self.kernel == 'catmull' and self.value == 'pixel':
            w = self._axis(fr, deriv)
            return w[:, 0], w[:, 1], w[:, 2]
        return self._z_weights(fr[:, 0], deriv), self._lat_weights(fr[:, 1], deriv), self._lat_weights(fr[:, 2], deriv)

    def _z_weights(self, fr, deriv):
        if self.kernel == 'catmull' and self.value == 'pixel':
            return self._axis(fr, deriv)
        m = torch.arange(4, device=fr.device, dtype=fr.dtype)
        s = fr[:, None] + 1 - m[None]
        return _dkernel(s, self.kernel) if deriv else _kernel(s, self.kernel)

    def evaluate_tables(self, pos, indices, grad=False, knots=None):
        """Evaluate at broadcastable positions [...,3] and table indices [...]."""
        K = self.knots if knots is None else knots
        indices = torch.as_tensor(indices, device=self.dev)
        batch = torch.broadcast_shapes(pos.shape[:-1], indices.shape)
        shape = (*batch, self.roisize, self.roisize)
        # Kernel weights are computed once per distinct position (before broadcasting to the tables) and gathered:
        # the weight matmul then has the shape of a per-table evaluation, so results do not depend on how many
        # tables share a position (cuBLAS picks its kernel by the batch shape).
        base = pos
        for dim in range(pos.dim() - 1):          # an expanded (stride 0) axis repeats one position: keep one copy
            if pos.shape[dim] > 1 and pos.stride(dim) == 0:
                base = base.narrow(dim, 0, 1)
        src = base.reshape(-1, 3).to(self.dtype)
        rows = torch.arange(len(src), device=src.device).reshape(base.shape[:-1]).expand(batch).reshape(-1)
        fr_src = self._knot_coords(src)[0]
        w_src = self._weights(fr_src, False)
        d_src = self._weights(fr_src, True) if grad else None
        pos = pos.expand(*batch, 3).reshape(-1, 3)
        ids = indices.expand(batch).reshape(-1)
        chunk = max(1, self.chunk_elems // (4 * self.Lw ** 2))
        parts = []
        for p, i, r in zip(pos.split(chunk), ids.split(chunk), rows.split(chunk)):
            w = tuple(x[r] for x in w_src)
            d = None if d_src is None else tuple(x[r] for x in d_src)
            parts.append(self._evaluate_block(p, i, grad, K, w, d))
        if grad:
            return tuple(torch.cat(deriv).reshape(shape) for deriv in zip(*parts))
        return torch.cat(parts).reshape(shape)

    def _knot_coords(self, pos):
        """(fractional knot offsets [N, 3] (z, y, x), first knot index [N, 3]) of positions [N, 3]."""
        S = self.S
        pos = pos.to(self.dtype)
        tz = ((pos[:, 2] - self.z0) / self.hz).clamp(1, self.nz - 2 - 1e-4)
        ty = self.t0 - pos[:, 1].clamp(-self.xmax, self.xmax) * S
        tx = self.t0 - pos[:, 0].clamp(-self.xmax, self.xmax) * S
        t = torch.stack([tz, ty, tx], 1)
        ix = torch.floor(t)
        return t - ix, ix.long() - 1

    def _evaluate_block(self, pos, ids, grad, K, w=None, d=None):
        """w, d: precomputed (z, y, x) weights and derivative weights of these rows (evaluate_tables), else computed."""
        N, P, S, Lw, T = len(pos), self.roisize, self.S, self.Lw, self.taps
        if N == 0:
            empty = K.reshape(-1)[:0].reshape(0, P, P)
            return (empty,) * 4 if grad else empty
        fr, i0 = self._knot_coords(pos)
        base = ((ids * self.nz + i0[:, 0]) * self.nv + i0[:, 1]) * self.nu + i0[:, 2]
        index = (base[:, None] + self.pat).reshape(-1)
        sub = K.reshape(-1).index_select(0, index).reshape(1, N * 4, Lw, Lw)
        wz, wy, wx = self._weights(fr, False) if w is None else w
        if grad and d is None:
            d = self._weights(fr, True)
        if self.backend == 'sep':
            return self._render_sep(sub.view(N, 4, Lw, Lw), wz, wy, wx, d, grad)
        w = wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]
        if not grad:
            return torch.nn.functional.conv2d(sub, w, groups=N, stride=S).view(N, P, P)
        dz, dy, dx = d
        Wx = wz[:, :, None, None] * wy[:, None, :, None] * dx[:, None, None, :]
        Wy = wz[:, :, None, None] * dy[:, None, :, None] * wx[:, None, None, :]
        Wz = dz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]
        Wt = torch.stack([w, Wx, Wy, Wz], 1).view(-1, 4, T, T)
        r = torch.nn.functional.conv2d(sub, Wt, groups=N, stride=S).view(N, 4, P, P)
        return r[:, 0], -S * r[:, 1], -S * r[:, 2], r[:, 3] / self.hz

    def _render_sep(self, sub, wz, wy, wx, d, grad):
        """Separable evaluation (backend 'sep'): contract z, then x (strided windows of the knot block), then y; plain
        float32 arithmetic (no cuDNN algorithm choice, no TF32)."""
        S, T = self.S, self.taps
        A = torch.einsum('nz,nzvu->nvu', wz, sub)                          # [N, Lw, Lw]
        cx = lambda X, w: torch.einsum('nvjt,nt->nvj', X.unfold(2, T, S), w)   # [N, Lw, P]
        cy = lambda X, w: torch.einsum('nijt,nt->nij', X.unfold(1, T, S), w)      # [N, P, P]
        Ax = cx(A, wx)
        val = cy(Ax, wy)
        if not grad:
            return val
        dz, dy, dx = d
        Az = torch.einsum('nz,nzvu->nvu', dz, sub)
        gx = cy(cx(A, dx), wy)
        gy = cy(Ax, dy)
        gz = cy(cx(Az, wx), wy)
        return val, -S * gx, -S * gy, gz / self.hz

    def render_batched(self, pos, nb=4000):
        """render() (value only) in batches of nb emitters."""
        return torch.cat([self.render(pos[s:s + nb]) for s in range(0, len(pos), nb)])

    def to_spline(self, roisize=None, z_range=None, **grid):
        """A cubic spline is already a spline: returns self, or resized(roisize) for another ROI size (other
        arguments are not supported)."""
        if (z_range is not None and tuple(z_range) != self.z_range) or grid:
            raise ValueError('re-tabulating a spline on another grid is not supported; build it from the source model')
        return self if roisize in (None, self.roisize) else self.resized(roisize)

    @classmethod
    def from_zstack(cls, zstack, z0_um, dz_um, roisize, pixel_um=None, device=None, normalize=False):
        """Spline with oversample 1 whose knots are a measured (or simulated) z-stack [nz, H, W] of an emitter centred
        on the middle pixel (H = W odd), plane k at z = z0_um + k dz_um. The offset limit follows from the stack size:
        xy_limit = (W - 1) / 2 - 2 - (roisize - 1) / 2. normalize: scale each plane to sum 1. pixel_um: lateral pixel
        size (um) of the stack (required for fit / localize)."""
        z0, hz = z0_um, dz_um
        zs = torch.as_tensor(zstack, dtype=torch.float32)
        if device is not None:
            zs = zs.to(device)
        nz, H, W = zs.shape
        assert H == W and W % 2 == 1, 'the z-stack must be square with an odd size'
        h = (roisize - 1) // 2
        xmax = (W - 1) / 2 - 2 - h
        assert xmax > 0, f'the z-stack ({W} px) is too small for roisize {roisize}'
        if normalize:
            zs = zs / zs.sum((1, 2), keepdim=True)
        return cls(roisize, (z0, z0 + (nz - 1) * hz), oversample=1, dz_um=hz, zpad=0, xy_limit=xmax, device=zs.device,
                   knots=zs.contiguous(), pixel_um=pixel_um, backend='conv')

    @property
    def has_dipole(self):
        return 'dipole' in self.axes or 'response' in self.axes

    def _volumes(self, knots=None):
        """Logical table layout, including a separate default in older dipole files."""
        bank = self.knots if knots is None else knots
        if self._ordered:
            return bank.reshape(*self.table_map.shape, self.nz, self.nv, self.nu)
        return bank[self.table_map]

    def default_volume(self, knots=None):
        """The coefficient volume for the selected angle and the file's default dipole tensor. For the model's own
        knots (no gradient) the result is cached, keyed on the knot tensor's storage, version and the angle, so a
        per-angle spline does not average its coefficients on every angle=None render."""
        own = knots is None and not (self.knots.requires_grad and torch.is_grad_enabled())
        if own:
            key = (self.knots.data_ptr(), self.knots._version, self.angle,
                   None if self.component_weights is None else self.component_weights._version)
            cached = getattr(self, '_default_cache', None)
            if cached is not None and cached[0] == key:
                return cached[1]
            volume = self._default_volume(knots)
            self._default_cache = (key, volume)
            return volume
        return self._default_volume(knots)

    def _default_volume(self, knots=None):
        tables = self._volumes(knots)
        weights = self.component_weights
        if self.angle is not None:
            tables = tables[self.angle]
            if weights is not None and weights.ndim == 2:
                weights = weights[self.angle]
        if weights is not None:
            tables = combine_tables(weights, tables)
        elif self.has_dipole:
            tables = tables.select(-4, 0)  # independently calibrated default
        elif not self.n_angles:
            if self.table_map.numel() != 1:
                raise ValueError('supply per-axis weights for a bank of component tables')
            return tables.reshape(self.nz, self.nv, self.nu)
        return tables.mean(0) if self.n_angles and self.angle is None else tables

    def _basis_indices(self):
        ids = self._basis_indices_for_all_angles()
        return ids[self.angle] if self.n_angles and self.angle is not None else ids

    def _angle_bank(self, dipole, knots=None):
        """Indices or coefficient combinations for every angle, before interpolation."""
        if not self.n_angles:
            raise ValueError('this spline has no per-angle terms')
        if dipole:
            return self._basis_indices_for_all_angles(), self.knots if knots is None else knots
        if self.component_weights is None:
            ids = self.table_map[..., 0] if self.has_dipole else self.table_map
            return ids, self.knots if knots is None else knots
        bank = combine_tables(self.component_weights, self._volumes(knots))
        return torch.arange(len(bank), device=self.dev), bank

    def _basis_indices_for_all_angles(self):
        if not self.has_dipole:
            raise ValueError('build with components="dipole" (dipole_components=True)')
        return self.table_map[..., -6:]

    def _angle_images(self, pos, angle, grad, dipole, knots=None):
        from torch_psf.vectorial import angle_indices
        ids, bank = self._angle_bank(dipole, knots)
        a = angle_indices(angle, self.n_angles, self.dev)
        if a.shape != (len(pos),):
            raise ValueError('expected one angle per emitter')
        return self.evaluate_tables(pos[:, None] if dipole else pos, ids[a], grad, bank)

    def _dipole_basis(self, pos, grad=False, *, angle=None, knots=None):
        if angle is not None and torch.as_tensor(angle).ndim:
            return self._angle_images(pos, angle, grad, True, knots)
        if angle is not None:
            return self.for_angle(angle)._dipole_basis(pos, grad, knots=knots)
        ids = self._basis_indices()
        if ids.ndim == 2:
            # Legacy angle=None uses the mean coefficients for each dipole basis.
            # Taking the mean after interpolation changes small axial derivatives.
            volumes = self._volumes(knots)[..., -6:, :, :, :].mean(0)
            ids = torch.arange(6, device=self.dev)
            return self.evaluate_tables(pos[:, None], ids, grad, volumes)
        return self.evaluate_tables(pos[:, None], ids, grad, knots)

    def _render(self, pos, grad=False, *, dipM=None, angle=None):
        return self.render_tables(pos, grad, dipM=dipM, angle=angle)

    def basis(self, pos, grad=False, *, weights=None, **options):
        if weights is None:
            return super().basis(pos, grad, **options)
        images = self.evaluate_tables(pos[:, None], self.table_map.flatten(), grad)
        return tuple(v[:, None] for v in images) if grad else images[:, None]

    def weights(self, pos, *, weights=None, **options):
        if weights is None:
            return super().weights(pos, **options)
        if isinstance(weights, dict):
            if set(weights) != set(self.axes):
                raise ValueError(f'expected weights for {self.axes}')
            product = pos.new_ones((len(pos), 1), dtype=self.dtype)
            for axis, size in zip(self.axes, self.table_map.shape):
                w = torch.as_tensor(weights[axis], device=self.dev, dtype=self.dtype).reshape(-1, size)
                product = (product[:, :, None] * w[:, None, :]).flatten(1)
            weights = product
        w = torch.as_tensor(weights, device=self.dev, dtype=self.dtype).reshape(-1, self.table_map.numel())
        return w[:, None].expand(len(pos), 1, -1)

    def render_tables(self, pos, grad=False, knots=None, dipM=None, *, weights=None, angle=None):
        """Render patches and optionally their analytic x, y, z derivatives.

        weights can be [T] or [N,T] in logical table order, or a dict of
        per-axis weights [size] / [N,size]. Angles and dipM select the
        calibrated response for the standard single-channel file layouts.
        """
        if angle is not None and torch.as_tensor(angle).ndim:
            if weights is not None:
                raise ValueError('supply angle or table weights')
            if dipM is None:
                return self._angle_images(pos, angle, grad, False, knots)
            images = self._angle_images(pos, angle, grad, True, knots)
            w = M_to_m6(torch.as_tensor(dipM).to(self.knots)).reshape(-1, 6)
            return tuple(_weighted(v, w) for v in images) if grad else _weighted(images, w)
        if angle is not None:
            return self.for_angle(angle).render_tables(pos, grad, knots, dipM, weights=weights)
        if weights is not None:
            if dipM is not None:
                raise ValueError('supply weights or dipM')
            w = self.weights(pos, weights=weights)[:, 0]
            basis = self.evaluate_tables(pos[:, None], self.table_map.flatten(), grad, knots)
            def contract(v):
                return torch.einsum('nk,nkij->nij', w, v)
            return tuple(contract(v) for v in basis) if grad else contract(basis)
        if dipM is not None:
            M = torch.as_tensor(dipM, device=self.dev, dtype=self.dtype)
            if M.shape not in ((3, 3), (len(pos), 3, 3)):
                raise ValueError('dipM must have shape [3,3] or [N,3,3]')
            w = M_to_m6(M).reshape(-1, 6)
            basis = self._dipole_basis(pos, grad, knots=knots)
        else:
            if knots is not None and knots.ndim == 3:
                return self.evaluate_tables(pos, self.table_map.new_zeros(()), grad, knots)
            if self.table_map.numel() == 1:
                return self.evaluate_tables(pos, self.table_map.reshape(()), grad, knots)
            if self.n_angles and self.angle is not None and not self.has_dipole:
                return self.evaluate_tables(pos, self.table_map[self.angle], grad, knots)
            volume = self.default_volume(knots)
            if volume.numel() != self.nz * self.nv * self.nu:
                raise ValueError('supply weights for a bank of component tables')
            return self.evaluate_tables(pos, self.table_map.new_zeros(()), grad, volume)
        return tuple(_weighted(v, w) for v in basis) if grad else _weighted(basis, w)

    @property
    def type_name(self):
        if self.kernel == 'bspline' and self.value == 'density':
            return 'integrated_spline'
        if self.kernel == 'catmull' and self.value == 'pixel' and not self.n_angles:
            return 'spline'
        return 'spline_table'

    def format_version(self, source=None):
        if self.type_name == 'spline_table':
            version = (0, 5)
        elif self.has_dipole:
            version = (0, 4)
        elif self.value == 'density':
            version = (0, 3)
        else:
            version = (0, 2)
        return version if source is None else max(version, source.format_version())

    @classmethod
    def file_grid(cls, block):
        """Find and validate a named grid in a channel or a standalone model."""
        typ = block.get('type') or next((k for k in cls.FILE_TYPES if k in block), None)
        if typ not in cls.FILE_TYPES or typ not in block:
            raise ValueError('missing spline grid')
        g = block[typ]
        if typ == 'spline_table' and not ('basis' in g and g.get('value') in ('pixel', 'density')):
            raise ValueError('spline_table requires explicit basis / value')
        if typ == 'integrated_spline' and (g.get('basis'), g.get('value')) != (cls.BASIS, 'pixel_integral'):
            raise ValueError('integrated_spline: unknown basis / value')
        return typ, g

    @classmethod
    def grid_interpolant(cls, g):
        """Decode a stored grid without consulting the defaults for newly built tables."""
        if 'basis' not in g and 'value' not in g:
            return 'catmull', 'pixel'
        if (g.get('basis'), g.get('value')) == (cls.BASIS, 'pixel_integral'):
            return 'bspline', 'density'
        kernels = {'catmull_rom': 'catmull', cls.BASIS: 'bspline'}
        if g.get('basis') in kernels and g.get('value') in ('density', 'pixel'):
            return kernels[g['basis']], g['value']
        raise ValueError(f"unknown spline basis / value: {g.get('basis')!r}, {g.get('value')!r}")

    def to_dict(self):
        density = self.value == 'density'
        g = dict(basis=self.BASIS, value='pixel_integral') if density else {}
        if self.type_name == 'spline_table':
            g = dict(basis='catmull_rom' if self.kernel == 'catmull' else self.BASIS, value=self.value)
        g.update(roisize=self.roisize, z_range=list(self.z_range), oversample=self.S, dz_um=self.hz,
                 zpad=self.zpad, xy_limit=self.xmax, c=self.c, z0_um=self.z0, nz=self.nz, u0_px=self.u0,
                 pixel_um=self.pixel_um)
        if density or self.n_angles:
            g['n_angles'] = self.n_angles
        if self.has_dipole:
            g['dipole_components'] = True
        if self.backend != ('sep' if density else 'conv'):
            g['backend'] = self.backend
        return g

    def _file_dict(self, source=None):
        if self.axes not in (('component',), ('dipole',), ('response',), ('angle',),
                             ('angle', 'dipole'), ('angle', 'response')):
            raise ValueError('these axes have no single-channel file block')
        if self.axes == ('component',) and self.table_map.numel() != 1:
            raise ValueError('a scalar file block requires one coefficient volume')
        d, arrays = ({}, {}) if source is None else source._file_dict()
        g = self.to_dict()
        d[self.type_name] = g
        key = 'coefs' if self.value == 'density' else 'knots'
        suffix = '_angle' if self.n_angles else ''
        tables = self._volumes()
        if self.has_dipole:
            arrays['dipole_' + key + suffix] = tables[..., -6:, :, :, :].detach().cpu().numpy()
        if self.component_weights is not None:
            g['default_weights'] = self.component_weights.tolist()
        else:
            fixed = tables.select(-4, 0) if self.has_dipole or not self.n_angles else tables
            arrays[key + suffix] = fixed.detach().cpu().numpy()
        return d, arrays

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, angle=None, **kw):
        typ, g = cls.file_grid(d)
        kernel, value = ('catmull', 'pixel') if typ == 'spline' else cls.grid_interpolant(g)
        density = value == 'density'
        arrays = arrays or {}
        key = 'coefs' if density else 'knots'
        weights = g.get('default_weights')
        per_angle = arrays.get(key + '_angle')
        na = int(g.get('n_angles', len(per_angle) if per_angle is not None else 0))
        suffix = '_angle' if na else ''
        fixed = None
        if weights is None:
            if (key in arrays) == (key + '_angle' in arrays):
                raise ValueError('expected exactly one fixed coefficient array')
            fixed = torch.as_tensor(arrays[key + suffix])
        axes = ('angle',) if na else ()
        if g.get('dipole_components'):
            tables = torch.as_tensor(arrays['dipole_' + key + suffix])
            if tables.shape[-4] != 6 or tables.ndim != 4 + bool(na):
                raise ValueError('expected six dipole volumes per angle')
            if fixed is not None:
                tables = torch.cat((fixed.unsqueeze(-4), tables), dim=-4)
            elif tuple(torch.as_tensor(weights).shape) != tuple(tables.shape[:-3]):
                raise ValueError('expected six default weights per angle')
            axes += ('dipole' if fixed is None else 'response',)
        else:
            tables = fixed
            if not na:
                tables = tables.unsqueeze(0)
                axes = ('component',)
        if (na and len(tables) != na) or tables.ndim != len(axes) + 3:
            raise ValueError('coefficient arrays do not match the stored axes')
        ids = torch.arange(math.prod(tables.shape[:-3])).reshape(tables.shape[:-3])
        model = cls(g['roisize'], g['z_range'], g['oversample'], g['dz_um'], g['zpad'], g['xy_limit'],
                    device=device, knots=tables, knot_u0_px=g.get('u0_px'), knot_z0_um=g.get('z0_um'),
                    pixel_um=g.get('pixel_um') or (d.get('optics') or {}).get('pixel_um'),
                    kernel=kernel, value=value,
                    backend=g.get('backend', 'sep' if density else 'conv'), axes=axes, table_map=ids, angle=angle,
                    component_weights=weights)
        if model.nz != g.get('nz', model.nz) or model.c != g.get('c', model.c):
            raise ValueError('coefficient arrays do not match the stored grid')
        return model if roisize in (None, model.roisize) else model.resized(roisize)

    @staticmethod
    def load_source(path, roisize=None, device=None, angle=None):
        d, arrays, _ = PSF.read_file(path)
        if 'zernike_waves' in d:
            return PSF._registry['vectorial']._from_file(d, arrays, roisize=roisize or d.get('roisize', 15),
                                                         device=device, angle=angle)

    def sample(self, model, pos_fn=None, *, params=None, chunk=4, focus=None, blur=None, dipM=None,
               differentiable=False):
        """Sample one response on this grid. differentiable=True retains the optical and prefilter graph."""
        render = self._sampler(model, pos_fn, params, focus, blur, dipM)
        with torch.set_grad_enabled(differentiable and torch.is_grad_enabled()):
            volumes = _sample_grid(self, render, chunk * self.S ** 2)
            return bspline_prefilter(volumes) if self.kernel == 'bspline' else volumes

    def _sampler(self, model, pos_fn, params, focus, blur, dipM):
        if model.roisize != self.Pl:
            raise ValueError(f'sampling the grid needs a model resized to {self.Pl}')
        if hasattr(model, '_grid_sampler'):
            return model._grid_sampler(self, pos_fn, params, focus, blur, dipM)
        if not hasattr(model, '_image'):
            if self.value == 'density' or pos_fn is not None or dipM is not None:
                raise ValueError('this model supports tabulating pixel values only')
            return model.render
        pixels = None
        if self.value == 'density':
            sampling = spectral_min_oversample(model.band_limit())
            if params is None:
                pixels = PixelIntegrator(model, model.blur if blur is None else blur, mode='point', oversample=sampling)
            else:
                # functional_call enters forward(), so configure the point integrator on a private optical copy.
                model = model.resampled(model.roisize, oversample=sampling, pixel_integration='point')

        def render(p):
            f = model.focus_um if focus is None else focus
            if pos_fn is not None:
                z, f = pos_fn(p[:, 2])
                p = torch.cat((p[:, :2], z[:, None]), dim=1)
            if pixels is not None:
                return model._image(p, f, pixels, dipM=dipM)
            if params is not None:
                return torch.func.functional_call(model, params, (p, f), dict(dipM=dipM, blur=blur))
            return model(p, f, dipM=dipM, blur=blur)
        return render

    def vjp_vectorial(self, vectorial, pos_fn, grad_knots, names=('zcoef',), chunk=4):
        """Pull one table's coefficient adjoint back to optical parameters, one z chunk at a time."""
        base = dict(vectorial.named_parameters())
        base.update(vectorial.named_buffers())
        prm = {k: v.detach().requires_grad_(k in names) for k, v in base.items()}
        render = self._sampler(vectorial, pos_fn, prm, None, None, None)
        adjoint = grad_knots.reshape(self.nz, self.nv, self.nu)
        if self.kernel == 'bspline':
            # Mirror filtering couples z planes. Apply its transpose to the entire adjoint before chunking.
            dummy = torch.zeros_like(adjoint, requires_grad=True)
            adjoint, = torch.autograd.grad(bspline_prefilter(dummy), dummy, adjoint)
        result = {k: torch.zeros_like(prm[k]) for k in names}
        for z in range(0, self.nz, chunk):
            planes = _sample_grid(self, render, chunk * self.S ** 2, self.zs[z:z + chunk])
            grads = torch.autograd.grad(planes, [prm[k] for k in names], adjoint[z:z + chunk])
            for k, g in zip(names, grads):
                result[k] += g
        return result


class IndexedSplines:
    """One coefficient bank for a channel or tile axis. Each member keeps its own angle and default response."""

    @staticmethod
    def compatible(models):
        if not models or not all(isinstance(m, SplinePSF) for m in models):
            return False
        keys = ('roisize', 'S', 'hz', 'z0', 'u0', 'nz', 'nv', 'xmax', 'kernel', 'value', 'dtype', 'dev')
        return all(all(getattr(m, k) == getattr(models[0], k) for k in keys) for m in models[1:])

    def __init__(self, models, axis):
        self.table = copy.copy(models[0])
        self.table.knots = torch.cat([m.knots for m in models])
        self.members = []
        self.slices = []
        start = 0
        for m in models:
            view = copy.copy(m)
            self.slices.append(slice(start, start + len(m.knots)))
            view.knots = self.table.knots[self.slices[-1]]
            self.members.append(view)
            start += len(m.knots)
        self.starts = self.table.table_map.new_tensor([0] + [len(m.knots) for m in models]).cumsum(0)[:-1]
        if all(m.axes == models[0].axes and m.table_map.shape == models[0].table_map.shape for m in models):
            self.table.axes = (axis, *models[0].axes)
            self.table.table_map = torch.stack([start + m.table_map for start, m in zip(self.starts, models)])
        else:
            self.table.axes = ('component',)
            self.table.table_map = torch.arange(len(self.table.knots), device=self.table.dev)
        self.table._ordered = torch.equal(self.table.table_map.flatten(),
                                          torch.arange(len(self.table.knots), device=self.table.dev))

    def evaluate(self, pos, member, grad=False, *, dipM=None, angle=None, basis=False):
        """pos [N,L,3] and member indices broadcast over [N,L]."""
        if angle is not None and torch.as_tensor(angle).ndim:
            from torch_psf.vectorial import angle_indices
            dipole = basis or dipM is not None
            volumes = [self.table.knots[sl] for sl in self.slices]
            selections, banks = zip(*(m._angle_bank(dipole, bank) for m, bank in zip(self.members, volumes)))
            a = angle_indices(angle, min(m.n_angles for m in self.members), self.table.dev)
            if a.shape != (len(pos),):
                raise ValueError('expected one angle per emitter')
            # Members may have different numbers of angles or independent calibrated defaults.
            direct = all(bank is v for bank, v in zip(banks, volumes))
            bank = self.table.knots if direct else torch.cat(banks)
            lengths = [len(b) for b in banks]
            starts = self.starts if direct else self.starts.new_tensor([0, *lengths[:-1]]).cumsum(0)
            rows = [ids[a] + start for ids, start in zip(selections, starts)]
            mapping = torch.stack(rows, 1)
            n = torch.arange(len(pos), device=a.device)[:, None]
            ids = mapping[n, member]
            images = self.table.evaluate_tables(pos.unsqueeze(-2) if dipole else pos, ids, grad, bank)
            if dipM is None:
                return images
            return self._contract(images, pos, dipM, grad)

        members = self.members if angle is None else [m.for_angle(angle) for m in self.members]
        if not basis and dipM is None:
            if all(m.table_map.numel() == 1 for m in members):
                ids = self.starts + torch.stack([m.table_map.reshape(()) for m in members])
                return self.table.evaluate_tables(pos, ids[member], grad)
            bank = torch.stack([m.default_volume(self.table.knots[sl]) for m, sl in zip(members, self.slices)])
            return self.table.evaluate_tables(pos, member, grad, bank)

        ids = [m._basis_indices() for m in members]
        if all(i.ndim == 1 for i in ids):
            bank = self.table.knots
            indices = torch.stack(ids) + self.starts[:, None]
        else:
            bank = torch.cat([self.table.knots[sl][i].mean(0) if i.ndim == 2 else self.table.knots[sl][i]
                              for sl, i in zip(self.slices, ids)])
            indices = torch.arange(len(bank), device=bank.device).reshape(-1, 6)
        images = self.table.evaluate_tables(pos.unsqueeze(-2), indices[member], grad, bank)
        if basis:
            return images
        return self._contract(images, pos, dipM, grad)

    def _contract(self, images, pos, dipM, grad):
        M = torch.as_tensor(dipM).to(self.table.knots)
        if M.shape not in ((3, 3), (len(pos), 3, 3)):
            raise ValueError('dipM must have shape [3,3] or [N,3,3]')
        w = M_to_m6(M).reshape(-1, 1, 6, 1, 1)
        def contract(v):
            return (v * w).sum(2)
        return tuple(contract(v) for v in images) if grad else contract(images)


for _type in SplinePSF.FILE_TYPES:
    PSF._registry[_type] = SplinePSF


def _weighted(images, weights):
    # Keep the component products rounded before summing (legacy dipole convention).
    return torch.einsum('nk,nkij->nkij', weights, images).sum(1)


def _sample_grid(grid, render, batch, zs=None):
    """Interleave subpixel-shifted images into coefficient volumes [...,nz,nv,nu]."""
    S, P = grid.S, grid.Pl
    shifts = torch.arange(S, device=grid.dev, dtype=torch.float64) / S
    z, y, x = torch.meshgrid(grid.zs if zs is None else zs, shifts, shifts, indexing='ij')
    pos = torch.stack((x, y, z), -1).reshape(-1, 3).float()
    images = torch.cat([render(p) for p in pos.split(batch)])
    axes = images.shape[1:-2]
    images = images.reshape(-1, S, S, *axes, P, P).flip((1, 2))
    order = (*range(3, 3 + len(axes)), 0, images.ndim - 2, 1, images.ndim - 1, 2)
    return images.permute(order).reshape(*axes, -1, P * S, P * S)


def to_spline(model, z_range=None, *, roisize=None, kernel=None, value=None, components=None, **grid):
    """Build spline tables for an optical model or per-angle models.

    kernel='catmull'|'bspline' and value='pixel'|'density' select the
    interpolant. Grid arguments are oversample, dz_um, zpad and xy_limit.
    components='dipole' adds the six dipole responses. Compositions keep
    their channel geometry and existing spline tables can be resized.
    """
    grid.update({k: v for k, v in dict(kernel=kernel, value=value, components=components).items() if v is not None})
    if isinstance(model, (list, tuple)) or getattr(type(model), 'to_spline', PSF.to_spline) is PSF.to_spline:
        return _tabulate(model, z_range, roisize=roisize, **grid)
    return model.to_spline(roisize, z_range, **grid)


@torch.no_grad()
def _tabulate(model, z_range=None, *, roisize=None, kernel=None, value=None, components=None, oversample=None,
              dz_um=None, zpad=None, xy_limit=None, chunk=None, backend='sep', pos_fn=None, kind=None,
              dipole_components=False):
    """Tabulate an optical PSF, or a sequence of PSFs for the pattern angles.

    Pass kernel, value and the grid explicitly for calibration builds.
    components='dipole' stores six responses per angle or phase/channel.
    chunk counts z planes for single-channel models and positions for 4Pi models.
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    if not models:
        raise ValueError('provide at least one source model')
    first = models[0]
    phase = hasattr(first, '_channel_images')
    legacy = kernel is None and value is None
    if phase or kind is not None:
        default = spline_kind(kind, first.pixel_integration)
        if value is None:
            value = 'density' if default == 'integrated' else 'pixel'
    elif value is None:
        default_kernel, value, default_sampling = DEFAULT_SPLINE
        kernel = default_kernel if kernel is None else kernel
        if oversample is None:
            oversample = default_sampling
    profile = ('phase_' if phase else '') + value
    if profile not in _GRID_PROFILES:
        raise ValueError('value must be pixel or density')
    k, S, dz, pad, xl, batch = _GRID_PROFILES[profile]
    kernel = k if kernel is None else kernel
    S = S if oversample is None else oversample
    dz = dz if dz_um is None else dz_um
    pad = (3 if value == 'density' and kernel == 'catmull' else pad) if zpad is None else zpad
    batch = batch if chunk is None else chunk
    P = roisize or first.roisize
    xl = (P / 2 + 1 if xl is None else xl) if xy_limit is None else xy_limit
    zr = first.z_range if z_range is None else z_range
    if zr is None:
        raise ValueError('pass z_range to tabulate this model')
    if components not in (None, 'dipole', 'phase') or (components == 'phase' and not phase):
        raise ValueError('unknown spline components')
    dipole = dipole_components or components == 'dipole'
    backend = backend or ('conv' if legacy and value == 'pixel' else 'sep')
    grid = SplinePSF(P, zr, S, dz, pad, xl + int(value == 'density'), device=first.dev, pixel_um=first.pixel_um,
                     kernel=kernel, value=value, backend=backend, dtype=getattr(first, 'dtype', torch.float32))
    grid.xmax = min(xl, grid.xmax)
    if phase:
        if len(models) != 1 or pos_fn is not None:
            raise ValueError('4Pi tables use one optical source with its configured zmode')
        from torch_psf.fourpi import FourPiSplinePSF
        big = first._sampling_model(grid.Pl, value == 'density')
        def render(p):
            return torch.stack([big._channel_images(p, phi=p.new_full((len(p),), ph), basis=dipole)
                                for ph in (0., math.pi, .5 * math.pi)], 1)
        g0, gpi, g90 = _sample_grid(grid, render, batch)
        inc = .5 * (g0 + gpi)
        tables = torch.stack((inc, .5 * (g0 - gpi), inc - g90))
        if kernel == 'bspline':
            tables = bspline_prefilter(tables)
        return FourPiSplinePSF(P, zr, tables, S, dz, pad, xl, first.tau.detach(), first.dev, first.pixel_um,
                               first.state(), integrated_grid=dict(u0_px=grid.u0, z0_um=grid.z0) if value == 'density' else None,
                               default_dipM=first.dipM if dipole else None, kernel=kernel, backend=backend)
    tables, weights = [], []
    for m in models:
        big = m if m.roisize == grid.Pl else m.resized(grid.Pl)
        if dipole:
            tables.append(torch.stack([grid.sample(big, pos_fn, dipM=M, chunk=batch) for M in unit_tensors(m.dev)]))
            weights.append(default_weights(m.dipM, m.dipw))
        else:
            tables.append(grid.sample(big, pos_fn, chunk=batch))
    multi = len(models) > 1
    bank = torch.stack(tables) if multi else tables[0]
    grid.knots = bank.reshape(-1, grid.nz, grid.nv, grid.nu)
    grid.axes = (('angle',) if multi else ()) + (('dipole',) if dipole else ())
    grid.axes = grid.axes or ('component',)
    grid.table_map = torch.arange(len(grid.knots), device=grid.dev).reshape(bank.shape[:-3] or (1,))
    if dipole:
        grid.component_weights = (torch.stack(weights) if multi else weights[0]).to(grid.knots)
    return grid
