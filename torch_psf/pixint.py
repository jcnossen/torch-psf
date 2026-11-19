"""Pixel spline evaluation and construction from the optical model in torch_psf.vectorial.

PixelSplinePSF: a tensor-product cubic spline (Catmull-Rom or interpolating cubic B-spline) in u, v, z on the
   knot grid of CubicSplinePSF (spacing 1 / S px, hz um). Two uses:
   - integrate=True: the knots are point samples of the continuous PSF density f(u, v, z); the pixel value is the exact
     integral of the spline over the pixel. Separable: int_pixel f du dv = (1 / S^2) sum_kl c_kl Wk(x) Wl(y), with
     Wk the box-integrated basis function Q(t_L + S - k) - Q(t_L - k) (Q = antiderivative of the cubic kernel, a
     piecewise quartic; t_L = knot coordinate of the pixel's left edge). All pixels of a patch are S knots apart, so a
     patch is one stride-S grouped conv2d with an (S + 4) x (S + 4) x 4 kernel (instead of 4 x 4 x 4): the same
     evaluation structure as CubicSplinePSF. d/dx, d/dy: dW/dt = K(t_L + S - k) - K(t_L - k) (the kernel itself).
   - integrate=False: the knots are pixel-integrated values at sub-pixel shifts, interpolated pointwise (= the
     current CubicSplinePSF for kernel 'catmull'; 'bspline' adds the B-spline prefilter).
"""
import torch
from torch_psf.optics import resolve_device
from torch_psf.psf import PSF
from torch_psf._dipole import DipoleSpline, combine_tables, default_weights, unit_tensors
from torch_psf.spline import CubicSplinePSF
from torch_psf.integrate import PixelIntegrator, spectral_min_oversample
from torch_psf.vectorial import _angle_index


# ============================================================================================ cubic kernels
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


def bspline_prefilter(samples, dims=(0, 1, 2)):
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


# ============================================================================================ spline
class PixelSplinePSF(PSF):
    """Cubic tensor-product spline PSF on the knot grid of a CubicSplinePSF `geo` (same knot layout: spacing 1 / S px,
    origin geo.u0, z planes geo.zs). kernel 'catmull' | 'bspline' (samples are prefiltered to B-spline coefficients).
    integrate=True: samples = point density f (per px^2) at the knots, pixel value = exact integral of the spline over
    the pixel; integrate=False: samples = pixel values at sub-pixel shifts, interpolated pointwise.
    backend 'sep': separable einsum contraction (z, x, y); 'conv': one grouped conv2d as CubicSplinePSF (cuDNN; uses
    TF32 on Ampere GPUs unless torch.backends.cudnn.allow_tf32 = False, ~2e-4 of the peak)."""

    def __init__(self, geo, samples, kernel='catmull', integrate=True, backend='sep', coefs=None):
        """coefs: the spline coefficients themselves instead of samples (kernel 'bspline': already prefiltered; samples
        is then None and not used)"""
        self.geo, self.kernel, self.integrate, self.backend = geo, kernel, integrate, backend
        g = geo
        self.roisize, self.S, self.hz, self.nz, self.nv, self.nu = g.roisize, g.S, g.hz, g.nz, g.nv, g.nu
        self.dev, self.dtype, self.pixel_um = g.dev, g.dtype, g.pixel_um
        self.z_range, self.z0 = g.z_range, g.z0
        if coefs is not None:
            self.samples, self.knots = None, coefs.to(self.dev, self.dtype)
        else:
            self.samples = samples.to(self.dev, self.dtype)
            self.knots = bspline_prefilter(self.samples) if kernel == 'bspline' else self.samples
        P, S, h = self.roisize, self.S, g.h
        self.T = S + 4 if integrate else 4                     # taps per lateral axis
        self.Lw = (P - 1) * S + self.T
        # knot coordinate of the first tap origin for x = 0: pixel centre (point) or left pixel edge (integrate)
        self.t0 = (-h - g.u0) * S - (S / 2 if integrate else 0.0)
        # first tap floor(t) - 1 >= 0 and last tap floor(t) - 2 + Lw <= nu - 1 for all |x| <= xmax
        lim = min((self.t0 - 1) / S, (self.nu - self.Lw - self.t0) / S - 1e-4)
        self.xmax = max(min(lim, g.xmax), 0.0)
        r = lambda n: torch.arange(n, device=self.dev)
        self.pat = (r(4)[:, None, None] * self.nv * self.nu + r(self.Lw)[None, :, None] * self.nu
                    + r(self.Lw)[None, None, :]).reshape(-1)

    @property
    def z_limits(self):
        return self.geo.z_limits

    @property
    def xy_limit(self):
        return self.xmax

    def _lat_weights(self, fr, deriv):
        """[N, T] lateral weights (or their d/dt) for the fractional knot offset fr [N]."""
        m = torch.arange(self.T, device=fr.device, dtype=fr.dtype)
        k = self.kernel
        if self.integrate:
            s = fr[:, None] + 1 - m[None]                      # t_L - k, k = floor(t_L) - 1 + m
            S = self.S
            if deriv:
                return (_kernel(s + S, k) - _kernel(s, k)) / S
            return (_qkernel(s + S, k) - _qkernel(s, k)) / S
        s = fr[:, None] + 1 - m[None]
        return _dkernel(s, k) if deriv else _kernel(s, k)

    def _z_weights(self, fr, deriv):
        m = torch.arange(4, device=fr.device, dtype=fr.dtype)
        s = fr[:, None] + 1 - m[None]
        return _dkernel(s, self.kernel) if deriv else _kernel(s, self.kernel)

    def render(self, pos, grad=False, knots=None):
        K = self.knots if knots is None else knots
        N, P, S, Lw, T = pos.shape[0], self.roisize, self.S, self.Lw, self.T
        pos = pos.to(self.dtype)
        tz = ((pos[:, 2] - self.z0) / self.hz).clamp(1, self.nz - 2 - 1e-4)
        ty = self.t0 - pos[:, 1].clamp(-self.xmax, self.xmax) * S
        tx = self.t0 - pos[:, 0].clamp(-self.xmax, self.xmax) * S
        t = torch.stack([tz, ty, tx], 1)
        ix = torch.floor(t)
        fr = t - ix
        i0 = ix.long() - 1
        if K.requires_grad and torch.is_grad_enabled():
            # index_select: its backward is a plain scatter-add onto the knots
            base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
            sub = K.reshape(-1).index_select(0, (base[:, None] + self.pat[None]).reshape(-1)).reshape(1, N * 4, Lw, Lw)
        else:
            # the same blocks read through a strided window view of the knots (no index tensor: ~2.3x faster at P = 41)
            V = K.unfold(0, 4, 1).unfold(1, Lw, 1).unfold(2, Lw, 1)            # [nz - 3, nv - Lw + 1, nu - Lw + 1, 4, Lw, Lw]
            sub = V[i0[:, 0], i0[:, 1], i0[:, 2]].reshape(1, N * 4, Lw, Lw)
        wz, wy, wx = self._z_weights(fr[:, 0], False), self._lat_weights(fr[:, 1], False), self._lat_weights(fr[:, 2], False)
        if self.backend == 'sep':
            return self._render_sep(sub.view(N, 4, Lw, Lw), wz, wy, wx, fr, grad)
        w = wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]
        if not grad:
            return torch.nn.functional.conv2d(sub, w, groups=N, stride=S).view(N, P, P)
        dz, dy, dx = self._z_weights(fr[:, 0], True), self._lat_weights(fr[:, 1], True), self._lat_weights(fr[:, 2], True)
        Wx = wz[:, :, None, None] * wy[:, None, :, None] * dx[:, None, None, :]
        Wy = wz[:, :, None, None] * dy[:, None, :, None] * wx[:, None, None, :]
        Wz = dz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]
        Wt = torch.stack([w, Wx, Wy, Wz], 1).view(-1, 4, T, T)
        r = torch.nn.functional.conv2d(sub, Wt, groups=N, stride=S).view(N, 4, P, P)
        return r[:, 0], -S * r[:, 1], -S * r[:, 2], r[:, 3] / self.hz

    def _render_sep(self, sub, wz, wy, wx, fr, grad):
        """Separable evaluation (backend 'sep'): contract z, then x (strided windows of the knot block), then y; plain
        float32 arithmetic (no cuDNN algorithm choice, no TF32)."""
        S, T = self.S, self.T
        A = torch.einsum('nz,nzvu->nvu', wz, sub)                          # [N, Lw, Lw]
        cx = lambda X, w: torch.einsum('nvjt,nt->nvj', X.unfold(2, T, S), w)   # [N, Lw, P]
        cy = lambda X, w: torch.einsum('nijt,nt->nij', X.unfold(1, T, S), w)      # [N, P, P]
        Ax = cx(A, wx)
        val = cy(Ax, wy)
        if not grad:
            return val
        dz, dy, dx = self._z_weights(fr[:, 0], True), self._lat_weights(fr[:, 1], True), self._lat_weights(fr[:, 2], True)
        Az = torch.einsum('nz,nzvu->nvu', dz, sub)
        gx = cy(cx(A, dx), wy)
        gy = cy(Ax, dy)
        gz = cy(cx(Az, wx), wy)
        return val, -S * gx, -S * gy, gz / self.hz


def build_pixel_spline(vec, roisize, z_range, S, dz_um=0.01, zpad=None, xy_limit=None, kernel='catmull',
                       integrate=True, sample_model=None, chunk=4, backend='sep'):
    """PixelSplinePSF of the VectorialPSF `vec`: integrate=True samples the point density (VectorialPSF 'point');
    integrate=False samples pixel values of `sample_model(vec_resized)` (default VectorialPSF 'spectral') at the
    S x S sub-pixel shifts. The knot grid is that of CubicSplinePSF(roisize, z_range, S, dz_um, zpad, xy_limit).
    zpad default: 3 planes (catmull), 6 (bspline: the mirror boundary of the prefilter perturbs the first ~4 planes)."""
    zpad = (6 if kernel == 'bspline' else 3) if zpad is None else zpad
    xy = roisize / 2 + 1 if xy_limit is None else xy_limit
    geo = CubicSplinePSF(roisize, z_range, oversample=S, dz_um=dz_um, zpad=zpad, xy_limit=xy + 1.0, device=vec.dev,
                         pixel_um=vec.pixel_um)
    if not integrate and sample_model is not None:
        model = sample_model(vec.resized(geo.Pl))
    else:
        model = vec.resized(geo.Pl, oversample=spectral_min_oversample(vec.band_limit()),
                            pixel_integration='point' if integrate else 'spectral')
    samples = geo.knots_from_model(model, chunk=chunk)
    spl = PixelSplinePSF(geo, samples, kernel=kernel, integrate=integrate, backend=backend)
    spl.xmax = min(spl.xmax, xy)
    return spl


# ============================================================================================ the file model
class IntegratedSplinePSF(DipoleSpline, PixelSplinePSF):
    """The integrated cubic B-spline PSF as a model of its own (psf.json type 'integrated_spline', docs/PSF_FORMAT.md):
    the coefficients c [nz, n, n] of a tensor-product cubic B-spline f(u, v, z) that interpolates point samples of the
    continuous PSF density (per px^2) on the knot grid (spacing 1 / S px laterally, dz_um in z; the grid layout of
    CubicSplinePSF: knot (k, i, j) at z = z0_um + k dz_um, v = u0_px + i / S, u = u0_px + j / S, u = pixel centre
    offset from the emitter); a pixel value is the exact integral of f over the pixel (PixelSplinePSF integrate=True,
    kernel 'bspline', backend 'sep': plain float32 arithmetic, the same pixel values on every device).

    Per-pattern-angle models (SIMCODE photoselection; VectorialPSF psf_dipole_M_angle / psf_blur_angle_px2):
    coefs_angle [A, nz, n, n], one coefficient set per angle; angle=None (the default) is the mean of the angle
    coefficients, i.e. the mean of the angle images (the spline is linear in its coefficients). render uses the
    coefficients of `angle`; for_angle(a) is the same model (shared tensors) for another angle.

    roisize: the ROI it renders; the coefficients tabulate the PSF independently of it (resized(roisize) shares them).
    xy_limit: the offset limit (px; the largest |x|, |y| the coefficients support for this ROI, or less when given).
    z_range: the heights (um) the spline is accurate on (zpad planes outside it absorb the boundary of the prefilter;
    z is clamped to the knot range beyond, and z_limits = z_range for the fitter). Build it with from_vectorial."""
    type_name = 'integrated_spline'
    BASIS = 'cubic_bspline'

    def __init__(self, roisize, coefs, oversample, dz_um, z_range, zpad, knot_u0_px, knot_z0_um, pixel_um=None,
                 coefs_angle=None, angle=None, xy_limit=None, device=None, chunk_elems=2 ** 26,
                 dipole_coefs=None, dipole_coefs_angle=None, dtype=torch.float32):
        dev = resolve_device(device)
        ca = None if coefs_angle is None else torch.as_tensor(coefs_angle).to(dev, dtype)
        if coefs is None:
            if ca is None:
                raise ValueError('IntegratedSplinePSF: coefs or coefs_angle needed')
            coefs = ca.mean(0)
        coefs = torch.as_tensor(coefs).to(dev, dtype).contiguous()
        if ca is not None:
            ca = ca.contiguous()
        if ca is not None and tuple(ca.shape[1:]) != tuple(coefs.shape):
            raise ValueError(f'coefs_angle {tuple(ca.shape)} does not match coefs {tuple(coefs.shape)}')
        if angle is not None and ca is None:
            angle = None                    # as VectorialPSF: a model without angle terms is the same for every angle
        if angle is not None:
            angle = _angle_index(angle, len(ca))      # refuses fractional / bool / out-of-range angles
        self._fixed_coefs, self._fixed_coefs_angle, self.angle = coefs, ca, angle
        self.zpad, self.chunk_elems = int(zpad), int(chunk_elems)
        active = coefs if angle is None else ca[int(angle)]
        geo = CubicSplinePSF(roisize, z_range, oversample=oversample, dz_um=dz_um, zpad=zpad, xy_limit=1e9, device=dev,
                             knots=active, knot_u0_px=knot_u0_px, knot_z0_um=knot_z0_um, pixel_um=pixel_um, dtype=dtype)
        super().__init__(geo, None, kernel='bspline', integrate=True, backend='sep', coefs=active)
        self.dipole_coefs_angle = None
        if dipole_coefs_angle is not None:
            tables = torch.as_tensor(dipole_coefs_angle, device=dev, dtype=self.dtype)
            if ca is None or tables.shape != (len(ca), 6, *active.shape):
                raise ValueError('dipole_coefs_angle must contain six coefficient tables per angle')
            self.dipole_coefs_angle = tables
            dipole_coefs = None if angle is None else tables[angle]
        self._set_dipole_knots(dipole_coefs)
        if xy_limit is not None:
            self.xmax = min(self.xmax, float(xy_limit))

    # ------------------------------------------------------------------ construction
    @classmethod
    def _from_components(cls, tables, weights, angle=None, **geometry):
        """One or A sets of six density volumes. Their first volume also supplies the grid geometry."""
        tables = torch.as_tensor(tables, device=geometry['device'], dtype=geometry.get('dtype', torch.float32))
        weights = torch.as_tensor(weights, device=tables.device, dtype=tables.dtype)
        if tables.ndim not in (4, 5) or tables.shape[-4] != 6 or weights.shape != tables.shape[:-3]:
            raise ValueError('expected six dipole coefficients and weights per angle')
        per_angle = tables.ndim == 5
        angle = None if angle is None or not per_angle else _angle_index(angle, len(tables))
        first = tables[0, 0] if per_angle else tables[0]
        model = cls(coefs=first, **geometry)
        model.angle = angle
        model._fixed_coefs = model._knots = None
        model.dipole_coefs_angle = tables if per_angle else None
        model._dipole_knots = None if per_angle else tables
        model.component_weights = weights
        return model

    @property
    def dipole_knots(self):
        if self._dipole_knots is None and self.dipole_coefs_angle is not None:
            return self.dipole_coefs_angle.mean(0) if self.angle is None else self.dipole_coefs_angle[self.angle]
        return self._dipole_knots

    @dipole_knots.setter
    def dipole_knots(self, value):
        self._dipole_knots = value

    @property
    def coefs_angle(self):
        if self.component_weights is not None and self.dipole_coefs_angle is not None:
            return combine_tables(self.component_weights, self.dipole_coefs_angle)
        return self._fixed_coefs_angle

    @property
    def coefs(self):
        if self._fixed_coefs is not None:
            return self._fixed_coefs
        if self.dipole_coefs_angle is not None:
            return self.coefs_angle.mean(0)
        return combine_tables(self.component_weights, self.dipole_knots)

    def _default_knots(self):
        if self.angle is None:
            return self.coefs
        return combine_tables(self.component_weights[self.angle], self.dipole_knots)

    @classmethod
    def from_vectorial(cls, vec, z_range, roisize=None, xy_limit=2.0, oversample=4, dz_um=0.01, zpad=6, chunk=4,
                       dipole_components=False):
        """The integrated spline of a VectorialPSF (its parameters, blur and dipole terms as they are), or of a list
        of per-angle VectorialPSFs (e.g. [PSF.load(calib, angle=a) for a in range(A)]: coefs_angle, one set per
        angle). dipole_components=True stores six density coefficient volumes per angle for per-emitter dipM.
        Point-density knots of the exact band-limited image (VectorialPSF 'point', point_knots),
        interpolated by the cubic B-spline (bspline_prefilter), on the knot grid of
        CubicSplinePSF(roisize, z_range, oversample, dz_um, zpad, xy_limit + 1) (roisize default vec.roisize). Knot
        z = emitter height at vec.focus_um. Accuracy (v4 in-situ PSFs, NA 1.49, 108 nm px, 0 - 0.4 um, offsets
        <= 2 px): max |spline - VectorialPSF 'spectral'| = 1.7e-5 of the peak at oversample 4, 5.3e-5 at 3, 8e-6
        at 5; dz_um 0.01 -> 0.005 changes it by < 1e-6."""
        vecs = list(vec) if isinstance(vec, (list, tuple)) else [vec]
        v0 = vecs[0]
        P = int(v0.roisize if roisize is None else roisize)
        geo = CubicSplinePSF(P, z_range, oversample=oversample, dz_um=dz_um, zpad=zpad, xy_limit=float(xy_limit) + 1.0,
                             device=v0.dev, pixel_um=v0.pixel_um)
        with torch.no_grad():
            K, bases = [], []
            for v in vecs:
                big = v.resized(geo.Pl)
                if dipole_components:
                    bases.append(torch.stack([bspline_prefilter(point_knots(big, geo, v.focus_um, dipM=M, chunk=chunk))
                                              for M in unit_tensors(big.dev)]))
                else:
                    K.append(bspline_prefilter(point_knots(big, geo, v.focus_um, chunk=chunk)))
        if dipole_components:
            weights = [default_weights(v.dipM, v.dipw).to(v0.dev) for v in vecs]
            return cls._from_components(torch.stack(bases) if len(bases) > 1 else bases[0],
                                        torch.stack(weights) if len(bases) > 1 else weights[0],
                                        roisize=P, oversample=oversample, dz_um=dz_um, z_range=z_range, zpad=zpad,
                                        knot_u0_px=geo.u0, knot_z0_um=geo.z0, pixel_um=v0.pixel_um,
                                        xy_limit=xy_limit, device=v0.dev)
        multi = isinstance(vec, (list, tuple)) and len(K) > 1
        return cls(P, None if multi else K[0], oversample, dz_um, z_range, zpad, geo.u0, geo.z0, pixel_um=v0.pixel_um,
                   coefs_angle=torch.stack(K) if multi else None, xy_limit=xy_limit, device=v0.dev)

    def _like(self, roisize=None, angle='same', xy_limit=None, device=None, dtype=None):
        geo = dict(roisize=self.roisize if roisize is None else roisize, oversample=self.S, dz_um=self.hz,
                   z_range=self.z_range, zpad=self.zpad, knot_u0_px=self.geo.u0, knot_z0_um=self.z0, pixel_um=self.pixel_um,
                   angle=self.angle if angle == 'same' else angle, xy_limit=xy_limit, chunk_elems=self.chunk_elems,
                   device=self.dev if device is None else device, dtype=self.dtype if dtype is None else dtype)
        if self.component_weights is not None:
            tables = self.dipole_coefs_angle if self.dipole_coefs_angle is not None else self.dipole_knots
            return self._from_components(tables, self.component_weights, **geo)
        return IntegratedSplinePSF(coefs=self.coefs, coefs_angle=self.coefs_angle, dipole_coefs=self.dipole_knots,
                                   dipole_coefs_angle=self.dipole_coefs_angle, **geo)

    def resized(self, roisize, xy_limit=None):
        """The same spline (shared coefficients) for roisize x roisize ROIs; offset limit = what the coefficients
        support for that ROI (or xy_limit when smaller)."""
        s = self._like(roisize, xy_limit=xy_limit)
        s.provenance = self.provenance
        return s

    def for_angle(self, angle):
        """The model of another pattern angle (None = the angle average), shared coefficients."""
        s = self._like(angle=angle, xy_limit=self.xmax)
        s.provenance = self.provenance
        return s

    @property
    def n_angles(self):
        if self.dipole_coefs_angle is not None:
            return len(self.dipole_coefs_angle)
        return 0 if self._fixed_coefs_angle is None else len(self._fixed_coefs_angle)

    def to(self, device=None, dtype=None):
        new = self._like(xy_limit=self.xmax, device=device, dtype=dtype)
        prov = self.provenance
        self.__dict__.update(new.__dict__)
        self.provenance = prov
        return self

    @property
    def z_limits(self):
        return self.z_range

    def render(self, pos, grad=False, knots=None, dipM=None):
        """PixelSplinePSF.render in chunks of emitters (the knot block of one emitter is 4 (P - 1) S + S + 4)^2
        floats: chunk_elems of them per chunk)"""
        if dipM is not None:
            return self._dipole_render(pos, grad, dipM, knots)
        n = max(1, self.chunk_elems // (4 * self.Lw * self.Lw))
        if pos.shape[0] <= n:
            return super().render(pos, grad, knots)
        table = self.knots if knots is None else knots
        parts = [super(IntegratedSplinePSF, self).render(p, grad, table) for p in pos.split(n)]
        if not grad:
            return torch.cat(parts)
        return tuple(torch.cat([p[k] for p in parts]) for k in range(4))

    # ------------------------------------------------------------------ files
    def to_dict(self):
        """Grid description (without the coefficients) for the PSF file."""
        return dict(basis=self.BASIS, value='pixel_integral', roisize=self.roisize, z_range=list(self.z_range),
                    oversample=self.S, dz_um=self.hz, zpad=self.zpad, xy_limit=self.xmax, c=self.geo.c, z0_um=self.z0,
                    nz=self.nz, u0_px=self.geo.u0, pixel_um=self.pixel_um, n_angles=self.n_angles,
                    **({'dipole_components': True} if self.dipole_coefs_angle is not None or self.dipole_knots is not None else {}))

    def _file_dict(self, source=None):
        d, arrays = {}, {}
        if source is not None:
            d, arrays = source._file_dict()
        d['integrated_spline'] = self.to_dict()
        if self.dipole_coefs_angle is not None:
            arrays['dipole_coefs_angle'] = self.dipole_coefs_angle.detach().cpu().numpy()
        elif self.dipole_knots is not None:
            arrays['dipole_coefs'] = self.dipole_knots.detach().cpu().numpy()
        if self.component_weights is not None:
            d['integrated_spline']['default_weights'] = self.component_weights.tolist()
        elif self.coefs_angle is None:
            arrays['coefs'] = self.coefs.detach().cpu().numpy()
        else:
            arrays['coefs_angle'] = self.coefs_angle.detach().cpu().numpy()
        return d, arrays

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, angle=None, **kw):
        g = d['integrated_spline']
        if g.get('basis') != cls.BASIS or g.get('value') != 'pixel_integral':
            raise ValueError(f"integrated_spline: unknown basis / value {g.get('basis')!r} / {g.get('value')!r} "
                             f"(this reader knows {cls.BASIS!r} / 'pixel_integral')")
        arrays = arrays or {}
        ca = arrays.get('coefs_angle')
        if 'default_weights' in g:
            key = 'dipole_coefs_angle' if g.get('n_angles') else 'dipole_coefs'
            m = cls._from_components(arrays[key], g['default_weights'], angle=angle, roisize=g['roisize'],
                                     oversample=g['oversample'], dz_um=g['dz_um'], z_range=g['z_range'], zpad=g['zpad'],
                                     knot_u0_px=g['u0_px'], knot_z0_um=g['z0_um'], pixel_um=g.get('pixel_um'),
                                     xy_limit=g.get('xy_limit'), device=resolve_device(device))
            if (m.n_angles, m.nz, m.geo.c) != (g['n_angles'], g['nz'], g['c']):
                raise ValueError('dipole coefficient arrays do not match the stored grid')
            return m if roisize in (None, m.roisize) else m.resized(roisize)
        if (ca is None) == ('coefs' not in arrays):
            raise ValueError("integrated_spline: exactly one of the arrays 'coefs' / 'coefs_angle' expected")
        if ca is not None and int(g.get('n_angles', len(ca))) != len(ca):
            raise ValueError(f"integrated_spline: n_angles {g.get('n_angles')} but coefs_angle has {len(ca)}")
        px = g.get('pixel_um') or (d['optics'].get('pixel_um') if d.get('optics') else None)
        component_kw = {}
        if g.get('dipole_components'):
            key = 'dipole_coefs' if ca is None else 'dipole_coefs_angle'
            component_kw[key] = arrays[key]
        m = cls(g['roisize'], arrays.get('coefs'), g['oversample'], g['dz_um'], tuple(g['z_range']), g['zpad'],
                g['u0_px'], g['z0_um'], pixel_um=px, coefs_angle=ca, angle=angle, xy_limit=g.get('xy_limit'),
                device=device, **component_kw)
        if m.nz != int(g['nz']) or m.geo.c != int(g['c']):
            raise ValueError(f"integrated_spline: coefficient array {tuple(m.coefs.shape)} does not match the grid "
                             f"(nz {g['nz']}, c {g['c']}, oversample {g['oversample']})")
        if roisize is not None and int(roisize) != m.roisize:
            m = m.resized(int(roisize))
        return m

    @staticmethod
    def load_source(path, roisize=None, device=None, angle=None):
        """The vectorial model the spline was built from (stored as its source), or None."""
        d, arrays, _ = PSF.read_file(path)
        if 'zernike_waves' not in d:
            return None
        return PSF._registry['vectorial']._from_file(d, arrays, roisize=roisize or d.get('roisize', 15), device=device,
                                                     angle=angle)


# ============================================================================================ spline-rendered track fit
def point_knots(big, geo, focus, blur=None, dipM=None, chunk=None):
    """Differentiable knot samples [nz, nv, nu] of the integrated spline (PixelSplinePSF integrate=True): the point
    density of the VectorialPSF `big` (roisize geo.Pl) at every knot, emitter height = knot z, at `focus` (tensor or
    float), with this blur / dipole tensor (None = big's attributes). Autograd flows to big's parameters, focus, blur
    and dipM (the same layout as CubicSplinePSF.knots_from_model)."""
    S, Pl = geo.S, geo.Pl
    A, B = geo._vec_positions()
    pixels = PixelIntegrator(big, big.blur if blur is None else blur, mode='point',
                             oversample=spectral_min_oversample(big.band_limit()))
    out = []
    step = geo.nz if chunk is None else chunk
    for z0 in range(0, geo.nz, step):
        z = geo.zs[z0:min(z0 + step, geo.nz)].float()
        n = len(z)
        pos = torch.stack([A[None].expand(n, -1), B[None].expand(n, -1), z[:, None].expand(-1, S * S)], -1)
        imgs = big._image(pos.reshape(-1, 3), focus, pixels, dipM=dipM).reshape(n, S, S, Pl, Pl)
        out.append(imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(n, Pl * S, Pl * S))
    return torch.cat(out)


def fit_tracks_spline(tracks, vectorial, readnoise_e=1.0, zmax_um=0.2, iters=600, warmup=100, focus_um=0.0,
                      blur='none', groups=None, ngroups=1, group_psf_kwargs=None, extra_params=(), batch=4000,
                      lr_xyz=0.03, lr_focus=0.005, lr_zernike=0.002, lr_blur=0.003, lr_extra=0.003, fix_zernike=False,
                      fix_focus=False, log=None, S=3, dz_um=0.01, xy_limit=None, final_exact=True, init_xy=None, init_z=None,
                      init_blur=None, lr_final=1.0):
    """estimation.fit_tracks with the samples rendered by the integrated cubic B-spline (PixelSplinePSF, S knots / px,
    dz_um, point-density knots of the exact vectorial model, max error ~3e-5 of the peak at S = 3) instead of the
    vectorial model: per iteration the knots are built once per PSF group from the current Zernikes / focus / blur /
    group terms (point_knots, with autograd), every chunk of samples is rendered by the spline with the knots detached
    (gradients to x, y, z and to the knots), then the knot gradient is pulled back to the PSF parameters (one backward
    through the knot build). Same arguments and result as fit_tracks (the model: the exact pixel integral of the
    vectorial PSF, psf.pixel_integration = 'spectral'); xy_limit: the spline's offset limit (px; default R / 2 - 0.5: a
    sample that runs off to a neighbour must render as in fit_tracks, a clamp at 2 px left ~2 % of the real-data
    samples stuck and cost 1e-3 nats / px). final_exact: the
    final NLL / residuals with the exact vectorial model (else with the spline).
    Warm start / refinement (2026-11-19): init_xy [B, 2] (px from the ROI centre) / init_z [B] (um, clamped into
    (0, zmax_um)) start every sample there instead of at xy_init / zmax_um / 2 (e.g. from a multi-start refit under the
    starting PSF, estimation.heldout_tracks); init_blur [sxx, syy, sxy] (px^2) starts the 'line' / 'cov' blur there (its
    Cholesky factor; default 0.05 px); lr_final: the learning rates of all parameters decay from their value at the end of
    the warmup to lr_final x that value at the last iteration (cosine; 1 = constant, as before). As in fit_tracks, the
    gradient goes through the closed-form I / bg (fit.solve_linear_grad): it is the Poisson-weighted IRLS estimate, not
    the minimiser of the Gaussian NLL that is evaluated, so holding I / bg fixed would bias the fit."""
    import math, time
    from torch_psf.fit import solve_linear_grad, readnoise_variance
    dev = tracks.rois.device
    B, T, R = tracks.rois.shape[0], tracks.rois.shape[1], tracks.rois.shape[-1]
    c = (R - 1) / 2
    xy_limit = R / 2 - 0.5 if xy_limit is None else xy_limit
    with torch.no_grad():
        spl = build_pixel_spline(vectorial.resized(R), R, (0.0, zmax_um), S, dz_um=dz_um, xy_limit=xy_limit,
                                 kernel='bspline', integrate=True)
    geo = spl.geo
    knot_os = spectral_min_oversample(vectorial.band_limit())
    big = vectorial.resized(geo.Pl, oversample=knot_os, pixel_integration='point')
    big.zcoef.requires_grad_(True)
    focus = torch.tensor(float(focus_um), device=dev, requires_grad=True)
    rn2, pix = readnoise_variance(readnoise_e, B, R, dev)
    xy = ((tracks.xy_init.float() - c) if init_xy is None else torch.as_tensor(init_xy, device=dev).float()).clone().requires_grad_(True)
    if init_z is None:
        zr = torch.zeros(B, device=dev, requires_grad=True)
    else:
        zf = (torch.as_tensor(init_z, device=dev).float() / zmax_um).clamp(1e-4, 1 - 1e-4)
        zr = torch.log(zf / (1 - zf)).clone().requires_grad_(True)
    BL = None
    if blur in ('line', 'cov') and init_blur is not None:
        sxx, syy, sxy = [float(v) for v in init_blur]
        a_ = math.sqrt(max(sxx, 1e-6)); b_ = sxy / a_
        BL = torch.tensor([a_, b_] if blur == 'line' else [a_, b_, math.sqrt(max(syy - b_ * b_, 1e-6))], device=dev, requires_grad=True)
    elif blur == 'line':
        BL = torch.tensor([0.05, 0.0], device=dev, requires_grad=True)
    elif blur == 'cov':
        BL = torch.tensor([0.05, 0.0, 0.05], device=dev, requires_grad=True)
    elif blur != 'none':
        raise ValueError(f"blur: 'none' | 'line' | 'cov', not {blur!r}")
    grp = torch.zeros(B, T, dtype=torch.long, device=dev) if groups is None else groups.to(dev)
    mask = tracks.mask
    extra = list(extra_params)
    opt = torch.optim.Adam([{'params': [xy, zr], 'lr': lr_xyz}, {'params': [focus], 'lr': lr_focus},
                            {'params': [big.zcoef], 'lr': lr_zernike}]
                           + ([{'params': [BL], 'lr': lr_blur}] if BL is not None else [])
                           + ([{'params': extra, 'lr': lr_extra}] if extra else []))
    lr0 = [g['lr'] for g in opt.param_groups]
    decay = lambda it: 1.0 if (lr_final == 1.0 or it < warmup) else \
        lr_final + (1 - lr_final) * 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(iters - 1 - warmup, 1)))
    npx = float(mask.sum()) * R * R
    ng = ngroups if group_psf_kwargs is not None else 1

    def blur_cov():
        if BL is None: return None
        if len(BL) == 2: return torch.stack([BL[0] ** 2, BL[1] ** 2, BL[0] * BL[1]])
        return torch.stack([BL[0] ** 2, BL[1] ** 2 + BL[2] ** 2, BL[0] * BL[1]])

    def group_kw(g):
        kw = dict(group_psf_kwargs(g)) if group_psf_kwargs is not None else {}
        if BL is not None: kw.setdefault('blur', blur_cov())
        return kw

    def knots():
        return [bspline_prefilter(point_knots(big, geo, focus, **group_kw(g))) for g in range(ng)]

    def loss(sl, K, render):
        data = tracks.rois[sl]
        b = data.shape[0]
        g_ = grp[sl]
        r2 = rn2[sl] if pix else rn2
        pos = torch.cat([xy[sl], zmax_um * torch.sigmoid(zr[sl])[:, None]], 1)
        psf = torch.stack([render(pos, g, K) for g in range(ng)], 1)                       # [b, ng, R, R]
        psf = torch.gather(psf, 1, (g_ if ng > 1 else torch.zeros_like(g_))[..., None, None].expand(-1, -1, R, R))
        I, bg = solve_linear_grad(psf.reshape(b, T, -1), data.reshape(b, T, -1), mask[sl], g_, r2, ngroups)
        mu = (I[..., None, None] * psf + torch.gather(bg, 1, g_)[..., None, None]).clamp(min=1e-3)
        var = mu + (r2.reshape(b, 1, R, R) if pix else r2)
        nll = 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
        nll = torch.where(mask[sl][..., None, None], nll, torch.zeros_like(nll)).sum()
        return nll, I, bg, mu, var, data

    spl_render = lambda pos, g, K: spl.render(pos, knots=K[g])
    chunks = torch.arange(B, device=dev).split(batch)
    hist, t0, tot = [], time.time(), float('nan')
    params = [focus, big.zcoef] + ([BL] if BL is not None else []) + extra
    for it in range(iters):
        for g_, l0 in zip(opt.param_groups, lr0):
            g_['lr'] = l0 * decay(it)
        opt.zero_grad()
        train_psf = it >= warmup
        with torch.set_grad_enabled(train_psf):
            Kg = knots()
        Kd = [k.detach().requires_grad_(train_psf) for k in Kg]
        tot = 0.0
        for ch in chunks:
            l = loss(ch, Kd, spl_render)[0] / npx
            l.backward()
            tot += float(l.detach())
        if train_psf:
            torch.autograd.backward(Kg, [k.grad for k in Kd])
        if fix_zernike:
            big.zcoef.grad = None
        if fix_focus:
            focus.grad = None
        opt.step()
        hist.append(dict(it=it, nll_per_px=tot, focus_um=float(focus.detach())))
        if log is not None:
            log(it, tot, time.time() - t0, big.zcoef.detach())
    out = vectorial.resized(R, oversample=max(vectorial.os, knot_os), pixel_integration='spectral')
    with torch.no_grad():
        out.zcoef.copy_(big.zcoef)
    out.focus_um = float(focus.detach())
    if BL is not None:
        out.blur = blur_cov().detach()
    with torch.no_grad():
        if final_exact:
            def ex_render(pos, g, K):
                kw = group_kw(g)
                return out(pos, focus.expand(pos.shape[0]), **kw)
            rend = ex_render
        else:
            rend = spl_render
        K = [k.detach() for k in knots()]
        Is, bgs, rs, final = [], [], [], 0.0
        for ch in chunks:
            l_, I, bg, mu, var, data = loss(ch, K, rend)
            final += float(l_) / npx
            Is.append(I); bgs.append(bg)
            rs.append(((data - mu) / var.sqrt())[mask[ch]])
        rs = torch.cat(rs)
    rep = dict(method='tracks_spline', n_samples=B, iterations=len(hist), resid_var=float(rs.var()), nll_per_px=final,
               nll_per_px_before_last_step=tot, focus_um=float(focus.detach()), seconds=time.time() - t0,
               spline=dict(S=S, dz_um=dz_um, xy_limit=xy_limit, kernel='bspline', integrate=True),
               at_limit=float((xy.detach().abs().max(1).values >= spl.xmax - 1e-3).float().mean()))
    return dict(psf=out, report=rep, history=hist, xy=xy.detach(), z=(zmax_um * torch.sigmoid(zr)).detach(),
                I=torch.cat(Is), bg=torch.cat(bgs), extra=[e.detach() for e in extra])
