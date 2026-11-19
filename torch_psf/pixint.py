"""
Exact pixel integration (prototype; not exported from torch_psf).

VectorialPSF integrates over a pixel by averaging s x s point samples (midpoint rule, `oversample` s). Two exact
alternatives and one spline representation are prototyped here:

1. ExactVectorialPSF: the same vectorial model with an exact (or near-exact) pixel integral.
   The image-plane intensity |E|^2 of a pupil-function model is strictly band-limited: E is a finite sum of plane waves
   with |k| <= k0 NA, so |E|^2 has no spatial frequencies above fc = 2 NA / lambda (cycles / um), 0.46 cycles / px for
   NA 1.49, 700 nm, 108 nm pixels. Two ways to use that:
   - mode 'spectral': point samples on a uniform grid with os samples per px (os >= 2 is alias-free), padded by `pad`
     px per side; one FFT; multiply by the exact transfer function of the pixel box, sinc(fx) sinc(fy), and by the
     optional image-plane blur exp(-2 pi^2 f^T C f); inverse FFT, sampled at the pixel centres. Exact up to the
     float32 round-off and the circular wrap at the padded border (a raised-cosine taper between fc and os / 2 keeps
     the kernel compact). Free in the blur path, which pads and FFTs already.
   - mode 'gauss': Gauss-Legendre quadrature with n nodes per axis instead of the midpoint rule (same cost as
     oversample n, no padding; error ~ (pi fc)^(2n) (n!)^4 / ((2n + 1) ((2n)!)^3): 1e-3 of the top frequency
     component for n = 2, 5e-6 for n = 3). No blur (a blur needs the uniform grid).
   - mode 'box': the current midpoint rule (= VectorialPSF), mode 'point': the point density at the pixel centres
     (per px^2; the samples for the integrated spline below).

2. PixelSplinePSF: a tensor-product cubic spline (Catmull-Rom or interpolating cubic B-spline) in u, v, z on the
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
import math
import numpy as np
import torch
from torch_psf.psf import PSF
from torch_psf.spline import CubicSplinePSF
from torch_psf.vectorial import dipole_tensor


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
    return c.to(samples.dtype)


# ============================================================================================ exact vectorial
def _gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x / 2, w / 2                                # nodes on [-1/2, 1/2], weights summing to 1


class ExactVectorialPSF(PSF):
    """Wraps a VectorialPSF `vec` (its parameters are used as they are) with another pixel integration (see the module
    docstring): mode 'spectral' (os samples per px, `pad` px padding, raised-cosine taper above the band limit),
    'gauss' (n = os Gauss-Legendre nodes per axis; no blur), 'box' (midpoint rule with os x os samples) or 'point' (the
    density at the pixel centres, per px^2). render(pos, grad) as VectorialPSF.render; the blur / dipole terms of vec
    apply."""

    def __init__(self, vec, mode='spectral', os=2, pad=3):
        self.vec, self.mode, self.os, self.pad = vec, mode, int(os), int(pad)
        self.roisize, self.pixel_um, self.dev = vec.roisize, vec.pixel_um, vec.dev
        self.z_range, self.focus_um = vec.z_range, vec.focus_um
        if mode == 'gauss' and vec.blur is not None:
            raise ValueError("mode 'gauss' does not support an image-plane blur; use 'spectral'")
        if mode in ('spectral', 'point'):
            assert self.os >= 2, 'the spectral modes need os >= 2 (alias-free sampling of |E|^2)'

    def resized(self, roisize):
        return ExactVectorialPSF(self.vec.resized(roisize), self.mode, self.os, self.pad)

    @property
    def xy_limit(self):
        return self.vec.xy_limit

    def band_limit(self):
        """Highest spatial frequency of |E|^2 in cycles / px: 2 NA / lambda * pixel size."""
        return 2 * float(self.vec.NA) / self.vec.wavelength_um * self.pixel_um

    def _grid(self):
        """1D sample positions (px, relative to the ROI centre) and their weights per pixel."""
        R, os_ = self.roisize, self.os
        if self.mode == 'gauss':
            x, w = _gauss_legendre(os_)
            c = np.arange(R) - (R - 1) / 2
            u = (c[:, None] + x[None]).reshape(-1)
            return torch.tensor(u, dtype=torch.float32, device=self.dev), torch.tensor(w, dtype=torch.float32, device=self.dev), 0
        pad = self.pad if self.mode in ('spectral', 'point') else (self.vec.BLUR_PAD if self.vec.blur is not None else 0)
        ns = (R + 2 * pad) * os_
        u = (torch.arange(ns, device=self.dev, dtype=torch.float32) - (ns - 1) / 2) / os_
        return u, None, pad

    def density(self, pos, focus, u, blur=None):
        """|E|^2 on the grid u x u (px) as a density per px^2 (normalized as VectorialPSF), [B, n, n]. A copy of the
        first part of VectorialPSF.forward with another image grid."""
        vec = self.vec
        B = pos.shape[0]
        P, kz_med, kz_imm, kx, ky = vec.pupil_fields()
        Mt = dipole_tensor(vec.dipM, vec.dipw, vec.dev)
        zc = vec.zcoef[None].expand(B, -1)
        W = (torch.einsum('bj,jnm->bnm', zc, vec.Z) + vec.pupil_phase[None]) * (2 * math.pi)
        x = pos[:, 0, None, None] * vec.pixel_um
        y = pos[:, 1, None, None] * vec.pixel_um
        ze = pos[:, 2, None, None]
        f = focus[:, None, None] if torch.is_tensor(focus) and focus.dim() > 0 else focus
        ph = torch.exp(1j * (W - kx * x - ky * y - kz_imm * f)) * torch.exp(1j * kz_med * ze)
        Pb = P[None] * ph[:, None]
        kx1, ky1 = kx[0], ky[:, 0]
        uu = u * vec.pixel_um
        Fx = torch.exp(1j * uu[:, None] * kx1[None, :])
        Fy = torch.exp(1j * uu[:, None] * ky1[None, :])
        E = Fy @ Pb.to(torch.complex64) @ Fx.T
        if Mt is None:
            I = (E.real ** 2 + E.imag ** 2).sum(1) / 3
        else:
            Ed = E.reshape(B, 3, 2, *E.shape[-2:])
            Mc = Mt.to(Ed.real.dtype)
            F = torch.einsum('ij,bjcuv->bicuv', Mc, Ed.real) + 1j * torch.einsum('ij,bjcuv->bicuv', Mc, Ed.imag)
            I = (Ed.real * F.real + Ed.imag * F.imag).sum((1, 2))
        dk = (kx1[1] - kx1[0]) * (ky1[1] - ky1[0])
        energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        return I * (dk / (2 * math.pi) ** 2) ** 2 * vec.pixel_um ** 2 / energy

    def _spectral(self, D, box):
        """Pixel integral (box) or point value (not box) at the pixel centres of the padded os-grid density D [B, n, n],
        with the optional blur; cropped by pad."""
        os_, pad, R = self.os, self.pad, self.roisize
        n = D.shape[-1]
        f = torch.fft.fftfreq(n, d=1.0 / os_, device=D.device)           # cycles / px
        fr = torch.fft.rfftfreq(n, d=1.0 / os_, device=D.device)
        fc = min(self.band_limit() * 1.02, 0.5 * os_ - 0.05)
        fn = 0.5 * os_

        def taper(q):
            a = ((q.abs() - fc) / (fn - fc)).clamp(0, 1)
            return 0.5 * (1 + torch.cos(math.pi * a))

        d = (os_ - 1) / (2 * os_)                                          # sample p os -> pixel centre p
        Ty = taper(f) * (torch.sinc(f) if box else 1.0) * torch.exp(2j * math.pi * f * d)
        Tx = taper(fr) * (torch.sinc(fr) if box else 1.0) * torch.exp(2j * math.pi * fr * d)
        H = Ty[:, None] * Tx[None, :]
        bl = self.vec.blur
        if bl is not None:
            cov = torch.as_tensor(bl, device=D.device, dtype=torch.float32).reshape(-1, 3)
            q = cov[:, 0, None, None] * fr[None, :] ** 2 + cov[:, 1, None, None] * f[:, None] ** 2 \
                + 2 * cov[:, 2, None, None] * fr[None, :] * f[:, None]
            H = H * torch.exp((-2 * math.pi ** 2 * q).clamp(max=3.0))
        J = torch.fft.irfft2(torch.fft.rfft2(D) * H, s=(n, n))
        J = J[:, ::os_, ::os_]
        return J[:, pad:pad + R, pad:pad + R]

    def forward(self, pos, focus=None):
        f = self.focus_um if focus is None else focus
        u, w, pad = self._grid()
        R, os_ = self.roisize, self.os
        D = self.density(pos, f, u)
        if self.mode in ('spectral', 'point'):
            return self._spectral(D, self.mode == 'spectral')
        if self.mode == 'gauss':
            B = D.shape[0]
            D = D.reshape(B, R, os_, R, os_)
            return torch.einsum('biajn,a,n->bij', D, w, w)
        # 'box': as VectorialPSF.forward
        if self.vec.blur is not None:
            from torch_psf.vectorial import gauss_blur_os
            D = gauss_blur_os(D, self.vec.blur, os_, pad)
        B = D.shape[0]
        return D.reshape(B, R, os_, R, os_).mean((2, 4))

    def render(self, pos, grad=False, focus=None):
        if not grad:
            return self.forward(pos, focus)
        fn = lambda p: self.forward(p, focus)
        out, val = [], None
        for k in range(3):
            t = torch.zeros_like(pos)
            t[:, k] = 1
            v, d = torch.func.jvp(fn, (pos,), (t,))
            val = v if val is None else val
            out.append(d)
        return val, out[0], out[1], out[2]


# ============================================================================================ spline
class PixelSplinePSF(PSF):
    """Cubic tensor-product spline PSF on the knot grid of a CubicSplinePSF `geo` (same knot layout: spacing 1 / S px,
    origin geo.u0, z planes geo.zs). kernel 'catmull' | 'bspline' (samples are prefiltered to B-spline coefficients).
    integrate=True: samples = point density f (per px^2) at the knots, pixel value = exact integral of the spline over
    the pixel; integrate=False: samples = pixel values at sub-pixel shifts, interpolated pointwise.
    backend 'sep': separable einsum contraction (z, x, y); 'conv': one grouped conv2d as CubicSplinePSF (cuDNN; uses
    TF32 on Ampere GPUs unless torch.backends.cudnn.allow_tf32 = False, ~2e-4 of the peak)."""

    def __init__(self, geo, samples, kernel='catmull', integrate=True, backend='sep'):
        self.geo, self.kernel, self.integrate, self.backend = geo, kernel, integrate, backend
        g = geo
        self.roisize, self.S, self.hz, self.nz, self.nv, self.nu = g.roisize, g.S, g.hz, g.nz, g.nv, g.nu
        self.dev, self.dtype, self.pixel_um = g.dev, g.dtype, g.pixel_um
        self.z_range, self.z0 = g.z_range, g.z0
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
        base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
        sub = K.reshape(-1).index_select(0, (base[:, None] + self.pat[None]).reshape(-1)).reshape(1, N * 4, Lw, Lw)
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
    """PixelSplinePSF of the VectorialPSF `vec`: integrate=True samples the point density (ExactVectorialPSF 'point');
    integrate=False samples pixel values of `sample_model(vec_resized)` (default ExactVectorialPSF 'spectral') at the
    S x S sub-pixel shifts. The knot grid is that of CubicSplinePSF(roisize, z_range, S, dz_um, zpad, xy_limit).
    zpad default: 3 planes (catmull), 6 (bspline: the mirror boundary of the prefilter perturbs the first ~4 planes)."""
    zpad = (6 if kernel == 'bspline' else 3) if zpad is None else zpad
    xy = roisize / 2 + 1 if xy_limit is None else xy_limit
    geo = CubicSplinePSF(roisize, z_range, oversample=S, dz_um=dz_um, zpad=zpad, xy_limit=xy + 1.0, device=vec.dev,
                         pixel_um=vec.pixel_um)
    big = vec.resized(geo.Pl)
    if integrate:
        model = ExactVectorialPSF(big, 'point', os=2, pad=3)
    else:
        model = sample_model(big) if sample_model is not None else ExactVectorialPSF(big, 'spectral', os=2, pad=3)
    samples = geo.knots_from_model(model, chunk=chunk)
    spl = PixelSplinePSF(geo, samples, kernel=kernel, integrate=integrate, backend=backend)
    spl.xmax = min(spl.xmax, xy)
    return spl


# ============================================================================================ spline-rendered track fit
def point_knots(big, geo, focus, blur=None, dipM=None, chunk=None):
    """Differentiable knot samples [nz, nv, nu] of the integrated spline (PixelSplinePSF integrate=True): the point
    density of the VectorialPSF `big` (roisize geo.Pl) at every knot, emitter height = knot z, at `focus` (tensor or
    float), with this blur / dipole tensor (None = big's attributes). Autograd flows to big's parameters, focus, blur
    and dipM (the same layout as CubicSplinePSF.knots_from_model)."""
    S, Pl = geo.S, geo.Pl
    A, B = geo._vec_positions()
    ob, od = big.blur, big.dipM
    if blur is not None:
        big.blur = blur
    if dipM is not None:
        big.dipM = dipM
    try:
        ex = ExactVectorialPSF(big, 'point', os=2, pad=3)
        out = []
        step = geo.nz if chunk is None else chunk
        for z0 in range(0, geo.nz, step):
            z = geo.zs[z0:min(z0 + step, geo.nz)].float()
            n = len(z)
            pos = torch.stack([A[None].expand(n, -1), B[None].expand(n, -1), z[:, None].expand(-1, S * S)], -1)
            imgs = ex.forward(pos.reshape(-1, 3), focus).reshape(n, S, S, Pl, Pl)
            out.append(imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(n, Pl * S, Pl * S))
        return torch.cat(out)
    finally:
        big.blur, big.dipM = ob, od


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
    big = vectorial.resized(geo.Pl, pixel_integration='spectral')
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
    out = vectorial.resized(R, pixel_integration='spectral')
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
