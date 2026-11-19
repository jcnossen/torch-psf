"""
Vectorial pupil-function PSF model (PyTorch, batched, differentiable).

Physics:
 - Emitter: freely rotating dipole (incoherent sum over x, y, z dipole orientations; or a given emission-dipole
   second-moment tensor M) at height z_e above the coverslip, in a medium with index n_med. The coverslip is assumed
   index-matched to the immersion (n_imm).
 - Fresnel transmission at the medium / coverslip interface (s and p), including supercritical-angle fluorescence
   (evanescent components for NA > n_med decay with the emitter height).
 - Aplanatic apodization 1 / sqrt(cos(theta_imm)), optional radial amplitude exp(sum_n apod_n rho^(2n)), optional
   asymmetry of the supercritical annulus (saf), optional detection diattenuation (diat).
 - Aberrations: Zernike phase (Noll index, in waves rms), optionally per emitter (field dependence), plus a free pupil
   phase map (waves) and a free pupil amplitude map (both neutral by default).
 - Defocus: focal-plane position f relative to the coverslip, in the sample's coordinate:
   phase = k0 (n_med cos(theta_med) z_e - n_imm cos(theta_imm) f).
 - Image: matrix Fourier transform (chirp-z) from the pupil to an oversampled pixel grid, optional image-plane Gaussian
   blur (covariance in px^2, e.g. a line blur v v^T), integration over the pixels (pixel_integration):
   'box'            the mean of the oversample x oversample point samples of each pixel (midpoint rule; the default).
                    Its error is mostly a missing isotropic blur of 1 / (12 oversample^2) px^2 (1.4 % of the peak at
                    oversample 2, NA 1.49, 700 nm, 108 nm pixels).
   'spectral'       exact: |E|^2 is band-limited (no spatial frequency above fc = 2 NA / lambda), so its point samples on
                    a uniform grid with oversample > 2 fc pixel_um per px (>= 2; 2 suffices for pixels < lambda / (2 NA),
                    otherwise ValueError) (padded by PIX_PAD px per side) determine it; one FFT,
                    times the transfer function of the pixel box sinc(fx) sinc(fy) and of the optional blur, inverse
                    FFT, sampled at the pixel centres (a raised-cosine taper between the band limit and the grid's
                    Nyquist frequency keeps the kernel compact). Exact up to float32 round-off and the wrap at the
                    padded border (~1e-5 of the peak at oversample 2); costs about oversample 3 of 'box'.
   'gauss_legendre' Gauss-Legendre quadrature with oversample nodes per axis (error ~1e-5 of the peak at 3 nodes;
                    the cost of 'box' at the same oversample); no image-plane blur.
   ('exact' = 'spectral'.)
 - Normalization: the PSF integrates to 1 over the whole image plane for an aberration-free emitter at z_e = 0, i.e.
   `intensity` = detected photons of an emitter at the coverslip.

Units: um for lengths (pixel size, wavelength, z, focus), px for lateral positions, waves for aberrations.

Besides the forward model: free pupil maps (neutral by default), the common PSF-model interface (focus_um, render,
z_range, xy_limit) and copy / resize helpers.
"""
import copy
import math
import numpy as np
import torch
from torch_psf.zernike import zernike
from torch_psf.optics import default_device, Optics
from torch_psf.psf import PSF
from torch_psf.spline import CubicSplinePSF


def _list(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    return np.asarray(v).tolist()


def gauss_blur_os(I, cov, os_, pad):
    """Anisotropic Gaussian blur of oversampled intensity images I [B, n, n] (n = (R + 2 pad) os) with covariance
    cov = [sxx, syy, sxy] (px^2; [3] or [B, 3]), by multiplication with exp(-2 pi^2 f^T C f) in the Fourier domain,
    then cropped by `pad` px per side. The padding keeps the circular wrap out of the ROI. A negative-definite cov
    (deblur) is allowed but its gain is capped."""
    B, n = I.shape[0], I.shape[-1]
    cov = torch.as_tensor(cov, device=I.device, dtype=torch.float32)
    cov = cov.reshape(-1, 3)
    f = torch.fft.fftfreq(n, d=1.0 / os_, device=I.device)            # cycles / px
    fr = torch.fft.rfftfreq(n, d=1.0 / os_, device=I.device)
    fy, fx = f[:, None], fr[None, :]
    q = (cov[:, 0, None, None] * fx ** 2 + cov[:, 1, None, None] * fy ** 2 + 2 * cov[:, 2, None, None] * fx * fy)
    H = torch.exp((-2 * math.pi ** 2 * q).clamp(max=3.0))
    J = torch.fft.irfft2(torch.fft.rfft2(I) * H, s=(n, n))
    c = pad * os_
    return J[:, c:n - c, c:n - c] if c > 0 else J


PIXEL_INTEGRATIONS = ('box', 'spectral', 'gauss_legendre')


def pixel_integration_mode(mode):
    """Validated pixel-integration name (VectorialPSF.pixel_integration); 'exact' = 'spectral', None = 'box'. Any other
    value (another string, a number, ...) raises ValueError: a model never silently falls back to 'box'."""
    if mode is None:
        return 'box'
    m = {'exact': 'spectral', 'gauss': 'gauss_legendre', 'midpoint': 'box'}.get(mode, mode) if isinstance(mode, str) else None
    if m not in PIXEL_INTEGRATIONS:
        raise ValueError(f"unknown pixel_integration {mode!r}: one of {PIXEL_INTEGRATIONS} (or 'exact')")
    return m


def spectral_min_oversample(fc):
    """Smallest oversample (>= 2) whose grid Nyquist frequency oversample / 2 is above the band limit fc (cycles / px) of
    |E|^2, so that its point samples do not alias (pixel_integration 'spectral')."""
    return max(2, int(math.floor(2 * float(fc))) + 1)


def check_spectral_oversample(fc, os_):
    """ValueError when the point samples of a density band-limited to fc (cycles / px) alias at os_ samples / px."""
    if float(fc) >= 0.5 * os_:
        raise ValueError(f"pixel_integration 'spectral': the band limit of |E|^2 (2 NA / wavelength x pixel size = "
                         f"{float(fc):.3f} cycles / px) is not below the Nyquist frequency of oversample {os_} "
                         f"({0.5 * os_:g} cycles / px), so the point samples alias; use oversample >= "
                         f"{spectral_min_oversample(fc)}")


def pixel_box_spectral(I, os_, pad, R, fc, cov=None, *, integrate=True):
    """Exact pixel means of a band-limited density from its point samples I [B, n, n] on the uniform grid with os_
    samples per px, n = (R + 2 pad) os_ (sample j at (j - (n - 1) / 2) / os_ px from the ROI centre): FFT, times the
    pixel-box transfer function sinc(fx) sinc(fy), the optional Gaussian blur exp(-2 pi^2 f^T C f) (cov = [sxx, syy, sxy]
    px^2, [3] or [B, 3]; gain capped as gauss_blur_os) and a raised-cosine taper from the band limit fc (cycles / px)
    to the Nyquist frequency os_ / 2, shifted onto the pixel centres; inverse FFT, every os_-th sample, cropped by pad
    -> [B, R, R] (pixel means, or centre densities when integrate=False). ValueError when fc >= os_ / 2."""
    check_spectral_oversample(fc, os_)
    n = I.shape[-1]
    f = torch.fft.fftfreq(n, d=1.0 / os_, device=I.device)            # cycles / px
    fr = torch.fft.rfftfreq(n, d=1.0 / os_, device=I.device)
    fn = 0.5 * os_
    fc = min(float(fc) * 1.02, fn - 0.05)

    def taper(q):
        a = ((q.abs() - fc) / (fn - fc)).clamp(0, 1)
        return 0.5 * (1 + torch.cos(math.pi * a))

    d = (os_ - 1) / (2 * os_)                                          # sample p os_ -> pixel centre p: shift by d px
    Ty = taper(f) * (torch.sinc(f) if integrate else 1) * torch.exp(2j * math.pi * f * d)
    Tx = taper(fr) * (torch.sinc(fr) if integrate else 1) * torch.exp(2j * math.pi * fr * d)
    H = Ty[:, None] * Tx[None, :]
    if cov is not None:
        cov = torch.as_tensor(cov, device=I.device, dtype=torch.float32).reshape(-1, 3)
        q = cov[:, 0, None, None] * fr[None, :] ** 2 + cov[:, 1, None, None] * f[:, None] ** 2 \
            + 2 * cov[:, 2, None, None] * fr[None, :] * f[:, None]
        H = H * torch.exp((-2 * math.pi ** 2 * q).clamp(max=3.0))
    J = torch.fft.irfft2(torch.fft.rfft2(I) * H, s=(n, n))
    return J[:, ::os_, ::os_][:, pad:pad + R, pad:pad + R]


def _angle_index(angle, n):
    """Index of a pattern angle into n per-angle terms; ValueError for bool, fractional or out-of-range values."""
    a = None if isinstance(angle, bool) else int(angle)
    if a is None or a != angle or not 0 <= a < n:
        raise ValueError(f'angle={angle!r}: this PSF has per-angle terms for angles 0..{n - 1}')
    return a


def dipole_tensor(M=None, w=None, dev=None):
    """Emission-dipole second-moment tensor (trace 1) from M ([3, 3] or [B, 3, 3]) or diagonal weights w ([3], mean 1);
    None if both are None (isotropic emitter)."""
    if M is not None:
        return torch.as_tensor(M, device=dev, dtype=torch.float32)
    if w is not None:
        return torch.diag_embed(torch.as_tensor(w, device=dev, dtype=torch.float32)) / 3
    return None


def photosel_tensor(gamma, e_inplane, eps=0.0, k_dir=None, ck=0.153):
    """Photoselection: M = (1 - gamma) I / 3 + gamma K, with K the excitation polarisation tensor (trace 1):
    K = (1 - eps) e e^T + eps (ck k k^T + (1 - ck) z z^T) (eps = p-polarised fraction of an evanescent excitation;
    ck = |E_x|^2 / |E|^2 of the evanescent p wave). e_inplane, k_dir: in-plane unit 2-vectors (x = column, y = row)."""
    dev = gamma.device if torch.is_tensor(gamma) else None
    e = torch.zeros(3, device=dev); e[:2] = torch.as_tensor(e_inplane, device=dev, dtype=torch.float32)
    K = e[:, None] * e[None, :]
    if k_dir is not None:
        kk = torch.zeros(3, device=dev); kk[:2] = torch.as_tensor(k_dir, device=dev, dtype=torch.float32)
        z = torch.zeros(3, device=dev); z[2] = 1
        Kp = ck * kk[:, None] * kk[None, :] + (1 - ck) * z[:, None] * z[None, :]
        K = (1 - eps) * K + eps * Kp
    return (1 - gamma) * torch.eye(3, device=dev) / 3 + gamma * K


class VectorialPSF(torch.nn.Module, PSF):
    """Vectorial PSF on a roisize x roisize patch (see the module docstring for the physics).

    Constructor: roisize (px), pixel_um, NA, n_imm, n_med, wavelength_um, zernike_indices (Noll), npupil (pupil
    samples per side), oversample (image samples per px per side), device (default: cuda if available); fit_NA / fit_nmed make NA / n_med trainable.
    focus_um (focal plane position used by render) and z_range (um, the emitter heights the model is meant for).
    pixel_integration: 'box' (default) | 'spectral' (= 'exact') | 'gauss_legendre' (module docstring; an attribute,
    copied by resized() and stored in the file when not 'box').

    Parameters / buffers: zcoef [nz] (waves rms), NA, n_med, apod [3], saf [4]; buffers pupil_phase [npupil, npupil]
    (waves, 0) and pupil_amp [npupil, npupil] (1). Optional plain attributes: blur (covariance [3] px^2), diat (Ex / Ey
    diattenuation), dipM (3 x 3 dipole tensor) or dipw (its diagonal, mean 1); None = not applied.
    """
    BLUR_PAD = 3   # px of padding per side for the image-plane blur (kernel sd <~ 0.7 px)
    PIX_PAD = 3    # px of padding per side of pixel_integration 'spectral' (also covers the blur)
    type_name = 'vectorial'

    def __init__(self, roisize, pixel_um=0.10833, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68,
                 zernike_indices=tuple(range(5, 22)), npupil=64, oversample=3, device=None, fit_NA=False,
                 fit_nmed=False, focus_um=0.0, z_range=(0.0, 1.0), pixel_integration='box'):
        super().__init__()
        self.roisize = roisize
        self.pixel_um = pixel_um
        self.npupil = npupil
        self.os = oversample
        self.pixel_integration = pixel_integration
        self.zidx = list(zernike_indices)
        self.n_imm, self.wavelength_um = n_imm, wavelength_um
        self.dev = torch.device(default_device() if device is None else device)
        self.focus_um = float(focus_um)
        self.z_range = None if z_range is None else (float(z_range[0]), float(z_range[1]))
        self.NA = torch.nn.Parameter(torch.tensor(float(NA), device=self.dev), requires_grad=fit_NA)
        self.n_med = torch.nn.Parameter(torch.tensor(float(n_med), device=self.dev), requires_grad=fit_nmed)
        self.zcoef = torch.nn.Parameter(torch.zeros(len(self.zidx), device=self.dev))  # waves (rms)
        # radial pupil amplitude (objective transmission): A(rho) = exp(sum_n c_n rho^(2n)), n = 1..3
        self.apod = torch.nn.Parameter(torch.zeros(3, device=self.dev), requires_grad=False)
        # supercritical-annulus asymmetry: on rho > n_med / NA only, phase tilt (waves per unit radius) and amplitude
        # modulation along x and y: [t_x, t_y, a_x, a_y]
        self.saf = torch.nn.Parameter(torch.zeros(4, device=self.dev), requires_grad=False)
        # optional terms (plain attributes; None = not applied, the images are then unchanged)
        self.diat = None    # detection Ex / Ey diattenuation d: Ex scaled by sqrt(1 + d), Ey by sqrt(1 - d)
        self.blur = None    # image-plane Gaussian blur covariance [sxx, syy, sxy] px^2 ([B, 3] per emitter)
        self.dipw = None    # emission-dipole diagonal weights [wx, wy, wz] (mean 1)
        self.dipM = None    # emission-dipole second-moment tensor (3 x 3, trace 1)
        # per illumination-pattern angle terms of a file (psf_dipole_M_angle [A, 3, 3], psf_blur_angle_px2 [A, 3]): kept
        # so that writing the model keeps them; dipM / blur hold the selected angle (or the average)
        self.dipM_angle = None
        self.blur_angle = None

        # normalized pupil grid (unit disk), fixed; physical k scales with NA
        g = (torch.arange(npupil, device=self.dev) - (npupil - 1) / 2) / (npupil / 2)
        self.register_buffer('px', g[None, :].expand(npupil, npupil).clone())
        self.register_buffer('py', g[:, None].expand(npupil, npupil).clone())
        rho = torch.sqrt(self.px ** 2 + self.py ** 2)
        self.register_buffer('rho', rho)
        self.register_buffer('phi', torch.atan2(self.py, self.px))
        self.register_buffer('mask', (rho <= 1).float())
        Z = torch.stack([zernike(j, rho.clamp(max=1), self.phi) for j in self.zidx]) * self.mask
        self.register_buffer('Z', Z)  # [nz, N, N]
        # free pupil maps (not Zernike): phase in waves, amplitude factor
        self.register_buffer('pupil_phase', torch.zeros(npupil, npupil, device=self.dev))
        self.register_buffer('pupil_amp', torch.ones(npupil, npupil, device=self.dev))

        # image sample positions (oversampled), relative to the ROI centre, in um
        ns = roisize * oversample
        u = (torch.arange(ns, device=self.dev) - (ns - 1) / 2) / oversample * pixel_um
        self.register_buffer('u', u)
        self._check_spectral()

    @property
    def pixel_integration(self):
        return self._pixint

    @pixel_integration.setter
    def pixel_integration(self, mode):
        m = pixel_integration_mode(mode)
        if m == 'spectral' and self.os < 2:
            raise ValueError("pixel_integration 'spectral' needs oversample >= 2 (alias-free point samples of |E|^2)")
        if m == 'spectral' and hasattr(self, 'NA'):            # (in __init__: checked once NA exists)
            check_spectral_oversample(self.band_limit(), self.os)
        self._pixint = m

    def _check_spectral(self):
        """'spectral': the oversampled grid must resolve the band limit of |E|^2 (ValueError otherwise)."""
        if self._pixint == 'spectral':
            check_spectral_oversample(self.band_limit(), self.os)

    def band_limit(self):
        """Highest spatial frequency of |E|^2 (cycles / px): 2 NA / lambda * pixel size."""
        return 2 * float(self.NA.detach()) / self.wavelength_um * self.pixel_um

    def _image_grid(self, blur):
        """(1D image sample positions in um from the ROI centre, Gauss-Legendre weights or None, padding px) of the
        pixel integration"""
        R, o, mode = self.roisize, self.os, self.pixel_integration
        if mode == 'gauss_legendre':
            if blur is not None:
                raise ValueError("pixel_integration 'gauss_legendre' does not support an image-plane blur; use 'spectral'")
            x, w = np.polynomial.legendre.leggauss(o)
            uc = (np.arange(R) - (R - 1) / 2)[:, None] + x[None] / 2
            t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.dev)
            return t(uc.reshape(-1)) * self.pixel_um, t(w / 2), 0
        pad = self.PIX_PAD if mode == 'spectral' else (self.BLUR_PAD if blur is not None else 0)
        if pad == 0:
            return self.u, None, 0
        ns_p = (R + 2 * pad) * o
        return (torch.arange(ns_p, device=self.dev) - (ns_p - 1) / 2) / o * self.pixel_um, None, pad

    # ------------------------------------------------------------------ physics
    def pupil_fields(self):
        """Per-dipole / polarization pupil amplitudes [6, N, N] (complex), kz in the medium and the immersion, kx, ky."""
        NA = self.NA
        k0 = 2 * math.pi / self.wavelength_um
        s_imm = NA * self.rho / self.n_imm
        c_imm = torch.sqrt((1 - s_imm ** 2).clamp(min=1e-6))
        s_med = (NA * self.rho / self.n_med).to(torch.complex64)
        c_med = torch.sqrt(1 - s_med ** 2)  # complex; imag > 0 for supercritical angles
        c_med = torch.where(c_med.imag < 0, -c_med, c_med)
        # a pupil sample exactly at the critical angle (NA rho = n_med, e.g. while NA is fitted) has c_med = 0; the
        # fields below are finite there (t_s, t_p ~ c_med cancel the 1 / c_med), so move it off the singularity
        c_med = torch.where(c_med.abs() < 1e-6, torch.full_like(c_med, 1e-6j), c_med)
        n1, n2 = self.n_med.to(torch.complex64), self.n_imm
        ts = 2 * n1 * c_med / (n1 * c_med + n2 * c_imm)
        tp = 2 * n1 * c_med / (n2 * c_med + n1 * c_imm)
        cphi, sphi = torch.cos(self.phi), torch.sin(self.phi)
        # far field of a dipole near an interface (Novotny & Hecht ch. 10): t (n2 / n1) cos2 / cos1, times the
        # aplanatic apodization 1 / sqrt(cos2)
        apod = self.mask / torch.sqrt(c_imm) * (n2 / n1) * c_imm / c_med
        r2 = self.rho ** 2
        apod = apod * torch.exp(self.apod[0] * r2 + self.apod[1] * r2 ** 2 + self.apod[2] * r2 ** 3)
        apod = apod * self.pupil_amp
        ann = (self.rho * NA > self.n_med).float()
        apod = apod * (1 + ann * (self.saf[2] * self.px + self.saf[3] * self.py))
        apod = apod * torch.exp(1j * 2 * math.pi * ann * (self.saf[0] * self.px + self.saf[1] * self.py))
        # p-vector in the medium (cos_med cos phi, cos_med sin phi, -sin_med), s-vector (-sin phi, cos phi, 0)
        # pupil field = tp (mu.p) (cos phi, sin phi) + ts (mu.s) (-sin phi, cos phi)
        Ex_x = tp * c_med * cphi * cphi + ts * sphi * sphi
        Ey_x = tp * c_med * cphi * sphi - ts * sphi * cphi
        Ex_y = tp * c_med * sphi * cphi - ts * cphi * sphi
        Ey_y = tp * c_med * sphi * sphi + ts * cphi * cphi
        Ex_z = -tp * s_med * cphi
        Ey_z = -tp * s_med * sphi
        P = torch.stack([Ex_x, Ey_x, Ex_y, Ey_y, Ex_z, Ey_z]) * apod
        d = self.diat
        if d is not None and (torch.is_tensor(d) or d != 0):
            d = d if torch.is_tensor(d) else torch.tensor(float(d), device=self.dev)
            wx, wy = torch.sqrt(1 + d), torch.sqrt(1 - d)
            P = P * torch.stack([wx, wy, wx, wy, wx, wy]).to(P.real.dtype)[:, None, None]
        kx = k0 * NA * self.px
        ky = k0 * NA * self.py
        return P, k0 * self.n_med * c_med, k0 * self.n_imm * c_imm, kx, ky

    def forward(self, pos, focus, zcoef_offset=None, intensity=None, bg=None, blur=None, dipw=None, dipM=None):
        """
        pos: [B, 3] (x, y relative to the ROI centre in px, z_e emitter height in um)
        focus: [B] (or a float) focal plane position in um relative to the coverslip
        zcoef_offset: optional [B, nz] per-emitter Zernike offsets (field dependence)
        blur / dipw / dipM: override the attributes of the same name for this call
        Returns [B, roisize, roisize]: the normalized PSF (times intensity, plus bg, when given).
        """
        B = pos.shape[0]
        P, kz_med, kz_imm, kx, ky = self.pupil_fields()
        Mt = dipole_tensor(dipM if dipM is not None else (None if dipw is not None else self.dipM),
                           dipw if dipw is not None else self.dipw, self.dev)
        zc = self.zcoef[None].expand(B, -1)
        if zcoef_offset is not None:
            zc = zc + zcoef_offset
        W = (torch.einsum('bj,jnm->bnm', zc, self.Z) + self.pupil_phase[None]) * (2 * math.pi)  # aberration phase
        x = pos[:, 0, None, None] * self.pixel_um
        y = pos[:, 1, None, None] * self.pixel_um
        ze = pos[:, 2, None, None]
        f = focus[:, None, None] if torch.is_tensor(focus) and focus.dim() > 0 else focus
        # complex phase; kz_med complex -> evanescent decay for z_e > 0
        ph = torch.exp(1j * (W - kx * x - ky * y - kz_imm * f)) * torch.exp(1j * kz_med * ze)
        Pb = P[None] * ph[:, None]  # [B, 6, N, N]
        # matrix Fourier transform: E(u, v) = sum_k P(kx, ky) exp(i (kx u + ky v))
        kx1 = kx[0]  # [N]
        ky1 = ky[:, 0]
        bl = blur if blur is not None else self.blur
        u, glw, pad = self._image_grid(bl)   # blur / 'spectral': a grid padded by `pad` px per side
        Fx = torch.exp(1j * u[:, None] * kx1[None, :])  # [ns, N]
        Fy = torch.exp(1j * u[:, None] * ky1[None, :])
        E = Fy @ Pb.to(torch.complex64) @ Fx.T  # [B, 6, ns, ns]
        if Mt is None:
            I = (E.real ** 2 + E.imag ** 2).sum(1) / 3
        else:   # sum_ij M_ij Re[E_i . E_j*], E_i = (Ex, Ey) of dipole i = channels (2i, 2i + 1)
            Ed = E.reshape(B, 3, 2, *E.shape[-2:])
            Mc = Mt.to(Ed.real.dtype)
            F = torch.einsum('ij,bjcuv->bicuv', Mc, Ed.real) + 1j * torch.einsum('ij,bjcuv->bicuv', Mc, Ed.imag) \
                if Mc.dim() == 2 else torch.einsum('bij,bjcuv->bicuv', Mc, Ed.real) + 1j * torch.einsum('bij,bjcuv->bicuv', Mc, Ed.imag)
            I = (Ed.real * F.real + Ed.imag * F.imag).sum((1, 2))
        # normalization: Parseval energy of an aberration-free emitter at z = 0 -> sum over the plane = 1
        dk = (kx1[1] - kx1[0]) * (ky1[1] - ky1[0])
        energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        # continuous field E = sum(...) dk / (2 pi)^2, so |E|^2 gets that factor squared
        I = I * (dk / (2 * math.pi) ** 2) ** 2
        R, o = self.roisize, self.os
        if self.pixel_integration == 'spectral':     # exact pixel mean (and blur) in the Fourier domain
            I = pixel_box_spectral(I, o, pad, R, self.band_limit(), bl) * self.pixel_um ** 2
        elif self.pixel_integration == 'gauss_legendre':
            I = torch.einsum('biajn,a,n->bij', I.reshape(B, R, o, R, o), glw, glw) * self.pixel_um ** 2
        else:
            if bl is not None:
                I = gauss_blur_os(I, bl, self.os, pad)
            # pixel integration: average over the oversampled subpixels times the pixel area
            I = I.reshape(B, R, o, R, o).mean((2, 4)) * self.pixel_um ** 2
        psf = I / energy
        if intensity is not None:
            psf = psf * intensity[:, None, None]
        if bg is not None:
            psf = psf + bg[:, None, None]
        return psf

    @torch.no_grad()
    def forward_batched(self, pos, focus, nb=400):
        """forward() without gradient in batches of nb emitters (large simulations). focus: [N] or float."""
        out = []
        for s in range(0, len(pos), nb):
            f = focus[s:s + nb] if torch.is_tensor(focus) and focus.dim() > 0 else focus
            out.append(self(pos[s:s + nb], f))
        return torch.cat(out) if out else torch.zeros(0, self.roisize, self.roisize, device=self.dev)

    # ------------------------------------------------------------------ common PSF-model interface
    @property
    def xy_limit(self):
        """Largest |x|, |y| offset (px) from the ROI centre that the fitter allows."""
        return self.roisize / 2 - 1.0

    def render(self, pos, grad=False, focus=None, dipM=None):
        """Normalized PSF at pos [N, 3] = (x px, y px, z um) at the focus self.focus_um (or `focus`) -> [N, P, P].
        grad=True also returns d/dx, d/dy (per px) and d/dz (per um), by forward-mode automatic differentiation."""
        f = self.focus_um if focus is None else focus
        if not grad:
            return self(pos, f, dipM=dipM)
        fn = lambda p: self(p, f, dipM=dipM)
        out = []
        val = None
        for k in range(3):
            t = torch.zeros_like(pos)
            t[:, k] = 1
            v, d = torch.func.jvp(fn, (pos,), (t,))
            val = v if val is None else val
            out.append(d)
        return val, out[0], out[1], out[2]

    def to_spline(self, roisize=None, z_range=None, oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0, chunk=8,
                  pos_fn=None, dipole_components=False):
        """CubicSplinePSF of this model (see PSF.to_spline). pos_fn: knot z -> (emitter z, focus) (default: the
        emitter height at self.focus_um; e.g. spline.stage_scan_pos_fn for a bead-stack geometry).
        dipole_components=True also tabulates xx, yy, zz, xy, xz, yz for per-emitter dipM rendering."""
        zr = self.z_range if z_range is None else z_range
        spl = CubicSplinePSF(roisize or self.roisize, zr, oversample=oversample, dz_um=dz_um, zpad=zpad,
                                 xy_limit=xy_limit, device=self.dev, pixel_um=self.pixel_um)
        big = self.resized(spl.Pl) if self.roisize != spl.Pl or dipole_components else self
        if dipole_components:
            from torch_psf._dipole import default_weights, unit_tensors
            tables = []
            for M in unit_tensors(big.dev):
                big.dipM = M
                tables.append(spl.knots_from_vectorial(big, pos_fn, chunk=chunk))
            spl._set_dipole_knots(torch.stack(tables), default_weights(self.dipM, self.dipw))
        else:
            spl.knots = spl.knots_from_vectorial(big, pos_fn, chunk=chunk)
        return spl

    def _apply(self, fn, *args, **kwargs):
        """torch.nn.Module semantics for .to(device / dtype), .cuda(), .cpu(), .float(): parameters and buffers are
        converted in place; the optional plain tensor attributes (blur, dipM, dipw, diat) and self.dev follow. Returns
        self. (A copy of another size: resized().)"""
        super()._apply(fn, *args, **kwargs)
        for k in ('blur', 'dipM', 'dipw', 'diat', 'dipM_angle', 'blur_angle'):
            v = getattr(self, k, None)
            if torch.is_tensor(v):
                setattr(self, k, fn(v))
        self.dev = self.u.device
        return self

    def __repr__(self):
        return (f'VectorialPSF(roisize={self.roisize}, pixel_um={self.pixel_um}, NA={float(self.NA):.3f}, '
                f'n_imm={self.n_imm}, n_med={float(self.n_med):.3f}, wavelength_um={self.wavelength_um}, '
                f'npupil={self.npupil}, oversample={self.os}, focus_um={self.focus_um}'
                + (f', pixel_integration={self.pixel_integration!r}' if self.pixel_integration != 'box' else '') + ')')

    # ------------------------------------------------------------------ files
    def _file_dict(self, source=None):
        o = self.optics
        d = dict(optics=o.to_dict(), zernike_noll=list(self.zidx), zernike_waves=_list(self.zcoef),
                 apod=_list(self.apod), saf=_list(self.saf), focus_um=self.focus_um, psf_blur_px2=_list(self.blur),
                 psf_dipole_M=_list(self.dipM), diattenuation=None if self.diat is None else float(self.diat),
                 npupil=self.npupil, oversample=self.os, roisize=self.roisize,
                 z_range_um=None if self.z_range is None else list(self.z_range))
        if self.dipw is not None:
            d['psf_dipole_w'] = _list(self.dipw)
        if self.pixel_integration != 'box':
            d['pixel_integration'] = self.pixel_integration
        if getattr(self, 'dipM_angle', None) is not None:
            d['psf_dipole_M_angle'] = _list(self.dipM_angle)
        if getattr(self, 'blur_angle', None) is not None:
            d['psf_blur_angle_px2'] = _list(self.blur_angle)
        d.update(pixelsize_um=o.pixel_um, wavelength_um=o.wavelength_um, NA=o.NA, n_imm=o.n_imm, n_med=o.n_med)
        arrays = {}
        if bool((self.pupil_phase != 0).any()):
            arrays['pupil_phase'] = self.pupil_phase.detach().cpu().numpy()
        if bool((self.pupil_amp != 1).any()):
            arrays['pupil_amp'] = self.pupil_amp.detach().cpu().numpy()
        return d, arrays

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, angle=None, npupil=None, oversample=None,
                   pixel_integration=None, **kw):
        """From a torch-psf json or a plain calibration json (the optics and Zernikes at the top level; optional terms
        psf_blur_px2, psf_dipole_M, psf_dipole_w, diattenuation, and per illumination angle psf_dipole_M_angle /
        psf_blur_angle_px2: angle 0 / 1 = that angle, None = the average)."""
        o = Optics.from_dict(d['optics'] if 'optics' in d else d)
        zr = d.get('z_range_um')
        pixint = pixel_integration if pixel_integration is not None else d.get('pixel_integration')
        try:
            pixint = pixel_integration_mode(pixint)
        except ValueError as e:
            raise ValueError(f'PSF file: {e}') from None
        vec = cls(roisize or d.get('roisize', 15), zernike_indices=tuple(d.get('zernike_noll', range(5, 22))),
                  npupil=npupil or d.get('npupil', 64), oversample=oversample or d.get('oversample', 3), device=device,
                  focus_um=float(d.get('focus_um', 0.0) or 0.0), z_range=tuple(zr) if zr else (0.0, 1.0),
                  pixel_integration=pixint, **o.vectorial_kwargs())
        t = lambda v: torch.tensor(v, device=vec.dev, dtype=torch.float32)
        with torch.no_grad():
            if d.get('zernike_waves') is not None:
                vec.zcoef.copy_(t(d['zernike_waves']))
            if d.get('apod') is not None:
                vec.apod.copy_(t(d['apod']))
            if d.get('saf') is not None:
                vec.saf.copy_(t(d['saf']))
            if arrays is not None:
                if 'pupil_phase' in arrays:
                    vec.pupil_phase.copy_(t(arrays['pupil_phase']))
                if 'pupil_amp' in arrays:
                    vec.pupil_amp.copy_(t(arrays['pupil_amp']))
        if d.get('psf_blur_px2') is not None:
            vec.blur = t([float(v) for v in d['psf_blur_px2']])
        if d.get('psf_dipole_M') is not None:
            vec.dipM = t(d['psf_dipole_M'])
        if d.get('psf_dipole_w') is not None:
            vec.dipw = t(d['psf_dipole_w'])
        if d.get('psf_dipole_M_angle') is not None:
            M = vec.dipM_angle = t(d['psf_dipole_M_angle'])
            vec.dipM = M.mean(0) if angle is None else M[_angle_index(angle, len(M))]
        if d.get('psf_blur_angle_px2') is not None:
            Bc = vec.blur_angle = t(d['psf_blur_angle_px2'])
            vec.blur = Bc.mean(0) if angle is None else Bc[_angle_index(angle, len(Bc))]
        if d.get('diattenuation') is not None:
            vec.diat = float(d['diattenuation'])
        return vec

    # ------------------------------------------------------------------ copies
    @classmethod
    def from_optics(cls, optics, roisize, **kw):
        """VectorialPSF with the optics of an Optics object (other arguments as the constructor)."""
        return cls(roisize, **optics.vectorial_kwargs(), **kw)

    @property
    def optics(self):
        return Optics(pixel_um=self.pixel_um, NA=float(self.NA), wavelength_um=self.wavelength_um, n_imm=self.n_imm,
                      n_med=float(self.n_med))

    def resized(self, roisize, npupil=None, oversample=None, device=None, pixel_integration=None):
        """A copy with another ROI size (and optionally pupil sampling / oversampling / device / pixel integration):
        same optics, Zernikes, apodization, SAF, pupil maps (resampled if npupil changes), optional terms and (default)
        pixel integration."""
        npupil = self.npupil if npupil is None else npupil
        new = VectorialPSF(roisize, pixel_um=self.pixel_um, NA=float(self.NA), n_imm=self.n_imm,
                           n_med=float(self.n_med), wavelength_um=self.wavelength_um, zernike_indices=self.zidx,
                           npupil=npupil, oversample=self.os if oversample is None else oversample,
                           device=self.dev if device is None else device, focus_um=self.focus_um, z_range=self.z_range,
                           pixel_integration=self.pixel_integration if pixel_integration is None else pixel_integration)
        new.NA.requires_grad_(self.NA.requires_grad)
        new.n_med.requires_grad_(self.n_med.requires_grad)
        with torch.no_grad():
            new.zcoef.copy_(self.zcoef); new.apod.copy_(self.apod); new.saf.copy_(self.saf)
            if npupil == self.npupil:
                new.pupil_phase.copy_(self.pupil_phase); new.pupil_amp.copy_(self.pupil_amp)
            else:
                rs = lambda m: torch.nn.functional.interpolate(m[None, None], size=(npupil, npupil), mode='bilinear',
                                                               align_corners=False)[0, 0]
                new.pupil_phase.copy_(rs(self.pupil_phase) * new.mask)
                new.pupil_amp.copy_(rs(self.pupil_amp))
        for k in ('blur', 'diat', 'dipw', 'dipM', 'dipM_angle', 'blur_angle'):
            v = getattr(self, k, None)
            if v is not None:
                setattr(new, k, v.detach().clone().to(new.dev) if torch.is_tensor(v) else copy.deepcopy(v))
        return new

    def set_zernike(self, coefs):
        """Set Zernike coefficients from a dict {noll: waves} (others unchanged) or a full vector [nz]."""
        with torch.no_grad():
            if isinstance(coefs, dict):
                for j, v in coefs.items():
                    self.zcoef[self.zidx.index(int(j))] = float(v)
            else:
                self.zcoef.copy_(torch.as_tensor(coefs, dtype=torch.float32, device=self.dev))
        return self
