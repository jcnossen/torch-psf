"""Pixel-integrated Gaussian PSFs: GaussianPSF (2D, fixed width) and AstigmaticGaussianPSF (3D, the usual
astigmatic calibration curves sigma_x(z) = sx0 sqrt(1 + ((z - gamma) / d)^2), sigma_y(z) = sy0 sqrt(1 + ((z + gamma)
/ d)^2); Huang et al., Science 2008, without the higher-order terms). Both integrate the Gaussian over the pixels
and sum to 1 over the image plane."""
import math
import torch
from torch_psf.optics import resolve_device
from torch_psf.psf import PSF


def _axis(o, c, s):
    """pixel-integrated 1D Gaussian at centres c [N] with widths s [N] on pixel offsets o [P]: value, d/dc, d/ds."""
    r2 = math.sqrt(2)
    a = (o[None] + 0.5 - c[:, None]) / (s[:, None] * r2)
    b = (o[None] - 0.5 - c[:, None]) / (s[:, None] * r2)
    v = 0.5 * (torch.erf(a) - torch.erf(b))
    ga, gb = torch.exp(-a ** 2), torch.exp(-b ** 2)
    k = 1 / math.sqrt(math.pi)
    return v, -k * (ga - gb) / (s[:, None] * r2), -k * (ga * a - gb * b) / s[:, None]


class _GaussianBase(PSF):
    dtype = torch.float32

    def _widths(self, z):
        raise NotImplementedError

    def _render(self, pos, grad=False):
        """pos [N, 3] (x, y px from the ROI centre, z um) -> [N, P, P]; grad: + d/dx, d/dy (per px), d/dz (per um)."""
        pos = torch.as_tensor(pos, dtype=self.dtype, device=self.dev)
        o = torch.arange(self.roisize, device=self.dev, dtype=self.dtype) - (self.roisize - 1) / 2
        sx, sy, dsx, dsy = self._widths(pos[:, 2])
        ex, dex, dsx_ = _axis(o, pos[:, 0], sx)
        ey, dey, dsy_ = _axis(o, pos[:, 1], sy)
        g = ey[:, :, None] * ex[:, None, :]
        if not grad:
            return g
        gx = ey[:, :, None] * dex[:, None, :]
        gy = dey[:, :, None] * ex[:, None, :]
        gz = ey[:, :, None] * (dsx_ * dsx[:, None])[:, None, :] + (dsy_ * dsy[:, None])[:, :, None] * ex[:, None, :]
        return g, gx, gy, gz


class GaussianPSF(_GaussianBase):
    """2D pixel-integrated Gaussian: roisize x roisize ROIs, width sigma_px (px; a float or (sx, sy)). z is ignored
    (z_range None, n_dims 2)."""
    type_name = 'gaussian'

    def _file_dict(self, source=None):
        return dict(gaussian=dict(sigma_px=list(self.sigma), pixel_um=self.pixel_um), roisize=self.roisize), {}

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        if 'gaussian' not in d:            # plain calibration json: psf_gauss_sigma_um, pixelsize_um
            px = float(d['pixelsize_um'])
            return cls(roisize or 15, sigma_px=float(d['psf_gauss_sigma_um']) / px, device=device, pixel_um=px)
        return cls(roisize or d.get('roisize', 15), sigma_px=tuple(d['gaussian']['sigma_px']), device=device,
                   pixel_um=d['gaussian'].get('pixel_um'))

    def __init__(self, roisize, sigma_px=1.3, device=None, pixel_um=None):
        self.roisize = roisize
        self.pixel_um = None if pixel_um is None else float(pixel_um)
        self.dev = resolve_device(device)
        s = (sigma_px, sigma_px) if not isinstance(sigma_px, (tuple, list)) else sigma_px
        self.sigma = (float(s[0]), float(s[1]))
        self.z_range = None

    def _widths(self, z):
        one = torch.ones_like(z)
        return self.sigma[0] * one, self.sigma[1] * one, 0 * one, 0 * one

    def to(self, device=None, dtype=None):
        """In place (the model holds no tensors); returns self."""
        self.dev = self.dev if device is None else torch.device(device)
        self.dtype = dtype or self.dtype
        return self

    def resized(self, roisize):
        return GaussianPSF(roisize, self.sigma, self.dev, self.pixel_um).to(dtype=self.dtype)

    def __repr__(self):
        return f'GaussianPSF(roisize={self.roisize}, sigma_px={self.sigma})'


class AstigmaticGaussianPSF(_GaussianBase):
    """3D astigmatic Gaussian: in-focus widths sigma_px = (sx0, sy0) (px), the x and y foci at z = +gamma_um and
    -gamma_um, depth of focus d_um; valid over z_range (um). pixel_um: lateral pixel size (um), required by fit /
    localize (z is stored in px)."""
    type_name = 'astig_gaussian'

    def _file_dict(self, source=None):
        return dict(gaussian=dict(sigma_px=list(self.sigma), gamma_um=self.gamma, d_um=self.d,
                                  z_range_um=list(self.z_range), pixel_um=self.pixel_um), roisize=self.roisize), {}

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        g = d['gaussian']
        return cls(roisize or d.get('roisize', 15), sigma_px=tuple(g['sigma_px']), gamma_um=g['gamma_um'],
                   d_um=g['d_um'], z_range=tuple(g['z_range_um']), device=device, pixel_um=g.get('pixel_um'))

    def __init__(self, roisize, sigma_px=(1.3, 1.3), gamma_um=0.2, d_um=0.4, z_range=(-0.5, 0.5), device=None,
                 pixel_um=None):
        self.roisize = roisize
        self.pixel_um = None if pixel_um is None else float(pixel_um)
        self.dev = resolve_device(device)
        s = (sigma_px, sigma_px) if not isinstance(sigma_px, (tuple, list)) else sigma_px
        self.sigma = (float(s[0]), float(s[1]))
        self.gamma, self.d = float(gamma_um), float(d_um)
        self.z_range = (float(z_range[0]), float(z_range[1]))

    def _widths(self, z):
        """sigma_x, sigma_y (px) and their z derivatives at z [N] (um)."""
        sx0, sy0 = self.sigma
        ax, ay = (z - self.gamma) / self.d, (z + self.gamma) / self.d
        qx, qy = torch.sqrt(1 + ax ** 2), torch.sqrt(1 + ay ** 2)
        return sx0 * qx, sy0 * qy, sx0 * ax / (qx * self.d), sy0 * ay / (qy * self.d)

    def to(self, device=None, dtype=None):
        """In place (the model holds no tensors); returns self."""
        self.dev = self.dev if device is None else torch.device(device)
        self.dtype = dtype or self.dtype
        return self

    def resized(self, roisize):
        return AstigmaticGaussianPSF(roisize, self.sigma, self.gamma, self.d, self.z_range, self.dev, self.pixel_um).to(dtype=self.dtype)

    def __repr__(self):
        return (f'AstigmaticGaussianPSF(roisize={self.roisize}, sigma_px={self.sigma}, gamma_um={self.gamma}, '
                f'd_um={self.d}, z_range={self.z_range})')


@torch.no_grad()
def fit_gaussian_free_sigma(rois, sigma0, sigma_limits=(0.5, 2.5), iters=30, min_photons=20.0, min_bg=0.05):
    """Spot fit with a free width: maximum-likelihood (Poisson) x, y (px from the ROI centre), photons, bg (photons /
    px) and sigma (px) of a pixel-integrated isotropic Gaussian on each ROI [N, P, P] (photons), by fit.lm_poisson
    (Levenberg-Marquardt, `iters` iterations). Start: x = y = 0, sigma = sigma0, bg = the 20 % quantile of the ROI (>=
    min_bg), photons = the sum above it (>= min_photons). Limits: |x|, |y| <= (P - 1) / 2 - 1, photons >= 1, bg >= 0.01,
    sigma within sigma_limits x sigma0. For spot finding and quality filters (width, chi2) on frames or frame sums.
    Returns (theta [N, 5] = x, y, photons, bg, sigma; crlb [N, 5] (sqrt of the diagonal of the pseudo-inverse of the
    Fisher matrix); chi2 [N] = sum((d - mu)^2 / mu) / P^2)."""
    from torch_psf.fit import lm_poisson
    d = torch.as_tensor(rois, dtype=torch.float32)
    N, P = d.shape[0], d.shape[-1]
    dev = d.device
    o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
    h = (P - 1) / 2

    def model(th):
        x, y, I, bg, s = th.unbind(1)
        ex, dex, dsx = _axis(o, x, s)
        ey, dey, dsy = _axis(o, y, s)
        g = ey[:, :, None] * ex[:, None, :]
        Ib = I[:, None, None]
        J = torch.stack([Ib * ey[:, :, None] * dex[:, None, :], Ib * dey[:, :, None] * ex[:, None, :], g,
                         torch.ones_like(g), Ib * (ey[:, :, None] * dsx[:, None, :] + dsy[:, :, None] * ex[:, None, :])], 1)
        return Ib * g + bg[:, None, None], J

    lo, hi = sigma_limits[0] * sigma0, sigma_limits[1] * sigma0

    def clamp(th):
        th[:, 0:2].clamp_(-h + 1, h - 1)
        th[:, 2].clamp_(min=1)
        th[:, 3].clamp_(min=1e-2)
        th[:, 4].clamp_(lo, hi)
        return th

    bg0 = d.flatten(1).quantile(0.2, 1).clamp(min=min_bg) if N else torch.zeros(0, device=dev)
    I0 = (d - bg0[:, None, None]).sum((1, 2)).clamp(min=min_photons)
    th0 = torch.stack([torch.zeros_like(I0), torch.zeros_like(I0), I0, bg0, torch.full_like(I0, float(sigma0))], 1)
    th, _ = lm_poisson(model, d, th0, clamp=clamp, iters=iters, mu_min=1e-3)
    mu, J = model(th)
    mu = mu.clamp(min=1e-3).flatten(1)
    Jf = J.flatten(2)
    H = torch.einsum('nap,nbp->nab', Jf / mu[:, None], Jf)
    crlb = torch.diagonal(torch.linalg.pinv(H), dim1=1, dim2=2).clamp(min=0).sqrt()
    chi2 = ((d.flatten(1) - mu) ** 2 / mu).sum(1) / (P * P)
    return th, crlb, chi2
