"""Pixel-integrated Gaussian PSFs: GaussianPSF (2D, fixed width) and AstigmaticGaussianPSF (3D, the usual
astigmatic calibration curves sigma_x(z) = sx0 sqrt(1 + ((z - gamma) / d)^2), sigma_y(z) = sy0 sqrt(1 + ((z + gamma)
/ d)^2); Huang et al., Science 2008, without the higher-order terms). Both integrate the Gaussian over the pixels
and sum to 1 over the image plane."""
import math
import torch
from torch_psf.config import resolve_device
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
    def _widths(self, z):
        raise NotImplementedError

    def render(self, pos, grad=False):
        """pos [N, 3] (x, y px from the ROI centre, z um) -> [N, P, P]; grad: + d/dx, d/dy (per px), d/dz (per um)."""
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.dev)
        o = torch.arange(self.roisize, device=self.dev, dtype=torch.float32) - (self.roisize - 1) / 2
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

    def to(self, device):
        """In place (the model holds no tensors); returns self."""
        self.dev = torch.device(device)
        return self

    def resized(self, roisize):
        return GaussianPSF(roisize, self.sigma, self.dev, self.pixel_um)

    def __repr__(self):
        return f'GaussianPSF(roisize={self.roisize}, sigma_px={self.sigma})'


class AstigmaticGaussianPSF(_GaussianBase):
    """3D astigmatic Gaussian: in-focus widths sigma_px = (sx0, sy0) (px), the x and y foci at z = +gamma_um and
    -gamma_um, depth of focus d_um; valid over z_range (um). pixel_um: lateral pixel size (um), required by fit /
    localize (z is stored in px)."""

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

    def to(self, device):
        """In place (the model holds no tensors); returns self."""
        self.dev = torch.device(device)
        return self

    def resized(self, roisize):
        return AstigmaticGaussianPSF(roisize, self.sigma, self.gamma, self.d, self.z_range, self.dev, self.pixel_um)

    def __repr__(self):
        return (f'AstigmaticGaussianPSF(roisize={self.roisize}, sigma_px={self.sigma}, gamma_um={self.gamma}, '
                f'd_um={self.d}, z_range={self.z_range})')
