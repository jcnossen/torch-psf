"""Pixel-integrated Gaussian PSF: 2D (fixed width), or astigmatic 3D with the widths
sigma_x(z) = sigma_x0 sqrt(1 + ((z - gamma) / d)^2), sigma_y(z) = sigma_y0 sqrt(1 + ((z + gamma) / d)^2)
(Huang et al., Science 2008, without the higher-order terms)."""
import math
import torch
from torch_psf.config import resolve_device


class GaussianPSF:
    """roisize x roisize patches; sigma_px: width (px; a float or (sx, sy)) in focus. 2D when gamma_um is None (z
    ignored, z_range None). gamma_um, d_um: focal offset of the x / y foci and depth of focus of the astigmatic 3D
    model (valid over z_range, um). device: default cuda if available."""

    def __init__(self, roisize, sigma_px=1.3, gamma_um=None, d_um=0.4, z_range=(-0.5, 0.5), device=None):
        self.roisize = roisize
        self.dev = resolve_device(device)
        s = (sigma_px, sigma_px) if not isinstance(sigma_px, (tuple, list)) else sigma_px
        self.sigma = (float(s[0]), float(s[1]))
        self.gamma = None if gamma_um is None else float(gamma_um)
        self.d = float(d_um)
        self.z_range = None if gamma_um is None else (float(z_range[0]), float(z_range[1]))

    @property
    def xy_limit(self):
        return self.roisize / 2 - 1.0

    @property
    def z_limits(self):
        return self.z_range

    def __repr__(self):
        return f'GaussianPSF(roisize={self.roisize}, sigma_px={self.sigma}, gamma_um={self.gamma}, d_um={self.d})'

    def widths(self, z):
        """sigma_x, sigma_y (px) and their z derivatives at z [N] (um)."""
        sx0, sy0 = self.sigma
        if self.gamma is None:
            one = torch.ones_like(z)
            return sx0 * one, sy0 * one, 0 * one, 0 * one
        ax, ay = (z - self.gamma) / self.d, (z + self.gamma) / self.d
        qx, qy = torch.sqrt(1 + ax ** 2), torch.sqrt(1 + ay ** 2)
        return sx0 * qx, sy0 * qy, sx0 * ax / (qx * self.d), sy0 * ay / (qy * self.d)

    def render(self, pos, grad=False):
        """pos [N, 3] (x, y px from the ROI centre, z um) -> [N, P, P] (sum 1 over the whole plane); grad: + d/dx, d/dy,
        d/dz (d/dz = 0 for the 2D model)."""
        pos = pos.float()
        o = torch.arange(self.roisize, device=pos.device, dtype=torch.float32) - (self.roisize - 1) / 2
        sx, sy, dsx, dsy = self.widths(pos[:, 2])
        r2 = math.sqrt(2)

        def axis(c, s):
            a = (o[None] + 0.5 - c[:, None]) / (s[:, None] * r2)
            b = (o[None] - 0.5 - c[:, None]) / (s[:, None] * r2)
            v = 0.5 * (torch.erf(a) - torch.erf(b))
            ga, gb = torch.exp(-a ** 2), torch.exp(-b ** 2)
            k = 1 / math.sqrt(math.pi)
            dc = -k * (ga - gb) / (s[:, None] * r2)               # d v / d c
            ds = -k * (ga * a - gb * b) / s[:, None]              # d v / d s
            return v, dc, ds

        ex, dex, dsx_ = axis(pos[:, 0], sx)
        ey, dey, dsy_ = axis(pos[:, 1], sy)
        g = ey[:, :, None] * ex[:, None, :]
        if not grad:
            return g
        gx = ey[:, :, None] * dex[:, None, :]
        gy = dey[:, :, None] * ex[:, None, :]
        gz = (ey[:, :, None] * (dsx_ * dsx[:, None])[:, None, :] + (dsy_ * dsy[:, None])[:, :, None] * ex[:, None, :])
        return g, gx, gy, gz

    def __call__(self, pos, intensity=None, bg=None):
        out = self.render(pos)
        if intensity is not None:
            out = out * intensity[:, None, None]
        if bg is not None:
            out = out + bg[:, None, None]
        return out
