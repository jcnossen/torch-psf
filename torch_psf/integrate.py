"""Image-plane sampling, Gaussian blur and pixel integration for optical PSFs."""
import math

import numpy as np
import torch

from torch_psf.optics import pixel_xy


MODES = ('midpoint', 'exact', 'gauss_legendre', 'point')
PAD = 3  # pixels on each side for circular convolution


def pixel_integration_mode(mode, *, from_file=False):
    """Validated pixel-integration name. from_file: also the names stored by earlier versions ('box', 'spectral',
    'gauss'); a file without the key is 'midpoint'. In the API None is the default 'exact', and 'box' / 'spectral'
    were renamed (no aliases)."""
    if mode is None:
        return 'midpoint' if from_file else 'exact'
    if isinstance(mode, str) and from_file:
        mode = {'box': 'midpoint', 'spectral': 'exact', 'gauss': 'gauss_legendre'}.get(mode, mode)
    elif mode in ('box', 'spectral'):
        raise ValueError(f"pixel_integration {mode!r} was renamed to {'midpoint' if mode == 'box' else 'exact'!r}")
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"unknown pixel_integration {mode!r}: one of {MODES}")
    return mode


def spectral_min_oversample(fc):
    """Smallest oversample >= 2 with Nyquist frequency strictly above fc (cycles/px)."""
    return max(2, math.floor(2 * fc) + 1)


def spectral_cutoff(model, mode, oversample):
    """Validate the current optical bandwidth, including when NA or wavelength changes."""
    if mode in ('exact', 'point'):
        fc = model.band_limit()
        minimum = spectral_min_oversample(fc)
        if oversample < minimum:
            raise ValueError(f"pixel_integration {mode!r}: alias-free sampling of the band limit "
                             f"{fc:.3f} cycles / px needs oversample >= {minimum} (or pixel_integration='midpoint')")
        return fc


def uniform_grid(size, oversample, pixel_um, device):
    n = size * oversample
    return (torch.arange(n, device=device) - (n - 1) / 2) / oversample * pixel_um


def _blur_transfer(cov, fx, fy):
    q = (cov[:, 0, None, None] * fx ** 2 + cov[:, 1, None, None] * fy ** 2
         + 2 * cov[:, 2, None, None] * fx * fy)
    return torch.exp((-2 * math.pi ** 2 * q).clamp(max=3.0))


def _pixel_transfer(f, fr, os_, fc, integrate):
    """Shift point samples onto pixel centres. For pixel means also apply the box
    transfer sinc(fx) sinc(fy). Taper above the optical band limit to limit wraparound."""
    fn = 0.5 * os_
    fc = min(float(fc) * 1.02, fn - 0.05)

    def taper(q):
        a = ((q.abs() - fc) / (fn - fc)).clamp(0, 1)
        return 0.5 * (1 + torch.cos(math.pi * a))

    d = (os_ - 1) / (2 * os_)                                          # sample p os_ -> pixel centre p: shift by d px
    Ty = taper(f) * (torch.sinc(f) if integrate else 1) * torch.exp(2j * math.pi * f * d)
    Tx = taper(fr) * (torch.sinc(fr) if integrate else 1) * torch.exp(2j * math.pi * fr * d)
    return Ty[:, None] * Tx[None, :]


class PixelIntegrator:
    """One render's image grid and its matching reduction. Input [N, ..., n, n] may
    include channel or dipole axes. Covariance in px^2 is shared [3] or per emitter [N, 3].
    FFT modes and blurred midpoint grids have a three-pixel border. Coherent spectral
    images need twice that margin because destructive interference can suppress the ROI peak.
    """

    def __init__(self, model, blur, *, mode=None, oversample=None, coherent=False):
        self.mode = model.pixel_integration if mode is None else pixel_integration_mode(mode)
        self.os = model.os if oversample is None else oversample
        if oversample is None and self.mode in ('exact', 'point'):
            # exact integration needs only alias-free point samples: raise the sampling grid to the minimum for the
            # band limit when the model's oversample is lower (the result does not depend on it beyond that)
            self.os = max(self.os, spectral_min_oversample(model.band_limit()))
        self.size = model.roisize
        self.fc = spectral_cutoff(model, self.mode, self.os)
        self.blur = blur
        self.weights = None
        self.pad = 0
        # image sample positions (um) along x (u) and y (uy; the same tensor for square pixels). Non-square pixels:
        # the grids follow the pixel sizes at every render (they may be tensors of a fitted aspect)
        square = not isinstance(model.pixel_um, tuple)
        px, py = pixel_xy(model.pixel_um)
        if self.mode == 'gauss_legendre':
            if blur is not None:
                raise ValueError("pixel_integration 'gauss_legendre' does not support an image-plane blur")
            nodes, weights = np.polynomial.legendre.leggauss(self.os)
            centres = np.arange(self.size) - (self.size - 1) / 2
            u = torch.tensor((centres[:, None] + nodes / 2).reshape(-1), dtype=torch.float32, device=model.dev)
            self.u = u * px
            self.uy = self.u if square else u * py
            self.weights = torch.tensor(weights / 2, dtype=model.u.dtype, device=model.dev)
        else:
            if self.fc is not None or blur is not None:
                self.pad = 2 * PAD if coherent and self.fc is not None else PAD
            if square and self.pad == 0 and self.os == model.os and hasattr(model, 'u'):
                self.u = model.u
            else:
                self.u = uniform_grid(self.size + 2 * self.pad, self.os, px, model.dev)
            self.uy = self.u if square else uniform_grid(self.size + 2 * self.pad, self.os, py, model.dev)

    def _filter(self, samples, batch):
        """Apply blur and the spectral pixel transfer together on the padded grid."""
        n = samples.shape[-1]
        f = torch.fft.fftfreq(n, d=1 / self.os, device=samples.device)
        fr = torch.fft.rfftfreq(n, d=1 / self.os, device=samples.device)
        transfer = None
        if self.fc is not None:
            transfer = _pixel_transfer(f, fr, self.os, self.fc, self.mode == 'exact')
        if self.blur is not None:
            cov = torch.as_tensor(self.blur, device=samples.device, dtype=torch.float32)
            if cov.shape not in ((3,), (1, 3), (batch[0], 3)):
                raise ValueError('blur covariance must have shape [3] or [N, 3]')
            cov = cov.reshape(-1, 3)
            if len(cov) > 1 and len(batch) > 1:
                cov = cov.repeat_interleave(math.prod(batch[1:]), dim=0)
            blur = _blur_transfer(cov, fr[None, :], f[:, None])
            transfer = blur if transfer is None else transfer * blur
        return torch.fft.irfft2(torch.fft.rfft2(samples) * transfer, s=(n, n))

    def __call__(self, intensity):
        """Reduce to pixel means, or sample the density at pixel centres in point mode."""
        batch = intensity.shape[:-2]
        samples = intensity.flatten(0, -3)
        if self.pad:
            samples = self._filter(samples, batch)
            border = self.pad * self.os
            samples = samples[:, border:-border, border:-border]
        R, o = self.size, self.os
        if self.fc is not None:
            pixels = samples[:, ::o, ::o]
        elif self.weights is not None:
            pixels = torch.einsum('biajn,a,n->bij', samples.reshape(-1, R, o, R, o), self.weights, self.weights)
        else:
            pixels = samples.reshape(-1, R, o, R, o).mean((2, 4))
        return pixels.reshape(*batch, R, R)
