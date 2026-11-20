"""Optical compositions. Wavelength and axial stage views share the underlying optics."""
from dataclasses import dataclass
import math

import torch

from torch_psf.psf import PSF, render_with_grad
from torch_psf.multichannel import ChannelsPSF, MultiChannelPSF


class _OpticalWrapper(PSF):
    roisize = property(lambda self: self.base.roisize)
    dtype = property(lambda self: self.base.dtype)
    pixel_um = property(lambda self: self.base.pixel_um)
    dev = property(lambda self: self.base.dev)
    xy_limit = property(lambda self: self.base.xy_limit)
    zidx = property(lambda self: self.base.zidx)
    zcoef = property(lambda self: self.base.zcoef)
    focus_um = property(lambda self: self.base.focus_um)
    pixel_integration = property(lambda self: self.base.pixel_integration)
    dipM = property(lambda self: self.base.dipM)
    dipw = property(lambda self: self.base.dipw)
    uses_field_xy = property(lambda self: getattr(self.base, "uses_field_xy", False))
    vortex = property(lambda self: getattr(self.base, 'vortex', None))

    def _with_base(self, base):
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        view.base = base
        return view

    def with_field_offset(self, offset):
        return self._with_base(self.base.with_field_offset(offset))

    def resized(self, roisize):
        return self._with_base(self.base.resized(roisize))

    def to(self, device=None, dtype=None):
        self.base.to(device=device, dtype=dtype)
        return self

    def __call__(self, pos, focus=None, **options):
        return self.render(pos, focus=focus, **options)


@dataclass(frozen=True)
class _AxialCoordinate:
    mode: str
    height: float = 0.
    scale: float = 1.

    def map(self, z, focus, depth=None):
        """Map axial displacement to emitter height and focus. depth is the calibrated arm depth."""
        if self.mode == 'stage':
            height = torch.full_like(z, self.height) if depth is None else depth.clamp_min(0)
            return height, focus - z / self.scale
        height = z if depth is None else (depth + z).clamp_min(0)
        return height, torch.as_tensor(focus, device=z.device, dtype=z.dtype).expand_as(z)


class ZMode(_OpticalWrapper):
    """Interpret z as emitter height or stage defocus. Stage focus = focus0 - z / scale.

    Coherent 4Pi arms use their signed arm coordinates and calibrated coverslip depths.
    """
    def __new__(cls, base, mode='stage', *, height=0., scale=1., focus=None, z_range=None):
        if mode not in ('emitter', 'stage') or not math.isfinite(scale) or scale == 0:
            raise ValueError("ZMode needs mode 'emitter' or 'stage' and a finite nonzero scale")
        if hasattr(base, '_with_zmode'):
            return base._with_zmode(mode, height, scale, focus, z_range)
        if isinstance(base, ChannelsPSF):
            return base.map_models(lambda m: cls(m, mode, height=height, scale=scale, focus=focus, z_range=z_range))
        return super().__new__(cls)

    def __init__(self, base, mode='stage', *, height=0., scale=1., focus=None, z_range=None):
        self.base, self.focus = base, focus
        self.axial = _AxialCoordinate(mode, float(height), float(scale))
        self.z_range = base.z_range if z_range is None else tuple(z_range)

    def coordinates(self, z, focus=None):
        f = self.focus if focus is None else focus
        f = self.focus_um if f is None else f
        return self.axial.map(z, f)

    def _mapped(self, fn, pos, focus):
        z, f = self.coordinates(pos[:, 2], focus)
        return fn(torch.cat((pos[:, :2], z[:, None]), 1), f)

    def _render(self, pos, grad, *, focus=None, **options):
        fn = lambda p, f: self.base.render(p, focus=f, **options)
        return render_with_grad(lambda p: self._mapped(fn, p, focus), pos, grad)

    def _dipole_basis(self, pos, grad, *, focus=None, **options):
        fn = lambda p, f: self.base.dipole_basis(p, focus=f, **options)
        return render_with_grad(lambda p: self._mapped(fn, p, focus), pos, grad)

    def basis(self, pos, grad=False, *, focus=None, **options):
        fn = lambda p, f: self.base.basis(p, focus=f, **options)
        return render_with_grad(lambda p: self._mapped(fn, p, focus), pos, grad)

    def weights(self, pos, *, focus=None, **options):
        return self._mapped(lambda p, f: self.base.weights(p, focus=f, **options), pos, focus)

    def at_wavelength(self, wavelength):
        return self._with_base(self.base.at_wavelength(wavelength))

    def _grid_sampler(self, grid, pos_fn, params, focus, blur, dipM):
        if pos_fn is not None:
            raise ValueError('spline grid already has a ZMode')
        return grid._sampler(self.base, lambda z: self.coordinates(z, focus), params, focus, blur, dipM)


class Polychromatic(_OpticalWrapper):
    """Spectrum-weighted intensities. Each wavelength uses its own normalized pupil and pixel integrator."""

    def __new__(cls, base, wavelengths_um, weights=None):
        if isinstance(base, ChannelsPSF):
            return base.map_models(lambda m: cls(m, wavelengths_um, weights))
        if isinstance(base, MultiChannelPSF):
            return object.__new__(_CoupledSpectrum)
        return super().__new__(cls)

    def __init__(self, base, wavelengths_um, weights=None):
        if not hasattr(base, 'at_wavelength'):
            raise TypeError(f'{type(base).__name__} has no wavelength-dependent optical stages')
        self.base = base
        self.wavelengths = tuple(float(v) for v in wavelengths_um)
        w = torch.ones(len(self.wavelengths)) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        if (not self.wavelengths or not all(v > 0 and math.isfinite(v) for v in self.wavelengths)
                or w.shape != (len(self.wavelengths),) or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0):
            raise ValueError('expected positive wavelengths and finite nonnegative weights with positive sum')
        self.fractions = tuple((w / w.sum()).tolist())
        self.z_range = base.z_range

    def _average(self, fn):
        values = (fn(self.base.at_wavelength(w)) for w in self.wavelengths)
        return self._sum(values)

    def _sum(self, values):
        out = None
        for weight, v in zip(self.fractions, values):
            if isinstance(v, tuple):
                out = tuple(x * weight for x in v) if out is None else tuple(a + x * weight for a, x in zip(out, v))
            else:
                out = v * weight if out is None else out + v * weight
        return out

    def _render(self, pos, grad, **options):
        return self._average(lambda m: m.render(pos, grad, **options))

    def _dipole_basis(self, pos, grad, **options):
        return self._average(lambda m: m.dipole_basis(pos, grad, **options))

    def basis(self, pos, grad=False, **options):
        return self._average(lambda m: m.basis(pos, grad, **options))

    def weights(self, pos, **options):
        return self.base.weights(pos, **options)

    def _grid_sampler(self, grid, pos_fn, params, focus, blur, dipM):
        samplers = [grid._sampler(self.base.at_wavelength(w), pos_fn, params, focus, blur, dipM)
                    for w in self.wavelengths]
        return lambda p: self._sum(sample(p) for sample in samplers)


class _CoupledSpectrum(Polychromatic, MultiChannelPSF):
    n_channels = property(lambda self: self.base.n_channels)
    extra_params = property(lambda self: self.base.extra_params)
    channel_offsets = property(lambda self: self.base.channel_offsets)
    intensity_ratio = property(lambda self: self.base.intensity_ratio)
    mappings = property(lambda self: self.base.mappings)

    def render_channels(self, pos_c, extra=None, grad=False, **options):
        return self._average(lambda m: m.render_channels(pos_c, extra, grad, **options))

    def default_extra_starts(self):
        return self.base.default_extra_starts()

    def wrap_extra(self, theta):
        return self.base.wrap_extra(theta)

    def _with_zmode(self, mode, height, scale, focus, z_range):
        base = ZMode(self.base, mode, height=height, scale=scale, focus=focus, z_range=z_range)
        view = self._with_base(base)
        view.z_range = base.z_range
        return view

    def _sampling_model(self, size, density):
        # Configure each wavelength's point integrator at its own optical bandwidth.
        models = [self.base.at_wavelength(w)._sampling_model(size, density) for w in self.wavelengths]
        view = self._with_base(self.base)
        view._sampling_models = models
        return view

    def _channel_images(self, pos, **options):
        return sum(w * m._channel_images(pos, **options) for w, m in zip(self.fractions, self._sampling_models))

    tau = property(lambda self: self.base.tau)

    def state(self):
        # A monochromatic source block cannot regenerate a spectrum-averaged table.
        return None
