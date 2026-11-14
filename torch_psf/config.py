"""Shared configuration objects: the optics and the camera of an acquisition.

A pipeline configuration with `optics` and `camera` blocks of the same field names maps 1:1 onto these classes:
    Optics.from_dict(cfg['optics']), Camera.from_dict(cfg['camera']).
"""
import dataclasses
import json
import os
from typing import Optional, Union
import numpy as np
import torch

Number = Union[float, np.ndarray, torch.Tensor]


def default_device():
    """'cuda' if a GPU is available, else 'cpu' (the default device of every torch_psf function)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def resolve_device(device):
    return torch.device(default_device() if device is None else device)


@dataclasses.dataclass
class Optics:
    """Detection optics. pixel_um: back-projected camera pixel size (um), NA, wavelength_um: emission wavelength (um),
    n_imm: immersion (= coverslip) index, n_med: sample medium index."""
    pixel_um: float = 0.10833
    NA: float = 1.49
    wavelength_um: float = 0.68
    n_imm: float = 1.518
    n_med: float = 1.33

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        """From an Optics dict, an optics config block (same names; extra keys are ignored), or a
        calib json (pixelsize_um / wavelength_um / NA / n_imm / n_med at the top level)."""
        d = dict(d)
        if 'pixel_um' not in d and 'pixelsize_um' in d:
            d['pixel_um'] = d['pixelsize_um']
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})

    def vectorial_kwargs(self):
        """Keyword arguments for VectorialPSF(...)."""
        return dict(pixel_um=self.pixel_um, NA=self.NA, wavelength_um=self.wavelength_um, n_imm=self.n_imm,
                    n_med=self.n_med)


def _crop(m, corners, P, frame_shape=None):
    """Per-ROI crops [n, P, P] of a per-pixel map m [H, W] at corners [n, 2] = (x0, y0) (torch)."""
    ii = torch.arange(P, device=m.device)
    yy = (corners[:, 1, None, None] + ii[None, :, None]).clamp(0, m.shape[0] - 1)
    xx = (corners[:, 0, None, None] + ii[None, None, :]).clamp(0, m.shape[1] - 1)
    return m[yy, xx]


@dataclasses.dataclass
class Camera:
    """Camera conversion. gain_e_per_adu (e- per ADU), offset_adu (ADU), readnoise_e (e- rms): each a scalar or a
    per-pixel map [H, W] (numpy or torch). maps: optional npz path with per-pixel 'offset_adu' / 'readnoise_e' maps,
    loaded by load_maps(). type: 'scmos' or 'emccd' (a label only).

    excess_noise: the noise factor F^2 of the camera (1 = Poisson; an EMCCD has F^2 ~ 2: the EM gain doubles the
    variance of the photoelectron count). to_photons divides by it, so the fits, CRLBs and deviances work with
    Poisson-equivalent ('effective') photons, var = mean, and the photon / background numbers they report are
    effective photons = photoelectrons / F^2. For an EMCCD give the true gain (photoelectrons per ADU, including the
    EM gain) with excess_noise=2, or equivalently the Poisson-equivalent gain (gain / 2, = 1 / the photon-transfer
    slope) with excess_noise=1. torch_psf.calibrate.photon_transfer measures it from a movie."""
    gain_e_per_adu: Number = 1.0
    offset_adu: Number = 0.0
    readnoise_e: Number = 0.0
    type: str = 'scmos'
    maps: Optional[str] = None
    excess_noise: float = 1.0

    @classmethod
    def from_dict(cls, d, base_dir=None, load_maps=True):
        """From a Camera dict or the camera block of a config / calibration json. A relative 'maps' path is resolved
        against base_dir (the directory of the calib file)."""
        names = {f.name for f in dataclasses.fields(cls)}
        c = cls(**{k: v for k, v in d.items() if k in names and v is not None})
        if c.maps and base_dir is not None and not os.path.isabs(c.maps):
            c.maps = os.path.join(base_dir, c.maps)
        if load_maps and c.maps:
            c.load_maps()
        return c

    @classmethod
    def from_json(cls, path, load_maps=True):
        """Camera block of a calibration / torch-psf json (relative map paths resolved against the json's directory), or
        None when the file has none."""
        with open(path) as f:
            c = json.load(f).get('camera')
        if not c:
            return None
        return cls.from_dict(c, base_dir=os.path.dirname(os.path.abspath(str(path))), load_maps=load_maps)

    def to_dict(self):
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                v = '<map>' if np.ndim(v) else float(v)   # maps live in the npz
            out[f.name] = v
        return out

    def load_maps(self, path=None):
        """Replace scalar offset / read noise by the per-pixel maps of the npz (keys offset_adu, readnoise_e)."""
        p = path or self.maps
        M = np.load(p)
        if 'offset_adu' in M:
            self.offset_adu = M['offset_adu'].astype(np.float32)
        if 'readnoise_e' in M:
            self.readnoise_e = M['readnoise_e'].astype(np.float32)
        self.maps = p
        return self

    def _t(self, v, like, corners=None, P=None):
        t = torch.as_tensor(v, dtype=torch.float32, device=like.device)
        if t.dim() == 2 and corners is not None:
            return _crop(t, corners.long(), P)
        return t

    def to_photons(self, adu, corners=None):
        """(adu - offset) * gain for frames [T, H, W] (full frames: maps apply directly) or ROIs [n, P, P] with their
        corners [n, 2] = (x0, y0) (maps are cropped)."""
        adu = torch.as_tensor(adu)
        x = adu.float()
        P = x.shape[-1]
        ph = (x - self._t(self.offset_adu, x, corners, P)) * self._t(self.gain_e_per_adu, x, corners, P)
        return ph if self.excess_noise == 1 else ph / float(self.excess_noise)

    def variance(self, like, corners=None):
        """Read-noise variance in e^2 for frames or ROIs shaped like `like` (scalar tensor if the read noise is a
        scalar). Pass it as `var` to localize()."""
        P = like.shape[-1]
        rn = self._t(self.readnoise_e, like, corners, P)
        return rn ** 2 if self.excess_noise == 1 else rn ** 2 / float(self.excess_noise) ** 2
