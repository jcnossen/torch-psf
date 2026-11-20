"""The detection optics of an acquisition (Optics) and the default device of every torch_psf function.

A pipeline configuration with an `optics` block of the same field names maps 1:1 onto Optics:
    Optics.from_dict(cfg['optics']).
"""
import dataclasses
import torch


def default_device():
    """'cuda' if a GPU is available, else 'cpu' (the default device of every torch_psf function)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def resolve_device(device):
    return torch.device(default_device() if device is None else device)


def pixel_size(v, *, floats=False):
    """A pixel size (um) as torch_psf keeps it: a float for square pixels, a pair (x, y) for non-square ones (a pair
    of equal values becomes a float). Tensors in a pair are kept (a fitted aspect) unless floats=True."""
    if v is None:
        return None
    if isinstance(v, (tuple, list)) or (hasattr(v, 'ndim') and v.ndim == 1):
        if len(v) != 2:
            raise ValueError(f'pixel_um: a number or a pair (x, y), not {v!r}')
        px, py = v[0], v[1]
        if not floats and (torch.is_tensor(px) or torch.is_tensor(py)):
            return (px, py)
        px, py = float(px), float(py)
        return px if px == py else (px, py)
    return float(v)


def pixel_xy(v):
    """(x, y) pixel sizes of a pixel size kept by pixel_size (a float or a pair)."""
    return (v[0], v[1]) if isinstance(v, tuple) else (v, v)


def pixel_x(v):
    """The x pixel size: the unit of z in px (z is stored in px of the x pixel size; = the pixel size if square)."""
    return v[0] if isinstance(v, tuple) else v


def pixel_area(v):
    """Pixel area (um^2): v ** 2 for square pixels (as before), x * y for non-square ones."""
    return v[0] * v[1] if isinstance(v, tuple) else v ** 2


def pixel_max(v):
    """The larger pixel size (sets the band limit in cycles / px)."""
    return max(float(v[0]), float(v[1])) if isinstance(v, tuple) else v


def pixel_meta(v):
    """Dataset metadata of a pixel size: pixel_nm (x; also the unit of z), and pixel_nm_y only for non-square pixels."""
    v = pixel_size(v, floats=True)
    if isinstance(v, tuple):
        return dict(pixel_nm=1000.0 * v[0], pixel_nm_y=1000.0 * v[1])
    return dict(pixel_nm=1000.0 * v)


def pixel_json(v):
    """A pixel size for a json file: a float, or [x, y] for non-square pixels."""
    v = pixel_size(v, floats=True)
    return list(v) if isinstance(v, tuple) else v


@dataclasses.dataclass
class Optics:
    """Detection optics. pixel_um: back-projected camera pixel size (um; a pair (x, y) for non-square pixels, stored only
    when they differ), NA, wavelength_um: emission wavelength (um), n_imm: immersion (= coverslip) index, n_med: sample
    medium index."""
    pixel_um: float = 0.10833
    NA: float = 1.49
    wavelength_um: float = 0.68
    n_imm: float = 1.518
    n_med: float = 1.33

    def __post_init__(self):
        self.pixel_um = pixel_size(self.pixel_um, floats=True)

    def to_dict(self):
        d = dataclasses.asdict(self)
        d['pixel_um'] = pixel_json(self.pixel_um)
        return d

    @classmethod
    def from_dict(cls, d):
        """From an Optics dict, an optics config block (same names; extra keys are ignored), or a
        calib json (pixelsize_um / wavelength_um / NA / n_imm / n_med at the top level)."""
        d = dict(d)
        if 'pixel_um' not in d and 'pixelsize_um' in d:
            d['pixel_um'] = d['pixelsize_um']
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: pixel_size(v, floats=True) if k == 'pixel_um' else float(v) for k, v in d.items()
                      if k in names})

    def vectorial_kwargs(self):
        """Keyword arguments for VectorialPSF(...)."""
        return dict(pixel_um=self.pixel_um, NA=self.NA, wavelength_um=self.wavelength_um, n_imm=self.n_imm,
                    n_med=self.n_med)
