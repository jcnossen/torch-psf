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
