"""torch-psf: PSF models (vectorial, convolutional spline, Gaussian), single-emitter fitting, spot detection and PSF
estimation from beads and in situ, in PyTorch.

Conventions: x = column, y = row, in px, pixel centres at integers; ROI positions are offsets from the ROI centre;
z = emitter height above the coverslip (um); focus_um = focal plane relative to the coverslip (um); lengths carry their
unit in the name (pixel_um, wavelength_um, dz_um); photons for intensities and background (photons / px).
Every PSF model has render(pos [N, 3], grad=False) -> PSF [N, P, P] (or (psf, d/dx, d/dy, d/dz)), roisize, z_range and
xy_limit. Device default: cuda if available, else cpu.

Top level: the models, files, configuration, the steps and the entry points. Helpers live in the submodules:
torch_psf.fit (lm_poisson, fisher, crlb, ...), .spots, .detect, .insitu (estimators), .simulate, .varpro, .io, .table.
"""
__version__ = '0.1.0'

from torch_psf.config import Optics, Camera
from torch_psf.movie import open_movie
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import ConvSplinePSF
from torch_psf.gaussian import GaussianPSF
from torch_psf.io import load_psf, save_psf
from torch_psf.detect import detect_spots
from torch_psf.spots import extract_spots
from torch_psf.fit import localize, crlb
from torch_psf.estimate import PSFEstimate
from torch_psf.pipeline import localize_movie, estimate_psf_beads, estimate_psf_insitu
from torch_psf import fit, insitu, simulate, spots, detect, io, zernike, varpro, table   # noqa: F401
