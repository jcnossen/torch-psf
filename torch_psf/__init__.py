"""torch-psf: a PSF library for single-molecule localization microscopy, in PyTorch.

The PSF is the main object: PSF (base class) with render, expected, fisher / crlb, fit (ROIs -> localizations),
localize (movie -> localizations), save / PSF.load and to_spline; the models VectorialPSF, ConvCubicSplinePSF,
GaussianPSF and AstigmaticGaussianPSF. PSFs are estimated with ZStackEstimator (bead z-stacks) and InSituEstimator
(from the SMLM movie itself).

Conventions: x = column, y = row, in px, pixel centres at integers; ROI positions are offsets from the ROI centre;
z = emitter height above the coverslip (um); focus_um = focal plane relative to the coverslip (um); lengths carry their
unit in the name (pixel_um, wavelength_um, dz_um); photons for intensities and background (photons / px). Device
default: cuda if available, else cpu. Lower-level pieces: torch_psf.fit, .spots, .detect, .insitu, .simulate, .io,
.table, .zernike, .varpro.
"""
__version__ = '0.1.0'

from torch_psf.config import Optics, Camera
from torch_psf.movie import open_movie
from torch_psf.psf import PSF
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import ConvCubicSplinePSF
from torch_psf.gaussian import GaussianPSF, AstigmaticGaussianPSF
from torch_psf.detect import detect_spots
from torch_psf.spots import extract_spots
from torch_psf.estimators import ZStackEstimator, InSituEstimator
from torch_psf import fit, insitu, simulate, spots, detect, io, zernike, varpro, table   # noqa: F401
