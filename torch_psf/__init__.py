"""torch-psf: a PSF library for single-molecule localization microscopy, in PyTorch.

The PSF is the main object: PSF (base class) with render, expected, fisher / crlb, fit (ROIs -> localizations),
localize (movie -> localizations), save / PSF.load (also psf.zip: io.save_zip, docs/PSF_FORMAT.md) and to_spline;
the models VectorialPSF, CubicSplinePSF,
GaussianPSF and AstigmaticGaussianPSF, and the multi-channel coherent 4Pi models
FourPiPSF and FourPiSplinePSF (torch_psf.fourpi; images [N, C, P, P]). PSFs are estimated with ZStackEstimator (bead z-stacks) and InSituEstimator
(from the SMLM movie itself).

Conventions: x = column, y = row, in px, pixel centres at integers; ROI positions are offsets from the ROI centre;
z = emitter height above the coverslip (um); focus_um = focal plane relative to the coverslip (um); lengths carry their
unit in the name (pixel_um, wavelength_um, dz_um); photons for intensities and background (photons / px). Device
default: cuda if available, else cpu. Lower-level pieces: torch_psf.fit (incl. solve_linear), .spots, .detect,
.estimation, .simulate, .io, .zernike, .optics, .camera (incl. photon_transfer). Localizations: Dataset
(torch_psf.dataset).

Module layers (each imports only from the layers before it): optics, zernike, fit -> detect, movie, spots, dataset,
camera -> psf (the base class) -> vectorial, spline, gaussian -> fourpi -> psfzip, psfreport, io, estimation, simulate ->
estimators -> viewer,
__main__. The model classes register themselves by their psf.json type name when they are defined
(PSF.__init_subclass__), and this __init__ imports all of them, so PSF.load knows every type whenever torch_psf is
imported; importing a submodule alone (e.g. `from torch_psf.psf import PSF`) runs this __init__ first and so
registers them too. Optional dependencies are imported where they are used: dme (Dataset.estimate_drift) and
scikit-image (viewer isosurfaces).
"""
__version__ = '0.1.0'

from torch_psf.optics import Optics
from torch_psf.camera import Camera, photon_transfer
from torch_psf.movie import open_movie
from torch_psf.psf import PSF
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import CubicSplinePSF
from torch_psf.gaussian import GaussianPSF, AstigmaticGaussianPSF
from torch_psf.fourpi import FourPiPSF, FourPiSplinePSF
from torch_psf.detect import detect_spots
from torch_psf.spots import extract_spots
from torch_psf.estimators import ZStackEstimator, InSituEstimator
from torch_psf.dataset import Dataset
from torch_psf import fit, estimation, simulate, spots, detect, io, zernike, optics, camera   # noqa: F401
