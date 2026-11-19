"""torch-psf: a PSF library for single-molecule localization microscopy, in PyTorch.

The PSF is the main object: PSF (base class) with render, expected, fisher / crlb, fit (ROIs -> localizations),
localize (movie -> localizations), save / PSF.load (also psf.zip: io.save_zip, docs/PSF_FORMAT.md) and to_spline;
the models VectorialPSF, CubicSplinePSF,
IntegratedSplinePSF (torch_psf.pixint: the integrated cubic B-spline of point samples of a vectorial model, exact pixel
integral; psf.json type 'integrated_spline'), GaussianPSF and AstigmaticGaussianPSF, and the multi-channel models (MultiChannelPSF, torch_psf.multichannel; images
[N, C, P, P], fit / crlb on all channels jointly): ChannelsPSF (independent channels) and the coherent 4Pi models
FourPiPSF and FourPiSplinePSF (torch_psf.fourpi), and the field-dependent models (torch_psf.field: FieldZernikes, a
polynomial of the field position per Zernike; FieldPSF, a VectorialPSF with it; TiledSplinePSF, its tiled spline
form; ChannelMap, torch_psf.mapping: the channel mappings of a multi-channel setup). PSFs are estimated with
ZStackEstimator (bead z-stacks; DenseBeadEstimator for dense, overlapping beads; PhaseDiversityEstimator: phase
diversity, the object eliminated; MultiChannelBeadEstimator: bead stacks seen in several channels, shared bead
positions, a fitted channel mapping), InSituEstimator (from the SMLM movie itself) and SOFIEstimator (torch_psf.sofi:
from dense blinking movies, SOFI or SMLM too dense to localize, by their second-order cross-cumulants).

Conventions: x = column, y = row, in px, pixel centres at integers; ROI positions are offsets from the ROI centre;
z = emitter height above the coverslip (um); focus_um = focal plane relative to the coverslip (um); lengths carry their
unit in the name (pixel_um, wavelength_um, dz_um); photons for intensities and background (photons / px). Device
default: cuda if available, else cpu. Lower-level pieces: torch_psf.fit (incl. solve_linear), .spots, .detect,
.estimation, .simulate, .io, .zernike, .optics, .camera (incl. photon_transfer). Localizations: Dataset
(torch_psf.dataset).

Module layers (each imports only from the layers before it): optics, zernike, fit, mapping -> detect, movie, spots,
dataset, camera -> psf (the base class) -> multichannel, vectorial, spline, gaussian -> fourpi -> psfzip, field,
matched_filter, psfreport, io, estimation, simulate ->
estimators, multichannel_estimator -> sofi -> viewer,
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
from torch_psf.pixint import IntegratedSplinePSF
from torch_psf.gaussian import GaussianPSF, AstigmaticGaussianPSF
from torch_psf.multichannel import MultiChannelPSF, ChannelsPSF
from torch_psf.fourpi import FourPiPSF, FourPiSplinePSF
from torch_psf.field import FieldZernikes, FieldPSF, TiledSplinePSF
from torch_psf.mapping import ChannelMap
from torch_psf.matched_filter import MatchedFilterDetector
from torch_psf.detect import detect_spots
from torch_psf.spots import extract_spots
from torch_psf.estimators import ZStackEstimator, DenseBeadEstimator, InSituEstimator, PhaseDiversityEstimator
from torch_psf.multichannel_estimator import MultiChannelBeadEstimator
from torch_psf.sofi import SOFIEstimator
from torch_psf.dataset import Dataset, Drift
from torch_psf import fit, estimation, simulate, spots, detect, io, zernike, optics, camera   # noqa: F401
