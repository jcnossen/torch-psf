"""PSF and calibration json files. The implementation is PSF.save / PSF.load (torch_psf.psf, registry by type name;
each model class writes and reads its own parameters); the functions here are the module-level entry points.

PSF json schema (format 'torch-psf', version '0.1.0'):
  type            'vectorial' | 'spline' | 'gaussian' | 'astig_gaussian'
  optics          {pixel_um, NA, wavelength_um, n_imm, n_med}   (also at the top level as pixelsize_um, wavelength_um, ...)
  zernike_noll, zernike_waves, apod, saf, focus_um, psf_blur_px2, psf_dipole_M, diattenuation, npupil, oversample,
  roisize, z_range_um
  spline          (type spline) {roisize, z_range, oversample, dz_um, zpad, xy_limit, c, z0_um, nz, u0_px, pixel_um};
                  knots in the npz ('knots')
  gaussian        (types gaussian / astig_gaussian) {sigma_px, gamma_um, d_um, z_range_um, pixel_um}; a plain
                  calibration json with psf_gauss_sigma_um is a 2D Gaussian
  arrays          name of the sidecar npz (pupil_phase, pupil_amp, knots), relative to the json
  camera          optional Camera dict; provenance: free-form
A plain calibration json (no 'format') with the vectorial keys at the top level (pixelsize_um, wavelength_um, NA,
n_imm, n_med, zernike_noll, zernike_waves, focus_um, psf_blur_px2, ...) is read as a vectorial PSF.
"""
import json
from torch_psf.optics import Optics
from torch_psf.camera import Camera
from torch_psf.psf import PSF, FORMAT, VERSION   # noqa: F401
from torch_psf.spline import CubicSplinePSF


def save(path, psf, source=None, camera=None, provenance=None):
    """psf.save(path, ...): write a PSF model (json plus an .npz for arrays). Returns the json dict."""
    return psf.save(path, camera=camera, provenance=provenance, source=source)


def load(path, roisize=None, device=None, angle=None, npupil=None, oversample=None):
    """PSF.load(path, ...): read a PSF json or a plain calibration json and return the model."""
    return PSF.load(path, roisize=roisize, device=device, angle=angle, npupil=npupil, oversample=oversample)


def load_spline_source(path, device=None):
    """The vectorial model a saved spline was built from (None if it was not stored)."""
    return CubicSplinePSF.load_source(path, device=device)


def load_camera(path, load_maps=True):
    """Camera block of a calib / torch-psf json (relative map paths resolved against the json's directory), or None."""
    return Camera.from_json(path, load_maps=load_maps)


def load_optics(path):
    with open(path) as f:
        d = json.load(f)
    return Optics.from_dict(d.get('optics', d))
