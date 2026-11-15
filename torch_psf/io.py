"""PSF and calibration json files. The implementation is PSF.save / PSF.load (torch_psf.psf, registry by type name;
each model class writes and reads its own parameters); the functions here are the module-level entry points.

PSF json schema (format 'torch-psf', version '0.1.0'):
  type            'vectorial' | 'spline' | 'gaussian' | 'astig_gaussian' | 'fourpi' | 'fourpi_spline'
  optics          {pixel_um, NA, wavelength_um, n_imm, n_med}   (also at the top level as pixelsize_um, wavelength_um, ...)
  zernike_noll, zernike_waves, apod, saf, focus_um, psf_blur_px2, psf_dipole_M, diattenuation, npupil, oversample,
  roisize, z_range_um
  spline          (type spline) {roisize, z_range, oversample, dz_um, zpad, xy_limit, c, z0_um, nz, u0_px, pixel_um};
                  knots in the npz ('knots')
  gaussian        (types gaussian / astig_gaussian) {sigma_px, gamma_um, d_um, z_range_um, pixel_um}; a plain
                  calibration json with psf_gauss_sigma_um is a 2D Gaussian
  fourpi          (types fourpi / fourpi_spline) the FourPiPSF parameter dict (FourPiPSF.state(): meta, zt, zb, phi,
                  tau, a, gamma, d_bot, blur_px2, depth); for fourpi_spline the source model
  fourpi_spline   (type fourpi_spline) the spline grid as `spline` plus n_channels, tau; knots [3, C, nz, n, n] in the npz
  arrays          name of the sidecar npz (pupil_phase, pupil_amp, knots), relative to the json
  camera          optional Camera dict; provenance: free-form
A plain calibration json (no 'format') with the vectorial keys at the top level (pixelsize_um, wavelength_um, NA,
n_imm, n_med, zernike_noll, zernike_waves, focus_um, psf_blur_px2, ...) is read as a vectorial PSF.

psf.zip (version '0.2.0', docs/PSF_FORMAT.md; torch_psf.psfzip): psf.json with a `channels` list (each channel the
0.1.0 keys of its model plus mapping / field / crop), .npy arrays, zstack.tif, preview.png, fit_report.pdf. save_zip
writes it; load / PSF.load read it like a json.
"""
import json
from torch_psf.optics import Optics
from torch_psf.camera import Camera
from torch_psf.psf import PSF, FORMAT, VERSION   # noqa: F401
from torch_psf.spline import CubicSplinePSF
from torch_psf import psfzip
from torch_psf.psfzip import PSFZip, save_zip   # noqa: F401


def save(path, psf, source=None, camera=None, provenance=None):
    """psf.save(path, ...): write a PSF model (json plus an .npz for arrays); a path ending in .zip writes a psf.zip
    (save_zip with its defaults). Returns the json dict."""
    if str(path).lower().endswith('.zip'):
        return save_zip(path, psf, camera=camera, provenance=provenance, source=source)
    return psf.save(path, camera=camera, provenance=provenance, source=source)


def load(path, roisize=None, device=None, angle=None, npupil=None, oversample=None, channel=None):
    """PSF.load(path, ...): read a PSF json, a psf.zip or a plain calibration json and return the model. channel
    (index or name): one channel of a psf.zip with independent channels (PSFZip.read gives all of them, with their
    mappings and field dependence)."""
    if channel is not None:
        return PSFZip.read(path, roisize=roisize, device=device).channel(channel).model
    return PSF.load(path, roisize=roisize, device=device, angle=angle, npupil=npupil, oversample=oversample)


def load_spline_source(path, device=None):
    """The vectorial model a saved spline was built from (None if it was not stored)."""
    return CubicSplinePSF.load_source(path, device=device)


def load_camera(path, load_maps=True):
    """Camera block of a calib / torch-psf json (relative map paths resolved against the json's directory) or of a
    psf.zip (per-pixel maps from the zip), or None."""
    if psfzip.is_psf_zip(path):
        return psfzip.camera_from_meta(*psfzip.read_zip(path))
    return Camera.from_json(path, load_maps=load_maps)


def load_optics(path):
    if psfzip.is_psf_zip(path):
        d = psfzip.read_zip(path)[0]
    else:
        with open(path) as f:
            d = json.load(f)
    return Optics.from_dict(d.get('optics', d))
