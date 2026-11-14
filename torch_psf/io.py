"""PSF files: one JSON description (parameters; optics; optional camera) plus an optional sidecar .npz for arrays
(pupil maps, spline knots). The vectorial PSF keys are those of the plain calibration json (pixelsize_um,
wavelength_um, NA, n_imm, n_med, zernike_noll, zernike_waves, focus_um, psf_blur_px2, ...), which load_psf also reads.

Schema (format 'torch-psf', version '0.1.0'):
  type            'vectorial' | 'spline' | 'gaussian'
  optics          {pixel_um, NA, wavelength_um, n_imm, n_med}   (also at the top level as pixelsize_um, wavelength_um, ...)
  zernike_noll, zernike_waves, apod, saf, focus_um, psf_blur_px2, psf_dipole_M, diattenuation, npupil, oversample,
  roisize, z_range_um
  spline          (type spline) {roisize, z_range, oversample, dz_um, zpad, xy_limit, c, z0_um, nz}; knots in the
                  npz ('knots')
  gaussian        (type gaussian) {sigma_px, gamma_um, d_um, z_range_um}; calibration key psf_gauss_sigma_um
  arrays          name of the sidecar npz (pupil_phase, pupil_amp, knots), relative to the json
  camera          optional Camera dict; provenance: free-form
"""
import json
import os
import numpy as np
import torch
from torch_psf.config import Optics, Camera
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import ConvSplinePSF
from torch_psf.gaussian import GaussianPSF

FORMAT, VERSION = 'torch-psf', '0.1.0'


def _list(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    return np.asarray(v).tolist()


def _vectorial_dict(vec):
    o = vec.optics
    d = dict(optics=o.to_dict(), zernike_noll=list(vec.zidx), zernike_waves=_list(vec.zcoef), apod=_list(vec.apod),
             saf=_list(vec.saf), focus_um=vec.focus_um, psf_blur_px2=_list(vec.blur), psf_dipole_M=_list(vec.dipM),
             diattenuation=None if vec.diat is None else float(vec.diat), npupil=vec.npupil, oversample=vec.os,
             roisize=vec.roisize, z_range_um=None if vec.z_range is None else list(vec.z_range))
    if vec.dipw is not None:
        d['psf_dipole_w'] = _list(vec.dipw)
    d.update(pixelsize_um=o.pixel_um, wavelength_um=o.wavelength_um, NA=o.NA, n_imm=o.n_imm, n_med=o.n_med)
    arrays = {}
    if bool((vec.pupil_phase != 0).any()):
        arrays['pupil_phase'] = vec.pupil_phase.detach().cpu().numpy()
    if bool((vec.pupil_amp != 1).any()):
        arrays['pupil_amp'] = vec.pupil_amp.detach().cpu().numpy()
    return d, arrays


def save_psf(path, psf, source=None, camera=None, provenance=None):
    """Write a PSF model (VectorialPSF, ConvSplinePSF or GaussianPSF) to path (.json) plus path minus .json + '.npz'
    when there are arrays. source: for a spline, the vectorial model it was built from (its parameters are stored
    too, so the spline can be rebuilt). camera: optional Camera. Returns the json dict."""
    arrays = {}
    if isinstance(psf, VectorialPSF):
        d, arrays = _vectorial_dict(psf)
        d = dict(format=FORMAT, version=VERSION, type='vectorial', **d)
    elif isinstance(psf, ConvSplinePSF):
        d = dict(format=FORMAT, version=VERSION, type='spline', spline=psf.to_dict())
        if source is not None:
            v, a = _vectorial_dict(source)
            d.update(v)
            arrays.update(a)
        arrays['knots'] = psf.knots.detach().cpu().numpy()
    elif isinstance(psf, GaussianPSF):
        d = dict(format=FORMAT, version=VERSION, type='gaussian',
                 gaussian=dict(sigma_px=list(psf.sigma), gamma_um=psf.gamma, d_um=psf.d,
                               z_range_um=None if psf.z_range is None else list(psf.z_range)), roisize=psf.roisize)
    else:
        raise TypeError(type(psf))
    if camera is not None:
        d['camera'] = camera.to_dict()
    d['provenance'] = provenance or {}
    path = str(path)
    if arrays:
        npz = os.path.splitext(path)[0] + '.npz'
        np.savez(npz, **arrays)
        d['arrays'] = os.path.basename(npz)
    with open(path, 'w') as f:
        json.dump(d, f, indent=1)
    return d


def _vectorial_from(d, roisize, device, angle, arrays, npupil=None, oversample=None):
    o = Optics.from_dict(d['optics'] if 'optics' in d else d)
    zr = d.get('z_range_um')
    vec = VectorialPSF(roisize, zernike_indices=tuple(d.get('zernike_noll', range(5, 22))),
                       npupil=npupil or d.get('npupil', 64), oversample=oversample or d.get('oversample', 3),
                       device=device, focus_um=float(d.get('focus_um', 0.0) or 0.0),
                       z_range=tuple(zr) if zr else (0.0, 1.0), **o.vectorial_kwargs())
    t = lambda v: torch.tensor(v, device=vec.dev, dtype=torch.float32)
    with torch.no_grad():
        if d.get('zernike_waves') is not None:
            vec.zcoef.copy_(t(d['zernike_waves']))
        if d.get('apod') is not None:
            vec.apod.copy_(t(d['apod']))
        if d.get('saf') is not None:
            vec.saf.copy_(t(d['saf']))
        if arrays is not None:
            if 'pupil_phase' in arrays:
                vec.pupil_phase.copy_(t(arrays['pupil_phase']))
            if 'pupil_amp' in arrays:
                vec.pupil_amp.copy_(t(arrays['pupil_amp']))
    # optional terms: global, or per illumination angle (angle None = average over angles)
    if d.get('psf_blur_px2') is not None:
        vec.blur = t([float(v) for v in d['psf_blur_px2']])
    if d.get('psf_dipole_M') is not None:
        vec.dipM = t(d['psf_dipole_M'])
    if d.get('psf_dipole_w') is not None:
        vec.dipw = t(d['psf_dipole_w'])
    if d.get('psf_dipole_M_angle') is not None:
        M = t(d['psf_dipole_M_angle'])
        vec.dipM = M.mean(0) if angle is None else M[angle]
    if d.get('psf_blur_angle_px2') is not None:
        Bc = t(d['psf_blur_angle_px2'])
        vec.blur = Bc.mean(0) if angle is None else Bc[angle]
    if d.get('diattenuation') is not None:
        vec.diat = float(d['diattenuation'])
    return vec


def load_psf(path, roisize=None, device=None, angle=None, npupil=None, oversample=None):
    """Read a torch-psf file or a plain calibration json and return the PSF model.

    roisize: ROI size for vectorial / Gaussian models (default: the stored one, or 15 for a calibration json).
    angle: for calibs with per-pattern-angle PSF terms: 0 / 1 = that angle's PSF, None = the angle average. npupil / oversample override the stored pupil sampling of a vectorial model.
    A calib with psf_gauss_sigma_um gives a GaussianPSF (sigma = psf_gauss_sigma_um / pixel size)."""
    path = str(path)
    with open(path) as f:
        d = json.load(f)
    arrays = None
    if d.get('arrays'):
        arrays = dict(np.load(os.path.join(os.path.dirname(path), d['arrays'])))
    typ = d.get('type', 'vectorial') if d.get('format') == FORMAT else 'calib'
    if typ == 'calib' and d.get('psf_gauss_sigma_um') is not None:
        return GaussianPSF(roisize or 15, sigma_px=float(d['psf_gauss_sigma_um']) / float(d['pixelsize_um']),
                           device=device)
    if typ in ('calib', 'vectorial'):
        return _vectorial_from(d, roisize or d.get('roisize', 15), device, angle, arrays, npupil, oversample)
    if typ == 'spline':
        g = d['spline']
        spl = ConvSplinePSF(g['roisize'], tuple(g['z_range']), oversample=g['oversample'], dz_um=g['dz_um'],
                            zpad=g['zpad'], xy_limit=g['xy_limit'], device=device)
        spl.knots = torch.as_tensor(arrays['knots'], device=spl.dev, dtype=spl.dtype)
        return spl
    if typ == 'gaussian':
        g = d['gaussian']
        return GaussianPSF(roisize or d.get('roisize', 15), sigma_px=tuple(g['sigma_px']), gamma_um=g.get('gamma_um'),
                           d_um=g.get('d_um', 0.4), z_range=g.get('z_range_um') or (-0.5, 0.5), device=device)
    raise ValueError(f'unknown PSF type {typ!r}')


def load_spline_source(path, device=None):
    """The vectorial model a saved spline was built from (None if it was not stored)."""
    with open(path) as f:
        d = json.load(f)
    if 'zernike_waves' not in d:
        return None
    arrays = dict(np.load(os.path.join(os.path.dirname(str(path)), d['arrays']))) if d.get('arrays') else None
    return _vectorial_from(d, d.get('roisize', 15), device, None, arrays)


def load_camera(path, load_maps=True):
    """Camera block of a calib / torch-psf json (relative map paths resolved against the json's directory), or None."""
    with open(path) as f:
        d = json.load(f)
    c = d.get('camera')
    if not c:
        return None
    return Camera.from_dict(c, base_dir=os.path.dirname(os.path.abspath(str(path))), load_maps=load_maps)


def load_optics(path):
    with open(path) as f:
        d = json.load(f)
    return Optics.from_dict(d.get('optics', d))
