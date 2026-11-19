"""A small, deterministic reference of the composed Vortex PSF in the CURRENT API, for the PSF refactor (psf-unify):
the new API's composition VectorialPSF + plate phase -> Polychromatic -> Field -> ZMode, with per-emitter dipM /
dipole_basis, must reproduce tests/data/orientation_composed_ref.json (written by this script; checked by
tests/test_orientation.py::test_composed_vortex_reference).

The composition in the current API (what each component means):
  base     VectorialPSF(roisize 15, pixel 0.065 um, NA 1.47, n_imm 1.518, n_med 1.0, wavelength 0.5975 um, npupil 32,
           oversample 1, pixel_integration "box" (point samples at the pixel centres), Zernikes Noll 5..21, focus_um 0,
           z_range (-0.6, 0.6)), on the CPU
  plate    orientation.vortex_phase(base, charge=+1, center=(-0.026, 0.013), levels=0, depth=593 / 597.5): the spiral
           phase (waves) written into base.pupil_phase (fixed; centre in pupil radii, x = column, y = row)
  constant Zernikes  base.set_zernike({5: 0.013, 6: -0.013, 11: 0.057})  (waves rms, Noll; at the field reference)
  spectrum orientation.Polychromatic(base, [0.5725, 0.5975, 0.6225], [1, 2, 1]): the weighted sum over wavelengths,
           pupil phase (Zernikes + pupil_phase) scaled by lambda_0 / lambda_i, each wavelength normalised as
           VectorialPSF, weights normalised to sum 1
  field    torch_psf.field.FieldZernikes(noll [5, 6, 8, 11], order {5: 2, 6: 2, 8: 1, 11: 1}, norm (512, 512, 512,
           512) px) with fixed coefficients COEF; emitter m at field position FIELD_XY[m] (camera px) gets
           zcoef_offset = field.offsets(FIELD_XY[m], base.zidx) (waves), passed per emitter to dipole_basis / render
           (and scaled by lambda_0 / lambda_i inside Polychromatic like the base Zernikes)
  z mode   orientation.FocusZ(poly, height=0.0): z = the emitter's position relative to the focal plane, the emitter
           at the coverslip; the model is poly(pos (x, y, 0), focus = focus_um - z)
  dipoles  M = (1 - g) I / 3 + g mu mu^T (orientation.dipole_M), mu from (theta, phi) (orientation.from_angles,
           theta from the optical axis, phi from +x = column towards +y = row); 4 dipoles and one isotropic emitter
           (g = 0, i.e. M = I / 3); image = I * sum_k m_k basis_k + bg, basis = dipole_basis (xx, yy, zz, xy, xz, yz;
           off-diagonal images with the factor 2), m = M_to_m6(M)
  fits     orientation.fit_orientation(model, rois, iters 20, z_limits (-0.5, 0.5), zstarts 1, iso_iters 15, zcoef_offset per ROI)
           on the noise-free image and on one Poisson draw (torch.Generator seed 1234) of each emitter

usage: python -I study/orientation/composed_reference.py   (writes tests/data/orientation_composed_ref.json)
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import numpy as np                                                                     # noqa: E402
import torch                                                                           # noqa: E402
from torch_psf import VectorialPSF                                                     # noqa: E402
from torch_psf.field import FieldZernikes                                              # noqa: E402
from torch_psf.orientation import (vortex_phase, Polychromatic, FocusZ, dipole_M, from_angles, M_to_m6,
                                   basis_images, model_and_jacobian, fit_orientation, angles)  # noqa: E402

REF_PATH = os.path.join(ROOT, 'tests', 'data', 'orientation_composed_ref.json')
CENTER = (-0.026, 0.013)
WAVELENGTHS, WEIGHTS = [0.5725, 0.5975, 0.6225], [1.0, 2.0, 1.0]
FIELD_NOLL, FIELD_ORDER = [5, 6, 8, 11], {5: 2, 6: 2, 8: 1, 11: 1}
FIELD_NORM = (512.0, 512.0, 512.0, 512.0)
# coefficients [k, nf - 1] for the order-2 feature set (u, v, u^2? ...: torch_psf.field.feature_powers(2)[1:])
COEF = [[0.010, -0.006, 0.004, 0.0, -0.003],
        [-0.004, 0.008, 0.0, 0.003, 0.0],
        [0.012, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.008, 0.0, 0.0, 0.0]]
FIELD_XY = [[100.0, 900.0], [512.0, 512.0], [950.0, 150.0]]
# emitters: (x px, y px, z um, theta deg, phi deg, g, field index); g = 0: isotropic
EMITTERS = [(0.21, -0.17, 0.08, 15.0, 30.0, 1.0, 0),
            (-0.33, 0.12, -0.12, 60.0, 135.0, 1.0, 1),
            (0.05, 0.28, 0.20, 85.0, 250.0, 0.5, 2),
            (-0.12, -0.31, -0.25, 40.0, 320.0, 0.5, 0),
            (0.17, 0.09, 0.03, 0.0, 0.0, 0.0, 1)]
INTENSITY, BG = 3000.0, 5.0
PIXELS = [(7, 7), (5, 9), (10, 4), (3, 3), (11, 12)]


def build():
    """the composed model, the field, per-emitter theta [5, 9] and zcoef_offset [5, nz]"""
    torch.manual_seed(0)
    v = VectorialPSF(15, pixel_um=0.065, NA=1.47, n_imm=1.518, n_med=1.0, wavelength_um=0.5975, npupil=32,
                     oversample=1, zernike_indices=tuple(range(5, 22)), device="cpu", focus_um=0.0, z_range=(-0.6, 0.6))
    for p in v.parameters():
        p.requires_grad_(False)
    vortex_phase(v, charge=1, center=CENTER, levels=0, depth=593.0 / 597.5)
    v.set_zernike({5: 0.013, 6: -0.013, 11: 0.057})
    poly = Polychromatic(v, WAVELENGTHS, WEIGHTS)
    model = FocusZ(poly, height=0.0, z_range=(-0.6, 0.6))
    field = FieldZernikes(FIELD_NOLL, FIELD_ORDER, norm=FIELD_NORM, coef=COEF, device='cpu')
    with torch.no_grad():
        zo_field = field.offsets(torch.tensor(FIELD_XY), v.zidx)
    th = torch.zeros(len(EMITTERS), 9)
    zo = torch.zeros(len(EMITTERS), len(v.zidx))
    for i, (x, y, z, t, ph, g, fi) in enumerate(EMITTERS):
        th[i, :5] = torch.tensor([x, y, z, INTENSITY, BG])
        th[i, 5:8] = from_angles(t, ph)
        th[i, 8] = g
        zo[i] = zo_field[fi]
    return model, field, th, zo


def image_hash(img):
    """sha256 of the image / its peak rounded to 5 decimals (float32): stable against last-bit differences"""
    a = np.round(np.asarray(img, np.float64) / float(np.max(img)), 5).astype(np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def compute():
    model, field, th, zo = build()
    with torch.no_grad():
        mu, _ = model_and_jacobian(model, th, zcoef_offset=zo)
        mu = mu[:, 0]
        B = basis_images(model, th[:, :3], zcoef_offset=zo)
        # the same images through the model's own render path (dipM per emitter) as a cross-check
        M = dipole_M(th[:, 5:8], th[:, 8])
        direct = INTENSITY * model(th[:, :3], dipM=M, zcoef_offset=zo) + BG
    out = dict(description=__doc__.split('\n\n')[0], emitters=[])
    noisy = torch.poisson(mu, generator=torch.Generator().manual_seed(1234))
    fits = {}
    for name, d in (('noise_free', mu), ('noisy_seed1234', noisy)):
        r = fit_orientation(model, d, iters=20, z_limits=(-0.5, 0.5), zstarts=1, iso_iters=15, zcoef_offset=zo, batch=8)
        fits[name] = r
    for i in range(len(EMITTERS)):
        img = mu[i].numpy()
        e = dict(emitter=dict(zip(('x_px', 'y_px', 'z_um', 'theta_deg', 'phi_deg', 'g', 'field_index'), EMITTERS[i])),
                 field_xy=FIELD_XY[EMITTERS[i][6]],
                 image_sum=float(img.sum()), image_peak=float(img.max()),
                 pixels={f'{r},{c}': float(img[r, c]) for r, c in PIXELS}, image_hash=image_hash(img),
                 render_vs_basis_max_abs=float((direct[i] - mu[i]).abs().max()),
                 dipole_basis_sums=[float(b) for b in B[i].sum((-1, -2))])
        for name, r in fits.items():
            t, p = angles(r['theta'][i:i + 1, 5:8])
            e['fit_' + name] = dict(theta_deg=float(t), phi_deg=float(p), g=float(r['theta'][i, 8]),
                                    z_um=float(r['theta'][i, 2]), x_px=float(r['theta'][i, 0]), y_px=float(r['theta'][i, 1]),
                                    intensity=float(r['theta'][i, 3]), bg=float(r['theta'][i, 4]))
        out['emitters'].append(e)
    return out


if __name__ == '__main__':
    res = compute()
    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    json.dump(res, open(REF_PATH, 'w'), indent=1)
    for e in res['emitters']:
        print(e['emitter'], round(e['image_sum'], 3), e['image_hash'], {k: round(v, 3) for k, v in e['fit_noise_free'].items()})
