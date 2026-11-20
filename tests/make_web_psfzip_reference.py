"""psf.zip files written by torch_psf for the web reader test (web/test/test_psfzip.ts), GENERATED into
web/test/.cache/psfzip/ (not tracked); web/test/run_tests.py runs this script when the output is missing or this file
has changed. Small models on the CPU (a few seconds).

    python tests/make_web_psfzip_reference.py [--out web/test/.cache/psfzip]

Outputs:
  python_vectorial.psf.zip   VectorialPSF (Zernikes, focus, z range) with a camera with per-pixel maps and extra map
                             entries of dtypes the web npy reader does not parse (bool, unicode) and int64;
                             zstack 9 planes at 1 / 2 px
  python_spline.psf.zip      the same model as a spline channel (SplinePSF, source = the vectorial model);
                             zstack 7 planes at 1 px
  python_ref.json            the model parameters as torch_psf holds them (float32 values as float64)
  integration_ref.json       Python renders at three emitter positions for the small 0.3.0 zip files
"""
import argparse, json, os, sys, warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--torch-psf', default=ROOT)
ap.add_argument('--out', default=os.path.join(ROOT, 'web', 'test', '.cache', 'psfzip'))
a = ap.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, a.torch_psf)
import torch
from torch_psf import VectorialPSF, Camera, PSF
from torch_psf.psfzip import save_zip
from torch_psf.spline import to_spline

os.makedirs(a.out, exist_ok=True)
v = VectorialPSF(13, pixel_um=0.108, NA=1.45, n_imm=1.518, n_med=1.33, wavelength_um=0.67, zernike_indices=tuple(range(5, 22)),
                 npupil=32, oversample=2, device='cpu', focus_um=0.35, z_range=(0.0, 0.7))
v.set_zernike({5: -0.03, 6: 0.09, 7: 0.015, 11: 0.025, 12: -0.01})
cam = Camera(gain_e_per_adu=0.47, offset_adu=np.full((8, 10), 101.5, np.float32), readnoise_e=1.6)
cam.scalars['offset_adu'] = 100.0
cam.map_extra = {'hot': np.zeros((8, 10), bool), 'n_frames': np.array(5000, np.int64), 'serial': np.array('cam-07'),
                 'variance': np.full((8, 10), 2.5, np.float32)}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    save_zip(os.path.join(a.out, 'python_vectorial.psf.zip'), v, camera=cam, zstack_planes=9, zstack_px=0.5,
             provenance={'source': 'tests/make_web_psfzip_reference.py'})
    s = v.to_spline(roisize=11, z_range=(0.1, 0.6), oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0)
    save_zip(os.path.join(a.out, 'python_spline.psf.zip'), s, source=v, zstack_planes=7, zstack_px=1.0)
ref = dict(zernike_noll=list(v.zidx), zernike_waves=[float(x) for x in v.zcoef.detach()], focus_um=v.focus_um,
           z_range_um=list(v.z_range), roisize=v.roisize, npupil=v.npupil, oversample=v.os, optics=v.optics.to_dict(),
           spline=s.to_dict())
json.dump(ref, open(os.path.join(a.out, 'python_ref.json'), 'w'), indent=1)

# The 0.3 evaluator oracle includes fractional pixels and a position outside the spline grid.
positions = torch.tensor([[0., 0., 0.15], [0.27, -0.38, 0.12], [-1.1, 0.7, 0.33]])
cases = []
v = v.resized(5)
v.z_range = (0.1, 0.2)


@torch.no_grad()
def reference(name, model, source=None, n_angles=0):
    path = os.path.join(a.out, name + '.psf.zip')
    m = save_zip(path, model, source=source, zstack_planes=3)
    assert m['version'] == '0.3.0'
    angles = [None, *range(n_angles or getattr(model, 'n_angles', 0))]
    for angle in angles:
        active = PSF.load(path, angle=angle, device='cpu')
        images = active.render(positions).detach().flatten(1).tolist()
        cases.append(dict(name=name, angle=angle, images=images))


for mode in ('exact', 'gauss_legendre'):
    v.pixel_integration = mode
    reference(mode, v)
reference('spline_nonbox_source', v.to_spline(roisize=5, z_range=(0.1, 0.2), oversample=2,
                                             dz_um=0.05, zpad=2, xy_limit=0.5), source=v)
v2 = v.resized(5)
v2.set_zernike({6: -0.18})
for name, models in [('integrated', v), ('integrated_angles', [v, v2])]:
    model = to_spline(models, (0.1, 0.2), kernel='bspline', integrated=True, oversample=2, dz_um=0.05, xy_limit=0.5, zpad=2)
    reference(name, model, source=v if isinstance(models, list) else None)
# A SIMCODE-style spectral calibration with a non-diagonal covariance and dipole tensor.
v.pixel_integration = 'exact'
v.dipM_angle = torch.tensor([[[0.4, 0.08, 0.1], [0.08, 0.3, 0.0], [0.1, 0.0, 0.3]],
                             [[0.3, 0.0, 0.0], [0.0, 0.45, -0.1], [0.0, -0.1, 0.25]]])
v.blur_angle = torch.tensor([[0.12, 0.16, 0.05], [0.16, 0.09, -0.06]])
v.dipM, v.blur, v.diat = v.dipM_angle.mean(0), v.blur_angle.mean(0), 0.15
v.pupil_phase.copy_(v.px * 0.03)
v.pupil_amp.copy_(1 + v.py * 0.15)
reference('spectral_angles', v, n_angles=2)
with open(os.path.join(a.out, 'integration_ref.json'), 'w') as f:
    json.dump(dict(positions=positions.tolist(), cases=cases), f)
print('wrote', a.out)
