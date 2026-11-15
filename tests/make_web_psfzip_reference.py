"""psf.zip files written by torch_psf for the web reader test (web/test/test_psfzip.ts), GENERATED into
web/test/.cache/psfzip/ (not tracked); web/test/run_tests.py runs this script when the output is missing or this file
has changed. Small models on the CPU (a few seconds).

    python tests/make_web_psfzip_reference.py [--out web/test/.cache/psfzip]

Outputs:
  python_vectorial.psf.zip   VectorialPSF (Zernikes, focus, z range) with a camera with per-pixel maps and extra map
                             entries of dtypes the web npy reader does not parse (bool, unicode) and int64;
                             zstack 9 planes at 1 / 2 px
  python_spline.psf.zip      the same model as a spline channel (CubicSplinePSF, source = the vectorial model);
                             zstack 7 planes at 1 px
  python_ref.json            the model parameters as torch_psf holds them (float32 values as float64)
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
from torch_psf import VectorialPSF, Camera
from torch_psf.psfzip import save_zip

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
print('wrote', a.out)
