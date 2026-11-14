"""Generate tests/.cache/reference_e8e6fc5.npz (git-ignored): images of the pre-restructuring code (package
torch_convspline at commit e8e6fc5: VectorialPSF with free pupil maps, KnotPSF spline), which test_vectorial.py /
test_spline.py compare the current code against. Run by the tests when the file is missing (needs the git history);
by hand:  python tests/make_legacy_reference.py"""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch

tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, 'torch_convspline')
os.makedirs(pkg)
for f in ('vectorial', 'knotpsf', 'conv_spline'):
    src = subprocess.run(['git', '-C', HERE, 'show', f'e8e6fc5:torch_convspline/{f}.py'], capture_output=True, text=True,
                         check=True).stdout
    src = src.replace('import matplotlib.pyplot as plt', '').replace('import tqdm', '')
    open(os.path.join(pkg, f + '.py'), 'w').write(src)
open(os.path.join(pkg, '__init__.py'), 'w').close()
sys.path.insert(0, tmp)
from torch_convspline.vectorial import VectorialPSF, zernike
from torch_convspline.knotpsf import KnotPSF

dev = 'cpu'
g = torch.Generator().manual_seed(0)
zc = torch.randn(17, generator=g) * 0.03
zc[1] = 0.1
pos = torch.stack([torch.rand(6, generator=g) - 0.5, torch.rand(6, generator=g) - 0.5, 0.8 * torch.rand(6, generator=g)], 1)
focus = torch.full((6,), 0.4)
v = VectorialPSF(13, pixelsize=0.108, wavelength=0.68, npupil=48, oversample=2, device=dev)
with torch.no_grad():
    v.zcoef.copy_(zc)
    v.pupil_phase.copy_(0.05 * zernike(22, v.rho.clamp(max=1), v.phi) * v.mask)
    v.pupil_amp.copy_(1 - 0.3 * torch.exp(-((v.px - 0.3) ** 2 + (v.py + 0.2) ** 2) / 0.05))
    img = v(pos, focus)
s = KnotPSF(13, 0.0, 0.8, S=2, hz=0.05, zpad=2, xmax=3.0, device=dev)
big = VectorialPSF(s.Pl, pixelsize=0.108, wavelength=0.68, npupil=48, oversample=2, device=dev)
with torch.no_grad():
    big.zcoef.copy_(zc)
s.knots = s.stack_from_vectorial(big, lambda z: (z, torch.full_like(z, 0.4)))
r = s.evaluate(pos, grad=True)
os.makedirs(os.path.join(HERE, '.cache'), exist_ok=True)
np.savez_compressed(os.path.join(HERE, '.cache', 'reference_e8e6fc5.npz'),
                    zcoef=zc.numpy(), pos=pos.numpy(), focus=focus.numpy(), pupil_phase=v.pupil_phase.numpy(),
                    pupil_amp=v.pupil_amp.numpy(), vec_img=img.numpy(), spline_img=r[0].numpy(), spline_dx=r[1].numpy(),
                    spline_dy=r[2].numpy(), spline_dz=r[3].numpy())
print('written')
