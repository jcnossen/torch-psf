"""Reference data for the SMLM-movie mode of the web version, computed with torch_psf itself (the web code is a port
of these functions and must agree with them). Written to web/test/.cache/smlm/ (git-ignored; nothing generated is
tracked).

    python tests/make_web_smlm_reference.py [--torch-psf DIR] [--device cuda]

Cases:
  movie.tif + movie.json       a seeded movie (torch_psf.simulate.simulate_movie, astigmatic vectorial PSF via its
                               ConvSplinePSF), its optics and truth
  detect.json                  torch_psf.spots.extract_spots on it (chunk = 100 frames, threshold 6, border h + 2):
                               frame, corner, score per spot, and the noise / median of each chunk
  localize.npz / .json         the spline knots (ConvSplinePSF.from_vectorial), the ROIs of the first 600 detections,
                               torch_psf.fit.localize (theta, nll) and crlb, and the table columns
                               (torch_psf.pipeline.localize_spots)
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--torch-psf', default=ROOT, help='directory that contains the torch_psf package')
ap.add_argument('--device', default='cuda')
ap.add_argument('--out', default=os.path.join(ROOT, 'web', 'test', '.cache', 'smlm'))
a = ap.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, a.torch_psf)
import torch, tifffile
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import ConvSplinePSF
from torch_psf.simulate import simulate_movie
from torch_psf.spots import extract_spots
from torch_psf.detect import filter_frames
from torch_psf.fit import localize, crlb
from torch_psf.pipeline import localize_spots

os.makedirs(a.out, exist_ok=True)
dev = a.device
OPT = dict(pixel_um=0.1, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68)
NOLL = list(range(5, 22))
ZC = {6: 0.1, 11: 0.02}
P, ZR, FOCUS, CHUNK = 15, (0.0, 0.8), 0.4, 100
GRID = dict(oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0)

vec = VectorialPSF(P, zernike_indices=NOLL, npupil=64, oversample=2, device=dev, focus_um=FOCUS, z_range=ZR, **OPT)
with torch.no_grad():
    for j, v in ZC.items():
        vec.zcoef[NOLL.index(j)] = v
spl = ConvSplinePSF.from_vectorial(vec, P, ZR, **GRID)

movie, truth = simulate_movie(spl.render, P, T=300, H=96, W=112, nsites=120, p_on=0.01, mean_on=3.0, bg=20.0,
                              photons=3000.0, z_range=(0.1, 0.7), seed=3, device=dev)
tifffile.imwrite(os.path.join(a.out, 'movie.tif'), movie.numpy())
json.dump(dict(optics=OPT, noll=NOLL, zernike=ZC, roisize=P, z_range=ZR, focus_um=FOCUS, npupil=64, oversample=2,
               grid=GRID, chunk=CHUNK, T=300, H=96, W=112, n_events=len(truth)),
          open(os.path.join(a.out, 'movie.json'), 'w'), indent=1)

# detection (extract_spots) and the per-chunk noise / median as computed by detect_spots
sp = extract_spots(movie.numpy(), P, None, chunk=CHUNK, device=dev, threshold=6.0)
chunks = []
for t0 in range(0, movie.shape[0], CHUNK):
    x = torch.as_tensor(movie[t0:t0 + CHUNK].numpy()).to(dev).float()
    f = filter_frames(x)
    med = f.median()
    chunks.append(dict(t0=t0, median=float(med), noise=float(1.4826 * (f - med).abs().median())))
json.dump(dict(frame=sp.frame.tolist(), corner=sp.corner.tolist(), score=sp.score.tolist(), chunks=chunks),
          open(os.path.join(a.out, 'detect.json'), 'w'))
print('detections', len(sp), 'events', len(truth))

# localization of the first n spots on the spline
n = min(600, len(sp))
s = sp.subset(slice(0, n))
th, nll = localize(spl, s.rois)
cr = crlb(spl, th)
tab = localize_spots(s, spl)
np.savez(os.path.join(a.out, 'localize.npz'), knots=spl.knots.cpu().numpy(), rois=s.rois.cpu().numpy(),
         theta=th.cpu().numpy(), nll=nll.cpu().numpy(), crlb=cr.cpu().numpy(),
         **{'tab_' + k: (tab[k].astype(np.uint8) if tab[k].dtype == bool else tab[k]) for k in tab.dtype.names})
json.dump(dict(n=n, z0=spl.z0, hz=spl.hz, S=spl.S, nz=spl.nz, nv=spl.nv, nu=spl.nu, xmax=spl.xmax, c=spl.c,
               z_limits=spl.z_limits), open(os.path.join(a.out, 'localize.json'), 'w'), indent=1)
print('localized', n, 'median |x - truth| etc. not checked here')
