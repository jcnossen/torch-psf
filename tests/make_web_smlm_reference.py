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

# ---------------------------------------------------------------------------------------------- in-situ fit
# a reference movie (simulate_reference_movie, astigmatic preset: Zernikes {5: 0.02, 6: 0.10, 11: 0.03}, focus 0.4 um,
# z 0 .. 0.8 um, camera gain 0.5 e-/ADU, offset 100, read noise 1.5 e-) and estimate_psf_insitu from a start PSF with
# only Noll 6 = 0.07
from torch_psf.simulate import simulate_reference_movie
from torch_psf.pipeline import estimate_psf_insitu
from torch_psf.insitu import knot_gradient, split_checkerboard
from torch_psf.spline import emitter_height_pos_fn
INS = dict(T=1500, H=96, W=96, emitters_per_frame=8.0, seed=5)
r = simulate_reference_movie('astigmatic', device=dev, **INS)
tifffile.imwrite(os.path.join(a.out, 'insitu_movie.tif'), r['movie_adu'])
truth_vec = r['psf']
start = VectorialPSF(15, zernike_indices=NOLL, npupil=64, oversample=3, device=dev, focus_um=0.4, z_range=(0.0, 0.8),
                     pixel_um=0.108, NA=1.49, wavelength_um=0.68, n_imm=1.518, n_med=1.33)
with torch.no_grad():
    start.zcoef[NOLL.index(6)] = 0.07
hist = []
est = estimate_psf_insitu(r['movie_adu'], psf=start, camera=r['camera'], roisize=15, z_range=(0.0, 0.8), nspots=3000,
                          nheldout=1500, iters=40, device=dev, log=lambda it, nll, s, zc: hist.append(dict(it=it, nll=nll, zcoef=zc.tolist())))
rep = est.report
json.dump(dict(params=r['params'], optics=dict(pixel_um=0.108, NA=1.49, wavelength_um=0.68, n_imm=1.518, n_med=1.33),
               noll=NOLL, truth_zernike={str(j): float(truth_vec.zcoef[truth_vec.zidx.index(j)]) for j in NOLL if j in truth_vec.zidx},
               start_zernike={'6': 0.07}, focus_um=0.4, z_range=[0.0, 0.8], roisize=15, nspots=3000, nheldout=1500, iters=40,
               final_zernike=est.psf.zcoef.tolist(), history=hist,
               report={k: v for k, v in rep.items() if isinstance(v, (int, float, str, dict, list)) or v is None}),
          open(os.path.join(a.out, 'insitu.json'), 'w'), indent=1, default=float)
print('insitu: spots', rep.get('n_spots'), 'train', rep.get('n_train'), 'final', np.round(est.psf.zcoef.tolist(), 4).tolist())
print('heldout', rep.get('heldout'), 'z_scale', rep.get('z_scale'), 'at_limit', rep.get('at_limit'))

# the gradient at the start: spline grid of estimate_psf_insitu (zpad 5), theta by localize, knot gradient, VJP
spl5 = ConvSplinePSF(15, (0.0, 0.8), oversample=2, dz_um=0.04, zpad=5, xy_limit=3.0, device=dev)
big = start.resized(spl5.Pl)
pf = emitter_height_pos_fn(0.4)
spl5.knots = spl5.knots_from_vectorial(big, pf, chunk=spl5.nz)
sp = extract_spots(r['movie_adu'], 15, r['camera'], device=dev, max_spots=2 * 4500, threshold=6.0)
par = split_checkerboard(sp.corner[:, 0] + 7, sp.corner[:, 1] + 7, 16)
tr = np.flatnonzero(par == 0)[:3000]
d = sp.rois[tr]
th, _ = localize(spl5, d)
gK, nll0 = knot_gradient(spl5, d, th)
gz = spl5.vjp_vectorial(big, pf, gK, chunk=spl5.nz)['zcoef']
np.savez(os.path.join(a.out, 'insitu_grad.npz'), knots=spl5.knots.cpu().numpy(), rois=d.cpu().numpy(),
         theta=th.cpu().numpy(), gK=gK.cpu().numpy(), gz=gz.cpu().numpy(), nll=np.float64(nll0),
         train_frame=sp.frame[tr].cpu().numpy(), train_corner=sp.corner[tr].cpu().numpy())
print('gradient: nll', nll0, 'gz', np.round(gz.cpu().numpy(), 2).tolist())
