"""First real-data test: EPFL SRM hub SOFI set Tubulin-COS7-Flip565-2D (Grussmayer et al., ACS Nano 2020, Fig. 1):
8000 frames 500 x 500 px, ORCA Flash 4.0, 20 ms, 108 nm px, Nikon 60x/1.27 NA water, Flip565 in 50 % glycerol.
SOFIEstimator on several ROIs; with no ground truth the check is the agreement between ROIs (and between the two
halves of the movie). The movie is used in ADU (a constant offset does not enter the covariances; the gain only
scales them). Data: https://srm.epfl.ch/datasets/real/LBEN/SOFI/Tubulin-COS7-Flip565-2D/Tubulin_SOFI_2D_flip.tif

python epfl_sofi2d.py [--movie PATH] [--roi 96] [--frames 0 8000]
"""
import argparse
from common import *
from torch_psf.sofi import SOFIEstimator

ap = argparse.ArgumentParser()
ap.add_argument('--movie', default='/mnt/data/SMLM_challenge/SOFI/Tubulin-COS7-Flip565-2D/Tubulin_SOFI_2D_flip.tif')
ap.add_argument('--roi', type=int, default=96)
ap.add_argument('--frames', type=int, nargs=2, default=None)
ap.add_argument('--iters', type=int, default=40)
ap.add_argument('--tag', default='')
a = ap.parse_args()
OPT = dict(pixel_um=0.108, NA=1.27, wavelength_um=0.60, n_imm=1.33, n_med=1.40)
S = a.roi
centres = [(250, 250), (130, 130), (130, 370), (370, 130), (370, 370)]
out = {}
for cy, cx in centres:
    roi = (cy - S // 2, cx - S // 2, S, S)
    est = SOFIEstimator(optics=OPT, roi=roi, iters=a.iters, block=500, nboot=100,
                        progress=lambda *x: None)
    t0 = time.time()
    psf = est.estimate(a.movie, frames=a.frames)
    r = est.report
    z = r['zernike']
    print(f"roi {roi}: {time.time() - t0:.0f}s dz {r['dz_um'] * 1e3:.0f} nm  " +
          ' '.join(f"{j}:{1e3 * z[j]:+.0f}" for j in sorted(z)), flush=True)
    out[str(roi)] = dict(roi=roi, zernike=z, dz_um=r['dz_um'], loss=r['loss'], n_frames=r['n_frames'])
save_json(out, f'epfl_sofi2d{a.tag}.json')
