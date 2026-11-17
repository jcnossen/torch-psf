"""EPFL SOFI 2D (Tubulin-COS7-Flip565): PSF estimate on a ROI, then the PSF-aware SOFI reconstruction with that PSF,
compared with the mean image and standard cross-cumulant SOFI2 (with / without Fourier reweighting).

python epfl_sofi2d_recon.py [--roi y0 x0 H W]  -> ~/sofi_psf_out/epfl_sofi2d_recon.png
"""
import argparse
from common import *
from torch_psf.sofi import SOFIEstimator
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--movie', default='/mnt/data/SMLM_challenge/SOFI/Tubulin-COS7-Flip565-2D/Tubulin_SOFI_2D_flip.tif')
ap.add_argument('--roi', type=int, nargs=4, default=[170, 170, 160, 160])
ap.add_argument('--l1', type=float, default=0.0)
ap.add_argument('--smooth', type=float, default=0.0)
ap.add_argument('--tag', default='')
a = ap.parse_args()
OPT = dict(pixel_um=0.108, NA=1.27, wavelength_um=0.60, n_imm=1.33, n_med=1.40)
est = SOFIEstimator(optics=OPT, roi=tuple(a.roi), iters=40, nboot=100, progress=lambda *x: None)
t0 = time.time()
psf = est.estimate(a.movie)
print(f'estimate {time.time() - t0:.0f}s dz {est.report["dz_um"] * 1e3:.0f} nm ' +
      ' '.join(f'{j}:{1e3 * v:+.0f}' for j, v in est.report['zernike'].items()), flush=True)
t0 = time.time()
r = est.reconstruct(os=4, l1=a.l1, smooth=a.smooth, iters=800)
print(f'reconstruct {time.time() - t0:.0f}s', flush=True)
flat = tp.VectorialPSF(13, zernike_indices=tuple(NOLL), device=DEV, **OPT)
rf = est.reconstruct(os=4, l1=a.l1, smooth=a.smooth, iters=800, psf=flat, standard=False)
panels = [('mean', r['mean']), ('SOFI2 (cross-cumulant)', r['sofi2']), ('SOFI2 + Fourier reweighting', r['sofi2_rw']),
          ('PSF-aware, estimated PSF', r['image'][0]), ('PSF-aware, aberration-free PSF', rf['image'][0])]
fig, ax = plt.subplots(2, len(panels), figsize=(4 * len(panels), 8.4))
H = a.roi[2]
zy0, zx0, zs = 40, 40, 40                                         # zoom box (px)
for k, (name, img) in enumerate(panels):
    im = img.cpu().numpy()
    up = im.shape[0] / H
    v = np.percentile(im, 99.7)
    ax[0, k].imshow(im, cmap='magma', vmin=np.percentile(im, 1) if k == 0 else 0, vmax=v, interpolation='nearest')
    ax[0, k].add_patch(plt.Rectangle((zx0 * up, zy0 * up), zs * up, zs * up, fill=False, ec='c', lw=0.8))
    ax[0, k].set_title(name, fontsize=10); ax[0, k].axis('off')
    z = im[int(zy0 * up):int((zy0 + zs) * up), int(zx0 * up):int((zx0 + zs) * up)]
    ax[1, k].imshow(z, cmap='magma', vmin=np.percentile(z, 1) if k == 0 else 0, vmax=np.percentile(z, 99.7),
                    interpolation='nearest')
    ax[1, k].axis('off')
fig.suptitle(f'EPFL Tubulin-COS7-Flip565 SOFI (8000 frames), ROI {a.roi}; zoom {zs} px = {zs * 0.108:.1f} um', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, f'epfl_sofi2d_recon{a.tag}.png'), dpi=100)
