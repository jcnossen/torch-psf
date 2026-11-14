"""Figures of a bead calibration: width curves of the spline and of the beads, data / model / residual montages.
usage: python calib_fig.py calibdir [tag] [zunit_nm_per_unit]"""
import sys, json
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from torch_psf import ConvCubicSplinePSF
from rdcommon import fit_gauss, gauss_patch

cd = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else ''
fit = json.load(open(f'{cd}/bead_fit{tag}.json'))
dat = np.load(f'{cd}/bead_fit_data{tag}.npz')
cal = torch.load(f'{cd}/psf_spline{tag}.pt')
spl = ConvCubicSplinePSF(cal['PB'], (cal['zlo'], cal['zhi']), oversample=cal['S'], dz_um=cal['hz'], zpad=cal['zpad'], xy_limit=cal['xmax'], device='cuda')
spl.knots = cal['knots'].cuda()
# Gaussian widths of the spline and of the bead frames
zg = np.arange(cal['zlo'], cal['zhi'] + 1e-6, 0.025)
off = torch.zeros(len(zg), 3, device='cuda'); off[:, 2] = torch.tensor(zg, device='cuda', dtype=torch.float32)
g = spl.render(off).cpu().numpy()
ws = np.array([fit_gauss(p)[0][2:4] for p in g])
d, z, mu = dat['d'], dat['z'], dat['mu']
th = dat['th']
sel = np.arange(0, len(d), 3)
wb = np.array([fit_gauss(d[i] - th[i, 4])[0][2:4] for i in sel])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(zg * 1000, ws[:, 0], 'b-', label='spline sx'); ax[0].plot(zg * 1000, ws[:, 1], 'r-', label='spline sy')
ax[0].plot(z[sel] * 1000, wb[:, 0], 'b.', ms=2, alpha=0.4, label='bead frames sx'); ax[0].plot(z[sel] * 1000, wb[:, 1], 'r.', ms=2, alpha=0.4, label='bead frames sy')
ax[0].set_ylim(0, 6); ax[0].set_xlabel('z (stage nm)'); ax[0].set_ylabel('Gaussian sigma (px)'); ax[0].legend(fontsize=8)
c2 = ((d - mu) ** 2 / mu).reshape(len(d), -1).mean(1)
ax[1].plot(z * 1000, c2, '.', ms=2); ax[1].set_xlabel('z (stage nm)'); ax[1].set_ylabel('chi2 / pixel (bead frames)')
ax[1].set_ylim(0, max(3, np.quantile(c2, 0.99)))
used = fit['beads_used']
ax[2].scatter([b['x'] for b in used], [b['y'] for b in used], c='k', s=12, label='used')
if 'far_beads' in fit:
    fb = fit['far_beads']
    sc = ax[2].scatter([b['x'] for b in fb], [b['y'] for b in fb], c=[b['chi2'] for b in fb], s=20, cmap='viridis', label='held out (colour: chi2)')
    plt.colorbar(sc, ax=ax[2])
roi = json.load(open(f'{cd}/beads_found.json'))['roi']
if roi:
    ax[2].add_patch(plt.Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, color='r', label='movie ROI'))
ax[2].set_xlim(0, 512); ax[2].set_ylim(512, 0); ax[2].set_aspect(1); ax[2].legend(fontsize=7); ax[2].set_title('bead positions on the chip')
fig.tight_layout(); fig.savefig(f'{cd}/calib_widths{tag}.png', dpi=110); plt.close(fig)
# montage: one bead, data / model / residual at 7 heights
b0 = int(np.bincount(dat['bid']).argmax())
idx = np.nonzero(dat['bid'] == b0)[0]
zsel = [-0.7, -0.45, -0.2, 0, 0.2, 0.45, 0.7]
fig, ax = plt.subplots(3, len(zsel), figsize=(2.2 * len(zsel), 6.6))
for j, zz in enumerate(zsel):
    i = idx[np.argmin(np.abs(z[idx] - zz))]
    vmax = d[i].max()
    ax[0, j].imshow(d[i], vmin=0, vmax=vmax); ax[0, j].set_title(f'z {z[i] * 1000:.0f} nm', fontsize=8)
    ax[1, j].imshow(mu[i], vmin=0, vmax=vmax)
    r = (d[i] - mu[i]) / np.sqrt(mu[i])
    ax[2, j].imshow(r, vmin=-4, vmax=4, cmap='RdBu')
    for a in ax[:, j]:
        a.axis('off')
ax[0, 0].set_title('data\n' + ax[0, 0].get_title(), fontsize=8)
fig.suptitle('one bead: data / spline model / normalised residual (+-4)')
fig.tight_layout(); fig.savefig(f'{cd}/calib_montage{tag}.png', dpi=100); plt.close(fig)
print('ok')
