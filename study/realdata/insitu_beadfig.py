"""Figure of the bead fits: chi2/pixel per z (free-knot spline vs vectorial variants) and a montage of one bead.
usage: python insitu_beadfig.py tag1 tag2 ... -> insitu/fig_beads.png"""
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from insitu_common import D, OUT

f = np.load(f'{D}/locs/calib/bead_fit_data.npz')
sel = np.abs(f['z']) <= 0.9 + 1e-6
d, z, bid, mus = f['d'][sel], f['z'][sel], f['bid'][sel], f['mu'][sel]
tags = sys.argv[1:]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
zb = np.round(z / 0.1) * 0.1
u = np.unique(zb)
c2s = ((d - mus) ** 2 / np.clip(mus, 1e-3, None)).mean((1, 2))
ax[0].plot(u, [c2s[zb == v].mean() for v in u], 'k-o', label='free-knot spline (beads)')
for t in tags:
    mu = np.load(f'{OUT}/vecbeads_{t}_mu.npy')
    c2 = ((d - mu) ** 2 / np.clip(mu, 1e-3, None)).mean((1, 2))
    ax[0].plot(u, [c2[zb == v].mean() for v in u], '-o', ms=3, label=f'vectorial {t}')
ax[0].axhline(1, color='k', lw=0.5); ax[0].set_xlabel('z (stage um)'); ax[0].set_ylabel('chi2 / pixel'); ax[0].legend(fontsize=7)
# radial residual profile, pooled: mean (d - mu) / sqrt(mu) against distance from the patch centre
r = np.hypot(*np.meshgrid(np.arange(23) - 11, np.arange(23) - 11))
for t in [None] + tags:
    mu = mus if t is None else np.load(f'{OUT}/vecbeads_{t}_mu.npy')
    res = ((d - mu) ** 2 / np.clip(mu, 1e-3, None)).mean(0)
    rb = np.arange(0, 12)
    ax[1].plot(rb, [res[(r >= a) & (r < a + 1)].mean() for a in rb], '-o', ms=3, label='spline' if t is None else t)
ax[1].set_xlabel('distance from patch centre (px)'); ax[1].set_ylabel('chi2 / pixel'); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_beads_chi2.png', dpi=120); plt.close(fig)

# montage of the bead with the most frames: data, spline, vectorial (last tag), residuals (in sigma)
b = np.bincount(bid).argmax()
mu = np.load(f'{OUT}/vecbeads_{tags[-1]}_mu.npy')
ks = np.nonzero(bid == b)[0]
ks = ks[np.argsort(z[ks])][::4]
fig, ax = plt.subplots(5, len(ks), figsize=(1.6 * len(ks), 8.2))
for j, k in enumerate(ks):
    vm = d[k].max()
    for i, (img, cm, lim) in enumerate([(d[k], 'gray', (0, vm)), (mus[k], 'gray', (0, vm)), (mu[k], 'gray', (0, vm)),
                                         ((d[k] - mus[k]) / np.sqrt(np.clip(mus[k], 1, None)), 'RdBu', (-5, 5)),
                                         ((d[k] - mu[k]) / np.sqrt(np.clip(mu[k], 1, None)), 'RdBu', (-5, 5))]):
        ax[i, j].imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1]); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[0, j].set_title(f'z {z[k]:+.2f}', fontsize=8)
for i, t in enumerate(['data', 'spline', f'vec {tags[-1]}', 'resid spline', 'resid vec']):
    ax[i, 0].set_ylabel(t, fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_beads_montage.png', dpi=110); plt.close(fig)
print('saved')
