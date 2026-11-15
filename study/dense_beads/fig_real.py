"""Figures of the real-data comparison (from the outputs of real_fit.py in OUT_DIR):
  clusters.png   data / model / normalized residual of dense clusters at a few planes (dense fit)
  zernikes.png   Zernike coefficients: isolated (ZStackEstimator), strictly isolated beads, dense
  regions.png    the main Zernikes of the separate fits of n x n field regions (field dependence)
  deviance.png   per-cluster Poisson deviance per pixel of the global dense fit over the field

  python fig_real.py OUT_DIR
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1]
res = json.load(open(os.path.join(out, 'real_results.json')))
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'font.size': 9})
NAMES = {5: 'astig 45°', 6: 'astig 0°', 7: 'coma y', 8: 'coma x', 9: 'trefoil y', 10: 'trefoil x', 11: 'spherical',
         12: '2nd astig 0°', 13: '2nd astig 45°', 14: 'quadrafoil', 15: 'quadrafoil', 16: '2nd coma x',
         17: '2nd coma y', 18: '2nd trefoil', 19: '2nd trefoil', 20: 'pentafoil', 21: 'pentafoil'}
zidx = res['dense']['zidx']

# ---- clusters: data / model / residual
f = np.load(os.path.join(out, 'clusters_fig.npz'))
ids = f['ids'][:4]
focus = f['focus']
nk = len(focus)
planes = np.linspace(0, nk - 1, 5).round().astype(int)
fig, axs = plt.subplots(3 * len(ids), len(planes), figsize=(1.9 * len(planes) + 0.6, 1.75 * 3 * len(ids)))
for r, c in enumerate(ids):
    d, m = f[f'data_{c}'], f[f'model_{c}']
    box = res['clusters'][int(c)]['box']
    for j, k in enumerate(planes):
        vmax = np.nanmax(d[k])
        res_k = (d[k] - m[k]) / np.sqrt(np.clip(m[k], 1, None))
        for i, (img, kw) in enumerate([(d[k], dict(cmap='gray', vmin=np.nanmin(m[k]), vmax=vmax)),
                                       (m[k], dict(cmap='gray', vmin=np.nanmin(m[k]), vmax=vmax)),
                                       (res_k, dict(cmap='RdBu_r', vmin=-4, vmax=4))]):
            ax = axs[3 * r + i, j]
            ax.imshow(np.ma.masked_invalid(img), **kw)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(['data', 'model', 'residual'][i] + (f'\ncluster {c} ({res["clusters"][int(c)]["n"]} beads)'
                                                                if i == 0 else ''), fontsize=8)
            if r == 0 and i == 0:
                ax.set_title(f'focus {focus[k]:+.2f} µm', fontsize=8)
fig.suptitle('Dense fit, training clusters of beads.tif (left channel): data, model, (d - mu) / sqrt(mu) in ±4',
             fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.985))
plt.savefig(os.path.join(out, 'clusters.png'), dpi=110)
plt.close()

# ---- Zernike bars
series = [('iso8', 'isolated: ZStackEstimator (nbeads=8)', S1), ('single', 'strictly isolated beads, joint fitter', S2),
          ('dense', 'dense: all clusters', S3)]
# ZStackEstimator's n_beads counts both stacks (beads2.tif held out): the training beads are those of beads.tif
n_beads = dict(iso8=res['iso8']['report']['n_beads'] // 2, single=res['single']['n_beads'],
               dense=res['dense']['report']['n_beads'])
x = np.arange(len(zidx))
w = 0.27
fig, ax = plt.subplots(figsize=(13, 4.4))
for i, (k, lab, col) in enumerate(series):
    z = np.array(res[k]['zcoef']) * 1e3
    ax.bar(x + (i - 1) * w, z, w * 0.9, color=col, label=f'{lab}, ~{n_beads[k]} training beads', zorder=2)
ax.axhline(0, color=INK2, lw=0.8, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f'Z{j}\n' + NAMES.get(j, '').replace(' ', '\n', 1) for j in zidx], fontsize=6.5)
ax.set_ylabel('coefficient (mλ rms)')
ax.grid(axis='y', color='#e4e3df', lw=0.6, zorder=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8)
ho = res['heldout']
ax.set_title('Zernikes, left channel, trained on beads.tif. Held-out deviance / px on beads2.tif clusters: ' +
             ', '.join(f'{k} {ho[k]["deviance_per_px"]:.3f}' for k in ('iso8', 'iso40', 'single', 'dense', 'start')),
             fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(out, 'zernikes.png'), dpi=120)
plt.close()

# ---- regions: main Zernikes per region (small multiples, diverging, symmetric per panel)
reg = [r for r in res.get('regions', []) if 'zcoef' in r]
if reg:
    n = max(max(r['ix'], r['iy']) for r in res['regions']) + 1
    main = [5, 6, 7, 8, 9, 10, 11, 12]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.4))
    zg = np.array(res['dense']['zcoef']) * 1e3
    for ax, j in zip(axs.ravel(), main):
        g = np.full((n, n), np.nan)
        for r in reg:
            g[r['iy'], r['ix']] = r['zcoef'][zidx.index(j)] * 1e3
        lim = max(np.nanmax(np.abs(g)), 1e-3)
        im = ax.imshow(g, cmap='RdBu_r', vmin=-lim, vmax=lim)
        for r in reg:
            ax.text(r['ix'], r['iy'], f'{g[r["iy"], r["ix"]]:+.0f}\n({r["n_beads"]})', ha='center', va='center',
                    fontsize=7, color=INK)
        ax.set_title(f'Z{j} {NAMES[j]} (global {zg[zidx.index(j)]:+.0f})', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f'Separate dense fits of {n} x {n} field regions (left channel, both stacks): mλ rms (beads)',
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(out, 'regions.png'), dpi=110)
    plt.close()

# ---- deviance per cluster over the field
det = np.load(os.path.join(out, 'detections.npy'), allow_pickle=True)
cl = res['clusters']
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
vals = np.array([c['deviance'] for c in cl if np.isfinite(c['deviance'])])
vmin, vmax = np.percentile(vals, [2, 98])
for s, ax in enumerate(axs):
    mp = det[s]['mp']
    ax.imshow(np.log(np.clip(mp - np.percentile(mp, 1), 1, None)), cmap='gray', alpha=0.6)
    cs = [c for c in cl if c['stack'] == s and np.isfinite(c['deviance'])]
    cx = [np.mean(np.array(c['xy'])[:, 0]) for c in cs]
    cy = [np.mean(np.array(c['xy'])[:, 1]) for c in cs]
    sc = ax.scatter(cx, cy, c=[c['deviance'] for c in cs], s=[30 + 12 * c['n'] for c in cs], cmap='viridis',
                    vmin=vmin, vmax=vmax, edgecolors=SURF, linewidths=1.5)
    ax.set_title(f'{"beads.tif (training)" if s == 0 else "beads2.tif (held out, refitted)"}: '
                 'deviance / px per cluster (global PSF)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(sc, ax=axs, fraction=0.025, pad=0.02, label='Poisson deviance per pixel')
plt.savefig(os.path.join(out, 'deviance.png'), dpi=110)
plt.close()
print('figures written to', out)
