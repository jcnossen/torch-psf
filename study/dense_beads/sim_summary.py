"""Summary table (markdown, stdout) and figure (sim_summary.png) of the sim_compare.py runs.

  python sim_summary.py OUT_DIR sim_mixed_tilt.json sim_dense_tilt.json ...
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out, files = sys.argv[1], sys.argv[2:]
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
COL = dict(iso='#2a78d6', single='#eb6834', dense='#1baf7a')
LAB = dict(iso='isolated: ZStackEstimator', single='strictly isolated beads', dense='dense: all clusters')
plt.rcParams.update({'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'font.size': 9})
rows, names = [], []
print('| scenario | method | beads used (mean) | Zernike rms error mλ (mean, per seed) | held-out deviance / px - truth |')
print('|---|---|---|---|---|')
for f in files:
    J = json.load(open(os.path.join(out, f)))
    a = J['args']
    name = f"{a['layout']}, tilt {a['tilt']} µm, {a['photons']:.0f} ph"
    names.append(name)
    R = J['results']
    row = {}
    for m in ('iso', 'single', 'dense'):
        e = np.array([r[m]['rms_mwaves'] for r in R])
        nb = np.mean([r[m]['n_beads'] if m != 'dense' else r['dense']['report']['n_beads'] for r in R])
        dv = np.mean([r['heldout'][m]['deviance_per_px'] - r['heldout']['truth']['deviance_per_px'] for r in R])
        row[m] = e
        print(f'| {name} | {m} | {nb:.1f} | {e.mean():.2f} ({", ".join(f"{v:.2f}" for v in e)}) | {dv:+.4f} |')
    rows.append(row)
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(rows))
w = 0.26
for i, m in enumerate(('iso', 'single', 'dense')):
    mean = [r[m].mean() for r in rows]
    ax.bar(x + (i - 1) * w, mean, w * 0.9, color=COL[m], label=LAB[m], zorder=2)
    for xi, r in zip(x, rows):
        ax.scatter(np.full(len(r[m]), xi + (i - 1) * w), r[m], s=9, color=INK, zorder=3, lw=0)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('Zernike rms error (mλ)')
ax.grid(axis='y', color='#e4e3df', lw=0.6, zorder=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8)
ax.set_title('Simulated dense bead fields: Zernike error (bars: mean, dots: seeds)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(out, 'sim_summary.png'), dpi=120)
