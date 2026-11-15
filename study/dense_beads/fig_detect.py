"""Detection / grouping overview (PNG) from OUT/detections.npy written by real_fit.py.

  python fig_detect.py OUT_DIR
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out = sys.argv[1]
det = np.load(os.path.join(out, 'detections.npy'), allow_pickle=True)
C_KEPT, C_DROP, C_PT = '#1baf7a', '#eb6834', '#2a78d6'
fig, axs = plt.subplots(1, len(det), figsize=(8 * len(det), 7.2), facecolor='#fcfcfb')
for i, (ax, d) in enumerate(zip(np.atleast_1d(axs), det)):
    mp = d['mp']
    ax.imshow(np.log(np.clip(mp - np.percentile(mp, 1), 1, None)), cmap='gray')
    if d['mask'] is not None:
        ax.contour(d['mask'], levels=[0.5], colors=['#c98500'], linewidths=0.8)
    for (x0, y0, x1, y1) in d['kept']:
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=C_KEPT, lw=1.2))
    for (x0, y0, x1, y1), why in d['dropped']:
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=C_DROP, lw=0.8, ls='--'))
        ax.text(x0, y0 - 1, why, color=C_DROP, fontsize=6)
    ax.scatter(d['xy'][:, 0], d['xy'][:, 1], s=8, c=C_PT, lw=0)
    ax.set_title(f'stack {i}: {len(d["xy"])} detections; green = clusters used, orange dashed = dropped',
                 fontsize=10, color='#0b0b0b')
    ax.set_xlim(-0.5, mp.shape[1] - 0.5); ax.set_ylim(mp.shape[0] - 0.5, -0.5)
plt.tight_layout()
plt.savefig(os.path.join(out, 'detections.png'), dpi=110, facecolor='#fcfcfb')
