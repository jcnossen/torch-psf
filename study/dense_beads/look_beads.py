"""Montage of a few beads over the planes (PNG) and the per-plane peak / width of one bead."""
import sys, os, numpy as np, tifffile
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
f, out = sys.argv[1], sys.argv[2]
a = tifffile.imread(f).astype(np.float32)[:, :, :475]
mp = gaussian_filter(a.max(0), 1.5); mp -= np.median(mp)
pk = (mp == maximum_filter(mp, 31)) & (mp > 0.2 * mp.max())
y, x = np.nonzero(pk)
ok = (x > 20) & (y > 20) & (x < 455) & (y < 380)
y, x = y[ok][:6], x[ok][:6]
planes = list(range(0, 150, 15)) + [149]
fig, ax = plt.subplots(len(y), len(planes), figsize=(len(planes) * 1.3, len(y) * 1.3))
for i in range(len(y)):
    roi = a[:, y[i] - 15:y[i] + 16, x[i] - 15:x[i] + 16]
    for j, k in enumerate(planes):
        ax[i, j].imshow(roi[k], cmap='gray'); ax[i, j].axis('off')
        if i == 0: ax[i, j].set_title(str(k), fontsize=8)
    pkk = roi.reshape(150, -1).max(1)
    print(f'bead {i} at ({x[i]},{y[i]}): plane of max peak {pkk.argmax()}  peak range {pkk.min():.0f}-{pkk.max():.0f}')
plt.tight_layout(); plt.savefig(os.path.join(out, 'look_beads.png'), dpi=80)
