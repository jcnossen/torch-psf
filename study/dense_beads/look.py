"""Quick look at a bead stack: shape, value ranges, projections (PNG)."""
import sys, os, numpy as np, tifffile
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
f, out = sys.argv[1], sys.argv[2]
a = tifffile.imread(f)
print(f, a.shape, a.dtype, a.min(), a.max(), np.percentile(a, [1, 5, 50, 95, 99.9]))
mp, mean = a.max(0), a.mean(0)
print('frame means first/last', a[:5].mean((1, 2)), a[-5:].mean((1, 2)))
fig, ax = plt.subplots(2, 1, figsize=(14, 12))
ax[0].imshow(np.log(mp - mp.min() + 1), cmap='gray'); ax[0].set_title('log max proj')
ax[1].imshow(mean, cmap='gray', vmax=np.percentile(mean, 99.5)); ax[1].set_title('mean')
plt.tight_layout(); plt.savefig(os.path.join(out, os.path.basename(f) + '_proj.png'), dpi=80)
# row/col profiles of the mean
print('row profile (every 25):', mean.mean(1)[::25].round(1))
print('col profile (every 25):', mean.mean(0)[::25].round(1))
