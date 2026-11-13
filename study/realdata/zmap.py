"""Median z per lateral bin (structure height map) and median z per block of frames (z drift + structure), PNG + json.
usage: python zmap.py locs.csv outdir [binpx]"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, binned_statistic

f, od = sys.argv[1], sys.argv[2]
bp = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
cols = open(f).readline().strip().split(',')
R = np.loadtxt(f, delimiter=',', skiprows=1, usecols=[cols.index(c) for c in ('frame', 'x', 'y', 'z', 'photons', 'flags')])
fr, x, y, z, ph, fl = R.T
g = (fl == 0) & (ph > 1000)
W, H = x.max(), y.max()
st = binned_statistic_2d(y[g], x[g], z[g], 'median', bins=[np.arange(0, H + bp, bp), np.arange(0, W + bp, bp)])
cnt = binned_statistic_2d(y[g], x[g], z[g], 'count', bins=[np.arange(0, H + bp, bp), np.arange(0, W + bp, bp)])
m = np.where(cnt.statistic >= 30, st.statistic, np.nan)
nb = np.arange(0, fr.max() + 2001, 2000)
zt = binned_statistic(fr[g], z[g], 'median', bins=nb).statistic
# z drift estimate independent of the lateral structure: z minus the height map of its lateral bin
ix = np.clip(((x[g]) // bp).astype(int), 0, m.shape[1] - 1); iy = np.clip(((y[g]) // bp).astype(int), 0, m.shape[0] - 1)
zr = z[g] - m[iy, ix]
ok = np.isfinite(zr)
ztr = binned_statistic(fr[g][ok], zr[ok], 'median', bins=nb).statistic
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im = ax[0].imshow(m, extent=(0, m.shape[1] * bp, m.shape[0] * bp, 0), cmap='coolwarm')
plt.colorbar(im, ax=ax[0], label='median z (nm)'); ax[0].set_title(f'median z per {bp:.0f} px bin (flags 0, photons > 1000)')
ax[1].plot(0.5 * (nb[1:] + nb[:-1]), zt, label='median z'); ax[1].plot(0.5 * (nb[1:] + nb[:-1]), ztr, label='median (z - height map)')
ax[1].set_xlabel('frame'); ax[1].set_ylabel('z (nm)'); ax[1].legend(); ax[1].set_title('z per 2000 frames (no drift correction)')
fig.tight_layout(); fig.savefig(f'{od}/z_map_time.png', dpi=110)
json.dump(dict(bin_px=bp, height_map_nm=np.where(np.isfinite(m), m, None).tolist(), frames=nb.tolist(), z_median_per_block=zt.tolist(),
               z_minus_map_median_per_block=ztr.tolist()), open(f'{od}/z_map_time.json', 'w'))
print('height map range (5-95 %):', np.nanquantile(m, [0.05, 0.95]).round(1).tolist(),
      ' z(t) - map: range', np.nanmin(ztr).round(1), np.nanmax(ztr).round(1))
