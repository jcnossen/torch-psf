"""Localizations per tile of the FD-DeepLoc example movie (from their csv), and how many are isolated: no other
localization in the same frame within `iso` px (Chebyshev distance) -- a proxy for the spots an in-situ fit with
iso x iso ROIs can use.

    python -I tile_counts.py LOCS_CSV OUT_JSON [--pixel_nm 110] [--grids 4 6] [--iso 31 41]
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('csv'); ap.add_argument('out')
ap.add_argument('--pixel_nm', type=float, default=110.0)
ap.add_argument('--size', type=int, default=1608)
ap.add_argument('--grids', type=int, nargs='*', default=[4, 6])
ap.add_argument('--iso', type=int, nargs='*', default=[31, 41])
a = ap.parse_args()
df = pd.read_csv(a.csv)
x = df['xnano'].values / a.pixel_nm - 0.5          # px, pixel centres at integers (xnano = (0.5 + col + dx) px)
y = df['ynano'].values / a.pixel_nm - 0.5
f = df['frame'].values.astype(int)
z = df['znano'].values
res = dict(n=len(df), frames=int(f.max()), pixel_nm=a.pixel_nm)
nn = np.full(len(df), np.inf)
for fr in np.unique(f):
    i = np.flatnonzero(f == fr)
    if len(i) < 2:
        continue
    t = cKDTree(np.stack([x[i], y[i]], 1))
    d, _ = t.query(np.stack([x[i], y[i]], 1), k=2, p=np.inf)
    nn[i] = d[:, 1]
for g in a.grids:
    T = a.size / g
    tx, ty = np.clip((x // T).astype(int), 0, g - 1), np.clip((y // T).astype(int), 0, g - 1)
    tab = dict(tile_px=T, all=np.zeros((g, g), int).tolist())
    cnt = np.zeros((g, g), int); np.add.at(cnt, (ty, tx), 1); tab['all'] = cnt.tolist()
    for r in a.iso:
        c = np.zeros((g, g), int); ok = nn > r
        np.add.at(c, (ty[ok], tx[ok]), 1); tab[f'isolated_{r}'] = c.tolist()
    zm = np.full((g, g), np.nan)
    for iy in range(g):
        for ix in range(g):
            s = (ty == iy) & (tx == ix)
            if s.any():
                zm[iy, ix] = np.median(z[s])
    tab['z_median_nm'] = np.round(zm, 0).tolist()
    res[f'grid{g}'] = tab
    print(f'grid {g}x{g} ({T:.0f} px):'); print(' all\n', cnt)
    for r in a.iso:
        print(f' isolated (>{r} px)\n', np.array(tab[f"isolated_{r}"]))
    print(' median z (nm)\n', np.round(zm, 0))
print('fraction isolated', {r: float((nn > r).mean()) for r in a.iso})
json.dump(res, open(a.out, 'w'), indent=1)
