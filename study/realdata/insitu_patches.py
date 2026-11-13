"""Step 2: training / validation / test patches from the NPC movie.

Detections with flags == 0 in locs/NPC-A647-3D_locs.csv (bead-spline fit: offset not at the 3 px limit, z not at the
+-800 nm limit, patch inside the frame, no other detection in the patch, photons/bg not clamped). The patches are the
same 17 x 17 px photoelectron patches the bead-spline fit used (centre = detection peak, peak_x, peak_y), so every PSF is
fitted to identical data. Split by frame blocks of 1000 frames: even blocks -> training pool, odd blocks -> test, so
that one on-event (a few frames) is never in both. Training: a random 150,000 of the even-block spots; validation: a
further random 30,000 even-block spots (for choices such as the pupil-map roughness weight); test: ALL odd-block spots
(needed for consecutive-frame pairs and for the pore analysis).
Outputs in insitu/: patches_{train,val,test}.npy (float32 [n, 17, 17]) and meta_{train,val,test}.npz (the CSV columns)."""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np, pandas as pd
from rdcommon import open_stack
from insitu_common import D, OUT, GAIN, OFFSET, P

h = P // 2
df = pd.read_csv(f'{D}/locs/NPC-A647-3D_locs.csv')
print('localizations', len(df), flush=True)
df = df[df['flags'] == 0].reset_index().rename(columns={'index': 'row'})
blk = (df.frame // 1000).values
even = np.nonzero(blk % 2 == 0)[0]
odd = np.nonzero(blk % 2 == 1)[0]
rng = np.random.default_rng(0)
perm = rng.permutation(even)
sets = dict(train=np.sort(perm[:150000]), val=np.sort(perm[150000:180000]), test=odd)
print({k: len(v) for k, v in sets.items()}, 'even pool', len(even), flush=True)
mm = [open_stack(f'{D}/NPC-A647-3D-stack_{i}.tif') for i in (1, 2)]
T = [m.shape[0] for m in mm]
cum = np.cumsum([0] + T)
Hh, Ww = mm[0].shape[1:]
allidx = np.sort(np.concatenate(list(sets.values())))
which = np.empty(len(df), dtype='U5')
for k, v in sets.items():
    which[v] = k
pos_in = {k: np.full(len(df), -1) for k in sets}
for k, v in sets.items():
    pos_in[k][v] = np.arange(len(v))
arr = {k: np.lib.format.open_memmap(f'{OUT}/patches_{k}.npy', mode='w+', dtype=np.float32, shape=(len(v), P, P))
       for k, v in sets.items()}
fr = df.frame.values; px = df.peak_x.values.astype(int); py = df.peak_y.values.astype(int)
assert (px - h >= 0).all() and (px + h < Ww).all() and (py - h >= 0).all() and (py + h < Hh).all()
ii = np.arange(P)
t0 = time.time()
CH = 1000
order = allidx[np.argsort(fr[allidx], kind='stable')]
fo = fr[order]
for a in range(0, int(cum[-1]), CH):
    lo, hi = np.searchsorted(fo, a), np.searchsorted(fo, a + CH)
    if lo == hi:
        continue
    sel = order[lo:hi]
    k = int(np.searchsorted(cum, a, side='right') - 1)
    frames = np.asarray(mm[k][a - cum[k]:min(a + CH, cum[k + 1]) - cum[k]]).astype(np.float32)
    ph = (frames - OFFSET) / GAIN
    t = fr[sel] - a
    pat = ph[t[:, None, None], (py[sel] - h)[:, None, None] + ii[None, :, None], (px[sel] - h)[:, None, None] + ii[None, None, :]]
    for kk in sets:
        m = which[sel] == kk
        if m.any():
            arr[kk][pos_in[kk][sel[m]]] = pat[m]
    if a % 20000 == 0:
        print(f'frames {a}: {time.time() - t0:.0f}s', flush=True)
for k, v in sets.items():
    arr[k].flush()
    sub = df.iloc[v]
    np.savez(f'{OUT}/meta_{k}.npz', **{c: sub[c].values for c in sub.columns})
print('done', time.time() - t0, flush=True)
