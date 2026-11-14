"""Check of the z-precision comparison (CPU): repeat precision of z in 200 nm bins of the reference z, each method's z
converted to the reference axis with a straight line fitted to (z_ref, z_X) WITHIN the bin (clipped least squares),
instead of the smoothed 50 nm local slope of insitu_tub_analyze.py (which is fragile where a method warps its z axis).
Also the same pairs split by photons (reference) inside the bins.
usage: INSITU_DS=tubulin python insitu_tub_check.py out_tag label=tag ...  -> insitu/check_<out_tag>.json"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np
from insitu_common import OUT, jdump

otag = sys.argv[1]
args = [a for a in sys.argv[2:] if not a.startswith('--')]
labs = [a.split('=')[0] for a in args]
tags = [a.split('=')[1] for a in args]
P = np.load(f'{OUT}/repeat_pairs_{otag}.npz')
a, b = P['a'], P['b']
Z, I = {}, None
for l, t in zip(labs, tags):
    th = np.load(f'{OUT}/eval_{t}.npz')['th']
    Z[l] = th[:, 2] * 1000
    if I is None:
        I = th[:, 3]
zr = Z[labs[0]]
rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 20 else float('nan')
zp = 0.5 * (zr[a] + zr[b])
Ip = 0.5 * (I[a] + I[b])
res = dict(bins=[], note='repeat z spread / sqrt 2 in reference units, line fitted within each 200 nm bin of z_ref')
print('z bin | ' + ' | '.join(f'{l}: slope, all, >4k' for l in labs))
for lo in range(-600, 600, 200):
    q = (zp >= lo) & (zp < lo + 200)
    s_all = (zr >= lo) & (zr < lo + 200)
    row = dict(z=[lo, lo + 200], n_pairs=int(q.sum()), n_bright=int((q & (Ip > 4000)).sum()), methods={})
    for l in labs:
        x, y = zr[s_all], Z[l][s_all]
        keep = np.ones(len(x), bool)
        for _ in range(5):
            co = np.polyfit(x[keep], y[keep], 1)
            r = y - np.polyval(co, x)
            keep = np.abs(r) <= 3 * rs(r[keep]) + 1e-6
        sl = co[0]
        d = (Z[l][b] - Z[l][a]) / sl
        row['methods'][l] = dict(slope=float(sl), rep_all=rs(d[q]) / np.sqrt(2), rep_bright=rs(d[q & (Ip > 4000)]) / np.sqrt(2),
                                 rep_dim=rs(d[q & (Ip < 2000)]) / np.sqrt(2), rep_own_units_all=rs((Z[l][b] - Z[l][a])[q]) / np.sqrt(2))
    res['bins'].append(row)
    print(f'[{lo},{lo + 200}) n {row["n_pairs"]} | ' + ' | '.join(
        f'{l}: {m["slope"]:.2f} {m["rep_all"]:.1f} {m["rep_bright"]:.1f}' for l, m in row['methods'].items()))
jdump(res, f'{OUT}/check_{otag}.json')
