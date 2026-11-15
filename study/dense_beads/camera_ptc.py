"""Photon transfer from a bead z-stack: per pixel mean over a block of planes and the variance from adjacent-plane
differences (the PSF hardly changes over 10 nm), binned by mean; line var = k (mean - offset) + c.
Also the dark level outside the field stop."""
import sys, numpy as np, tifffile
f = sys.argv[1]
x0, x1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, None)
a = tifffile.imread(f).astype(np.float64)[:, :, x0:x1]
nk = a.shape[0]
res = []
for b0 in range(0, nk - 30, 30):
    blk = a[b0:b0 + 30]
    mu = blk.mean(0)
    d = np.diff(blk, axis=0)
    # remove the slow per-pixel trend in the differences (focus change): subtract the mean difference
    v = ((d - d.mean(0)) ** 2).sum(0) / (d.shape[0] - 1) / 2
    m, vv = mu.ravel(), v.ravel()
    lo, hi = np.percentile(m, [1, 99.5])
    e = np.quantile(m[(m > lo) & (m < hi)], np.linspace(0, 1, 41))
    xm, xv = [], []
    for i in range(40):
        s = (m >= e[i]) & (m < e[i + 1])
        if s.sum() > 20:
            xm.append(np.median(m[s])); xv.append(np.median(vv[s]))
    k, c = np.polyfit(xm, xv, 1)
    res.append((k, c, -c / k))
    print(f'planes {b0}-{b0+30}: slope {k:.4f} ADU/ADU  intercept {c:.2f}  offset(var=0) {-c/k:.1f}  bins mean {xm[0]:.0f}..{xm[-1]:.0f}')
r = np.array(res)
print('median slope', np.median(r[:, 0]), '-> gain e/ADU (Poisson eq.)', 1 / np.median(r[:, 0]), 'offset', np.median(r[:, 2]))
mean = a.mean(0)
print('dark level: 0.5/2/5 percentiles of the mean image', np.percentile(mean, [0.5, 2, 5]))
