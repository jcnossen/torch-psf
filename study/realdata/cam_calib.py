"""EMCCD conversion from the SMLM movie itself (no dark frames are available).

For an EMCCD at high EM gain, a pixel with mean m (ADU) has variance  var = 2 g (m - o) + r^2,  with g the overall
conversion (ADU per photoelectron, EM gain included), the factor 2 the excess noise factor F^2, o the offset (ADU) and
r the read noise (ADU). Per pixel and per block of frames the mean is the temporal median and the variance is taken
from differences of consecutive frames (removes the slow background drift): var = (1.4826 MAD(x[t+1]-x[t]))^2 / 2.
The median and the MAD ignore frames in which an emitter is on at that pixel (a minority of frames).
A straight line through (m, var) of all pixels and blocks gives slope 2g and intercept; o = -intercept/slope if r^2 is
negligible (checked: r^2 / slope is the error in o).
usage: python cam_calib.py out.json movie1.tif [movie2.tif ...]
"""
import sys, json
import numpy as np, tifffile

out, files = sys.argv[1], sys.argv[2:]
from rdcommon import open_stack
mm = [open_stack(f) for f in files]
T = [m.shape[0] for m in mm]
cum = np.cumsum([0] + T)
TT = cum[-1]
B = 400
starts = sorted(set([0, 1000, 5000] + list(np.linspace(0, TT - B, 14).astype(int))))
rows = []
for s in starts:
    k = np.searchsorted(cum, s, side='right') - 1
    s0 = s - cum[k]
    x = np.asarray(mm[k][s0:s0 + B], dtype=np.float64)
    med = np.median(x, 0)
    d = np.diff(x, axis=0)
    mad = np.median(np.abs(d - np.median(d, 0)), 0)
    var_mad = (1.4826 * mad) ** 2 / 2
    # clipped variance of the differences (no quantisation of the MAD): |d - median| < 3.5 sigma_MAD, corrected for the
    # truncation of a Gaussian at 3.5 sigma (factor 0.99491)
    sd = np.sqrt(2 * var_mad)[None]
    dc = d - np.median(d, 0)
    keep = np.abs(dc) < 3.5 * sd
    var = (np.where(keep, dc ** 2, 0).sum(0) / keep.sum(0)) / 2 / 0.99491
    # trimmed mean as a second mean estimate: frames above median + 4 sigma removed (emitters)
    thr = med + 4 * np.sqrt(var)
    xm = np.where(x < thr, x, np.nan)
    tmean = np.nanmean(xm, 0)
    rows.append((s, med.ravel(), tmean.ravel(), var.ravel()))
    print(f'block {s:6d}: median {np.median(med):7.1f}  var {np.median(var):8.1f}  var/median slope-guess', flush=True)
m = np.concatenate([r[1] for r in rows]); tm = np.concatenate([r[2] for r in rows]); v = np.concatenate([r[3] for r in rows])
res = {}
for name, mx in (('median', m), ('trimmed_mean', tm)):
    # robust line: bin by mean, take the median variance per bin, weighted least squares over bins
    q = np.quantile(mx, np.linspace(0.01, 0.99, 41))
    bc, bv, bn = [], [], []
    for a, b in zip(q[:-1], q[1:]):
        s = (mx >= a) & (mx < b)
        if s.sum() > 50:
            bc.append(np.median(mx[s])); bv.append(np.median(v[s])); bn.append(s.sum())
    bc, bv = np.array(bc), np.array(bv)
    A = np.stack([np.ones_like(bc), bc], 1)
    co, *_ = np.linalg.lstsq(A, bv, rcond=None)
    slope, icpt = co[1], co[0]
    # also a plain least squares on all points (outlier pixels with emitters inflate it)
    co2 = np.polyfit(mx, v, 1)
    res[name] = dict(slope=float(slope), intercept=float(icpt), g_ADU_per_e=float(slope / 2),
                     offset_ADU=float(-icpt / slope), slope_allpoints=float(co2[0]), offset_allpoints=float(-co2[1] / co2[0]),
                     mean_range=[float(bc.min()), float(bc.max())],
                     bins=[[float(a), float(b)] for a, b in zip(bc, bv)])
    print(name, {k: res[name][k] for k in res[name] if k != 'bins'})
    print('  bins (mean, var, line residual %):', [(round(a), round(b), round(100 * (b - icpt - slope * a) / b, 2)) for a, b in zip(bc, bv)][::4])
# the median of the MAD-based variance is biased low for a skewed (gamma-like) distribution; check the ratio of the
# MAD-based variance to the plain variance of the differences on the cleanest pixels (lowest 20 % of variance)
# per-pixel temporal regression: each pixel's own (mean, var) over the blocks (the background changes in time), which
# removes pixel-to-pixel differences in content (structure with many blinking emitters); robust medians over pixels
import os
skip = int(os.environ.get('CAL_SKIP', '0'))        # blocks starting before this frame are left out of the temporal fit
Mb = np.stack([r[1] for r in rows if r[0] >= skip]); Vb = np.stack([r[3] for r in rows if r[0] >= skip])   # [blocks, px]
res['temporal_skip_frames'] = skip
mc = Mb - Mb.mean(0); vc = Vb - Vb.mean(0)
rng = Mb.max(0) - Mb.min(0)
sl = (mc * vc).sum(0) / (mc ** 2).sum(0).clip(1e-9)
icp = Vb.mean(0) - sl * Mb.mean(0)
for lim in (20, 50, 100, 200):
    s_ = rng > lim
    if s_.sum() > 100:
        res[f'per_pixel_temporal_range>{lim}'] = dict(n_pixels=int(s_.sum()), slope_median=float(np.median(sl[s_])),
                                                     g_ADU_per_e=float(np.median(sl[s_]) / 2),
                                                     offset_ADU=float(np.median(-icp[s_] / sl[s_])))
        print(f'per-pixel temporal fits (mean range > {lim} ADU, {int(s_.sum())} px):', res[f'per_pixel_temporal_range>{lim}'])
res['n_points'] = int(len(m)); res['blocks'] = [int(s) for s in starts]; res['block_len'] = B
json.dump(res, open(out, 'w'), indent=1)
