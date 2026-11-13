"""Step 4b: held-out metrics of the PSFs on the test set (CPU).
usage: python insitu_analyze.py out_tag label1=tag1 label2=tag2 ...   (the first label is the reference, the bead spline)
-> insitu/results_<out_tag>.json and PNGs insitu/fig_<out_tag>_*.png

  likelihood  deviance of the halved data and model per spot (EMCCD-correct log-likelihood ratio against the saturated
              model; lower is better; all PSFs have the same 5 free parameters per spot), chi2/pixel (variance 2 mu);
              paired difference against the reference with its standard error; binned by z and by photons (bins from
              the reference fit, so that every method is compared on the same spots).
  repeat      same emitter in consecutive frames (mutual nearest neighbours within 1 px in the reference fit, both
              frames in the same test block; one fixed pair set for every method): robust std of the difference / sqrt 2
              against the median CRLB of the method, and the robust std of the CRLB-normalised difference.
  pores       time windows of 10,000 movie frames (5,000 test frames); in each window, nuclear-pore candidates from the
              reference localizations (maxima of a 0.4 px-blurred density, localizations within 1.2 px, at least 15,
              algebraic circle fit with radius 0.2-1 px and centre within 0.5 px of the maximum). Per pore and window
              each method's own centre (circle fit to the SAME spots in that method's coordinates) and its own median
              z are removed, so drift between windows does not enter. Lateral: fitted radius in px (x, y scaled to nm
              with the bead-fit pixel sizes) and its ratio to the reference. Axial: residual z of bright spots
              (photons > 2000) pooled over pores; robust spread, and a two-Gaussian fit (equal widths) against one.
  z scale     z_X against z_ref (|z_ref| < 600 nm, iterative 3-sigma clipped least squares) and the mean lateral
              difference x_X - x_ref, y_X - y_ref per 100 nm of z_ref.
"""
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import minimize
from insitu_common import OUT, jdump, XMAX, ZLIM

otag = sys.argv[1]
opts = [a for a in sys.argv[2:] if a.startswith('--')]
args = [a for a in sys.argv[2:] if not a.startswith('--')]
labs = [a.split('=')[0] for a in args]
tags = [a.split('=')[1] for a in args]
W = 10000
MINL = 15
for o in opts:
    k, v = o[2:].split('=')
    if k == 'window': W = int(v)
    if k == 'minlocs': MINL = int(v)
M = np.load(f'{OUT}/meta_test.npz')
try:
    vb = json.load(open(f'{OUT}/vecbeads_best.json'))
    PXN, PYN = vb['pxy'][0] * 1000, vb['pxy'][1] * 1000
except FileNotFoundError:
    PXN, PYN = 134.9, 142.0
E = {}
for l, t in zip(labs, tags):
    e = np.load(f'{OUT}/eval_{t}.npz')
    th = e['th']
    E[l] = dict(x=M['peak_x'] + th[:, 0], y=M['peak_y'] + th[:, 1], z=th[:, 2] * 1000, I=th[:, 3], bg=th[:, 4],
                ox=th[:, 0], oy=th[:, 1], dev=e['dev'].astype(np.float64), chi2=e['chi2'].astype(np.float64),
                crx=e['crlb'][:, 0], cry=e['crlb'][:, 1], crz=e['crlb'][:, 2] * 1000, sec=float(e['seconds']))
ref = labs[0]
R = E[ref]
fr = M['frame']
n = len(fr)
res = dict(n_test=n, labels=dict(zip(labs, tags)), pixel_nm_from_beads=[PXN, PYN], methods={})
rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 5 else float('nan')

# ------------------------------------------------------------------------------------------------ fit limits / likelihood
zb = np.arange(-800, 801, 200)
pb = [0, 1000, 2000, 4000, 8000, 1e9]
zr, Ir = R['z'], R['I']
for l in labs:
    e = E[l]
    xy = np.maximum(np.abs(e['ox']), np.abs(e['oy'])) > XMAX - 0.06
    zl = np.abs(e['z']) >= ZLIM * 1000 - 5
    dd = e['dev'] - R['dev']
    m = dict(limits=dict(xy=float(xy.mean()), z=float(zl.mean()), any=float((xy | zl).mean())),
             dev_per_spot=float(e['dev'].mean()), dev_per_pixel=float(e['dev'].mean() / 289),
             chi2_mean=float(e['chi2'].mean()), chi2_median=float(np.median(e['chi2'])),
             ddev_vs_ref=float(dd.mean()), ddev_vs_ref_se=float(dd.std() / np.sqrt(n)),
             frac_spots_better_than_ref=float((dd < 0).mean()),
             photons_ratio_median=float(np.median(e['I'] / R['I'])), seconds_test=e['sec'])
    m['by_z'] = []
    for a, b in zip(zb[:-1], zb[1:]):
        s = (zr >= a) & (zr < b)
        if s.sum() > 100:
            m['by_z'].append(dict(z=[float(a), float(b)], n=int(s.sum()), ddev=float(dd[s].mean()),
                                  ddev_se=float(dd[s].std() / np.sqrt(s.sum())), chi2=float(e['chi2'][s].mean()),
                                  dev=float(e['dev'][s].mean())))
    m['by_photons'] = []
    for a, b in zip(pb[:-1], pb[1:]):
        s = (Ir >= a) & (Ir < b)
        if s.sum() > 100:
            m['by_photons'].append(dict(I=[a, b], n=int(s.sum()), ddev=float(dd[s].mean()),
                                        ddev_se=float(dd[s].std() / np.sqrt(s.sum())), chi2=float(e['chi2'][s].mean()),
                                        dev=float(e['dev'][s].mean())))
    res['methods'][l] = m

# ------------------------------------------------------------------------------------------------ repeat precision
blk = fr // 1000
pts = np.stack([R['x'], R['y'], 10.0 * fr], 1)
tree = cKDTree(pts)
pr = tree.query_pairs(10.05, output_type='ndarray')
a, b = pr[:, 0], pr[:, 1]
m_ = (np.abs(fr[b] - fr[a]) == 1) & (blk[a] == blk[b])
a, b = a[m_], b[m_]
sw = fr[a] > fr[b]
a[sw], b[sw] = b[sw], a[sw].copy()
ua, ca = np.unique(a, return_counts=True); ub, cb = np.unique(b, return_counts=True)
ok = np.isin(a, ua[ca == 1]) & np.isin(b, ub[cb == 1])
a, b = a[ok], b[ok]
res['repeat_pairs'] = int(len(a))
for l in labs:
    e = E[l]
    rep = {}
    for sel_name, s in (('all', np.ones(len(a), bool)), ('photons>8000', (Ir[a] > 8000) & (Ir[b] > 8000)),
                        ('photons 2000-8000', (Ir[a] > 2000) & (Ir[b] > 2000) & (Ir[a] <= 8000) & (Ir[b] <= 8000))):
        row = dict(n=int(s.sum()))
        for k, ck in (('x', 'crx'), ('y', 'cry'), ('z', 'crz')):
            dlt = (e[k][b] - e[k][a])[s]
            crl = np.sqrt((e[ck][a] ** 2 + e[ck][b] ** 2) / 2)[s]
            row[k] = dict(robust_std_over_sqrt2=rs(dlt) / np.sqrt(2), median_crlb=float(np.median(crl)),
                          ratio=rs(dlt) / np.sqrt(2) / float(np.median(crl)),
                          normalized_robust_std=rs(dlt / (np.sqrt(2) * crl)))
        rep[sel_name] = row
    res['methods'][l]['repeat'] = rep

# ------------------------------------------------------------------------------------------------ pores
good = (Ir > 500) & (np.abs(zr) < 750)
Wd, Hd = float(R['x'].max()) + 1, float(R['y'].max()) + 1
clusters = []
for w0 in range(0, int(fr.max()) + 1, W):
    ws = np.nonzero(good & (fr >= w0) & (fr < w0 + W))[0]
    if len(ws) < 1000:
        continue
    xw, yw = R['x'][ws], R['y'][ws]
    img, _, _ = np.histogram2d(yw, xw, bins=[np.arange(0, Hd + 0.05, 0.1), np.arange(0, Wd + 0.05, 0.1)])
    dimg = gaussian_filter(img, 4)
    pk = (dimg == maximum_filter(dimg, 15)) & (dimg > np.quantile(dimg, 0.995))
    py_, px_ = np.nonzero(pk)
    tr = cKDTree(np.stack([xw, yw], 1))
    for cy, cx in zip(py_ * 0.1 + 0.05, px_ * 0.1 + 0.05):
        idx = tr.query_ball_point([cx, cy], 1.2)
        if len(idx) < MINL:
            continue
        X, Y = xw[idx], yw[idx]
        Mm = np.stack([X, Y, np.ones_like(X)], 1)
        sol, *_ = np.linalg.lstsq(Mm, X ** 2 + Y ** 2, rcond=None)
        xc, yc = sol[0] / 2, sol[1] / 2
        r = np.sqrt(max(sol[2] + xc ** 2 + yc ** 2, 0))
        if 0.2 < r < 1.0 and np.hypot(xc - cx, yc - cy) < 0.5:
            clusters.append(ws[idx])
res['pores'] = dict(n_clusters=len(clusters), n_locs=int(sum(len(c) for c in clusters)), window_frames=W)


def circle(X, Y):
    Mm = np.stack([X, Y, np.ones_like(X)], 1)
    sol, *_ = np.linalg.lstsq(Mm, X ** 2 + Y ** 2, rcond=None)
    xc, yc = sol[0] / 2, sol[1] / 2
    return xc, yc, np.sqrt(max(sol[2] + xc ** 2 + yc ** 2, 0))


def mix_fit(v):
    """1 vs 2 Gaussians (equal widths) by maximum likelihood; v in nm."""
    v = v[np.abs(v) < 300]
    s1 = v.std()
    ll1 = -len(v) * (np.log(s1 * np.sqrt(2 * np.pi)) + 0.5)

    def nll2(p):
        w, m1, m2, ls = 1 / (1 + np.exp(-p[0])), p[1], p[2], p[3]
        s = np.exp(ls)
        pdf = (w * np.exp(-0.5 * ((v - m1) / s) ** 2) + (1 - w) * np.exp(-0.5 * ((v - m2) / s) ** 2)) / (s * np.sqrt(2 * np.pi))
        return -np.log(pdf + 1e-300).sum()
    best = None
    for sep in (20, 40, 60, 80):
        r = minimize(nll2, [0, -sep / 2, sep / 2, np.log(s1 * 0.8)], method='Nelder-Mead',
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    w, m1, m2, s = 1 / (1 + np.exp(-best.x[0])), best.x[1], best.x[2], np.exp(best.x[3])
    return dict(n=len(v), sd_one=float(s1), sep=float(abs(m2 - m1)), sigma=float(s), weight=float(w),
                dloglik_2_vs_1=float(-best.fun - ll1),
                resolved=bool(abs(m2 - m1) > 2 * s))


def mix_fit_crlb(v, cr):
    """As mix_fit, with each spot's own CRLB: v_i ~ w N(m1, cr_i^2 + s^2) + (1 - w) N(m2, cr_i^2 + s^2) against
    N(m, cr_i^2 + s^2); s = spread beyond the localization precision (structure + model error)."""
    k = np.abs(v) < 300
    v, cr = v[k], cr[k]

    def nll1(p):
        var = cr ** 2 + np.exp(2 * p[1])
        return 0.5 * (np.log(2 * np.pi * var) + (v - p[0]) ** 2 / var).sum()

    def nll2(p):
        w = 1 / (1 + np.exp(-p[0]))
        var = cr ** 2 + np.exp(2 * p[3])
        a = w * np.exp(-0.5 * (v - p[1]) ** 2 / var) + (1 - w) * np.exp(-0.5 * (v - p[2]) ** 2 / var)
        return -(np.log(a + 1e-300) - 0.5 * np.log(2 * np.pi * var)).sum()
    r1 = minimize(nll1, [0, np.log(30)], method='Nelder-Mead')
    best = None
    for sep in (30, 50, 70, 90):
        r = minimize(nll2, [0, -sep / 2, sep / 2, np.log(15)], method='Nelder-Mead', options=dict(maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return dict(n=int(len(v)), s_one=float(np.exp(r1.x[1])), sep=float(abs(best.x[2] - best.x[1])),
                s_two=float(np.exp(best.x[3])), weight=float(1 / (1 + np.exp(-best.x[0]))),
                dloglik_2_vs_1=float(r1.fun - best.fun), median_crlb=float(np.median(cr)))


zres_all = {}
for l in labs:
    e = E[l]
    rad, rad_ref, zres, zres_b, crb = [], [], [], [], []
    for c in clusters:
        X, Y = e['x'][c] * PXN, e['y'][c] * PYN
        xc, yc, r = circle(X, Y)
        rad.append(r)
        zc = e['z'][c]
        bright = R['I'][c] > 2000
        zres.append(zc - np.median(zc))
        if bright.sum() >= 3:
            zres_b.append(zc[bright] - np.median(zc[bright]))
            crb.append(e['crz'][c][bright])
    rad = np.array(rad)
    zres, zres_b, crb = np.concatenate(zres), np.concatenate(zres_b), np.concatenate(crb)
    zres_all[l] = zres_b
    pm = dict(radius_nm_median=float(np.median(rad)), radius_nm_iqr=[float(np.quantile(rad, 0.25)), float(np.quantile(rad, 0.75))],
              radius_px_median=float(np.median(rad) / np.sqrt(PXN * PYN)),
              z_within_pore_robust_sd_all=rs(zres), z_within_pore_robust_sd_bright=rs(zres_b),
              z_within_pore_iqr_bright=float(np.subtract(*np.quantile(zres_b, [0.75, 0.25]))),
              z_mix_bright=mix_fit(zres_b), z_mix_crlb_bright=mix_fit_crlb(zres_b, crb))
    res['methods'][l]['pores'] = pm
# radius ratio per pore against the reference (same spots)
radref = []
for c in clusters:
    radref.append(circle(R['x'][c] * PXN, R['y'][c] * PYN)[2])
radref = np.array(radref)
for l in labs:
    e = E[l]
    rl = np.array([circle(e['x'][c] * PXN, e['y'][c] * PYN)[2] for c in clusters])
    res['methods'][l]['pores']['radius_ratio_vs_ref_median'] = float(np.median(rl / radref))

# ------------------------------------------------------------------------------------------------ z scale, lateral shift vs z
zbins = np.arange(-600, 601, 100)
for l in labs:
    e = E[l]
    s = (np.abs(zr) < 600) & (np.abs(e['z']) < 750)
    xx, yy = zr[s], e['z'][s]
    keep = np.ones(len(xx), bool)
    for _ in range(5):
        co = np.polyfit(xx[keep], yy[keep], 1)
        rr = yy - np.polyval(co, xx)
        keep = np.abs(rr) < 3 * rs(rr[keep])
    sh = []
    for a_, b_ in zip(zbins[:-1], zbins[1:]):
        q = (zr >= a_) & (zr < b_) & (np.abs(e['z'] - zr) < 150)
        if q.sum() > 50:
            sh.append(dict(z=float(0.5 * (a_ + b_)), n=int(q.sum()), dx_nm=float(np.mean(e['x'][q] - R['x'][q]) * PXN),
                           dy_nm=float(np.mean(e['y'][q] - R['y'][q]) * PYN), dz_nm=float(np.median(e['z'][q] - zr[q]))))
    res['methods'][l]['z_vs_ref'] = dict(slope=float(co[0]), intercept_nm=float(co[1]), resid_robust_sd=rs(rr[keep]),
                                         frac_clipped=float(1 - keep.mean()), per_z=sh)

# ------------------------------------------------------------------------------------------------ figures
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for i, l in enumerate(labs[1:]):
    m = res['methods'][l]
    zc = [0.5 * sum(r['z']) for r in m['by_z']]
    ax[0].errorbar(zc, [r['ddev'] for r in m['by_z']], [r['ddev_se'] for r in m['by_z']], label=l, color=cols[i + 1], capsize=2)
    pc = [i_ for i_ in range(len(m['by_photons']))]
    ax[1].errorbar(pc, [r['ddev'] for r in m['by_photons']], [r['ddev_se'] for r in m['by_photons']], label=l, color=cols[i + 1], capsize=2)
for i, l in enumerate(labs):
    m = res['methods'][l]
    ax[2].plot([0.5 * sum(r['z']) for r in m['by_z']], [r['chi2'] for r in m['by_z']], 'o-', label=l, color=cols[i])
ax[0].axhline(0, color='k', lw=0.5); ax[1].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel(f'z of {ref} (nm)'); ax[0].set_ylabel(f'mean deviance per spot minus {ref} (lower = better)')
ax[1].set_xticks(range(len(pb) - 1)); ax[1].set_xticklabels(['<1k', '1-2k', '2-4k', '4-8k', '>8k']); ax[1].set_xlabel(f'photons ({ref})')
ax[2].set_xlabel(f'z of {ref} (nm)'); ax[2].set_ylabel('mean chi2 / pixel (test)'); ax[2].axhline(1, color='k', lw=0.5)
for a_ in ax:
    a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_{otag}_likelihood.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, len(labs), figsize=(3.6 * len(labs), 3.4), sharey=True, squeeze=False)
ax = ax[0]
hb = np.arange(-250, 251, 7.5)
for i, l in enumerate(labs):
    v = zres_all[l]
    ax[i].hist(v, hb, density=True, color=cols[i], alpha=0.7)
    mf = res['methods'][l]['pores']['z_mix_bright']
    ax[i].set_title(f'{l}: robust sd {res["methods"][l]["pores"]["z_within_pore_robust_sd_bright"]:.1f} nm\n'
                    f'2-G sep {mf["sep"]:.0f}, sigma {mf["sigma"]:.0f}, dlogL {mf["dloglik_2_vs_1"]:.0f}', fontsize=8)
    ax[i].set_xlabel('z - pore median (nm)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_{otag}_pore_z.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, l in enumerate(labs[1:]):
    sh = res['methods'][l]['z_vs_ref']['per_z']
    zz = [r['z'] for r in sh]
    ax[0].plot(zz, [r['dz_nm'] for r in sh], 'o-', color=cols[i + 1], label=l)
    ax[1].plot(zz, [r['dx_nm'] for r in sh], 'o-', color=cols[i + 1], label=l)
    ax[2].plot(zz, [r['dy_nm'] for r in sh], 'o-', color=cols[i + 1], label=l)
for a_, t_ in zip(ax, ('median z - z_ref (nm)', 'mean x - x_ref (nm)', 'mean y - y_ref (nm)')):
    a_.set_xlabel(f'z of {ref} (nm)'); a_.set_ylabel(t_); a_.axhline(0, color='k', lw=0.5); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_{otag}_z_lateral.png', dpi=120); plt.close(fig)

# z-scale-corrected precision metrics: each method's z divided by its slope against the reference, so that a
# stretched or compressed z axis does not count as worse or better precision
for l in labs:
    m = res['methods'][l]
    sl = m['z_vs_ref']['slope']
    m['scale_corrected'] = dict(slope=sl, repeat_z_all=m['repeat']['all']['z']['robust_std_over_sqrt2'] / sl,
                                repeat_z_2000_8000=m['repeat']['photons 2000-8000']['z']['robust_std_over_sqrt2'] / sl,
                                repeat_z_gt8000=m['repeat']['photons>8000']['z']['robust_std_over_sqrt2'] / sl,
                                pore_z_sd_bright=m['pores']['z_within_pore_robust_sd_bright'] / sl,
                                pore_z_iqr_bright=m['pores']['z_within_pore_iqr_bright'] / sl)
# local z scale: z_X is a nonlinear function of z_ref (offsets of up to ~100 nm that vary with z), so a single slope is
# not enough. f(z) = z + median(z_X - z_ref) in 50 nm bins of z_ref (smoothed over 3 bins); local scale f'(z_ref) per
# spot; repeat differences and within-pore residuals are divided by it (precision in the reference's z units).
pa, pb_ = a, b
zg = np.arange(-775, 776, 50)
allc = np.concatenate(clusters)
for l in labs:
    e = E[l]
    med = np.array([np.median(e['z'][(zr >= g - 25) & (zr < g + 25)] - zr[(zr >= g - 25) & (zr < g + 25)])
                    if ((zr >= g - 25) & (zr < g + 25)).sum() > 50 else np.nan for g in zg])
    ok_ = np.isfinite(med)
    med = np.interp(zg, zg[ok_], med[ok_])
    med = np.convolve(np.pad(med, 1, mode='edge'), np.ones(3) / 3, mode='valid')
    sl = np.clip(1 + np.gradient(med, zg), 0.3, 3.0)
    sloc = np.interp(zr, zg, sl)
    sp = 0.5 * (sloc[pa] + sloc[pb_])
    lc = {}
    for nm_, sel in (('all', np.ones(len(pa), bool)), ('photons 2000-8000', (Ir[pa] > 2000) & (Ir[pb_] > 2000) & (Ir[pa] <= 8000) & (Ir[pb_] <= 8000)),
                     ('photons>8000', (Ir[pa] > 8000) & (Ir[pb_] > 8000))):
        dlt = ((e['z'][pb_] - e['z'][pa]) / sp)[sel]
        lc['repeat_z_' + nm_] = rs(dlt) / np.sqrt(2)
    zres_l = []
    for c in clusters:
        bright = R['I'][c] > 2000
        if bright.sum() >= 3:
            zc = e['z'][c][bright]
            zres_l.append((zc - np.median(zc)) / sloc[c][bright])
    zres_l = np.concatenate(zres_l)
    lc['pore_z_sd_bright'] = rs(zres_l)
    lc['pore_z_iqr_bright'] = float(np.subtract(*np.quantile(zres_l, [0.75, 0.25])))
    lc['local_scale_by_z'] = [[float(g), float(v)] for g, v in zip(zg, sl)]
    res['methods'][l]['local_scale_corrected'] = lc
print('local-scale-corrected (reference z units): repeat z all / 2-8k / >8k, pore z robust sd / IQR (bright)')
for l in labs:
    lc = res['methods'][l]['local_scale_corrected']
    print(f'  {l}: {lc["repeat_z_all"]:.2f} / {lc["repeat_z_photons 2000-8000"]:.2f} / {lc["repeat_z_photons>8000"]:.2f}   '
          f'{lc["pore_z_sd_bright"]:.2f} / {lc["pore_z_iqr_bright"]:.2f}')
jdump(res, f'{OUT}/results_{otag}.json')
# compact table
print(f'test spots {n}, repeat pairs {res["repeat_pairs"]}, pore clusters {len(clusters)}')
hdr = ('method', 'dev/spot', 'd vs ref', 'se', 'chi2', 'lim any', 'rep x/crlb', 'rep z/crlb', 'z>8k', 'x>8k', 'pore z sd',
       'mix sep', 'sigma', 'dlogL', 'crlb-mix sep', 's', 'dlogL', 'radius', 'r ratio', 'z slope', 'z off')
print(' | '.join(hdr))
for l in labs:
    m = res['methods'][l]
    rp, pp, zs = m['repeat'], m['pores'], m['z_vs_ref']
    print(' | '.join([l, f'{m["dev_per_spot"]:.3f}', f'{m["ddev_vs_ref"]:+.4f}', f'{m["ddev_vs_ref_se"]:.4f}', f'{m["chi2_mean"]:.4f}',
                      f'{m["limits"]["any"]:.4f}', f'{rp["all"]["x"]["ratio"]:.3f}', f'{rp["all"]["z"]["ratio"]:.3f}',
                      f'{rp["photons>8000"]["z"]["ratio"]:.3f}', f'{rp["photons>8000"]["x"]["ratio"]:.3f}',
                      f'{pp["z_within_pore_robust_sd_bright"]:.1f}', f'{pp["z_mix_bright"]["sep"]:.1f}',
                      f'{pp["z_mix_bright"]["sigma"]:.1f}', f'{pp["z_mix_bright"]["dloglik_2_vs_1"]:.1f}',
                      f'{pp["z_mix_crlb_bright"]["sep"]:.1f}', f'{pp["z_mix_crlb_bright"]["s_two"]:.1f}',
                      f'{pp["z_mix_crlb_bright"]["dloglik_2_vs_1"]:.1f}',
                      f'{pp["radius_nm_median"]:.1f}', f'{pp["radius_ratio_vs_ref_median"]:.4f}', f'{zs["slope"]:.4f}',
                      f'{zs["intercept_nm"]:+.1f}']))
