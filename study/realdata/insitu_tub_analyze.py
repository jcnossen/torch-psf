"""Step 4b (tubulin): held-out metrics of the PSFs on the test set (CPU). Same metrics as insitu_analyze.py (NPC) except
the nuclear-pore part, plus the repeat precision in five photon bins (the axial ruler is insitu_tub_mt.py).
usage: INSITU_DS=tubulin python insitu_tub_analyze.py out_tag label1=tag1 label2=tag2 ...  (first label = reference)
-> insitu/results_<out_tag>.json and PNGs insitu/fig_<out_tag>_*.png

  likelihood  deviance of the halved data and model per spot (EMCCD), chi2/pixel (variance 2 mu); paired difference
              against the reference with its standard error; binned by z and by photons of the reference fit.
  limits      fraction of fits at the xy limit (3 px), at the z limit (+-800 nominal nm), either.
  repeat      same emitter in consecutive frames (mutual nearest neighbours within 1 px in the reference fit, both
              frames in the same test block; one fixed pair set for every method): robust std of the difference /
              sqrt 2 against the median CRLB of the method, and the robust std of the CRLB-normalised difference;
              overall and in photon bins (mean photons of the pair in the reference fit).
  z scale     z_X against z_ref (|z_ref| < 600 nm, clipped least squares), local slope f'(z_ref) from 50 nm bins, and
              the mean lateral difference x_X - x_ref, y_X - y_ref per 100 nm of z_ref (nm with 117.6 nm/px).
"""
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from insitu_common import OUT, jdump, XMAX, ZLIM

PXNM = 117.6
otag = sys.argv[1]
args = [a for a in sys.argv[2:] if not a.startswith('--')]
labs = [a.split('=')[0] for a in args]
tags = [a.split('=')[1] for a in args]
M = np.load(f'{OUT}/meta_test.npz')
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
res = dict(n_test=n, labels=dict(zip(labs, tags)), pixel_nm=PXNM, methods={})
rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 5 else float('nan')

zb = np.arange(-800, 801, 200)
pb = [0, 1000, 2000, 4000, 8000, 1e9]
pbn = ['<1k', '1-2k', '2-4k', '4-8k', '>8k']
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
             photons_ratio_median=float(np.median(e['I'] / R['I'])), bg_diff_median=float(np.median(e['bg'] - R['bg'])),
             seconds_test=e['sec'])
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

# ------------------------------------------------------------------------------------------------ repeat pairs
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
np.savez(f'{OUT}/repeat_pairs_{otag}.npz', a=a, b=b)
res['repeat_pairs'] = int(len(a))
Ip = 0.5 * (Ir[a] + Ir[b])
sels = [('all', np.ones(len(a), bool))] + [(f'photons {nm}', (Ip >= lo) & (Ip < hi)) for nm, lo, hi in zip(pbn, pb[:-1], pb[1:])]
for l in labs:
    e = E[l]
    rep = {}
    for sel_name, s in sels:
        row = dict(n=int(s.sum()))
        for k, ck in (('x', 'crx'), ('y', 'cry'), ('z', 'crz')):
            dlt = (e[k][b] - e[k][a])[s]
            crl = np.sqrt((e[ck][a] ** 2 + e[ck][b] ** 2) / 2)[s]
            row[k] = dict(robust_std_over_sqrt2=rs(dlt) / np.sqrt(2), median_crlb=float(np.median(crl)),
                          ratio=rs(dlt) / np.sqrt(2) / float(np.median(crl)),
                          normalized_robust_std=rs(dlt / (np.sqrt(2) * crl)),
                          normalized_std=float(np.std(dlt / (np.sqrt(2) * crl))))
        rep[sel_name] = row
    res['methods'][l]['repeat'] = rep

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
            sh.append(dict(z=float(0.5 * (a_ + b_)), n=int(q.sum()), dx_nm=float(np.mean(e['x'][q] - R['x'][q]) * PXNM),
                           dy_nm=float(np.mean(e['y'][q] - R['y'][q]) * PXNM), dz_nm=float(np.median(e['z'][q] - zr[q]))))
    res['methods'][l]['z_vs_ref'] = dict(slope=float(co[0]), intercept_nm=float(co[1]), resid_robust_sd=rs(rr[keep]),
                                         frac_clipped=float(1 - keep.mean()), per_z=sh,
                                         max_abs_lateral_shift_nm=float(max([max(abs(r['dx_nm']), abs(r['dy_nm'])) for r in sh] or [0])))

# local z scale f'(z_ref) per spot; repeat z differences divided by it (precision in the reference's z units)
zg = np.arange(-775, 776, 50)
for l in labs:
    e = E[l]
    med = np.array([np.median(e['z'][(zr >= g - 25) & (zr < g + 25)] - zr[(zr >= g - 25) & (zr < g + 25)])
                    if ((zr >= g - 25) & (zr < g + 25)).sum() > 50 else np.nan for g in zg])
    ok_ = np.isfinite(med)
    med = np.interp(zg, zg[ok_], med[ok_])
    med = np.convolve(np.pad(med, 1, mode='edge'), np.ones(3) / 3, mode='valid')
    sl = np.clip(1 + np.gradient(med, zg), 0.3, 3.0)
    sloc = np.interp(zr, zg, sl)
    sp = 0.5 * (sloc[a] + sloc[b])
    lc = {}
    for nm_, sel in sels:
        dlt = ((e['z'][b] - e['z'][a]) / sp)[sel]
        lc['repeat_z_' + nm_] = rs(dlt) / np.sqrt(2)
    lc['local_scale_by_z'] = [[float(g), float(v)] for g, v in zip(zg, sl)]
    lc['offset_by_z'] = [[float(g), float(v)] for g, v in zip(zg, med)]
    res['methods'][l]['local_scale_corrected'] = lc

# ------------------------------------------------------------------------------------------------ figures
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for i, l in enumerate(labs[1:]):
    m = res['methods'][l]
    zc = [0.5 * sum(r['z']) for r in m['by_z']]
    ax[0].errorbar(zc, [r['ddev'] for r in m['by_z']], [r['ddev_se'] for r in m['by_z']], label=l, color=cols[i + 1], capsize=2)
    ax[1].errorbar(range(len(m['by_photons'])), [r['ddev'] for r in m['by_photons']], [r['ddev_se'] for r in m['by_photons']],
                   label=l, color=cols[i + 1], capsize=2)
for i, l in enumerate(labs):
    m = res['methods'][l]
    ax[2].plot([0.5 * sum(r['z']) for r in m['by_z']], [r['chi2'] for r in m['by_z']], 'o-', label=l, color=cols[i])
ax[0].axhline(0, color='k', lw=0.5); ax[1].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel(f'z of {ref} (nominal nm)'); ax[0].set_ylabel(f'mean deviance per spot minus {ref} (lower = better)')
ax[1].set_xticks(range(len(pbn))); ax[1].set_xticklabels(pbn); ax[1].set_xlabel(f'photons ({ref})')
ax[2].set_xlabel(f'z of {ref} (nominal nm)'); ax[2].set_ylabel('mean chi2 / pixel (test)'); ax[2].axhline(1, color='k', lw=0.5)
for a_ in ax:
    a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_{otag}_likelihood.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, l in enumerate(labs[1:]):
    sh = res['methods'][l]['z_vs_ref']['per_z']
    zz = [r['z'] for r in sh]
    ax[0].plot(zz, [r['dz_nm'] for r in sh], 'o-', color=cols[i + 1], label=l)
    ax[1].plot(zz, [r['dx_nm'] for r in sh], 'o-', color=cols[i + 1], label=l)
    ax[2].plot(zz, [r['dy_nm'] for r in sh], 'o-', color=cols[i + 1], label=l)
for a_, t_ in zip(ax, ('median z - z_ref (nominal nm)', 'mean x - x_ref (nm)', 'mean y - y_ref (nm)')):
    a_.set_xlabel(f'z of {ref} (nominal nm)'); a_.set_ylabel(t_); a_.axhline(0, color='k', lw=0.5); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_{otag}_z_lateral.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, l in enumerate(labs):
    rp = res['methods'][l]['repeat']
    ax[0].plot(range(len(pbn)), [rp[f'photons {nm}']['x']['ratio'] for nm in pbn], 'o-', color=cols[i], label=l)
    ax[1].plot(range(len(pbn)), [rp[f'photons {nm}']['z']['ratio'] for nm in pbn], 'o-', color=cols[i], label=l)
for a_, t_ in zip(ax, ('x', 'z')):
    a_.set_xticks(range(len(pbn))); a_.set_xticklabels(pbn); a_.set_xlabel(f'photons ({ref}, pair mean)')
    a_.set_ylabel(f'repeat {t_} spread / median CRLB'); a_.axhline(1, color='k', lw=0.5); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_{otag}_repeat.png', dpi=120); plt.close(fig)

jdump(res, f'{OUT}/results_{otag}.json')
print(f'test spots {n}, repeat pairs {res["repeat_pairs"]}')
hdr = ('method', 'dev/spot', 'd vs ref', 'se', 'chi2', 'lim xy', 'lim z', 'rep x/crlb', 'rep z/crlb', 'rep x px', 'rep z (a units)',
       'z slope', 'z off', 'max lat shift')
print(' | '.join(hdr))
for l in labs:
    m = res['methods'][l]
    rp, zs = m['repeat'], m['z_vs_ref']
    print(' | '.join([l, f'{m["dev_per_spot"]:.3f}', f'{m["ddev_vs_ref"]:+.4f}', f'{m["ddev_vs_ref_se"]:.4f}', f'{m["chi2_mean"]:.4f}',
                      f'{m["limits"]["xy"]:.4f}', f'{m["limits"]["z"]:.4f}', f'{rp["all"]["x"]["ratio"]:.3f}',
                      f'{rp["all"]["z"]["ratio"]:.3f}', f'{rp["all"]["x"]["robust_std_over_sqrt2"]:.4f}',
                      f'{m["local_scale_corrected"]["repeat_z_all"]:.2f}', f'{zs["slope"]:.4f}', f'{zs["intercept_nm"]:+.1f}',
                      f'{zs["max_abs_lateral_shift_nm"]:.1f}']))
print('repeat x / CRLB by photon bin:', pbn)
for l in labs:
    rp = res['methods'][l]['repeat']
    print(f'  {l}: x ' + ' '.join(f'{rp[f"photons {nm}"]["x"]["ratio"]:.3f}' for nm in pbn) +
          '   z ' + ' '.join(f'{rp[f"photons {nm}"]["z"]["ratio"]:.3f}' for nm in pbn) +
          '   z(a units) ' + ' '.join(f'{res["methods"][l]["local_scale_corrected"]["repeat_z_photons " + nm]:.1f}' for nm in pbn))
