"""Sanity checks of a localization table written by loc.py.
usage: python checks.py locs.csv outdir [--render-frames a:b] [--npc]  -> checks.json and PNGs in outdir"""
import sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter

ap = argparse.ArgumentParser()
ap.add_argument('csv'); ap.add_argument('outdir')
ap.add_argument('--windows', type=int, default=10000)
ap.add_argument('--npc', action='store_true')
ap.add_argument('--zlim', type=float, default=800.0)
ap.add_argument('--tag', default='')
A = ap.parse_args()
R = np.loadtxt(A.csv, delimiter=',', skiprows=1)
cols = open(A.csv).readline().strip().split(',')
C = {c: R[:, i] for i, c in enumerate(cols)}
n = len(R)
fl = C['flags'].astype(int)
out = dict(n=n, frames=[int(C['frame'].min()), int(C['frame'].max())])
names = {1: 'xy_at_limit', 2: 'z_at_limit', 4: 'edge', 8: 'neighbour_in_patch', 16: 'I_or_bg_at_limit'}
out['flag_fraction'] = {v: float(((fl & k) > 0).mean()) for k, v in names.items()}
out['flag_fraction']['any'] = float((fl > 0).mean())
clean = (fl == 0)
out['clean_fraction'] = float(clean.mean())
q = lambda v: {f'q{int(p * 100):02d}': float(np.quantile(v, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
for k in ('photons', 'bg', 'crlb_x', 'crlb_y', 'crlb_z', 'chi2', 'nll_ratio', 'z'):
    out[k + '_all'] = q(C[k]); out[k + '_clean'] = q(C[k][clean])
out['locs_per_frame'] = float(n / (out['frames'][1] - out['frames'][0] + 1))
# z histogram
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
bins = np.arange(-A.zlim - 25, A.zlim + 26, 25)
ax[0].hist(C['z'], bins, histtype='step', label='all')
ax[0].hist(C['z'][clean], bins, histtype='step', label='flags = 0')
ax[0].set_xlabel('z (stage nm, bead calibration)'); ax[0].set_ylabel('count'); ax[0].legend()
good = clean & (C['photons'] > 1000)
ax[1].hist(C['z'][good], bins, histtype='step', color='k')
ax[1].set_xlabel('z (nm), flags = 0 and photons > 1000'); fig.tight_layout()
fig.savefig(f'{A.outdir}/z_hist{A.tag}.png', dpi=110); plt.close(fig)
h, e = np.histogram(C['z'][clean], bins)
out['z_hist_clean'] = dict(edges=e.tolist(), counts=h.tolist())
zc = C['z'][good]
out['z_clean_bright'] = dict(median=float(np.median(zc)), iqr=float(np.subtract(*np.quantile(zc, [0.75, 0.25]))),
                             frac_within_100nm_of_limit=float((np.abs(zc) > A.zlim - 100).mean()))
# photons / bg / chi2 histograms
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
ax[0].hist(np.log10(C['photons'].clip(1)), 100); ax[0].set_xlabel('log10 photons')
ax[1].hist(C['bg'], np.linspace(0, np.quantile(C['bg'], 0.995), 100)); ax[1].set_xlabel('bg (photons/px)')
ax[2].hist(C['chi2'], np.linspace(0, 4, 100)); ax[2].axvline(1, color='k'); ax[2].set_xlabel('chi2 / pixel (expect 1)')
ax[3].scatter(C['z'][::20], C['crlb_x'][::20], s=1, label='x'); ax[3].scatter(C['z'][::20], C['crlb_y'][::20], s=1, label='y')
ax[3].set_ylim(0, 0.5); ax[3].set_xlabel('z (nm)'); ax[3].set_ylabel('CRLB (px)'); ax[3].legend()
fig.tight_layout(); fig.savefig(f'{A.outdir}/dists{A.tag}.png', dpi=110); plt.close(fig)
# chi2 against photons (model error grows with signal)
pb = np.quantile(C['photons'][clean], np.linspace(0, 1, 11))
out['chi2_vs_photons_clean'] = [[float(a), float(b), float(np.median(C['chi2'][clean & (C['photons'] >= a) & (C['photons'] < b)]))]
                                for a, b in zip(pb[:-1], pb[1:])]
zb = np.arange(-A.zlim, A.zlim + 1, 100)
out['chi2_vs_z_clean'] = [[float(a), float(np.median(C['chi2'][clean & (C['z'] >= a) & (C['z'] < a + 100)]))
                           if (clean & (C['z'] >= a) & (C['z'] < a + 100)).sum() > 20 else None] for a in zb[:-1]]

# consecutive-frame repeats: same emitter on in frames t and t+1 (mutual unique neighbours within 1 px)
sel = np.nonzero(clean)[0]
pts = np.stack([C['x'][sel], C['y'][sel], 10.0 * C['frame'][sel]], 1)
tree = cKDTree(pts)
pr = tree.query_pairs(10.05, output_type='ndarray')
a, b = sel[pr[:, 0]], sel[pr[:, 1]]
df = C['frame'][b] - C['frame'][a]
m = np.abs(df) == 1
a, b = a[m], b[m]
sw = C['frame'][a] > C['frame'][b]
a[sw], b[sw] = b[sw], a[sw].copy()
ua, ca = np.unique(a, return_counts=True); ub, cb = np.unique(b, return_counts=True)
okp = np.isin(a, ua[ca == 1]) & np.isin(b, ub[cb == 1])
a, b = a[okp], b[okp]
rep = dict(n_pairs=int(len(a)))
for k, ck, s in (('x', 'crlb_x', 1.0), ('y', 'crlb_y', 1.0), ('z', 'crlb_z', 1.0)):
    dlt = C[k][b] - C[k][a]
    rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))
    nrm = dlt / np.sqrt(C[ck][a] ** 2 + C[ck][b] ** 2)
    rep[k] = dict(std_diff_over_sqrt2=float(np.std(dlt) / np.sqrt(2)), robust_std_diff_over_sqrt2=rs(dlt) / np.sqrt(2),
                  median_crlb=float(np.median(np.sqrt((C[ck][a] ** 2 + C[ck][b] ** 2) / 2))),
                  normalized_std=float(np.std(nrm)), normalized_robust_std=rs(nrm), mean_diff=float(np.mean(dlt)))
# the same in photon bins (CRLB depends on photons)
pbins = [0, 1000, 2000, 4000, 8000, 1e9]
rep['by_photons'] = []
for lo, hi in zip(pbins[:-1], pbins[1:]):
    s = (C['photons'][a] >= lo) & (C['photons'][a] < hi)
    if s.sum() < 50:
        continue
    row = dict(lo=lo, hi=hi, n=int(s.sum()))
    for k, ck in (('x', 'crlb_x'), ('z', 'crlb_z')):
        dlt = (C[k][b] - C[k][a])[s]
        row[k + '_robust_std_over_sqrt2'] = float(1.4826 * np.median(np.abs(dlt - np.median(dlt))) / np.sqrt(2))
        row[k + '_median_crlb'] = float(np.median(np.sqrt((C[ck][a][s] ** 2 + C[ck][b][s] ** 2) / 2)))
    rep['by_photons'].append(row)
out['repeat'] = rep


# renders
def render(x, y, x0, x1, y0, y1, px, blur=1.0):
    img, _, _ = np.histogram2d(y, x, bins=[np.arange(y0, y1 + px / 2, px), np.arange(x0, x1 + px / 2, px)])
    return gaussian_filter(img, blur)


good = clean & (C['photons'] > 500) & (np.abs(C['z']) < A.zlim - 50)
xs, ys, fs = C['x'][good], C['y'][good], C['frame'][good]
W, H = float(C['x'].max()), float(C['y'].max())
img = render(xs, ys, 0, W, 0, H, 0.1)
fig = plt.figure(figsize=(10, 10 * H / W))
plt.imshow(img, cmap='hot', vmax=np.quantile(img, 0.999), extent=(0, W, H, 0)); plt.xlabel('x (px)'); plt.ylabel('y (px)')
plt.title('all frames, flags = 0, photons > 500 (no drift correction), 0.1 px bins')
fig.savefig(f'{A.outdir}/render_full{A.tag}.png', dpi=130); plt.close(fig)
# first window only (less drift blur), a 20 x 20 px region in the densest part
f0 = C['frame'].min()
wsel = fs < f0 + A.windows
dens = render(xs[wsel], ys[wsel], 0, W, 0, H, 1.0, blur=5)
iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
x0 = float(np.clip(ix - 10, 0, W - 20)); y0 = float(np.clip(iy - 10, 0, H - 20))
for tag, ws in (('win1', wsel), ('all', np.ones_like(wsel))):
    img = render(xs[ws], ys[ws], x0, x0 + 20, y0, y0 + 20, 0.1, blur=0.8)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='hot', vmax=np.quantile(img, 0.998), extent=(x0, x0 + 20, y0 + 20, y0))
    plt.title(f'{"frames < " + str(int(f0 + A.windows)) if tag == "win1" else "all frames"}; 20 x 20 px; 0.1 px bins')
    plt.xlabel('x (px)'); plt.ylabel('y (px)')
    fig.savefig(f'{A.outdir}/render_zoom_{tag}{A.tag}.png', dpi=130); plt.close(fig)
out['zoom_region'] = [x0, y0, 20, 20]

if A.npc:
    # ring radius: per time window (drift < blur), NPC candidates = maxima of a 0.4 px-blurred density, circle fit to
    # the localizations within 1.2 px of the candidate centre
    radii, zsep = [], []
    for w0 in range(int(f0), int(C['frame'].max()) + 1, A.windows):
        ws = (fs >= w0) & (fs < w0 + A.windows)
        if ws.sum() < 1000:
            continue
        xw, yw = xs[ws], ys[ws]
        zw = C['z'][good][ws]
        dimg = render(xw, yw, 0, W, 0, H, 0.1, blur=4)
        pk = (dimg == maximum_filter(dimg, 15)) & (dimg > np.quantile(dimg, 0.995))
        py, px_ = np.nonzero(pk)
        tr = cKDTree(np.stack([xw, yw], 1))
        for cy, cx in zip(py * 0.1 + 0.05, px_ * 0.1 + 0.05):
            idx = tr.query_ball_point([cx, cy], 1.2)
            if len(idx) < 30:
                continue
            X, Y = xw[idx], yw[idx]
            # algebraic circle fit
            M = np.stack([X, Y, np.ones_like(X)], 1)
            sol, *_ = np.linalg.lstsq(M, X ** 2 + Y ** 2, rcond=None)
            xc, yc = sol[0] / 2, sol[1] / 2
            r = np.sqrt(sol[2] + xc ** 2 + yc ** 2)
            rr = np.hypot(X - xc, Y - yc)
            if 0.2 < r < 1.0 and np.hypot(xc - cx, yc - cy) < 0.5:
                radii.append(dict(r=float(r), r_med=float(np.median(rr)), n=len(idx), xc=float(xc), yc=float(yc), w0=w0,
                                  z_med=float(np.median(zw[idx])), z_iqr=float(np.subtract(*np.quantile(zw[idx], [0.75, 0.25])))))
    rr = np.array([d['r'] for d in radii])
    out['npc'] = dict(n_rings=len(radii), radius_px_median=float(np.median(rr)) if len(rr) else None,
                      radius_px_iqr=[float(np.quantile(rr, 0.25)), float(np.quantile(rr, 0.75))] if len(rr) else None,
                      z_iqr_within_ring_median_nm=float(np.median([d['z_iqr'] for d in radii])) if len(rr) else None)
    if len(rr):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(rr, 40); ax[0].set_xlabel('fitted ring radius (px)')
        # pooled radial profile around the fitted centres
        allr = []
        for d in radii[:2000]:
            ws = (fs >= d['w0']) & (fs < d['w0'] + A.windows)
            X, Y = xs[ws], ys[ws]
            s = np.hypot(X - d['xc'], Y - d['yc']) < 1.5
            allr.append(np.hypot(X[s] - d['xc'], Y[s] - d['yc']))
        allr = np.concatenate(allr)
        hh, ee = np.histogram(allr, np.linspace(0, 1.5, 61))
        dens_r = hh / (np.pi * (ee[1:] ** 2 - ee[:-1] ** 2))
        ax[1].plot(0.5 * (ee[1:] + ee[:-1]), dens_r); ax[1].set_xlabel('distance from ring centre (px)'); ax[1].set_ylabel('density')
        out['npc']['radial_density_peak_px'] = float(0.5 * (ee[1:] + ee[:-1])[np.argmax(dens_r[2:]) + 2])
        fig.tight_layout(); fig.savefig(f'{A.outdir}/npc_rings{A.tag}.png', dpi=110); plt.close(fig)
json.dump(out, open(f'{A.outdir}/checks{A.tag}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('z_hist_clean',)}, indent=1))
