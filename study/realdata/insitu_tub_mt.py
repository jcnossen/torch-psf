"""Axial ruler on the tubulin test set: cross-sections of straight, isolated microtubule segments (CPU).

A microtubule is a cylinder, so its label distribution seen in (lateral u perpendicular to the filament, height z)
must be round. For each method the z-to-lateral width ratio (after removing the localization precision in quadrature)
tells which factor kappa = true height / fitted z makes the cross-section round. Nominal z = the bead-stack axis with
10 nm per plane assumed; the refractive-index factor of Li et al. is 0.75.

usage: INSITU_DS=tubulin python insitu_tub_mt.py out_tag label1=tag1 ... [--window=2000] [--seglen=1.5] [--boot=300]
needs eval_<tag>.npz of every method and results_<out_tag>.json of insitu_tub_analyze.py (repeat precision per photon
bin: each spot's precision = its CRLB x the robust std of the CRLB-normalised consecutive-frame differences of its
photon bin, per method and axis).

1 segments (from the reference method, the first label): render of all test localizations (photons > 800, |z| < 700
  nominal nm, not at a fit limit) in 0.2 px bins, Gaussian blur 1.5 bins; candidates = local maxima (9 x 9 bins) of
  a Sato ridge filter (sigma 1.5-2.5 bins) in its top 3 %; at each, a straight segment of length seglen (um at
  117.6 nm/px) along the local structure-tensor direction, angle (+-12 deg) and offset (+-1 bin) refined to maximise
  the render along it. Kept if: no gap (< 5 % of the samples below 1/4 of the median), straight (ridge position in 8
  sub-pieces off a line by < 0.15 px rms), isolated (mean render 1.5-6 px from the axis < ISO x the mean within
  0.5 px), >= MINLOC localizations within 0.5 px, ends >= 3 px from the frame edge; then greedily (most isolated
  first) no two segments closer than 3 px; also the band 0.9-1.5 px must be < ISO_NEAR x the core (a second
  filament close by). Segments whose drift- and tilt-corrected residuals in the reference have more than UTAIL beyond
  +-100 nm laterally or more than ZTAIL beyond +-200 nominal nm axially are dropped (a second filament within the
  4 px window, beside or above; both cuts are > 4 sd of a single filament's profile).
2 drift: not corrected in the data. Per method, every localization within UWIN (2) px of a segment axis (the same spots for
  all methods, selected in the reference coordinates) is modelled as
    u_i = n_seg . (Dx_w, Dy_w) + a0 + a1 s + a2 s^2,     z_i = Dz_w + b0 + b1 s + b2 s^2   (per segment a, b)
  with one drift vector per time window w of --window frames (shared by all segments; the segments have different
  orientations, so Dx, Dy are determined), robust (iteratively clipped at 3.5 robust sd) least squares. The residuals
  are the cross-section. --drift=segment instead fits one offset per (segment, window) (no assumption that the
  drift is uniform over the field); the residual variances are then corrected by n / (n - parameters) per segment.
3 widths per segment and method: Gaussian + uniform background, maximum likelihood (EM) in |u| < 250 nm, |z| < 400
  nominal nm; sigma_tot^2 - <precision^2> (responsibility-weighted mean over the segment's spots) = W^2.
  ratio = W_z / W_u (nominal z); kappa_round = W_u / W_z; ratio with z x 0.75 = 0.75 x ratio. Pooled over segments as
  sqrt(sum W_z^2 / sum W_u^2), bootstrap over segments.
4 hollow cylinder: pooled residuals (core) fitted with a Gaussian and with a projected thin ring of radius R, both
  convolved with each spot's precision plus a common extra width s (+ uniform background); and a 2D fit of an
  elliptical ring in (u, z): semi-axes R and R / kappa, isotropic extra width s (s / kappa in z), each spot's
  precision; kappa from the shape and width of the 2D cross-section.
"""
import sys, json, time
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from skimage.filters import sato
from skimage.feature import structure_tensor
from insitu_common import OUT, jdump, XMAX, ZLIM

t00 = time.time()
otag = sys.argv[1]
opts = dict(window=2000, seglen=1.0, boot=300, boot2d=30, iso=0.12, iso_near=0.20, nfit=40000, uwin=2.0, drift='global', utail=0.12, ztail=0.15, minloc=200, px=117.6, tagout='')
for o in sys.argv[2:]:
    if o.startswith('--'):
        k, v = o[2:].split('=')
        opts[k] = type(opts[k])(v) if not isinstance(opts[k], str) else v
args = [a for a in sys.argv[2:] if not a.startswith('--')]
labs = [a.split('=')[0] for a in args]
tags = [a.split('=')[1] for a in args]
PXNM = opts['px']
W = opts['window']
otag2 = otag + opts['tagout']
rng = np.random.default_rng(1)
M = np.load(f'{OUT}/meta_test.npz')
fr = M['frame']
try:
    RA = json.load(open(f'{OUT}/results_{otag}.json'))
except FileNotFoundError:            # test runs only: no repeat-precision calibration, precision = CRLB
    RA = None
    print('WARNING: no results json, precision = CRLB', flush=True)
pb = [0, 1000, 2000, 4000, 8000, 1e9]
pbn = ['<1k', '1-2k', '2-4k', '4-8k', '>8k']
E = {}
for l, t in zip(labs, tags):
    if t == 'csv':                   # test runs: the bead-spline localizations of the CSV (= method a)
        E[l] = dict(x=M['x'], y=M['y'], z=M['z'], I=M['photons'], ox=M['x'] - M['peak_x'], oy=M['y'] - M['peak_y'],
                    crx=M['crlb_x'], cry=M['crlb_y'], crz=M['crlb_z'])
        continue
    e = np.load(f'{OUT}/eval_{t}.npz')
    th = e['th']
    E[l] = dict(x=M['peak_x'] + th[:, 0], y=M['peak_y'] + th[:, 1], z=th[:, 2] * 1000, I=th[:, 3],
                ox=th[:, 0], oy=th[:, 1], crx=e['crlb'][:, 0], cry=e['crlb'][:, 1], crz=e['crlb'][:, 2] * 1000)
ref = labs[0]
R = E[ref]
lim = (np.maximum(np.abs(R['ox']), np.abs(R['oy'])) > XMAX - 0.06) | (np.abs(R['z']) >= ZLIM * 1000 - 5)
good = (R['I'] > 800) & (np.abs(R['z']) < 700) & ~lim
rs = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 5 else float('nan')
out = dict(opts=opts, labels=dict(zip(labs, tags)), n_test=int(len(fr)))

# ------------------------------------------------------------------------------------------------ 1 segments
B = 0.2
Wd, Hd = 324, 271
img, _, _ = np.histogram2d(R['y'][good], R['x'][good], bins=[np.arange(0, Hd + 1e-6, B), np.arange(0, Wd + 1e-6, B)])
sm = gaussian_filter(img, 1.5)
rid = sato(sm, sigmas=[1.5, 2.5], black_ridges=False)
Arr, Arc, Acc = structure_tensor(sm, sigma=4, order='rc')
theta = 0.5 * np.arctan2(2 * Arc, Acc - Arr) + np.pi / 2       # local filament direction (x = column, y = row)
Lb = opts['seglen'] * 1000 / PXNM / B                            # segment length in bins
pk = (rid == maximum_filter(rid, 9)) & (rid > np.quantile(rid, 0.97))
cy, cx = np.nonzero(pk)
nsb = np.linspace(-Lb / 2, Lb / 2, 65)
xs_all, ys_all = R['x'][good], R['y'][good]
from scipy.spatial import cKDTree
tree_good = cKDTree(np.stack([xs_all, ys_all], 1))
cands, rejects = [], dict(candidates=int(len(cy)), gap=0, curved=0, crowded=0, few=0, edge=0, overlap=0)
for y, x in zip(cy, cx):
    best = None
    for dth in np.deg2rad(np.arange(-12, 12.1, 2)):
        th = theta[y, x] + dth
        t = np.array([np.cos(th), np.sin(th)]); n = np.array([-t[1], t[0]])
        for off in (-1, 0, 1):
            v = map_coordinates(sm, [y + nsb * t[1] + off * n[1], x + nsb * t[0] + off * n[0]], order=1, mode='constant')
            sc = v.mean() - 0.5 * v.std()
            if best is None or sc > best[0]:
                best = (sc, th, off, v)
    sc, th, off, v = best
    t = np.array([np.cos(th), np.sin(th)]); n = np.array([-t[1], t[0]])
    c = np.array([x, y], float) + off * n
    if np.median(v) <= 0 or (v < 0.25 * np.median(v)).mean() > 0.05:        # gap along the segment
        rejects['gap'] += 1
        continue
    pos = []                                                                 # ridge position in 8 sub-pieces
    uu = np.arange(-5, 5.01, 0.25)
    for k in range(8):
        s_ = nsb[k * 8:(k + 1) * 8 + 1]
        P_ = [map_coordinates(sm, [c[1] + s_ * t[1] + u * n[1], c[0] + s_ * t[0] + u * n[0]], order=1).mean() for u in uu]
        pos.append(uu[int(np.argmax(P_))])
    pos = np.array(pos)
    co = np.polyfit(np.arange(8), pos, 1)
    rms_ = float(np.sqrt(np.mean((pos - np.polyval(co, np.arange(8))) ** 2)))
    if rms_ > 0.75:                                                          # 0.15 px = 18 nm
        rejects['curved'] += 1
        continue
    uu = np.arange(-30, 30.01, 1.0)
    prof = np.array([map_coordinates(sm, [c[1] + nsb * t[1] + u * n[1], c[0] + nsb * t[0] + u * n[0]], order=1, mode='constant').mean() for u in uu])
    core_r = prof[np.abs(uu) <= 2.5].mean()
    iso = float(prof[np.abs(uu) >= 7.5].mean() / core_r)                     # density 1.5-6 px / core
    iso_near = float(prof[(np.abs(uu) >= 4.5) & (np.abs(uu) < 7.5)].mean() / core_r)   # 0.9-1.5 px / core
    if iso > opts['iso'] or iso_near > opts['iso_near']:
        rejects['crowded'] += 1
        continue
    cp = (c + 0.5) * B
    half = Lb / 2 * B
    ends = np.array([cp - half * t, cp + half * t])
    if ends[:, 0].min() < 3 or ends[:, 1].min() < 3 or ends[:, 0].max() > Wd - 3 or ends[:, 1].max() > Hd - 3:
        rejects['edge'] += 1
        continue
    idx = np.array(tree_good.query_ball_point(cp, half + 1), dtype=int)
    d = np.stack([xs_all[idx], ys_all[idx]], 1) - cp
    core = int(((np.abs(d @ t) <= half) & (np.abs(d @ n) < 0.5)).sum())
    if core < opts['minloc']:
        rejects['few'] += 1
        continue
    cands.append(dict(c=cp, t=t, n=n, s_lo=-half, s_hi=half, n_core=core, flank_ratio=iso, near_ratio=iso_near, straight_rms_px=rms_ * B))
cands.sort(key=lambda d: d['flank_ratio'])
segs = []
for d in cands:                                  # greedy: most isolated first, no two segments closer than 3 px
    pts = d['c'][None] + np.linspace(d['s_lo'], d['s_hi'], 33)[:, None] * d['t'][None]
    if any(np.min(np.linalg.norm(pts[:, None] - (e['c'][None] + np.linspace(e['s_lo'], e['s_hi'], 33)[:, None] * e['t'][None])[None], axis=2)) < 3
           for e in segs):
        rejects['overlap'] += 1
        continue
    segs.append(d)
print(f'segments: {len(segs)} kept; rejected {rejects}  ({time.time() - t00:.0f}s)', flush=True)

# spots of each segment (reference coordinates, same spots for every method): |u| < uwin px, s within the piece
allpts = np.stack([R['x'], R['y']], 1)
tree_all = cKDTree(allpts)
okspot = ~lim & (np.abs(R['z']) < 750)
seg_idx = []
for sg in segs:
    idx = np.array(tree_all.query_ball_point(sg['c'], 0.5 * (sg['s_hi'] - sg['s_lo']) + 5), dtype=int)
    d = allpts[idx] - sg['c']
    ss, uu = d @ sg['t'], d @ sg['n']
    q = (ss >= sg['s_lo']) & (ss <= sg['s_hi']) & (np.abs(uu) < opts['uwin']) & okspot[idx]
    seg_idx.append(idx[q])
# a spot can lie near two segments (crossing ends): keep it only in the first
seen = np.zeros(len(fr), bool)
for i in range(len(seg_idx)):
    v = seg_idx[i]
    seg_idx[i] = v[~seen[v]]
    seen[seg_idx[i]] = True
nwin = int(fr.max() // W) + 1


def fit_drift(l, seg_ids, clipk=3.5, iters=4):
    """Global drift per window + quadratic per segment, for u (nm) and z (nominal nm) of method l. Returns residuals
    per segment (lists) and the drift traces."""
    e = E[l]
    rows_u, cols, vals_u, yu, yz, segof, sloc, wloc = [], [], [], [], [], [], [], []
    U, Z, S, Wn, G, NX, NY = [], [], [], [], [], [], []
    for j, k in enumerate(seg_ids):
        sg, idx = segs[k], seg_idx[k]
        X = np.stack([e['x'][idx], e['y'][idx]], 1)
        d = (X - sg['c']) * PXNM
        U.append(d @ sg['n']); S.append(d @ sg['t'] - 0.5 * (sg['s_lo'] + sg['s_hi']) * PXNM)
        Z.append(e['z'][idx]); Wn.append(fr[idx] // W); G.append(np.full(len(idx), j))
        NX.append(np.full(len(idx), sg['n'][0])); NY.append(np.full(len(idx), sg['n'][1]))
    U, Z, S, Wn, G, NX, NY = map(np.concatenate, (U, Z, S, Wn, G, NX, NY))
    ns = len(seg_ids)
    N = len(U)
    sn = S / 1000.0
    # columns: drift x (nwin), drift y (nwin) [u] or drift z (nwin) [z]; then 3 per segment
    if opts['drift'] == 'segment':
        # offset per (segment, window): no assumption that the drift is the same over the field
        key = G * nwin + Wn
        uk, inv = np.unique(key, return_inverse=True)
        nk = len(uk)
        npar_seg = np.bincount(uk // nwin, minlength=ns) + 2     # offsets of its windows + linear + quadratic in s

        def design(lateral):
            r = np.arange(N)
            rows = [r, r, r]; cl = [inv, nk + 2 * G, nk + 2 * G + 1]; vv = [np.ones(N), sn, sn ** 2]
            A = csr_matrix((np.concatenate(vv), (np.concatenate(rows), np.concatenate(cl))), shape=(N, nk + 2 * ns))
            return A, 0
    else:
        npar_seg = np.full(ns, 3)

    def design_global(lateral):
        r = np.arange(N)
        if lateral:
            rows = [r, r]; cl = [Wn, nwin + Wn]; vv = [NX, NY]; off = 2 * nwin
        else:
            rows = [r]; cl = [Wn]; vv = [np.ones(N)]; off = nwin
        for p in range(3):
            rows.append(r); cl.append(off + 3 * G + p); vv.append(sn ** p)
        A = csr_matrix((np.concatenate(vv), (np.concatenate(rows), np.concatenate(cl))), shape=(N, off + 3 * ns))
        return A, off
    if opts['drift'] != 'segment':
        design = design_global
    res_ = {}
    for nm_, y, lat in (('u', U, True), ('z', Z, False)):
        A, off = design(lat)
        keep = np.ones(N, bool)
        for it in range(iters):
            Ak = A[keep]
            sol = lsqr(Ak, y[keep], damp=1e-6, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
            r = y - A @ sol
            sd = np.array([rs(r[(G == j) & keep]) for j in range(ns)])
            keep = np.abs(r) < clipk * sd[G]
        # gauge: drift of window 0 is degenerate with the segment offsets; report drift relative to its mean
        if opts['drift'] == 'segment':
            dr = np.zeros((nwin, 2 if lat else 1))
        elif lat:
            dr = np.stack([sol[:nwin], sol[nwin:2 * nwin]], 1)
        else:
            dr = sol[:nwin, None]
        res_[nm_] = (r, dr - dr.mean(0))
    return dict(u=res_['u'][0], z=res_['z'][0], G=G, drift_xy=res_['u'][1], drift_z=res_['z'][1], N=N, npar_seg=npar_seg)


def em_gauss_unif(v, half, it=200):
    """Gaussian (mu, sigma) + uniform on [-half, half] by EM; returns mu, sigma, background fraction, responsibilities."""
    inside = np.abs(v) < half
    v = v[inside]
    mu, sd, f = np.median(v), rs(v), 0.1
    for _ in range(it):
        g = (1 - f) * np.exp(-0.5 * ((v - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        u = f / (2 * half)
        w = g / (g + u)
        f = 1 - w.mean()
        mu = (w * v).sum() / w.sum()
        sd = np.sqrt((w * (v - mu) ** 2).sum() / w.sum())
    return mu, sd, f, w, inside


def prec(l, idx, n_seg, mode='repeat'):
    """Per-spot precision (u in nm along the segment normal, z in nominal nm)."""
    e = E[l]
    if mode == 'crlb' or RA is None:
        fx = fy = fz = np.ones(len(idx))
    else:
        rp = RA['methods'][l]['repeat']
        key = 'normalized_robust_std'
        b = np.clip(np.searchsorted(pb, R['I'][idx], side='right') - 1, 0, len(pbn) - 1)
        fx = np.array([rp[f'photons {nm}']['x'][key] for nm in pbn])[b]
        fy = np.array([rp[f'photons {nm}']['y'][key] for nm in pbn])[b]
        fz = np.array([rp[f'photons {nm}']['z'][key] for nm in pbn])[b]
        if mode == 'repeat+10':
            fx, fy, fz = 1.1 * fx, 1.1 * fy, 1.1 * fz
    su = np.sqrt((n_seg[0] * e['crx'][idx] * fx) ** 2 + (n_seg[1] * e['cry'][idx] * fy) ** 2) * PXNM
    sz = e['crz'][idx] * fz
    return su, sz


def widths(l, fd, seg_ids, mode='repeat', bright=None):
    rows = []
    for j, k in enumerate(seg_ids):
        q = fd['G'] == j
        idx = seg_idx[k]
        ru, rz = fd['u'][q], fd['z'][q]
        su, sz = prec(l, idx, segs[k]['n'], mode)
        if bright is not None:
            lo_, hi_ = (bright, 1e12) if np.isscalar(bright) else bright
            b = (R['I'][idx] > lo_) & (R['I'][idx] <= hi_)
            ru, rz, su, sz = ru[b], rz[b], su[b], sz[b]
        if len(ru) < 50:
            rows.append(None)
            continue
        mu_u, sd_u, f_u, w_u, in_u = em_gauss_unif(ru, 250.0)
        mu_z, sd_z, f_z, w_z, in_z = em_gauss_unif(rz, 400.0)
        p_u = (w_u * su[in_u] ** 2).sum() / w_u.sum()
        p_z = (w_z * sz[in_z] ** 2).sum() / w_z.sum()
        nall = int(q.sum())
        dof = nall / max(nall - int(fd['npar_seg'][j]), 1)        # residual variance -> variance (fitted offsets)
        Wu2 = sd_u ** 2 * dof - p_u
        Wz2 = sd_z ** 2 * dof - p_z
        rows.append(dict(n=int(len(ru)), sd_u=float(sd_u), sd_z=float(sd_z), prec_u=float(np.sqrt(p_u)), prec_z=float(np.sqrt(p_z)),
                         bg_u=float(f_u), bg_z=float(f_z), Wu2=float(Wu2), Wz2=float(Wz2),
                         ratio=float(np.sqrt(Wz2 / Wu2)) if Wu2 > 0 and Wz2 > 0 else float('nan')))
    return rows


def pooled(rows, boot):
    ok = [r for r in rows if r is not None]
    Wu2 = np.array([r['Wu2'] for r in ok]); Wz2 = np.array([r['Wz2'] for r in ok])
    est = np.sqrt(max(Wz2.sum(), 0) / Wu2.sum()) if Wu2.sum() > 0 else float('nan')
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(ok), len(ok))
        bs.append(np.sqrt(max(Wz2[i].sum(), 0) / Wu2[i].sum()))
    bs = np.array(bs)
    rat = np.array([r['ratio'] for r in ok])
    rat = rat[np.isfinite(rat)]
    return dict(n_seg=len(ok), ratio_pooled=float(est), ratio_boot_sd=float(bs.std()),
                ratio_boot_ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                kappa_round=float(1 / est), kappa_round_ci95=[float(1 / np.quantile(bs, 0.975)), float(1 / np.quantile(bs, 0.025))],
                ratio_x075=float(0.75 * est), ratio_x075_ci95=[float(0.75 * np.quantile(bs, 0.025)), float(0.75 * np.quantile(bs, 0.975))],
                ratio_median_seg=float(np.median(rat)) if len(rat) else float('nan'),
                ratio_iqr_seg=[float(np.quantile(rat, 0.25)), float(np.quantile(rat, 0.75))] if len(rat) else [],
                n_seg_negative_W2=int(sum((r['Wu2'] <= 0) or (r['Wz2'] <= 0) for r in ok)),
                Wu_nm_rms=float(np.sqrt(max(Wu2.mean(), 0))), Wz_nm_rms=float(np.sqrt(max(Wz2.mean(), 0))),
                sd_u_median=float(np.median([r['sd_u'] for r in ok])), sd_z_median=float(np.median([r['sd_z'] for r in ok])),
                prec_u_median=float(np.median([r['prec_u'] for r in ok])), prec_z_median=float(np.median([r['prec_z'] for r in ok])))


_trap = getattr(np, 'trapezoid', None) or np.trapz


def ring1d(v, sig, R_):
    """density of a thin ring of radius R_ projected on one axis, convolved with N(0, sig^2) (per spot)."""
    th = np.linspace(0, np.pi, 33)
    c = R_ * np.cos(th)[None, :]
    g = np.exp(-0.5 * ((v[:, None] - c) / sig[:, None]) ** 2) / (sig[:, None] * np.sqrt(2 * np.pi))
    return _trap(g, th, axis=1) / np.pi


def fit1d(v, sp, half):
    """Gaussian vs projected ring, both with the spot precision sp plus a common extra width s, + uniform bg."""
    k = np.abs(v) < half
    v, sp = v[k], sp[k]

    def nllg(p):
        s2 = sp ** 2 + np.exp(2 * p[1]); f = 1 / (1 + np.exp(-p[2]))
        g = np.exp(-0.5 * (v - p[0]) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
        return -np.log((1 - f) * g + f / (2 * half)).sum()

    def nllr(p):
        sig = np.sqrt(sp ** 2 + np.exp(2 * p[1])); f = 1 / (1 + np.exp(-p[2]))
        return -np.log((1 - f) * ring1d(v - p[0], sig, abs(p[3])) + f / (2 * half)).sum()
    rg = minimize(nllg, [0, np.log(15), -2.5], method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-4, fatol=1e-4))
    best = None
    for R0 in (15, 25, 35):
        r_ = minimize(nllr, [rg.x[0], np.log(8), rg.x[2], R0], method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
        if best is None or r_.fun < best.fun:
            best = r_
    Rr, sr = abs(best.x[3]), np.exp(best.x[1])
    sm = float(np.median(sp))
    sig = np.sqrt(sm ** 2 + sr ** 2)
    p0 = ring1d(np.array([0.0]), np.array([sig]), Rr)[0]
    pR = ring1d(np.array([Rr]), np.array([sig]), Rr)[0]
    return dict(n=int(len(v)), gauss_extra_sd=float(np.exp(rg.x[1])), ring_R=float(Rr), ring_extra_sd=float(sr),
                dloglik_ring_vs_gauss=float(rg.fun - best.fun), median_prec=sm,
                dip_depth=float(1 - p0 / pR), resolved_dip=bool(p0 < pR))


def fit2d(u, z, su, sz, halfu=150.0, halfz=300.0, x0=None):
    """Elliptical thin ring (R in u, R / kappa in nominal z) + isotropic extra width s (s / kappa in z) + spot
    precision, + uniform background in the box."""
    k = (np.abs(u) < halfu) & (np.abs(z) < halfz)
    u, z, su, sz = u[k], z[k], su[k], sz[k]
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    cu, sn = np.cos(th)[None, :], np.sin(th)[None, :]

    def nll(p):
        Rr, kap, s, f = abs(p[0]), abs(p[1]), np.exp(p[2]), 1 / (1 + np.exp(-p[3]))
        vu = su ** 2 + s ** 2
        vz = sz ** 2 + (s / kap) ** 2
        g = np.exp(-0.5 * (u[:, None] - Rr * cu) ** 2 / vu[:, None] - 0.5 * (z[:, None] - Rr / kap * sn) ** 2 / vz[:, None]).mean(1)
        g = g / (2 * np.pi * np.sqrt(vu * vz))
        return -np.log((1 - f) * g + f / (4 * halfu * halfz)).sum()
    best = None
    starts = [[R0, k0, np.log(8), -2.5] for R0 in (15, 25) for k0 in (0.6, 0.9)] if x0 is None else [x0]
    for st in starts:
        r_ = minimize(nll, st, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-4, fatol=1e-3))
        if best is None or r_.fun < best.fun:
            best = r_
    if x0 is not None:
        return dict(kappa=float(abs(best.x[1])), R=float(abs(best.x[0])))
    # same with R fixed at 0 (a filled blob: Gaussian in both axes, its own widths) for the comparison
    def nll0(p):
        a, b_, f = np.exp(p[0]), np.exp(p[1]), 1 / (1 + np.exp(-p[2]))
        vu, vz = su ** 2 + a ** 2, sz ** 2 + b_ ** 2
        g = np.exp(-0.5 * u ** 2 / vu - 0.5 * z ** 2 / vz) / (2 * np.pi * np.sqrt(vu * vz))
        return -np.log((1 - f) * g + f / (4 * halfu * halfz)).sum()
    r0 = minimize(nll0, [np.log(15), np.log(20), -2.5], method='Nelder-Mead', options=dict(maxiter=3000))
    return dict(n=int(len(u)), R=float(abs(best.x[0])), kappa=float(abs(best.x[1])), extra_sd=float(np.exp(best.x[2])),
                bg=float(1 / (1 + np.exp(-best.x[3]))), dloglik_ring_vs_blob=float(r0.fun - best.fun),
                blob_extra_u=float(np.exp(r0.x[0])), blob_extra_z=float(np.exp(r0.x[1])), x=best.x.tolist())


# ------------------------------------------------------------------------------------------------ reference pass: z check
all_ids = list(range(len(segs)))
fdr = fit_drift(ref, all_ids)
keep_ids = []
for j, k in enumerate(all_ids):
    q = fdr['G'] == j
    frac = float((np.abs(fdr['z'][q]) > 200).mean())
    ufrac = float((np.abs(fdr['u'][q]) > 100).mean())
    segs[k]['z_out_frac'] = frac
    segs[k]['u_out_frac'] = ufrac
    segs[k]['z_median'] = float(np.median(R['z'][seg_idx[k]]))
    segs[k]['n_spots'] = int(q.sum())
    if frac <= opts['ztail'] and ufrac <= opts['utail'] and q.sum() >= opts['minloc']:
        keep_ids.append(k)
print('reference residual tails per segment (u > 100 nm, z > 200 nm, near ratio):',
      ' '.join(f'{segs[k]["u_out_frac"]:.2f}/{segs[k]["z_out_frac"]:.2f}/{segs[k]["near_ratio"]:.2f}' for k in all_ids), flush=True)
print(f'{len(keep_ids)} segments after the overlap check (residuals of the reference: > {opts["utail"]:.0%} beyond +-100 nm lateral or > {opts["ztail"]:.0%} beyond +-200 nm axial: {len(segs) - len(keep_ids)} dropped)', flush=True)
out['segments'] = [dict(id=k, centre_px=segs[k]['c'].tolist(), direction=segs[k]['t'].tolist(),
                        length_px=segs[k]['s_hi'] - segs[k]['s_lo'], n_spots=segs[k]['n_spots'], z_median_ref=segs[k]['z_median'],
                        flank_ratio=segs[k]['flank_ratio'], near_ratio=segs[k]['near_ratio'], z_out_frac=segs[k]['z_out_frac'], u_out_frac=segs[k]['u_out_frac'], kept=k in keep_ids) for k in all_ids]
out['n_segments'] = len(keep_ids)
out['n_spots_segments'] = int(sum(segs[k]['n_spots'] for k in keep_ids))
out['segment_selection_rejects'] = rejects

# ------------------------------------------------------------------------------------------------ per method
zmed = np.array([segs[k]['z_median'] for k in keep_ids])
zcut = np.quantile(zmed, [1 / 3, 2 / 3])
out['z_terciles_ref_nominal_nm'] = zcut.tolist()
out['methods'] = {}
resid_store = {}
for l in labs:
    t0 = time.time()
    fd = fit_drift(l, keep_ids)
    m = dict(drift_xy_nm_range=[float(np.ptp(fd['drift_xy'][:, 0])), float(np.ptp(fd['drift_xy'][:, 1]))],
             drift_z_nm_range=float(np.ptp(fd['drift_z'])), drift_xy=fd['drift_xy'], drift_z=fd['drift_z'][:, 0])
    rows = widths(l, fd, keep_ids)
    m['per_segment'] = rows
    m['moments'] = pooled(rows, opts['boot'])
    for mode in ('crlb', 'repeat+10'):
        m[f'moments_{mode}'] = pooled(widths(l, fd, keep_ids, mode=mode), opts['boot'])
    m['moments_bright3000'] = pooled(widths(l, fd, keep_ids, bright=3000), opts['boot'])
    # consistency of the precision model: the structure widths W_u, W_z must not depend on the brightness of the spots
    m['moments_by_photons'] = []
    for lo, hi in ((0, 1500), (1500, 3000), (3000, 6000), (6000, 1e12)):
        p_ = pooled(widths(l, fd, keep_ids, bright=(lo, hi)), max(opts['boot'] // 3, 20))
        p_['photons'] = [lo, hi]
        m['moments_by_photons'].append(p_)
    # by depth (terciles of the segments' median z in the reference)
    m['moments_by_z'] = []
    for lo, hi in ((-1e9, zcut[0]), (zcut[0], zcut[1]), (zcut[1], 1e9)):
        sel = [r for r, zz in zip(rows, zmed) if lo <= zz < hi]
        p_ = pooled(sel, opts['boot'])
        p_['z_range'] = [float(max(lo, -800)), float(min(hi, 800))]
        p_['z_median'] = float(np.median(zmed[(zmed >= lo) & (zmed < hi)]))
        m['moments_by_z'].append(p_)
    # pooled residuals for the hollow test (core spots)
    U, Z, SU, SZ = [], [], [], []
    for j, k in enumerate(keep_ids):
        q = fd['G'] == j
        su, sz = prec(l, seg_idx[k], segs[k]['n'])
        U.append(fd['u'][q]); Z.append(fd['z'][q]); SU.append(su); SZ.append(sz)
    U, Z, SU, SZ = map(np.concatenate, (U, Z, SU, SZ))
    resid_store[l] = (U, Z, SU, SZ)
    # the shape fits use a fixed random subset of at most nfit spots (same spots for every method: same seed, same N)
    GG = np.concatenate([np.full((fd['G'] == j).sum(), j) for j in range(len(keep_ids))])
    sub = np.sort(np.random.default_rng(7).permutation(len(U))[:opts['nfit']])
    U, Z, SU, SZ, GG = U[sub], Z[sub], SU[sub], SZ[sub], GG[sub]
    m['hollow_u'] = fit1d(U, SU, 150.0)
    m['hollow_z'] = fit1d(Z, SZ, 300.0)
    m['ring2d'] = fit2d(U, Z, SU, SZ)
    bk = []
    Gs = [GG == j for j in range(len(keep_ids))]
    for bi in range(opts['boot2d']):
        ii = rng.integers(0, len(keep_ids), len(keep_ids))
        q = np.concatenate([np.nonzero(Gs[i])[0] for i in ii])
        bk.append(fit2d(U[q], Z[q], SU[q], SZ[q], x0=m['ring2d']['x'])['kappa'])
    m['ring2d']['kappa_boot_sd'] = float(np.std(bk))
    m['ring2d']['kappa_boot_ci95'] = [float(np.quantile(bk, 0.025)), float(np.quantile(bk, 0.975))]
    m['ring2d']['n_boot'] = len(bk)
    out['methods'][l] = m
    mo = m['moments']
    print(f'{l}: {mo["n_seg"]} seg; sd u {mo["sd_u_median"]:.1f} z {mo["sd_z_median"]:.1f} nm; prec u {mo["prec_u_median"]:.1f} '
          f'z {mo["prec_z_median"]:.1f}; W u {mo["Wu_nm_rms"]:.1f} z {mo["Wz_nm_rms"]:.1f}; ratio z/u {mo["ratio_pooled"]:.3f} '
          f'[{mo["ratio_boot_ci95"][0]:.3f}, {mo["ratio_boot_ci95"][1]:.3f}] (x0.75: {mo["ratio_x075"]:.3f}); kappa_round '
          f'{mo["kappa_round"]:.3f}; crlb-only {m["moments_crlb"]["ratio_pooled"]:.3f}, prec+10% {m["moments_repeat+10"]["ratio_pooled"]:.3f}, '
          f'bright {m["moments_bright3000"]["ratio_pooled"]:.3f}; by photons (W_u, W_z) ' +
          ' '.join(f'{p["Wu_nm_rms"]:.0f},{p["Wz_nm_rms"]:.0f}' for p in m['moments_by_photons']) + '; by z ' +
          ' '.join(f'{p["ratio_pooled"]:.3f}' for p in m['moments_by_z']) +
          f'; ring2d kappa {m["ring2d"]["kappa"]:.3f} +- {m["ring2d"]["kappa_boot_sd"]:.3f} R {m["ring2d"]["R"]:.1f} s {m["ring2d"]["extra_sd"]:.1f} '
          f'dlogL {m["ring2d"]["dloglik_ring_vs_blob"]:.1f}; hollow u R {m["hollow_u"]["ring_R"]:.1f} dip {m["hollow_u"]["dip_depth"]:.2f} '
          f'dlogL {m["hollow_u"]["dloglik_ring_vs_gauss"]:.1f}, z R {m["hollow_z"]["ring_R"]:.1f} dip {m["hollow_z"]["dip_depth"]:.2f} '
          f'dlogL {m["hollow_z"]["dloglik_ring_vs_gauss"]:.1f}  {time.time() - t0:.0f}s', flush=True)
jdump(out, f'{OUT}/mt_{otag2}.json')

# ------------------------------------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
ax[0].imshow(np.log1p(sm), cmap='gray', origin='upper', extent=[0, Wd, Hd, 0])
for k in all_ids:
    sg = segs[k]
    p0 = sg['c'] + sg['s_lo'] * sg['t']; p1 = sg['c'] + sg['s_hi'] * sg['t']
    ax[0].plot([p0[0], p1[0]], [p0[1], p1[1]], '-', color='lime' if k in keep_ids else 'red', lw=1.5)
ax[0].set_title(f'{len(keep_ids)} segments used (green), {len(all_ids) - len(keep_ids)} dropped by the overlap check (red)')
sc = ax[1].scatter([segs[k]['c'][0] for k in keep_ids], [segs[k]['c'][1] for k in keep_ids], c=zmed, cmap='viridis', s=40)
ax[1].set_xlim(0, Wd); ax[1].set_ylim(Hd, 0); ax[1].set_aspect('equal'); plt.colorbar(sc, ax=ax[1], label=f'median z ({ref}, nominal nm)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_mt_{otag2}_segments.png', dpi=110); plt.close(fig)

nL = len(labs)
fig, ax = plt.subplots(3, nL, figsize=(3.3 * nL, 9.5), squeeze=False)
for i, l in enumerate(labs):
    U, Z, SU, SZ = resid_store[l]
    ax[0, i].hist2d(U, Z, bins=[np.arange(-120, 121, 6), np.arange(-200, 201, 10)], cmap='magma')
    ax[0, i].set_title(f'{l}: kappa(2D ring) {out["methods"][l]["ring2d"]["kappa"]:.2f}', fontsize=9)
    ax[0, i].set_xlabel('u (nm)'); ax[0, i].set_ylabel('z (nominal nm)')
    ax[1, i].hist(U[np.abs(U) < 150], np.arange(-150, 151, 5), density=True, color='C0', alpha=0.8)
    ax[1, i].set_xlabel('u (nm)'); ax[1, i].set_title(f'lateral; dip {out["methods"][l]["hollow_u"]["dip_depth"]:.2f}', fontsize=9)
    ax[2, i].hist(Z[np.abs(Z) < 300], np.arange(-300, 301, 10), density=True, color='C1', alpha=0.8)
    ax[2, i].set_xlabel('z (nominal nm)'); ax[2, i].set_title(f'axial; dip {out["methods"][l]["hollow_z"]["dip_depth"]:.2f}', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_mt_{otag2}_profiles.png', dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, l in enumerate(labs):
    mo = out['methods'][l]
    rr = [r['ratio'] for r in mo['per_segment'] if r is not None]
    ax[0].plot(np.full(len(rr), i) + rng.uniform(-0.15, 0.15, len(rr)), rr, '.', alpha=0.5)
    ax[0].errorbar(i + 0.3, mo['moments']['ratio_pooled'], yerr=np.abs(np.array(mo['moments']['ratio_boot_ci95'])[:, None] - mo['moments']['ratio_pooled']), fmt='ks')
    zz = [p['z_median'] for p in mo['moments_by_z']]
    ax[1].errorbar(zz, [1 / p['ratio_pooled'] for p in mo['moments_by_z']],
                   yerr=[p['ratio_boot_sd'] / p['ratio_pooled'] ** 2 for p in mo['moments_by_z']], fmt='o-', label=l, capsize=2)
ax[0].set_xticks(range(nL)); ax[0].set_xticklabels(labs); ax[0].set_ylim(0, 3); ax[0].axhline(1 / 0.75, color='k', lw=0.5, ls='--')
ax[0].set_ylabel('W_z / W_u (nominal z), per segment; pooled with 95 % CI'); ax[0].axhline(1, color='k', lw=0.5)
ax[1].axhline(0.75, color='k', lw=0.5, ls='--'); ax[1].set_xlabel(f'median z of the segment ({ref}, nominal nm)')
ax[1].set_ylabel('kappa_round = W_u / W_z'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_mt_{otag2}_ratio.png', dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
for l in labs:
    d_ = out['methods'][l]['drift_xy']
    ax.plot(np.arange(len(d_)) * W, d_[:, 0], '-', label=f'{l} x'); ax.plot(np.arange(len(d_)) * W, d_[:, 1], '--', label=f'{l} y')
    ax.plot(np.arange(len(d_)) * W, out['methods'][l]['drift_z'], ':', label=f'{l} z')
ax.set_xlabel('frame'); ax.set_ylabel('drift (nm)'); ax.legend(fontsize=6, ncol=3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_mt_{otag2}_drift.png', dpi=110); plt.close(fig)
print('done', time.time() - t00, flush=True)
