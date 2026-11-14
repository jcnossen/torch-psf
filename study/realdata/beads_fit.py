"""Step 2 of the bead calibration: one free-knot cubic (Catmull-Rom) spline PSF fitted to the stacks of many beads
with a Poisson likelihood, alternating (a) bead parameters at fixed knots and (b) knots at fixed bead parameters.

Bead model: frame k of bead b is  mu = I_bk * g(o - x_b, o - y_b, z_k - dz_b) + bg_bk,  z_k the stage position of the
plane (um), (x_b, y_b) shared by all planes of the bead (so the spline carries any z-dependent lateral shift of the
PSF), dz_b the stage position of the bead's own focus, initialised at the crossing sx = sy of Gaussian widths and refined
by a 1D likelihood search with the mean of the refinements held fixed (gauge). I_bk, bg_bk free per frame. Because I is
free per frame, the overall scale of each z plane of the spline is not identified by the data: every knot plane is
normalised so that the PSF sums to 1 over the PB x PB (23 x 23) bead patch at x = y = 0.

Noise: the stack is EMCCD data at EM gain 4 (excess factor unknown, between 1 and 2, plus read noise). Pass 1 treats
ADU as Poisson counts; the residuals of pass 1 give var = a + b mu (ADU); pass 2 fits the transformed data
y = (ADU + a/b) / b, which has var(y) = E(y) (a Poisson-like weighting). Absolute bead photon counts are not needed.

usage: python beads_fit.py stack.tif beads_found.json outdir [--maxdist 100] [--lam 1e4] [--lobo N] [--lams 0,1e3,...]
"""
import sys, json, time, argparse, math
import numpy as np, torch, tifffile
sys.path.insert(0, '.')
from torch_psf import ConvCubicSplinePSF
from torch_psf.fit import localize
from torch_psf.fit import poisson_nll
from rdcommon import gauss_knots, spline_widths, DEV

ap = argparse.ArgumentParser()
ap.add_argument('stack'); ap.add_argument('found'); ap.add_argument('outdir')
ap.add_argument('--maxdist', type=float, default=100.0)      # px from the movie ROI
ap.add_argument('--Imax', type=float, default=6.0)
ap.add_argument('--lam', type=float, default=1e4)
ap.add_argument('--lams', default='')                         # comma list: LOBO scan of the roughness weight
ap.add_argument('--lobo', type=int, default=0)                # number of held-out beads for the LOBO check (0: none, -1: all)
ap.add_argument('--outer', type=int, default=14)
ap.add_argument('--zr', type=float, default=0.8)
ap.add_argument('--hz', type=float, default=0.05)
ap.add_argument('--S', type=int, default=2)
ap.add_argument('--tag', default='')
ap.add_argument('--chi2max', type=float, default=1.5)
ap.add_argument('--noise', default='linear')        # linear (NPC) or quad (Tubulin: very bright beads)
A = ap.parse_args()
torch.manual_seed(0)
fd = json.load(open(A.found))
zs = np.array(fd['zs'])
PB = fd['PB']; hb = PB // 2
st = tifffile.imread(A.stack).astype(np.float32)
allb = [b for b in fd['beads'] if np.isfinite(b['zc']) and b['shape_rms'] < 0.35 and 0.3 < b['Irel'] < A.Imax
        and abs(b['zc']) < 0.5]
near = [b for b in allb if b['dist_roi'] < A.maxdist]
far = [b for b in allb if b['dist_roi'] >= A.maxdist]
print(f'beads passing shape/brightness: {len(allb)}; used (within {A.maxdist:.0f} px of the movie ROI): {len(near)}; '
      f'held out for the field check: {len(far)}', flush=True)


def crop(beads):
    d, z, bid, kk = [], [], [], []
    for i, b in enumerate(beads):
        p = st[:, b['y'] - hb:b['y'] + hb + 1, b['x'] - hb:b['x'] + hb + 1]
        for k in range(len(zs)):
            zk = zs[k] - b['zc']
            if abs(zk) <= A.zr + 0.1:
                d.append(p[k]); z.append(zk); bid.append(i); kk.append(k)
    return (torch.tensor(np.array(d), device=DEV), torch.tensor(z, device=DEV, dtype=torch.float32),
            torch.tensor(bid, device=DEV), np.array(kk))


def new_spline():
    return ConvCubicSplinePSF(PB, (-A.zr, A.zr), oversample=A.S, dz_um=A.hz, zpad=4, xy_limit=1.0, device=DEV)


def init_knots(spl):
    mc = fd['median_curve']
    zk = spl.zs.cpu().numpy()
    sx = np.interp(zk, mc['z'], mc['sx']); sy = np.interp(zk, mc['z'], mc['sy'])
    return normalize(spl, gauss_knots(spl, zk, sx, sy))


def normalize(spl, K):
    """(1) A constant added to a knot plane is exactly degenerate with the free background of the bead frames; the
    pedestal is fixed by setting the mean of the knots on the border ring of the bead patch (|u| or |v| between
    hb - 0.5 and hb + 0.5 px) to zero. (2) Each plane is scaled to sum to 1 over the PB x PB patch at x = y = 0."""
    u = spl.u0 + torch.arange(spl.nu, device=DEV, dtype=torch.float32) / spl.S
    r = torch.maximum(u.abs()[None, :], u.abs()[:, None])
    ring = (r >= hb - 0.5) & (r <= hb + 0.5)
    K = K - K[:, ring].mean(1)[:, None, None]
    off = torch.zeros(spl.nz, 3, device=DEV)
    off[:, 2] = spl.zs.float()
    off[:, 2] = off[:, 2].clamp(spl.z0 + spl.hz, spl.z0 + (spl.nz - 2) * spl.hz - 1e-4)
    s = spl.render(off, knots=K).sum((1, 2))
    return K / s[:, None, None].clamp_min(1e-12)


def bead_fit(spl, d, z, bid, nb, dz, th=None, iters=10):
    """x, y shared per bead (mean over |z| < 0.5 um planes), z = z_plane - dz_b fixed, I and bg per frame."""
    zz = z - dz[bid]
    if th is None:
        bg = d.flatten(1).quantile(0.1, 1)
        I = (d.flatten(1).sum(1) - bg * PB * PB).clamp_min(100)
        th = torch.stack([torch.zeros_like(I), torch.zeros_like(I), zz, I, bg], 1)
    th = th.clone(); th[:, 2] = zz
    th, _ = localize(spl, d, th, iters=iters, fixed=(2,))
    w = (zz.abs() < 0.5).float()
    xy = torch.zeros(nb, 2, device=DEV).index_add_(0, bid, th[:, :2] * w[:, None]) / \
        torch.zeros(nb, device=DEV).index_add_(0, bid, w)[:, None].clamp_min(1)
    th[:, :2] = xy[bid]
    th, nll = localize(spl, d, th, iters=iters, fixed=(0, 1, 2))
    return th, nll


def refine_dz(spl, d, z, bid, nb, dz, th, span=0.04, n=9):
    """1D likelihood search of the focus offset of each bead (I, bg refitted per trial), parabolic refinement."""
    grid = torch.linspace(-span, span, n, device=DEV)
    tot = torch.zeros(nb, n, device=DEV)
    for j, g in enumerate(grid):
        t = th.clone(); t[:, 2] = z - dz[bid] - g
        t, nll = localize(spl, d, t, iters=6, fixed=(0, 1, 2))
        tot[:, j] = torch.zeros(nb, device=DEV).index_add_(0, bid, nll)
    j = tot.argmin(1).clamp(1, n - 2)
    a, b_, c = tot.gather(1, (j - 1)[:, None])[:, 0], tot.gather(1, j[:, None])[:, 0], tot.gather(1, (j + 1)[:, None])[:, 0]
    den = (a - 2 * b_ + c)
    sh = torch.where(den > 0, 0.5 * (a - c) / den, torch.zeros_like(den)).clamp(-1, 1)
    step = grid[j] + sh * (grid[1] - grid[0])
    step = step - step.mean()                        # gauge: the mean focus offset stays where it was
    return dz + step, step


def rough(K):
    dzz = K[2:] - 2 * K[1:-1] + K[:-2]
    dy = K[:, 2:] - 2 * K[:, 1:-1] + K[:, :-2]
    dx = K[:, :, 2:] - 2 * K[:, :, 1:-1] + K[:, :, :-2]
    return dzz, dy, dx


def fit(beads, d, z, bid, lam, outer, log=True, K0=None):
    nb = len(beads)
    spl = new_spline()
    K = init_knots(spl) if K0 is None else K0.clone()
    dz = torch.zeros(nb, device=DEV)
    wsm = 1.0 / (K.abs() + 1e-4)
    th = None
    t0 = time.time()
    for o in range(outer):
        spl.knots = K
        th, nll = bead_fit(spl, d, z, bid, nb, dz, th)
        if o in (2, 5, 8, 11) or (o > 11 and o % 3 == 2):
            dz, step = refine_dz(spl, d, z, bid, nb, dz, th)
            th, nll = bead_fit(spl, d, z, bid, nb, dz, th)
        Kp = K.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([Kp], max_iter=12, history_size=10, line_search_fn='strong_wolfe')
        zz = z - dz[bid]
        tt = th.clone(); tt[:, 2] = zz

        def closure():
            opt.zero_grad()
            mu = tt[:, 3, None, None] * spl.render(tt[:, :3], knots=Kp) + tt[:, 4, None, None]
            l = poisson_nll(d, mu).sum()
            if lam > 0:
                a, b_, c = rough(Kp)
                l = l + lam * ((wsm[1:-1] * a ** 2).sum() + (wsm[:, 1:-1] * b_ ** 2).sum() + (wsm[:, :, 1:-1] * c ** 2).sum())
            l.backward()
            return l
        l = opt.step(closure)
        K = normalize(spl, Kp.detach())
        if log:
            print(f'  outer {o:2d}: loss {float(l.detach()):.6e}  nll {float(nll.sum()):.6e}  dz range {float(dz.min()):+.3f} '
                  f'{float(dz.max()):+.3f}  {time.time() - t0:.1f}s', flush=True)
    spl.knots = K
    th, nll = bead_fit(spl, d, z, bid, nb, dz, th)
    return spl, K, dz, th, nll


def chi2(spl, d, th):
    mu = (th[:, 3, None, None] * spl.render(th[:, :3]) + th[:, 4, None, None]).clamp_min(1e-6)
    return ((d - mu) ** 2 / mu).flatten(1).mean(1), mu


def heldout(spl, beads_h, d, z, bid, iters_dz=3):
    """Fit held-out beads with a fixed spline: x, y, I, bg and the focus offset dz (no gauge constraint)."""
    nb = len(beads_h)
    dz = torch.zeros(nb, device=DEV)
    th, _ = bead_fit(spl, d, z, bid, nb, dz)
    for _ in range(iters_dz):
        grid = torch.linspace(-0.05, 0.05, 11, device=DEV)
        tot = torch.zeros(nb, 11, device=DEV)
        for j, g in enumerate(grid):
            t = th.clone(); t[:, 2] = z - dz[bid] - g
            t, nll = localize(spl, d, t, iters=6, fixed=(0, 1, 2))
            tot[:, j] = torch.zeros(nb, device=DEV).index_add_(0, bid, nll)
        dz = dz + grid[tot.argmin(1)]
        th, _ = bead_fit(spl, d, z, bid, nb, dz, th)
    c2, mu = chi2(spl, d, th)
    return c2, th, dz


def per_bead_mean(v, bid, nb):
    return (torch.zeros(nb, device=DEV).index_add_(0, bid, v) /
            torch.zeros(nb, device=DEV).index_add_(0, bid, torch.ones_like(v)))


def lobo_one(beads, d, z, bid, h, lam, outer):
    nbd = len(beads)
    keep = [i for i in range(nbd) if i != h]
    sel = torch.isin(bid, torch.tensor(keep, device=DEV))
    remap = torch.full((nbd,), -1, device=DEV, dtype=torch.long)
    remap[torch.tensor(keep, device=DEV)] = torch.arange(len(keep), device=DEV)
    splh, _, _, _, _ = fit([beads[i] for i in keep], d[sel], z[sel], remap[bid[sel]], lam, outer, log=False)
    hs = bid == int(h)
    c2h, thh, dzh = heldout(splh, [beads[h]], d[hs], z[hs], torch.zeros(int(hs.sum()), dtype=torch.long, device=DEV))
    # localization of the held-out frames with all five parameters free, started at the truth: the systematic error
    # of the spline for a bead it has not seen (z truth = stage position - fitted focus offset of the bead)
    ztrue = z[hs] - dzh[0]
    thf, _ = localize(splh, d[hs], thh, iters=30)
    m = ztrue.abs() < 0.6
    ez = (thf[:, 2] - ztrue)[m] * 1000
    ex = (thf[:, 0] - thf[m, 0].mean())[m]; ey = (thf[:, 1] - thf[m, 1].mean())[m]
    return dict(chi2=float(c2h.mean()), z_rmse_nm=float(ez.pow(2).mean().sqrt()), z_bias_max_nm=float(ez.abs().max()),
                x_rms_px=float(ex.pow(2).mean().sqrt()), y_rms_px=float(ey.pow(2).mean().sqrt()))


# ------------------------------------------------------------------------------------------------ pass 1: noise model
d0, z0, bid0, kk0 = crop(near)
print(f'bead frames used: {len(d0)} ({len(near)} beads)', flush=True)
spl, K, dz, th, nll = fit(near, d0, z0, bid0, 0.0, 6, log=False)
_, mu = chi2(spl, d0, th)
r2 = ((d0 - mu) ** 2).flatten().cpu().numpy(); m = mu.flatten().cpu().numpy()
q = np.quantile(m, np.linspace(0, 1, 31))
bc, bv = [], []
for lo_, hi_ in zip(q[:-1], q[1:]):
    s = (m >= lo_) & (m < hi_)
    if s.sum() > 100:
        bc.append(m[s].mean()); bv.append(r2[s].mean())
bc, bv = np.array(bc), np.array(bv)
# var = a + b mu + c mu^2: the quadratic term takes the PSF model error of pass 1 (it grows with the signal), a + b mu
# is used as the noise model
if A.noise == 'ptc':
    # photon transfer from the stack itself: with fine z steps adjacent planes differ mainly by noise, so the second
    # difference along z, (x[k+1] - 2 x[k] + x[k-1])^2 / 6, estimates the pixel variance at mean x[k]
    raw = np.stack([st[:, b['y'] - hb:b['y'] + hb + 1, b['x'] - hb:b['x'] + hb + 1] for b in near]).astype(np.float64)
    s2 = ((raw[:, 2:] - 2 * raw[:, 1:-1] + raw[:, :-2]) ** 2 / 6).ravel()
    mm_ = raw[:, 1:-1].ravel()
    qq = np.quantile(mm_, np.linspace(0, 0.995, 41))
    pc, pv = [], []
    for lo_, hi_ in zip(qq[:-1], qq[1:]):
        s_ = (mm_ >= lo_) & (mm_ < hi_)
        if s_.sum() > 200:
            v_ = s2[s_]
            v_ = v_[v_ < 10 * np.median(v_)]
            pc.append(mm_[s_].mean()); pv.append(v_.mean())
    pc, pv = np.array(pc), np.array(pv)
    (nb_b, na), nc_ = np.polyfit(pc, pv, 1, w=1.0 / pv), 0.0
    print('photon transfer from second differences along z (mean:var): '
          + ', '.join(f'{a:.0f}:{b:.1f}' for a, b in zip(pc[::4], pv[::4])), flush=True)
    bc, bv = pc, pv
elif A.noise == 'quad':
    nc_, nb_b, na = np.polyfit(bc, bv, 2, w=1.0 / bv)
else:   # linear fit below the 90 % quantile of mu (the brightest decile carries the PSF model error)
    lowm = bc < np.quantile(m, 0.9)
    (nb_b, na), nc_ = np.polyfit(bc[lowm], bv[lowm], 1), 0.0
print(f'quadratic fit of the residual variance: c = {nc_:.3e} (model error), b = {nb_b:.4f}, a = {na:.2f}', flush=True)
print(f'noise model of the bead stack (ADU): var = {na:.1f} + {nb_b:.3f} mu   (bins mu:var '
      + ', '.join(f'{a:.0f}:{b:.0f}' for a, b in zip(bc[::4], bv[::4])) + ')', flush=True)


def conv(x):
    return (x + na / nb_b) / nb_b


# ------------------------------------------------------------------------------------------------ pass 2
d, z, bid, kk = crop(near)
d = conv(d)
res = dict(noise=dict(a=float(na), b=float(nb_b), c_model_error=float(nc_), bins=[[float(a), float(b)] for a, b in zip(bc, bv)]))
nbd = len(near)
# outlier beads (neighbours, debris): a first fit, then beads with chi2/pixel above chi2max are dropped
spl, K, dz, th, nll = fit(near, d, z, bid, A.lam, A.outer, log=False)
c2, mu = chi2(spl, d, th)
pb0 = per_bead_mean(c2, bid, nbd).cpu().numpy()
drop = [i for i in range(nbd) if pb0[i] > A.chi2max]
print(f'first fit chi2/pixel per bead: {np.round(pb0, 2).tolist()}; dropped (> {A.chi2max}): '
      f'{[(near[i]["x"], near[i]["y"], round(float(pb0[i]), 2)) for i in drop]}', flush=True)
res['dropped'] = [dict(x=near[i]['x'], y=near[i]['y'], chi2=float(pb0[i])) for i in drop]
near = [b for i, b in enumerate(near) if i not in drop]
d, z, bid, kk = crop(near)
d = conv(d)
nbd = len(near)
print(f'beads used after the outlier cut: {nbd}', flush=True)

if A.lams:
    # leave-one-bead-out scan of the roughness weight on up to 8 held-out beads
    lams = [float(x) for x in A.lams.split(',')]
    hold = np.random.default_rng(0).choice(nbd, size=min(8, nbd), replace=False)
    scan = {}
    for lam in lams:
        rr = [lobo_one(near, d, z, bid, int(h), lam, A.outer) for h in hold]
        c2s = [r_['chi2'] for r_ in rr]
        print(f'   LOBO lam {lam:g}: z rmse (nm) {[round(r_["z_rmse_nm"], 1) for r_ in rr]}  x rms (px) {[round(r_["x_rms_px"], 3) for r_ in rr]}', flush=True)
        scan[lam] = c2s
        print(f'LOBO lam {lam:g}: held-out chi2/pixel per bead {np.round(c2s, 3).tolist()}  mean {np.mean(c2s):.4f}', flush=True)
    res['lam_scan'] = {str(k): v for k, v in scan.items()}
    res['lam_scan_beads'] = [int(h) for h in hold]
    json.dump(res, open(f'{A.outdir}/lam_scan{A.tag}.json', 'w'), indent=1)
    sys.exit(0)

spl, K, dz, th, nll = fit(near, d, z, bid, A.lam, A.outer)
c2, mu = chi2(spl, d, th)
per_bead = per_bead_mean(c2, bid, nbd)
print('in-sample chi2/pixel per bead:', np.round(per_bead.cpu().numpy(), 3).tolist())
zb = np.round((z - dz[bid]).cpu().numpy() / 0.1) * 0.1
c2n = c2.cpu().numpy()
per_z = {f'{u:+.1f}': float(c2n[zb == u].mean()) for u in np.unique(zb)}
print('in-sample chi2/pixel per z (0.1 um bins):', {k: round(v, 3) for k, v in per_z.items()})
zg = np.round(np.arange(-A.zr, A.zr + 1e-6, 0.05), 3)
sx, sy, cx, cy, tot = spline_widths(spl, zg)
dd_ = sx - sy
zc_spl = float('nan')
for i in range(len(zg) - 1):
    if dd_[i] * dd_[i + 1] <= 0 and dd_[i] != dd_[i + 1]:
        zc_spl = float(zg[i] + (zg[i + 1] - zg[i]) * dd_[i] / (dd_[i] - dd_[i + 1]))
print('spline widths (z, sx, sy, cx, cy):', [(float(a), round(float(b), 2), round(float(c), 2), round(float(e), 2), round(float(f), 2))
                                             for a, b, c, e, f in zip(zg[::2], sx[::2], sy[::2], cx[::2], cy[::2])])
print(f'z where the spline second-moment widths are equal: {zc_spl:+.3f} um')
out = dict(res, beads_used=[dict(x=b['x'], y=b['y'], zc=b['zc'], dz=float(dz[i]), chi2=float(per_bead[i]),
                                  dist_roi=b['dist_roi'], Irel=b['Irel']) for i, b in enumerate(near)],
           chi2_per_z=per_z, widths=dict(z=zg.tolist(), sx=sx.tolist(), sy=sy.tolist(), cx=cx.tolist(), cy=cy.tolist()),
           z_equal_width=zc_spl, settings=dict(PB=PB, S=A.S, hz=A.hz, zr=A.zr, zpad=4, xmax_bead=1.0, lam=A.lam,
                                                outer=A.outer, z0=float(spl.z0), nz=spl.nz, c=spl.c, maxdist=A.maxdist))
if far:
    df, zf, bf, _ = crop(far)
    c2f, thf, dzf = heldout(spl, far, conv(df), zf, bf)
    pf = per_bead_mean(c2f, bf, len(far))
    out['far_beads'] = [dict(x=b['x'], y=b['y'], dist_roi=b['dist_roi'], chi2=float(pf[i]), dz=float(dzf[i])) for i, b in enumerate(far)]
    print(f'beads far from the ROI (>= {A.maxdist:.0f} px), fitted with this spline: chi2/pixel median {float(pf.median()):.3f} '
          f'(range {float(pf.min()):.3f}-{float(pf.max()):.3f})', flush=True)
if A.lobo:
    idx = range(nbd) if A.lobo < 0 else np.random.default_rng(1).choice(nbd, size=min(A.lobo, nbd), replace=False)
    lo = []
    for h in idx:
        r_ = lobo_one(near, d, z, bid, int(h), A.lam, A.outer)
        lo.append(dict(bead=int(h), chi2_heldout=r_['chi2'], chi2_insample=float(per_bead[h]), **{k: v for k, v in r_.items() if k != 'chi2'}))
        print(f'  LOBO bead {int(h):2d}: held-out chi2/pixel {r_["chi2"]:.3f}  in-sample {float(per_bead[h]):.3f}  z rmse {r_["z_rmse_nm"]:.1f} nm  '
              f'max |z err| {r_["z_bias_max_nm"]:.1f} nm  x,y rms {r_["x_rms_px"]:.3f},{r_["y_rms_px"]:.3f} px', flush=True)
    out['lobo'] = lo
torch.save(dict(knots=K.cpu(), PB=PB, S=A.S, hz=A.hz, zlo=-A.zr, zhi=A.zr, zpad=4, xmax=1.0, c=spl.c, z0=float(spl.z0),
                nz=spl.nz, normalization=f'each knot plane sums to 1 over the {PB}x{PB} patch at x=y=0',
                z_convention='z = stage position (um) relative to the focus of each bead (dz_b); dz_b initialised at the '
                             'crossing sx=sy of Gaussian widths, mean of the likelihood refinements held at 0',
                noise=res['noise']), f'{A.outdir}/psf_spline{A.tag}.pt')
json.dump(out, open(f'{A.outdir}/bead_fit{A.tag}.json', 'w'), indent=1)
np.savez(f'{A.outdir}/bead_fit_data{A.tag}.npz', d=d.cpu().numpy(), mu=mu.cpu().numpy(), z=(z - dz[bid]).cpu().numpy(),
         bid=bid.cpu().numpy(), th=th.cpu().numpy())
print('saved', flush=True)
