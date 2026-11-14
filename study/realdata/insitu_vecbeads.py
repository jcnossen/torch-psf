"""Step 1: vectorial pupil model of the bead PSF (NPC-A647-3D bead stack, the 1,080 frames of the 30 calibration beads
that the free-knot spline was fitted to, `locs/calib/bead_fit_data.npz`, data already in the noise-transformed units
y = (ADU + a/b) / b with var(y) = E(y)).

Model of frame k of bead b: mu = I_k * PSF(x_b, y_b, z_e = 0; focus f0 + s (z_k - dz_b)) + bg_k with
  z_k        stage position of the plane relative to the bead focus of the spline calibration (um),
  x_b, y_b   bead position (px), shared by the planes of a bead,
  dz_b       focus correction of the bead (mean held at 0: gauge with f0),
  I_k, bg_k  per frame, profiled out (Newton per frame at every evaluation; envelope theorem for the gradient),
global: Zernike coefficients (waves rms), NA, pixel size x and y (um), wavelength (fixed unless --fit-lam), f0, and
optionally a radial pupil amplitude exp(a1 r^2 + a2 r^4 + a3 r^6) (--apod). Beads on the coverslip in water: n_med
1.33, n_imm 1.518. s = +-1: direction of the stage axis relative to the model's focus axis (not known; both tried).
L-BFGS on all of these except I_k, bg_k.

usage: python insitu_vecbeads.py --tag T [--nmax 21] [--s 1] [--z6 0.1] [--fix NA=1.45,lam=0.68,...] [--apod]
"""
import sys, json, time, argparse, math
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np, torch
from insitu_common import AVecPSF, DEV, D, OUT, jdump, PX0, PY0
from torch_psf.fit import poisson_nll

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--nmax', type=int, default=21)          # highest Noll index (from 5)
ap.add_argument('--s', type=float, default=1.0)
ap.add_argument('--z6', type=float, default=0.1)
ap.add_argument('--fix', default='lam=0.68')               # comma list name=value of fixed globals (NA, px, py, lam, ratio)
ap.add_argument('--apod', action='store_true')
ap.add_argument('--iters', type=int, default=400)
ap.add_argument('--npupil', type=int, default=64)
ap.add_argument('--os', type=int, default=3)
ap.add_argument('--init', default='')                      # json state to start from
ap.add_argument('--NA0', type=float, default=1.45)
ap.add_argument('--px0', type=float, default=PX0)
ap.add_argument('--py0', type=float, default=PY0)
ap.add_argument('--zmax', type=float, default=0.9)
ap.add_argument('--adam', type=int, default=3000)
ap.add_argument('--lr', type=float, default=1e-2)
ap.add_argument('--height', action='store_true')         # fit the height of the bead centre above the glass
ap.add_argument('--zscale', action='store_true')         # fit a scale factor of the stage axis (piezo calibration)
ap.add_argument('--poly', type=float, default=0.0)       # um: also lam -+ poly (weights 1/4, 1/2, 1/4), broad emission
ap.add_argument('--maps', action='store_true')           # free pupil phase and log-amplitude maps (as method d)
ap.add_argument('--blur', action='store_true')           # fit a Gaussian lateral blur (finite bead size)
A = ap.parse_args()
torch.manual_seed(0)
fix = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in A.fix.split(',') if kv}

f = np.load(f'{D}/locs/calib/bead_fit_data.npz')
sel = np.abs(f['z']) <= A.zmax + 1e-6
d = torch.tensor(f['d'][sel], device=DEV)
zk = torch.tensor(f['z'][sel], device=DEV)
bid = torch.tensor(f['bid'][sel], device=DEV)
mu_spl = torch.tensor(f['mu'][sel], device=DEV)
th_spl = torch.tensor(f['th'][sel], device=DEV)
nb, N, PB = int(bid.max()) + 1, len(d), d.shape[1]
noll = tuple(range(5, A.nmax + 1))
print(f'{N} bead frames, {nb} beads, patch {PB}, Noll 5-{A.nmax} ({len(noll)} terms), s {A.s:+.0f}, fixed {fix}', flush=True)

psf = AVecPSF(PB, px=A.px0, py=A.py0, lam=fix.get('lam', 0.68), NA=A.NA0, zernike_indices=noll, npupil=A.npupil,
              oversample=A.os, device=DEV)
base = {k: v.detach() for k, v in list(psf.named_parameters()) + list(psf.named_buffers())}

# ------------------------------------------------------------------------------------------------ parameters
xy0 = torch.zeros(nb, 2, device=DEV).index_add_(0, bid, th_spl[:, :2]) / torch.bincount(bid, minlength=nb)[:, None]
T = dict(zc=torch.zeros(len(noll), device=DEV), aNA=torch.zeros(1, device=DEV), apx=torch.zeros(1, device=DEV),
         apy=torch.zeros(1, device=DEV), alam=torch.zeros(1, device=DEV), af0=torch.zeros(1, device=DEV),
         xy=xy0.clone(), adz=torch.zeros(nb, device=DEV), apod=torch.zeros(3, device=DEV),
         ablur=torch.full((1,), 3.0, device=DEV), ahe=torch.full((1,), 3.0, device=DEV),
         azs=torch.zeros(1, device=DEV), phase=torch.zeros(A.npupil, A.npupil, device=DEV),
         la=torch.zeros(A.npupil, A.npupil, device=DEV))
from torch_psf.zernike import zernike
_mask = psf.mask.bool()
_low = torch.stack([zernike(j, psf.rho.clamp(max=1), psf.phi)[_mask] for j in (1, 2, 3, 4)], 1)
_Q, _ = torch.linalg.qr(_low)
T['zc'][noll.index(6)] = A.z6
if A.init:
    st = json.load(open(A.init))
    for j, c in zip(st['noll'], st['zcoef']):
        if j in noll:
            T['zc'][noll.index(j)] = c
    T['aNA'][0] = (st['NA'] - A.NA0) / 0.01
    T['apx'][0] = (st['pxy'][0] / A.px0 - 1) / 0.01
    T['apy'][0] = (st['pxy'][1] / A.py0 - 1) / 0.01
    T['af0'][0] = st['f0'] / 0.01
    T['xy'] = torch.tensor(st['bead_xy'], device=DEV)
    T['adz'] = torch.tensor(st['bead_dz'], device=DEV) / 0.01
    if 'apod' in st and A.apod:
        T['apod'] = torch.tensor(st['apod'], device=DEV)
    if st.get('bead_height', 0) > 0 and A.height:
        T['ahe'][0] = st['bead_height'] / 0.01
    if 'blur_sigma' in st and A.blur and st['blur_sigma'] > 0:
        T['ablur'][0] = st['blur_sigma'] / 0.01
free = ['zc', 'af0', 'xy', 'adz']
if 'NA' not in fix: free.append('aNA')
if 'px' not in fix and 'ratio' not in fix: free.append('apx')
if 'py' not in fix: free.append('apy')
if 'lam' not in fix: free.append('alam')
if A.apod: free.append('apod')
if A.blur: free.append('ablur')
if A.height and 'he' not in fix: free.append('ahe')
if A.zscale: free.append('azs')
if A.maps: free += ['phase', 'la']
for k in free:
    T[k].requires_grad_(True)


def globals_(T):
    NA = fix['NA'] + 0 * T['aNA'][0] if 'NA' in fix else (A.NA0 + 0.01 * T['aNA'][0]).clamp(1.30, 1.515)
    py = torch.as_tensor(fix['py'], device=DEV) if 'py' in fix else A.py0 * (1 + 0.01 * T['apy'][0])
    if 'ratio' in fix:
        px = py * fix['ratio']
    elif 'px' in fix:
        px = torch.as_tensor(fix['px'], device=DEV)
    else:
        px = A.px0 * (1 + 0.01 * T['apx'][0])
    lam = torch.as_tensor(fix['lam'], device=DEV) if 'lam' in fix else 0.68 * (1 + 0.01 * T['alam'][0])
    f0 = 0.01 * T['af0'][0]
    return NA, px, py, lam, f0


def model_images(T, idx):
    NA, px, py, lam, f0 = globals_(T)
    prm = dict(base)
    prm.update(zcoef=T['zc'], NA=NA, pxy=torch.stack([px, py]), lam=lam, apod=T['apod'])
    if A.maps:
        ph = T['phase'][_mask]
        ph = ph - _Q @ (_Q.T @ ph)                        # piston, tilt, defocus: gauges of bead position and focus
        la = T['la'][_mask]
        la = la - la.mean()                               # overall transmission: gauge of the bead intensity
        prm['pupil_phase'] = torch.zeros_like(T['phase']).masked_scatter(_mask, ph)
        prm['pupil_amp'] = torch.ones_like(T['la']).masked_scatter(_mask, torch.exp(la))
    dz = 0.01 * (T['adz'] - T['adz'].mean())
    b = bid[idx]
    he = (torch.as_tensor(fix['he'], device=DEV) if 'he' in fix else 0.01 * T['ahe'][0].abs()) if (A.height or 'he' in fix)         else torch.zeros((), device=DEV)
    pos = torch.cat([T['xy'][b], he.expand(len(idx), 1)], 1)
    foc = f0 + A.s * (1 + 0.01 * T['azs'][0]) * (zk[idx] - dz[b])
    if A.poly > 0:
        g = 0
        for wl, wt in ((lam - A.poly, 0.25), (lam, 0.5), (lam + A.poly, 0.25)):
            p2 = dict(prm, lam=wl, zcoef=T['zc'] * lam / wl)      # aberrations are fixed path differences
            g = g + wt * torch.func.functional_call(psf, p2, (pos, foc))
    else:
        g = torch.func.functional_call(psf, prm, (pos, foc))
    if A.blur:
        g = blur_img(g, 0.01 * T['ablur'][0].abs(), px, py)
    return g


def blur_img(g, sig, px, py):
    """Lateral Gaussian blur (sigma um) of the pixel images, applied as a multiplication of their discrete Fourier
    transform (the images are zero-padded to twice their size first): a finite bead diameter."""
    n = g.shape[-1]
    G = torch.fft.rfft2(torch.nn.functional.pad(g, (0, n, 0, n)))
    fy = torch.fft.fftfreq(2 * n, device=DEV)[:, None] / py
    fx = torch.fft.rfftfreq(2 * n, device=DEV)[None, :] / px
    H = torch.exp(-2 * math.pi ** 2 * sig ** 2 * (fx ** 2 + fy ** 2))
    return torch.fft.irfft2(G * H, s=(2 * n, 2 * n))[..., :n, :n]


IB = None


@torch.no_grad()
def profile_Ibg(g, dd, IB0, it=8):
    """Poisson MLE of (I, bg) per frame for fixed PSF images g (damped Newton)."""
    if IB0 is None:
        bg = dd.flatten(1).quantile(0.1, 1)
        I = ((dd.flatten(1).sum(1) - bg * PB * PB) / g.flatten(1).sum(1)).clamp_min(10)
        IB0 = torch.stack([I, bg], 1)
    I, bg = IB0[:, 0].clone(), IB0[:, 1].clone()
    gf, df = g.flatten(1), dd.flatten(1)
    for _ in range(it):
        mu = (I[:, None] * gf + bg[:, None]).clamp_min(1e-3)
        r = 1 - df / mu
        gI, gb = (r * gf).sum(1), r.sum(1)
        w = df / mu ** 2
        hII, hIb, hbb = (w * gf * gf).sum(1), (w * gf).sum(1), w.sum(1)
        det = (hII * hbb - hIb ** 2).clamp_min(1e-12)
        sI = (hbb * gI - hIb * gb) / det
        sb = (hII * gb - hIb * gI) / det
        I = (I - sI).clamp_min(1.0)
        bg = (bg - sb).clamp_min(1e-2)
    return torch.stack([I, bg], 1)


CH = 360
chunks = [torch.arange(s, min(s + CH, N), device=DEV) for s in range(0, N, CH)]


def loss_and_grad(T, need_grad=True):
    global IB
    newIB = []
    tot = 0.0
    for idx in chunks:
        g = model_images(T, idx)
        ib = profile_Ibg(g.detach(), d[idx], None if IB is None else IB[idx])
        newIB.append(ib)
        mu = ib[:, 0, None, None] * g + ib[:, 1, None, None]
        mu = mu.clamp_min(1e-3).double()
        dd = d[idx].double()
        l = (mu - dd * torch.log(mu)).sum()          # float64: L-BFGS line search needs the loss to ~1e-3
        if need_grad:
            l.backward()
        tot += float(l.detach())
    IB = torch.cat(newIB)
    return tot


opt = torch.optim.LBFGS([T[k] for k in free], max_iter=A.iters, max_eval=int(A.iters * 1.3), history_size=30,
                        line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
ncall = [0]
t0 = time.time()
L0 = [None]


def closure():
    opt.zero_grad()
    l = loss_and_grad(T)
    ncall[0] += 1
    if ncall[0] % 25 == 1:
        NA, px, py, lam, f0 = globals_(T)
        print(f'  eval {ncall[0]:4d}: nll {l:.6e}  NA {float(NA):.4f}  px {float(px) * 1000:.2f} py {float(py) * 1000:.2f} nm  '
              f'lam {float(lam) * 1000:.1f}  f0 {float(f0) * 1000:+.1f} nm  {time.time() - t0:.0f}s', flush=True)
    if L0[0] is None:
        L0[0] = l
    # float64 and offset: in float32 a loss of 5e7 has a resolution of 4 and L-BFGS stops on 'no change'
    return torch.tensor(l - L0[0], device=DEV, dtype=torch.float64)


# warm-up: Adam for the shape (L-BFGS from a crude start can stall), then L-BFGS
adam = torch.optim.Adam([T[k] for k in free], lr=A.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(adam, max(A.adam, 1), eta_min=A.lr * 0.01)
for it in range(A.adam):
    adam.zero_grad()
    l = loss_and_grad(T)
    if not math.isfinite(l) or any(not torch.isfinite(T[k].grad).all() for k in free):
        print(f'  non-finite loss/gradient at Adam step {it}: stop the warm-up here', flush=True)
        with torch.no_grad():
            for k in free:
                T[k].copy_(last[k])
        IB = None
        break
    last = {k: T[k].detach().clone() for k in free}
    adam.step(); sched.step()
    if it % 250 == 0:
        NA, px, py, lam, f0 = globals_(T)
        print(f'  adam {it}: nll {l:.6e}  NA {float(NA):.4f}  px {float(px) * 1000:.2f} py {float(py) * 1000:.2f}  '
              f'lam {float(lam) * 1000:.1f}  f0 {float(f0) * 1000:+.1f}  blur {float(T["ablur"][0].abs()) * 10:.1f} he {float(T["ahe"][0].abs()) * 10:.1f} nm  {time.time() - t0:.0f}s', flush=True)
opt.step(closure)
with torch.no_grad():
    nll_tot = loss_and_grad(T, need_grad=False)
    mu = torch.cat([IB[idx][:, 0, None, None] * model_images(T, idx) + IB[idx][:, 1, None, None] for idx in chunks])
    nll_spl = float((mu_spl.clamp_min(1e-3).double() - d.double() * torch.log(mu_spl.clamp_min(1e-3).double())).sum())
    c2 = ((d - mu) ** 2 / mu.clamp_min(1e-3)).flatten(1).mean(1)
    c2s = ((d - mu_spl) ** 2 / mu_spl.clamp_min(1e-3)).flatten(1).mean(1)
    per_bead = (torch.zeros(nb, device=DEV).index_add_(0, bid, c2) / torch.bincount(bid, minlength=nb)).cpu().numpy()
    per_bead_s = (torch.zeros(nb, device=DEV).index_add_(0, bid, c2s) / torch.bincount(bid, minlength=nb)).cpu().numpy()
    zb = np.round(zk.cpu().numpy() / 0.1) * 0.1
    c2n, c2sn = c2.cpu().numpy(), c2s.cpu().numpy()
    per_z = {f'{u:+.1f}': [float(c2n[zb == u].mean()), float(c2sn[zb == u].mean())] for u in np.unique(zb)}
    NA, px, py, lam, f0 = [float(v) for v in globals_(T)]
    dz = (0.01 * (T['adz'] - T['adz'].mean())).cpu().numpy()
    # photometry: fraction of the vectorial PSF inside the 23 x 23 patch at the bead focus
    res = dict(tag=A.tag, noll=list(noll), zcoef=T['zc'].detach().cpu().numpy(), NA=NA, pxy=[px, py], lam=lam, f0=f0,
               s=A.s, n_imm=1.518, n_med=1.33, apod=T['apod'].detach().cpu().numpy(), npupil=A.npupil, os=A.os,
               blur_sigma=float(0.01 * T['ablur'][0].abs()) if A.blur else 0.0,
               bead_height=fix['he'] if 'he' in fix else (float(0.01 * T['ahe'][0].abs()) if A.height else 0.0),
               zscale=float(1 + 0.01 * T['azs'][0]), poly=A.poly,
               maps=bool(A.maps),
               bead_xy=T['xy'].detach().cpu().numpy(), bead_dz=dz, fixed=fix, nll=nll_tot, nll_spline=nll_spl,
               n_pix=int(d.numel()), chi2_mean=float(c2.mean()), chi2_bead_median=float(np.median(per_bead)),
               chi2_bead=per_bead, chi2_spline_mean=float(c2s.mean()), chi2_spline_bead_median=float(np.median(per_bead_s)),
               chi2_per_z_vec_spline=per_z, n_eval=ncall[0], seconds=time.time() - t0, n_frames=N,
               n_params_global=sum(T[k].numel() for k in free if k not in ('xy', 'adz')))
print(f'done {A.tag}: nll {nll_tot:.6e} (spline {nll_spl:.6e}, diff {nll_tot - nll_spl:+.1f}); chi2/pixel mean {res["chi2_mean"]:.4f} '
      f'(spline {res["chi2_spline_mean"]:.4f}), per-bead median {res["chi2_bead_median"]:.4f} (spline '
      f'{res["chi2_spline_bead_median"]:.4f}); blur {res["blur_sigma"] * 1000:.1f} nm, height {res["bead_height"] * 1000:.1f} nm, zscale {res["zscale"]:.4f}; NA {NA:.4f} px {px * 1000:.2f} py {py * 1000:.2f} nm ratio {px / py:.4f} '
      f'lam {lam * 1000:.1f} f0 {f0 * 1000:+.1f}; {time.time() - t0:.0f}s', flush=True)
print('zcoef (waves):', {j: round(float(c), 4) for j, c in zip(noll, res['zcoef'])}, flush=True)
print('chi2 per z (vec, spline):', {k: [round(a, 3), round(b, 3)] for k, (a, b) in per_z.items()}, flush=True)
if A.maps:
    with torch.no_grad():
        ph = T['phase'][_mask]; ph = ph - _Q @ (_Q.T @ ph); la = T['la'][_mask]; la = la - la.mean()
        res['pupil_phase'] = torch.zeros_like(T['phase']).masked_scatter(_mask, ph).cpu().numpy()
        res['pupil_amp'] = torch.ones_like(T['la']).masked_scatter(_mask, torch.exp(la)).cpu().numpy()
        res['pupil_phase_rms'] = float(ph.std()); res['pupil_amp_rms'] = float(torch.exp(la).std())
jdump(res, f'{OUT}/vecbeads_{A.tag}.json')
np.save(f'{OUT}/vecbeads_{A.tag}_mu.npy', mu.cpu().numpy())
