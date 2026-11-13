"""PSF estimators of the in-situ study.

  fit_vec_stack     vectorial Zernike model fitted to the bead stack (per-emitter vectorial evaluation; 162 patches)
  fit_knots         free spline knots from bead stack and / or SMLM patches (+ a reference PSF as prior)
  fit_vec_insitu    vectorial Zernike model fitted to SMLM patches with the gradient collected on the spline knots
                    and passed through the vectorial model once per iteration (route K), or evaluated per emitter
                    with the vectorial model (route V, the conventional way)
"""
import math, time
import torch
import common as C
from torch_convspline.knotpsf import KnotPSF, localize, poisson_nll, crlb


def new_spline(S=2, hz=0.04):
    return KnotPSF(C.P, *C.ZR, S=S, hz=hz, zpad=5, device=C.DEV)       # zpad 5: knots cover the bead stack range


def knots_from(spl, zc, cond='insitu'):
    big = C.model_vec(spl.Pl, zc)
    return spl.stack_from_vectorial(big, C.insitu_pos if cond == 'insitu' else C.bead_pos)


# ------------------------------------------------------------------------------------------- vectorial, bead stack
def fit_vec_stack(stack, iters=300, lr=3e-3):
    d = stack['d']
    nb, nk = d.shape[:2]
    psf = C.model_vec(C.P)
    psf.zcoef.requires_grad_(True)
    xy = torch.zeros(nb, 2, device=C.DEV, requires_grad=True)
    bg0 = d.flatten(2).quantile(0.1, 2)
    logI = torch.log((d.flatten(2).sum(2) - bg0 * C.P ** 2).clamp_min(100)).requires_grad_(True)
    bg = bg0.clone().requires_grad_(True)
    f = (C.F0 - stack['z_nom'] / C.ALPHA)[None].expand(nb, -1).reshape(-1)
    opt = torch.optim.Adam([dict(params=[psf.zcoef], lr=lr), dict(params=[xy], lr=1e-2),
                            dict(params=[logI], lr=1e-2), dict(params=[bg], lr=0.3)])
    for it in range(iters):
        opt.zero_grad()
        pos = torch.cat([xy[:, None].expand(-1, nk, -1), torch.zeros(nb, nk, 1, device=C.DEV)], -1).reshape(-1, 3)
        mu = torch.exp(logI).reshape(-1, 1, 1) * psf(pos, f) + bg.reshape(-1, 1, 1)
        loss = poisson_nll(d.reshape(-1, C.P, C.P), mu).sum()
        loss.backward()
        opt.step()
    return psf.zcoef.detach().clone()


# ------------------------------------------------------------------------------------------- free knots
@torch.no_grad()
def bead_params(spl, stack, th=None):
    """Per bead frame (x, y, z, I, bg): z is the known nominal plane height, x and y are shared by the planes of a
    bead (they fix the lateral position of every z plane of the PSF), I and bg per frame."""
    d = stack['d']
    nb, nk = d.shape[:2]
    dd = d.reshape(-1, C.P, C.P)
    if th is None:
        bg = dd.flatten(1).quantile(0.1, 1)
        I = (dd.flatten(1).sum(1) - bg * C.P ** 2).clamp_min(100)
        z = stack['z_nom'][None].expand(nb, -1).reshape(-1)
        th = torch.stack([torch.zeros_like(I), torch.zeros_like(I), z, I, bg], 1)
    th, _ = localize(spl, dd, th, iters=10, fixed=(2,))
    inr = ((stack['z_nom'] > 0.1) & (stack['z_nom'] < 0.7)).float()[None, :, None]
    xy = (th[:, :2].reshape(nb, nk, 2) * inr).sum(1) / inr.sum(1)
    th[:, :2] = xy[:, None].expand(-1, nk, -1).reshape(-1, 2)
    th, nll = localize(spl, dd, th, iters=10, fixed=(0, 1, 2))
    return th, nll


def knot_loss(spl, K, d, th, nb=4000, backward=True, weight=1.0):
    """Poisson NLL of patches d at fixed emitter parameters th, as a function of the knots K (gradient accumulated
    into K.grad chunk by chunk)."""
    tot = 0.0
    for s in range(0, len(d), nb):
        t = th[s:s + nb]
        mu = t[:, 3, None, None] * spl.evaluate(t[:, :3], knots=K) + t[:, 4, None, None]
        l = weight * poisson_nll(d[s:s + nb], mu).sum()
        if backward:
            l.backward()
        tot += float(l)
    return tot


def fit_knots(spl, K0, insitu=None, stack=None, ref=None, lam_ref=0.0, lam_s=0.0, w_stack=1.0, outer=12, inner=8,
              th_ins=None, log=None):
    """Free-knot PSF. Alternates (1) emitter parameters at fixed knots (Newton per patch) and (2) the knots at fixed
    emitter parameters (L-BFGS; the model is linear in the knots, so this subproblem is convex).
    loss = NLL(SMLM patches) + w_stack NLL(bead stack) + lam_ref sum (K - ref)^2 / (ref + eps) + lam_s roughness."""
    K = K0.clone().requires_grad_(True)
    eps = 1e-4
    wref = None if ref is None else 1.0 / (ref.abs() + eps)
    wsm = 1.0 / (K0.abs() + eps)
    th_b = None
    t0 = time.time()
    for o in range(outer):
        spl.knots = K.detach()
        if insitu is not None:
            th_ins, nll_i = localize(spl, insitu, th_ins, iters=30 if th_ins is None else 5)
        if stack is not None:
            th_b, nll_b = bead_params(spl, stack, th_b)
        opt = torch.optim.LBFGS([K], max_iter=inner, history_size=10, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            tot = 0.0
            if insitu is not None:
                tot += knot_loss(spl, K, insitu, th_ins)
            if stack is not None:
                tot += knot_loss(spl, K, stack['d'].reshape(-1, C.P, C.P), th_b, weight=w_stack)
            reg = 0.0
            if ref is not None and lam_ref > 0:
                reg = reg + lam_ref * (wref * (K - ref) ** 2).sum()
            if lam_s > 0:
                dz = K[2:] - 2 * K[1:-1] + K[:-2]
                dy = K[:, 2:] - 2 * K[:, 1:-1] + K[:, :-2]
                dx = K[:, :, 2:] - 2 * K[:, :, 1:-1] + K[:, :, :-2]
                reg = reg + lam_s * ((wsm[1:-1] * dz ** 2).sum() + (wsm[:, 1:-1] * dy ** 2).sum()
                                     + (wsm[:, :, 1:-1] * dx ** 2).sum())
            if torch.is_tensor(reg):
                reg.backward()
                tot += float(reg)
            return torch.tensor(tot, device=K.device)

        l = opt.step(closure)
        if log is not None:
            log(o, float(l), time.time() - t0, K.detach())
    spl.knots = K.detach()
    return K.detach(), th_ins


# ------------------------------------------------------------------------------------------- vectorial, in situ
def knot_grad(spl, d, th, nb=4000):
    """d NLL / d knots at fixed emitter parameters: the adjoint of the spline evaluation, summed over all patches."""
    K = spl.knots.clone().requires_grad_(True)
    nll = knot_loss(spl, K, d, th, nb)
    return K.grad, nll


def vec_grad_direct(big_or_small, d, th, nb=250):
    """d NLL / d zcoef with the vectorial model evaluated for every emitter (route V)."""
    psf = big_or_small
    zc = psf.zcoef.detach().clone().requires_grad_(True)
    tot, g = 0.0, torch.zeros_like(zc)
    prm = {k: v.detach() for k, v in psf.named_parameters()}
    for s in range(0, len(d), nb):
        t = th[s:s + nb]
        prm['zcoef'] = zc
        g_ = torch.func.functional_call(psf, prm, (t[:, :3], torch.full((len(t),), C.F0, device=C.DEV)))
        l = poisson_nll(d[s:s + nb], t[:, 3, None, None] * g_ + t[:, 4, None, None]).sum()
        g += torch.autograd.grad(l, zc)[0]
        tot += float(l)
    return g, tot


def fit_vec_insitu(spl, d, zc0, iters=80, lr=2e-3, route='K', loc_iters=3, log=None, th=None, n_loc_v=None):
    """Vectorial Zernike model from SMLM patches. Per iteration: (1) emitter parameters (Newton on the spline built
    from the current Zernikes), (2) Zernike gradient: route K = knot gradient + one VJP through the knot build;
    route V = the vectorial model evaluated for every emitter. Adam step on the Zernikes."""
    big = C.model_vec(spl.Pl, zc0)
    small = C.model_vec(C.P, zc0)
    zc = zc0.clone().to(C.DEV)
    m, v = torch.zeros_like(zc), torch.zeros_like(zc)
    t0 = time.time()
    for it in range(iters):
        with torch.no_grad():
            big.zcoef.copy_(zc); small.zcoef.copy_(zc)
        spl.knots = spl.stack_from_vectorial(big, C.insitu_pos)
        th, nll = localize(spl, d, th, iters=30 if th is None else loc_iters)
        if route == 'K':
            gK, tot = knot_grad(spl, d, th)
            g = spl.vjp_vectorial(big, C.insitu_pos, gK)['zcoef']
        else:
            g, tot = vec_grad_direct(small, d, th)
        torch.cuda.synchronize()
        if log is not None:
            log(it, tot, time.time() - t0, zc.clone())
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        zc = zc - lr * (m / (1 - 0.9 ** (it + 1))) / ((v / (1 - 0.999 ** (it + 1))).sqrt() + 1e-12)
    return zc, th


def fit_pupil_insitu(spl, d, zc0, iters=200, lr_z=1e-3, lr_p=2e-3, lr_a=1e-2, lam=0.0, th=None, log=None):
    """Free pupil from SMLM patches through the knot route: the Zernikes (Noll 5-21) plus a pixel map of pupil phase
    (waves) and of log pupil amplitude. The physics of the pupil model still ties lateral shift, z scale and brightness
    to the PSF shape. Piston, tilt and defocus are projected out of the phase map (global shifts of all emitters) and
    the mean out of the log amplitude (global brightness). lam: roughness penalty on both maps."""
    from torch_convspline.vectorial import zernike
    big = C.model_vec(spl.Pl, zc0)
    mask = big.mask.bool()
    low = torch.stack([zernike(j, big.rho.clamp(max=1), big.phi)[mask] for j in (1, 2, 3, 4)], 1)     # [npix, 4]
    Q, _ = torch.linalg.qr(low)
    prm = dict(zcoef=zc0.clone().to(C.DEV), phase=torch.zeros_like(big.pupil_phase), la=torch.zeros_like(big.pupil_phase))
    lr = dict(zcoef=lr_z, phase=lr_p, la=lr_a)
    m = {k: torch.zeros_like(v) for k, v in prm.items()}; v2 = {k: torch.zeros_like(v) for k, v in prm.items()}

    def lap(x):
        return (4 * x[1:-1, 1:-1] - x[2:, 1:-1] - x[:-2, 1:-1] - x[1:-1, 2:] - x[1:-1, :-2]) * mask[1:-1, 1:-1]

    def set_model():
        with torch.no_grad():
            big.zcoef.copy_(prm['zcoef']); big.pupil_phase.copy_(prm['phase'] * mask); big.pupil_amp.copy_(torch.exp(prm['la']))

    t0 = time.time()
    for it in range(iters):
        set_model()
        spl.knots = spl.stack_from_vectorial(big, C.insitu_pos)
        th, nll = localize(spl, d, th, iters=30 if th is None else 3)
        gK, tot = knot_grad(spl, d, th)
        g = spl.vjp_vectorial(big, C.insitu_pos, gK, names=('zcoef', 'pupil_phase', 'pupil_amp'))
        gr = dict(zcoef=g['zcoef'], phase=g['pupil_phase'] * mask, la=g['pupil_amp'] * big.pupil_amp * mask)
        if lam > 0:
            for k in ('phase', 'la'):
                x = prm[k].clone().requires_grad_(True)
                gr[k] = gr[k] + torch.autograd.grad(lam * (lap(x) ** 2).sum(), x)[0]
        if log is not None:
            log(it, tot, time.time() - t0, prm)
        for k in prm:
            m[k] = 0.9 * m[k] + 0.1 * gr[k]; v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
            prm[k] = prm[k] - lr[k] * (m[k] / (1 - 0.9 ** (it + 1))) / ((v2[k] / (1 - 0.999 ** (it + 1))).sqrt() + 1e-12)
        ph = prm['phase'][mask]
        prm['phase'][mask] = ph - Q @ (Q.T @ ph)
        prm['la'][mask] = prm['la'][mask] - prm['la'][mask].mean()
        prm['phase'] = prm['phase'] * mask; prm['la'] = prm['la'] * mask
    set_model()
    return spl.stack_from_vectorial(big, C.insitu_pos), prm, th


# ------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(spl, knots, test, nll_true=None):
    """Localize the test emitters with the PSF `knots`; errors against the truth and the likelihood deficit."""
    spl.knots = knots
    th, nll = localize(spl, test['d'])
    e = th - test['th']
    z = test['th'][:, 2]
    ok = (e[:, 2].abs() < 0.15) & (e[:, :2].abs().max(1).values < 0.5)       # gross outliers (wrong z branch)
    # z error after the best straight line (offset and scale of the z axis are reported separately)
    A = torch.stack([torch.ones_like(z), z], 1)[ok]
    co = torch.linalg.lstsq(A.double(), th[ok, 2:3].double()).solution[:, 0]
    zres = th[:, 2] - (co[0] + co[1] * z)
    out = dict(outlier=float(1 - ok.float().mean()),
               x_rmse=float(e[ok, 0].pow(2).mean().sqrt()) * 108, y_rmse=float(e[ok, 1].pow(2).mean().sqrt()) * 108,
               z_rmse=float(e[ok, 2].pow(2).mean().sqrt()) * 1000, z_rmse_lin=float(zres[ok].pow(2).mean().sqrt()) * 1000,
               z_scale=float(co[1]), z_off=float(co[0]) * 1000,
               x_bias=float(e[ok, 0].mean()) * 108, y_bias=float(e[ok, 1].mean()) * 108,
               I_ratio=float((th[ok, 3] / test['th'][ok, 3]).median()), th=th, ok=ok)
    if nll_true is not None:
        out['dnll'] = float((nll - nll_true).mean())
    out['nll'] = nll
    return out


def psf_rel_err(spl, knots, knots_true):
    """Relative L2 difference of the PSF planes within the emitter z range."""
    zs = spl.zs
    m = (zs >= C.ZR[0]) & (zs <= C.ZR[1])
    return float((knots[m] - knots_true[m]).norm() / knots_true[m].norm())
