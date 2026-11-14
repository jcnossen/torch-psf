"""In-situ PSF estimation on the spline knots (the "knot route").

All emitters are evaluated on a ConvSplinePSF; the Poisson likelihood gradient is accumulated on the knots (autograd
through the spline) and, for a vectorial model, passed to its parameters with one vector-Jacobian product through
the knot build (ConvSplinePSF.vjp_vectorial). The cost of the vectorial model is then independent of the number of
emitters.

  fit_zernike_knots   Zernike coefficients of a VectorialPSF from SMLM ROIs (route 'K'; route 'V' evaluates the
                      vectorial model for every emitter, the conventional way, for comparison)
  fit_pupil_knots     Zernikes plus a free pupil phase map (waves) and log-amplitude map
  fit_free_knots      free knots (model-free PSF) from SMLM ROIs and / or a bead stack, with prior / smoothness terms

Inner step: per-emitter theta = (x, y, z, photons, bg) by Levenberg-Marquardt on the current spline
(torch_psf.fit.localize). Every result (PSFEstimate) reports the fraction of emitters at the lateral offset limit of
the spline (at_limit; should be ~0, raise xy_limit otherwise) and the z scale relative to the starting PSF.
"""
import time
import torch
from torch_psf.fit import localize, poisson_nll, at_limit
from torch_psf.zernike import zernike
from torch_psf.spline import ConvSplinePSF, emitter_height_pos_fn
from torch_psf.estimate import PSFEstimate, z_scale, FREE_KNOT_WARNING


def limit_fraction(spline, theta):
    """Fraction of emitters whose fitted x or y sits at the spline's offset limit."""
    return float(at_limit(theta, spline).float().mean())


def _copy(spline, knots):
    s = ConvSplinePSF(spline.roisize, spline.z_range, oversample=spline.S, dz_um=spline.hz, zpad=spline.zpad,
                      xy_limit=spline.xmax, device=spline.dev, dtype=spline.dtype)
    s.knots = knots.detach().clone()
    return s


def _zscale_report(spline, knots_ref, knots_new):
    try:
        sc, off = z_scale(_copy(spline, knots_new), _copy(spline, knots_ref))
    except Exception:          # e.g. a degenerate PSF; the report must not fail
        sc, off = float('nan'), float('nan')
    return dict(z_scale=sc, z_offset_um=off)


def knot_loss(spline, knots, rois, theta, batch=4000, backward=True, weight=1.0, var=None):
    """Poisson NLL of rois [N, P, P] at fixed emitter parameters theta [N, 5] as a function of the knots (a tensor
    that requires grad; the gradient is accumulated into knots.grad batch by batch when backward). Returns the total
    NLL (float)."""
    tot = 0.0
    for s in range(0, len(rois), batch):
        t = theta[s:s + batch]
        mu = t[:, 3, None, None] * spline.render(t[:, :3], knots=knots) + t[:, 4, None, None]
        v = var[s:s + batch] if (torch.is_tensor(var) and var.dim() > 0) else var
        l = weight * poisson_nll(rois[s:s + batch], mu, v).sum()
        if backward:
            l.backward()
        tot += float(l.detach())
    return tot


def knot_gradient(spline, rois, theta, batch=4000, var=None):
    """d NLL / d knots at fixed emitter parameters (the adjoint of the spline evaluation summed over all ROIs).
    Returns (gradient [nz, nv, nu], NLL)."""
    K = spline.knots.clone().requires_grad_(True)
    nll = knot_loss(spline, K, rois, theta, batch, var=var)
    return K.grad, nll


def vectorial_gradient(vectorial, rois, theta, focus_um, batch=1000, names=('zcoef',)):
    """d NLL / d parameters with the vectorial model (roisize = ROI size) evaluated for every emitter (route V).
    Returns ({name: gradient}, NLL)."""
    prm = {k: v.detach() for k, v in vectorial.named_parameters()}
    leaf = {k: prm[k].clone().requires_grad_(True) for k in names}
    tot, g = 0.0, {k: torch.zeros_like(prm[k]) for k in names}
    for s in range(0, len(rois), batch):
        t = theta[s:s + batch]
        prm.update(leaf)
        g_ = torch.func.functional_call(vectorial, prm,
                                        (t[:, :3], torch.full((len(t),), float(focus_um), device=t.device)))
        l = poisson_nll(rois[s:s + batch], t[:, 3, None, None] * g_ + t[:, 4, None, None]).sum()
        for k, gk in zip(names, torch.autograd.grad(l, [leaf[k] for k in names])):
            g[k] += gk
        tot += float(l)
    return g, tot


def _adam(m, v, g, it):
    """Adam moments (beta 0.9, 0.999) and the bias-corrected step direction."""
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    return m, v, (m / (1 - 0.9 ** (it + 1))) / ((v / (1 - 0.999 ** (it + 1))).sqrt() + 1e-12)


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def fit_zernike_knots(spline, rois, vectorial, pos_fn=None, iters=80, lr=2e-3, route='K', loc_iters=3, log=None,
                      theta=None, localizer=None, stop=None):
    """Zernike coefficients of `vectorial` (a VectorialPSF of any roisize; copied to the sizes needed) from SMLM ROIs
    [N, P, P] (photons; P = spline.roisize). The spline (ConvSplinePSF) defines the knot grid; its knots are
    overwritten. Per iteration:
      (1) emitter parameters by Levenberg-Marquardt on the spline built from the current Zernikes (30 iterations the
          first time, loc_iters afterwards; or localizer(spline, rois, theta) -> (theta, nll)),
      (2) the Zernike gradient: route 'K' = knot gradient + one VJP through the knot build; route 'V' = the
          vectorial model for every emitter,
      (3) an Adam step (lr, waves) on the Zernikes.
    pos_fn: knot z -> (emitter z, focus) (default: emitter height at vectorial.focus_um).
    log(it, nll, seconds, zcoef): progress callback; stop(): early stop ("use what we have").
    Returns PSFEstimate: psf = copy of vectorial with the fitted Zernikes; data = dict(zcoef, theta (of the last
    localization), knots (the spline of the last localization step))."""
    spl, d, vec = spline, rois, vectorial
    pos_fn = emitter_height_pos_fn(vec.focus_um) if pos_fn is None else pos_fn
    big = vec.resized(spl.Pl)
    small = vec.resized(spl.roisize) if route == 'V' else None
    zc = vec.zcoef.detach().clone().to(spl.dev)
    m, v = torch.zeros_like(zc), torch.zeros_like(zc)
    hist = []
    th = theta
    K0 = None
    t0 = time.time()
    for it in range(iters):
        with torch.no_grad():
            big.zcoef.copy_(zc)
            if small is not None:
                small.zcoef.copy_(zc)
        spl.knots = spl.knots_from_vectorial(big, pos_fn, chunk=spl.nz)
        if K0 is None:
            K0 = spl.knots
        th, nll = (localize(spl, d, th, iters=30 if th is None else loc_iters) if localizer is None
                   else localizer(spl, d, th))
        if route == 'K':
            gK, tot = knot_gradient(spl, d, th)
            g = spl.vjp_vectorial(big, pos_fn, gK, chunk=spl.nz)['zcoef']      # one build + one backward
        else:
            gd, tot = vectorial_gradient(small, d, th, focus_um=float(pos_fn(torch.zeros(1, device=spl.dev))[1][0]))
            g = gd['zcoef']
        _sync(spl.dev)
        hist.append(dict(it=it, nll=tot, seconds=time.time() - t0))
        if log is not None:
            log(it, tot, time.time() - t0, zc.clone())
        m, v, step = _adam(m, v, g, it)
        zc = zc - lr * step
        if stop is not None and stop():
            break
    sec = time.time() - t0
    out = vec.resized(vec.roisize)
    with torch.no_grad():
        out.zcoef.copy_(zc)
        big.zcoef.copy_(zc)
    last_knots = spl.knots
    rep = dict(method='zernike_knots', route=route, iterations=len(hist), n_emitters=len(d), seconds=sec,
               nll=hist[-1]['nll'] if hist else float('nan'), at_limit=limit_fraction(spl, th))
    if K0 is not None:
        rep.update(_zscale_report(spl, K0, spl.knots_from_vectorial(big, pos_fn, chunk=spl.nz)))
    spl.knots = last_knots
    return PSFEstimate(out, rep, hist, dict(zcoef=zc, theta=th, knots=last_knots))


def fit_pupil_knots(spline, rois, vectorial, pos_fn=None, iters=200, lr_z=1e-3, lr_p=2e-3, lr_a=1e-2, lam=0.0,
                    theta=None, log=None, localizer=None, stop=None):
    """Free pupil from SMLM ROIs through the knot route: the Zernikes of `vectorial` plus a pixel map of pupil phase
    (waves) and of log pupil amplitude. Piston, tilt and defocus are projected out of the phase map (global shifts of
    all emitters), the mean out of the log amplitude (global brightness). lam: Laplacian roughness penalty on both
    maps; lr_z / lr_p / lr_a: Adam step sizes of the Zernikes, phase and log amplitude. Other arguments as
    fit_zernike_knots. Returns PSFEstimate: psf = copy of vectorial with the fitted Zernikes and maps; data =
    dict(zcoef, phase, la, theta, knots (built from the final model))."""
    spl, d, vec = spline, rois, vectorial
    pos_fn = emitter_height_pos_fn(vec.focus_um) if pos_fn is None else pos_fn
    big = vec.resized(spl.Pl)
    mask = big.mask.bool()
    low = torch.stack([zernike(j, big.rho.clamp(max=1), big.phi)[mask] for j in (1, 2, 3, 4)], 1)     # [npix, 4]
    Q, _ = torch.linalg.qr(low)
    prm = dict(zcoef=vec.zcoef.detach().clone().to(spl.dev), phase=torch.zeros_like(big.pupil_phase),
               la=torch.zeros_like(big.pupil_phase))
    lr = dict(zcoef=lr_z, phase=lr_p, la=lr_a)
    m = {k: torch.zeros_like(v) for k, v in prm.items()}; v2 = {k: torch.zeros_like(v) for k, v in prm.items()}

    def lap(x):
        return (4 * x[1:-1, 1:-1] - x[2:, 1:-1] - x[:-2, 1:-1] - x[1:-1, 2:] - x[1:-1, :-2]) * mask[1:-1, 1:-1]

    def set_model():
        with torch.no_grad():
            big.zcoef.copy_(prm['zcoef']); big.pupil_phase.copy_(prm['phase'] * mask)
            big.pupil_amp.copy_(torch.exp(prm['la']))

    th = theta
    hist = []
    K0 = None
    t0 = time.time()
    for it in range(iters):
        set_model()
        spl.knots = spl.knots_from_vectorial(big, pos_fn, chunk=spl.nz)
        if K0 is None:
            K0 = spl.knots
        th, nll = localize(spl, d, th, iters=30 if th is None else 3) if localizer is None else localizer(spl, d, th)
        gK, tot = knot_gradient(spl, d, th)
        g = spl.vjp_vectorial(big, pos_fn, gK, names=('zcoef', 'pupil_phase', 'pupil_amp'), chunk=spl.nz)
        gr = dict(zcoef=g['zcoef'], phase=g['pupil_phase'] * mask, la=g['pupil_amp'] * big.pupil_amp * mask)
        if lam > 0:
            for k in ('phase', 'la'):
                x = prm[k].clone().requires_grad_(True)
                gr[k] = gr[k] + torch.autograd.grad(lam * (lap(x) ** 2).sum(), x)[0]
        hist.append(dict(it=it, nll=tot, seconds=time.time() - t0))
        if log is not None:
            log(it, tot, time.time() - t0, prm)
        for k in prm:
            m[k] = 0.9 * m[k] + 0.1 * gr[k]; v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
            prm[k] = prm[k] - lr[k] * (m[k] / (1 - 0.9 ** (it + 1))) / ((v2[k] / (1 - 0.999 ** (it + 1))).sqrt() + 1e-12)
        ph = prm['phase'][mask]
        prm['phase'][mask] = ph - Q @ (Q.T @ ph)
        prm['la'][mask] = prm['la'][mask] - prm['la'][mask].mean()
        prm['phase'] = prm['phase'] * mask; prm['la'] = prm['la'] * mask
        if stop is not None and stop():
            break
    sec = time.time() - t0
    set_model()
    K = spl.knots_from_vectorial(big, pos_fn)
    out = big.resized(vec.roisize)
    rep = dict(method='pupil_knots', iterations=len(hist), n_emitters=len(d), seconds=sec,
               nll=hist[-1]['nll'] if hist else float('nan'), at_limit=limit_fraction(spl, th))
    if K0 is not None:
        rep.update(_zscale_report(spl, K0, K))
    return PSFEstimate(out, rep, hist, dict(prm, theta=th, knots=K))


@torch.no_grad()
def bead_theta(spline, stack, z_nom, theta=None, xy_zrange=None):
    """Per bead plane theta = (x, y, z, photons, bg) for bead stacks [nbeads, nk, P, P] (photons) at the nominal knot
    heights z_nom [nk] (um): z is held at z_nom; x, y are shared by the planes of a bead (the mean over the planes
    with z_nom inside xy_zrange, default the middle 75 % of the range); photons and bg per plane.
    Returns theta [nbeads nk, 5], nll [nbeads nk]."""
    spl, d, th = spline, stack, theta
    nb, nk, P = d.shape[0], d.shape[1], d.shape[-1]
    dd = d.reshape(-1, P, P)
    if th is None:
        bg = dd.flatten(1).quantile(0.1, 1)
        I = (dd.flatten(1).sum(1) - bg * P ** 2).clamp_min(100)
        z = z_nom[None].expand(nb, -1).reshape(-1)
        th = torch.stack([torch.zeros_like(I), torch.zeros_like(I), z, I, bg], 1)
    th, _ = localize(spl, dd, th, iters=10, fixed=(2,))
    if xy_zrange is None:
        lo, hi = float(z_nom.min()), float(z_nom.max())
        xy_zrange = (lo + 0.125 * (hi - lo), hi - 0.125 * (hi - lo))
    inr = ((z_nom > xy_zrange[0]) & (z_nom < xy_zrange[1])).float()[None, :, None]
    xy = (th[:, :2].reshape(nb, nk, 2) * inr).sum(1) / inr.sum(1)
    th[:, :2] = xy[:, None].expand(-1, nk, -1).reshape(-1, 2)
    return localize(spl, dd, th, iters=10, fixed=(0, 1, 2))


def fit_free_knots(spline, knots0, rois=None, stack=None, ref=None, lam_ref=0.0, lam_s=0.0, w_stack=1.0, outer=12,
                   inner=8, theta=None, log=None, localizer=None, xy_zrange=None, stop=None):
    """Free-knot PSF. Alternates (1) the emitter parameters at fixed knots (Levenberg-Marquardt per ROI) and (2) the
    knots at fixed emitter parameters (L-BFGS, `inner` iterations; the model is linear in the knots, so this
    subproblem is convex), `outer` times.
    spline: ConvSplinePSF (knot grid; its knots are overwritten); knots0: starting knots [nz, nv, nu];
    rois: SMLM ROIs [N, P, P] (photons); stack: dict(d=[nbeads, nk, P, P], z_nom=[nk]) bead stack with known plane
    heights; ref: reference knots for the prior.
    loss = NLL(SMLM ROIs) + w_stack NLL(bead stack) + lam_ref sum (K - ref)^2 / (|ref| + eps)
           + lam_s sum of squared second differences / (|knots0| + eps).
    Returns PSFEstimate: psf = a ConvSplinePSF with the fitted knots; data = dict(knots, theta (of the SMLM ROIs));
    report includes z_scale against knots0 and a warning (free knots can warp z)."""
    spl, insitu, K0, th_ins = spline, rois, knots0, theta
    K = K0.clone().requires_grad_(True)
    eps = 1e-4
    wref = None if ref is None else 1.0 / (ref.abs() + eps)
    wsm = 1.0 / (K0.abs() + eps)
    th_b = None
    hist = []
    t0 = time.time()
    P = spl.roisize
    for o in range(outer):
        spl.knots = K.detach()
        if insitu is not None:
            th_ins, nll_i = (localize(spl, insitu, th_ins, iters=30 if th_ins is None else 5) if localizer is None
                             else localizer(spl, insitu, th_ins))
        if stack is not None:
            th_b, nll_b = bead_theta(spl, stack['d'], stack['z_nom'], th_b, xy_zrange)
        opt = torch.optim.LBFGS([K], max_iter=inner, history_size=10, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            tot = 0.0
            if insitu is not None:
                tot += knot_loss(spl, K, insitu, th_ins)
            if stack is not None:
                tot += knot_loss(spl, K, stack['d'].reshape(-1, P, P), th_b, weight=w_stack)
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
                tot += float(reg.detach())
            return torch.tensor(tot, device=K.device)

        l = opt.step(closure)
        hist.append(dict(it=o, loss=float(l), seconds=time.time() - t0))
        if log is not None:
            log(o, float(l), time.time() - t0, K.detach())
        if stop is not None and stop():
            break
    sec = time.time() - t0
    spl.knots = K.detach()
    rep = dict(method='free_knots', iterations=len(hist), seconds=sec,
               n_emitters=0 if insitu is None else len(insitu),
               at_limit=limit_fraction(spl, th_ins) if th_ins is not None else float('nan'),
               warnings=[FREE_KNOT_WARNING])
    rep.update(_zscale_report(spl, K0, K.detach()))
    return PSFEstimate(_copy(spl, K.detach()), rep, hist, dict(knots=K.detach(), theta=th_ins))
