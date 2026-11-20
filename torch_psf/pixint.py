"""Differentiable track fitting through a spline sampled from the optical model."""
import torch
from torch_psf.integrate import spectral_min_oversample
from torch_psf.spline import to_spline

def fit_tracks_spline(tracks, vectorial, readnoise_e=1.0, zmax_um=0.2, iters=600, warmup=100, focus_um=0.0,
                      blur='none', groups=None, ngroups=1, group_psf_kwargs=None, extra_params=(), batch=4000,
                      lr_xyz=0.03, lr_focus=0.005, lr_zernike=0.002, lr_blur=0.003, lr_extra=0.003, fix_zernike=False,
                      fix_focus=False, log=None, S=3, dz_um=0.01, xy_limit=None, final_exact=True, init_xy=None, init_z=None,
                      init_blur=None, lr_final=1.0):
    """estimation.fit_tracks with the samples rendered by the integrated cubic B-spline (SplinePSF, S knots / px,
    dz_um, point-density knots of the exact vectorial model, max error ~3e-5 of the peak at S = 3) instead of the
    vectorial model: per iteration the knots are built once per PSF group from the current Zernikes / focus / blur /
    group terms (SplinePSF.sample, with autograd), every chunk of samples is rendered by the spline with the knots detached
    (gradients to x, y, z and to the knots), then the knot gradient is pulled back to the PSF parameters (one backward
    through the knot build). Same arguments and result as fit_tracks (the model: the exact pixel integral of the
    vectorial PSF, psf.pixel_integration = 'exact'); xy_limit: the spline's offset limit (px; default R / 2 - 0.5: a
    sample that runs off to a neighbour must render as in fit_tracks, a clamp at 2 px left ~2 % of the real-data
    samples stuck and cost 1e-3 nats / px). final_exact: the
    final NLL / residuals with the exact vectorial model (else with the spline).
    Warm start / refinement (2026-11-19): init_xy [B, 2] (px from the ROI centre) / init_z [B] (um, clamped into
    (0, zmax_um)) start every sample there instead of at xy_init / zmax_um / 2 (e.g. from a multi-start refit under the
    starting PSF, estimation.heldout_tracks); init_blur [sxx, syy, sxy] (px^2) starts the 'line' / 'cov' blur there (its
    Cholesky factor; default 0.05 px); lr_final: the learning rates of all parameters decay from their value at the end of
    the warmup to lr_final x that value at the last iteration (cosine; 1 = constant, as before). As in fit_tracks, the
    gradient goes through the closed-form I / bg (fit.solve_linear_grad): it is the Poisson-weighted IRLS estimate, not
    the minimiser of the Gaussian NLL that is evaluated, so holding I / bg fixed would bias the fit."""
    import math, time
    from torch_psf.fit import solve_linear_grad, readnoise_variance
    dev = tracks.rois.device
    B, T, R = tracks.rois.shape[0], tracks.rois.shape[1], tracks.rois.shape[-1]
    c = (R - 1) / 2
    xy_limit = R / 2 - 0.5 if xy_limit is None else xy_limit
    with torch.no_grad():
        spl = to_spline(vectorial, (0.0, zmax_um), roisize=R, oversample=S, dz_um=dz_um, xy_limit=xy_limit,
                                 kernel='bspline', value='density')
    knot_os = spectral_min_oversample(vectorial.band_limit())
    big = vectorial.resampled(spl.Pl, oversample=knot_os, pixel_integration='point')
    big.zcoef.requires_grad_(True)
    focus = torch.tensor(float(focus_um), device=dev, requires_grad=True)
    rn2, pix = readnoise_variance(readnoise_e, B, R, dev)
    xy = ((tracks.xy_init.float() - c) if init_xy is None else torch.as_tensor(init_xy, device=dev).float()).clone().requires_grad_(True)
    if init_z is None:
        zr = torch.zeros(B, device=dev, requires_grad=True)
    else:
        zf = (torch.as_tensor(init_z, device=dev).float() / zmax_um).clamp(1e-4, 1 - 1e-4)
        zr = torch.log(zf / (1 - zf)).clone().requires_grad_(True)
    BL = None
    if blur in ('line', 'cov') and init_blur is not None:
        sxx, syy, sxy = [float(v) for v in init_blur]
        a_ = math.sqrt(max(sxx, 1e-6)); b_ = sxy / a_
        BL = torch.tensor([a_, b_] if blur == 'line' else [a_, b_, math.sqrt(max(syy - b_ * b_, 1e-6))], device=dev, requires_grad=True)
    elif blur == 'line':
        BL = torch.tensor([0.05, 0.0], device=dev, requires_grad=True)
    elif blur == 'cov':
        BL = torch.tensor([0.05, 0.0, 0.05], device=dev, requires_grad=True)
    elif blur != 'none':
        raise ValueError(f"blur: 'none' | 'line' | 'cov', not {blur!r}")
    grp = torch.zeros(B, T, dtype=torch.long, device=dev) if groups is None else groups.to(dev)
    mask = tracks.mask
    extra = list(extra_params)
    opt = torch.optim.Adam([{'params': [xy, zr], 'lr': lr_xyz}, {'params': [focus], 'lr': lr_focus},
                            {'params': [big.zcoef], 'lr': lr_zernike}]
                           + ([{'params': [BL], 'lr': lr_blur}] if BL is not None else [])
                           + ([{'params': extra, 'lr': lr_extra}] if extra else []))
    lr0 = [g['lr'] for g in opt.param_groups]
    decay = lambda it: 1.0 if (lr_final == 1.0 or it < warmup) else \
        lr_final + (1 - lr_final) * 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(iters - 1 - warmup, 1)))
    npx = float(mask.sum()) * R * R
    ng = ngroups if group_psf_kwargs is not None else 1

    def blur_cov():
        if BL is None: return None
        if len(BL) == 2: return torch.stack([BL[0] ** 2, BL[1] ** 2, BL[0] * BL[1]])
        return torch.stack([BL[0] ** 2, BL[1] ** 2 + BL[2] ** 2, BL[0] * BL[1]])

    def group_kw(g):
        kw = dict(group_psf_kwargs(g)) if group_psf_kwargs is not None else {}
        if BL is not None: kw.setdefault('blur', blur_cov())
        return kw

    def knots():
        return [spl.sample(big, focus=focus, chunk=spl.nz, differentiable=True, **group_kw(g)) for g in range(ng)]

    def loss(sl, K, render):
        data = tracks.rois[sl]
        b = data.shape[0]
        g_ = grp[sl]
        r2 = rn2[sl] if pix else rn2
        pos = torch.cat([xy[sl], zmax_um * torch.sigmoid(zr[sl])[:, None]], 1)
        psf = torch.stack([render(pos, g, K) for g in range(ng)], 1)                       # [b, ng, R, R]
        psf = torch.gather(psf, 1, (g_ if ng > 1 else torch.zeros_like(g_))[..., None, None].expand(-1, -1, R, R))
        I, bg = solve_linear_grad(psf.reshape(b, T, -1), data.reshape(b, T, -1), mask[sl], g_, r2, ngroups)
        mu = (I[..., None, None] * psf + torch.gather(bg, 1, g_)[..., None, None]).clamp(min=1e-3)
        var = mu + (r2.reshape(b, 1, R, R) if pix else r2)
        nll = 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
        nll = torch.where(mask[sl][..., None, None], nll, torch.zeros_like(nll)).sum()
        return nll, I, bg, mu, var, data

    spl_render = lambda pos, g, K: spl.render_tables(pos, knots=K[g])
    chunks = torch.arange(B, device=dev).split(batch)
    hist, t0, tot = [], time.time(), float('nan')
    params = [focus, big.zcoef] + ([BL] if BL is not None else []) + extra
    for it in range(iters):
        for g_, l0 in zip(opt.param_groups, lr0):
            g_['lr'] = l0 * decay(it)
        opt.zero_grad()
        train_psf = it >= warmup
        with torch.set_grad_enabled(train_psf):
            Kg = knots()
        Kd = [k.detach().requires_grad_(train_psf) for k in Kg]
        tot = 0.0
        for ch in chunks:
            l = loss(ch, Kd, spl_render)[0] / npx
            l.backward()
            tot += float(l.detach())
        if train_psf:
            torch.autograd.backward(Kg, [k.grad for k in Kd])
        if fix_zernike:
            big.zcoef.grad = None
        if fix_focus:
            focus.grad = None
        opt.step()
        hist.append(dict(it=it, nll_per_px=tot, focus_um=float(focus.detach())))
        if log is not None:
            log(it, tot, time.time() - t0, big.zcoef.detach())
    out = vectorial.resampled(R, oversample=max(vectorial.os, knot_os), pixel_integration='exact')
    with torch.no_grad():
        out.zcoef.copy_(big.zcoef)
    out.focus_um = float(focus.detach())
    if BL is not None:
        out.blur = blur_cov().detach()
    with torch.no_grad():
        if final_exact:
            def ex_render(pos, g, K):
                kw = group_kw(g)
                return out(pos, focus.expand(pos.shape[0]), **kw)
            rend = ex_render
        else:
            rend = spl_render
        K = [k.detach() for k in knots()]
        Is, bgs, rs, final = [], [], [], 0.0
        for ch in chunks:
            l_, I, bg, mu, var, data = loss(ch, K, rend)
            final += float(l_) / npx
            Is.append(I); bgs.append(bg)
            rs.append(((data - mu) / var.sqrt())[mask[ch]])
        rs = torch.cat(rs)
    rep = dict(method='tracks_spline', n_samples=B, iterations=len(hist), resid_var=float(rs.var()), nll_per_px=final,
               nll_per_px_before_last_step=tot, focus_um=float(focus.detach()), seconds=time.time() - t0,
               spline=dict(S=S, dz_um=dz_um, xy_limit=xy_limit, kernel='bspline', integrate=True),
               at_limit=float((xy.detach().abs().max(1).values >= spl.xmax - 1e-3).float().mean()))
    return dict(psf=out, report=rep, history=hist, xy=xy.detach(), z=(zmax_um * torch.sigmoid(zr)).detach(),
                I=torch.cat(Is), bg=torch.cat(bgs), extra=[e.detach() for e in extra])
