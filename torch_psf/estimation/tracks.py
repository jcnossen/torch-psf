"""In-situ vectorial PSF fit on emitter tracks with variable projection.

Each sample is one emitter seen on up to T consecutive frames (torch_psf.spots.Tracks). Per frame
the intensity, per (sample, group) the background are solved in closed form (Poisson-weighted IRLS,
torch_psf.fit.solve_linear_grad) at every iteration, and the gradient goes through that solve (it is not the exact
minimiser of the NLL below, so holding I / bg fixed would not give the gradient of the profiled NLL); the per-sample position (x, y, and z through a sigmoid on
[0, zmax_um]) and the global PSF parameters (Zernikes, focus, optional line blur, optional extra parameters of a
per-group PSF hook) are fitted with Adam on the Gaussian approximation of the Poisson + read-noise likelihood. The
first `warmup` iterations move only the per-sample positions.

Per-group PSF hook (e.g. photoselection by a polarised excitation that differs per illumination-pattern angle):
group_psf_kwargs(g) -> dict of extra VectorialPSF.forward arguments (e.g. dict(dipM=..., blur=...)) for frames of
group g; their trainable tensors are passed in extra_params (fitted with lr_extra)."""
import math
import time
import torch


def _result(psf, report, history, data):
    """Result of the low-level estimators: dict(psf, report, history, **data)."""
    return dict(psf=psf, report=report, history=history, **data)
from torch_psf.fit import solve_linear_grad, readnoise_variance


def fit_tracks(tracks, vectorial, readnoise_e=1.0, zmax_um=0.2, iters=600, warmup=100, focus_um=0.0, blur='none',
               groups=None, ngroups=1, group_psf_kwargs=None, extra_params=(), batch=2000, lr_xyz=0.03,
               lr_focus=0.005, lr_zernike=0.002, lr_blur=0.003, lr_extra=0.003, fix_zernike=False, fix_focus=False,
               log=None, stop=None, pixel_integration=None):
    """tracks: Tracks (rois [B, T, R, R] in photons, mask, xy_init). vectorial: starting VectorialPSF (copied;
    roisize = R). readnoise_e: read noise (e- rms): a scalar, or per pixel of every sample [B, R, R] (an sCMOS
    read-noise map cut at each sample's crop; the variance model is then mu + rn^2 per pixel). zmax_um: emitter heights 0..zmax_um. focus_um: starting
    focus (fitted). blur: 'none', 'line' (image-plane line blur v v^T, px^2) or 'cov' (a general 2 x 2 Gaussian blur
    covariance L L^T, px^2; the line blur plus an isotropic / second-axis part). groups [B, T] (long, values
    0..ngroups-1): background groups and PSF groups (default one group). group_psf_kwargs / extra_params: per-group
    PSF hook (module docstring). fix_zernike / fix_focus: hold the Zernikes / the focus at their starting values.
    pixel_integration: the pixel integration of the fitted model (VectorialPSF.pixel_integration: 'midpoint' = the midpoint
    rule with the model's oversample, 'exact' = the exact pixel integral, 'gauss_legendre'); None = that
    of `vectorial`. The returned psf keeps it.
    Returns dict(psf, report, history, ...): psf = fitted VectorialPSF (focus_um = fitted focus, blur set if fitted); report: resid_var
    (variance of the normalized residuals, 1 for a perfect model), nll_per_px, focus_um; data: xy [B, 2] px rel. to
    the ROI centre, z [B] um, I [B, T] photons, bg [B, ngroups]."""
    dev = tracks.rois.device
    B, T, R = tracks.rois.shape[0], tracks.rois.shape[1], tracks.rois.shape[-1]
    c = (R - 1) / 2
    model = vectorial.resampled(R, pixel_integration=pixel_integration)
    model.zcoef.requires_grad_(True)
    focus = torch.tensor(float(focus_um), device=dev, requires_grad=True)
    rn2, pix = readnoise_variance(readnoise_e, B, R, dev)
    xy = (tracks.xy_init.float() - c).clone().requires_grad_(True)
    zr = torch.zeros(B, device=dev, requires_grad=True)
    BL = None
    if blur == 'line':
        BL = torch.tensor([0.05, 0.0], device=dev, requires_grad=True)   # start off zero (the gradient of v v^T vanishes there)
    elif blur == 'cov':
        BL = torch.tensor([0.05, 0.0, 0.05], device=dev, requires_grad=True)   # Cholesky factor (a, b, c) of the covariance
    elif blur != 'none':
        raise ValueError(f"blur: 'none' | 'line' | 'cov', not {blur!r}")
    grp = torch.zeros(B, T, dtype=torch.long, device=dev) if groups is None else groups.to(dev)
    mask = tracks.mask
    extra = list(extra_params)
    opt = torch.optim.Adam([{'params': [xy, zr], 'lr': lr_xyz}, {'params': [focus], 'lr': lr_focus},
                            {'params': [model.zcoef], 'lr': lr_zernike}]
                           + ([{'params': [BL], 'lr': lr_blur}] if BL is not None else [])
                           + ([{'params': extra, 'lr': lr_extra}] if extra else []))
    npx = float(mask.sum()) * R * R

    def blur_cov():
        if BL is None: return None
        if len(BL) == 2: return torch.stack([BL[0] ** 2, BL[1] ** 2, BL[0] * BL[1]])                  # line: v v^T
        return torch.stack([BL[0] ** 2, BL[1] ** 2 + BL[2] ** 2, BL[0] * BL[1]])                   # cov: L L^T, L = [[a, 0], [b, c]]

    def psf_of(pos, b):
        if group_psf_kwargs is None:
            return model(pos, focus.expand(b), blur=blur_cov())
        out = []
        for g in range(ngroups):
            kw = dict(group_psf_kwargs(g))
            if BL is not None:
                kw.setdefault('blur', blur_cov())
            out.append(model(pos, focus.expand(b), **kw))
        return torch.stack(out, 1)                        # [b, ngroups, R, R]

    def loss(sl):
        data = tracks.rois[sl]
        b = data.shape[0]
        g_ = grp[sl]
        r2 = rn2[sl] if pix else rn2                       # [b, 1, R * R] or scalar
        pos = torch.cat([xy[sl], zmax_um * torch.sigmoid(zr[sl])[:, None]], 1)
        psf = psf_of(pos, b)
        if psf.dim() == 4:      # per-group PSF -> per-frame PSF [b, T, R, R]
            psf = torch.gather(psf, 1, g_[..., None, None].expand(-1, -1, R, R))
            I, bg = solve_linear_grad(psf.reshape(b, T, -1), data.reshape(b, T, -1), mask[sl], g_, r2, ngroups)
            mu = I[..., None, None] * psf
        else:
            I, bg = solve_linear_grad(psf.reshape(b, -1), data.reshape(b, T, -1), mask[sl], g_, r2, ngroups)
            mu = I[..., None, None] * psf[:, None]
        mu = (mu + torch.gather(bg, 1, g_)[..., None, None]).clamp(min=1e-3)
        var = mu + (r2.reshape(b, 1, R, R) if pix else r2)
        nll = 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
        nll = torch.where(mask[sl][..., None, None], nll, torch.zeros_like(nll)).sum()
        return nll, I, bg, mu, var, data

    chunks = torch.arange(B, device=dev).split(batch)
    hist = []
    t0 = time.time()
    tot = float('nan')
    for it in range(iters):
        opt.zero_grad()
        tot = 0.0
        for ch in chunks:
            l = loss(ch)[0] / npx
            l.backward()
            tot += float(l.detach())
        if it < warmup:
            for v in [focus, model.zcoef] + ([BL] if BL is not None else []) + extra:
                v.grad = None
        if fix_zernike:
            model.zcoef.grad = None
        if fix_focus:
            focus.grad = None
        opt.step()
        hist.append(dict(it=it, nll_per_px=tot, focus_um=float(focus.detach())))
        if log is not None:
            log(it, tot, time.time() - t0, model.zcoef.detach())
        if stop is not None and stop():
            break
    with torch.no_grad():
        Is, bgs, rs, final = [], [], [], 0.0
        for ch in chunks:
            l_, I, bg, mu, var, data = loss(ch)
            final += float(l_) / npx
            Is.append(I); bgs.append(bg)
            rs.append(((data - mu) / var.sqrt())[mask[ch]])
        rs = torch.cat(rs)
    out = model.resized(R)
    out.focus_um = float(focus.detach())
    if BL is not None:
        out.blur = blur_cov().detach()
    rep = dict(method='tracks', n_samples=B, iterations=len(hist), resid_var=float(rs.var()), nll_per_px=final,
               nll_per_px_before_last_step=tot,
               focus_um=float(focus.detach()), seconds=time.time() - t0,
               at_limit=float((xy.detach().abs().max(1).values > c - 1).float().mean()))
    return _result(out, rep, hist, dict(xy=xy.detach(), z=(zmax_um * torch.sigmoid(zr)).detach(),
                                            I=torch.cat(Is), bg=torch.cat(bgs),
                                            extra=[e.detach() for e in extra]))
