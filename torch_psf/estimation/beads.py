"""Vectorial PSF fit to bead z-stacks (Poisson likelihood, Adam): the Zernike coefficients (and optionally NA /
n_med / apodization) of a VectorialPSF, with per bead x, y (shared by its planes), per plane photons and background,
an optional finite bead size and an optional polynomial lateral drift of each stack over the planes."""
import time
import torch


def _result(psf, report, history, data):
    """Result of the low-level estimators: dict(psf, report, history, **data)."""
    return dict(psf=psf, report=report, history=history, **data)
from torch_psf.fit import poisson_nll


def sphere_points(diameter_um, n=32, seed=0):
    """n points (um) uniformly on the surface of a sphere (a surface-stained bead), centred at 0: [n, 3]."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g)
    return v / v.norm(dim=1, keepdim=True) * (diameter_um / 2)


def fit_beads(stack, focus_um, vectorial, iters=300, lr=3e-3, fit=('zcoef',), emitter_z_um=0.0, stack_id=None,
              drift=False, bead_diameter_um=None, nball=32, chunk=None, log=None, stop=None, weights=None,
              fit_dz_scale=False, fit_height=False, intensity='free', heldout_iters=150, drift_order=3):
    """Fit a VectorialPSF (copied; roisize = the ROI size) to bead stacks [nbeads, nk, P, P] (photons) imaged at the
    focus positions focus_um [nk] (focal plane relative to the coverslip, um; a stage scan gives
    focus = f0 + sign dz k). The beads sit at height emitter_z_um. Poisson likelihood, Adam (lr for the PSF
    parameters).

    fit: names of the vectorial parameters to fit ('zcoef', 'NA', 'n_med', 'apod'). stack_id [nbeads] (long) and
    drift=True: per stack a lateral xy drift over the planes, a polynomial of degree drift_order (1 = linear, default 3:
    cubic) in the plane index without a constant term (the constant is the bead position). bead_diameter_um: the bead
    is modelled as nball points on a sphere (else a point). With fit_height the sphere stays tied to the fitted
    centre height: a bead lies on the coverslip, so its diameter = 2 x the height (bead_diameter_um only switches the
    sphere on and gives the starting height when emitter_z_um is 0). fit_dz_scale: fit a scale factor of the focus steps (stage calibration; the focus
    positions become focus_um * scale). fit_height: fit the bead (centre) height above the coverslip, starting at
    emitter_z_um. intensity: 'free' (photons per bead and plane) or 'bead_x_plane' (photons per bead times a per-plane
    factor shared by all beads, e.g. bleaching or illumination drift; fewer parameters). chunk: beads per backward
    pass (memory; None = all at once). weights [nbeads]: e.g. 0 for held-out beads; after the fit those beads get
    their own position / photons / bg (and the drift of their stack) refitted with the PSF fixed (heldout_iters Adam
    steps) and are scored
    (report: heldout_nll_per_px). log(it, nll, seconds, zcoef), stop(): progress / early stop.
    Returns dict(psf, report, history, xy [nbeads, 2] px rel. to the ROI centre, photons [nbeads, nk],
    bg [nbeads, nk], model [nbeads, nk, P, P] (the fitted expected images), chi2 [nbeads, nk] (mean (d - mu)^2 / mu
    per pixel), drift [nstacks, drift_order, 2] px (polynomial coefficients for t = plane index / half the
    stack, t in [-1, 1]) or None, dz_scale, height_um, bead_diameter_um)."""
    d, focus, vec, emitter_z = stack, focus_um, vectorial, emitter_z_um
    nb, nk, P = d.shape[0], d.shape[1], d.shape[-1]
    dev = d.device
    psf = vec.resized(P)
    for k in fit:
        getattr(psf, k).requires_grad_(True)
    if 'zcoef' not in fit:
        psf.zcoef.requires_grad_(False)
    xy = torch.zeros(nb, 2, device=dev, requires_grad=True)
    bg0 = d.flatten(2).quantile(0.1, 2)
    logI = torch.log((d.flatten(2).sum(2) - bg0 * P ** 2).clamp_min(100))
    if intensity == 'bead_x_plane':
        logIb = logI.mean(1).clone().requires_grad_(True)
        logIk = torch.zeros(nk, device=dev, requires_grad=True)
        int_params = [logIb, logIk]
        photons = lambda sl: torch.exp(logIb[sl][:, None] + logIk[None, :])
    elif intensity == 'free':
        logI = logI.requires_grad_(True)
        int_params = [logI]
        photons = lambda sl: torch.exp(logI[sl])
    else:
        raise ValueError(intensity)
    bg = bg0.clone().requires_grad_(True)
    f = focus[None].expand(nb, -1).reshape(-1)
    groups = [dict(params=[getattr(psf, k) for k in fit], lr=lr), dict(params=[xy], lr=1e-2),
              dict(params=int_params, lr=1e-2), dict(params=[bg], lr=0.3)]
    dr = None
    if drift:
        sid = torch.zeros(nb, dtype=torch.long, device=dev) if stack_id is None else stack_id.to(dev)
        dr = torch.zeros(int(sid.max()) + 1, int(drift_order), 2, device=dev, requires_grad=True)
        groups.append(dict(params=[dr], lr=2e-3))
    log_s = height = None
    if fit_dz_scale:
        log_s = torch.zeros((), device=dev, requires_grad=True)
        groups.append(dict(params=[log_s], lr=1e-3))
    if fit_height:
        z_start = float(emitter_z) if (emitter_z or bead_diameter_um is None) else 0.5 * float(bead_diameter_um)
        height = torch.tensor(z_start, device=dev, requires_grad=True)
        groups.append(dict(params=[height], lr=2e-3))
    opt = torch.optim.Adam(groups)
    ball = None if bead_diameter_um is None else sphere_points(bead_diameter_um, nball).to(dev)
    unit_ball = None if bead_diameter_um is None else sphere_points(1.0, nball).to(dev)     # diameter 1
    kk = torch.arange(nk, device=dev, dtype=torch.float32) - (nk - 1) / 2
    tt = kk / max((nk - 1) / 2, 1.0)                                     # plane index in [-1, 1]
    tpow = torch.stack([tt ** (k + 1) for k in range(int(drift_order))], 0)          # [order, nk]
    w = None if weights is None else weights.to(dev).float()
    sl_all = [slice(None)] if chunk is None else [slice(s, s + chunk) for s in range(0, nb, chunk)]

    def model(sl):
        n_ = d[sl].shape[0]
        pxy = xy[sl][:, None].expand(-1, nk, -1)
        if dr is not None:
            pxy = pxy + torch.einsum('bkc,kn->bnc', dr[sid[sl]], tpow)
        ze = torch.full((n_, nk, 1), float(emitter_z), device=dev) if height is None else height.expand(n_, nk, 1)
        pos = torch.cat([pxy, ze], -1).reshape(-1, 3)
        fs = focus[None].expand(n_, -1).reshape(-1) if (chunk is not None or not isinstance(sl, slice)) else f
        if log_s is not None:
            fs = fs * torch.exp(log_s)
        if ball is None:
            img = psf(pos, fs)
        else:
            b = ball if height is None else unit_ball * (2 * height)      # tied: diameter = 2 x centre height
            q = b / psf.pixel_um
            pb = pos[:, None, :] + torch.stack([q[:, 0], q[:, 1], b[:, 2]], 1)[None]
            img = psf(pb.reshape(-1, 3), fs.repeat_interleave(len(ball))).reshape(len(pos), len(ball), P, P).mean(1)
        return photons(sl).reshape(-1, 1, 1) * img + bg[sl].reshape(-1, 1, 1)

    hist = []
    t0 = time.time()
    tot = float('nan')
    for it in range(iters):
        opt.zero_grad()
        tot = 0.0
        for sl in sl_all:
            l = poisson_nll(d[sl].reshape(-1, P, P), model(sl))
            if w is not None:
                l = l * w[sl].repeat_interleave(nk)
            l = l.sum()
            l.backward()
            tot += float(l.detach())
        opt.step()
        hist.append(dict(it=it, nll=tot))
        if log is not None:
            log(it, tot, time.time() - t0, psf.zcoef.detach())
        if stop is not None and stop():
            break
    for k in fit:
        getattr(psf, k).requires_grad_(False)
    rep = dict(method='beads', n_beads=nb, n_planes=nk, iterations=len(hist), seconds=time.time() - t0, at_limit=0.0)
    ntrain = float(d.numel() if w is None else (w > 0).sum() * d[0].numel())
    rep['nll_per_px'] = tot / ntrain
    if w is not None and bool((w == 0).any()):
        # held-out beads: their own position / photons / bg with the PSF (and all global terms) fixed, then scored
        ho = torch.nonzero(w == 0)[:, 0]
        frozen = [log_s, height] + [psf.zcoef] + int_params[1:]
        for v in frozen:
            if v is not None:
                v.requires_grad_(False)
        own_int = int_params[:1]                     # photons per bead (the per-plane factors stay fixed)
        g2 = [dict(params=[xy], lr=1e-2), dict(params=own_int, lr=1e-2), dict(params=[bg], lr=0.3)]
        if dr is not None:                           # the drift of the held-out stacks (only they get a gradient)
            g2.append(dict(params=[dr], lr=2e-3))
        opt2 = torch.optim.Adam(g2)
        for it in range(heldout_iters):
            opt2.zero_grad()
            l = poisson_nll(d[ho].reshape(-1, P, P), model(ho)).sum()
            l.backward()
            opt2.step()
        with torch.no_grad():
            rep['heldout_nll_per_px'] = float(poisson_nll(d[ho].reshape(-1, P, P), model(ho)).sum()) / d[ho].numel()
    psf.zcoef.requires_grad_(True)
    if log_s is not None:
        rep['dz_scale'] = float(torch.exp(log_s))
    if height is not None:
        rep['height_um'] = float(height)
    with torch.no_grad():
        ph = photons(slice(None)).detach()
        # gain / statistics check: median over bead planes of the mean (d - mu)^2 / mu per pixel (~1 if Poisson)
        c2, mus = [], []
        for i in range(nb):
            mu = model(torch.tensor([i], device=dev))
            mus.append(mu.reshape(nk, P, P))
            c2.append(((d[i].reshape(-1, P, P) - mu) ** 2 / mu.clamp_min(1e-6)).flatten(1).mean(1))
        c2 = torch.stack(c2)
        tr = torch.ones(nb, dtype=torch.bool, device=c2.device) if w is None else (w > 0)
        rep['chi2_median'] = float(c2[tr].median())
        if not bool(tr.all()):
            rep['heldout_chi2_median'] = float(c2[~tr].median())
    return _result(psf, rep, hist, dict(xy=xy.detach(), photons=ph, bg=bg.detach(), model=torch.stack(mus), chi2=c2,
                                        drift=None if dr is None else dr.detach(),
                                        dz_scale=None if log_s is None else float(torch.exp(log_s)),
                                        height_um=None if height is None else float(height),
                                        bead_diameter_um=None if bead_diameter_um is None else
                                        (float(2 * height) if height is not None else float(bead_diameter_um))))


def stage_focus(nk, dz_um, sign=1.0, f0=0.0):
    """Focus positions of a stage scan: f0 + sign dz (k - (nk - 1) / 2), k = 0..nk-1 (um)."""
    return f0 + sign * dz_um * (torch.arange(nk, dtype=torch.float32) - (nk - 1) / 2)
