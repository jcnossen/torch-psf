"""Vectorial PSF fit to bead z-stacks (Poisson likelihood, Adam): the Zernike coefficients (and optionally NA /
n_med / apodization) of a VectorialPSF, with per bead x, y (shared by its planes), per plane photons and background,
an optional finite bead size and an optional linear lateral drift of each stack over the planes."""
import time
import torch
from torch_psf.fit import poisson_nll
from torch_psf.estimate import PSFEstimate


def sphere_points(diameter_um, n=32, seed=0):
    """n points (um) uniformly on the surface of a sphere (a surface-stained bead), centred at 0: [n, 3]."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g)
    return v / v.norm(dim=1, keepdim=True) * (diameter_um / 2)


def fit_beads(stack, focus_um, vectorial, iters=300, lr=3e-3, fit=('zcoef',), emitter_z_um=0.0, stack_id=None,
              drift=False, bead_diameter_um=None, nball=32, chunk=None, log=None, stop=None, weights=None):
    """Fit a VectorialPSF (copied; roisize = the ROI size) to bead stacks [nbeads, nk, P, P] (photons) imaged at the
    focus positions focus_um [nk] (focal plane relative to the coverslip, um; a stage scan gives
    focus = f0 + sign dz k). The beads sit at height emitter_z_um. Poisson likelihood, Adam (lr for the PSF
    parameters).

    fit: names of the vectorial parameters to fit ('zcoef', 'NA', 'n_med', 'apod'). stack_id [nbeads] (long) and
    drift=True: per stack a linear xy drift (px per plane). bead_diameter_um: the bead is modelled as nball points on
    a sphere (else a point). chunk: beads per backward pass (memory; None = all at once). weights [nbeads]: e.g. 0
    for held-out beads. log(it, nll, seconds, zcoef), stop(): progress / early stop.
    Returns PSFEstimate: psf = the fitted VectorialPSF; data = dict(xy [nbeads, 2] px rel. to the ROI centre,
    photons [nbeads, nk], bg [nbeads, nk], drift [nstacks, 2] px per plane or None)."""
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
    logI = torch.log((d.flatten(2).sum(2) - bg0 * P ** 2).clamp_min(100)).requires_grad_(True)
    bg = bg0.clone().requires_grad_(True)
    f = focus[None].expand(nb, -1).reshape(-1)
    groups = [dict(params=[getattr(psf, k) for k in fit], lr=lr), dict(params=[xy], lr=1e-2),
              dict(params=[logI], lr=1e-2), dict(params=[bg], lr=0.3)]
    dr = None
    if drift:
        sid = torch.zeros(nb, dtype=torch.long, device=dev) if stack_id is None else stack_id.to(dev)
        dr = torch.zeros(int(sid.max()) + 1, 2, device=dev, requires_grad=True)
        groups.append(dict(params=[dr], lr=1e-4))
    opt = torch.optim.Adam(groups)
    ball = None if bead_diameter_um is None else sphere_points(bead_diameter_um, nball).to(dev)
    kk = torch.arange(nk, device=dev, dtype=torch.float32) - (nk - 1) / 2
    w = None if weights is None else weights.to(dev).float()
    sl_all = [slice(None)] if chunk is None else [slice(s, s + chunk) for s in range(0, nb, chunk)]
    hist = []
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        tot = 0.0
        for sl in sl_all:
            n_ = d[sl].shape[0]
            pxy = xy[sl][:, None].expand(-1, nk, -1)
            if dr is not None:
                pxy = pxy + dr[sid[sl]][:, None, :] * kk[None, :, None]
            pos = torch.cat([pxy, torch.full((n_, nk, 1), float(emitter_z), device=dev)], -1).reshape(-1, 3)
            fs = focus[None].expand(n_, -1).reshape(-1) if chunk is not None else f
            if ball is None:
                img = psf(pos, fs)
            else:
                q = ball / psf.pixel_um
                pb = pos[:, None, :] + torch.stack([q[:, 0], q[:, 1], ball[:, 2]], 1)[None]
                img = psf(pb.reshape(-1, 3), fs.repeat_interleave(len(ball))).reshape(len(pos), len(ball), P, P).mean(1)
            mu = torch.exp(logI[sl]).reshape(-1, 1, 1) * img + bg[sl].reshape(-1, 1, 1)
            l = poisson_nll(d[sl].reshape(-1, P, P), mu)
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
    psf.zcoef.requires_grad_(True)
    rep = dict(method='beads', n_beads=nb, n_planes=nk, iterations=len(hist), nll_per_px=tot / d.numel(),
               seconds=time.time() - t0, at_limit=0.0)
    return PSFEstimate(psf, rep, hist, dict(xy=xy.detach(), photons=torch.exp(logI.detach()), bg=bg.detach(),
                                            drift=None if dr is None else dr.detach()))


def stage_focus(nk, dz_um, sign=1.0, f0=0.0):
    """Focus positions of a stage scan: f0 + sign dz (k - (nk - 1) / 2), k = 0..nk-1 (um)."""
    return f0 + sign * dz_um * (torch.arange(nk, dtype=torch.float32) - (nk - 1) / 2)
