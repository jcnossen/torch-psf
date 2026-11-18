"""Held-out checks of a PSF estimate: split the spots by position into a checkerboard of tiles (train on one parity,
test on the other: the same emitters / structures never appear in both), and compare PSF models by the mean
negative log-likelihood per pixel of the test spots after a full refit of each spot. heldout_tracks: the same for
track samples (torch_psf.spots.Tracks, the data of estimation.fit_tracks) with the PSF frozen; paired_nll: the paired
per-sample NLL difference of two PSFs on the same samples."""
import math
import numpy as np
import torch
from torch_psf.fit import localize, at_limit, solve_linear


def split_checkerboard(x, y, tile=16):
    """Parity (0 / 1) of the tile (tile x tile px) that holds each position (image px; numpy or torch)."""
    x, y = np.asarray(x.cpu() if torch.is_tensor(x) else x), np.asarray(y.cpu() if torch.is_tensor(y) else y)
    return ((x // tile).astype(int) + (y // tile).astype(int)) % 2


def _localize(psf, rois, theta0, var, zstarts, xy):
    if xy is not None and hasattr(psf, 'bind'):           # a field-dependent model (torch_psf.field)
        from torch_psf.estimation.field_knots import localize_field
        return localize_field(psf, rois, xy, var=var, theta0=theta0, zstarts=zstarts)
    return localize(psf, rois, theta0, var=var, zstarts=zstarts)


@torch.no_grad()
def heldout_nll(psf, rois, var=None, theta0=None, zstarts=5, xy=None):
    """Mean NLL per pixel of the patches rois [N, P, P] after localizing each with psf (Poisson, constant dropped);
    also the fraction at the offset limit and the NLL per spot (nll [N] numpy). Lower is better; differences between
    PSFs are what matters. zstarts: z starts of the multi-start localization. xy [N, 2]: field positions of the ROIs
    (camera px) for a field-dependent model (torch_psf.field.FieldPSF / TiledSplinePSF; ignored by other models)."""
    th, nll = _localize(psf, rois, theta0, var, zstarts, xy)
    lim = at_limit(th, psf).float().mean() if hasattr(psf, 'xy_limit') else torch.zeros(())
    return dict(nll_per_px=float(nll.mean()) / rois[0].numel(), at_limit=float(lim), theta=th,
                nll=nll.double().cpu().numpy())


def compare_heldout(psfs, rois, var=None, zstarts=5, xy=None):
    """{name: heldout_nll(psf)} for a dict of PSF models, plus the difference to the first one (nll_per_px_vs_first)
    and, per spot, the mean NLL difference to the first one with its standard error (dnll_per_spot_vs_first,
    dnll_se) and the fraction of spots that it fits better (frac_better). xy: field positions for field-dependent
    models (heldout_nll)."""
    out = {k: heldout_nll(p, rois, var, zstarts=zstarts, xy=xy) for k, p in psfs.items()}
    v0 = next(iter(out.values()))
    first, n0 = v0['nll_per_px'], v0['nll']
    for v in out.values():
        v['nll_per_px_vs_first'] = v['nll_per_px'] - first
        dd = v['nll'] - n0
        v['dnll_per_spot_vs_first'] = float(dd.mean()) if len(dd) else 0.0
        v['dnll_se'] = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
        v['frac_better'] = float((dd < 0).mean()) if len(dd) else 0.0
    for v in out.values():
        v.pop('theta'); v.pop('nll')
    return out


@torch.no_grad()
def spot_deviance(psf, rois, var=None, theta=None, zstarts=5, xy=None, batch=4000):
    """Poisson deviance per pixel of every ROI under psf after its MLE fit (theta: fitted parameters, else fitted
    here): 2 sum(d log(d / mu) - (d - mu)) / P^2, about 1 for a spot the model explains. Overlapping emitters and
    other spots the single-emitter model does not explain give larger values. Returns (deviance [N] numpy, theta).
    zstarts / xy: as heldout_nll (a field-dependent model needs xy)."""
    if theta is None:
        theta, _ = _localize(psf, rois, None, var, zstarts, xy)
    field = xy is not None and hasattr(psf, 'bind')
    if field:
        from torch_psf.estimation.field_knots import expected_field
        xy = torch.as_tensor(xy, dtype=torch.float32, device=rois.device).reshape(-1, 2)
    out = []
    for s in range(0, len(rois), batch):                   # in batches: [N, P, P] images of a large set
        th = theta[s:s + batch]
        mu = expected_field(psf, th, xy[s:s + batch]) if field else psf.expected(th)
        d = rois[s:s + batch]
        v = var[s:s + batch] if (torch.is_tensor(var) and var.dim() > 0) else var
        if v is not None:
            d, mu = d + v, mu + v
        dv = 2 * (torch.where(d > 0, d * torch.log(d.clamp_min(1e-9) / mu.clamp_min(1e-6)), torch.zeros_like(d))
                  - (d - mu))
        out.append((dv.flatten(1).sum(1) / d[0].numel()).cpu().numpy())
    return (np.concatenate(out) if out else np.zeros(0)), theta


def paired_nll(nll_a, nll_b):
    """Paired NLL difference b - a of two PSFs on the same samples (nll [N] per sample, numpy or torch): mean, its
    standard error (ddof 1), z = mean / se and the fraction of samples that b fits better. Negative mean: b is better."""
    a = np.asarray(nll_a.cpu() if torch.is_tensor(nll_a) else nll_a, dtype=np.float64)
    b = np.asarray(nll_b.cpu() if torch.is_tensor(nll_b) else nll_b, dtype=np.float64)
    d = b - a
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
    m = float(d.mean()) if len(d) else 0.0
    return dict(mean=m, se=se, z=m / se if se > 0 else 0.0, frac_better=float((d < 0).mean()) if len(d) else 0.0)


def heldout_tracks(tracks, psfs, readnoise_e=1.0, zmax_um=0.2, groups=None, ngroups=1, refit_psfs=None,
                   z_starts_um=(0.02, 0.08, 0.15), iters=300, lr=0.03, batch=3000):
    """Held-out NLL of a frozen PSF on track samples (the model of estimation.fit_tracks): per sample x, y and z
    (through a sigmoid on [0, zmax_um]) are refitted by Adam (lr, iters) from every start z in z_starts_um, with the
    per-frame intensities I [B, T] and the per-(sample, group) backgrounds solved in closed form (fit.solve_linear)
    at every step; per sample the start with the lowest NLL is kept.

    tracks: Tracks (rois [B, T, R, R] in photons, mask, xy_init). psfs: the PSF model, or one per group (a list; frames
    of group g use psfs[g]); any model with render(pos [N, 3]) -> [N, R, R] (pos = x, y px from the ROI centre, z um),
    e.g. a VectorialPSF at its focus_um with its per-group dipM / blur set. groups [B, T] long in [0, ngroups)
    (default 0): the background groups (one background per sample and group) and, with a list of ngroups PSFs, the
    PSF of each frame (as fit_tracks with group_psf_kwargs). refit_psfs: models used for the refit iterations (same layout as
    psfs, e.g. their to_spline() tables, much faster); the NLL is then evaluated with psfs. readnoise_e: read noise
    (e- rms). NLL: Gaussian approximation of Poisson + read noise, 0.5 (d - mu)^2 / v + 0.5 log(2 pi v), v = mu + rn^2,
    over the present frames.

    Returns dict(nll [B], chi2 [B] (mean (d - mu)^2 / v per pixel of the sample's present frames), xy [B, 2] (px
    from the ROI centre), z [B] (um), I [B, T], bg [B, ngroups], nll_per_px (sum NLL / pixels), chi2_per_px (mean of
    chi2), n); arrays as numpy."""
    rois, mask = tracks.rois, tracks.mask
    dev = rois.device
    B, T, R = rois.shape[0], rois.shape[1], rois.shape[-1]
    c = (R - 1) / 2
    as_list = lambda p: list(p) if isinstance(p, (list, tuple)) else [p]
    psfs = as_list(psfs)
    refit = psfs if refit_psfs is None else as_list(refit_psfs)
    assert len(refit) == len(psfs)
    ng = int(ngroups)
    assert len(psfs) in (1, ng), 'psfs: one model or one per group'
    grp = torch.zeros(B, T, dtype=torch.long, device=dev) if groups is None else groups.to(dev).long()
    grp = torch.where(mask, grp, torch.zeros_like(grp))
    rn2 = torch.tensor(float(readnoise_e) ** 2, device=dev)
    chunks = torch.arange(B, device=dev).split(batch)

    def evaluate(models, xy, zr, sl):
        data, g_, m_ = rois[sl], grp[sl], mask[sl]
        b = data.shape[0]
        pos = torch.cat([xy[sl], zmax_um * torch.sigmoid(zr[sl])[:, None]], 1)
        if len(models) == 1:
            psf = models[0].render(pos)
            I, bg = solve_linear(psf.reshape(b, -1).detach(), data.reshape(b, T, -1), m_, g_, rn2, ng)
            mu = I[..., None, None] * psf[:, None]
        else:
            psf = torch.stack([m.render(pos) for m in models], 1)                       # [b, ng, R, R]
            psf = torch.gather(psf, 1, g_[..., None, None].expand(-1, -1, R, R))         # [b, T, R, R]
            I, bg = solve_linear(psf.reshape(b, T, -1).detach(), data.reshape(b, T, -1), m_, g_, rn2, ng)
            mu = I[..., None, None] * psf
        mu = (mu + torch.gather(bg, 1, g_)[..., None, None]).clamp(min=1e-3)
        var = mu + rn2
        mk = m_[..., None, None]
        nll = torch.where(mk, 0.5 * (data - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var), torch.zeros_like(mu))
        chi = torch.where(mk, (data - mu) ** 2 / var, torch.zeros_like(mu))
        npx = (m_.sum(1) * R * R).clamp_min(1)
        return nll.sum((1, 2, 3)), chi.sum((1, 2, 3)) / npx, I, bg

    best = None
    for z0 in z_starts_um:
        xy = (tracks.xy_init.float() - c).clone().requires_grad_(True)
        zr = torch.full((B,), math.log(z0 / (zmax_um - z0)), device=dev, requires_grad=True)
        opt = torch.optim.Adam([xy, zr], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            for ch in chunks:
                evaluate(refit, xy, zr, ch)[0].sum().backward()
            opt.step()
        with torch.no_grad():
            r = [evaluate(psfs, xy, zr, ch) for ch in chunks]
            cur = dict(nll=torch.cat([v[0] for v in r]), chi2=torch.cat([v[1] for v in r]), xy=xy.detach(),
                       z=zmax_um * torch.sigmoid(zr.detach()), I=torch.cat([v[2] for v in r]),
                       bg=torch.cat([v[3] for v in r]))
        if best is None:
            best = cur
        else:
            w = cur['nll'] < best['nll']
            for k in best:
                best[k] = torch.where(w.reshape(-1, *([1] * (best[k].dim() - 1))), cur[k], best[k])
    out = {k: v.cpu().numpy() for k, v in best.items()}
    npx = float(mask.sum()) * R * R
    out.update(nll_per_px=float(out['nll'].astype(np.float64).sum() / max(npx, 1.0)),
               chi2_per_px=float(np.mean(out['chi2'])) if B else float('nan'), n=int(B))
    return out
