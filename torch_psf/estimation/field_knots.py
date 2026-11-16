"""In-situ estimation of a field-dependent PSF on the spline knots (the knot route of knots.py, per field tile).

The field model (torch_psf.field.FieldZernikes) gives every emitter the Zernikes base + poly(x, y). The knot route
needs one spline per PSF, so the field is split into tiles: per tile t one spline is built from the Zernikes at the
tile's centre c_t (the mean field position of its spots), its spots are localized on it, and the knot gradient of
their NLL is passed to the Zernikes by one VJP through the knot build (CubicSplinePSF.vjp_vectorial):

    g_t = d NLL_t / d zcoef   at  zcoef_t = base + field(c_t)
    d NLL / d base = sum_t g_t,     d NLL / d coef = sum_t g_t . d field(c_t) / d coef     (+ the ridge prior)

Adam on the base Zernikes and the polynomial coefficients. The tile size sets the resolution of the approximation
(within a tile the PSF is that of its centre); the polynomial itself is continuous. Cost per iteration: one knot build
and one VJP per tile (the cost of fit_zernike_knots times the number of tiles), plus the localization of every spot.

  fit_field_knots    the joint fit
  tile_assign        tiles of field positions on a square grid
  localize_field     localization of ROIs with a field-dependent model (FieldPSF / TiledSplinePSF) or any PSF
"""
import time
import numpy as np
import torch
from torch_psf.fit import localize, at_limit
from torch_psf.spline import emitter_height_pos_fn
from torch_psf.estimation.knots import knot_gradient, _adam, _sync, _result


def tile_assign(xy, tile, extent=None):
    """(tile index [N] numpy, nx, ny, extent) of field positions xy [N, 2] px on a grid of tile x tile px over extent
    (x0, y0, x1, y1) (default the bounding box of xy); positions outside the extent go to the nearest tile."""
    xy = np.asarray(xy.detach().cpu() if torch.is_tensor(xy) else xy, np.float64).reshape(-1, 2)
    if extent is None:
        lo, hi = xy.min(0), xy.max(0) + 1e-6
        extent = (lo[0], lo[1], hi[0], hi[1])
    x0, y0, x1, y1 = (float(v) for v in extent)
    nx = max(1, int(np.ceil((x1 - x0) / tile - 1e-9)))
    ny = max(1, int(np.ceil((y1 - y0) / tile - 1e-9)))
    ix = np.clip(np.floor((xy[:, 0] - x0) / tile).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((xy[:, 1] - y0) / tile).astype(int), 0, ny - 1)
    return iy * nx + ix, nx, ny, (x0, y0, x1, y1)


def _sub(v, idx):
    if v is None or not torch.is_tensor(v) or v.dim() == 0:
        return v
    return v[torch.as_tensor(idx, device=v.device)]


@torch.no_grad()
def localize_field(psf, rois, xy=None, var=None, theta0=None, iters=30, zstarts=5, batch=1500):
    """(theta [N, 5], nll [N]) of ROIs [N, P, P] localized with psf at field positions xy [N, 2] px: a TiledSplinePSF
    per tile, a FieldPSF in bound batches (each ROI with the Zernikes at its position), any other PSF (or xy None)
    with torch_psf.fit.localize."""
    if xy is None or not hasattr(psf, 'bind'):
        return localize(psf, rois, theta0, var=var, iters=iters, zstarts=zstarts)
    xy = torch.as_tensor(xy, dtype=torch.float32, device=rois.device).reshape(-1, 2)
    N = len(rois)
    th = torch.zeros(N, 5, device=rois.device)
    nll = torch.zeros(N, device=rois.device)
    if hasattr(psf, 'tile_index'):
        ti = psf.tile_index(xy).cpu().numpy()
        groups = [(psf.spline(int(t)), np.flatnonzero(ti == t)) for t in np.unique(ti)]
    else:
        groups = [(None, np.arange(s, min(s + batch, N))) for s in range(0, N, batch)]
    for m, idx in groups:
        it = torch.as_tensor(idx, device=rois.device)
        model = m if m is not None else psf.bind(xy[it])
        t, l = localize(model, rois[it], None if theta0 is None else theta0[it], var=_sub(var, idx), iters=iters,
                        zstarts=zstarts, batch=max(len(idx), 1))
        th[it], nll[it] = t, l
    return th, nll


@torch.no_grad()
def expected_field(psf, theta, xy, batch=1500):
    """expected images [N, P, P] = I psf(x, y, z; field position xy) + bg of a field-dependent model"""
    xy = torch.as_tensor(xy, dtype=torch.float32, device=theta.device).reshape(-1, 2)
    out = []
    for s in range(0, len(theta), batch):
        t = theta[s:s + batch]
        out.append(t[:, 3, None, None] * psf.render(t[:, :3], xy=xy[s:s + batch]) + t[:, 4, None, None])
    return torch.cat(out) if out else torch.zeros(0, psf.roisize, psf.roisize, device=theta.device)


def fit_field_knots(spline, rois, xy, vectorial, field, tile=256.0, extent=None, pos_fn=None, iters=60, lr=2e-3,
                    lr_field=None, loc_iters=3, zstarts=5, var=None, fit_base=True, min_spots=30, theta=None,
                    log=None, stop=None):
    """Joint in-situ fit of the base Zernikes of `vectorial` and the field polynomial `field` (FieldZernikes; its
    modes must be modes of the model) from SMLM ROIs [N, P, P] (photons) at field positions xy [N, 2] (camera px,
    e.g. the ROI centres). spline: CubicSplinePSF that defines the knot grid (roisize P; its knots are overwritten).
    tile (px) / extent: the tile grid (tile_assign; default extent = the field box of `field`); tiles with fewer than
    min_spots spots are merged into the nearest kept tile. lr / lr_field (default lr): Adam steps (waves) of the base
    and of the polynomial coefficients; fit_base: also fit the base (the constant terms). loc_iters / zstarts: the
    localization per iteration (30 Levenberg-Marquardt iterations with zstarts starts the first time). var: read-noise
    variance (scalar or [N, P, P]). theta: starting emitter parameters [N, 5]. pos_fn: knot z -> (emitter z, focus).
    log(it, nll, seconds, zcoef, coef); stop() -> True stops early.

    Returns dict(psf, report, history, ...): psf = torch_psf.field.FieldPSF (copies of vectorial and field with the
    fitted values); data = dict(zcoef, coef, theta [N, 5], nll [N] (the last localization), tile [N], centers [T, 2]
    (the tile centres used), tile_counts)."""
    from torch_psf.field import FieldPSF
    spl, d = spline, rois
    dev = spl.dev
    vec = vectorial
    pos_fn = emitter_height_pos_fn(vec.focus_um) if pos_fn is None else pos_fn
    big = vec.resized(spl.Pl)
    fz = field.copy().to(dev)
    xy = torch.as_tensor(xy, dtype=torch.float32, device=dev).reshape(-1, 2)
    if extent is None:
        cx, cy, sx, sy = fz.norm
        extent = (cx - sx, cy - sy, cx + sx, cy + sy)
    ti, nx, ny, extent = tile_assign(xy, tile, extent)
    # merge sparse tiles into the nearest kept one
    cnt = np.bincount(ti, minlength=nx * ny)
    keep = np.flatnonzero(cnt >= min_spots)
    if len(keep) == 0:
        keep = np.array([int(np.argmax(cnt))])
    gx, gy = keep % nx, keep // nx
    for t in np.flatnonzero((cnt > 0) & (cnt < min_spots)):
        if t in keep:
            continue
        j = keep[np.argmin((gx - t % nx) ** 2 + (gy - t // nx) ** 2)]
        ti[ti == t] = j
    tiles = [int(t) for t in np.unique(ti)]
    idx = {t: np.flatnonzero(ti == t) for t in tiles}
    xyn = xy.cpu().numpy()
    centers = np.stack([xyn[idx[t]].mean(0) for t in tiles])                       # [T, 2] mean spot position
    ct = torch.as_tensor(centers, dtype=torch.float32, device=dev)
    zc = vec.zcoef.detach().clone().to(dev)
    m_z, v_z = torch.zeros_like(zc), torch.zeros_like(zc)
    m_c, v_c = torch.zeros_like(fz.coef), torch.zeros_like(fz.coef)
    lr_f = lr if lr_field is None else lr_field
    th = torch.zeros(len(d), 5, device=dev) if theta is None else theta.clone()
    first = theta is None
    nll_spot = torch.zeros(len(d), device=dev)
    hist = []
    t0 = time.time()
    for it in range(iters):
        with torch.no_grad():
            off = fz.offsets(ct, vec.zidx)                                         # [T, nz]
        g_tiles = torch.zeros(len(tiles), len(zc), device=dev)
        tot = 0.0
        for k, t in enumerate(tiles):
            ii = idx[t]
            it_ = torch.as_tensor(ii, device=dev)
            with torch.no_grad():
                big.zcoef.copy_(zc + off[k])
            spl.knots = spl.knots_from_vectorial(big, pos_fn, chunk=spl.nz)
            dt, vt = d[it_], _sub(var, ii)
            if first:
                tt, nl = localize(spl, dt, None, var=vt, iters=30, zstarts=zstarts)
            else:
                tt, nl = localize(spl, dt, th[it_], var=vt, iters=loc_iters, zstarts=zstarts)
            th[it_], nll_spot[it_] = tt, nl
            gK, tl = knot_gradient(spl, dt, tt, var=vt)
            g_tiles[k] = spl.vjp_vectorial(big, pos_fn, gK, chunk=spl.nz)['zcoef']
            tot += tl
        first = False
        # chain rule into the polynomial coefficients, plus the ridge prior
        fz.coef.grad = None
        offg = fz.offsets(ct, vec.zidx)
        pen = fz.penalty()
        ((offg * g_tiles).sum() + pen).backward()
        gc = fz.coef.grad * fz.mask
        _sync(dev)
        hist.append(dict(it=it, nll=tot, penalty=float(pen.detach()), seconds=time.time() - t0))
        if log is not None:
            log(it, tot, time.time() - t0, zc.clone(), fz.coef.detach().clone())
        if fit_base:
            m_z, v_z, sz = _adam(m_z, v_z, g_tiles.sum(0), it)
            zc = zc - lr * sz
        m_c, v_c, sc = _adam(m_c, v_c, gc, it)
        with torch.no_grad():
            fz.coef.sub_(lr_f * sc * fz.mask)
        if stop is not None and stop():
            break
    sec = time.time() - t0
    out_vec = vec.resized(vec.roisize)
    with torch.no_grad():
        out_vec.zcoef.copy_(zc)
    fz.coef.requires_grad_(False)
    out = FieldPSF(out_vec, fz)
    lim = float(at_limit(th, spl).float().mean())
    rep = dict(method='field_knots', iterations=len(hist), n_emitters=len(d), n_tiles=len(tiles), tile_px=float(tile),
               grid=[nx, ny], seconds=sec, nll=hist[-1]['nll'] if hist else float('nan'),
               penalty=hist[-1]['penalty'] if hist else float('nan'), at_limit=lim,
               n_field_params=fz.n_params, prior_sd=fz.prior_sd)
    tile_of = np.zeros(len(d), int)
    for k, t in enumerate(tiles):
        tile_of[idx[t]] = k
    return _result(out, rep, hist, dict(zcoef=zc, coef=fz.coef.detach().clone(), theta=th, nll=nll_spot,
                                        tile=tile_of, centers=centers,
                                        tile_counts=[int(len(idx[t])) for t in tiles]))
