"""First-order Legendre NAT (Droste et al., Optica 2025), in-situ local/global MLE."""
from math import sqrt
import time

import numpy as np
import torch
from scipy.optimize import minimize

from torch_psf.field import FieldPSF, FieldZernikes, bbox_norm
from torch_psf.fit import poisson_nll
from torch_psf.spline import SplinePSF
from torch_psf.zernike import zernike
from .field_knots import localize_field, tile_assign, _sub
from .knots import knot_gradient

NAT_MODES = (4, 5, 6, 7, 8, 11)


def nat_polynomial(gamma, *, convention="torch"):
    """Rows in NAT_MODES order, columns [1,v,v²,u,uv,u²]. Units are Noll RMS waves.

    Uses the field scale of Vectorfit 0.3.4 compute_xy_gamma_legendre, including
    its unit quadratic astigmatism coefficient (as distinct from Eq. S58).
    Default gamma coordinates multiply RMS-normalized Noll modes. For the
    literal barred gamma in Supplement Eqs. S43/S58, pass convention="paper"
    with coefficients in waves. Both conventions span the same subspace.
    """
    if gamma.shape != (13,):
        raise ValueError("expected 13 first-order NAT parameters")
    if convention == "paper":
        factors = [sqrt(3)] * 4 + [sqrt(6)] * 5 + [sqrt(8)] * 3 + [sqrt(5)]
        factors[8] /= sqrt(15 / 7)
        gamma = gamma / gamma.new_tensor(factors)
    elif convention != "torch":
        raise ValueError("expected torch or paper coefficient convention")
    d, dx, dy, curvature, a0, a45, ax, ay, aq, cx, cy, cr, spherical = gamma
    zero = torch.zeros_like(d)
    s, t = sqrt(3), sqrt(5)
    return torch.stack(
        [
            torch.stack(row)
            for row in (
                (
                    d - t * curvature,
                    s * dy,
                    1.5 * t * curvature,
                    s * dx,
                    zero,
                    1.5 * t * curvature,
                ),
                (a45, s * ax, zero, s * ay, 3 * aq, zero),
                (a0, -s * ay, -1.5 * aq, s * ax, zero, 1.5 * aq),
                (cy, s * cr, zero, zero, zero, zero),
                (cx, zero, zero, s * cr, zero, zero),
                (spherical, zero, zero, zero, zero, zero),
            )
        ]
    )


def nat_field(vectorial, gamma, norm):
    """Build a continuous FieldPSF from a starting pupil, fit gamma, and field normalization."""
    vec = vectorial.resized(vectorial.roisize)
    missing = [j for j in NAT_MODES if j not in vec.zidx]
    if missing:
        extra = torch.stack([zernike(j, vec.rho.clamp(max=1), vec.phi) * vec.mask for j in missing])
        vec.Z = torch.cat((vec.Z, extra))
        vec.zcoef = torch.nn.Parameter(torch.cat((vec.zcoef.detach(), vec.zcoef.new_zeros(len(missing)))))
        vec.zidx = list(vec.zidx) + missing
    field = FieldZernikes(NAT_MODES, [2, 2, 2, 1, 1, 0], norm=norm, device=vec.dev)
    if gamma is not None:
        coef = nat_polynomial(torch.as_tensor(gamma, device=vec.dev, dtype=vec.dtype))
        with torch.no_grad():
            vec.zcoef[[vec.zidx.index(j) for j in NAT_MODES]] = coef[:, 0]
            field.coef.copy_(coef[:, 1:])
    return FieldPSF(vec, field)


def fit_nat(
    vectorial,
    rois,
    xy,
    *,
    norm=None,
    backend="bspline",
    tile=64.0,
    iters=10,
    global_iters=4,
    loc_iters=5,
    zstarts=5,
    fraction=1.0,
    seed=0,
    var=None,
    oversample=3,
    dz_um=0.025,
    xy_limit=3.0,
    batch=96,
    gamma0=None,
    field_varying=True,
    max_rois=None,
    log=None,
    stop=None,
):
    """Alternate local xyz/photons/background fits and a shared 13-parameter pupil fit.

    xy are camera (column,row) pixels; norm=(cx,cy,sx,sy) defines the physical FOV.
    direct propagates each emitter's pupil; bspline shares an integrated B-spline
    and one optical VJP per occupied tile. The output FieldPSF is continuous in xy.
    Other starting pupil modes, focus and optical settings are held fixed. gamma0
    is optional; otherwise the six constant terms start at the input pupil.
    fraction selects a reproducible random subset for each global update. max_rois
    caps the calibration set by sampling evenly across occupied 10x10 field bins.
    Pass a matrix of initial pupils in gamma0 to fit independent starts and select
    the lowest final likelihood on the same calibration set.
    With field_varying=False, fit only the six spatially constant pupil terms.
    backend="web" uses the browser's pixel-value Catmull-Rom table and one Adam
    update per outer iteration. Match the browser's midpoint optics, oversample=2,
    dz_um=.04, tile and field normalization for cross-runtime comparisons.
    """
    web = backend == "web"
    if backend not in ("direct", "bspline", "web") or min(iters, global_iters, loc_iters, batch, zstarts) < 1:
        raise ValueError("expected direct/bspline/web and positive iteration counts and batch size")
    if not (0 < fraction <= 1) or tile <= 0:
        raise ValueError("expected fraction in (0,1] and positive tile size")
    d = torch.as_tensor(rois, device=vectorial.dev, dtype=vectorial.dtype)
    xy = torch.as_tensor(xy, device=vectorial.dev, dtype=vectorial.dtype)
    if (
        d.ndim != 3
        or len(d) == 0
        or d.shape[1:] != (vectorial.roisize, vectorial.roisize)
        or xy.shape != (len(d), 2)
    ):
        raise ValueError("expected nonempty matching ROIs and camera coordinates")
    if var is not None:
        var = torch.as_tensor(var, device=vectorial.dev, dtype=vectorial.dtype)
        if var.ndim and var.shape != d.shape:
            raise ValueError("var must be scalar or have the ROI stack shape")
    norm = tuple(bbox_norm(xy) if norm is None else norm)
    if len(norm) != 4 or min(norm[2:]) <= 0 or not np.isfinite(norm).all():
        raise ValueError("invalid normalized field domain")
    rng = np.random.default_rng(seed)
    sample_indices = np.arange(len(d))
    if max_rois is not None:
        if max_rois < 1:
            raise ValueError("max_rois must be positive")
        if len(d) > max_rois:
            points = xy.detach().cpu().numpy()
            cells = np.clip(((points - norm[:2]) / norm[2:] + 1) * 5, 0, 9).astype(int)
            cell_id = cells[:, 0] + 10 * cells[:, 1]
            groups = [rng.permutation(np.flatnonzero(cell_id == k)) for k in np.unique(cell_id)]
            order = np.concatenate(groups)
            rank = np.concatenate([np.arange(len(group)) for group in groups])
            sample_indices = order[np.lexsort((rng.random(len(order)), rank))[:max_rois]]
            d, xy, var = d[sample_indices], xy[sample_indices], _sub(var, sample_indices)
    if gamma0 is not None:
        initial = torch.as_tensor(gamma0)
        if initial.ndim == 2:
            if initial.shape[1] != 13 or len(initial) == 0:
                raise ValueError("expected a nonempty set of 13-parameter starts")
            best, scores = None, []
            for start in initial:
                candidate = fit_nat(
                    vectorial,
                    d,
                    xy,
                    norm=norm,
                    backend=backend,
                    tile=tile,
                    iters=iters,
                    global_iters=global_iters,
                    loc_iters=loc_iters,
                    zstarts=zstarts,
                    fraction=fraction,
                    seed=seed,
                    var=var,
                    oversample=oversample,
                    dz_um=dz_um,
                    xy_limit=xy_limit,
                    batch=batch,
                    gamma0=start,
                    field_varying=field_varying,
                    log=log,
                    stop=stop,
                )
                scores.append(candidate["report"]["nll"])
                if best is None or scores[-1] < best["report"]["nll"]:
                    best = candidate
                    best["report"]["selected_start"] = len(scores) - 1
                if stop and stop():
                    break
            best["report"]["start_nll"] = scores
            best["indices"] = sample_indices
            return best
    psf = nat_field(vectorial, None, norm)
    vec, field = psf.base, psf.field
    columns = [vec.zidx.index(j) for j in NAT_MODES]
    base = vec.zcoef.detach().clone()
    gamma = base.new_zeros(13)
    gamma[[0, 5, 4, 10, 9, 12]] = base[columns]
    if gamma0 is not None:
        gamma = torch.as_tensor(gamma0, device=vec.dev, dtype=base.dtype).detach().clone()
    base[columns] = 0.0
    bounds = None
    if not field_varying:
        varying = [1, 2, 3, 6, 7, 8, 11]
        gamma[varying] = 0
        bounds = [(0., 0.) if i in varying else (None, None) for i in range(13)]

    def update_model(g):
        coef = nat_polynomial(g)
        with torch.no_grad():
            vec.zcoef.copy_(base)
            vec.zcoef[columns] = coef[:, 0]
            field.coef.copy_(coef[:, 1:])
        return psf

    grid = dict(
        kernel="catmull" if web else "bspline",
        integrated=not web,
        oversample=oversample,
        dz_um=dz_um,
        xy_limit=xy_limit,
        zpad=5 if web else 4,
    )
    cx, cy, sx, sy = norm
    extent = (cx - sx, cy - sy, cx + sx, cy + sy)
    bins, nx, _, _ = tile_assign(xy, tile, extent)
    spl = big = None
    if backend != "direct":
        spl = SplinePSF(
            vec.roisize,
            vec.z_range,
            device=vec.dev,
            pixel_um=vec.pixel_um,
            **dict(grid, xy_limit=xy_limit + int(not web)),
        )
        spl.xmax = min(xy_limit, spl.xmax)
        big = vec.resized(spl.Pl)

    def global_loss(x, indices, theta):
        g = base.new_tensor(x, requires_grad=True)
        poly = nat_polynomial(g)
        loss = 0.0
        grad = torch.zeros_like(g)
        if backend == "direct":
            for idx in indices.split(batch):
                t = theta[idx]
                coeff = base.expand(len(idx), -1).clone()
                coeff[:, columns] = field.features(xy[idx]).to(poly) @ poly.T
                signal = vec(t[:, :3], vec.focus_um, zcoef_offset=coeff - vec.zcoef.detach())
                mu = t[:, 3, None, None] * signal + t[:, 4, None, None]
                value = poisson_nll(d[idx], mu.double(), _sub(var, idx)).sum()
                grad += torch.autograd.grad(value, g, retain_graph=True)[0].detach()
                loss += value.item()
        else:
            selected = bins[indices.cpu().numpy()]
            for tile_id in np.unique(selected):
                idx = indices[torch.as_tensor(selected == tile_id, device=vec.dev)]
                center = [
                    [
                        cx - sx + (tile_id % nx + 0.5) * tile,
                        cy - sy + (tile_id // nx + 0.5) * tile,
                    ]
                ]
                c = (field.features(center).to(poly) @ poly.T)[0]
                with torch.no_grad():
                    big.zcoef.copy_(base)
                    big.zcoef[columns] = c
                    spl.knots = spl.sample(big)
                gk, value = knot_gradient(spl, d[idx], theta[idx], var=_sub(var, idx))
                gz = spl.vjp_vectorial(big, None, gk)["zcoef"]
                grad += torch.autograd.grad(c, g, gz[columns], retain_graph=True)[0].detach()
                loss += value
        return loss / len(indices), grad.cpu().double().numpy() / len(indices)

    theta = None
    history = []
    tic = time.monotonic()

    def local(g, initial, steps):
        psf = update_model(g)
        if backend != "direct":
            psf = psf.to_tiled(tile=tile, extent=extent, **grid)
        with torch.no_grad():
            return localize_field(psf, d, xy, var, initial, iters=steps, zstarts=zstarts, batch=batch)

    if web:
        gamma.requires_grad_(True)
        adam = torch.optim.Adam([gamma], lr=.002, eps=1e-12)
    for it in range(iters):
        theta, nll = local(gamma, theta, 30 if theta is None else loc_iters)
        count = max(1, round(len(d) * fraction))
        indices = torch.as_tensor(rng.choice(len(d), count, replace=False), device=vec.dev)
        x = gamma.detach().cpu().double().numpy()
        if web:
            _, gradient = global_loss(x, indices, theta)
            gamma.grad = base.new_tensor(gradient)
            if not field_varying:
                gamma.grad[varying] = 0
            adam.step()
            status, message = 0, "Adam update"
        else:
            opt = minimize(global_loss, x, args=(indices, theta), method="L-BFGS-B", bounds=bounds, jac=True,
                           options=dict(maxiter=global_iters, maxls=20, ftol=1e-10))
            gamma = base.new_tensor(opt.x)
            status, message = int(opt.status), str(opt.message)
        history.append(
            dict(
                it=it,
                local_nll=float(nll.sum()),
                global_status=status,
                global_message=message,
                seconds=time.monotonic() - tic,
            )
        )
        if log:
            log(it, float(nll.sum()), history[-1]["seconds"], gamma)
        if stop and stop():
            break
    theta, nll = local(gamma, theta, 30)
    out = update_model(gamma)
    report = dict(
        method="nat",
        field_varying=field_varying,
        backend=backend,
        gamma=gamma.tolist(),
        norm=list(norm),
        n_emitters=len(d),
        n_tiles=len(np.unique(bins)) if backend != "direct" else None,
        iterations=len(history),
        nll=float(nll.sum()),
        seconds=time.monotonic() - tic,
    )
    return dict(psf=out, report=report, theta=theta, nll=nll, history=history, indices=sample_indices)
