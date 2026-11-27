"""Calibrations of a trained SMLMNet: fresh BatchNorm statistics, and the sigma calibration of its localizations.

BatchNorm recalibration (bn_recalibrate): the running mean / variance stored with a checkpoint follow the last few
training batches (momentum 0.1). Every BatchNorm layer is reset and switched to a cumulative average (momentum None),
the network runs (train-mode BatchNorm only, no gradients, weights unchanged) on n_scenes fresh scene batches of the
training scenes with a fixed seed disjoint from the training (seed + 1000 k) and validation (seed + 7) seeds, `mix`
scene batches concatenated per forward.

Sigma calibration (fit_calibration): the network's uncertainties are rescaled per bin of its photon estimate, fitted on
simulated scenes of the training recipe (seed `seed`; a second, independent set at seed + 1 checks the fit). The
detections (rule, p >= p_min) are matched one-to-one to the labels per field (Hungarian, sqrt((dxy / 250 nm)^2 +
(dz / 500 nm)^2) <= 1), emitters with true z in z_range_um are used.
  full_xyz models: per photon bin a lower-triangular matrix A, applied as L -> L A to the Cholesky factor L of the
    predicted x, y, z covariance (calibrated covariance L A A^T L^T). method 'rms': A = chol of the second moment of
    the whitened errors (whitened rms 1); 'quantile' (robust): the same A, scaled so that the 68.27 % quantile of the
    squared Mahalanobis radius equals the chi2(3) quantile (3.5267), so that the 1-sigma ellipsoid holds 68.27 % of
    the errors also for heavy-tailed errors.
  diagonal models: per photon bin and axis a factor = the 68.27 % quantile of |error / sigma|.

calibration.json: format 'torch-psf nn calibration', version 1, bin 'photons', edges [...] (bin edges in
photoelectrons), xyz_matrix [bins][3][3] or factors {x: [bins], y: [...], z: [...]}, method, fit (how it was fitted),
check (coverages before / after on the check set), counts.
"""
import copy
import numpy as np
import torch

FORMAT, VERSION = 'torch-psf nn calibration', 1
DEFAULT_EDGES = (0, 200, 400, 700, 1000, 1500, 2500, 1e9)
BN_SEED = 91170001
FIT_SEED = 777001
BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


# ---------------------------------------------------------------------------------------------- BatchNorm
def bn_layers(net):
    return [m for m in net.modules() if isinstance(m, BN_TYPES)]


@torch.no_grad()
def recalibrate_bn(net, batches):
    """Reset the BatchNorm running statistics of net and re-estimate them as the cumulative average over the
    forwards of `batches` (input tensors [B, window, H, W]); weights unchanged. -> (net in eval mode, forwards)."""
    layers = bn_layers(net)
    if not layers:
        raise ValueError('no BatchNorm layer in the model: nothing to recalibrate')
    mom = [m.momentum for m in layers]
    net.eval()
    for m in layers:
        m.reset_running_stats(); m.momentum = None; m.train()
    n = 0
    for x in batches:
        net(x); n += 1
    if n == 0:
        raise ValueError('no calibration batch')
    for m, mo in zip(layers, mom):
        m.momentum = mo
    net.eval()
    return net, n


def bn_recalibrate(net, scenes, n_scenes=256, mix=4, seed=BN_SEED):
    """recalibrate_bn on n_scenes scene batches of `scenes` (torch_psf.nn.scenes.Scenes) drawn from `seed`, `mix` per
    forward; checks that no parameter changed. -> (net, info dict)."""
    ref = copy.deepcopy(net.state_dict())

    def gen():
        buf = []
        for f, _, _ in scenes.batches(n_scenes, seed=seed, z=net.z):
            buf.append(f)
            if len(buf) == mix:
                yield torch.cat(buf); buf = []
        if buf:
            yield torch.cat(buf)
    net, nf = recalibrate_bn(net, gen())
    sd = net.state_dict()
    bufs = {n for n, _ in net.named_buffers()}
    for k, v in ref.items():
        if k not in bufs and not torch.equal(v, sd[k]):
            raise RuntimeError(f'parameter {k} changed during the recalibration')
    return net, dict(seed=seed, n_scenes=n_scenes, batch=scenes.batch, mix=mix, forwards=nf,
                     bn_layers=len(bn_layers(net)), momentum='None (cumulative average)')


# ---------------------------------------------------------------------------------------------- matching
def match_window(lab_xyz, det_xyz, pixel_nm, pixel_nm_y=None, dxy_nm=250., dz_nm=500.):
    """One field: labels [g, 3] and detections [r, 3] (x, y px, z um; numpy) -> (label index, detection index) of the
    one-to-one matches with sqrt((dxy / dxy_nm)^2 + (dz / dz_nm)^2) <= 1 (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    py = pixel_nm if pixel_nm_y is None else pixel_nm_y
    dxy = np.hypot(pixel_nm * (lab_xyz[:, None, 0] - det_xyz[None, :, 0]),
                   py * (lab_xyz[:, None, 1] - det_xyz[None, :, 1]))
    dz = np.abs(lab_xyz[:, None, 2] - det_xyz[None, :, 2]) * 1e3
    c = np.sqrt((dxy / dxy_nm) ** 2 + (dz / dz_nm) ** 2)
    c[c > 1] = 1e6
    a, b = linear_sum_assignment(c)
    ok = c[a, b] < 1e5
    return a[ok], b[ok]


@torch.no_grad()
def collect_errors(net, scenes, n_batches, seed, p_min=0.5, rule='nms', pixel_nm=117.6, pixel_nm_y=None, **rule_kw):
    """Matched errors of the detections on n_batches scene batches drawn from `seed`: dict photons (estimate), z_true,
    ex, ey (px), ez (um), sx, sy, sz, cov [m, 3, 3] (full_xyz) and counts."""
    from torch_psf.nn.merge import detect
    if rule == 'nms':
        rule_kw.setdefault('raw_th', 0.05)
    E = {k: [] for k in ('photons', 'z_true', 'ex', 'ey', 'ez', 'sx', 'sy', 'sz', 'cov')}
    nwin = ngt = ndet = 0
    for frames, _, lab in scenes.batches(n_batches, seed=seed, z=net.z):
        d = detect(net(frames), rule, p_min=p_min, n_params=net.n_params, cov=net.cov, **rule_kw)
        d = {k: v.float().cpu().numpy() for k, v in d.items()}
        L = {k: v.float().cpu().numpy() for k, v in lab.items()}
        B = frames.shape[0]
        nwin += B; ngt += len(L['x']); ndet += len(d['x'])
        for w in range(B):
            gi = np.nonzero(L['fov'] == w)[0]; ri = np.nonzero(d['b'] == w)[0]
            if not len(gi) or not len(ri):
                continue
            a, b = match_window(np.stack([L[k][gi] for k in 'xyz'], 1), np.stack([d[k][ri] for k in 'xyz'], 1),
                                pixel_nm, pixel_nm_y)
            g, r = gi[a], ri[b]
            E['photons'].append(d['photons'][r]); E['z_true'].append(L['z'][g])
            for k in 'xyz':
                E['e' + k].append(d[k][r] - L[k][g]); E['s' + k].append(d['s' + k][r])
            if 'cov' in d:
                E['cov'].append(d['cov'][r])
    E = {k: np.concatenate(v) if v else np.zeros(0) for k, v in E.items()}
    if not len(E['cov']):
        del E['cov']
    E['counts'] = dict(windows=nwin, gt=ngt, det=ndet, matched=int(len(E['photons'])))
    return E


# ---------------------------------------------------------------------------------------------- fits
def fit_xyz_matrix(fit, check, edges=DEFAULT_EDGES, z_range_um=(-np.inf, np.inf), method='rms'):
    """Per-bin matrices A (module docstring) from the errors `fit` (collect_errors), verified on `check`.
    -> (matrices [bins][3][3] as lists, report dict)."""
    from scipy.stats import chi2, norm
    if method not in ('rms', 'quantile'):
        raise ValueError(f"matrix calibration method {method!r}: 'rms' or 'quantile'")
    edges = np.asarray(edges, np.float64)

    def residuals(e):
        use = (e['z_true'] >= z_range_um[0]) & (e['z_true'] <= z_range_um[1])
        bins = np.searchsorted(edges[1:-1], e['photons'][use], side='right')
        cov = e['cov'][use].astype(np.float64)
        error = np.stack([e['e' + k][use] for k in 'xyz'], -1)
        return bins, np.linalg.solve(np.linalg.cholesky(cov), error[..., None])[..., 0]

    bins, w = residuals(fit)
    if len(w) < 50:
        raise ValueError('matrix calibration needs at least 50 matched positions')
    q3 = chi2.ppf(chi2.cdf(1.0, 1), 3)             # 68.27 % quantile of chi2(3)
    factors = []
    for i in range(len(edges) - 1):
        v = w[bins == i]
        if len(v) < 50:
            v = w
        A = np.linalg.cholesky(v.T @ v / len(v) + 1e-6 * np.eye(3))
        if method == 'quantile':
            r2 = (np.linalg.solve(A, v.T) ** 2).sum(0)
            A = A * np.sqrt(np.quantile(r2, chi2.cdf(1.0, 1)) / q3)
        factors.append(A)
    bins, w = residuals(check)
    z = np.linalg.solve(np.asarray(factors)[bins], w[..., None])[..., 0]

    def report(v):
        return dict(n=len(v), rms=np.sqrt((v * v).mean(0)).tolist(), coverage=(abs(v) <= 1).mean(0).tolist(),
                    marginal_pit=[np.histogram(u, bins=10, range=(0, 1))[0].tolist() for u in norm.cdf(v).T],
                    joint_pit=np.histogram(chi2.cdf((v * v).sum(1), 3), bins=10, range=(0, 1))[0].tolist())

    def cov12(v):
        if not len(v):
            return dict(n=0)
        r2 = (v * v).sum(1)
        return dict(n=len(v), white_rms=np.sqrt((v * v).mean(0)).tolist(), cov1=(abs(v) <= 1).mean(0).tolist(),
                    cov2=(abs(v) <= 2).mean(0).tolist(), ellipsoid_1sigma=float((r2 <= q3).mean()),
                    ellipsoid_2sigma=float((r2 <= chi2.ppf(chi2.cdf(4.0, 1), 3)).mean()))
    out = dict(before=report(w), after=report(z), method=method, all=dict(before=cov12(w), after=cov12(z)),
               check_bins=[dict(bin=[float(edges[i]), float(edges[i + 1])], before=cov12(w[bins == i]),
                                after=cov12(z[bins == i])) for i in range(len(edges) - 1)])
    return np.asarray(factors).tolist(), out


def _q68(v):
    return float(np.quantile(np.abs(v), 0.6827)) if len(v) else float('nan')


def fit_axis_factors(fit, check, edges=DEFAULT_EDGES, z_range_um=(-np.inf, np.inf), axes='xyz', nmin=50):
    """Diagonal models: per bin and axis the 68.27 % quantile of |error / sigma| (bins with < nmin matches: linear
    interpolation between the neighbouring bins). -> (factors {axis: [bins]}, report)."""
    edges = np.asarray(edges, np.float64)
    nb = len(edges) - 1

    def sel(e):
        m = (e['z_true'] >= z_range_um[0]) & (e['z_true'] <= z_range_um[1])
        return m, np.clip(np.searchsorted(edges[1:-1], e['photons'], side='right'), 0, nb - 1)
    m, b = sel(fit)
    F = {}
    for k in axes:
        f = np.array([_q68((fit['e' + k] / fit['s' + k])[m & (b == i)]) if (m & (b == i)).sum() >= nmin else np.nan
                      for i in range(nb)])
        good = np.isfinite(f)
        if not good.any():
            raise ValueError(f'axis {k}: no photon bin with >= {nmin} matches')
        F[k] = np.interp(np.arange(nb), np.nonzero(good)[0], f[good]).tolist()
    m, b = sel(check)
    rep = {}
    for k in axes:
        zb = check['e' + k][m] / check['s' + k][m]
        za = zb / np.asarray(F[k])[b[m]]
        rep[k] = dict(q68_before=_q68(zb), q68_after=_q68(za), coverage_before=float((abs(zb) <= 1).mean()),
                      coverage_after=float((abs(za) <= 1).mean()))
    return F, rep


def fit_calibration(net, scenes, n_batches=150, seed=FIT_SEED, p_min=0.5, z_range_um=None, method='quantile',
                    edges=DEFAULT_EDGES, rule='nms', pixel_nm=117.6, pixel_nm_y=None, **rule_kw):
    """The calibration.json dict (module docstring) of net on `scenes`: fit set at `seed`, check set at seed + 1."""
    zr = tuple(scenes.p['z_range_um']) if z_range_um is None else tuple(z_range_um)
    Ef = collect_errors(net, scenes, n_batches, seed, p_min, rule, pixel_nm, pixel_nm_y, **rule_kw)
    Ec = collect_errors(net, scenes, n_batches, seed + 1, p_min, rule, pixel_nm, pixel_nm_y, **rule_kw)
    S = dict(format=FORMAT, version=VERSION, bin='photons', edges=[float(e) for e in edges])
    if net.cov:
        S['xyz_matrix'], S['check'] = fit_xyz_matrix(Ef, Ec, edges, zr, method)
        S['method'] = method
    else:
        S['factors'], S['check'] = fit_axis_factors(Ef, Ec, edges, zr, 'xyz' if net.z else 'xy')
        S['method'] = 'axis_q68'
    S['fit'] = dict(batches=n_batches, seeds=[seed, seed + 1], p_min=p_min, z_range_um=list(zr), rule=rule,
                    rule_kw=rule_kw, batch=scenes.batch, size=scenes.size, halo=scenes.halo)
    S['counts'] = dict(fit=Ef['counts'], check=Ec['counts'])
    return S


# ---------------------------------------------------------------------------------------------- apply
def _bins(cal, v):
    e = torch.as_tensor(np.asarray(cal['edges'], np.float64)[1:-1], dtype=v.dtype, device=v.device)
    return torch.bucketize(v, e, right=True)


def apply(cal, rows):
    """rows (torch_psf.nn.merge: tensors photons, sx, sy, sz, cov) -> a copy with calibrated sx, sy, sz (and cov)."""
    r = dict(rows)
    i = _bins(cal, r[cal.get('bin', 'photons')].float())
    if 'xyz_matrix' in cal:
        if 'cov' not in r:
            raise ValueError('an xyz_matrix calibration needs a full_xyz model')
        cov = r['cov'].double()
        A = torch.as_tensor(np.asarray(cal['xyz_matrix'], np.float64), dtype=cov.dtype, device=cov.device)[i]
        LA = torch.linalg.cholesky(cov) @ A
        r['cov'] = (LA @ LA.mT).to(rows['cov'].dtype)
        r['sx'], r['sy'], r['sz'] = LA.norm(dim=-1).to(rows['sx'].dtype).unbind(-1)
    for k, f in (cal.get('factors') or {}).items():
        fk = torch.as_tensor(np.asarray(f, np.float64), dtype=r['s' + k].dtype, device=r['s' + k].device)[i]
        r['s' + k] = r['s' + k] * fk
    return r
