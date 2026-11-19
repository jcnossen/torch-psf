"""Single-emitter maximum-likelihood fitting (batched Levenberg-Marquardt) and the Cramer-Rao lower bound.

Noise model: Poisson photon counts, optionally with sCMOS read noise in the usual shifted-Poisson approximation
(d + var ~ Poisson(mu + var), var = read-noise variance in e^2 per pixel; Huang et al., Nat. Methods 2013). Data and
model in photons (e-).

Parameter vector of an emitter: theta = (x px, y px, z um, photons, bg photons / px); x, y relative to the ROI centre.
Any PSF model with render(pos, grad=True) -> (psf, d/dx, d/dy, d/dz), xy_limit and z_range (None for 2D models; z is
then held fixed) can be fitted with localize().

Variable projection (solve_linear): the closed-form per-frame intensities and grouped backgrounds of a track sample for
a given PSF (used by torch_psf.estimation.fit_tracks).
"""
import torch


def poisson_nll(d, mu, var=None, weight=None, mu_min=1e-6):
    """Negative Poisson log-likelihood without the data-only constant, summed over the pixels -> [N].
    var: sCMOS read-noise variance (e^2; broadcastable to d), weight: pixel mask / weights (e.g. 0 outside the
    frame)."""
    if var is None:
        mu = mu.clamp_min(mu_min)
        l = mu - d * torch.log(mu)
    else:
        mv = (mu + var).clamp_min(mu_min)
        l = mv - (d + var).clamp_min(0) * torch.log(mv)
    if weight is not None:
        l = l * weight
    return l.flatten(1).sum(1)


def model_and_jacobian(psf, theta):
    """Expected image mu = I psf(x, y, z) + bg [N, P, P] and its Jacobian [N, 5, P, P] for theta [N, 5]."""
    th = theta
    g, gx, gy, gz = psf.render(th[:, :3], grad=True)
    I = th[:, 3, None, None]
    return I * g + th[:, 4, None, None], torch.stack([I * gx, I * gy, I * gz, g, torch.ones_like(g)], 1)


def clamp_theta(th, psf, margin=0.05, z_limits=None, xy_limit=None):
    """Clamp theta [N, 5] in place to the model's support: |x|, |y| <= xy_limit - margin, z within the model's z
    limits, photons >= 1, bg >= 1e-3. Returns th."""
    xm = (psf.xy_limit if xy_limit is None else xy_limit) - margin
    th[:, 0].clamp_(-xm, xm)
    th[:, 1].clamp_(-xm, xm)
    zl = z_limits if z_limits is not None else getattr(psf, 'z_limits', None) or psf.z_range
    if zl is not None:
        th[:, 2].clamp_(zl[0], zl[1])
    th[:, 3].clamp_min_(1.0)
    th[:, 4].clamp_min_(1e-3)
    return th


@torch.no_grad()
def lm_poisson(model_fn, d, th0, var=None, weight=None, clamp=None, fixed=(), iters=30, lam0=1.0, mu_min=1e-6):
    """Batched Levenberg-Marquardt for Poisson data.

    model_fn(theta [N, p]) -> (mu [N, ...], J [N, p, ...]) (expected image and its Jacobian). d: data [N, ...].
    var: sCMOS read-noise variance, weight: pixel weights (both broadcastable to d). clamp(theta) -> theta: projection
    onto the allowed region (in place is fine). fixed: indices of theta held at th0.
    Damping: H + lam diag(H); lam x 0.3 after an accepted step, x 5 after a rejected one (1e-4 .. 1e6).
    Returns theta [N, p], nll [N]."""
    th = th0.clone()
    if clamp is not None:
        th = clamp(th)
    fx = list(fixed)
    lam = torch.full((len(d),), float(lam0), device=d.device)
    v = 0.0 if var is None else var

    def nllf(mu):
        return poisson_nll(d, mu, var, weight, mu_min)

    mu, J = model_fn(th)
    nll = nllf(mu)
    for _ in range(iters):
        mu = mu.clamp_min(mu_min)
        if fx:
            J[:, fx] = 0
        Jw = J if weight is None else J * weight[:, None]
        mv = mu + v
        gr = (Jw * ((d + v) / mv - 1)[:, None]).flatten(2).sum(2)
        H = torch.einsum('nap,nbp->nab', Jw.flatten(2) / mv.flatten(1)[:, None], J.flatten(2))
        dg = torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-12)
        H = H + torch.diag_embed(dg * lam[:, None])
        if fx:
            H[:, fx, fx] = 1.0
        step = torch.linalg.solve(H.double(), gr.double()[..., None])[..., 0].to(th.dtype)
        tn = th + step
        if clamp is not None:
            tn = clamp(tn)
        mun, Jn = model_fn(tn)
        nn_ = nllf(mun)
        ok = nn_ < nll
        th[ok], mu[ok], J[ok], nll[ok] = tn[ok], mun[ok], Jn[ok], nn_[ok]
        lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-4, 1e6)
    return th, nll


def initial_theta(d, z=0.0, bg_quantile=0.2, min_photons=50.0):
    """Starting values (x = y = 0, z, photons = sum above the background, bg = a low quantile of the patch)."""
    bg = d.flatten(1).quantile(bg_quantile, 1)
    I = (d.flatten(1).sum(1) - bg * d[0].numel()).clamp_min(min_photons)
    return torch.stack([torch.zeros_like(I), torch.zeros_like(I), torch.full_like(I, float(z)), I, bg], 1)


@torch.no_grad()
def localize(psf, rois, theta0=None, var=None, iters=30, zstarts=5, batch=3000, fixed=(), weight=None):
    """Maximum-likelihood theta = (x px, y px, z um, photons, bg photons / px) per ROI with Levenberg-Marquardt.

    psf: any PSF model (render(pos, grad=True), xy_limit, z_range). rois [N, P, P]: photons (e-), P = psf.roisize;
    x, y relative to the ROI centre. theta0 [N, 5]: starting values; without it, a 3D model is started at zstarts
    heights evenly inside psf.z_range and the best likelihood wins; a 2D model (z_range None) starts at z = 0 with z
    fixed. var: sCMOS read-noise variance (e^2; scalar or [N, P, P]); weight [N, P, P]: pixel weights (0 = ignore);
    fixed: indices of theta held at theta0; batch: ROIs per GPU batch.
    Returns theta [N, 5] and the negative log-likelihood [N] (constant dropped)."""
    d, th0, nb = rois, theta0, batch
    is2d = getattr(psf, 'z_range', None) is None
    if is2d:
        fixed = tuple(sorted(set(fixed) | {2}))
    if th0 is None:
        zl = [0.0] if is2d else [float(z) for z in torch.linspace(psf.z_range[0], psf.z_range[1], zstarts + 2)[1:-1]]
        best, bnll = None, None
        for z in zl:
            th, nll = localize(psf, d, initial_theta(d, z), var, iters, batch=nb, fixed=fixed, weight=weight)
            if best is None:
                best, bnll = th, nll
            else:
                m = nll < bnll
                best[m], bnll[m] = th[m], nll[m]
        return best, bnll
    out, onll = [], []
    dtype = getattr(psf, 'dtype', torch.float32)
    sl = lambda a, s: a if (a is None or not torch.is_tensor(a) or a.dim() == 0 or len(a) != len(d)) else a[s:s + nb]
    for s in range(0, len(d), nb):
        th, nll = lm_poisson(lambda t: model_and_jacobian(psf, t), d[s:s + nb], th0[s:s + nb].to(dtype),
                             var=sl(var, s), weight=sl(weight, s), clamp=lambda t: clamp_theta(t, psf),
                             fixed=fixed, iters=iters)
        out.append(th); onll.append(nll)
    return torch.cat(out), torch.cat(onll)


@torch.no_grad()
def fisher(psf, theta, var=None, weight=None, params=(0, 1, 2, 3, 4)):
    """Fisher information [N, p, p] of the listed parameters (indices into theta) at theta [N, 5]; var / weight as in
    localize."""
    mu, J = model_and_jacobian(psf, theta)
    mu = mu.clamp_min(1e-6)
    J = J[:, list(params)].flatten(2)
    w = 1.0 / (mu + (0.0 if var is None else var)).flatten(1)
    if weight is not None:
        w = w * weight.flatten(1)
    return torch.einsum('nap,nbp->nab', J * w[:, None], J)


@torch.no_grad()
def crlb(psf, theta, var=None, weight=None, ridge=0.0):
    """Cramer-Rao lower bound (standard deviation) [N, 5] of (x px, y px, z um, photons, bg) at theta [N, 5]. For a 2D model
    the z column is NaN (z not estimated). ridge: added to the diagonal of the Fisher matrix (e.g. 1e-9 when pixels
    are masked out)."""
    is2d = getattr(psf, 'z_range', None) is None
    params = (0, 1, 3, 4) if is2d else (0, 1, 2, 3, 4)
    F = fisher(psf, theta, var, weight, params)
    if ridge:
        F = F + ridge * torch.eye(len(params), device=F.device)
    F = F.double()
    sd = torch.sqrt(torch.diagonal(torch.linalg.inv(F), dim1=1, dim2=2).clamp_min(0)).float()
    if is2d:
        sd = torch.stack([sd[:, 0], sd[:, 1], torch.full_like(sd[:, 0], float('nan')), sd[:, 2], sd[:, 3]], 1)
    return sd


def at_limit(theta, psf, margin=0.06):
    """Bool [N]: fitted x or y at the model's offset limit (clamped)."""
    return theta[:, :2].abs().max(1).values > psf.xy_limit - margin


@torch.no_grad()
def readnoise_variance(readnoise_e, B, R, device):
    """read-noise variance (e^2) of track samples for the variable-projection fits: a scalar read noise (e rms) -> 0-dim
    tensor; a per-pixel map per sample [B, R, R] (e rms, e.g. an sCMOS read-noise map cut at each sample's crop) ->
    [B, 1, R * R] (broadcasts against [B, T, R * R]). Returns (rn2, per_pixel)."""
    r = torch.as_tensor(readnoise_e, dtype=torch.float32, device=device)
    if r.dim() > 0:
        assert r.shape == (B, R, R), f'per-pixel read noise must be [B, R, R] = {(B, R, R)}, not {tuple(r.shape)}'
        return (r ** 2).reshape(B, 1, R * R), True
    return r ** 2, False


def solve_linear_grad(psf, data, mask, group, rn2, ngroups=2, iters=3, I_min=0.0):
    """Variable projection helper: closed-form (iteratively reweighted linear least squares) solution of the linear
    parameters of a track sample,
        mu[b, t, :] = I[b, t] psf[b, :] + bg[b, group[b, t]],
    per-frame intensities and a background shared per group (e.g. per illumination-pattern angle). The weights are
    1 / (mu + rn2) (Poisson + read noise).

    psf [B, P] (or [B, T, P]: per-frame PSF), data [B, T, P], mask [B, T] bool, group [B, T] long in [0, ngroups),
    rn2: read-noise variance (e^2). Returns I [B, T], bg [B, ngroups].

    Differentiable (solve_linear is the same without gradient). The solution is the Poisson-weighted IRLS estimate, not
    the minimiser of the Gaussian NLL that fit_tracks / heldout_tracks evaluate, so the envelope theorem does not hold:
    the gradient of the profiled NLL must go through this solve (holding I / bg fixed biases the PSF fit)."""
    B, T, P = data.shape
    pf = psf[:, None, :] if psf.dim() == 2 else psf                  # [B, 1|T, P]
    n = T + ngroups
    m = mask.float()
    G = torch.nn.functional.one_hot(group.clamp(min=0), ngroups).float() * m[..., None]   # [B, T, ng]
    w = torch.ones(B, T, P, device=data.device) * m[..., None]
    I = torch.zeros(B, T, device=data.device); bg = torch.zeros(B, ngroups, device=data.device)
    for _ in range(iters):
        # normal equations; columns: I_t (psf on frame t), bg_g (ones on the frames of group g)
        pw = pf * w
        A_II = torch.diag_embed((pw * pf).sum(-1))
        A_Ib = (pw.sum(-1)[..., None] * G)
        A_bb = torch.diag_embed((w.sum(-1)[..., None] * G).sum(1))
        rhs_I = (pw * data).sum(-1)
        rhs_b = ((w * data).sum(-1)[..., None] * G).sum(1)
        A = torch.zeros(B, n, n, device=data.device)
        A[:, :T, :T] = A_II; A[:, :T, T:] = A_Ib; A[:, T:, :T] = A_Ib.transpose(1, 2); A[:, T:, T:] = A_bb
        A = A + torch.eye(n, device=data.device) * 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]
        pad = torch.cat([1 - m, torch.zeros(B, ngroups, device=data.device)], 1)   # masked frames: trivial rows
        A = A + torch.diag_embed(pad)
        x = torch.linalg.solve(A, torch.cat([rhs_I, rhs_b], 1))
        I = x[:, :T].clamp(min=I_min) * m; bg = x[:, T:]
        mu = I[..., None] * pf + (G * bg[:, None, :]).sum(-1, keepdim=True)
        w = m[..., None] / (mu.clamp(min=1e-3) + rn2)
    return I, bg


solve_linear = torch.no_grad()(solve_linear_grad)
solve_linear.__doc__ = """solve_linear_grad without gradient (I / bg as constants)."""
