"""Single-molecule emission-dipole orientation with the vectorial PSF (torch_psf.vectorial).

Model of one emitter: a dipole along the unit vector mu that wobbles in a cone, described by the order parameter
g in [0, 1]; its emission second-moment tensor is
    M = (1 - g) I / 3 + g mu mu^T          (trace 1; g = 1 fixed dipole, g = 0 freely rotating = isotropic emitter).
For a cone of half-angle a (uniform wobble), g = cos a (1 + cos a) / 2 (cone_to_g / g_to_cone).
mu is parametrised by an unconstrained 3-vector v, mu = v / |v| (no pole singularity; M depends on mu mu^T only, so
mu and -mu are the same orientation). angles(mu) gives (theta, phi): theta = polar angle from the optical axis in
[0, 90] deg, phi = azimuth (x = column, y = row) in [0, 360) deg after flipping mu to mu_z >= 0. (theta, phi) and
(theta, phi + 180) are distinct orientations (tilted towards +x vs -x with the same in-plane axis; mirror-image
PSFs); fold=True folds phi to [0, 180) (the in-plane axis only, the tilt sign dropped).

Linear structure: for fixed position / focus the image is linear in the 6 independent entries
m = (M_xx, M_yy, M_zz, M_xy, M_xz, M_yz):  psf = sum_k m_k B_k  (VectorialPSF.dipole_basis / basis_images). This gives
a closed-form start: weighted least squares of (I m, bg) per spot (solve_M), then a projection of the estimated M on
(mu, g) (project_M: top eigenvector, g from the eigenvalues).

Excitation: per frame (illumination pattern angle) a polarisation tensor K (trace 1; excitation_tensor, the K of
vectorial.photosel_tensor) scales the brightness by 3 tr(M K) (= 1 for the isotropic emitter; = 3 mu^T K mu for a
fixed dipole). An ensemble of emitters of order g with uniformly distributed mu then has the photoselected tensor
M_ens = (1 - gamma) I / 3 + gamma K, gamma = 0.4 g^2 (ensemble_gamma).

Fitting (fit_orientation): Poisson maximum likelihood of theta = (x px, y px, z um, I, bg, v_x, v_y, v_z, g) per spot
or per track (frames sharing position and orientation; per-frame known brightness factors h and optional K per frame),
Levenberg-Marquardt (torch_psf.fit.lm_poisson) with |v| renormalised after each step, started from the isotropic fit
plus the linear-M solution (and its mirror, mu_z -> -mu_z). CRLB (crlb_orientation): Fisher matrix in the tangent
coordinates (a, b) of mu (mu = normalize(mu0 + a e_theta + b e_phi)), finite at the pole; sd(theta) = sd(a),
sd(phi) = sd(b) / sin(theta).

Intensity I: photons an isotropic emitter (g = 0) at the coverslip would give per frame (times h, times the excitation
factor); a fixed dipole's detected total also depends on its orientation and height (the vectorial normalization).
"""
import math
import time
import torch
from torch_psf.vectorial import photosel_tensor
from torch_psf.fit import lm_poisson, localize

from torch_psf._dipole import M_to_m6, m6_to_M

BASIS = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')
NPAR = 9      # x, y, z, I, bg, v_x, v_y, v_z, g
PAR_NAMES = ('x', 'y', 'z', 'I', 'bg', 'vx', 'vy', 'vz', 'g')


# ------------------------------------------------------------------ orientation helpers
def unit(v):
    """mu = v / |v| ([..., 3])."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def canonical(mu):
    """The representative of +-mu with mu_z >= 0 (unit vectors [..., 3])."""
    s = torch.where(mu[..., 2:3] < 0, -1.0, 1.0)
    return mu * s


def angles(mu, deg=True, fold=False):
    """(theta, phi) of the orientation axis mu [..., 3] (any length): theta = angle from the optical axis in [0, 90],
    phi = azimuth from +x (column) towards +y (row) in [0, 360) of the representative with mu_z >= 0 (fold=True:
    phi mod 180, the in-plane axis only). Degrees (deg=True) or radians."""
    m = canonical(unit(mu))
    th = torch.acos(m[..., 2].clamp(-1, 1))
    ph = torch.atan2(m[..., 1], m[..., 0]) % (2 * math.pi)
    if fold:
        ph = ph % math.pi
    if deg:
        th, ph = torch.rad2deg(th), torch.rad2deg(ph)
    return th, ph


def from_angles(theta, phi, deg=True):
    """Unit vector mu [..., 3] of polar angle theta (from z) and azimuth phi (from x towards y)."""
    th = torch.as_tensor(theta, dtype=torch.float32)
    ph = torch.as_tensor(phi, dtype=torch.float32)
    if deg:
        th, ph = torch.deg2rad(th), torch.deg2rad(ph)
    th, ph = torch.broadcast_tensors(th, ph)
    return torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], -1)


def tangent_basis(mu):
    """(e_theta, e_phi) [..., 3] at the unit vectors mu (of the canonical representative's angles; at the pole
    phi = 0, i.e. e_theta = x, e_phi = y)."""
    th, ph = angles(mu, deg=False)
    ct, st, cp, sp = torch.cos(th), torch.sin(th), torch.cos(ph), torch.sin(ph)
    e_th = torch.stack([ct * cp, ct * sp, -st], -1)
    e_ph = torch.stack([-sp, cp, torch.zeros_like(sp)], -1)
    s = torch.where(mu[..., 2:3] < 0, -1.0, 1.0)      # tangents of mu itself (not of -mu): same plane, any sign ok
    return e_th * s, e_ph * s


def cone_to_g(a, deg=True):
    """Order parameter g of uniform wobble in a cone of half-angle a: g = cos a (1 + cos a) / 2."""
    a = torch.as_tensor(a, dtype=torch.float32)
    c = torch.cos(torch.deg2rad(a) if deg else a)
    return c * (1 + c) / 2


def g_to_cone(g, deg=True):
    """Cone half-angle of order parameter g (inverse of cone_to_g; g in [0, 1] -> a in [90, 0] deg)."""
    g = torch.as_tensor(g, dtype=torch.float32).clamp(0, 1)
    a = torch.acos(((-1 + torch.sqrt(1 + 8 * g)) / 2).clamp(-1, 1))
    return torch.rad2deg(a) if deg else a


def ensemble_gamma(g):
    """Photoselection strength gamma of an ensemble of emitters with order g and uniformly distributed axes:
    M_ens = (1 - gamma) I / 3 + gamma K, gamma = 0.4 g^2 (fixed dipoles: 2/5). (Inverse: g = sqrt(gamma / 0.4).)"""
    return 0.4 * torch.as_tensor(g, dtype=torch.float32) ** 2


def dipole_M(v, g):
    """M = (1 - g) I / 3 + g mu mu^T, mu = v / |v|; v [..., 3], g [...] -> [..., 3, 3]."""
    mu = unit(v)
    g = torch.as_tensor(g, dtype=mu.dtype, device=mu.device)
    eye = torch.eye(3, dtype=mu.dtype, device=mu.device)
    return (1 - g)[..., None, None] * eye / 3 + g[..., None, None] * mu[..., :, None] * mu[..., None, :]


def project_M(M):
    """Nearest (mu, g) of an estimated M [..., 3, 3]: mu = top eigenvector (canonical, mu_z >= 0) of the
    trace-normalised symmetric part, g = l1 - (l2 + l3) / 2 (least-squares fit of the eigenvalues
    ((1 - g) / 3 + g, (1 - g) / 3, (1 - g) / 3)), clamped to [0, 1]. Returns mu [..., 3], g [...]."""
    S = (M + M.transpose(-1, -2)) / 2
    tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1)
    S = S / torch.where(tr.abs() > 1e-12, tr, torch.ones_like(tr))[..., None, None]
    w, V = torch.linalg.eigh(S.double())
    mu = canonical(V[..., :, 2].to(M.dtype))
    g = (w[..., 2] - (w[..., 0] + w[..., 1]) / 2).to(M.dtype).clamp(0, 1)
    return mu, g


# ------------------------------------------------------------------ excitation
def excitation_tensor(e_inplane, eps=0.0, k_dir=None, ck=0.153, device=None):
    """Excitation polarisation tensor K (trace 1) of vectorial.photosel_tensor (gamma = 1): s-polarised in-plane field
    e e^T plus an eps fraction of evanescent p (ck k k^T + (1 - ck) z z^T; ck = |E_x|^2 / |E|^2 of the p wave).
    e_inplane, k_dir: in-plane 2-vectors (x = column, y = row; normalised here). Returns [3, 3]."""
    e = torch.as_tensor(e_inplane, dtype=torch.float32)
    e = e / e.norm()
    k = None
    if k_dir is not None:
        k = torch.as_tensor(k_dir, dtype=torch.float32)
        k = k / k.norm()
    K = photosel_tensor(torch.tensor(1.0), e, eps, k, ck)
    if k is None and eps:       # photosel_tensor applies eps only with k_dir; without it: p = pure z
        z = torch.zeros(3, 3); z[2, 2] = 1
        K = (1 - eps) * K + eps * z
    return K if device is None else K.to(device)


def excitation_factor(M, K):
    """Brightness factor 3 tr(M K): M [..., 3, 3], K [..., 3, 3] (broadcast) -> [...]; 1 for M = I / 3."""
    return 3 * (M * K.transpose(-1, -2)).sum((-1, -2))


# ------------------------------------------------------------------ Vortex PSF
def vortex_phase(model, charge=1, offset=0.0, apply=True, center=(0.0, 0.0), levels=0, depth=1.0):
    """Spiral phase plate of topological charge `charge` in the emission pupil (Vortex PSF, Hulleman et al., Nat.
    Commun. 12, 5934 (2021)): phase = depth * 2 pi * frac((charge * atan2(p_y - c_y, p_x - c_x) + offset) / 2 pi)
    (rad), p_x along +x (column), p_y along +y (row) of the pupil grid in units of the pupil radius, centre
    c = center (a lateral offset of the plate, pupil radii; a tensor makes the map differentiable in it). levels > 0:
    the spiral quantised to that many steps (the plate V-593-10-1 has 64); depth: the full phase step in waves
    (design wavelength / emission wavelength, e.g. 593 / 597.5; 1 = an exact 2 pi). Their MATLAB code
    (vecfitcpu_vortex, doevortexflip = 1, phase +atan2(y, x) with x the first array index = the image column in
    the DIPimage convention) is charge +1 here; their real data (4TU 10.4121/13031837) prefer charge +1 by the
    likelihood (study/orientation/vortex_real.py).
    Returns the phase map in waves ([npupil, npupil], inside the pupil); apply=True also writes it to
    model.pupil_phase (replacing the free phase map)."""
    c = torch.as_tensor(center, dtype=model.px.dtype, device=model.px.device)
    f = (charge * torch.atan2(model.py - c[1], model.px - c[0]) + offset) / (2 * math.pi) % 1.0
    if levels:
        f = torch.floor(f * levels + 0.5) / levels % 1.0
    W = depth * f * model.mask
    if apply:
        with torch.no_grad():
            model.pupil_phase.copy_(W.detach())
    return W


# ------------------------------------------------------------------ basis images
def basis_images(psf, pos, focus=None, grad=False, xy=None):
    """The six basis images [N, 6, R, R] (BASIS order) of a VectorialPSF at pos [N, 3] (x px, y px, z um) and focus
    (default psf.focus_um): psf(dipM=M) = sum_k m_k B_k, m = M_to_m6(M). grad=True also returns d/dx, d/dy (per px),
    d/dz (per um), each [N, 6, R, R] (forward-mode autodiff)."""
    return psf.dipole_basis(pos, grad, focus=focus, xy=xy)


@torch.no_grad()
def solve_M(B, data, iters=5, var=None, h=None):
    """Closed-form start: per spot, iteratively reweighted (Poisson weights 1 / (mu + var)) linear least squares of
    data [N, R, R] (or a track [N, T, R, R] with per-frame known brightness factors h [N, T], default 1) =
    sum_k c_k (h_t B_k) + bg (B: basis [N, 6, R, R] at the spot position). I = c_xx + c_yy + c_zz (tr M = 1),
    M = c / I. Returns M [N, 3, 3], I [N], bg [N]."""
    d = data if data.dim() == 4 else data[:, None]
    N, T = d.shape[:2]
    hh = torch.ones(N, T, device=d.device) if h is None else torch.as_tensor(h, dtype=d.dtype, device=d.device).expand(N, T)
    A = torch.cat([(hh[:, :, None, None, None] * B[:, None]),
                   torch.ones(N, T, 1, *B.shape[-2:], device=d.device)], 2)       # [N, T, 7, R, R]
    A = A.transpose(1, 2).flatten(2)                                               # [N, 7, T R R]
    y = d.flatten(1)
    w = torch.ones_like(y)
    rn = 0.0 if var is None else var
    for _ in range(iters):
        Aw = A * w[:, None]
        H = torch.einsum('nap,nbp->nab', Aw, A).double()
        H = H + 1e-9 * H.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(7, device=d.device, dtype=H.dtype)
        c = torch.linalg.solve(H, torch.einsum('nap,np->na', Aw, y).double()).float()
        mu = torch.einsum('na,nap->np', c, A)
        w = 1.0 / (mu.clamp_min(1e-3) + rn)
    I = c[:, 0] + c[:, 1] + c[:, 2]
    M = m6_to_M(c[:, :6] / torch.where(I.abs() > 1e-6, I, torch.ones_like(I))[:, None])
    return M, I, c[:, 6]


# ------------------------------------------------------------------ model and Jacobian
def _amp_m(p6, K):
    """(m [6], brightness factor f [T]) from p6 = (I, bg, v_x, v_y, v_z, g) (single emitter; K [T, 3, 3] or None)."""
    M = dipole_M(p6[2:5], p6[5])
    m = M_to_m6(M)
    f = excitation_factor(M[None], K) if K is not None else None
    return m, f


def model_and_jacobian(psf, theta, T=1, focus=None, h=None, K=None, xy=None, basis=None, *, zcoef_offset=None):
    """Expected images mu [N, T, R, R] and Jacobian [N, 9, T, R, R] for theta [N, 9] = (x, y, z, I, bg, v, g):
    frame t of a track has mu_t = I h_t f_t sum_k m_k B_k(x, y, z) + bg, f_t = 3 tr(M K_t) (1 without K).
    h: [N, T] or [T] known per-frame brightness factors (default 1); K: [T, 3, 3] or [N, T, 3, 3] (default None:
    no orientation dependence of the brightness). The v columns are d/dv at the given v (tangent to |v| = const).
    basis: precomputed basis_images(psf, theta[:, :3], focus, grad=True) (4 tensors [N, 6, R, R]; e.g. expanded from
    one position for many orientations)."""
    N = theta.shape[0]
    dev = theta.device
    if zcoef_offset is not None:
        psf = psf.with_field_offset(zcoef_offset)
    B, Bx, By, Bz = basis if basis is not None else basis_images(psf, theta[:, :3], focus, grad=True,
                                                                  xy=xy)
    hh = torch.ones(N, T, device=dev) if h is None else torch.as_tensor(h, dtype=theta.dtype, device=dev).expand(N, T)
    p6 = theta[:, 3:]
    Kb = None if K is None else torch.as_tensor(K, dtype=theta.dtype, device=dev).expand(N, T, 3, 3)
    if Kb is None:
        fm = lambda p: (_amp_m(p, None)[0],)
        m, = torch.func.vmap(fm)(p6)
        dm, = torch.func.vmap(torch.func.jacfwd(fm))(p6)                 # [N, 6, 6]
        f = torch.ones(N, T, device=dev); df = torch.zeros(N, T, 6, device=dev)
    else:
        fm = lambda p, k: _amp_m(p, k)
        m, f = torch.func.vmap(fm)(p6, Kb)
        dm, df = torch.func.vmap(torch.func.jacfwd(fm))(p6, Kb)        # [N, 6, 6], [N, T, 6]
    I, bg = p6[:, 0], p6[:, 1]
    img = torch.einsum('nk,nkuv->nuv', m, B)                              # [N, R, R]
    amp = I[:, None] * hh * f                                             # [N, T]
    mu = amp[..., None, None] * img[:, None] + bg[:, None, None, None]
    J = []
    for dB in (Bx, By, Bz):
        J.append(amp[..., None, None] * torch.einsum('nk,nkuv->nuv', m, dB)[:, None])
    J.append((hh * f)[..., None, None] * img[:, None])                    # I
    J.append(torch.ones_like(mu))                                         # bg
    for c in range(2, 6):                                                 # v_x, v_y, v_z, g
        dimg = torch.einsum('nk,nkuv->nuv', dm[:, :, c], B)
        J.append(amp[..., None, None] * dimg[:, None] + (I[:, None] * hh * df[..., c])[..., None, None] * img[:, None])
    return mu, torch.stack(J, 1)


def _clamp(th, psf, margin=0.05, z_limits=(0.0, 1.0), g_limits=(0.0, 1.0)):
    xm = psf.xy_limit - margin
    th[:, 0].clamp_(-xm, xm)
    th[:, 1].clamp_(-xm, xm)
    th[:, 2].clamp_(*z_limits)
    th[:, 3].clamp_min_(1.0)
    th[:, 4].clamp_min_(1e-3)
    th[:, 5:8] = unit(th[:, 5:8])
    th[:, 8].clamp_(*g_limits)
    return th


# ------------------------------------------------------------------ Fisher / CRLB
@torch.no_grad()
def fisher_orientation(psf, theta, T=1, focus=None, h=None, K=None, var=None, xy=None, basis=None):
    """Poisson Fisher matrix [N, 8, 8] of (x, y, z, I, bg, a, b, g) at theta [N, 9], (a, b) = tangent coordinates of
    mu along (e_theta, e_phi). Track of T frames as model_and_jacobian (Fisher summed over the frames). var: read-noise
    variance (e^2)."""
    th = theta.clone()
    th[:, 5:8] = unit(th[:, 5:8])
    mu, J = model_and_jacobian(psf, th, T, focus, h, K, xy, basis)
    e_th, e_ph = tangent_basis(th[:, 5:8])
    Jv = J[:, 5:8]
    Ja = torch.einsum('nc...,nc->n...', Jv, e_th)
    Jb = torch.einsum('nc...,nc->n...', Jv, e_ph)
    Jt = torch.stack([J[:, 0], J[:, 1], J[:, 2], J[:, 3], J[:, 4], Ja, Jb, J[:, 8]], 1).flatten(2)
    w = 1.0 / (mu.clamp_min(1e-6) + (0.0 if var is None else var)).flatten(1)
    return torch.einsum('nap,nbp->nab', Jt * w[:, None], Jt).double()


@torch.no_grad()
def crlb_orientation(psf, theta, T=1, focus=None, h=None, K=None, var=None, fixed=(), xy=None, basis=None):
    """CRLB of an orientation fit at theta [N, 9]. fixed: indices into (x, y, z, I, bg, a, b, g) held known (e.g. (7,)
    for a known g). Returns a dict of standard deviations [N]: x, y (px), z (um), I, bg, theta, phi (deg; phi =
    sd(b) / sin theta), inplane (deg, sd(b): the in-plane angular error, finite at the pole), g; 'cov' [N, 8, 8] and
    'corr' [N, 8, 8] in the (x, y, z, I, bg, a, b, g) coordinates (rad for a, b)."""
    F = fisher_orientation(psf, theta, T, focus, h, K, var, xy=xy, basis=basis)
    keep = [i for i in range(8) if i not in fixed]
    Fk = F[:, keep][:, :, keep]
    Ck = torch.linalg.pinv(Fk, hermitian=True)
    C = torch.full((len(theta), 8, 8), 0.0, dtype=F.dtype, device=F.device)
    for a, i in enumerate(keep):
        for b, j in enumerate(keep):
            C[:, i, j] = Ck[:, a, b]
    sd = torch.sqrt(torch.diagonal(C, dim1=1, dim2=2).clamp_min(0))
    corr = C / (sd[:, :, None] * sd[:, None, :]).clamp_min(1e-30)
    th, _ = angles(theta[:, 5:8], deg=False)
    out = dict(x=sd[:, 0], y=sd[:, 1], z=sd[:, 2], I=sd[:, 3], bg=sd[:, 4],
               theta=torch.rad2deg(sd[:, 5]), inplane=torch.rad2deg(sd[:, 6]),
               phi=torch.rad2deg(sd[:, 6] / torch.sin(th).double().clamp_min(1e-9)), g=sd[:, 7], cov=C, corr=corr)
    return {k: v.float() for k, v in out.items()}


# ------------------------------------------------------------------ fit
def _track_dims(rois):
    return (rois, False) if rois.dim() == 4 else (rois[:, None], True)


@torch.no_grad()
def fit_orientation(psf, rois, focus=None, h=None, K=None, var=None, iters=40, theta0=None, z_limits=(0.0, 0.4),
                    g_limits=(0.0, 1.0), batch=500, mirror=True, iso_iters=30, zstarts=4, zgrid=None,
                    xy=None, zcoef_offset=None):
    """Poisson MLE of theta = (x px, y px, z um, I, bg, v_x, v_y, v_z, g) per spot (rois [N, R, R]) or per track
    (rois [N, T, R, R]: frames sharing position, orientation, I and bg; per-frame known factors h [N, T] / [T] and
    optional excitation tensors K [T, 3, 3] or [N, T, 3, 3], see model_and_jacobian).

    Start (theta0 None): isotropic-PSF fit of the (summed) frames (torch_psf.fit.localize, zstarts heights) ->
    x, y, z, then the linear-M solution at that position (solve_M) -> (mu, g) (g clamped to [0.1, 0.9]); with
    mirror=True also the mirror start (mu_z -> -mu_z, the other tilt sign); the best likelihood wins.
    zgrid: extra starts at these heights (um; x, y, I, bg from the isotropic fit, (mu, g) from the linear-M
    solution at each height, each with its mirror): for PSFs whose isotropic fit is a poor guide to z (Vortex).
    xy [N, 2]: per-ROI field coordinates in camera pixels (fixed at each
    ROI's field position); the isotropic start ignores them.
    Returns dict: theta [N, 9] (v unit, canonical mu_z >= 0), nll [N], theta_deg / phi_deg [N] (angles(mu)), g [N],
    crlb (crlb_orientation at the estimate), M_lin [N, 3, 3] (the linear solution)."""
    d, single = _track_dims(rois)
    N, T = d.shape[:2]
    dev = d.device
    zl = tuple(z_limits)
    hh = None if h is None else torch.as_tensor(h, dtype=d.dtype, device=dev).expand(N, T)
    out_th, out_nll, out_Ml = [], [], []
    for s in range(0, N, batch):
        db = d[s:s + batch]
        n = len(db)
        hb = None if hh is None else hh[s:s + batch]
        field_xy = None if xy is None else xy[s:s + batch]
        model = psf if zcoef_offset is None else psf.with_field_offset(zcoef_offset[s:s + batch])
        Kb = None
        if K is not None:
            Kt = torch.as_tensor(K, dtype=d.dtype, device=dev)
            Kb = Kt.expand(N, T, 3, 3)[s:s + batch]
        if theta0 is not None:
            starts = [theta0[s:s + batch].to(dev).clone()]
            Ml = torch.zeros(n, 3, 3, device=dev)
        else:
            hm = torch.ones(n, T, device=dev) if hb is None else hb
            dsum = db.sum(1)
            th_iso, _ = localize(psf, dsum, iters=iso_iters, zstarts=zstarts, batch=batch, z_limits=zl)
            Bb = basis_images(model, th_iso[:, :3], focus, xy=field_xy)
            Ml, I0, bg0 = solve_M(Bb, db, h=hm)
            mu0, g0 = project_M(Ml)
            g0 = g0.clamp(0.1, 0.9)
            Kf = torch.ones(n, T, device=dev) if Kb is None else excitation_factor(dipole_M(mu0, g0)[:, None], Kb)
            I0 = th_iso[:, 3] / (hm * Kf).sum(1).clamp_min(1e-3)
            bg0 = th_iso[:, 4] / T
            base = torch.cat([th_iso[:, :3], I0[:, None], bg0[:, None], mu0, g0[:, None]], 1)
            starts = [base]
            for zg in (zgrid or ()):
                pz = th_iso[:, :3].clone(); pz[:, 2] = float(zg)
                Mz, _, _ = solve_M(basis_images(model, pz, focus, xy=field_xy), db, h=hm)
                mz, gz = project_M(Mz)
                st = base.clone(); st[:, 2] = float(zg); st[:, 5:8] = mz; st[:, 8] = gz.clamp(0.1, 0.9)
                starts.append(st)
            if mirror:
                for st in list(starts):
                    alt = st.clone(); alt[:, 7] = -alt[:, 7]
                    starts.append(alt)
        best, bnll = None, None
        for st in starts:
            th, nll = lm_poisson(lambda t: model_and_jacobian(model, t, T, focus, hb, Kb, field_xy), db, st, var=var,
                                 clamp=lambda t: _clamp(t, psf, z_limits=zl, g_limits=g_limits), iters=iters)
            if best is None:
                best, bnll = th, nll
            else:
                m = nll < bnll
                best[m], bnll[m] = th[m], nll[m]
        best[:, 5:8] = canonical(unit(best[:, 5:8]))
        out_th.append(best); out_nll.append(bnll); out_Ml.append(Ml)
    th = torch.cat(out_th)
    tdeg, pdeg = angles(th[:, 5:8])
    model = psf if zcoef_offset is None else psf.with_field_offset(zcoef_offset)
    cr = crlb_orientation(model, th, T, focus, hh, K, var, xy=xy)
    return dict(theta=th, nll=torch.cat(out_nll), theta_deg=tdeg, phi_deg=pdeg, g=th[:, 8], crlb=cr,
                M_lin=torch.cat(out_Ml))


def angle_error(mu_est, mu_true, deg=True):
    """Angle between the orientation axes (|mu_est . mu_true|, sign-free) [N]."""
    c = (unit(mu_est) * unit(mu_true)).sum(-1).abs().clamp(max=1)
    a = torch.acos(c)
    return torch.rad2deg(a) if deg else a


# ------------------------------------------------------------------ pipeline
class OrientationEstimator:
    """Orientation + 3D localization of sparse single molecules with a calibrated (e.g. Vortex) VectorialPSF.

    Pipeline: spot detection (torch_psf.spots.extract_spots; roisize = psf.roisize) -> per-spot Poisson MLE of
    (x, y, z, I, bg, mu, g) (fit_orientation: isotropic start, linear-M start and its mirror) -> CRLBs -> Dataset.

    psf: VectorialPSF (its focus_um is the focal plane of the fits; z = emitter height in the model's coordinate).
    camera: Camera / calibration json / None (data in photons). z_limits (um): the fitted z range (default
    psf.z_range); g_limits; iters (LM iterations per start); mirror (also the mirrored-tilt start); batch (spots per
    GPU batch); zgrid (extra z starts, see fit_orientation); field: torch_psf.field.FieldZernikes (field-dependent
    Zernikes; each spot gets the offsets at its ROI centre in camera px, so pass the full-frame coordinates or
    corners); threshold and detect_kw: detection (see PSF.localize). progress(done, total, n) / stop() hooks.

    estimate(data, frames=None) -> Dataset. data: a movie (path / array [T, H, W] / reader) or Spots;
    fit_rois(rois, corners=None, frame_ids=None) for ROIs [N, P, P] in photons (corners [N, 2] = (x0, y0)).
    Dataset fields (one row per spot): frame, pos [n, 3] px (image x, y; z = emitter height / lateral pixel size),
    crlb [n, 3] px, photons (signal photons in the ROI), bg, intensity (I: photons of an isotropic emitter at the
    coverslip, the model's scale), theta_deg (polar angle from the optical axis, 0..90), phi_deg (azimuth from +x
    towards +y, 0..360, of the axis with mu_z >= 0), g (order parameter, g2 of Hulleman et al.; within g_limits:
    with g_limits wider than [0, 1], e.g. (-0.5, 1.5), g is the unconstrained estimate whose spread matches crlb_g
    near the physical bounds), g_display (g clamped to [0, 1]), crlb_theta_deg,
    crlb_phi_deg, crlb_inplane_deg, crlb_g, crlb_photons, crlb_bg, mu [n, 3], loglik (-NLL), chi2 (mean
    (d - mu)^2 / mu per pixel), at_limit (x / y at the offset limit)."""

    def __init__(self, psf, camera=None, threshold=6.0, iters=40, z_limits=None, g_limits=(0.0, 1.0), mirror=True,
                 batch=500, zstarts=4, zgrid=None, field=None, device=None, progress=None, stop=None, **detect_kw):
        from torch_psf.estimators import _camera
        self.psf, self.camera = psf, _camera(camera)
        self.threshold, self.iters, self.mirror, self.batch, self.zstarts = threshold, iters, mirror, batch, zstarts
        self.zgrid, self.field = zgrid, field
        self.z_limits = tuple(z_limits) if z_limits is not None else tuple(psf.z_range)
        self.g_limits = tuple(g_limits)
        self.dev = psf.dev if device is None else torch.device(device)
        self.progress, self.stop, self.detect_kw = progress, stop, detect_kw
        self.report = {}

    @torch.no_grad()
    def fit_rois(self, rois, corners=None, frame_ids=None):
        """The fit part of estimate for ROIs [N, P, P] (photons) -> Dataset."""
        from torch_psf.dataset import Dataset
        from torch_psf.fit import at_limit
        psf = self.psf
        rois = torch.as_tensor(rois, dtype=torch.float32, device=self.dev)
        N, P = len(rois), psf.roisize
        cols = {k: [] for k in ('th', 'nll', 'chi2', 'ph', 'cr')}
        zo_all = None
        if self.field is not None:
            if corners is None:
                raise ValueError('a field model needs the ROI corners (field positions)')
            from torch_psf.field import FieldPSF
            psf = FieldPSF(psf, self.field)
            cxy = torch.as_tensor(corners, dtype=torch.float32, device=self.dev) + (P - 1) / 2
            with torch.no_grad():
                zo_all = cxy
        for s in range(0, N, self.batch):
            d = rois[s:s + self.batch]
            zo = None if zo_all is None else zo_all[s:s + self.batch]
            r = fit_orientation(psf, d, iters=self.iters, z_limits=self.z_limits, g_limits=self.g_limits,
                                batch=self.batch, mirror=self.mirror, zstarts=self.zstarts, zgrid=self.zgrid,
                                xy=zo)
            th = r['theta']
            mu, _ = model_and_jacobian(psf, th, xy=zo)
            mu = mu[:, 0]
            cols['th'].append(th); cols['nll'].append(r['nll']); cols['cr'].append(r['crlb'])
            cols['chi2'].append(((d - mu) ** 2 / mu.clamp_min(1e-6)).flatten(1).mean(1))
            cols['ph'].append((mu - th[:, 4, None, None]).flatten(1).sum(1))
            if self.progress is not None:
                self.progress(min(s + self.batch, N), N, min(s + self.batch, N))
            if self.stop is not None and self.stop():
                break
        n = sum(len(t) for t in cols['th'])
        if n == 0:
            th = torch.zeros(0, NPAR, device=self.dev)
            cr = {k: torch.zeros(0) for k in ('x', 'y', 'z', 'I', 'bg', 'theta', 'phi', 'inplane', 'g')}
            nll = chi2 = ph = torch.zeros(0)
        else:
            th = torch.cat(cols['th']); nll = torch.cat(cols['nll']); chi2 = torch.cat(cols['chi2'])
            ph = torch.cat(cols['ph'])
            cr = {k: torch.cat([c[k] for c in cols['cr']]) for k in cols['cr'][0] if k not in ('cov', 'corr')}
        h = (P - 1) / 2
        xy = th[:, :2].double()
        if corners is not None:
            xy = xy + torch.as_tensor(corners[:n], device=self.dev).double() + h
        px = psf.pixel_um
        tdeg, pdeg = angles(th[:, 5:8])
        cpu = lambda t: t.detach().cpu()
        fr = torch.as_tensor(frame_ids[:n]).cpu() if frame_ids is not None else torch.zeros(n, dtype=torch.int32)
        I = th[:, 3]
        fields = dict(frame=fr, pos=cpu(torch.cat([xy, th[:, 2:3].double() / px], 1)),
                      crlb=cpu(torch.stack([cr['x'], cr['y'], cr['z'] / px], 1).float()),
                      photons=cpu(ph), bg=cpu(th[:, 4]), intensity=cpu(I),
                      crlb_photons=cpu(cr['I'] * ph / I.clamp_min(1e-6)), crlb_bg=cpu(cr['bg']),
                      theta_deg=cpu(tdeg), phi_deg=cpu(pdeg), g=cpu(th[:, 8]), g_display=cpu(th[:, 8].clamp(0, 1)),
                      mu=cpu(th[:, 5:8]),
                      crlb_theta_deg=cpu(cr['theta']), crlb_phi_deg=cpu(cr['phi']),
                      crlb_inplane_deg=cpu(cr['inplane']), crlb_g=cpu(cr['g']),
                      loglik=cpu(-nll), chi2=cpu(chi2), at_limit=cpu(at_limit(th, psf)))
        meta = dict(method='torch_psf OrientationEstimator', n_dims=3, pixel_nm=1000.0 * float(px),
                    focus_um=float(psf.focus_um), z_limits=list(self.z_limits),
                    vortex=getattr(psf, 'vortex', None))
        if n:
            meta['chi2_median'] = float(chi2.median())
        return Dataset(fields, meta)

    def estimate(self, data, frames=None):
        """data: a movie (path / array [T, H, W] / reader; frames = (f0, f1)) or Spots -> Dataset (fit_rois for bare
        ROIs)."""
        from torch_psf.spots import Spots, extract_spots
        t0 = time.time()
        sp = data if isinstance(data, Spots) else extract_spots(
            data, self.psf.roisize, self.camera, frames, device=self.dev, threshold=self.threshold, stop=self.stop,
            **self.detect_kw)
        ds = self.fit_rois(sp.rois, sp.corner, sp.frame)
        ds['id'] = torch.arange(len(ds), dtype=torch.int64)
        ds.meta['n_spots'] = len(sp)
        self.report = dict(n=len(ds), seconds=time.time() - t0, chi2_median=ds.meta.get('chi2_median'))
        return ds
