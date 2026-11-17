"""Fisher information of the Zernike coefficients from second-order cross-cumulants, with the density profiled out.

Data: g = the zero-lag covariances G(i, j) of all pixel pairs (i, j = i + d), d in D(R), both pixels in a P x P patch.
Model: g = B(theta) rho with B[(i, j), m] = h_m(i) h_m(j) (m = grid point, see model.py).
Noise: Sigma = Cov(g_hat) for T frames, from the moments of the movie (no simulation needed), with C(tau) the pixel
covariance at lag tau = r(tau) S + delta_tau0 N (S = signal covariance, N = diag noise variance in photons^2:
F2 * mean + read variance; F2 = 2 for EMCCD), all emitters sharing the normalized autocorrelation r(tau):
    T Sigma[(ij), (kl)] = alpha (S_ik S_jl + S_il S_jk) + (S_ik N_jl + N_ik S_jl + S_il N_jk + N_il S_jk)
                          + N_ik N_jl + N_il N_jk                                   (Gaussian / Wick part)
                          + F2 [delta_ik X(ij, l) + delta_jl X(ij, k) + delta_il X(ij, k) + delta_jk X(ij, l)]
                          + sum_m c4_m h_m(i) h_m(j) h_m(k) h_m(l)                 (non-Gaussian blinking)
with X(ij, l) = sum_m k3_m h_m(i) h_m(j) h_m(l) (shot noise modulated by the blinking, third order, zero lag),
alpha = sum_tau r(tau)^2, k3_m = E[(a - mu)^3] and c4_m = LRV[(a - mu)^2] - 2 alpha Var[a]^2 summed over the
emitters at m (a = eps s, photons per frame). Validated against Monte-Carlo movies and a block bootstrap
(study/sofi/validate_noise.py).

Profiled Fisher matrix of theta (Zernikes + nuisances such as a common z shift):
    J = A_w^T (I - P_B) A_w,   A_w = L^-1 dg/dtheta,  B_w = L^-1 B,  Sigma = L L^T,
P_B = projector onto range(B_w) (rank cut-off `tol` relative to the largest singular value squared). support=None
profiles rho at every grid point (arbitrary object; positivity ignored, conservative); support = bool mask profiles
only those grid points (rho known to be zero elsewhere, e.g. the true emitter positions: an optimistic bound).
"""
import torch
from torch_psf.sofi.cumulants import separations


def patch_matrix(h, P):
    """Hm [P*P, Ng] (Ng = Np (P + Q - 1)^2): column m = phase PSF of grid point m placed in the P x P patch."""
    Np, Q, _ = h.shape
    c = (Q - 1) // 2
    Pm = P + Q - 1
    dev, dt = h.device, h.dtype
    # grid point (p, jy, jx) <-> n = (jy - c, jx - c); pixel i = n - c + u = (jy - 2c + uy, ...)
    jy, jx = torch.meshgrid(torch.arange(Pm, device=dev), torch.arange(Pm, device=dev), indexing='ij')
    u = torch.arange(Q, device=dev)
    yy = jy.reshape(-1, 1, 1) - 2 * c + u[None, :, None]
    xx = jx.reshape(-1, 1, 1) - 2 * c + u[None, None, :]
    ok = (yy >= 0) & (yy < P) & (xx >= 0) & (xx < P)                       # [Pm^2, Q, Q]
    Hm = torch.zeros(P * P, Np, Pm * Pm, device=dev, dtype=dt)
    for p in range(Np):
        vals = h[p][None].expand(Pm * Pm, Q, Q)
        cols = torch.arange(Pm * Pm, device=dev)[:, None, None].expand(-1, Q, Q)
        Hm[:, p].index_put_(((yy * P + xx)[ok], cols[ok]), vals[ok], accumulate=True)
    return Hm.reshape(P * P, Np * Pm * Pm)


def pairs(P, R, device='cuda'):
    """Pair index arrays I, J (flat pixel indices, both inside the P x P patch) and the separation index k."""
    D = separations(R)
    I, J, K = [], [], []
    y, x = torch.meshgrid(torch.arange(P, device=device), torch.arange(P, device=device), indexing='ij')
    for k, (dy, dx) in enumerate(D):
        ok = (y + dy >= 0) & (y + dy < P) & (x + dx >= 0) & (x + dx < P)
        I.append((y * P + x)[ok])
        J.append(((y + dy) * P + x + dx)[ok])
        K.append(torch.full((int(ok.sum()),), k, device=device))
    return torch.cat(I), torch.cat(J), torch.cat(K), D


def noise_cov(Hm, rho, k3, c4, Nvar, I, J, T, alpha, F2=1.0, chunk=1024):
    """Sigma [M, M] (see the module docstring), assembled in row chunks (memory ~ M^2 + chunk * M * few).
    rho, k3, c4 [Ng] per grid point; Nvar [P*P] noise variance."""
    S = (Hm * rho[None]) @ Hm.T
    B = Hm[I] * Hm[J]                                     # [M, Ng]
    M = len(I)
    SI, SJ = S[I], S[J]                                   # [M, P*P]
    SIt_I, SIt_J = SI[:, I], SI[:, J]                     # rows a, columns b: S[i_a, k_b], S[i_a, l_b]
    del SI
    SJt_I, SJt_J = SJ[:, I], SJ[:, J]                     # S[j_a, k_b], S[j_a, l_b]
    del SJ
    X = None if k3 is None else (B * k3[None]) @ Hm.T     # [M, P*P]: X[a, l] = sum_m k3 h(i_a) h(j_a) h(l)
    NI, NJ = Nvar[I], Nvar[J]
    Sig = torch.empty(M, M, device=S.device, dtype=S.dtype)
    for a0 in range(0, M, chunk):
        a1 = min(M, a0 + chunk)
        Ia, Ja = I[a0:a1, None], J[a0:a1, None]
        Sik, Sil, Sjk, Sjl = SIt_I[a0:a1], SIt_J[a0:a1], SJt_I[a0:a1], SJt_J[a0:a1]
        Eik, Eil = (Ia == I[None]).to(S.dtype), (Ia == J[None]).to(S.dtype)
        Ejk, Ejl = (Ja == I[None]).to(S.dtype), (Ja == J[None]).to(S.dtype)
        nI, nJ = NI[a0:a1, None], NJ[a0:a1, None]
        blk = alpha * (Sik * Sjl + Sil * Sjk)
        blk += Sik * nJ * Ejl + nI * Eik * Sjl + Sil * nJ * Ejk + nI * Eil * Sjk
        blk += nI * nJ * (Eik * Ejl + Eil * Ejk)
        if X is not None:
            XI, XJ = X[a0:a1][:, I], X[a0:a1][:, J]
            blk += F2 * (Eik * XJ + Ejl * XI + Eil * XI + Ejk * XJ)
        Sig[a0:a1] = blk
    del SIt_I, SIt_J, SJt_I, SJt_J
    if c4 is not None:
        Sig += (B * c4[None]) @ B.T
    Sig /= T
    return Sig, B, S


def _proj_fisher(Aw, Bw, tol):
    """Aw^T (I - P_B) Aw with P_B the projector onto range(Bw) (rank cut-off tol * largest eigenvalue)."""
    M, Ng = Bw.shape
    if Ng <= M:
        e, V = torch.linalg.eigh(Bw.T @ Bw)
        keep = e > tol * e.max()
        Qb = (Bw @ V[:, keep]) / torch.sqrt(e[keep])[None]
    else:
        e, V = torch.linalg.eigh(Bw @ Bw.T)
        keep = e > tol * e.max()
        Qb = V[:, keep]
    QA = Qb.T @ Aw
    return Aw.T @ Aw - QA.T @ QA, int(keep.sum())


def fisher(h, dh, rho, P, R, T, alpha, Nvar, k3=None, c4=None, F2=1.0, support=None, tol=1e-10, extra=None):
    """Profiled Fisher matrix J [ntheta, ntheta] for a P x P patch (fp64 recommended).
    h [Np, Q, Q], dh [ntheta, Np, Q, Q], rho / k3 / c4 [Np, P + Q - 1, P + Q - 1] grid maps, Nvar [P, P].
    Returns dict(J, J_known (rho known: no profiling), rank, M, Ng)."""
    dev = h.device
    I, J, _, D = pairs(P, R, dev)
    Hm = patch_matrix(h, P)
    dHm = torch.stack([patch_matrix(d, P) for d in dh])            # [nt, P*P, Ng]
    rv = rho.reshape(-1)
    Sig, B, S = noise_cov(Hm, rv, None if k3 is None else k3.reshape(-1), None if c4 is None else c4.reshape(-1),
                          Nvar.reshape(-1), I, J, T, alpha, F2)
    A = torch.stack([((dHm[t][I] * Hm[J] + Hm[I] * dHm[t][J]) @ rv) for t in range(len(dh))], 1)   # [M, nt]
    L = torch.linalg.cholesky(Sig)
    Aw = torch.linalg.solve_triangular(L, A, upper=False)
    sel = torch.ones(B.shape[1], dtype=torch.bool, device=dev) if support is None else support.reshape(-1)
    Bw = torch.linalg.solve_triangular(L, B[:, sel], upper=False)
    Jp, rank = _proj_fisher(Aw, Bw, tol)
    return dict(J=Jp, J_known=Aw.T @ Aw, rank=rank, M=len(I), Ng=int(sel.sum()))


def crlb(J, names=None, cutoff=1e12):
    """sqrt(diag(J^-1)) (joint bound, all parameters unknown), inf where J is (numerically) singular."""
    e, V = torch.linalg.eigh(J)
    good = e > e.max() / cutoff
    Jinv = (V[:, good] / e[good][None]) @ V[:, good].T
    out = torch.sqrt(torch.diag(Jinv).clamp(min=0))
    bad = (V[:, ~good] ** 2).sum(1) > 1e-3
    out[bad] = float('inf')
    return out
