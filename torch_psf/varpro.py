"""Variable projection helper: closed-form (iteratively reweighted linear least squares) solution of the linear
parameters of a track sample,
    mu[b, t, :] = I[b, t] psf[b, :] + bg[b, group[b, t]],
per-frame intensities and a background shared per group (e.g. per illumination-pattern angle). The weights are
1 / (mu + rn2) (Poisson + read noise). Nonlinear parameters then get gradients with I / bg held fixed."""
import torch


@torch.no_grad()
def solve_linear(psf, data, mask, group, rn2, ngroups=2, iters=3, I_min=0.0):
    """psf [B, P] (or [B, T, P]: per-frame PSF), data [B, T, P], mask [B, T] bool, group [B, T] long in [0, ngroups),
    rn2: read-noise variance (e^2). Returns I [B, T], bg [B, ngroups]."""
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
