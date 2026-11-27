"""Training loss of SMLMNet: the Gaussian-mixture likelihood of DECODE (Speiser et al. 2021) with a count term.

Per image, every pixel k is one mixture component with weight p_k / sum_k p_k, mean mu_k and either independent
Gaussians per parameter (cov=None) or, for cov='full_xyz', a full x, y, z covariance (Cholesky factor xyz_cholesky
of the marginal SDs and the partial correlations) times independent Gaussians for bg and photons. The loss of an
image is

  - sum_j log sum_k w_k N(theta_j; mu_k, Sigma_k)      (the targets j of the image)
  - count_loss_weight x log N(n_targets; sum_k p_k, sum_k p_k (1 - p_k))

without the constant -0.5 log(2 pi) per parameter of the mixture term. The training loss is the mean over images.
"""
import math
import torch
import torch.nn as nn
from torch_psf.nn.model import xyz_cholesky


def targets_tensor(active, xyz, bg, photons):
    """Targets [B, S, 1 + n]: columns active (0/1), x, y[, z], bg, photons (the parameter order of SMLMNet).
    active [B, S], xyz [B, S, 2 or 3] (x, y px image coordinates; z um), bg [B, S], photons [B, S]."""
    return torch.cat([active[..., None], xyz, bg[..., None], photons[..., None]], dim=-1)


def _diag_log_prob(values, mu, sigma, logw):
    """values [B, S, n], mu / sigma [B, n, K], logw [B, K] -> log mixture density [B, S]."""
    d = values[:, :, :, None] - mu[:, None]
    s = sigma[:, None]
    log_normal = -d ** 2 / (2 * s ** 2) - torch.log(s)
    return torch.logsumexp(logw[:, None] + log_normal.sum(2), dim=2)


def _xyz_log_prob(values, mu, sigma, rho, logw):
    """full_xyz: values [B, S, n], mu / sigma [B, n, K], rho [B, 3, K], logw [B, K] -> [B, S]. One triangular solve per
    pixel with all targets of the image as right-hand sides."""
    L = xyz_cholesky(sigma[:, :3].transpose(1, 2), rho.transpose(1, 2))          # [B, K, 3, 3]
    d = values[:, :, :, None] - mu[:, None]                                       # [B, S, n, K]
    rhs = d[:, :, :3].permute(0, 3, 2, 1).contiguous()                            # [B, K, 3, S]
    w = torch.linalg.solve_triangular(L, rhs, upper=False)
    log_xyz = -0.5 * w.square().sum(2).transpose(1, 2) - L.diagonal(dim1=-2, dim2=-1).log().sum(-1)[:, None]
    rest = sigma[:, None, 3:]
    log_rest = -0.5 * (d[:, :, 3:] / rest).square() - rest.log()
    return torch.logsumexp(logw[:, None] + log_xyz + log_rest.sum(2), -1)


class GMMLoss(nn.Module):
    """loss(output [B, C, H, W] of SMLMNet, targets [B, S, 1 + n]) -> mean loss over the images (module docstring).

    n_params: n (5 for 3D, 4 for 2D). cov: None or 'full_xyz' (the output then ends with the three correlation
    channels). count_loss_weight: weight of the count term (0 = off). components: 0 = every pixel is a component,
    else only the `components` pixels of highest p (the weights stay normalised over all pixels)."""
    def __init__(self, n_params, cov=None, count_loss_weight=0.01, components=0, epsilon=1e-8):
        super().__init__()
        self.n_params, self.cov = int(n_params), cov
        self.count_loss_weight, self.components, self.epsilon = float(count_loss_weight), int(components), epsilon

    def per_image(self, output, targets):
        """-> loss [B] (one value per image)."""
        n = self.n_params
        rho = None
        if self.cov:
            rho = output[:, -3:]
            output = output[:, :-3]
            dtype = torch.float64 if output.dtype == torch.float64 else torch.float32
            output, targets, rho = output.to(dtype), targets.to(dtype), rho.to(dtype)
        b, c, h, w = output.shape
        if c != 1 + 2 * n:
            raise ValueError(f'{c} output channels for {n} parameters')
        if targets.shape[-1] != n + 1 or targets.shape[0] != b:
            raise ValueError(f'targets {tuple(targets.shape)}: [batch, max targets, 1 + {n}]')
        mask, values = targets[..., 0], targets[..., 1:]
        prob = torch.clamp(output[:, 0], 1e-20).reshape(b, h * w)
        mu = output[:, 1:n + 1].reshape(b, n, h * w)
        sigma = torch.clamp(output[:, n + 1:], 1e-10).reshape(b, n, h * w)
        logw = torch.log(prob) - torch.log(prob.sum(-1, keepdim=True))
        if rho is not None:
            rho = rho.reshape(b, 3, h * w)
        if self.components:
            top, idx = torch.topk(logw[:, None].expand(-1, n, -1), self.components, dim=-1, largest=True, sorted=False)
            mu, sigma, logw = mu.gather(2, idx), sigma.gather(2, idx), top[:, 0]
            if rho is not None:
                rho = rho.gather(2, idx[:, :3])
        if rho is None:
            logp = _diag_log_prob(values, mu, sigma, logw)
        else:
            with torch.autocast(output.device.type, enabled=False):
                logp = _xyz_log_prob(values, mu, sigma, rho, logw)
        loss = -torch.sum(logp * mask, -1)
        if self.count_loss_weight > 0:
            mean = prob.sum(-1)
            var = torch.clamp((prob - prob ** 2).sum(-1), min=self.epsilon)
            count = mask.sum(-1)
            loss = loss - self.count_loss_weight * (-0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var)
                                                    - 0.5 * (count - mean) ** 2 / var)
        return loss

    def forward(self, output, targets):
        return self.per_image(output, targets).mean()
