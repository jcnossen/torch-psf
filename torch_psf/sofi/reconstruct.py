"""SOFI image reconstruction from the second-order cross-cumulants, with a known (estimated) PSF, and the standard
cross-cumulant SOFI2 image as a reference.

PSF-aware reconstruction (`reconstruct`): the model of torch_psf.sofi.model, G_d = sum_p rho_p (*) h_p h_p(. + d),
is a forward model of the SOFI data; its density rho (eps^2 Var[s] per sub-pixel grid point: the brightness-weighted
emitter density) is the image. rho >= 0 is fitted by weighted least squares on all separations 0 < |d| <= R at once,
with the true (aberrated) PSF, by FISTA, with optional priors: l1 (sparsity; soft threshold) and a quadratic
smoothness penalty on the fine image (gradient energy). The second-order kernels h h(. + d) have twice the bandwidth
of h, so the attainable resolution is ~2x the widefield one without the brightness / noise nonlinearity of higher
orders; positivity / sparsity extrapolate beyond. This is the approach of SPARCOM (Solomon et al.), here with the
estimated vectorial PSF and the cumulant noise weights.

Standard SOFI2 (`sofi2_xc`): every cross-cumulant G(r, d) is put on the virtual pixel at the midpoint r + d / 2 of a
2x grid, corrected by the distance factor exp(-|d|^2 / (4 sigma^2)) of a Gaussian PSF (Dertinger et al. 2010), and
averaged; optional Fourier reweighting (Wiener filter with the OTF of the squared Gaussian PSF).
"""
import math
import torch
from torch_psf.sofi.model import GridPSF, kernels, model_G, model_G_adjoint


def to_fine(rho, os, nz=1):
    """Phase maps rho [Np = nz os^2, Hm, Wm] -> fine images [nz, Hm os, Wm os] (fine index j os + i = grid point
    j + i / os of the map)."""
    Np, Hm, Wm = rho.shape
    return rho.reshape(nz, os, os, Hm, Wm).permute(0, 3, 1, 4, 2).reshape(nz, Hm * os, Wm * os)


def from_fine(img, os):
    nz, Hf, Wf = img.shape
    Hm, Wm = Hf // os, Wf // os
    return img.reshape(nz, Hm, os, Wm, os).permute(0, 2, 4, 1, 3).reshape(nz * os * os, Hm, Wm)


def _grad_energy(f):
    """sum of squared forward differences of fine images [nz, H, W] and its gradient (closed form)."""
    dy, dx = f[:, 1:] - f[:, :-1], f[:, :, 1:] - f[:, :, :-1]
    g = torch.zeros_like(f)
    g[:, 1:] += 2 * dy
    g[:, :-1] -= 2 * dy
    g[:, :, 1:] += 2 * dx
    g[:, :, :-1] -= 2 * dx
    return float((dy ** 2).sum() + (dx ** 2).sum()), g


@torch.no_grad()
def reconstruct(G, mask, psf, var=None, os=4, Q=13, z_layers=(0.0,), focus=None, dz=0.0, D=None, l1=0.0,
                smooth=0.0, iters=500, crop=True, verbose=False):
    """PSF-aware SOFI reconstruction. G [Nd, H, W] cross-cumulants (CumulantAccumulator.result()['G']), mask, var
    (their variances, e.g. a smoothed block bootstrap; None = per-separation variance), psf: VectorialPSF (the
    estimate), D: the separations of G. l1: sparsity weight relative to the largest data gradient (0 .. ~0.1);
    smooth: gradient-energy weight relative to the data term curvature (0 .. ~1). Returns dict(image [nz, H os, W os]
    (cropped to the frame when crop) or the full map, rho, loss)."""
    if D is None:
        raise ValueError('pass D (the separations of G)')
    nz = len(z_layers)
    gp = GridPSF(psf, Q, os, z_layers, focus)
    h = gp.h(psf.zcoef.detach(), float(dz)).double()
    K = kernels(h, D)
    Nd, H, W = G.shape
    Gh = torch.where(mask, G, torch.zeros_like(G)).double()
    if var is None:
        sc = torch.stack([Gh[k][mask[k]].var() for k in range(Nd)])
        w = mask.double() / sc[:, None, None]
    else:
        w = torch.where(mask & torch.isfinite(var), 1.0 / var.clamp(min=1e-30), torch.zeros_like(var)).double()
    w = w / w[mask].mean()
    shape = (gp.Np, H + Q - 1, W + Q - 1)
    x = torch.rand(shape, device=G.device, dtype=torch.float64)
    for _ in range(30):
        y = model_G_adjoint(w * model_G(x, K), K)
        lam = y.norm() / x.norm()
        x = y / y.norm()
    Ld = 2 * float(lam) * 1.05
    g0 = 2 * model_G_adjoint(w * Gh, K)
    tau = l1 * float(g0.max())                 # l1 weight in data-gradient units
    mu = smooth * Ld / 16.0                    # gradient energy: Hessian norm <= 16 mu
    L = Ld + 16.0 * mu
    rho = torch.zeros(shape, device=G.device, dtype=torch.float64)
    xk, yk, t = rho, rho, 1.0
    for it in range(iters):
        r = model_G(yk, K) - Gh
        grad = 2 * model_G_adjoint(w * r, K)
        if mu > 0:
            _, gs = _grad_energy(to_fine(yk, os, nz))
            grad = grad + mu * from_fine(gs, os)
        xn = (yk - (grad + tau) / L).clamp(min=0)
        tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
        yk = xn + (t - 1) / tn * (xn - xk)
        xk, t = xn, tn
        if verbose and it % 100 == 0:
            print(f'  recon it {it} data {float((w * (model_G(xk, K) - Gh) ** 2).sum()):.4e}')
    loss = float((w * (model_G(xk, K) - Gh) ** 2).sum())
    img = to_fine(xk, os, nz)
    if crop:
        c = (Q - 1) // 2
        img = img[:, c * os:(c + H) * os, c * os:(c + W) * os]
    return dict(image=img, rho=xk, loss=loss)


@torch.no_grad()
def sofi2_xc(G, mask, D, sigma_px, rmax=2.0, reweight=None):
    """Standard cross-cumulant SOFI2 on a 2x grid [2H - 1, 2W - 1] (virtual pixels at the pair midpoints, distance
    factor of a Gaussian PSF of sigma_px). reweight: None, or the Wiener regularization (relative to the OTF peak,
    e.g. 0.02) of a Fourier reweighting with the OTF of the squared PSF (Gaussian sigma / sqrt(2), on the 2x grid)."""
    Nd, H, W = G.shape
    out = torch.zeros(2 * H - 1, 2 * W - 1, device=G.device, dtype=torch.float64)
    cnt = torch.zeros_like(out)
    for k, (dy, dx) in enumerate(D):
        d2 = dy * dy + dx * dx
        if d2 > rmax * rmax:
            continue
        g = torch.where(mask[k], G[k], torch.zeros_like(G[k])).double() * math.exp(d2 / (4 * sigma_px ** 2))
        m = mask[k].double()
        ys, xs = 2 * torch.arange(H, device=G.device) + dy, 2 * torch.arange(W, device=G.device) + dx
        oky, okx = (ys >= 0) & (ys < 2 * H - 1), (xs >= 0) & (xs < 2 * W - 1)
        yy, xx = ys[oky], xs[okx]
        out[yy[:, None], xx[None, :]] += g[oky][:, okx]
        cnt[yy[:, None], xx[None, :]] += m[oky][:, okx]
    img = out / cnt.clamp(min=1)
    if reweight:
        s = sigma_px * 2 / math.sqrt(2)          # sigma of h^2 in 2x-grid pixels
        fy = torch.fft.fftfreq(img.shape[0], device=G.device, dtype=torch.float64)[:, None]
        fx = torch.fft.fftfreq(img.shape[1], device=G.device, dtype=torch.float64)[None, :]
        otf = torch.exp(-2 * math.pi ** 2 * s ** 2 * (fx ** 2 + fy ** 2))
        img = torch.fft.ifft2(torch.fft.fft2(img) * otf / (otf ** 2 + reweight)).real
    return img
