"""
KnotPSF: a 3D PSF stored as Catmull-Rom knots on an oversampled grid, evaluated with a stride-S grouped conv2d.

The pixel-integrated PSF is shift invariant: patch[i, j] = g(o_j - x, o_i - y, z), with o the pixel offset from the
patch centre and (x, y) the emitter offset in px. g is stored on a uniform grid: 1/S px in u and v, hz um in z:
knots [nz, nv, nu]. Output pixels are S knots apart and share the fractional knot offset, so a P x P patch is the
4x4x4 Catmull-Rom kernel applied with stride S to the block of knots around the emitter.

The knots are either
  - free parameters (model-free PSF; the patch is linear in the knots, d patch / d knots by autograd), or
  - a function of the parameters of a vectorial model (stack_from_vectorial), in which case a gradient with respect
    to the knots is passed to the vectorial parameters with one vector-Jacobian product (vjp_vectorial).
"""
import math
import torch
from torch_convspline.conv_spline import geometry_matrix


class KnotPSF:
    def __init__(self, P, zlo, zhi, S=3, hz=0.04, zpad=2, xmax=1.5, device='cuda', dtype=torch.float32):
        assert P % 2 == 1
        self.P, self.S, self.hz, self.xmax = P, S, float(hz), float(xmax)
        self.h = (P - 1) // 2
        self.dev, self.dtype = torch.device(device), dtype
        self.c = int(math.ceil(self.h + xmax)) + 2            # half width (px) of the stored PSF
        self.Pl = 2 * self.c + 1
        self.nz = int(round((zhi - zlo) / hz)) + 1 + 2 * zpad
        self.zlo, self.zhi = float(zlo), float(zhi)
        self.z0 = zlo - zpad * hz
        self.nv = self.nu = self.Pl * S
        self.u0 = -(self.c * S + S - 1) / S                   # u (px) of knot 0
        self.oS0 = float((-self.h - self.u0) * S)             # knot coordinate of patch pixel 0 for x = 0
        self.Lw = (P - 1) * S + 4
        r = lambda n: torch.arange(n, device=self.dev)
        self.pat = (r(4)[:, None, None] * self.nv * self.nu + r(self.Lw)[None, :, None] * self.nu
                    + r(self.Lw)[None, None, :]).reshape(-1)
        self.G = geometry_matrix(self.dev).to(dtype)
        self.e = torch.arange(4, device=self.dev)
        self.knots = torch.zeros(self.nz, self.nv, self.nu, device=self.dev, dtype=dtype)

    @property
    def zs(self):
        return self.z0 + self.hz * torch.arange(self.nz, device=self.dev, dtype=torch.float64)

    # ------------------------------------------------------------------ evaluation
    def _axis(self, t, deriv):
        p = t[..., None] ** self.e
        if deriv:
            p = self.e * t[..., None] ** (self.e - 1).clamp_min(0)
        return p @ self.G

    def evaluate(self, off, grad=False, knots=None):
        """off [N, 3] = (x px, y px, z um) -> patches [N, P, P]; with grad also d/dx, d/dy (per px), d/dz (per um).
        Differentiable with respect to knots."""
        K = self.knots if knots is None else knots
        N, P, S, Lw = off.shape[0], self.P, self.S, self.Lw
        off = off.to(self.dtype)
        tz = ((off[:, 2] - self.z0) / self.hz)
        zin = ((tz >= 1) & (tz <= self.nz - 2)).to(self.dtype)[:, None, None]
        tz = tz.clamp(1, self.nz - 2 - 1e-4)
        t = torch.stack([tz, self.oS0 - off[:, 1].clamp(-self.xmax, self.xmax) * S,
                         self.oS0 - off[:, 0].clamp(-self.xmax, self.xmax) * S], 1)
        ix = torch.floor(t)
        fr = t - ix
        i0 = ix.long() - 1
        base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
        sub = K.reshape(-1)[base[:, None] + self.pat[None]].reshape(1, N * 4, Lw, Lw)
        wa = self._axis(fr, False)                             # [N, 3, 4]
        w = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
        if not grad:
            return torch.nn.functional.conv2d(sub, w, groups=N, stride=S).view(N, P, P)
        da = self._axis(fr, True)
        wz = da[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
        wy = wa[:, 0, :, None, None] * da[:, 1, None, :, None] * wa[:, 2, None, None, :]
        wx = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * da[:, 2, None, None, :]
        W = torch.stack([w, wx, wy, wz], 1).view(-1, 4, 4, 4)
        r = torch.nn.functional.conv2d(sub, W, groups=N, stride=S).view(N, 4, P, P)
        return r[:, 0], -S * r[:, 1], -S * r[:, 2], zin * r[:, 3] / self.hz

    def evaluate_batched(self, off, nb=4000):
        return torch.cat([self.evaluate(off[s:s + nb]) for s in range(0, len(off), nb)])

    # ------------------------------------------------------------------ knots from a vectorial model
    def _vec_positions(self):
        a = torch.arange(self.S, device=self.dev, dtype=torch.float64) / self.S
        B, A = torch.meshgrid(a, a, indexing='ij')
        return A.reshape(-1).float(), B.reshape(-1).float()

    def stack_planes(self, big, params, zi0, zi1, pos_fn):
        """Knot planes zi0..zi1-1 from the vectorial model `big` (roisize Pl) with parameters `params` (a dict for
        torch.func.functional_call; autograd-able). pos_fn(z_knot [n]) -> (z_emitter [n], focus [n]) maps the knot z
        coordinate to the arguments of the vectorial model."""
        S, Pl = self.S, self.Pl
        A, B = self._vec_positions()
        z = self.zs[zi0:zi1].float()
        ze, f = pos_fn(z)
        n = zi1 - zi0
        pos = torch.stack([A[None].expand(n, -1), B[None].expand(n, -1), ze[:, None].expand(-1, S * S)], -1)
        imgs = torch.func.functional_call(big, params, (pos.reshape(-1, 3), f[:, None].expand(-1, S * S).reshape(-1)))
        imgs = imgs.reshape(n, S, S, Pl, Pl)                   # [z, b, a, i, j]
        return imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(n, Pl * S, Pl * S)

    def stack_from_vectorial(self, big, pos_fn, params=None, chunk=4):
        params = dict(big.named_parameters()) if params is None else params
        with torch.no_grad():
            return torch.cat([self.stack_planes(big, params, z, min(z + chunk, self.nz), pos_fn)
                              for z in range(0, self.nz, chunk)]).to(self.dtype)

    def vjp_vectorial(self, big, pos_fn, gK, names=('zcoef',), chunk=4):
        """(d knots / d theta)^T gK for the named vectorial parameters: one build + backward, chunked over z."""
        base = {k: v.detach() for k, v in big.named_parameters()}
        out = {k: torch.zeros_like(base[k]) for k in names}
        for z in range(0, self.nz, chunk):
            z1 = min(z + chunk, self.nz)
            prm = dict(base)
            for k in names:
                prm[k] = base[k].clone().requires_grad_(True)
            st = self.stack_planes(big, prm, z, z1, pos_fn)
            gr = torch.autograd.grad(st, [prm[k] for k in names], gK[z:z1].float())
            for k, g in zip(names, gr):
                out[k] += g
        return out


# ---------------------------------------------------------------------- Poisson likelihood and localization
def poisson_nll(d, mu):
    """Sum over pixels of mu - d log mu (constant in d dropped) -> [N]."""
    mu = mu.clamp_min(1e-6)
    return (mu - d * torch.log(mu)).flatten(1).sum(1)


def _clamp(th, spl):
    th[:, 0].clamp_(-spl.xmax + 0.05, spl.xmax - 0.05)
    th[:, 1].clamp_(-spl.xmax + 0.05, spl.xmax - 0.05)
    th[:, 2].clamp_(spl.z0 + spl.hz, spl.z0 + (spl.nz - 2) * spl.hz)
    th[:, 3].clamp_min_(1.0)
    th[:, 4].clamp_min_(1e-3)
    return th


@torch.no_grad()
def localize(spl, d, th0=None, iters=30, zstarts=5, nb=3000, fixed=()):
    """Maximum-likelihood (x, y, z, I, bg) per patch d [N, P, P] with a damped Newton (Levenberg-Marquardt) iteration
    on the spline. Without th0: zstarts starting heights, the best likelihood wins. fixed: indices of theta held at th0.
    Returns theta [N, 5], nll [N]."""
    if th0 is None:
        best, bnll = None, None
        for z in torch.linspace(spl.zlo, spl.zhi, zstarts + 2)[1:-1]:
            bg = d.flatten(1).quantile(0.2, 1)
            I = (d.flatten(1).sum(1) - bg * d[0].numel()).clamp_min(50.0)
            t0 = torch.stack([torch.zeros_like(I), torch.zeros_like(I), torch.full_like(I, float(z)), I, bg], 1)
            th, nll = localize(spl, d, t0, iters, nb=nb)
            if best is None:
                best, bnll = th, nll
            else:
                m = nll < bnll
                best[m], bnll[m] = th[m], nll[m]
        return best, bnll
    out, onll = [], []
    for s in range(0, len(d), nb):
        dd, th = d[s:s + nb], _clamp(th0[s:s + nb].clone().to(spl.dtype), spl)
        fx = list(fixed)
        lam = torch.full((len(dd),), 1.0, device=dd.device)

        def model(t):
            g, gx, gy, gz = spl.evaluate(t[:, :3], grad=True)
            I = t[:, 3, None, None]
            return I * g + t[:, 4, None, None], torch.stack([I * gx, I * gy, I * gz, g, torch.ones_like(g)], 1)

        mu, J = model(th)
        nll = poisson_nll(dd, mu)
        for _ in range(iters):
            mu = mu.clamp_min(1e-6)
            if fx:
                J[:, fx] = 0
            gr = (J * (dd / mu - 1)[:, None]).flatten(2).sum(2)
            H = torch.einsum('nap,nbp->nab', J.flatten(2) / mu.flatten(1)[:, None], J.flatten(2))
            dg = torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-12)
            H = H + torch.diag_embed(dg * lam[:, None])
            if fx:
                H[:, fx, fx] = 1.0
            step = torch.linalg.solve(H.double(), gr.double()[..., None])[..., 0].to(th.dtype)
            tn = _clamp(th + step, spl)
            mun, Jn = model(tn)
            nn_ = poisson_nll(dd, mun)
            ok = nn_ < nll
            th[ok], mu[ok], J[ok], nll[ok] = tn[ok], mun[ok], Jn[ok], nn_[ok]
            lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-4, 1e6)
        out.append(th); onll.append(nll)
    return torch.cat(out), torch.cat(onll)


@torch.no_grad()
def crlb(spl, th):
    """Cramer-Rao lower bound [N, 5] (std) of (x, y, z, I, bg) at theta."""
    g, gx, gy, gz = spl.evaluate(th[:, :3], grad=True)
    I = th[:, 3, None, None]
    mu = (I * g + th[:, 4, None, None]).clamp_min(1e-6)
    J = torch.stack([I * gx, I * gy, I * gz, g, torch.ones_like(g)], 1).flatten(2)
    F = torch.einsum('nap,nbp->nab', J / mu.flatten(1)[:, None], J)
    return torch.sqrt(torch.diagonal(torch.linalg.inv(F.double()), dim1=1, dim2=2)).float()
