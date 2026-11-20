"""Grid model of the second-order cumulants: G_d = sum_p rho_p (*) K_{d,p}.

The density rho (= eps^2 Var[s] summed over the emitters at a grid point) lives on a sub-pixel grid: integer pixel n
plus a phase delta = (iy, ix) / os in [0, 1)^2 and, for thick samples, one of Nz depth layers. Phase p = (layer,
iy, ix), Np = Nz * os^2. h_p [Q, Q] is the pixel-integrated PSF of an emitter at ROI-centre offset delta_p and depth
z_p, so the emitter at grid point (p, n) puts h_p[u] into frame pixel n - c + u (c = (Q - 1) / 2). Then
    G_d(i) = sum_p sum_n rho_p(n) h_p(i - n + c) h_p(i + d - n + c) = sum_p (rho_p (*) K_{d,p})(i),
    K_{d,p}(u) = h_p(u) h_p(u + d).
rho maps are stored for n in [-c, H - 1 + c] x [-c, W - 1 + c], i.e. as [Np, H + Q - 1, W + Q - 1].
"""
import itertools
import torch
from torch.func import functional_call, jacfwd


class GridPSF:
    """Phase PSFs h_p and their derivatives with respect to the Zernike coefficients (and a common z shift)."""

    def __init__(self, psf, Q, os=2, z_layers=(0.0,), focus=None):
        self.psf = psf if psf.roisize == Q else psf.resized(Q)
        self.Q, self.os, self.c = Q, os, (Q - 1) // 2
        self.z_layers = [float(z) for z in z_layers]
        self.focus = psf.focus_um if focus is None else float(focus)
        dev = self.psf.dev
        ph = [(z, iy / os, ix / os) for z, iy, ix in itertools.product(self.z_layers, range(os), range(os))]
        self.pos = torch.tensor([[dx, dy, z] for z, dy, dx in ph], dtype=torch.float32, device=dev)
        self.Np = len(ph)
        self.dev = dev

    def h(self, zcoef=None, dz=0.0):
        p = self.pos.clone()
        p[:, 2] += dz
        if zcoef is None:
            with torch.no_grad():
                return self.psf.render(p, focus=self.focus)
        return functional_call(self.psf, {'zcoef': zcoef}, (p, self.focus))

    def jac(self, zcoef=None):
        """h [Np, Q, Q] and derivatives dh [nz + 1, Np, Q, Q]: d/d zcoef_j (waves) and d/d dz (um, common z shift)."""
        zc = (self.psf.zcoef.detach() if zcoef is None else zcoef).clone()
        h = self.h(zc).detach()
        dzc = jacfwd(lambda c: self.h(c))(zc).detach()                     # [Np, Q, Q, nz]
        dzz = jacfwd(lambda s: self.h(zc, s))(torch.zeros((), device=self.dev)).detach()   # [Np, Q, Q]
        d = torch.cat([dzc.permute(3, 0, 1, 2), dzz[None]])
        return h, d


def kernels(h, D):
    """K [Nd, Np, Q, Q]: K_{d,p}(u) = h_p(u) h_p(u + d) (0 where u + d leaves the patch)."""
    Np, Q, _ = h.shape
    out = torch.zeros(len(D), Np, Q, Q, device=h.device, dtype=h.dtype)
    for k, (dy, dx) in enumerate(D):
        ys, ye = max(0, -dy), min(Q, Q - dy)
        xs, xe = max(0, -dx), min(Q, Q - dx)
        out[k, :, ys:ye, xs:xe] = h[:, ys:ye, xs:xe] * h[:, ys + dy:ye + dy, xs + dx:xe + dx]
    return out


def model_G(rho, K):
    """Expected G [Nd, H, W] from rho [Np, H + Q - 1, W + Q - 1] and K [Nd, Np, Q, Q] (see the module docstring)."""
    return torch.nn.functional.conv2d(rho[None], torch.flip(K, (-2, -1)))[0]


def model_G_adjoint(R, K):
    """Adjoint of model_G: R [Nd, H, W] -> [Np, H + Q - 1, W + Q - 1]."""
    return torch.nn.functional.conv_transpose2d(R[None], torch.flip(K, (-2, -1)))[0]


def rho_from_emitters(x, y, z, weight, H, W, Q, os, z_layers):
    """Grid density [Np, H + Q - 1, W + Q - 1] from emitters at grid positions (x = column, y = row px; z on the
    layers): the sum of `weight` (e.g. eps^2 Var[s]) per grid point. Emitters outside the map are dropped."""
    c = (Q - 1) // 2
    dev = x.device
    zl = torch.tensor(z_layers, device=dev, dtype=x.dtype)
    li = torch.argmin((z[:, None] - zl[None]).abs(), 1)
    nx, ny = torch.floor(x + 1e-4), torch.floor(y + 1e-4)
    ix = torch.round((x - nx) * os).long().clamp(0, os - 1)
    iy = torch.round((y - ny) * os).long().clamp(0, os - 1)
    p = (li * os + iy) * os + ix
    jy, jx = ny.long() + c, nx.long() + c
    Hm, Wm = H + Q - 1, W + Q - 1
    ok = (jy >= 0) & (jy < Hm) & (jx >= 0) & (jx < Wm)
    rho = torch.zeros(len(z_layers) * os * os, Hm, Wm, device=dev, dtype=weight.dtype)
    rho.index_put_((p[ok], jy[ok], jx[ok]), weight[ok], accumulate=True)
    return rho
