"""Shared helpers for the real-data localization (EPFL SMLM challenge 2016, 3D astigmatic data of the Ries lab)."""
import json, math
import numpy as np, torch
from scipy.special import erf

DEV = 'cuda'


def gauss_patch(P, x, y, sx, sy, I, bg):
    """Pixel-integrated elliptical Gaussian on a P x P patch, (x, y) offset from the patch centre in px. numpy."""
    h = (P - 1) / 2
    o = np.arange(P) - h
    ex = 0.5 * (erf((o + 0.5 - x) / (np.sqrt(2) * sx)) - erf((o - 0.5 - x) / (np.sqrt(2) * sx)))
    ey = 0.5 * (erf((o + 0.5 - y) / (np.sqrt(2) * sy)) - erf((o - 0.5 - y) / (np.sqrt(2) * sy)))
    return I * ey[:, None] * ex[None, :] + bg


def fit_gauss(d, w=None):
    """Least-squares elliptical Gaussian (x, y, sx, sy, I, bg) of one patch d (numpy) -> params, rms residual."""
    from scipy.optimize import least_squares
    P = d.shape[0]
    bg0 = np.quantile(d, 0.1)
    I0 = max(d.sum() - bg0 * d.size, 1.0)
    h = (P - 1) / 2
    yy, xx = np.mgrid[:P, :P] - h
    dd = np.clip(d - bg0, 0, None)
    cx, cy = (dd * xx).sum() / dd.sum(), (dd * yy).sum() / dd.sum()
    p0 = [cx, cy, 2.0, 2.0, I0, bg0]
    f = lambda p: (gauss_patch(P, *p) - d).ravel()
    r = least_squares(f, p0, bounds=([-h, -h, 0.5, 0.5, 0, -np.inf], [h, h, P, P, np.inf, np.inf]))
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def gauss_knots(spl, zs, sx, sy):
    """Knot array of a pixel-integrated elliptical Gaussian with widths sx(z), sy(z) (arrays over the knot planes)."""
    u = (spl.u0 + torch.arange(spl.nu, dtype=torch.float64) / spl.S).numpy()
    K = np.zeros((spl.nz, spl.nv, spl.nu))
    for k in range(spl.nz):
        ex = 0.5 * (erf((u + 0.5) / (np.sqrt(2) * sx[k])) - erf((u - 0.5) / (np.sqrt(2) * sx[k])))
        ey = 0.5 * (erf((u + 0.5) / (np.sqrt(2) * sy[k])) - erf((u - 0.5) / (np.sqrt(2) * sy[k])))
        K[k] = ey[:, None] * ex[None, :]
    return torch.tensor(K, dtype=spl.dtype, device=spl.dev)


def transfer_knots(Ksrc, c_src, c_dst, S):
    """Knots of a spline with stored half width c_src -> half width c_dst (same S, hz, z grid): crop or zero-pad."""
    d = (c_src - c_dst) * S
    if d == 0:
        return Ksrc.clone()
    if d > 0:
        return Ksrc[:, d:-d, d:-d].clone()
    return torch.nn.functional.pad(Ksrc, (-d, -d, -d, -d))


def spline_widths(spl, zs, P=None):
    """Second-moment widths sx, sy (px) and x, y centroid of the spline PSF evaluated at x = y = 0 for heights zs."""
    off = torch.zeros(len(zs), 3, device=spl.dev)
    off[:, 2] = torch.as_tensor(zs, device=spl.dev, dtype=torch.float32)
    g = spl.evaluate(off).double().cpu().numpy()
    Pp = g.shape[1]
    o = np.arange(Pp) - (Pp - 1) / 2
    gs = np.clip(g, 0, None)
    n = gs.sum((1, 2))
    cx = (gs.sum(1) * o).sum(1) / n
    cy = (gs.sum(2) * o).sum(1) / n
    sx = np.sqrt((gs.sum(1) * (o[None] - cx[:, None]) ** 2).sum(1) / n)
    sy = np.sqrt((gs.sum(2) * (o[None] - cy[:, None]) ** 2).sum(1) / n)
    return sx, sy, cx, cy, g.sum((1, 2))


def localize_w(spl, d, w, th0, iters=30, nb=4000, fixed=(), lam0=1.0):
    """As torch_convspline.knotpsf.localize (Levenberg-Marquardt Poisson MLE of x, y, z, I, bg on the spline) with a
    pixel weight mask w [N, P, P] in {0, 1}: pixels with w = 0 (outside the frame) do not enter the likelihood."""
    from torch_convspline.knotpsf import _clamp
    out, onll = [], []
    with torch.no_grad():
        for s in range(0, len(d), nb):
            dd, ww = d[s:s + nb], w[s:s + nb]
            th = _clamp(th0[s:s + nb].clone().to(spl.dtype), spl)
            fx = list(fixed)
            lam = torch.full((len(dd),), lam0, device=dd.device)

            def model(t):
                g, gx, gy, gz = spl.evaluate(t[:, :3], grad=True)
                I = t[:, 3, None, None]
                return I * g + t[:, 4, None, None], torch.stack([I * gx, I * gy, I * gz, g, torch.ones_like(g)], 1)

            def nllf(mu):
                mu = mu.clamp_min(1e-6)
                return (ww * (mu - dd * torch.log(mu))).flatten(1).sum(1)

            mu, J = model(th)
            nll = nllf(mu)
            for _ in range(iters):
                mu = mu.clamp_min(1e-6)
                if fx:
                    J[:, fx] = 0
                Jw = J * ww[:, None]
                gr = (Jw * (dd / mu - 1)[:, None]).flatten(2).sum(2)
                H = torch.einsum('nap,nbp->nab', Jw.flatten(2) / mu.flatten(1)[:, None], J.flatten(2))
                dg = torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-12)
                H = H + torch.diag_embed(dg * lam[:, None])
                if fx:
                    H[:, fx, fx] = 1.0
                step = torch.linalg.solve(H.double(), gr.double()[..., None])[..., 0].to(th.dtype)
                tn = _clamp(th + step, spl)
                mun, Jn = model(tn)
                nn_ = nllf(mun)
                ok = nn_ < nll
                th[ok], mu[ok], J[ok], nll[ok] = tn[ok], mun[ok], Jn[ok], nn_[ok]
                lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-4, 1e6)
            out.append(th); onll.append(nll)
    return torch.cat(out), torch.cat(onll)


def crlb_w(spl, th, w):
    """Poisson CRLB (std of x, y, z, I, bg) with a pixel mask."""
    with torch.no_grad():
        g, gx, gy, gz = spl.evaluate(th[:, :3], grad=True)
        I = th[:, 3, None, None]
        mu = (I * g + th[:, 4, None, None]).clamp_min(1e-6)
        J = torch.stack([I * gx, I * gy, I * gz, g, torch.ones_like(g)], 1).flatten(2)
        F = torch.einsum('nap,nbp->nab', J * (w.flatten(1) / mu.flatten(1))[:, None], J)
        F = F + 1e-9 * torch.eye(5, device=F.device)
        return torch.sqrt(torch.diagonal(torch.linalg.inv(F.double()), dim1=1, dim2=2).clamp_min(0)).float()


class PageStack:
    """Frames of a TIFF whose pages are uncompressed but not equally spaced (MicroManager OME stacks with per-frame
    metadata): one memmap of the file plus the data offset of every page."""

    def __init__(self, f):
        import tifffile, os
        self.os = os
        cache = f + '.offsets.npy'
        cache = os.path.join('/mnt/data/SMLM_challenge/_offsets', os.path.basename(cache))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with tifffile.TiffFile(f) as t:
            p0 = t.pages[0]
            self.H, self.W = p0.shape
            self.dtype = np.dtype(t.byteorder + p0.dtype.char)
            if os.path.exists(cache):
                self.offs = np.load(cache)
            else:
                t.pages.cache = False
                offs = []
                for i in range(len(t.pages)):
                    p = t.pages[i]
                    assert len(p.dataoffsets) == 1 and p.is_contiguous
                    offs.append(p.dataoffsets[0])
                self.offs = np.array(offs, dtype=np.int64)
                np.save(cache, self.offs)
        self.mm = np.memmap(f, dtype=np.uint8, mode="r")      # byte map: page offsets can be odd
        self.shape = (len(self.offs), self.H, self.W)
        self.n = self.H * self.W

    def __getitem__(self, s):
        if isinstance(s, int):
            o = int(self.offs[s])
            return np.array(self.mm[o:o + self.n * self.dtype.itemsize]).view(self.dtype).reshape(self.H, self.W)
        idx = range(*s.indices(self.shape[0]))
        return np.stack([self[i] for i in idx]) if len(idx) else np.zeros((0, self.H, self.W), self.dtype)


def open_stack(f):
    import tifffile
    try:
        return tifffile.memmap(f)
    except ValueError:
        return PageStack(f)
