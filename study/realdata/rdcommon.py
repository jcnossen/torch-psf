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
    g = spl.render(off).double().cpu().numpy()
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
    """torch_psf.localize-style Levenberg-Marquardt Poisson MLE of x, y, z, I, bg on the spline with a pixel weight
    mask w [N, P, P] in {0, 1}: pixels with w = 0 (outside the frame) do not enter the likelihood."""
    from torch_psf.fit import lm_poisson, model_and_jacobian, clamp_theta
    out, onll = [], []
    for s in range(0, len(d), nb):
        th, nll = lm_poisson(lambda t: model_and_jacobian(spl, t), d[s:s + nb], th0[s:s + nb].to(spl.dtype),
                             weight=w[s:s + nb], clamp=lambda t: clamp_theta(t, spl), fixed=fixed, iters=iters,
                             lam0=lam0)
        out.append(th); onll.append(nll)
    return torch.cat(out), torch.cat(onll)


def crlb_w(spl, th, w):
    """Poisson CRLB (std of x, y, z, I, bg) with a pixel mask."""
    from torch_psf import crlb
    return crlb(spl, th, weight=w, ridge=1e-9)


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
