"""Shared code of the real-data in-situ PSF test (NPC-A647-3D): a vectorial pupil model with anisotropic pixels and a
fitted wavelength, the movie spline geometry, a Levenberg-Marquardt localizer with the movie z limits, and the knot-route
gradient for any vectorial parameters (data-agnostic versions of study/insitu/fit.py)."""
import math, json, time, os
import numpy as np, torch
from torch_psf import VectorialPSF
from torch_psf.zernike import zernike
from torch_psf import SplinePSF
from torch_psf.fit import poisson_nll, lm_poisson, model_and_jacobian
from torch_psf.estimation import knot_loss
from rdcommon import transfer_knots

DEV = 'cuda'
# data set switch: environment variable INSITU_DS = npc (default, the original NPC test) or tubulin
DS = os.environ.get('INSITU_DS', 'npc').lower()
if DS == 'npc':
    D = '/mnt/data/SMLM_challenge/NPC-A647-3D'
    GAIN, OFFSET = 10.9, 295.0
    LOCS_CSV = f'{D}/locs/NPC-A647-3D_locs.csv'
    MOVIES = [f'{D}/NPC-A647-3D-stack_{i}.tif' for i in (1, 2)]
    PX0, PY0 = 0.135, 0.142          # um, Cspline-submission scale (start values of the bead vectorial fit)
elif DS == 'tubulin':
    D = '/mnt/data/SMLM_challenge/Tubulin-A647-3D'
    GAIN, OFFSET = 17.0, 400.0       # README_Tubulin.md
    LOCS_CSV = f'{D}/locs/Tubulin-A647-3D_locs.csv'
    MOVIES = [f'{D}/Tubulin-A647-3D-stacks_{i}.tif' for i in (1, 2, 3, 4)]
    PX0, PY0 = 0.117, 0.1182
else:
    raise ValueError(f'unknown INSITU_DS {DS}')
OUT = os.environ.get('INSITU_OUT', f'{D}/insitu')   # INSITU_OUT: write / read the in-situ files elsewhere
P, XMAX, ZLIM = 17, 3.0, 0.8


class AVecPSF(VectorialPSF):
    """VectorialPSF with separate pixel sizes in x and y (pxy, um; the cylindrical lens can give a different
    magnification in x and y: the image is sampled on a rectangular grid, the pupil stays round) and the wavelength as
    a tensor (lam, um) so that it can be fitted. Everything else as VectorialPSF."""

    def __init__(self, roisize, px=0.135, py=0.142, lam=0.68, **kw):
        kw.setdefault('pixel_um', 0.5 * (px + py))
        super().__init__(roisize, wavelength_um=lam, **kw)
        self.pxy = torch.nn.Parameter(torch.tensor([px, py], device=self.dev), requires_grad=False)
        self.lam = torch.nn.Parameter(torch.tensor(float(lam), device=self.dev), requires_grad=False)
        # lateral Gaussian blur (um) of the image: finite bead size or any other blur (vibration, camera); 0 = none
        self.blur = torch.nn.Parameter(torch.zeros((), device=self.dev), requires_grad=False)
        self.use_blur = False
        self.poly = 0.0           # um: > 0 averages lam - poly, lam, lam + poly (weights 1/4, 1/2, 1/4): broad emission band

    def pupil_fields(self, lam=None):
        NA = self.NA
        k0 = 2 * math.pi / (self.lam if lam is None else lam)
        s_imm = NA * self.rho / self.n_imm
        c_imm = torch.sqrt((1 - s_imm ** 2).clamp(min=1e-6))
        s_med = (NA * self.rho / self.n_med).to(torch.complex64)
        # small imaginary part: keeps the gradient finite when a pupil sample sits at the critical angle while NA moves
        c_med = torch.sqrt(1 - s_med ** 2 + 1e-5j)
        c_med = torch.where(c_med.imag < 0, -c_med, c_med)
        n1, n2 = self.n_med.to(torch.complex64), self.n_imm
        ts = 2 * n1 * c_med / (n1 * c_med + n2 * c_imm)
        tp = 2 * n1 * c_med / (n2 * c_med + n1 * c_imm)
        cphi, sphi = torch.cos(self.phi), torch.sin(self.phi)
        apod = self.mask / torch.sqrt(c_imm) * (n2 / n1) * c_imm / c_med
        r2 = self.rho ** 2
        apod = apod * torch.exp(self.apod[0] * r2 + self.apod[1] * r2 ** 2 + self.apod[2] * r2 ** 3)
        apod = apod * self.pupil_amp
        Ex_x = tp * c_med * cphi * cphi + ts * sphi * sphi
        Ey_x = tp * c_med * cphi * sphi - ts * sphi * cphi
        Ex_y = tp * c_med * sphi * cphi - ts * cphi * sphi
        Ey_y = tp * c_med * sphi * sphi + ts * cphi * cphi
        Ex_z = -tp * s_med * cphi
        Ey_z = -tp * s_med * sphi
        Pf = torch.stack([Ex_x, Ey_x, Ex_y, Ey_y, Ex_z, Ey_z]) * apod
        kx = k0 * NA * self.px
        ky = k0 * NA * self.py
        return Pf, k0 * self.n_med * c_med, k0 * self.n_imm * c_imm, kx, ky

    def forward(self, pos, focus, zcoef_offset=None, intensity=None, bg=None):
        if self.poly > 0:
            psf = sum(w * self._mono(pos, focus, self.lam + dl) for dl, w in ((-self.poly, 0.25), (0.0, 0.5), (self.poly, 0.25)))
        else:
            psf = self._mono(pos, focus, self.lam)
        if self.use_blur:
            psf = blur_img(psf, self.blur, self.pxy[0], self.pxy[1])
        if intensity is not None:
            psf = psf * intensity[:, None, None]
        if bg is not None:
            psf = psf + bg[:, None, None]
        return psf

    def _mono(self, pos, focus, lam):
        """One wavelength; the aberrations (zcoef, pupil_phase in waves at self.lam) are fixed path differences."""
        B = pos.shape[0]
        Pf, kz_med, kz_imm, kx, ky = self.pupil_fields(lam)
        zc = self.zcoef[None].expand(B, -1)
        W = (torch.einsum('bj,jnm->bnm', zc, self.Z) + self.pupil_phase[None]) * (2 * math.pi) * (self.lam / lam)
        pxs, pys = self.pxy[0], self.pxy[1]
        x = pos[:, 0, None, None] * pxs
        y = pos[:, 1, None, None] * pys
        ze = pos[:, 2, None, None]
        f = focus[:, None, None] if torch.is_tensor(focus) and focus.dim() > 0 else focus
        ph = torch.exp(1j * (W - kx * x - ky * y - kz_imm * f)) * torch.exp(1j * kz_med * ze)
        Pb = Pf[None] * ph[:, None]
        R, o = self.roisize, self.os
        ns = R * o
        g = (torch.arange(ns, device=self.dev, dtype=torch.float32) - (ns - 1) / 2) / o
        kx1, ky1 = kx[0], ky[:, 0]
        Fx = torch.exp(1j * (g * pxs)[:, None] * kx1[None, :])
        Fy = torch.exp(1j * (g * pys)[:, None] * ky1[None, :])
        E = Fy @ Pb.to(torch.complex64) @ Fx.T
        I = (E.real ** 2 + E.imag ** 2).sum(1) / 3
        dk = (kx1[1] - kx1[0]) * (ky1[1] - ky1[0])
        energy = (Pf.real ** 2 + Pf.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        I = I * (dk / (2 * math.pi) ** 2) ** 2
        I = I.reshape(B, R, o, R, o).mean((2, 4)) * pxs * pys
        return I / energy


def blur_img(g, sig, px, py):
    """Lateral Gaussian blur (sigma um) of pixel images: multiplication of their DFT (zero-padded to twice the size)."""
    n = g.shape[-1]
    G = torch.fft.rfft2(torch.nn.functional.pad(g, (0, n, 0, n)))
    fy = torch.fft.fftfreq(2 * n, device=g.device)[:, None] / py
    fx = torch.fft.rfftfreq(2 * n, device=g.device)[None, :] / px
    H = torch.exp(-2 * math.pi ** 2 * sig ** 2 * (fx ** 2 + fy ** 2))
    return torch.fft.irfft2(G * H, s=(2 * n, 2 * n))[..., :n, :n]


def make_vec(roisize, st, npupil=64, oversample=3, use_blur=False):
    """AVecPSF from a state dict saved by insitu_vecbeads.py."""
    v = AVecPSF(roisize, px=st['pxy'][0], py=st['pxy'][1], lam=st['lam'], NA=st['NA'], n_imm=st.get('n_imm', 1.518),
                n_med=st.get('n_med', 1.33), zernike_indices=tuple(st['noll']), npupil=npupil, oversample=oversample,
                device=DEV)
    with torch.no_grad():
        v.zcoef.copy_(torch.as_tensor(st['zcoef'], dtype=torch.float32))
        if 'apod' in st:
            v.apod.copy_(torch.as_tensor(st['apod'], dtype=torch.float32))
        if 'pupil_phase' in st:
            v.pupil_phase.copy_(torch.as_tensor(st['pupil_phase'], dtype=torch.float32))
        if 'pupil_amp' in st:
            v.pupil_amp.copy_(torch.as_tensor(st['pupil_amp'], dtype=torch.float32))
        v.poly = float(st.get('poly', 0.0))
        if use_blur and st.get('blur_sigma', 0) > 0:
            v.blur.fill_(float(st['blur_sigma']))
            v.use_blur = True
    return v


# ---------------------------------------------------------------------------------------------- movie spline
def movie_spline():
    """Same knot grid as the bead calibration (S 2, hz 50 nm, z -0.8...0.8 um + 4 pad planes), 17 px patch, +-3 px."""
    cal = torch.load(f'{D}/locs/calib/psf_spline.pt')
    spl = SplinePSF(P, (cal['zlo'], cal['zhi']), oversample=cal['S'], dz_um=cal['hz'], zpad=cal['zpad'], xy_limit=XMAX, device=DEV)
    assert spl.nz == cal['nz'] and abs(spl.z0 - cal['z0']) < 1e-9
    Kb = transfer_knots(cal['knots'].to(DEV), cal['c'], spl.c, cal['S'])
    return spl, Kb


def pos_fn_maker(f0, s, D=0.0, kappa=1.0):
    """knot z (um, stage-equivalent) -> (emitter height D, focus f0 + s z): the PSF of an emitter at a fixed height D
    above the coverslip seen with the focus moved by z (a z-stack at depth D). D = 0 is the bead geometry."""
    def pos_fn(z):
        return torch.full_like(z, float(D)), float(f0) + float(s) * float(kappa) * z
    return pos_fn


def knots_from_vec(spl, big, pos_fn):
    return spl.sample(big, pos_fn, chunk=spl.nz)


# ---------------------------------------------------------------------------------------------- localization
def _clamp(th, zlim=ZLIM, xmax=XMAX):
    th[:, 0].clamp_(-xmax + 0.05, xmax - 0.05)
    th[:, 1].clamp_(-xmax + 0.05, xmax - 0.05)
    th[:, 2].clamp_(-zlim, zlim)
    th[:, 3].clamp_min_(1.0)
    th[:, 4].clamp_min_(1e-3)
    return th


def _model(spl, t):
    return model_and_jacobian(spl, t)


@torch.no_grad()
def lm(spl, d, th, iters, nb=8000):
    """Levenberg-Marquardt Poisson MLE (torch_psf.lm_poisson with the movie z limits and mu >= 1e-4), patches on the
    GPU or CPU (moved per batch)."""
    out, onll = [], []
    for s in range(0, len(d), nb):
        dd = d[s:s + nb].to(DEV, non_blocking=True)
        t, nll = lm_poisson(lambda t_: _model(spl, t_), dd, th[s:s + nb].to(DEV), clamp=_clamp, iters=iters,
                            mu_min=1e-4)
        out.append(t); onll.append(nll)
    return torch.cat(out), torch.cat(onll)


def init_theta(d, z, frac=0.97):
    d = d.to(DEV)
    q = d.flatten(1).sort(1).values
    bg0 = q[:, int(0.2 * q.shape[1])].clamp_min(1.0)
    I0 = (d - bg0[:, None, None]).flatten(1).sum(1).clamp_min(50.0) / frac
    return torch.stack([torch.zeros_like(I0), torch.zeros_like(I0), torch.full_like(I0, z), I0, bg0], 1)


@torch.no_grad()
def localize_full(spl, d, starts=(-0.45, 0.0, 0.45), iters=12, iters2=20, nb=20000):
    """3 starting heights (loc.py settings), best continued. d may live on the CPU; processed in blocks of nb."""
    ths, nlls = [], []
    for s in range(0, len(d), nb):
        dd = d[s:s + nb].to(DEV)
        best, bnll = None, None
        for z in starts:
            th, nll = lm(spl, dd, init_theta(dd, z), iters)
            if best is None:
                best, bnll = th, nll
            else:
                m = nll < bnll
                best[m], bnll[m] = th[m], nll[m]
        th, nll = lm(spl, dd, best, iters2)
        ths.append(th); nlls.append(nll)
    return torch.cat(ths), torch.cat(nlls)


@torch.no_grad()
def fit_stats(spl, d, th, nb=20000):
    """Per spot: chi2/pixel with the EMCCD variance 2 mu, Poisson deviance of the halved data and model (EMCCD-correct
    log-likelihood ratio against the saturated model, summed over the patch), CRLB x sqrt(2) of (x, y, z, I, bg)."""
    c2s, devs, crs = [], [], []
    for s in range(0, len(d), nb):
        dd = d[s:s + nb].to(DEV); t = th[s:s + nb].to(DEV)
        mu, J = _model(spl, t)
        mu = mu.clamp_min(1e-4)
        c2s.append(((dd - mu) ** 2 / (2 * mu)).flatten(1).mean(1))
        dh, mh = dd.clamp_min(0) / 2, mu / 2
        dev = torch.where(dh > 0, dh * torch.log(dh.clamp_min(1e-9) / mh), torch.zeros_like(dh)) - (dh - mh)
        devs.append(dev.flatten(1).sum(1))
        J = J.flatten(2)
        F = torch.einsum('nap,nbp->nab', J / mu.flatten(1)[:, None], J) + 1e-9 * torch.eye(5, device=DEV)
        crs.append(torch.sqrt(torch.diagonal(torch.linalg.inv(F.double()), dim1=1, dim2=2).clamp_min(0)).float()
                   * math.sqrt(2.0))
    return torch.cat(c2s), torch.cat(devs), torch.cat(crs)


def limit_fractions(th):
    xy = (th[:, :2].abs().max(1).values > XMAX - 0.06)
    zl = (th[:, 2].abs() >= ZLIM - 0.005)
    return dict(xy=float(xy.float().mean()), z=float(zl.float().mean()), any=float((xy | zl).float().mean()))


# ---------------------------------------------------------------------------------------------- knot route
def knot_grad(spl, d, th, nb=20000):
    """d NLL / d knots at fixed emitter parameters (Poisson NLL of the photoelectron data), summed over all patches."""
    K = spl.knots.clone().requires_grad_(True)
    tot = 0.0
    for s in range(0, len(d), nb):
        tot += knot_loss(spl, K, d[s:s + nb].to(DEV), th[s:s + nb].to(DEV), batch=nb)
    return K.grad, tot


class Adam:
    def __init__(self, params, lrs, b1=0.9, b2=0.999):
        self.p, self.lr, self.b1, self.b2 = params, lrs, b1, b2
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, g):
        self.t += 1
        for k in self.p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] = self.p[k] - self.lr[k] * mh / (vh.sqrt() + 1e-12)


def jdump(obj, f):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        raise TypeError(type(o))
    json.dump(obj, open(f, 'w'), indent=1, default=conv)
