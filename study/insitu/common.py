"""Simulation for the in-situ PSF study: vectorial ground-truth PSFs, a bead z-stack and an SMLM movie.

Optics: NA 1.49 oil objective (n 1.518), sample in water (n 1.33), 680 nm, 108 nm pixels, astigmatism for 3D.
  bead stack : beads ON the coverslip (z_e = 0), the stage (focus f) is scanned.
  SMLM movie : focus fixed at F0, emitters at heights z_e in [0, 0.8] um above the coverslip.
Because of the index mismatch these are different PSFs, and during the experiment the aberrations have also moved
away from the calibration (DZ_INSITU), as in a real sample.

Scenarios
  A  the pupil is a Noll 5-21 Zernike series (a vectorial Zernike model can represent the truth exactly).
  B  as A plus pupil structure outside that series: a high-order phase screen (Noll 22-66, HO_RMS waves rms) and an
     amplitude defect (a partially obscured patch of the pupil).
"""
import math
import torch
from torch_convspline.vectorial import VectorialPSF, zernike

DEV = 'cuda'
P = 15                      # fit ROI
PR = 23                     # render ROI for the movie
F0 = 0.45                   # um, focus during the SMLM acquisition
ZR = (0.0, 0.8)             # um, emitter heights
ALPHA = 1.33 / 1.518        # paraxial focal-shift factor used to label the bead stack with a nominal emitter z
OPT = dict(pixelsize=0.108, NA=1.49, n_imm=1.518, n_med=1.33, wavelength=0.68, npupil=64, oversample=3)
# The data are simulated with a finer pupil sampling than the fitted model uses, so that the simulation is not the
# fitted model itself, and the pupil structure of scenario B does not lie on the pixel grid of the fitted pupil map.
SIM_OPT = dict(OPT, npupil=128, oversample=4)
NOLL = tuple(range(5, 22))
HO_RMS = 0.06


def zc_bead():
    g = torch.Generator().manual_seed(3)
    zc = torch.randn(len(NOLL), generator=g) * 0.012
    zc[NOLL.index(6)] = 0.10                               # astigmatism for 3D
    return zc


DZ_INSITU = {5: 0.02, 7: 0.03, 8: -0.02, 11: 0.04}        # waves rms: aberration change bead -> experiment


def zc_insitu():
    zc = zc_bead()
    for j, v in DZ_INSITU.items():
        zc[NOLL.index(j)] += v
    return zc


def make_vec(roisize, scn='A', cond='insitu', ho_rms=HO_RMS, device=DEV):
    """Ground-truth vectorial PSF for scenario scn and condition cond ('bead' or 'insitu')."""
    psf = VectorialPSF(roisize, zernike_indices=NOLL, device=device, **SIM_OPT)
    with torch.no_grad():
        psf.zcoef.copy_((zc_bead() if cond == 'bead' else zc_insitu()).to(device))
        if scn == 'B':
            g = torch.Generator().manual_seed(11)
            js = list(range(22, 67))
            co = torch.randn(len(js), generator=g)
            co = co / co.norm() * ho_rms
            rho, phi = psf.rho.clamp(max=1), psf.phi
            psf.pupil_phase.copy_(sum(c * zernike(j, rho, phi) for c, j in zip(co.tolist(), js)) * psf.mask)
            psf.pupil_amp.copy_(1 - 0.6 * torch.exp(-((psf.px - 0.35) ** 2 + (psf.py + 0.25) ** 2) / (2 * 0.15 ** 2)))
    return psf


def model_vec(roisize, zc=None, device=DEV):
    """The fitted model class: vectorial PSF with Noll 5-21 Zernikes only (no free pupil maps)."""
    psf = VectorialPSF(roisize, zernike_indices=NOLL, device=device, **OPT)
    if zc is not None:
        with torch.no_grad():
            psf.zcoef.copy_(torch.as_tensor(zc).to(device))
    return psf


def insitu_pos(z):
    """knot z -> (emitter height, focus) for the SMLM condition."""
    return z, torch.full_like(z, F0)


def bead_pos(z):
    """knot z (nominal emitter height) -> (emitter height 0, focus) for the bead stack."""
    return torch.zeros_like(z), F0 - z / ALPHA


def render(psf, pos, focus, nb=400):
    with torch.no_grad():
        return torch.cat([psf(pos[s:s + nb], focus[s:s + nb]) for s in range(0, len(pos), nb)])


def sim_stack(scn, nbeads=6, I=40000.0, bg=30.0, seed=0, **kw):
    """Bead z-stack: d [nbeads, nk, P, P] photon counts; z_nom [nk] nominal z of the planes; bead xy truth."""
    g = torch.Generator(device=DEV).manual_seed(100 + seed)
    psf = make_vec(P, scn, 'bead', **kw)
    z_nom = torch.arange(-0.10, 0.9001, 0.04, device=DEV)
    xy = torch.rand(nbeads, 2, generator=g, device=DEV) - 0.5
    nk = len(z_nom)
    pos = torch.cat([xy[:, None].expand(-1, nk, -1), torch.zeros(nbeads, nk, 1, device=DEV)], -1).reshape(-1, 3)
    f = (F0 - z_nom / ALPHA)[None].expand(nbeads, -1).reshape(-1)
    mu = I * render(psf, pos, f) + bg
    d = torch.poisson(mu, generator=g).reshape(nbeads, nk, P, P)
    return dict(d=d, z_nom=z_nom, xy=xy, I=I, bg=bg)


def sim_test(scn, N=4000, bg=30.0, seed=0, **kw):
    """Isolated test emitters under the SMLM condition with known truth: d [N, P, P], theta [N, 5]."""
    g = torch.Generator(device=DEV).manual_seed(200 + seed)
    psf = make_vec(P, scn, 'insitu', **kw)
    r = lambda *s: torch.rand(*s, generator=g, device=DEV)
    th = torch.stack([r(N) - 0.5, r(N) - 0.5, 0.05 + 0.7 * r(N),
                      torch.exp(math.log(4000.0) + 0.4 * torch.randn(N, generator=g, device=DEV)),
                      torch.full((N,), bg, device=DEV)], 1)
    mu = th[:, 3, None, None] * render(psf, th[:, :3], torch.full((N,), F0, device=DEV)) + bg
    return dict(d=torch.poisson(mu, generator=g), th=th, mu=mu)


def sim_movie(scn, T=6000, H=128, W=128, nsites=700, p_on=0.002, mean_on=3.0, bg=30.0, seed=0, **kw):
    """SMLM movie: fixed binding sites that blink (geometric on-times), Poisson noise.
    Returns movie [T, H, W] (uint16, cpu) and the truth table [n, 6] = (frame, X, Y, z, photons, site)."""
    g = torch.Generator(device=DEV).manual_seed(300 + seed)
    r = lambda *s: torch.rand(*s, generator=g, device=DEV)
    psf = make_vec(PR, scn, 'insitu', **kw)
    sx, sy = 4 + (W - 8) * r(nsites), 4 + (H - 8) * r(nsites)
    sz = ZR[0] + (ZR[1] - ZR[0]) * r(nsites)
    # two-state chain per site: off -> on with p_on, on -> off with 1 / mean_on
    on = torch.zeros(T, nsites, dtype=torch.bool, device=DEV)
    state = torch.zeros(nsites, dtype=torch.bool, device=DEV)
    for t in range(T):                                     # sequential in time (Markov chain)
        u = r(nsites)
        state = torch.where(state, u > 1 / mean_on, u < p_on)
        on[t] = state
    tt, ss = torch.nonzero(on, as_tuple=True)
    n = len(tt)
    phot = torch.exp(math.log(4000.0) + 0.4 * torch.randn(n, generator=g, device=DEV))
    X, Y, Z = sx[ss], sy[ss], sz[ss]
    hr = (PR - 1) // 2
    ox, oy = torch.round(X).long() - hr, torch.round(Y).long() - hr
    pos = torch.stack([X - torch.round(X), Y - torch.round(Y), Z], 1)
    movie = torch.empty(T, H, W, dtype=torch.uint16)
    ii = torch.arange(PR, device=DEV)
    CH = 500
    for t0 in range(0, T, CH):
        m = (tt >= t0) & (tt < t0 + CH)
        fr = torch.full((min(CH, T - t0), H, W), bg, device=DEV)
        if m.any():
            pat = phot[m, None, None] * render(psf, pos[m], torch.full((int(m.sum()),), F0, device=DEV))
            yy = (oy[m, None, None] + ii[None, :, None]).expand(-1, PR, PR)
            xx = (ox[m, None, None] + ii[None, None, :]).expand(-1, PR, PR)
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            flat = ((tt[m] - t0)[:, None, None] * H + yy.clamp(0, H - 1)) * W + xx.clamp(0, W - 1)
            fr.view(-1).index_add_(0, flat[ok], pat[ok])
        movie[t0:t0 + CH] = torch.poisson(fr, generator=g).clamp(max=65535).to(torch.uint16).cpu()
    truth = torch.stack([tt.float(), X, Y, Z, phot, ss.float()], 1)
    return movie, truth


def detect_crop(movie, bg_guess=None, thresh=6.0, min_sep=14, nb=500):
    """Spot detection on the movie (no ground truth used): difference of Gaussians, local maxima above thresh x the
    noise, spots closer than min_sep px to another spot or to the border are dropped. Returns rois [n, P, P] (float,
    photons), frame [n], corner (x0, y0) [n, 2]."""
    T, H, W = movie.shape
    h = (P - 1) // 2

    def gk(s):
        x = torch.arange(-int(4 * s), int(4 * s) + 1, device=DEV, dtype=torch.float32)
        k = torch.exp(-x ** 2 / (2 * s * s))
        return k / k.sum()

    def blur(x, s):
        k = gk(s)
        x = torch.nn.functional.conv2d(x[:, None], k[None, None, :, None], padding=(len(k) // 2, 0))
        return torch.nn.functional.conv2d(x, k[None, None, None, :], padding=(0, len(k) // 2))[:, 0]

    rois, frames, corners = [], [], []
    ii = torch.arange(P, device=DEV)
    for t0 in range(0, T, nb):
        fr = movie[t0:t0 + nb].to(DEV).float()
        dog = blur(fr, 2.0) - blur(fr, 6.0)
        sig = dog.flatten(1).std(1)[:, None, None]
        mx = torch.nn.functional.max_pool2d(dog[:, None], 9, 1, 4)[:, 0]
        pk = (dog == mx) & (dog > thresh * sig)
        t, y, x = torch.nonzero(pk, as_tuple=True)
        # isolation: no other peak within min_sep px (Chebyshev) in the same frame
        cnt = torch.nn.functional.conv2d(pk[:, None].float(), torch.ones(1, 1, 2 * min_sep + 1, 2 * min_sep + 1, device=DEV),
                                         padding=min_sep)[:, 0]
        ok = (cnt[t, y, x] < 1.5) & (x >= h + 2) & (x < W - h - 2) & (y >= h + 2) & (y < H - h - 2)
        t, y, x = t[ok], y[ok], x[ok]
        yy = (y[:, None, None] - h + ii[None, :, None]).expand(-1, P, P)
        xx = (x[:, None, None] - h + ii[None, None, :]).expand(-1, P, P)
        rois.append(fr[t[:, None, None], yy, xx])
        frames.append(t + t0)
        corners.append(torch.stack([x - h, y - h], 1))
    return torch.cat(rois), torch.cat(frames), torch.cat(corners)
