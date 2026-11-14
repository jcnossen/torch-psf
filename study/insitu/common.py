"""Configuration of the in-situ PSF study (simulation and detection come from torch_psf): vectorial ground-truth PSFs, a bead z-stack and an SMLM movie.

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
from torch_psf import VectorialPSF
from torch_psf.zernike import zernike
from torch_psf import simulate
from torch_psf.spots import extract_spots

DEV = 'cuda'
P = 15                      # fit ROI
PR = 23                     # render ROI for the movie
F0 = 0.45                   # um, focus during the SMLM acquisition
ZR = (0.0, 0.8)             # um, emitter heights
ALPHA = 1.33 / 1.518        # paraxial focal-shift factor used to label the bead stack with a nominal emitter z
OPT = dict(pixel_um=0.108, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68, npupil=64, oversample=3)
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


def sim_stack(scn, nbeads=6, I=40000.0, bg=30.0, seed=0, **kw):
    """Bead z-stack: d [nbeads, nk, P, P] photon counts; z_nom [nk] nominal z of the planes; bead xy truth."""
    g = torch.Generator(device=DEV).manual_seed(100 + seed)
    psf = make_vec(P, scn, 'bead', **kw)
    z_nom = torch.arange(-0.10, 0.9001, 0.04, device=DEV)
    s = simulate.simulate_bead_stack(psf.forward_batched, F0 - z_nom / ALPHA, nbeads, I, bg, generator=g, device=DEV)
    return dict(d=s['d'], z_nom=z_nom, xy=s['xy'], I=I, bg=bg)


def sim_test(scn, N=4000, bg=30.0, seed=0, **kw):
    """Isolated test emitters under the SMLM condition with known truth: d [N, P, P], theta [N, 5]."""
    g = torch.Generator(device=DEV).manual_seed(200 + seed)
    psf = make_vec(P, scn, 'insitu', **kw)
    return simulate.simulate_emitters(lambda p: psf.forward_batched(p, torch.full((len(p),), F0, device=DEV)), N,
                                      bg=bg, generator=g, device=DEV)


def sim_movie(scn, T=6000, H=128, W=128, nsites=700, p_on=0.002, mean_on=3.0, bg=30.0, seed=0, **kw):
    """SMLM movie: fixed binding sites that blink (geometric on-times), Poisson noise.
    Returns movie [T, H, W] (uint16, cpu) and the truth table [n, 6] = (frame, X, Y, z, photons, site)."""
    g = torch.Generator(device=DEV).manual_seed(300 + seed)
    psf = make_vec(PR, scn, 'insitu', **kw)
    return simulate.simulate_movie(lambda p: psf.forward_batched(p, torch.full((len(p),), F0, device=DEV)), PR, T, H,
                                   W, nsites, p_on, mean_on, bg, z_range=ZR, generator=g, device=DEV)


def detect_crop(movie, thresh=6.0, min_sep=14, nb=500):
    """Spot detection on the movie (no ground truth used): difference of Gaussians, local maxima above thresh x the
    noise, spots closer than min_sep px to another spot or to the border are dropped. Returns rois [n, P, P] (float,
    photons), frame [n], corner (x0, y0) [n, 2]."""
    sp = extract_spots(movie, P, frames=None, chunk=nb, device=DEV, threshold=thresh, sigma=(2.0, 6.0), nms=9,
                       noise='std', isolation=min_sep)
    return sp.rois, sp.frame, sp.corner


def render(psf, pos, focus, nb=400):
    """Normalized vectorial PSF images without gradient, in batches."""
    return psf.forward_batched(pos, focus, nb)
