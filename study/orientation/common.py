"""Shared settings of the orientation study (study/orientation): SIMCODE optics, the in-situ PSF files (read-only), the
excitation tensors derived from the in-situ per-angle ensemble dipole tensors, plotting style."""
import json
import math
import os
import torch
from torch_psf import VectorialPSF
from torch_psf.psf import PSF
from torch_psf.orientation import excitation_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
SIM2 = os.path.expanduser('~/simcode2/models')
FILES = dict(m1809=os.path.join(SIM2, 'm1809/calib_1809_2_psf_v3.json'),
             m25=os.path.join(SIM2, 'm25/calib_2p5nM_m2_psf_v3.json'))
AXIS = dict(m1809=os.path.join(SIM2, 'm1809/calib_1809_2_psf_v3_axis.json'),
            m25=os.path.join(SIM2, 'm25/calib_2p5nM_m2_psf_v3_axis.json'))
R = 15
PIX = 0.10833
CK = 0.1525      # |E_x|^2 / |E|^2 of the evanescent p wave: 642 nm, 82 nm penetration depth, n_med 1.33 (see note)
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRID, AXISC, INK, INK2 = '#e1e0d9', '#c3c2b7', '#0b0b0b', '#5f5e58'


def style(plt):
    plt.rcParams.update({'axes.edgecolor': AXISC, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 9, 'legend.frameon': False, 'lines.linewidth': 1.6,
                         'figure.dpi': 110, 'savefig.dpi': 110})


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def free_model(focus, dev, R=R):
    v = VectorialPSF(R, pixel_um=PIX, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.70, npupil=64, oversample=2,
                     device=dev, focus_um=focus, z_range=(0.0, 0.4))
    for p in v.parameters():
        p.requires_grad_(False)
    return v


def file_model(name, dev, R=R):
    v = PSF.load(FILES[name], roisize=R, device=dev)
    v.dipM = None                     # the orientation code sets M itself; isotropic default for the bias fits
    v.z_range = (0.0, 0.4)
    for p in v.parameters():
        p.requires_grad_(False)
    return v


def models(dev):
    """name -> (VectorialPSF, description). Aberration-free at both in-situ foci, and the two in-situ PSFs."""
    f1809 = json.load(open(FILES['m1809']))['focus_um']
    f25 = json.load(open(FILES['m25']))['focus_um']
    return {'free_f%.2f' % f25: (free_model(f25, dev), f'aberration-free, focus {f25 * 1e3:.0f} nm'),
            'm25': (file_model('m25', dev), f'2.5 nM in-situ PSF (focus {f25 * 1e3:.0f} nm, Zernikes + blur)'),
            'free_f%.2f' % f1809: (free_model(f1809, dev), f'aberration-free, focus {f1809 * 1e3:.0f} nm'),
            'm1809': (file_model('m1809', dev), f'1809 in-situ PSF (focus {f1809 * 1e3:.0f} nm, Zernikes)')}


def ensemble_to_K(name, ck=CK):
    """Excitation tensors K_a (a = pattern angle 0, 1) for single dipoles consistent with SIMCODE's fitted per-angle
    ensemble tensors M_a (psf_dipole_M_angle): model M_a = (1 - gamma) I / 3 + gamma K_a with
    K_a = (1 - eps) e_a e_a^T + eps (ck k_a k_a^T + (1 - ck) z z^T) (e_a = SIMCODE e_dir, s-polarised, perpendicular
    to the fringe direction k_a; eps = evanescent p fraction). Solved from the diagonal in the (k_a, e_a, z) frame,
    averaged over both angles. Returns dict(gamma, eps, g_ens = sqrt(gamma / 0.4), K = [2, 3, 3] (eps fitted),
    K_s = [2, 3, 3] (eps = 0, pure s), e = [2, 2], k = [2, 2], M = [2, 3, 3])."""
    d = json.load(open(AXIS[name]))
    M = torch.tensor(d['psf_dipole_M_angle'], dtype=torch.float64)
    e = torch.tensor(d['psf3d']['e_dir'], dtype=torch.float64)
    k = torch.stack([torch.stack([e[a, 1], -e[a, 0]]) for a in range(2)])      # perpendicular to e
    gs, es = [], []
    for a in range(2):
        ee = torch.cat([e[a], torch.zeros(1, dtype=torch.float64)])
        kk = torch.cat([k[a], torch.zeros(1, dtype=torch.float64)])
        Mee, Mkk, Mzz = float(ee @ M[a] @ ee), float(kk @ M[a] @ kk), float(M[a, 2, 2])
        ge = (Mzz - Mkk) / (1 - 2 * ck)          # gamma eps
        gam = (Mee - Mkk) + ge * (1 + ck)
        gs.append(gam); es.append(ge / gam)
    gam, eps = sum(gs) / 2, sum(es) / 2
    K = torch.stack([excitation_tensor(e[a].float(), eps, k[a].float(), ck) for a in range(2)])
    Ks = torch.stack([excitation_tensor(e[a].float(), 0.0) for a in range(2)])
    return dict(gamma=gam, eps=eps, g_ens=math.sqrt(gam / 0.4), K=K, K_s=Ks, e=e.float(), k=k.float(), M=M.float(),
                gamma_simcode=d['psf3d']['gamma'], zeta_simcode=d['psf3d']['zeta'])
