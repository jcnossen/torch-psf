"""Numerical cross-check of our VectorialPSF against simcode2's line-by-line PyTorch port of their MATLAB code
(~/vortex_ref/vecfitcpu_vortex-pytorch, validated against Octave): the same expected image for the same emitter,
including the vortex plate, the per-spot NAT Zernikes and their immersion-mismatch / focus phase (Wfocus).

Mapping of their model onto ours (found here, see main()):
  * arrays: their XPupil runs along the first array index (image row) and so does their image x; our py / image y
    run along the rows, so their pupil maps are our pupil maps array by array (a transposition of the axis names:
    their (x, y) = our (y, x)); the dipole vector: their (mu_x, mu_y, mu_z) = our (mu_y, mu_x, mu_z), x, y positions
    swapped likewise;
  * phase convention: their pupil uses exp(+i 2 pi W / lambda) with cos(theta_med) = -i |.| beyond the critical
    angle (and propagation exp(-i k.r)); ours uses +i |.|; checked here: conj (their W -> -W in ours) or not;
  * z: their stage z enters as exp(-i kz_imm z); our FocusZ z as exp(+i kz_imm z) (focus = -z): z_ours = -z_theirs.

usage: python -I study/orientation/vortex_crosscheck.py      (CPU is enough)
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
PORT = os.environ.get('VECFIT_PORT', os.path.expanduser('~/vortex_ref/vecfitcpu_vortex-pytorch/python'))
sys.path.insert(0, PORT)
import numpy as np                                                                     # noqa: E402
import torch                                                                           # noqa: E402
from torch_psf import VectorialPSF                                                     # noqa: E402
from torch_psf.orientation import FocusZ, dipole_M                                     # noqa: E402
from vecfit.set_parameters_vortex_lambdaDNA import set_parameters_vortex_lambdaDNA    # noqa: E402
from vecfit.funfit.get_pupil_matrix import pupil_static, get_pupil_matrix             # noqa: E402
from vecfit.funfit.poissonrate import poissonrate                                     # noqa: E402
from vecfit.pipeline import nat_predictions                                           # noqa: E402


def their_image(p, theta, zc_nm):
    """their expected image(s) [S, Mx, My] for theta [S, 8] (x, y, z nm, N, bg, azim, pola, g2), zc_nm [S, 12]"""
    st = pupil_static(p)
    wv, wvz, PM = get_pupil_matrix(p, st, torch.as_tensor(zc_nm, dtype=torch.float64), 'cpu')
    Cx = torch.as_tensor(p['Cx']); Cy = torch.as_tensor(p['Cy'])
    mu, _ = poissonrate(p, torch.as_tensor(theta, dtype=torch.float64), PM, wv, wvz, Cx, Cy)
    return mu, st


def their_pupil_phase_waves(p, st, zc_nm):
    """their total pupil phase (waves) per spot [S, N, N]: Zernikes + plate + Wfocus"""
    Z = st['allzernikes']; nf = st['normfac']
    Wab = np.einsum('xyj,sj->sxy', Z, np.asarray(zc_nm) * nf)
    W = Wab + np.real(st['W0'])[None] + np.real(st['Wfocus'])[None]
    return W / p['lambda'] * st['ApertureMask'][None]


def our_image(p, st, theta, zc_nm, conj, swap, zsign=-1.0, xysign=1.0, dipflip=False):
    """our image for the same emitter: pupil_phase = +-(their total phase) array by array, positions / dipole swapped
    if swap, z_ours = -z_theirs, the same normalisation as theirs (their N is the whole-plane photon count)."""
    n = len(theta)
    v = VectorialPSF(p['Mx'], pixel_um=p['pixelsize'] / 1e3, NA=p['NA'], n_imm=p['refimm'], n_med=p['refmed'],
                     wavelength_um=p['lambda'] / 1e3, npupil=p['Npupil'], oversample=1, device='cpu',
                     zernike_indices=(5,), focus_um=0.0, z_range=(-1.1, 1.1))
    for q in v.parameters():
        q.requires_grad_(False)
    Wt = their_pupil_phase_waves(p, st, zc_nm)
    out = []
    for i in range(n):
        W = torch.as_tensor(Wt[i], dtype=torch.float32)
        W = W.T if not swap else W                 # swap: their axis 0 -> our rows (array identity)
        with torch.no_grad():
            v.pupil_phase.copy_(-W if conj else W)
        x, y, z = (float(t) / 1e3 for t in theta[i, :3])
        az, po, g2 = theta[i, 5], theta[i, 6], theta[i, 7]
        mu_t = np.array([math.sin(po) * math.cos(az), math.sin(po) * math.sin(az), math.cos(po)])
        if swap:
            pos = torch.tensor([[y / v.pixel_um, x / v.pixel_um, z]], dtype=torch.float32)
            mu = torch.tensor([[mu_t[1], mu_t[0], mu_t[2]]], dtype=torch.float32)
        else:
            pos = torch.tensor([[x / v.pixel_um, y / v.pixel_um, z]], dtype=torch.float32)
            mu = torch.tensor([mu_t], dtype=torch.float32)
        f = FocusZ(v, height=0.0, z_range=(-1.1, 1.1))
        pos[:, 2] = zsign * pos[:, 2]              # z_ours = -z_theirs (zsign = -1)
        pos[:, :2] = xysign * pos[:, :2]
        if dipflip:
            mu = mu * torch.tensor([[-1.0, -1.0, 1.0]])
        # their image: N [(1 - g2) free / n_free + g2 fixed / n_fixed] + bg; ours: build both parts separately
        with torch.no_grad():
            free = f(pos, dipM=torch.eye(3)[None] / 3)[0]
            fixed = f(pos, dipM=(mu[:, :, None] * mu[:, None, :]))[0]
        out.append((free, fixed))
    return out, v


def main():
    torch.set_default_dtype(torch.float32)
    rng = np.random.default_rng(0)
    for refmed in (1.333, 1.0):
        p = set_parameters_vortex_lambdaDNA(refmed=refmed)
        S = 4
        roixy = np.stack([rng.uniform(100, 900, S), rng.uniform(100, 900, S)], 1)
        nat, _, _ = nat_predictions(p, roixy, 1024)
        theta = np.zeros((S, 8))
        theta[:, 0] = rng.uniform(-60, 60, S); theta[:, 1] = rng.uniform(-60, 60, S); theta[:, 2] = rng.uniform(-150, 150, S)
        theta[:, 3] = 5000; theta[:, 4] = 0.0
        theta[:, 5] = rng.uniform(0, 2 * np.pi, S); theta[:, 6] = rng.uniform(0.3, 2.8, S); theta[:, 7] = 1.0
        mu_t, st = their_image(p, theta, nat)
        mu_t = mu_t.numpy()
        for conj, zsign in ((False, -1.0), (False, 1.0), (True, -1.0), (True, 1.0)):
            ours, v = our_image(p, st, theta, nat, conj, True, zsign)
            errs = []
            for i, (free, fixed) in enumerate(ours):
                a = mu_t[i] / mu_t[i].sum()
                b = fixed.numpy(); b = b / b.sum()
                errs.append(np.abs(a - b).max() / a.max())
            print(f'refmed {refmed}: conj {conj!s:5} zsign {zsign:+.0f}: max rel diff {np.round(errs, 4).tolist()}',
                  flush=True)
        for dipflip in (False, True):
            for zsign in (1.0, -1.0):
                ours, v = our_image(p, st, theta, nat, True, True, zsign, -1.0, dipflip)
                errs = [float(np.abs(mu_t[i] / mu_t[i].sum() - np.flip(f.numpy(), (0, 1)) / f.numpy().sum()).max()
                              / (mu_t[i] / mu_t[i].sum()).max()) for i, (_, f) in enumerate(ours)]
                print(f'  conj + rot180 + x -> -x, zsign {zsign:+.0f}, dipole flip {dipflip}: max rel diff '
                      f'{np.round(errs, 4).tolist()}', flush=True)


if __name__ == '__main__':
    main()
