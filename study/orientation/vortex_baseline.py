"""SUPERSEDED by vortex_exact.py, which reproduces their model exactly (checked against simcode2's port of their code).
Two settings here are wrong: the NAT field axes (their xfov is the image COLUMN, not the row) and the sign of their
pupil phase in our convention (their pupil is the complex conjugate of ours: plate, Zernikes and W_mis enter with
the opposite sign, and their image is our image rotated by 180 deg). Kept for the record of the first attempt.

Baseline reproduction of Hulleman et al. (Nat. Commun. 12, 5934, 2021) on their data (4TU 10.4121/13031837), with
their processing settings taken from their MATLAB code (vecfitcpu_vortex, 4TU 10.4121/15175494):

  optics      NA 1.47, n_imm 1.518, wavelength 597.5 nm, pixel 65 nm, ROI 15 x 15, the PSF point-sampled at the pixel
              centres (no pixel integration: VectorialPSF oversample 1), pupil 32 x 32 samples
  sample      lambda-DNA: n_med 1.333, emitter at the coverslip (depth 0); PMMA: n_med 1.49 (the paper's value; the
              PMMA settings are not in the released code)
  immersion   their model has the coverslip (1.523) and the immersion oil (1.518) with the objective designed for 1.523
              over a 120 um free working distance: the phase fwd (n_imm cos_imm - n_nom cos_nom), best focus removed,
              is added to the pupil (W_mis; refnom 1.523, or their optimised 1.5209)
  plate       the spiral plate, 64 levels, phase depth 593 nm, charge +1 in our pupil coordinates (their
              doevortexflip = 1 with x the first array index = the image column); centred (aligned optically)
  aberrations their Nodal Aberration Theory map (read/nat/natCoefficients_mlambda_oil.mat, calibrated on beads in
              oil): per spot the Zernikes (n, m) (2,-2) (2,2) (3,-1) (3,1) (3,-3) (3,3) (4,0) (4,-2) (4,2) (5,-1)
              (5,1) (6,0) = Noll 5 6 7 8 9 10 11 13 12 17 16 22 at the field position, as fixed zcoef_offset;
              their field coordinates: xn = (row - FOV / 2) x 65 nm, yn = (column - FOV / 2) x 65 nm (roixy(:, 1) is
              the row index in their array layout)
  camera      (ADU - 106) x 0.33 e / ADU, values <= 0 set to 1e-3
  detection   FastPeakFind(threshold 25 photons, 3 x 3 median, 7 x 7 Gaussian sigma 1, weighted centroids, areas
              <= mean + 2 sd), 15 px edge, spots with a neighbour closer than 5.3 px dropped (both)
  fit         their initial values (moments: centroid, pola0 = pi eccentricity, azim0 = -atan2(mu12, mu21) mod pi,
              g2 0.75, z 0 (stage), N = 2.5 x sum, bg = median of the ROI rim), one start, Levenberg-Marquardt
              50 iterations, z in [-1.1, 1.1] um (stage), g in [0, 1]
  filters     outliers: |x| or |y| > 3 px from the ROI centre, log-likelihood below mean - 3 sd; chi^2 in [0.75, 3]
  g2          their mixture H = (1 - g2) H_free / n_free + g2 H_fixed / n_fixed (each part normalised by its pupil
              energy) is our M = (1 - g) I / 3 + g mu mu^T up to the per-orientation weight:
              g2 = g n_fix / (g n_fix + (1 - g) n_free) (n = pupil energies); the fits are the same, only the
              reported g2 differs. N (their photons, whole image plane) = I ((1 - g) n_free + g n_fix) / n_free.

Stages: pmma (the PMMA through-focus validation), dna (lambda-DNA, first movie, frames 0-600, the densest 256 px crop),
both with variants (argv[2:]: 'refnom1.5209', 'nonat', 'nowmis', 'nmed1.0', 'zgrid': starts on a z grid, 'insitu':
our in-situ calibrated model from vortex_real.py with their initial values / start / filters).
usage: CUDA_VISIBLE_DEVICES=2 python -I study/orientation/vortex_baseline.py pmma|dna [variant ...]
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np                                                                     # noqa: E402
import torch                                                                           # noqa: E402
import vortex_real as VR                                                              # noqa: E402
from torch_psf import VectorialPSF                                                     # noqa: E402
from torch_psf.composition import ZMode  # noqa: E402
from torch_psf.orientation import fit_orientation, model_and_jacobian, vortex_phase, unit, angles  # noqa: E402

OUT = VR.OUT
DEV = VR.DEV
VF = os.environ.get('VECFIT_DIR', os.path.join(VR.DATA, 'code', 'vecfitcpu_vortex-master'))   # their MATLAB code
OPT = dict(NA=1.47, n_imm=1.518, wavelength_um=0.5975, pixel_um=0.065)
NOLL = tuple(range(5, 23))
NAT_NOLL = (5, 6, 7, 8, 9, 10, 11, 13, 12, 17, 16, 22)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ model
def wmis_waves(v, refnom=1.523, fwd_um=120.0):
    """Their immersion / design mismatch phase fwd (n_imm cos_imm - n_nom cos_nom) (waves), with the best-focus term
    (a multiple of n_imm cos_imm) and piston removed by least squares over the pupil."""
    NA, lam = float(v.NA), v.wavelength_um
    r = v.rho.double()
    m = v.mask.bool()
    ci = torch.sqrt((1 - (NA * r / v.n_imm) ** 2).clamp_min(0))
    cn = torch.sqrt((1 - (NA * r / refnom) ** 2).clamp_min(0))
    W = fwd_um * (v.n_imm * ci - refnom * cn)                     # um of optical path
    A = torch.stack([v.n_imm * ci[m], torch.ones_like(ci[m])], 1)
    coef = torch.linalg.lstsq(A, W[m][:, None]).solution[:, 0]
    W = (W - coef[0] * v.n_imm * ci - coef[1]) * m
    return (W / lam).float()


def their_model(n_med, refnom=1.523, wmis=True, vortex=True, roisize=15, npupil=32):
    v = VectorialPSF(roisize, n_med=n_med, npupil=npupil, oversample=1, zernike_indices=NOLL, device=DEV,
                     focus_um=0.0, z_range=(-1.1, 1.1), **OPT)
    for p in v.parameters():
        p.requires_grad_(False)
    W = torch.zeros_like(v.pupil_phase)
    if vortex:
        W = W + vortex_phase(v, charge=1, levels=64, depth=593.0 / 597.5, apply=False)
    if wmis:
        W = W + wmis_waves(v, refnom)
    with torch.no_grad():
        v.pupil_phase.copy_(W * v.mask)
    return v


def nat_offsets(rows, cols, fov, wl_nm=597.5):
    """their NAT Zernikes (waves) at image (row, col) (0-based px) of a frame of width fov -> [N, len(NOLL)]"""
    import scipy.io as sio
    m = sio.loadmat(os.path.join(VF, 'read/nat/natCoefficients_mlambda_oil.mat'))
    g = lambda k: np.asarray(m[k], np.float64).ravel()
    A3, A5, C3, C5, S5, T, S6 = (g(k) for k in ('RAstig3', 'RAstig5', 'RComa3', 'RComa5', 'RCurv5', 'RTrefoil',
                                                'RCurv6'))
    A3, A5, C3, C5, S5, T, S6 = [np.concatenate([[0.0], a]) for a in (A3, A5, C3, C5, S5, T, S6)]  # 1-based
    xn = 0.065 * (np.asarray(rows, np.float64) + 1 - fov / 2)
    yn = 0.065 * (np.asarray(cols, np.float64) + 1 - fov / 2)
    p = np.zeros((len(xn), 12))
    p[:, 0] = (2 * A3[1] * (xn ** 3 * yn + xn * yn ** 3) + A3[2] * (xn ** 2 * yn + yn ** 3) + A3[3] * (xn ** 3 + xn * yn ** 2)
               + A3[4] * (xn ** 2 + yn ** 2) + A3[6] * (xn * yn ** 2 - xn ** 3) + A3[7] * (2 * xn * yn ** 2) + A3[8] * yn
               - A3[9] * xn + A3[10] * (2 * xn * yn) + A3[11] * xn + A3[12] * yn + A3[13])
    p[:, 1] = (A3[1] * (yn ** 4 - xn ** 4) - A3[2] * (xn ** 3 + xn * yn ** 2) + A3[3] * (xn ** 2 * yn + yn ** 3)
               + A3[5] * (xn ** 2 + yn ** 2) - A3[6] * (2 * xn ** 2 * yn) + A3[7] * (yn ** 3 - xn ** 2 * yn) + A3[8] * xn
               + A3[9] * yn + A3[10] * (yn ** 2 - xn ** 2) + A3[11] * yn - A3[12] * xn + A3[14])
    p[:, 2] = (C3[1] * (xn ** 3 + xn * yn ** 2) + C3[2] * xn ** 2 + C3[3] * xn * yn + C3[4] * xn + C3[5] * (xn ** 2 + yn ** 2)
               + C3[7] * yn + C3[8] * xn + C3[9])
    p[:, 3] = (C3[1] * (yn ** 3 + xn ** 2 * yn) + C3[2] * xn * yn + C3[3] * yn ** 2 + C3[4] * yn + C3[6] * (xn ** 2 + yn ** 2)
               + C3[7] * xn - C3[8] * yn + C3[10])
    p[:, 4] = T[1] * (3 * yn ** 2 * xn - xn ** 3) + T[2] * (yn ** 2 - xn ** 2) + T[3] * (2 * xn * yn) + T[4] * xn + T[5] * yn + T[6]
    p[:, 5] = T[1] * (yn ** 3 - 3 * xn ** 2 * yn) - T[2] * (2 * xn * yn) + T[3] * (yn ** 2 - xn ** 2) + T[4] * yn - T[5] * xn + T[7]
    p[:, 6] = S5[1] * (xn ** 2 + yn ** 2) + S5[2] * xn + S5[3] * yn + S5[4]
    p[:, 7] = A5[1] * (2 * xn * yn) + A5[2] * yn + A5[3] * xn + A5[4]
    p[:, 8] = A5[1] * (yn ** 2 - xn ** 2) - A5[2] * xn + A5[3] * yn + A5[5]
    p[:, 9] = C5[1] * xn + C5[2]
    p[:, 10] = C5[1] * yn + C5[3]
    p[:, 11] = S6[1] * (xn ** 2 + yn ** 2) + S6[2] * xn + S6[3] * yn + S6[4]
    out = np.zeros((len(xn), len(NOLL)), np.float32)
    for k, j in enumerate(NAT_NOLL):
        out[:, NOLL.index(j)] = p[:, k] * 1e-3              # mlambda -> waves
    return torch.as_tensor(out, device=DEV)


def pupil_energy(v):
    """IM_ij = sum_pupil Re[P_i . P_j*] (3 x 3), normalised to trace 3 (the isotropic emitter: 1 per dipole)."""
    P = v.pupil_fields()[0].reshape(3, 2, *v.rho.shape)
    IM = torch.einsum('icuv,jcuv->ij', P.real, P.real) + torch.einsum('icuv,jcuv->ij', P.imag, P.imag)
    return IM / (torch.trace(IM) / 3)


def to_their_g2(v, mu, g):
    """their g2 and the photon-weight factor N / I for our (mu, g)"""
    T = pupil_energy(v).to(mu.dtype)
    nfix = torch.einsum('ni,ij,nj->n', mu, T, mu)
    g2 = g * nfix / (g * nfix + (1 - g) * 1.0)
    return g2, (1 - g) + g * nfix


# ------------------------------------------------------------------ their detection and initial values
def their_detect(img, thr=25.0, edge=15, maxdist=None, roisize=15):
    """FastPeakFind(img, thr, gauss 7x7 sigma 1, edge, res 2) + their neighbour removal; img [H, W] photons ->
    (rows, cols) int arrays (0-based)."""
    from scipy.ndimage import median_filter, convolve, label, center_of_mass, sum as nsum
    maxdist = maxdist or math.sqrt(2 * roisize ** 2) / 4
    d = np.floor(np.clip(img, 0, 65535)).astype(np.float64)
    d = median_filter(d, size=3)
    d = d * (d > thr)
    x = np.arange(7) - 3
    k = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 2.0); k /= k.sum()
    d = convolve(d, k, mode='constant')
    d = d * (d > 0.9 * thr)
    lab, n = label(d > 0, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    idx = np.arange(1, n + 1)
    area = nsum(np.ones_like(d), lab, idx)
    keep = area <= area.mean() + 2 * area.std()
    cm = np.array(center_of_mass(d, lab, idx))[keep]
    r, c = np.round(cm[:, 0]).astype(int), np.round(cm[:, 1]).astype(int)
    H, W = img.shape
    h = roisize // 2
    ok = (r - h > 0) & (c - h > 0) & (r + h < H - 1) & (c + h < W - 1)
    r, c = r[ok], c[ok]
    if len(r) > 1:
        dd = np.sqrt((r[:, None] - r[None]) ** 2 + (c[:, None] - c[None]) ** 2)
        np.fill_diagonal(dd, np.inf)
        far = (dd >= maxdist).all(1)
        r, c = r[far], c[far]
    return r, c


def their_init(rois):
    """theta0 [N, 9] from their initialvalues (K = 1, vortex); rois [N, P, P] photons (row, column)."""
    N, P = rois.shape[0], rois.shape[-1]
    d = rois.double()
    rim = torch.cat([d[:, 0, :], d[:, -1, :], d[:, :, 0], d[:, :, -1]], 1)
    bg = rim.median(1).values.clamp(1, 50)
    f = (d - bg[:, None, None]).clamp_min(1e-3)
    c = torch.arange(P, device=d.device, dtype=d.dtype) - (P - 1) / 2
    X, Y = c[None, None, :], c[None, :, None]                       # X along columns, Y along rows (px)
    m00 = f.sum((1, 2))
    m10 = (X * f).sum((1, 2)); m01 = (Y * f).sum((1, 2))
    m20 = (X ** 2 * f).sum((1, 2)); m02 = (Y ** 2 * f).sum((1, 2)); m11 = (X * Y * f).sum((1, 2))
    xc, yc = m10 / m00, m01 / m00
    mu20 = m20 - m00 * xc ** 2; mu11 = m11 - m00 * xc * yc; mu02 = m02 - m00 * yc ** 2
    mu21 = m20 - m20 * yc - 2 * m11 * yc + 2 * m00 * xc ** 2 * yc      # as in their code
    mu12 = m02 - m02 * xc - 2 * m11 * yc + 2 * m00 * xc * yc ** 2
    s = torch.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    e1, e2 = (mu20 + mu02 + s) / 2, (mu20 + mu02 - s) / 2
    ecc = torch.sqrt((1 - e2 / e1).clamp(0, 1))
    pola = torch.remainder(math.pi * ecc, math.pi)
    azim = torch.remainder(-torch.atan2(mu12, mu21), math.pi)
    mu = torch.stack([torch.sin(pola) * torch.cos(azim), torch.sin(pola) * torch.sin(azim), torch.cos(pola)], 1)
    th = torch.zeros(N, 9, device=rois.device)
    th[:, 0], th[:, 1] = xc.float(), yc.float()
    th[:, 3] = (2.5 * m00).float(); th[:, 4] = bg.float()
    th[:, 5:8] = mu.float(); th[:, 8] = 0.75
    return th


def their_fit(psf, rois, zo, z_limits=(-1.1, 1.1), iters=50, zgrid=None):
    """one start from their initial values (z = 0), or with zgrid one per height (their initial values otherwise; the
    best likelihood wins); returns fit_orientation's dict + chi2 + their outlier flag"""
    th0 = their_init(rois)
    r = None
    for zg in (zgrid or (0.0,)):
        t0 = th0.clone(); t0[:, 2] = zg
        q = fit_orientation(psf, rois, theta0=t0, iters=iters, z_limits=z_limits, g_limits=(0.0, 1.0),
                            batch=400, zcoef_offset=zo)
        if r is None:
            r = q
        else:
            b = q['nll'] < r['nll']
            for k in ('theta', 'nll', 'theta_deg', 'phi_deg', 'g'):
                r[k][b] = q[k][b]
            for k in r['crlb']:
                r['crlb'][k][b] = q['crlb'][k][b]
    th = r['theta']
    mu = torch.cat([model_and_jacobian(psf, th[s:s + 400], zcoef_offset=None if zo is None else zo[s:s + 400])[0][:, 0]
                    for s in range(0, len(th), 400)])
    chi2 = ((rois - mu) ** 2 / mu.clamp_min(1e-6)).flatten(1).mean(1)
    ll = -r['nll']
    out = (th[:, :2].abs() > 3).any(1) | (ll < ll.mean() - 3 * ll.std())
    r.update(chi2=chi2, outlier=out)
    return r


def variant(argv):
    v = dict(refnom=1.523, nat=True, wmis=True, n_med=None, zgrid=None, insitu=False)
    for a in argv:
        if a.startswith('refnom'):
            v['refnom'] = float(a[6:])
        elif a == 'nonat':
            v['nat'] = False
        elif a == 'nowmis':
            v['wmis'] = False
        elif a.startswith('nmed'):
            v['n_med'] = float(a[4:])
        elif a == 'insitu':
            v['insitu'] = True
        elif a == 'zgrid':
            v['zgrid'] = (-0.8, -0.4, 0.0, 0.4, 0.8)
    return v


# ------------------------------------------------------------------ PMMA
def run_pmma(var, tag):
    t0 = time.time()
    n_med = var['n_med'] or 1.49
    mols = VR.pmma_molecules()
    rv, _ = VR.stack_rois(VR.PMMA_V, [m[0] for m in mols])
    nm, nk, P = rv.shape[:3]
    cent = np.array([m[0] for m in mols])                      # (x = column, y = row) of the molecule
    if var['insitu']:      # our in-situ calibrated model (vortex_real.py plate2: plate centre, Zernikes, n_med, apod)
        psf = VR.sm_focusz(z_range=(-1.1, 1.1))
        v = psf.base
        zo = None
    else:
        v = their_model(n_med, refnom=var['refnom'], wmis=var['wmis'])
        psf = ZMode(v, height=0.0, z_range=(-1.1, 1.1))
        zo = nat_offsets(cent[:, 1], cent[:, 0], 2048) if var['nat'] else None
    zz = None if zo is None else zo[:, None].expand(-1, nk, -1).reshape(nm * nk, -1)
    r = their_fit(psf, rv.reshape(-1, P, P), zz, zgrid=var['zgrid'])
    g2, _ = to_their_g2(v, r['theta'][:, 5:8], r['theta'][:, 8])
    T = lambda a: a.reshape(nm, nk).cpu().numpy()
    rows = []
    for i in range(nm):
        pp = dict(z_vortex=T(r['theta'][:, 2])[i].tolist(), theta_vortex=T(r['theta_deg'])[i].tolist(),
                  phi_vortex=T(r['phi_deg'])[i].tolist(), g_vortex=T(r['theta'][:, 8])[i].tolist(),
                  g2=T(g2)[i].tolist(), chi2=T(r['chi2'])[i].tolist(), outlier=T(r['outlier'].float())[i].tolist(),
                  crlb_theta=T(r['crlb']['theta'])[i].tolist(), crlb_phi=T(r['crlb']['phi'])[i].tolist(),
                  z_std=[0.0] * nk, theta_std=[0.0] * nk, phi_std=[0.0] * nk)
        rows.append(dict(mol=i, xy=mols[i][0], per_plane=pp))
    summ = VR.summarize_pmma(rows, zr=0.5)                    # their 11 slices: +-500 nm around focus
    g2m = [np.mean([g for g, z in zip(rw['per_plane']['g2'], rw['per_plane']['z_vortex']) if abs(z) <= 0.5])
           for rw in rows]
    summ.update(g2_mean=float(np.nanmean(g2m)), g2_sd_between=float(np.nanstd(g2m)),
                chi2_median=float(r['chi2'].median()), variant=var, seconds=time.time() - t0)
    summ.pop('vs_standard', None)
    log(tag, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in summ.items()
                         if k != 'per_molecule'}))
    json.dump(dict(summary=summ, molecules=rows), open(os.path.join(OUT, f'baseline_pmma_{tag}.json'), 'w'))


# ------------------------------------------------------------------ lambda-DNA
def run_dna(var, tag, frames=(0, 600), crop=(400, 16, 256, 256)):
    t0 = time.time()
    n_med = var['n_med'] or 1.333
    x0, y0, w, h = crop
    mov = VR._movie_photons(VR.DNA_MOVIE, crop=None, frames=frames).numpy()       # full frames: their detection
    mov = np.where(mov <= 0, 1e-3, mov)
    rois, rc, fr = [], [], []
    hP = 7
    for t in range(mov.shape[0]):
        r, c = their_detect(mov[t])
        sel = (c >= x0) & (c < x0 + w) & (r >= y0) & (r < y0 + h)
        for rr, cc in zip(r[sel], c[sel]):
            rois.append(mov[t, rr - hP:rr + hP + 1, cc - hP:cc + hP + 1]); rc.append((rr, cc)); fr.append(t)
    rois = torch.as_tensor(np.stack(rois), device=DEV, dtype=torch.float32)
    rc = np.array(rc); fr = np.array(fr)
    log(f'{tag}: {len(rois)} spots detected ({time.time() - t0:.0f} s)')
    v = their_model(n_med, refnom=var['refnom'], wmis=var['wmis'])
    psf = ZMode(v, height=0.0, z_range=(-1.1, 1.1))
    zo = nat_offsets(rc[:, 0], rc[:, 1], 1024) if var['nat'] else None
    res = {k: [] for k in ('theta', 'theta_deg', 'phi_deg', 'chi2', 'outlier', 'crlb_xy', 'crlb_inplane', 'crlb_theta',
                           'crlb_g', 'crlb_z', 'nll')}
    for s in range(0, len(rois), 2000):
        r = their_fit(psf, rois[s:s + 2000], None if zo is None else zo[s:s + 2000], zgrid=var['zgrid'])
        for k in ('theta', 'theta_deg', 'phi_deg', 'chi2', 'outlier', 'nll'):
            res[k].append(r[k].cpu())
        res['crlb_xy'].append(((r['crlb']['x'] + r['crlb']['y']) / 2).cpu())
        for k in ('inplane', 'theta', 'g', 'z'):
            res['crlb_' + k].append(r['crlb'][k].cpu())
        log(f'{tag}: fitted {min(s + 2000, len(rois))} / {len(rois)} ({time.time() - t0:.0f} s)')
    R = {k: torch.cat(v_) for k, v_ in res.items()}
    th = R['theta']
    g2, Nf = to_their_g2(v, th[:, 5:8].to(DEV), th[:, 8].to(DEV))
    g2 = g2.cpu(); N = (th[:, 3] * Nf.cpu())
    keep = (~R['outlier']) & (R['chi2'] > 0.75) & (R['chi2'] < 3)
    xy = np.stack([rc[:, 1] + th[:, 0].numpy(), rc[:, 0] + th[:, 1].numpy()], 1)    # (x = column, y = row) px
    k = keep.numpy()
    ax = VR.local_axis(xy[k], radius=8.0, min_n=12, min_aniso=10.0)
    ph = R['phi_deg'].numpy()[k] % 180
    dphi = (ph - ax) % 180
    ok = np.isfinite(dphi)
    # distance of each localisation to the local line through its neighbours (FWHM across the strand)
    from scipy.spatial import cKDTree
    P = xy[k]; tr = cKDTree(P); dr = np.full(len(P), np.nan)
    for i, nb in enumerate(tr.query_ball_point(P, 8.0)):
        nb = [j for j in nb if j != i]
        if len(nb) < 12:
            continue
        q = P[nb]; c = q.mean(0); u, s_, vt = np.linalg.svd(q - c, full_matrices=False)
        if s_[0] < 3.16 * s_[1]:
            continue
        nrm = vt[1]
        dr[i] = (P[i] - c) @ nrm * 65.0
    drv = dr[np.isfinite(dr)]
    hist, edges = np.histogram(drv, bins=np.arange(-60, 61, 4))
    half = hist.max() / 2
    above = np.nonzero(hist >= half)[0]
    fwhm = float((above[-1] - above[0] + 1) * 4) if len(above) else float('nan')
    mad = lambda a: float(np.median(np.abs(a - np.median(a))))
    hg, _ = np.histogram(g2.numpy()[k], bins=20, range=(0, 1))
    L = VR._their_locs(frames, crop)
    tax = VR.local_axis(np.stack([L['x'], L['y']], 1), radius=8.0, min_n=12, min_aniso=10.0)
    tdphi = (L['sx'] % 180 - tax) % 180
    tok = np.isfinite(tdphi)
    thg, _ = np.histogram(L['ellipticity'], bins=20, range=(0, 1))
    summ = dict(variant=var, n_spots=len(rois), n_kept=int(k.sum()), n_axis=int(ok.sum()),
                dphi_median=float(np.median(dphi[ok])), dphi_mad=mad(dphi[ok]),
                g2_peak=float((np.argmax(hg) + 0.5) / 20), g2_median=float(np.median(g2.numpy()[k])),
                g_ours_median=float(np.median(th[k, 8].numpy())),
                theta_median=float(np.median(R['theta_deg'].numpy()[k])),
                fwhm_across_nm=fwhm, n_across=int(len(drv)), sd_across_nm=float(np.std(drv[np.abs(drv) < 60])),
                N_median=float(np.median(N.numpy()[k])), bg_median=float(np.median(th[k, 4].numpy())),
                z_median_nm=float(np.median(th[k, 2].numpy()) * 1e3),
                crlb_xy_nm=float(np.median(R['crlb_xy'].numpy()[k]) * 65), crlb_z_nm=float(np.median(R['crlb_z'].numpy()[k]) * 1e3),
                crlb_inplane=float(np.median(R['crlb_inplane'].numpy()[k])), crlb_theta=float(np.median(R['crlb_theta'].numpy()[k])),
                crlb_g=float(np.median(R['crlb_g'].numpy()[k])), chi2_median=float(R['chi2'].median()),
                theirs=dict(n=len(L), n_axis=int(tok.sum()), dphi_median=float(np.median(tdphi[tok])),
                            dphi_mad=mad(tdphi[tok]), g2_peak=float((np.argmax(thg) + 0.5) / 20),
                            g2_median=float(np.median(L['ellipticity'])), N_median=float(np.median(L['photons'])),
                            bg_median=float(np.median(L['bg'])), z_median_nm=float(np.median(L['net_gradient']))),
                seconds=time.time() - t0)
    log(tag, json.dumps(summ))
    json.dump(dict(summary=summ, dphi=dphi[ok].tolist(), g2=g2.numpy()[k].tolist(), across_nm=drv.tolist(),
                   dphi_theirs=tdphi[tok].tolist(), g2_theirs=L['ellipticity'].tolist()),
              open(os.path.join(OUT, f'baseline_dna_{tag}.json'), 'w'))


if __name__ == '__main__':
    stage, rest = sys.argv[1], sys.argv[2:]
    var = variant(rest)
    tag = '_'.join(rest) or 'base'
    {'pmma': run_pmma, 'dna': run_dna}[stage](var, tag)
