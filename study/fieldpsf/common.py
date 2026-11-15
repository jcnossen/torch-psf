"""Shared pieces of the field-dependent PSF study on the FD-DeepLoc data (Fu et al., Nat. Methods 2023).

Data paths are always passed on the command line (the data are untrusted input; run the scripts with python -I).
Under python -I PYTHONPATH is ignored, so every script puts the worktree root on sys.path itself (see `setup_path`).

Their conventions (docs/notes/field_psf_study.md has the sources):
  - aberrations: 21 Zernike modes in the (n, m) order THEIR_NM below; per-bead fits in nm, aber_map in waves
    (nm / 680); m < 0 sine, m > 0 cosine;
  - aber_map [1608, 1608, 23] (MATLAB order [row, col, ch]): ch 0..20 the 21 modes (waves), ch 21 / 22 the image
    blur sigma x / y (constant over the field); aber_map[y - 1, x - 1, :21] = value_P / 680 at their bead pos (x, y),
    1-based;
  - optics: NA 1.35, n_imm = n_med = 1.406 (silicone), n_cov 1.525, lambda 680 nm, pixel 108 nm.
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def setup_path():
    """put the worktree root first on sys.path (python -I ignores PYTHONPATH)"""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


setup_path()
from torch_psf.optics import Optics            # noqa: E402
from torch_psf.zernike import noll_to_nm       # noqa: E402

LAMBDA_NM = 680.0
THEIR_NM = [(2, -2), (2, 2), (3, -1), (3, 1), (4, 0), (3, -3), (3, 3), (4, -2), (4, 2), (5, -1), (5, 1), (6, 0),
            (4, -4), (4, 4), (5, -3), (5, 3), (6, -2), (6, 2), (7, 1), (7, -1), (8, 0)]
_NM2NOLL = {noll_to_nm(j): j for j in range(1, 80)}
THEIR_NOLL = [_NM2NOLL[t] for t in THEIR_NM]   # [5, 6, 7, 8, 11, 9, 10, 13, 12, 17, 16, 22, 15, 14, 19, 18, 23, 24, 30, 29, 37]
MODE_NAMES = {5: 'astig 45', 6: 'astig 0', 7: 'coma y', 8: 'coma x', 9: 'trefoil y', 10: 'trefoil x', 11: 'spherical',
              12: '2nd astig 0', 13: '2nd astig 45', 14: 'quadrafoil 0', 15: 'quadrafoil 45', 16: '2nd coma x',
              17: '2nd coma y', 18: '2nd trefoil x', 19: '2nd trefoil y', 22: '2nd spherical', 23: '3rd astig 45',
              24: '3rd astig 0', 29: '3rd coma y', 30: '3rd coma x', 37: '3rd spherical'}

OPTICS = Optics(pixel_um=0.108, NA=1.35, wavelength_um=0.68, n_imm=1.406, n_med=1.406)


def mode_label(j):
    n, m = noll_to_nm(j)
    return f'Z{j} ({n},{m:+d}) {MODE_NAMES.get(j, "")}'.strip()


def their_waves_to_ours(c_theirs, signs=None):
    """their 21 coefficients (waves, THEIR_NM order) -> ours (waves rms, Noll THEIR_NOLL order): identical
    normalization, times a per-mode sign (signs [21]; None = +1) for a convention difference."""
    c = np.asarray(c_theirs, np.float64)
    return c if signs is None else c * np.asarray(signs)


def zflip_signs(nm=THEIR_NM):
    """the coefficient change that is equivalent to flipping the focus axis (n_med = n_imm, real pupil amplitude):
    c_j -> -(-1)^m c_j"""
    return np.array([-((-1) ** abs(m)) for n, m in nm], np.float64)


def load_their_beads(prefix):
    """their per-bead calibration of one bead stack (files prefix + '_maps.mat', prefix + '_3dcal.mat'):
    dict(pos [n, 2] (x, y) 0-based px, coef [n, 21] waves, z_pos [n] nm, sigmaxy [n, 2], mean [21] waves (their
    global fit of the stack), objstage [nk] nm)"""
    import scipy.io as sio
    m = sio.loadmat(prefix + '_maps.mat', squeeze_me=True, struct_as_record=False)['beads_zrnikecoeff']
    c = sio.loadmat(prefix + '_3dcal.mat', squeeze_me=True, struct_as_record=False)['SXY'].zernikefit
    pos = np.array([p for p in m.pos], np.float64)[:, :2] - 1.0
    coef = np.array([v for v in m.value_P], np.float64) / LAMBDA_NM
    model = np.asarray(m.model)
    assert [tuple(r) for r in model.tolist()] == THEIR_NM, model
    return dict(pos=pos, coef=coef, z_pos=np.array([z for z in m.z_pos], np.float64)[:, 0],
                sigmaxy=np.array([s for s in m.sigmaxy], np.float64), mean=np.asarray(c.aberrations)[:, 2] / LAMBDA_NM,
                objstage=np.asarray(c.objStageStack, np.float64))


def load_aber_map(path, cache_dir=None):
    """aber_map [1608, 1608, 23] float32 in [row, col, ch] order (MATLAB indexing); a .npy cache in cache_dir"""
    if cache_dir is not None:
        key = os.path.basename(os.path.dirname(path)) + '__' + os.path.basename(path) + '.npy'
        cp = os.path.join(cache_dir, key)
        if os.path.exists(cp):
            return np.load(cp, mmap_mode='r')
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            A = np.ascontiguousarray(f['aber_map'][:].transpose(2, 1, 0))
    except OSError:
        import scipy.io as sio
        A = sio.loadmat(path)['aber_map']
    A = A.astype(np.float32)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cp, A)
    return A


def map_at(A, x, y, half=0):
    """aber_map channels 0..20 at 0-based (x, y) (nearest px), or the mean over a (2 half + 1)^2 window"""
    xi, yi = int(round(x)), int(round(y))
    H, W = A.shape[:2]
    y0, y1 = max(yi - half, 0), min(yi + half + 1, H)
    x0, x1 = max(xi - half, 0), min(xi + half + 1, W)
    return np.asarray(A[y0:y1, x0:x1, :21], np.float64).reshape(-1, 21).mean(0)


def compare_modes(ours, theirs, noll=THEIR_NOLL):
    """per-mode agreement of two sets [n, 21] (waves): mean diff, rms diff, std of each over the set, Pearson r
    (over the set; nan if one is constant) and the slope of a least-squares fit theirs = a ours + b"""
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    out = []
    for k, j in enumerate(noll):
        a, b = ours[:, k], theirs[:, k]
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
        slope = float(np.polyfit(a, b, 1)[0]) if a.std() > 0 else float('nan')
        out.append(dict(noll=j, nm=list(noll_to_nm(j)), label=mode_label(j), mean_ours=float(a.mean()),
                        mean_theirs=float(b.mean()), mean_diff=float((a - b).mean()),
                        rms_diff=float(np.sqrt(((a - b) ** 2).mean())), std_ours=float(a.std()),
                        std_theirs=float(b.std()), r=r, slope=slope))
    return out


def fmt_table(rows, keys, fmt='%.4f'):
    """markdown table of a list of dicts"""
    head = '| ' + ' | '.join(keys) + ' |\n|' + '---|' * len(keys) + '\n'
    body = ''
    for r in rows:
        body += '| ' + ' | '.join((fmt % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + ' |\n'
    return head + body


def their_to_ours_matrix(transpose=True, zflip=False, npupil=256):
    """M [21, 21] with c_ours (Noll THEIR_NOLL order, our pupil axes px = image column, py = row) = M @ c_theirs
    (THEIR_NM order). Their vectorial model's pupil x is the image row (first array index), so their angle is
    phi_FD = atan2(col, row) = pi / 2 - phi_ours (transpose=True); transpose=False: the same axes as ours.
    zflip: also flip the focus axis (c_j -> -(-1)^m c_j). Computed by projecting their Zernikes, evaluated on our
    grid, onto ours (an exact signed permutation)."""
    import torch
    from torch_psf.zernike import zernike
    g = (torch.arange(npupil, dtype=torch.float64) - (npupil - 1) / 2) / (npupil / 2)
    px, py = g[None, :].expand(npupil, npupil), g[:, None].expand(npupil, npupil)
    rho = torch.sqrt(px ** 2 + py ** 2)
    msk = rho <= 1
    phi_o = torch.atan2(py, px)
    phi_t = torch.atan2(px, py) if transpose else phi_o
    Zo = torch.stack([zernike(j, rho, phi_o)[msk] for j in THEIR_NOLL])       # our basis
    Zt = torch.stack([zernike(j, rho, phi_t)[msk] for j in THEIR_NOLL])       # their basis on our grid
    M = torch.linalg.lstsq(Zo.T, Zt.T).solution.numpy()                         # Zt_k = sum_j M[j, k] Zo_j
    M[np.abs(M) < 1e-3] = 0.0
    M = np.round(M, 3)
    if zflip:
        M = np.diag(zflip_signs()) @ M
    return M


def theirs_to_ours(c_theirs, **kw):
    """their coefficients [..., 21] (waves) -> ours [..., 21] (waves rms, Noll THEIR_NOLL order)"""
    return np.asarray(c_theirs, np.float64) @ their_to_ours_matrix(**kw).T
