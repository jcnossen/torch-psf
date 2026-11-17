"""Shared settings for the SOFI-PSF simulation study (study/sofi)."""
import os, sys, json, math, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # this worktree's torch_psf first
import torch_psf as tp
from torch_psf import sofi
from torch_psf.sofi import BlinkModel, CameraModel, DenseMovieSpec, DenseMovie, CumulantAccumulator, GridPSF

OUT = os.environ.get('SOFI_OUT', os.path.join(os.path.expanduser('~'), 'sofi_psf_out'))   # generated data: not in git
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda'

OPTICS = dict(pixel_um=0.108, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68)
NOLL = list(range(5, 22))


def make_psf(Q=13, zern=None, astig=0.0, focus_um=0.0, z_range=(0.0, 1.0), npupil=64, oversample=3, device=DEV,
             n_med=None):
    """VectorialPSF with Noll 5-21; zern: dict or vector of the true aberrations (waves rms), astig: extra Noll 6
    (vertical astigmatism, waves rms) bias."""
    psf = tp.VectorialPSF(Q, zernike_indices=tuple(NOLL), npupil=npupil, oversample=oversample, focus_um=focus_um,
                          z_range=z_range, device=device, **{**OPTICS, **({} if n_med is None else dict(n_med=n_med))})
    zc = np.zeros(len(NOLL)) if zern is None else (np.asarray([zern.get(j, 0.0) for j in NOLL]) if isinstance(zern, dict)
                                                    else np.asarray(zern, float))
    zc = zc.copy()
    zc[NOLL.index(6)] += astig
    psf.set_zernike(zc)
    return psf


def random_aberrations(rms_total=0.05, seed=0):
    """Random Noll 5-21 coefficients with a total rms of rms_total waves (falling spectrum ~ 1 / n)."""
    rng = np.random.default_rng(seed)
    n = np.array([tp.zernike.noll_to_nm(j)[0] for j in NOLL], float)
    c = rng.normal(size=len(NOLL)) / n
    return c * rms_total / np.sqrt((c ** 2).sum())


def save_json(obj, name):
    p = os.path.join(OUT, 'results', name)                     # generated: not in git
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return p
