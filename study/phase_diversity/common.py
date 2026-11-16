"""Shared pieces of the phase-diversity study on the 2-colour bead stacks (left channel): loading, the held-out
cluster set of the dense-beads study (beads2.tif, scored by dense_nll exactly as study/dense_beads/real_fit.py) and
the scoring of a PSF on it. The data are read as plain arrays (tifffile); nothing in the data folder is executed."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import tifffile
import torch
from torch_psf import Camera, VectorialPSF
from torch_psf.estimation import stage_focus
from torch_psf.estimation.dense_beads import (field_stop_mask, find_all_beads, find_bead_clusters, BeadClusters,
                                              dense_nll)

DATA = '/mnt/data/multicolor/june/astig/2col/beads100nm_10nm_steps'
# assumptions of docs/notes/dense_beads.md
OPT = dict(pixel_um=0.097, NA=1.49, wavelength_um=0.67, n_imm=1.518, n_med=1.33)
CAM = Camera(gain_e_per_adu=0.597, offset_adu=100.6, readnoise_e=0.0)
REGION = (0, 0, 475, 400)
DZ = 0.010
# dense-beads settings of the real-data runs (run_real.sh / real_variants.py)
DENSE = dict(patch=21, plane_step=2, max_box=84, max_members=10, threshold=6.0, faint=0.1, focus_offset='bead',
             bg='linear', reject=3.0, holdout=1)
HELDOUT_DROPPED = [48, 59]       # real_fit.py: held-out clusters with chi2 > 3 x the training median (aggregates)


def load(dev='cuda', files=('beads.tif', 'beads2.tif')):
    x0, y0, x1, y1 = REGION
    return [CAM.to_photons(torch.as_tensor(tifffile.imread(os.path.join(DATA, f))[:, y0:y1, x0:x1].astype(np.float32),
                                           device=dev)) for f in files]


def clusters(st, dev='cuda'):
    """BeadClusters of both stacks with the dense-beads settings (as DenseBeadEstimator.estimate builds them), the
    focus positions of the used planes and the held-out cluster ids (beads2, minus the dropped aggregates)."""
    ps = DENSE['plane_step']
    st2 = [s[::ps] for s in st]
    nk = st[0].shape[0]
    focus = stage_focus(nk, DZ, 1.0).to(dev)[::ps]
    P = DENSE['patch']
    cl = []
    for i, a in enumerate(st2):
        an = a.cpu().numpy()
        mask = field_stop_mask(an)
        xy, amp = find_all_beads(an, threshold=DENSE['threshold'], min_dist=3)
        kept, _ = find_bead_clusters(xy, amp, P, an.shape[1:], mask=mask, max_members=DENSE['max_members'],
                                     max_box=DENSE['max_box'], faint=DENSE['faint'])
        for c in kept:
            c['stack'] = i
        cl += kept
    data = BeadClusters.from_stacks(st2, cl, P, dev, render_pad=P // 3)
    ho = [c for c in range(data.n_clusters) if data.clusters[c]['stack'] == 1 and c not in HELDOUT_DROPPED]
    return data, focus, ho


FIT_KW = dict(drift=True, drift_order=3, focus_offset='bead', bg='linear', intensity='free', fit_dz_scale=False,
              emitter_z_um=0.0, max_renders=3000)


def score(ho_data, focus, psf, iters=200, **kw):
    """dense_nll on the held-out clusters (as real_fit.py): deviance / NLL per pixel, chi2 median, the result."""
    r = dense_nll(ho_data, focus, psf, iters=iters, **{**FIT_KW, **kw})
    rep = r['report']
    return dict(deviance_per_px=rep['deviance_per_px'], nll_per_px=rep['nll_per_px'],
                chi2_median=rep['chi2_median'], n_clusters=ho_data.n_clusters, n_beads=ho_data.n_beads), r
