"""Dense bead stacks: detection / grouping of overlapping beads, the joint cluster fit recovers known Zernikes, the
isolated-bead estimator on the same field for comparison, and the held-out scoring."""
import numpy as np
import torch
from conftest import DEV, needs_gpu
from torch_psf import VectorialPSF, ZStackEstimator, DenseBeadEstimator
from torch_psf.estimation import stage_focus
from torch_psf.estimation.dense_beads import (find_bead_clusters, field_stop_mask, simulate_dense_bead_stack,
                                              dense_nll, BeadClusters, psf_extent)

TRUE = {5: 0.05, 6: 0.02, 7: -0.03, 8: 0.02, 11: 0.04}
OPT = dict(pixel_um=0.1, NA=1.49, wavelength_um=0.67)
PUP = dict(npupil=32, oversample=2)
# a 96 x 96 field: one isolated bead, a pair 1.1 um apart, a triplet (1.0 / 1.3 um) and a pair 1.4 um apart
XY = np.array([[20.3, 22.6], [70.2, 18.4], [79.6, 23.9], [24.1, 70.5], [33.8, 72.9], [26.0, 80.7],
               [66.7, 66.1], [76.4, 75.8]])


def _sim(seed=0, nk=21):
    truth = VectorialPSF(29, device=DEV, **OPT, **PUP).set_zernike(TRUE)
    focus = stage_focus(nk, 1.0 / (nk - 1), 1.0).to(DEV)
    sim = simulate_dense_bead_stack(truth, focus, XY, 96, 96, photons=30000.0, bg=30.0, seed=seed)
    return truth, focus, sim


def _err(psf, truth):
    return float((psf.zcoef.detach() - truth.zcoef.detach()).pow(2).mean().sqrt())


def test_grouping_faint_and_field_stop():
    xy = np.array([[20.0, 20.0], [30.0, 20.0], [60.0, 60.0], [25.0, 27.0], [3.0, 60.0], [90.0, 30.0]])
    amp = np.array([100.0, 90.0, 80.0, 5.0, 70.0, 60.0])
    kept, dropped = find_bead_clusters(xy, amp, 15, (100, 100), faint=0.2)
    sizes = sorted(len(c['xy']) for c in kept)
    assert sizes == [1, 1, 3]                                   # the faint one joins the pair, does not group
    pair = [c for c in kept if len(c['xy']) == 3][0]
    assert pair['faint'].tolist() == [False, False, True] and pair['box'] == (13, 13, 38, 28)
    assert [c['why'] for c in dropped] == ['frame']
    mask = np.zeros((100, 100), bool)
    mask[:, :85] = True
    kept, dropped = find_bead_clusters(xy, amp, 15, (100, 100), mask=mask, faint=0.2)
    assert sorted(c['why'] for c in dropped) == ['field_stop', 'frame']
    # a dark field stop around a bright disc is found; a flat frame has none
    yy, xx = np.mgrid[:80, :80]
    img = np.where((xx - 40) ** 2 + (yy - 40) ** 2 < 30 ** 2, 50.0, 2.0)[None].repeat(3, 0)
    m = field_stop_mask(img, margin=2)
    assert m[40, 40] and not m[2, 2] and not m[40, 70] and m[40, 66]
    assert field_stop_mask(np.full((3, 50, 50), 40.0)).all()


@needs_gpu
def test_dense_fit_recovers_zernikes_and_beats_isolated():
    truth, focus, sim = _sim()
    est = DenseBeadEstimator(optics=OPT, patch=15, render_pad=5, iters=250, field_stop=False, device=DEV,
                             npupil=32, oversample_pupil=2)
    psf = est.estimate([sim['d']], 0.05)
    rep = est.report
    assert rep['n_beads'] == 8 and rep['cluster_sizes'] == {1: 1, 2: 2, 3: 1}, rep
    e_dense = _err(psf, truth)
    assert e_dense < 0.003, e_dense                            # a few milli-waves
    assert 0.9 < rep['deviance_per_px'] < 1.2, rep
    # the isolated-bead estimator on the same field: its ROIs (same size) hold parts of the neighbours
    iso = ZStackEstimator('vectorial', optics=OPT, roisize=15, nbeads=8, iters=250, device=DEV, npupil=32,
                          oversample_pupil=2)
    p_iso = iso.estimate([sim['d'].cpu().numpy()], 0.05)
    e_iso = _err(p_iso, truth)
    assert e_iso > 2 * e_dense, (e_iso, e_dense)


@needs_gpu
def test_heldout_clusters_scored():
    truth, focus, sim = _sim(seed=1)
    est = DenseBeadEstimator(optics=OPT, patch=15, render_pad=5, iters=150, field_stop=False, holdout_fraction=0.5,
                             holdout_seed=3, device=DEV, npupil=32, oversample_pupil=2)
    psf = est.estimate([sim['d']], 0.05)
    rep = est.report
    assert rep['heldout_n_clusters'] == 2 and 'heldout_nll_per_px' in rep
    ho = est.data.subset(np.nonzero(est.result['heldout'])[0])
    start = VectorialPSF(15, device=DEV, **OPT, **PUP)
    s_true = dense_nll(ho, est.focus_um, truth, iters=100, **est.fit_kw)['report']['nll_per_px']
    s_fit = dense_nll(ho, est.focus_um, psf, iters=100, **est.fit_kw)['report']['nll_per_px']
    s_start = dense_nll(ho, est.focus_um, start, iters=100, **est.fit_kw)['report']['nll_per_px']
    assert s_start > s_fit and abs(s_fit - s_true) < 0.02 * abs(s_start - s_true), (s_true, s_fit, s_start)


def test_bead_clusters_layout_and_extent():
    truth, focus, sim = _sim()
    kept, _ = find_bead_clusters(XY, np.ones(len(XY)), 15, (96, 96))
    for c in kept:
        c['stack'] = 0
    data = BeadClusters.from_stacks([sim['d']], kept, 15, DEV, render_pad=3)
    assert data.n_beads == 8 and data.R == 21 and data.idx.shape == (8, 21 * 21)
    assert data.d.shape == (21, data.n_px) and int(data.idx.max()) < data.n_px
    c = int(np.argmax(data.cluster_sizes()))
    img = data.unflatten(data.d, c)
    x0, y0, x1, y1 = kept[c]['box']
    m = torch.as_tensor(data.masks[c], device=DEV)
    assert torch.equal(img[:, m], sim['d'][:, y0:y1, x0:x1][:, m])
    P = psf_extent(truth, focus)
    assert 11 <= P <= 41 and P % 2 == 1
