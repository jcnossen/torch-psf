"""Spot detection recall / precision on a simulated blinking movie."""
import numpy as np
import torch
from conftest import DEV
from torch_psf import GaussianPSF, extract_spots
from torch_psf.simulate import simulate_movie


def test_detection_recall_on_simulated_movie():
    psf = GaussianPSF(11, sigma_px=1.3, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(3)
    movie, truth = simulate_movie(psf.render, 11, T=300, H=96, W=96, nsites=150, p_on=0.02, mean_on=2.0, bg=20.0,
                                  photons=2000.0, photons_logsd=0.2, z_range=(0.0, 0.0), margin=8, generator=g,
                                  device=DEV)
    sp = extract_spots(movie, 11, device=DEV, threshold=6.0)
    det = np.stack([sp.frame.cpu().numpy(), (sp.corner[:, 0] + 5).cpu().numpy(), (sp.corner[:, 1] + 5).cpu().numpy()], 1)
    tr = truth.cpu().numpy()
    inside = (tr[:, 1] > 8) & (tr[:, 1] < 87) & (tr[:, 2] > 8) & (tr[:, 2] < 87)
    tr = tr[inside]
    # isolated true events only (no other event within 7 px in the same frame): those must be found
    found, iso = 0, 0
    by_frame = {}
    for i, f in enumerate(det[:, 0]):
        by_frame.setdefault(int(f), []).append(i)
    for f, x, y in tr[:, :3]:
        same = tr[(tr[:, 0] == f)]
        if (np.hypot(same[:, 1] - x, same[:, 2] - y) < 7).sum() > 1:
            continue
        iso += 1
        idx = by_frame.get(int(f), [])
        if idx and (np.hypot(det[idx, 1] - x, det[idx, 2] - y) < 1.5).any():
            found += 1
    assert found / iso > 0.95, (found, iso)
    # precision: every detection near a true event of its frame
    good = 0
    for f, x, y in det:
        same = truth.cpu().numpy()
        same = same[same[:, 0] == f]
        good += bool(len(same) and (np.hypot(same[:, 1] - x, same[:, 2] - y) < 2.5).any())
    assert good / len(det) > 0.97


def test_reference_movie_is_deterministic():
    from torch_psf.simulate import simulate_reference_movie
    a = simulate_reference_movie('astigmatic', T=40, seed=3, device=DEV)
    b = simulate_reference_movie('astigmatic', T=40, seed=3, device=DEV)
    assert np.array_equal(a['movie_adu'], b['movie_adu']) and np.array_equal(a['truth'], b['truth'])
    on = len(a['truth']) / 40
    assert 2 < on < 9, on
    sp = extract_spots(a['movie_adu'], 15, a['camera'], device=DEV)
    assert len(sp) > 0.3 * len(a['truth'])      # border (9 px) and defocused spots are not detected
