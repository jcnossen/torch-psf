"""Native first-order field parity and the two in-situ evaluation paths."""
import numpy as np
from pathlib import Path
import pytest
import torch
from torch_psf import VectorialPSF, InSituEstimator
from torch_psf.estimation import fit_nat, nat_polynomial


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_native_field_grid():
    fixture = Path(__file__).parents[1] / "method_comparison/droste/vectorfit_field_reference.csv"
    data = np.loadtxt(fixture, delimiter=",", skiprows=1)
    for k, g in enumerate(torch.eye(13, dtype=torch.float64), start=1):
        rows = data[data[:, 0] == k]
        u, v = rows[:, 1:3].T
        basis = torch.from_numpy(np.stack([np.ones_like(u), v, v*v, u, u*v, u*u], axis=1))
        torch.testing.assert_close(basis @ nat_polynomial(g).T, torch.from_numpy(rows[:, 3:]), rtol=0, atol=2e-14)


def test_web_reference(single_thread):
    import json
    from make_web_nat_reference import reference

    recorded = json.loads((Path(__file__).parents[1] / "web/test/nat_reference.json").read_text())
    current = reference(recorded["data"])
    np.testing.assert_allclose(current["expected"], recorded["expected"], rtol=0, atol=3e-4)
    # Float32 CPU backends differ most on the independent defocus cross term.
    delta = np.asarray(current["independent"]) - recorded["independent"]
    assert np.max(np.abs(delta)) < 5e-4
    assert np.sqrt(np.mean(delta**2)) < 1e-4
    np.testing.assert_allclose(current["nll"], recorded["nll"], rtol=1e-5)
    np.testing.assert_allclose(current["pixels"], recorded["pixels"], rtol=1e-4, atol=1e-7)


def test_native_field_columns():
    # Vectorfit 0.3.4 compute_xy_gamma_legendre at u=.4, v=-.7, mode order
    # (2,0),(2,-2),(2,2),(3,-1),(3,1),(4,0), for unit gamma_1 ... gamma_13.
    ref = torch.zeros(13, 6, dtype=torch.float64)
    ref[0, 0], ref[1, 0], ref[2, 0], ref[3, 0] = (
        1,
        0.692820323,
        -1.212435565,
        -0.055901699,
    )
    ref[4, 2] = ref[5, 1] = 1
    ref[6, 1], ref[6, 2] = -1.212435565, 0.692820323
    ref[7, 1], ref[7, 2] = 0.692820323, 1.212435565
    ref[8, 1], ref[8, 2] = -0.84, -0.495
    ref[9, 4] = ref[10, 3] = ref[12, 5] = 1
    ref[11, 3], ref[11, 4] = -1.212435565, 0.692820323
    features = torch.tensor([1, -0.7, 0.49, 0.4, -0.28, 0.16], dtype=torch.float64)
    actual = torch.stack([nat_polynomial(g) @ features for g in torch.eye(13, dtype=torch.float64)])
    torch.testing.assert_close(actual, ref, atol=1e-8, rtol=0)
    paper = torch.stack(
        [nat_polynomial(g, convention="paper") @ features for g in torch.eye(13, dtype=torch.float64)]
    )
    ref[8] *= np.sqrt(15 / 7)
    norms = ref.new_tensor([3, 6, 6, 8, 8, 5]).sqrt()
    torch.testing.assert_close(paper, ref / norms, atol=1e-8, rtol=0)


@pytest.mark.parametrize("backend, field_varying", [("direct", True), ("bspline", True), ("bspline", False)])
def test_fit_from_missing_defocus(backend, field_varying, single_thread):
    psf = VectorialPSF(
        9,
        npupil=16,
        oversample=2,
        zernike_indices=[5, 6, 7, 8, 11],
        z_range=(0.0, 0.2),
        focus_um=0.1,
        device="cpu",
    )
    original = psf.zcoef.detach().clone()
    xy = torch.tensor([[x, y] for x in [4, 12, 20, 28] for y in [4, 12, 20, 28]])
    pos = torch.tensor([[0.15, -0.11, 0.1]]).repeat(len(xy), 1)
    true = psf.resized(9).set_zernike({6: 0.09})
    with torch.no_grad():
        data = 2000 * true.render(pos) + 8
    result = fit_nat(
        psf,
        data,
        xy,
        norm=(16, 16, 16, 16),
        backend=backend,
        field_varying=field_varying,
        tile=16,
        iters=2,
        global_iters=2,
        zstarts=2,
        batch=8,
        oversample=2,
        dz_um=0.05,
    )
    assert psf.zidx == [5, 6, 7, 8, 11]
    torch.testing.assert_close(psf.zcoef.detach(), original)
    assert len(result["report"]["gamma"]) == 13
    assert np.isfinite(result["report"]["nll"])
    assert result["report"]["nll"] < result["history"][0]["local_nll"]
    assert result["theta"].shape == (16, 5)
    assert 4 in result["psf"].zidx
    if not field_varying:
        coef = result["psf"].zernikes_at(xy)
        np.testing.assert_array_equal(coef, np.broadcast_to(coef[0], coef.shape))
        assert np.any(coef[0] != 0)
    assert InSituEstimator(method="nat", device="cpu").field["model"] == "nat"


def test_movie_nat_path(single_thread):
    psf = VectorialPSF(
        9,
        npupil=16,
        oversample=2,
        zernike_indices=[5, 6, 11],
        z_range=(0.0, 0.2),
        focus_um=0.1,
        device="cpu",
    ).set_zernike({6: 0.08})
    movie = torch.full((3, 64, 64), 8.0)
    with torch.no_grad():
        image = psf.render(torch.tensor([[0.1, -0.2, 0.08]]))[0] * 5000
        for x in [10, 26, 42]:
            for y in [10, 26, 42]:
                movie[:, y - 4 : y + 5, x - 4 : x + 5] += image
    est = InSituEstimator(
        method="nat",
        start=psf,
        iters=1,
        nspots=12,
        nheldout=8,
        roisize=9,
        z_range=(0.0, 0.2),
        device="cpu",
        zstarts=1,
        dz_um=0.05,
        reject_overlap=False,
        field=dict(
            tile=64, iters=1, global_iters=1, dz_um=0.05, oversample=2, max_rois=8, gamma0=[[0.0] * 13] * 2
        ),
    )
    est.estimate(movie.numpy())
    report = est.report["field"]
    assert report["n_emitters"] == 8 and len(report["start_nll"]) == 2
    assert report["nll"] == min(report["start_nll"])
    assert np.isfinite(est.report["heldout"]["field"]["nll_per_px"])
