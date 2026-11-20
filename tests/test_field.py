"""Field-dependent PSFs (torch_psf.field): the field polynomial (features, per-mode order, clipping, the psf.zip
'zernike_poly' block), the vectorial and tiled-spline field models (render, fit, psf.zip round trip), recovery of a
known field polynomial by the in-situ knot route and the dense-bead fit, and the matched-filter detector."""
import numpy as np
import pytest
import torch
from conftest import DEV, needs_gpu
from torch_psf import VectorialPSF, SplinePSF, PSF, io
from torch_psf.field import FieldZernikes, FieldPSF, TiledSplinePSF, field_from_spec, feature_powers
from torch_psf.psfzip import PSFZip, FieldZernikePoly, poly_features
from torch_psf.estimation import fit_field_knots, compare_heldout, stage_focus
from torch_psf.matched_filter import MatchedFilterDetector
from torch_psf.spots import extract_spots

PUP = dict(npupil=32, oversample=2)


def vec(R=11, device=DEV, **kw):
    kw.setdefault('pixel_integration', 'midpoint')
    v = VectorialPSF(R, pixel_um=0.1, NA=1.4, zernike_indices=(5, 6, 7, 11), device=device, focus_um=0.3,
                     z_range=(0.0, 0.6), **PUP, **kw)
    return v.set_zernike({6: 0.1, 7: 0.02})


def field(device=DEV):
    """Z5 linear in x (+0.04 waves at the right edge), Z6 linear in y (-0.03 at the bottom), on 256 x 256 px"""
    f = FieldZernikes([5, 6], order=1, norm=(128.0, 128.0, 128.0, 128.0), device=device)
    with torch.no_grad():
        f.coef.copy_(torch.tensor([[0.0, 0.04], [-0.03, 0.0]]))           # features (v, u)
    return f


def test_field_zernikes_features_orders_clip_and_block():
    norm = (100.0, 80.0, 50.0, 40.0)
    f = FieldZernikes([5, 6, 11], order={5: 1, 6: 2, 11: 0}, norm=norm, prior_sd=0.02)
    assert f.order == 2 and f.orders == [1, 2, 0] and f.n_params == 2 + 5
    assert feature_powers(2) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
    with torch.no_grad():
        f.coef.copy_(torch.arange(1, 16, dtype=torch.float32).reshape(3, 5) / 100)
    f.project_()
    c = f.coef.detach().numpy()
    assert np.all(c[0, [1, 3, 4]] == 0) and np.all(c[0, [0, 2]] != 0) and np.all(c[1] != 0) and np.all(c[2] == 0)
    x, y = np.array([60.0, 100.0, 140.0]), np.array([50.0, 80.0, 110.0])
    F = poly_features(x, y, 2, norm)
    assert np.allclose(f.features(torch.tensor(np.stack([x, y], 1))).numpy(), F, atol=1e-6)
    # relative to the reference point (the centre): zero there
    assert torch.allclose(f.relative(torch.tensor([[100.0, 80.0]])), torch.zeros(1, 3))
    # no extrapolation: outside the box the value is that at the box edge (per coordinate)
    assert torch.allclose(f(torch.tensor([[400.0, -300.0]])), f(torch.tensor([[150.0, 40.0]])))
    pen = float(f.penalty().detach())
    assert abs(pen - 0.5 * float(((f.coef * f.mask / 0.02) ** 2).sum())) < 1e-6
    # psf.zip block: coef[:, 0] = the model's value at the reference point; the numpy reader agrees, also clipped
    blk = f.to_block({5: 0.01, 6: 0.08, 11: -0.02})
    assert blk['order'] == 2 and blk['zernike_noll'] == [5, 6, 11] and 'reference_point' not in blk
    assert np.allclose(np.asarray(blk['coef'])[:, 0], [0.01, 0.08, -0.02])
    rd = FieldZernikePoly(blk)
    pts = np.array([[60.0, 50.0], [130.0, 101.0], [500.0, 90.0], [-20.0, -40.0]])
    W = rd.zernike_at(pts[:, 0], pts[:, 1])
    ref = np.array([0.01, 0.08, -0.02]) + f.relative(torch.tensor(pts)).detach().numpy()
    assert np.allclose(W, ref, atol=1e-6)
    g, const = FieldZernikes.from_block(blk)
    assert g.orders == [1, 2, 0] and np.allclose(const, [0.01, 0.08, -0.02])
    assert torch.allclose(g.relative(torch.tensor(pts)).detach(), f.relative(torch.tensor(pts)).detach(), atol=1e-6)
    # another reference point
    f2 = FieldZernikes([6], order=1, norm=norm, ref=(120.0, 60.0))
    with torch.no_grad():
        f2.coef.fill_(0.02)
    b2 = f2.to_block({6: 0.08})
    assert b2['reference_point'] == dict(x=120.0, y=60.0)
    assert np.allclose(FieldZernikePoly(b2).zernike_at(120.0, 60.0), [0.08])
    # spec of the estimators
    fz, s = field_from_spec(dict(order={5: 2}, modes=[5, 6]), (5, 6, 7), torch.tensor([[0.0, 0.0], [50.0, 20.0]]))
    assert fz.orders == [2, 0] and fz.norm == (25.0, 10.0, 25.0, 10.0) and s['score_tile'] == 128.0
    with pytest.raises(ValueError):
        field_from_spec(dict(bad=1), (5,), torch.zeros(2, 2))


def test_field_psf_render_fit_and_zip(tmp_path):
    v, f = vec(13), field()
    fp = FieldPSF(v, f)
    xy = torch.tensor([[10.0, 20.0], [250.0, 40.0], [128.0, 128.0], [600.0, 500.0]], device=DEV)
    pos = torch.tensor([[0.2, -0.3, 0.2], [0.0, 0.1, 0.4], [-0.4, 0.2, 0.3], [0.1, 0.1, 0.1]], device=DEV)
    r = fp.render(pos, xy=xy)
    for i in range(len(xy)):
        assert torch.allclose(r[i], fp.at(*xy[i].tolist()).render(pos[i:i + 1])[0], atol=1e-7)
    assert torch.allclose(fp.render(pos[2:3]), v.render(pos[2:3]))       # no xy: the reference point (the centre)
    z = fp.zernikes_at(xy)
    assert np.allclose(z[:, 0], [0.04 * (10 - 128) / 128, 0.04 * (250 - 128) / 128, 0.0, 0.04], atol=1e-6)
    assert np.allclose(z[:, 1], 0.1 - 0.03 * np.clip((xy[:, 1].cpu().numpy() - 128) / 128, -1, 1), atol=1e-6)
    val, gx, gy, gz = fp.render(pos, grad=True, xy=xy)
    eps = 1e-2
    d = torch.zeros_like(pos); d[:, 2] = eps
    num = (fp.render(pos + d, xy=xy) - fp.render(pos - d, xy=xy)) / (2 * eps)
    assert float((num - gz).abs().max()) < 2e-3 * float(gz.abs().max())
    # psf.zip: PSF.load and PSFZip.read return the field-dependent model
    meta = io.save_zip(tmp_path / 'f.zip', fp, zstack_planes=3)
    assert meta['channels'][0]['field']['type'] == 'zernike_poly' and 'field' in meta['report']['sections']
    p2 = PSF.load(tmp_path / 'f.zip', device=DEV)
    assert isinstance(p2, FieldPSF) and p2.field.orders == [1, 1]
    assert torch.allclose(p2.render(pos, xy=xy), r, atol=1e-6)
    pz = PSFZip.read(tmp_path / 'f.zip', device=DEV)
    assert isinstance(pz.psf, FieldPSF) and pz.channels[0].psf is pz.psf
    assert torch.allclose(pz.channels[0].model_at(250.0, 40.0).render(pos), fp.at(250.0, 40.0).render(pos), atol=1e-6)
    # fit: every ROI at its own field position (the ROI centre)
    N = 120
    g = torch.Generator().manual_seed(0)
    fxy = (torch.rand(N, 2, generator=g) * 256).to(DEV)
    pp = torch.cat([torch.rand(N, 2, generator=g) - 0.5, 0.1 + 0.4 * torch.rand(N, 1, generator=g)], 1).to(DEV)
    with torch.no_grad():
        rois = torch.poisson(3000 * fp.render(pp, xy=fxy) + 10)
    ds = fp.fit(rois, corners=fxy - 6, batch=50, zstarts=2, iters=15)
    assert len(ds) == N and torch.equal(ds['id'], torch.arange(N))
    ds0 = v.fit(rois, corners=fxy - 6, zstarts=2, iters=15)
    assert float(ds['loglik'].mean()) > float(ds0['loglik'].mean())     # the field model fits better
    zf = ds['pos'][:, 2].to(DEV) * 0.1
    assert float((zf - pp[:, 2]).abs().median()) < 0.03


@torch.no_grad()
def test_integrated_field_tiles_and_channels(tmp_path, monkeypatch):
    from torch_psf import ChannelsPSF, SplinePSF, psfzip

    fp = FieldPSF(vec(11, pixel_integration='exact'), field())
    tiles = fp.to_tiled(tile=128, z_range=(.2, .3), xy_limit=1)
    xy = torch.tensor(tiles.centers[[0, 3]], device=DEV, dtype=torch.float32)
    pos = torch.tensor([[.19, -.31, .241], [-.33, .12, .264]], device=DEV)
    ref = fp.render(pos, xy=xy)
    for i, point in enumerate(xy):
        vec_at = fp.at(*point.tolist())
        torch.testing.assert_close(ref[i], vec_at.render(pos[i:i+1])[0], atol=1e-7, rtol=0)
        vec_at.pixel_integration = 'midpoint'
        assert (ref[i] - vec_at.render(pos[i:i+1])[0]).abs().max() > 1e-4 * ref[i].max()
    expected = tiles.render(pos, xy=xy)
    assert torch.all((expected - ref).abs().amax((-2, -1)) / ref.amax((-2, -1)) < 2e-5)
    path = tmp_path / 'tiles.zip'
    assert io.save_zip(path, tiles, zstack_planes=3)['version'] == '0.4.0'
    loaded = PSF.load(path, device=DEV)
    assert isinstance(loaded.reference_spline(), SplinePSF)
    torch.testing.assert_close(loaded.render(pos, xy=xy), expected)
    models = [tiles.spline(0), tiles.spline(3)]
    channels = ChannelsPSF(models)
    assert torch.equal(channels.render(pos), torch.stack([s.render(pos) for s in models], 1))
    monkeypatch.setattr(psfzip, 'READ_VERSIONS', ((0, 2), (0, 3)))
    with pytest.raises(ValueError, match='version 0.4.0'):
        PSF.load(path)
    with pytest.raises(ValueError, match='kind'):
        fp.to_tiled(kind='unknown')


def test_tiled_spline_psf_fit_and_zip(tmp_path):
    fp = FieldPSF(vec(11), field())
    tp = fp.to_tiled(tile=64, roisize=11, oversample=2, dz_um=0.04, zpad=2, xy_limit=2.0)
    assert isinstance(tp, TiledSplinePSF) and tp.n_tiles == 16
    xy = torch.tensor([[32.0, 32.0], [224.0, 96.0], [-50.0, 300.0]], device=DEV)
    assert tp.tile_index(xy).tolist() == [0, 7, 12]
    pos = torch.tensor([[0.1, -0.2, 0.3]] * 3, device=DEV)
    ref = fp.render(pos, xy=torch.tensor([[32.0, 32.0], [224.0, 96.0], [32.0, 224.0]], device=DEV))
    assert float((tp.render(pos, xy=xy) - ref).abs().max()) < 0.02 * float(ref.max())     # spline at the tile centres
    assert len(tp._splines) == 3                                                           # built on first use
    N = 400
    g = torch.Generator().manual_seed(1)
    fxy = (torch.rand(N, 2, generator=g) * 256).to(DEV)
    pp = torch.cat([torch.rand(N, 2, generator=g) - 0.5, 0.1 + 0.4 * torch.rand(N, 1, generator=g)], 1).to(DEV)
    with torch.no_grad():
        rois = torch.poisson(3000 * fp.render(pp, xy=fxy) + 10)
    ds = tp.fit(rois, corners=fxy - 5)
    assert len(ds) == N
    x_true = (fxy[:, 0] + pp[:, 0]).double().cpu()
    assert float((ds['pos'][:, 0] - x_true).abs().median()) < 0.05      # input order kept
    ho = compare_heldout({'centre': fp.to_spline(11, oversample=2, dz_um=0.04, zpad=2, xy_limit=2.0), 'tiled': tp},
                         rois, xy=fxy)
    assert ho['tiled']['dnll_per_spot_vs_first'] < 0 and ho['tiled']['frac_better'] > 0.5
    # psf.zip: a spline channel (the reference spline) with its vectorial source and the field block
    meta = io.save_zip(tmp_path / 't.zip', tp, zstack_planes=3)
    ch = meta['channels'][0]
    assert ch['type'] == 'spline' and ch['source']['type'] == 'vectorial' and ch['field']['type'] == 'zernike_poly'
    assert meta['provenance']['field_tiles']['tile_px'] == 64.0
    t2 = PSF.load(tmp_path / 't.zip', device=DEV)
    assert isinstance(t2, TiledSplinePSF) and t2.n_tiles == 16 and t2.roisize == 11
    assert torch.allclose(t2.render(pos, xy=xy), tp.render(pos, xy=xy), atol=1e-6)


def test_fit_field_knots_recovers_a_linear_field():
    """in situ, simulated: 2500 spots over 256 x 256 px with Z5 ~ x and Z6 ~ y; 2 x 2 tiles; from a constant field"""
    v, truth = vec(11), field()
    fp = FieldPSF(v, truth)
    N = 2500
    g = torch.Generator().manual_seed(1)
    xy = (torch.rand(N, 2, generator=g) * 256).to(DEV)
    pos = torch.cat([torch.rand(N, 2, generator=g) - 0.5, 0.05 + 0.5 * torch.rand(N, 1, generator=g)], 1).to(DEV)
    with torch.no_grad():
        rois = torch.poisson(3000 * torch.cat([fp.render(pos[s:s + 1000], xy=xy[s:s + 1000])
                                               for s in range(0, N, 1000)]) + 10)
    start = FieldZernikes([5, 6], order=1, norm=(128.0, 128.0, 128.0, 128.0), device=DEV)
    spl = SplinePSF(11, (0.0, 0.6), device=DEV, oversample=2, dz_um=0.04, zpad=3, xy_limit=2.0)
    r = fit_field_knots(spl, rois, xy, v, start, tile=128, iters=25, lr=5e-3)
    assert r['report']['n_tiles'] == 4 and sorted(r['tile_counts']) and sum(r['tile_counts']) == N
    c = r['psf'].field.coef.detach().cpu().numpy()
    err = np.abs(c - truth.coef.detach().cpu().numpy()).max()
    assert err < 0.008, c                                        # field terms 0.03 - 0.04 waves
    assert float((r['psf'].base.zcoef.detach() - v.zcoef.detach()).abs().max()) < 0.008


def test_matched_filter_detector():
    """two well separated spots and a close pair in a frame: centres to a pixel, the template z, isolation"""
    v = vec(31)
    det = MatchedFilterDetector(v, dz_um=0.1, size=21)
    assert det.T.shape == (7, 21, 21)
    H = W = 96
    pts = [(20.0, 25.0, 0.1), (70.0, 60.0, 0.5), (30.0, 75.0, 0.3), (45.0, 78.0, 0.3)]
    mu = torch.full((2, H, W), 5.0, device=DEV)
    with torch.no_grad():
        for x, y, z in pts:
            img = 4000 * v.render(torch.tensor([[x - round(x), y - round(y), z]], device=DEV))[0]
            mu[:, int(y) - 15:int(y) + 16, int(x) - 15:int(x) + 16] += img
    frames = torch.poisson(mu, generator=torch.Generator(device=DEV).manual_seed(0))
    d = det(frames, threshold=8.0, isolation=20)
    assert len(d['x']) == 4                                      # 2 frames x the 2 isolated spots
    got = sorted(zip(d['x'][:2].tolist(), d['y'][:2].tolist(), d['z'][:2].tolist()))
    assert abs(got[0][0] - 20) <= 1 and abs(got[0][1] - 25) <= 1 and abs(got[0][2] - 0.1) < 0.15
    assert abs(got[1][0] - 70) <= 1 and abs(got[1][1] - 60) <= 1 and abs(got[1][2] - 0.5) < 0.15
    sp = extract_spots(frames, 15, detector=det, device=DEV, threshold=8.0)
    assert len(sp) >= 6 and sp.rois.shape[1:] == (15, 15)        # without isolation the close pair is kept too


@needs_gpu
def test_dense_beads_field_recovers_polynomial():
    """dense bead stack with a linear Z5 / Z6 field over 160 x 160 px: the field fit recovers it and beats the
    field-constant fit on held-out clusters"""
    from torch_psf import DenseBeadEstimator
    from torch_psf.estimation.dense_beads import simulate_dense_bead_stack, dense_bead_layout
    truth = VectorialPSF(31, pixel_um=0.1, NA=1.49, wavelength_um=0.67, device=DEV, **PUP, pixel_integration='midpoint').set_zernike(
        {5: 0.02, 6: 0.06, 7: -0.03})
    f = FieldZernikes([5, 6], order=1, norm=(80.0, 80.0, 80.0, 80.0), device=DEV)
    with torch.no_grad():
        f.coef.copy_(torch.tensor([[0.0, 0.03], [-0.025, 0.0]]))
    nk = 21
    focus = stage_focus(nk, 1.0 / (nk - 1), 1.0).to(DEV)
    stacks = []
    for seed in (0, 1):
        xy, _ = dense_bead_layout(160, 160, [1] * 8 + [2] * 6, sep_px=(10, 16), margin=12, gap_px=20, seed=seed)
        stacks.append(simulate_dense_bead_stack(truth, focus, xy, 160, 160, photons=30000.0, bg=30.0, seed=seed,
                                                field=f)['d'])
    est = DenseBeadEstimator(optics=dict(pixel_um=0.1, NA=1.49, wavelength_um=0.67), patch=15, render_pad=5,
                             iters=200, field_stop=False, holdout=1, device=DEV, npupil=32, oversample_pupil=2,
                             zernike_indices=(5, 6, 7, 8, 11), field=dict(modes=[5, 6], order=1, iters=200))
    psf = est.estimate(stacks, 0.05)
    assert isinstance(psf, FieldPSF) and est.report['field']['kept']
    assert est.report['field']['heldout_nll_per_px'] < est.report['heldout_nll_per_px']
    # the polynomial in the fitted normalisation, evaluated against the truth on the field
    X, Y = np.meshgrid(np.linspace(20, 140, 7), np.linspace(20, 140, 7))
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32, device=DEV)
    zt = (truth.zcoef[None] + f.offsets(pts, truth.zidx)).detach().cpu().numpy()[:, :2]
    zf = psf.zernikes_at(pts)[:, :2]
    assert np.abs(zf - zt).max() < 0.006, np.abs(zf - zt).max()


@needs_gpu
def test_insitu_estimator_field_option():
    """InSituEstimator(field=) on a simulated movie with a linear Z6 field: the field model wins on held-out spots"""
    from torch_psf import InSituEstimator
    v = VectorialPSF(13, pixel_um=0.1, NA=1.4, zernike_indices=(5, 6, 11), device=DEV, focus_um=0.3,
                     z_range=(0.0, 0.6), **PUP, pixel_integration='midpoint').set_zernike({6: 0.08})
    f = FieldZernikes([6], order=1, norm=(128.0, 128.0, 128.0, 128.0), device=DEV)
    with torch.no_grad():
        f.coef.copy_(torch.tensor([[0.0, 0.06]]))                         # Z6 0.02 .. 0.14 waves across x
    fp = FieldPSF(v.resized(17), f)
    g = torch.Generator(device=DEV).manual_seed(0)
    T, H, n = 80, 256, 60
    mov = torch.full((T, H, H), 10.0, device=DEV)
    for t in range(T):
        xy = 12 + torch.rand(n, 2, device=DEV, generator=g) * (H - 24)
        z = 0.05 + 0.5 * torch.rand(n, 1, device=DEV, generator=g)
        r = torch.round(xy)
        with torch.no_grad():
            img = 3000 * fp.render(torch.cat([xy - r, z], 1), xy=xy)
        for i in range(n):
            x0, y0 = int(r[i, 0]) - 8, int(r[i, 1]) - 8
            mov[t, y0:y0 + 17, x0:x0 + 17] += img[i]
    mov = torch.poisson(mov, generator=g)
    est = InSituEstimator(start=v, roisize=13, z_range=(0.0, 0.6), nspots=2500, nheldout=1000, iters=15,
                          xy_limit=2.0, isolation=8, device=DEV, reject_overlap=False,
                          field=dict(modes=[6], order=1, tile=128, iters=25, lr=4e-3, score_tile=64))
    psf = est.estimate(mov.cpu().numpy())
    rep = est.report
    assert isinstance(psf, FieldPSF) and rep['field']['kept'], rep['heldout']
    assert rep['heldout']['field']['dnll_per_spot_vs_first'] < rep['heldout']['vectorial']['dnll_per_spot_vs_first']
    c = psf.field.coef.detach().cpu().numpy()
    assert abs(c[0, 1] - 0.06) < 0.015 and abs(c[0, 0]) < 0.015, c
