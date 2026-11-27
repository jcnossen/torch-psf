"""torch_psf.nn: network, loss, merge, scenes, calibration, files, localization, training (CPU, generated data)."""
import json
import math
import os
import numpy as np
import pytest
import torch
import torch_psf as tp
import torch_psf.nn as tn
from torch_psf.nn import merge as M
from torch_psf.nn.scenes import gamma_unit
from conftest import needs_gpu

SMALL = dict(frame_features=(8, 16), window_features=(8, 16, 32), window_out=16, head_features=4)


@pytest.fixture(scope='module')
def psf_file(tmp_path_factory):
    p = tmp_path_factory.mktemp('nn') / 'astig.json'
    tp.AstigmaticGaussianPSF(23, sigma_px=(1.3, 1.3), gamma_um=0.25, d_um=0.4, z_range=(-0.5, 0.5),
                             device='cpu').save(str(p))
    return str(p)


def scene_cfg(psf_file, **ranges):
    return dict(psf=[psf_file], pixel_nm=117.0, pixel_nm_y=127.0, camera=dict(adu_per_e=15.34, offset_adu=394.1,
                readnoise_adu=4.5), label_min=100.0, min_on=0.0, ranges=dict(dict(z_range_um=(-0.4, 0.4)), **ranges))


# ---------------------------------------------------------------------------------------------- network
@pytest.mark.parametrize('z,cov', [(True, None), (True, 'full_xyz'), (False, None)])
def test_output_layout(z, cov):
    net = tn.SMLMNet(z=z, cov=cov, max_bg=200., **SMALL).eval()
    x = torch.rand(2, 3, 16, 16) * 50
    with torch.no_grad():
        o = net(x)
    n = 5 if z else 4
    assert o.shape == (2, 1 + 2 * n + (3 if cov else 0), 16, 16) == (2, net.channels, 16, 16)
    assert ((o[:, 0] >= 0) & (o[:, 0] <= 1)).all()
    col = torch.arange(16.)[None, None, :]
    assert ((o[:, 1] - col).abs() <= 1 + 1e-6).all()                    # x offset within xyz_scale of the pixel
    assert ((o[:, 2] - torch.arange(16.)[None, :, None]).abs() <= 1 + 1e-6).all()
    assert (o[:, n + 1:2 * n + 1] >= 1e-3 * 0.999).all()
    if cov:
        assert (o[:, -3:] == 0).all()                                  # zero-initialised correlation head
    with pytest.raises(ValueError):
        net(torch.rand(1, 2, 16, 16))


def test_bad_options():
    with pytest.raises(ValueError):
        tn.SMLMNet(window=2)
    with pytest.raises(ValueError):
        tn.SMLMNet(z=False, cov='full_xyz')


def test_xyz_cholesky():
    sigma = torch.tensor([[0.3, 0.5, 0.05]])
    rho = torch.tensor([[0.4, -0.2, 0.3]])
    L = tn.xyz_cholesky(sigma, rho)
    cov = L @ L.mT
    assert torch.allclose(cov.diagonal(dim1=-2, dim2=-1).sqrt(), sigma, atol=1e-6)
    r = tn.correlations(cov)
    assert torch.allclose(r[:, 0], rho[:, 0], atol=1e-6) and torch.allclose(r[:, 1], rho[:, 1], atol=1e-6)


# ---------------------------------------------------------------------------------------------- loss
def _random_output(net, b=2, h=8):
    torch.manual_seed(3)
    with torch.no_grad():
        return net(torch.rand(b, 3, h, h) * 40)


def test_full_xyz_loss_with_zero_correlation_is_diagonal():
    net = tn.SMLMNet(cov='full_xyz', **SMALL).eval()
    out = _random_output(net)
    out[:, -3:] = 0
    t = torch.zeros(2, 3, 6)
    t[0, :2] = torch.tensor([[1, 3.2, 4.1, 0.1, 20., 800.], [1, 6.0, 1.5, -0.2, 15., 400.]])
    t[1, 0] = torch.tensor([1, 2.5, 2.5, 0.0, 30., 1200.])
    full = tn.GMMLoss(5, cov='full_xyz').per_image(out, t)
    diag = tn.GMMLoss(5).per_image(out[:, :-3], t)
    assert torch.allclose(full, diag, rtol=1e-5, atol=1e-4)


def test_loss_gradients_finite():
    net = tn.SMLMNet(cov='full_xyz', **SMALL)
    x = torch.rand(2, 3, 8, 8) * 40
    t = torch.zeros(2, 1, 6)
    t[0, 0] = torch.tensor([1, 3.2, 4.1, 0.1, 20., 800.])
    loss = net.loss(0.01)(net(x), t)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------------------------- merge
def _grid(B=1, H=8, W=8):
    G = {k: torch.zeros(B, H, W) for k in ('p', 'z', 'photons', 's_photons', 'bg', 's_bg')}
    G['x'] = torch.arange(W).float()[None, None].expand(B, H, W).clone()
    G['y'] = torch.arange(H).float()[None, :, None].expand(B, H, W).clone()
    for k in ('sx', 'sy'):
        G[k] = torch.full((B, H, W), 0.1)
    G['sz'] = torch.full((B, H, W), 0.05)
    return G


def test_merge_offsets():
    G = _grid()
    # one emitter at (3.5, 4.0) seen by the pixels (3, 4) and (4, 4): one row with p summed
    for ix in (3, 4):
        G['p'][0, 4, ix] = 0.5; G['x'][0, 4, ix] = 3.5; G['photons'][0, 4, ix] = 1000.
    # two emitters in adjacent pixels at distinct positions: two rows
    G['p'][0, 1, 1] = 0.9; G['p'][0, 1, 2] = 0.9
    G['x'][0, 1, 1] = 0.8; G['x'][0, 1, 2] = 2.2
    r = M.merge(G, **M.MERGE_DEFAULTS)
    assert len(r['p']) == 3
    i = torch.nonzero((r['x'] - 3.5).abs() < 1e-6)[:, 0]
    assert len(i) == 1 and abs(float(r['p'][i]) - 1.0) < 1e-6 and abs(float(r['photons'][i]) - 1000.) < 1e-3
    assert sorted(float(v) for v in r['x'][r['iy'] == 1]) == pytest.approx([0.8, 2.2])
    n = M.nms(G, raw_th=0.05)
    assert len(n['p']) >= 2


def test_merge_covariance_parallel_axis():
    G = _grid()
    G['cov'] = torch.zeros(3, 3, 1, 8, 8)
    for i, s in enumerate((0.1, 0.1, 0.05)):
        G['cov'][i, i] = s ** 2
    G['p'][0, 4, 3] = 0.5; G['p'][0, 4, 4] = 0.5
    G['x'][0, 4, 3] = 3.45; G['x'][0, 4, 4] = 3.55
    r = M.merge(G, **M.MERGE_DEFAULTS)
    assert len(r['p']) == 1
    assert float(r['x'][0]) == pytest.approx(3.5, abs=1e-6)
    assert float(r['cov'][0, 0, 0]) == pytest.approx(0.01 + 0.05 ** 2, rel=1e-5)     # mean cov + scatter


# ---------------------------------------------------------------------------------------------- scenes
def test_gamma_sampler():
    g = torch.Generator().manual_seed(1)
    a = torch.full((200000,), 7.0)
    s = gamma_unit(a, g)
    assert float(s.mean()) == pytest.approx(7.0, rel=0.01) and float(s.var()) == pytest.approx(7.0, rel=0.03)
    a = torch.full((100000,), 1.0)
    assert float(gamma_unit(a, g).mean()) == pytest.approx(1.0, rel=0.02)


def test_scenes_deterministic_and_labels(psf_file):
    cfg = scene_cfg(psf_file)
    a = list(tn.Scenes(cfg, 4, 24, 0, 'cpu', seed=11).batches(2))
    b = list(tn.Scenes(cfg, 4, 24, 0, 'cpu', seed=11).batches(2))
    c = list(tn.Scenes(cfg, 4, 24, 0, 'cpu', seed=12).batches(1))
    for (fa, ta, _), (fb, tb, _) in zip(a, b):
        assert torch.equal(fa, fb) and torch.equal(ta, tb)
    assert not torch.equal(a[0][0], c[0][0])
    f, t, lab = a[0]
    assert f.shape == (4, 3, 24, 24) and t.shape[0] == 4 and t.shape[2] == 6
    assert int(t[..., 0].sum()) == len(lab['x'])
    assert (lab['photons'] >= 100).all()
    assert ((lab['x'].round() >= 0) & (lab['x'].round() < 24)).all()
    # targets in row order per field
    for w in range(4):
        sel = lab['fov'] == w
        k = int(sel.sum())
        assert torch.allclose(t[w, :k, 1], lab['x'][sel].float())


def test_scenes_halo(psf_file):
    S = tn.Scenes(scene_cfg(psf_file), 2, 16, 4, 'cpu', seed=1)
    f, t, lab = next(S.batches(1))
    assert f.shape[-1] == 24
    assert ((lab['x'].round() >= 0) & (lab['x'].round() < 16)).all()


def test_scene_physics(psf_file):
    """Background level and camera chain: the mean of an empty scene equals the background in photoelectrons."""
    S = tn.Scenes(scene_cfg(psf_file, dens=(1e-9, 1e-9 * 1.0001), bg=(20., 20.0001), bg_sd=0., bg_hp=0., bg_fluct=0.,
                            gain_err=0., off_err=0.), 8, 32, 0, 'cpu', seed=3)
    sc = S.scene()
    assert float(sc['bg'].mean()) == pytest.approx(20., rel=1e-3)
    # EMCCD: variance of the photoelectron frames ~ 2 x mean (excess noise) + read noise
    v = float(sc['frames'].var())
    assert v == pytest.approx(2 * 20 + (4.5 / 15.34) ** 2, rel=0.05)
    assert float(sc['frames'].mean()) == pytest.approx(20., rel=0.02)


# ---------------------------------------------------------------------------------------------- calibration
def test_xyz_matrix_recovers_scale():
    rng = np.random.default_rng(0)
    n = 6000
    cov = np.tile(np.diag([0.01, 0.01, 0.0025]), (n, 1, 1))
    err = rng.standard_normal((n, 3)) * np.sqrt([0.01, 0.01, 0.0025]) * np.array([1.5, 1.2, 2.0])
    E = dict(photons=rng.uniform(0, 3000, n), z_true=np.zeros(n), ex=err[:, 0], ey=err[:, 1], ez=err[:, 2], cov=cov)
    A, rep = tn.calibrate.fit_xyz_matrix(E, E, method='rms')
    A = np.asarray(A)
    assert np.allclose(np.diagonal(A, axis1=1, axis2=2).mean(0), [1.5, 1.2, 2.0], rtol=0.08)
    assert rep['after']['rms'] == pytest.approx([1, 1, 1], abs=0.05)
    rows = dict(photons=torch.tensor([100., 2000.]), sx=torch.full((2,), 0.1), sy=torch.full((2,), 0.1),
                sz=torch.full((2,), 0.05), cov=torch.tensor(cov[:2], dtype=torch.float32))
    cal = dict(edges=list(tn.calibrate.DEFAULT_EDGES), xyz_matrix=np.tile(np.eye(3) * 2, (7, 1, 1)).tolist())
    r = tn.apply_calibration(cal, rows)
    assert torch.allclose(r['sx'], torch.full((2,), 0.2)) and torch.allclose(r['sz'], torch.full((2,), 0.1))


def test_bn_recalibration(psf_file):
    net = tn.SMLMNet(**SMALL)
    S = tn.Scenes(scene_cfg(psf_file), 2, 16, 0, 'cpu', seed=1)
    w0 = {k: v.clone() for k, v in net.state_dict().items() if 'running' not in k and 'num_batches' not in k}
    net, info = tn.bn_recalibrate(net, S, n_scenes=4, mix=2)
    assert info['forwards'] == 2 and not net.training
    for k, v in w0.items():
        assert torch.equal(v, net.state_dict()[k])
    assert float(net.frame_unet.stem.bn.running_mean.abs().sum()) > 0


# ---------------------------------------------------------------------------------------------- files, localize
def test_model_files_roundtrip(tmp_path):
    net = tn.SMLMNet(cov='full_xyz', **SMALL).eval()
    meta = tn.files.model_meta(net, 117.0, 127.0, [], tn.files.camera_dict(dict(adu_per_e=15.34, offset_adu=394.1,
                                                                                readnoise_adu=4.5)))
    tn.save_model(tmp_path / 'm', net, meta, dict(edges=[0, 1e9], xyz_matrix=[np.eye(3).tolist()]))
    net2, meta2, cal = tn.load_model(tmp_path / 'm', 'cpu')
    x = torch.rand(1, 3, 16, 16) * 30
    with torch.no_grad():
        assert torch.equal(net(x), net2(x))
    assert meta2['pixel_nm_y'] == 127.0 and cal['xyz_matrix'][0][0][0] == 1.0


def test_localize_movie_and_cli(tmp_path, psf_file):
    net = tn.SMLMNet(cov='full_xyz', **SMALL).eval()
    with torch.no_grad():
        net.head[1].bias[0] = 2.0                      # p high everywhere: many detections
    cam = tn.files.camera_dict(dict(adu_per_e=15.34, offset_adu=394.1, readnoise_adu=4.5))
    tn.save_model(tmp_path / 'm', net, tn.files.model_meta(net, 117.0, 127.0, [psf_file], cam),
                  dict(edges=[0, 1e9], xyz_matrix=[(np.eye(3) * 2).tolist()]))
    movie = (np.random.default_rng(0).poisson(30, (5, 18, 22)) + 400).astype(np.uint16)
    np.save(tmp_path / 'movie.npy', movie)
    ds = tn.localize_movie(str(tmp_path / 'movie.npy'), str(tmp_path / 'm'), camera='model', device='cpu', batch=2)
    assert len(ds) > 0 and ds.n_dims == 3
    assert set(ds['frame'].tolist()) <= set(range(5))
    assert torch.allclose(ds['crlb'], 2 * ds['sigma'], rtol=1e-4)
    assert ds['cov'].shape == (len(ds), 3) and ds['rho'].shape == (len(ds), 3)
    assert ds.meta['pixel_nm_y'] == 127.0 and ds.meta['image_size'] == [18, 22]
    from torch_psf.__main__ import main
    out = tmp_path / 'locs.hdf5'
    assert main(['nn', 'localize', str(tmp_path / 'movie.npy'), '--model', str(tmp_path / 'm'), '--camera', 'model',
                 '--device', 'cpu', '-o', str(out)]) == 0
    assert len(tp.Dataset.load(str(out))) == len(ds)


# ---------------------------------------------------------------------------------------------- training
def test_lion_matches_lion_pytorch():
    lion = pytest.importorskip('lion_pytorch')
    torch.manual_seed(0)
    a = torch.nn.Linear(4, 3); b = torch.nn.Linear(4, 3); b.load_state_dict(a.state_dict())
    oa = tn.Lion(a.parameters(), lr=1e-2, weight_decay=0.1); ob = lion.Lion(b.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(5):
        x = torch.randn(8, 4)
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad(); m(x).square().sum().backward(); o.step()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def _train_cfg(psf_file, out, steps=4):
    return dict(out=str(out), seed=5, steps=steps, batch=2, size=16, lr=1e-3, warmup=2, val_batches=1, log_every=2,
                save_every=100, device='cpu', model=dict(cov='full_xyz', max_bg=200., **SMALL),
                scenes=scene_cfg(psf_file))


def test_train_deterministic_and_resume(tmp_path, psf_file):
    log = []
    tn.train_model(_train_cfg(psf_file, tmp_path / 'a'), progress=log.append)
    tn.train_model(_train_cfg(psf_file, tmp_path / 'b'), progress=lambda s: None)
    sa = torch.load(tmp_path / 'a' / 'weights.pt', weights_only=True)
    sb = torch.load(tmp_path / 'b' / 'weights.pt', weights_only=True)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    lines = [json.loads(s) for s in open(tmp_path / 'a' / 'train.log')]
    assert [d['step'] for d in lines] == [0, 2, 4] and 'recall' in lines[-1]['validation']
    net, meta, _ = tn.load_model(tmp_path / 'a', 'cpu')
    assert meta['training']['steps'] == 4 and meta['architecture']['cov'] == 'full_xyz'
    tn.train_model(_train_cfg(psf_file, tmp_path / 'a', steps=6), progress=lambda s: None)     # resume 4 -> 6
    assert json.loads(open(tmp_path / 'a' / 'train.log').read().splitlines()[-1])['step'] == 6


def test_halo_training(tmp_path, psf_file):
    cfg = _train_cfg(psf_file, tmp_path / 'h', steps=2)
    cfg['halo'] = 4
    tn.train_model(cfg, progress=lambda s: None)
    assert os.path.exists(tmp_path / 'h' / 'model.json')


@needs_gpu
def test_gpu_smoke_training_loss_decreases(tmp_path, psf_file):
    """Small network, 300 steps on the GPU (~20 s): the training loss goes down."""
    cfg = dict(_train_cfg(psf_file, tmp_path / 'g', steps=300), device='cuda', batch=8, size=24, log_every=100,
               lr=3e-4, warmup=50, model=dict(cov='full_xyz', max_bg=200., frame_features=(16, 32),
                                              window_features=(16, 32, 64), window_out=64, head_features=8))
    tn.train_model(cfg, progress=lambda s: None)
    L = [json.loads(s) for s in open(tmp_path / 'g' / 'train.log')]
    assert L[-1]['train_loss'] < L[1]['train_loss'] and L[-1]['validation_loss'] < L[0]['validation_loss']
