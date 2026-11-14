"""Dataset: construction, indexing and filters, concatenate, crop, drift, render / FRC, files (hdf5, Picasso, csv)."""
import numpy as np
import pytest
import torch
from torch_psf import Dataset


def make(n=500, D=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, D, generator=g, dtype=torch.float64) * torch.tensor([100.0, 80.0, 5.0][:D], dtype=torch.float64)
    ds = Dataset.from_arrays(pos, torch.randint(0, 1000, (n,), generator=g), crlb=torch.rand(n, D, generator=g) * 0.2,
                             photons=1000 + 500 * torch.rand(n, generator=g), bg=torch.full((n,), 10.0),
                             n_frames=torch.full((n,), 6), meta=dict(pixel_nm=108.0, image_size=[80, 100]),
                             intensities=torch.rand(n, 6, 2, generator=g), score=torch.rand(n, generator=g))
    return ds


def test_fields_and_dtypes():
    ds = make()
    assert len(ds) == 500 and ds.n_dims == 3 and ds['pos'].dtype == torch.float64
    assert ds.frame.dtype == torch.int32 and ds.id.dtype == torch.int64 and ds.crlb.dtype == torch.float32
    assert torch.equal(ds.end_frame, ds.frame + 5) and torch.equal(ds.center_frame, ds.frame + 3)
    with pytest.raises(ValueError):
        ds['bad'] = torch.zeros(3)
    ds.add_fields(flag=torch.zeros(500, dtype=torch.bool))
    assert 'flag' in ds and isinstance(ds.numpy()['pos'], np.ndarray)


def test_indexing_filters_concatenate():
    ds = make()
    m = ds.photons > 1200
    sub = ds[m]
    assert len(sub) == int(m.sum()) and sub.intensities.shape[1:] == (6, 2) and torch.equal(sub.id, ds.id[m])
    assert len(ds[10:20]) == 10 and len(ds[torch.tensor([1, 3])]) == 2 and len(ds[5]) == 1
    f = ds.filter_photons(1200)
    assert len(f) == len(sub)
    n0 = len(ds)
    ds2 = ds.copy().filter_crlb_(max_px=0.1)
    assert len(ds2) < n0 and float(ds2.crlb[:, :2].max()) <= 0.1
    c = Dataset.concatenate([ds[:100], ds[100:]], set_group=True)
    assert len(c) == n0 and set(c.group.tolist()) == {0, 1} and torch.equal(c.pos, ds.pos)
    assert len(ds[:10] + ds[10:30]) == 30
    cr = ds.crop(10, 10, 50, 40, shift=True)
    assert float(cr.pos[:, 0].min()) >= 0 and float(cr.pos[:, 0].max()) < 40 and cr.meta['origin'] == [10.0, 10.0]
    fr = ds.filter_frames(100, 500)
    assert bool((fr.frame >= 100).all()) and bool((fr.end_frame < 500).all())


def test_drift_apply_undo():
    ds = make()
    drift = torch.stack([torch.linspace(0, 1, 1010), torch.linspace(0, -2, 1010)], 1)
    p0 = ds.pos.clone()
    ds.set_drift(drift).apply_drift()
    t = (ds.frame + 3).long()
    assert torch.allclose(ds.pos[:, :2], p0[:, :2] - drift[t].double())
    ds.undo_drift()
    assert torch.allclose(ds.pos, p0)


def test_files_roundtrip(tmp_path):
    ds = make()
    ds.set_drift(torch.zeros(1006, 2))
    ds.save(tmp_path / 'a.hdf5')
    b = Dataset.load(tmp_path / 'a.hdf5')
    assert set(b.keys()) == set(ds.keys()) and all(torch.equal(b[k], ds[k]) for k in ds.keys())
    assert b.meta['image_size'] == [80, 100] and b.pixel_nm == 108.0 and b.drift.shape == (1006, 2)
    ds.save_picasso(tmp_path / 'p.hdf5')
    p = Dataset.load(tmp_path / 'p.hdf5')
    assert torch.allclose(p.pos, ds.pos, atol=1e-4) and p.meta['image_size'] == [80, 100] and torch.allclose(p.score, ds.score)
    ds.save(tmp_path / 'c.csv')
    c = Dataset.load_csv(tmp_path / 'c.csv', pixel_nm=108.0)
    assert torch.allclose(c.pos, ds.pos, atol=1e-5) and torch.equal(c.frame, ds.frame)
    assert torch.allclose(c.crlb[:, :2].float(), ds.crlb[:, :2], atol=1e-6)


def test_render_and_frc():
    g = torch.Generator().manual_seed(1)
    n = 20000
    # rings of radius 2 px, localization noise 0.05 px
    cx, cy = torch.rand(40, generator=g) * 80 + 10, torch.rand(40, generator=g) * 60 + 10
    k = torch.randint(0, 40, (n,), generator=g)
    a = torch.rand(n, generator=g) * 6.283
    pos = torch.stack([cx[k] + 2 * torch.cos(a), cy[k] + 2 * torch.sin(a)], 1) + 0.05 * torch.randn(n, 2, generator=g)
    ds = Dataset.from_arrays(pos, torch.randint(0, 3000, (n,), generator=g), crlb=torch.full((n, 2), 0.05),
                             meta=dict(pixel_nm=100.0, image_size=[80, 100]))
    img = ds.render(pixel_nm=20)
    assert img.shape == (400, 500) and abs(img.sum() - n) < 1
    res = ds.frc(block=300)
    assert 5 < res < 100, res


def test_estimate_drift():
    import importlib, os, sys
    if os.environ.get('DME_PATH'):
        sys.path.insert(0, os.environ['DME_PATH'])
    try:
        dme = importlib.import_module('dme')
        importlib.import_module('dme.dme_prior')
        assert int(str(dme.__version__).split('.')[0]) >= 2
    except Exception:
        pytest.skip('DME v2 (github.com/jcnossen/drift-estimation) not available')
    g = torch.Generator().manual_seed(0)
    n, F = 40000, 4000
    sites = torch.rand(400, 2, generator=g) * 100
    k = torch.randint(0, 400, (n,), generator=g)
    fr = torch.randint(0, F, (n,), generator=g)
    true = torch.stack([torch.linspace(0, 1.5, F), torch.linspace(0, -1.0, F)], 1).double()
    ds = Dataset.from_arrays(sites[k] + true[fr] + 0.05 * torch.randn(n, 2, generator=g, dtype=torch.float64), fr,
                             crlb=torch.full((n, 2), 0.05), meta=dict(pixel_nm=100.0, n_frames=F))
    d, rec = ds.estimate_drift(return_info=True)
    assert float((d - d.mean(0) - (true - true.mean(0))).abs().max()) < 0.05
    assert ds.meta['drift'] is rec and rec['method'] == 'DME v2' and rec['parameters']['n_rows'] == n
    assert len(rec['result']['ptp_px']) == 2 and abs(rec['result']['ptp_px'][0] - 1.5) < 0.1


def test_fit_needs_a_pixel_size_for_3d():
    from torch_psf import AstigmaticGaussianPSF, GaussianPSF
    a = AstigmaticGaussianPSF(11, gamma_um=0.2, device='cpu')
    th = torch.tensor([[0.1, -0.1, 0.2, 2000.0, 10.0]]).repeat(50, 1)
    rois = torch.poisson(a.expected(th))
    with pytest.raises(ValueError):
        a.fit(rois)
    ds = a.fit(rois, pixel_um=0.1)
    assert ds.pixel_nm == 100.0 and abs(float(ds.pos[:, 2].median()) - 2.0) < 0.5      # 0.2 um = 2 px of 0.1 um
    g = GaussianPSF(11, device='cpu')
    assert g.fit(torch.poisson(g.expected(th))).n_dims == 2                          # 2D: no pixel size needed


def test_drift_record():
    ds = make()
    ds.set_drift(torch.zeros(1006, 2), info=dict(method='given'))
    assert ds.meta['drift'] == dict(method='given')
    p = ds.copy()
    assert p.meta['drift']['method'] == 'given'


def test_picasso_3d_z_in_nm(tmp_path):
    import h5py
    n = 50
    pos = torch.rand(n, 3, dtype=torch.float64) * 10
    ds = Dataset.from_arrays(pos, torch.arange(n), crlb=torch.full((n, 3), 0.1), meta=dict(pixel_nm=100.0, image_size=[20, 20]))
    ds.save_picasso(tmp_path / 'z.hdf5')
    with h5py.File(tmp_path / 'z.hdf5') as f:   # Picasso convention: x, y, lpx, lpy in px; z, lpz in nm
        L = f['locs'][:]
    assert np.allclose(L['z'], pos[:, 2].numpy() * 100, atol=1e-3) and np.allclose(L['lpz'], 10.0)
    assert np.allclose(L['x'], pos[:, 0].numpy(), atol=1e-5)
    p = Dataset.load(tmp_path / 'z.hdf5')
    assert torch.allclose(p.pos, pos, atol=1e-5) and torch.allclose(p.crlb[:, 2], torch.full((n,), 0.1))
    with pytest.raises(ValueError):
        Dataset.from_arrays(pos, torch.arange(n)).save_picasso(tmp_path / 'nopx.hdf5')
