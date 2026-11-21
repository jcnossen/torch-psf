"""Dataset: construction, indexing and filters, concatenate, crop, drift, render / FRC, files (hdf5, Picasso, csv, ShareLoc .smlm)."""
import json

import h5py
import numpy as np
import pytest
import torch
from torch_psf import Dataset, Drift


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


def test_drift_corrected():
    ds = Dataset.from_arrays([[5, 6, 7]] * 3, [-2, 0, 9], n_frames=[1, 2, 1])
    drift = Drift([[1, 2], [3, 4], [5, 6]], {'method': 'given'})
    corrected = ds.drift_corrected(drift)
    assert corrected.pos.tolist() == [[4, 4, 7], [2, 2, 7], [0, 0, 7]]
    assert corrected.meta['drift'] == drift.info
    assert ds.pos.tolist() == [[5, 6, 7]] * 3 and 'drift' not in ds.meta
    with pytest.raises(ValueError, match='already drift corrected'):
        corrected.drift_corrected(drift)
    second = Drift([[1]], {'method': 'residual'})
    twice = corrected.drift_corrected(second, force=True)
    assert torch.equal(twice.pos[:, 0], corrected.pos[:, 0] - 1)
    assert twice.meta['drift'] == second.info


def test_drift_roundtrip(tmp_path):
    drift = Drift([[1, 2], [3, 4]], {'parameters': {'dims': 2}})
    path = tmp_path / 'drift.npz'
    drift.save(path)
    loaded = Drift.load(path)
    assert loaded.curve.dtype == torch.float64
    assert torch.equal(loaded.curve, drift.curve) and loaded.info == drift.info


@pytest.mark.parametrize('applied', [True, False])
def test_legacy_drift(tmp_path, applied):
    path = tmp_path / 'v1.h5'
    info = {'method': 'given'}
    with h5py.File(path, 'w') as f:
        f.attrs.update(format='torch-psf dataset', format_version=1)
        f['localizations/pos'] = [[4., 5.]]
        f['localizations/frame'] = [0]
        f.create_group('metadata').attrs['drift'] = 'json:' + json.dumps(info)
        d = f.create_dataset('drift', data=[[1., 2.]])
        d.attrs.update(applied=applied, info=json.dumps(info))
    if applied:
        ds = Dataset.load(path)
        assert ds.meta['drift'] == info
    else:
        with pytest.warns(UserWarning, match=r'unapplied drift curve.*tp\.Drift\.load'):
            ds = Dataset.load(path)
        assert 'drift' not in ds.meta
    assert ds.pos.tolist() == [[4., 5.]]
    drift = Drift.load(path)
    assert drift.curve.tolist() == [[1., 2.]] and drift.info == info


def test_files_roundtrip(tmp_path):
    ds = make().drift_corrected(Drift([[1, 2]], {'method': 'given'}))
    ds.save(tmp_path / 'a.hdf5')
    b = Dataset.load(tmp_path / 'a.hdf5')
    assert set(b.keys()) == set(ds.keys()) and all(torch.equal(b[k], ds[k]) for k in ds.keys())
    assert b.meta['image_size'] == [80, 100] and b.pixel_nm == 108.0 and b.meta['drift'] == ds.meta['drift']
    with h5py.File(tmp_path / 'a.hdf5') as f:
        assert 'drift' not in f and f.attrs['format_version'] == 2
    ds.save_picasso(tmp_path / 'p.hdf5')
    p = Dataset.load(tmp_path / 'p.hdf5')
    assert torch.allclose(p.pos, ds.pos, atol=1e-4) and p.meta['image_size'] == [80, 100] and torch.allclose(p.score, ds.score)
    ds.save(tmp_path / 'c.csv')
    c = Dataset.load_csv(tmp_path / 'c.csv', pixel_nm=108.0)
    assert torch.allclose(c.pos, ds.pos, atol=1e-5) and torch.equal(c.frame, ds.frame)
    assert torch.allclose(c.crlb[:, :2].float(), ds.crlb[:, :2], atol=1e-6)


def test_smlm_roundtrip(tmp_path):
    import zipfile
    ds = make()
    ds.add_fields(flag=ds.photons > 1200, track=torch.arange(len(ds), dtype=torch.int32) // 3)
    b = Dataset.load(ds.save(tmp_path / 'a.smlm', dtype='float64'))            # pixel size etc. from the manifest
    assert set(b.keys()) == set(ds.keys()) - {'intensities'}                    # 1D fields only
    for k in b.keys():
        if k in ('id', 'frame', 'n_frames', 'flag', 'track'):
            assert b[k].dtype == ds[k].dtype and torch.equal(b[k], ds[k]), k
        else:
            assert torch.allclose(b[k].double(), ds[k].double(), rtol=1e-12, atol=1e-12), k
    assert b.pixel_nm == 108.0 and b.meta['image_size'] == [80, 100] and b.num_frames == ds.num_frames
    with zipfile.ZipFile(tmp_path / 'a.smlm') as z:
        m = json.loads(z.read('manifest.json'))
    f = m['formats'][m['files'][0]['format']]
    assert m['format_version'] == '0.2' and f['mode'] == 'binary' and f['headers'][:5] == ['id', 'frame', 'x', 'y', 'z']
    assert m['files'][0]['rows'] == len(ds) and f['units'][2] == 'nm'
    assert np.isclose(m['files'][0]['min']['x'], float((ds.pos[:, 0].min() + 0.5) * 108.0))
    # float32 (ShareLoc's usual table type), 2D, non-square pixels, other conventions
    d2 = Dataset.from_arrays(ds.pos[:, :2], ds.frame, crlb=ds.crlb[:, :2], photons=ds.photons,
                             meta=dict(pixel_nm=100.0, pixel_nm_y=120.0))
    c = Dataset.load(d2.save(tmp_path / 'b.smlm', frame_base=0, offset_px=0.0))
    assert c.n_dims == 2 and c.meta['pixel_nm_y'] == 120.0 and torch.equal(c.frame, d2.frame)
    assert torch.allclose(c.pos, d2.pos, atol=1e-5) and torch.allclose(c.crlb, d2.crlb, atol=1e-6)
    assert torch.allclose(Dataset.load_smlm(tmp_path / 'b.smlm', pixel_nm=50.0).pos[:, 0], d2.pos[:, 0] * 2, atol=1e-5)
    with pytest.raises(ValueError):
        Dataset.from_arrays(ds.pos, ds.frame).save_smlm(tmp_path / 'c.smlm')    # no pixel size


def test_smlm_foreign_file(tmp_path):
    """a ShareLoc-style table without the torch_psf block: pixel_nm is needed; one lateral uncertainty column"""
    import zipfile
    n, hdr = 20, ['frame', 'x', 'y', 'uncertainty_xy_nm_']
    a = np.zeros(n, np.dtype([(h, '<f4') for h in hdr]))
    a['frame'], a['x'], a['y'], a['uncertainty_xy_nm_'] = np.arange(1, n + 1), 150.0, 250.0, 10.0
    p = tmp_path / 'f.smlm'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('manifest.json', json.dumps(dict(format_version='0.2', formats={'t': dict(
            type='table', mode='binary', headers=hdr, dtype=['float32'] * 4)}, files=[dict(name='t.bin', format='t', rows=n)])))
        z.writestr('t.bin', a.tobytes())
    with pytest.raises(ValueError):
        Dataset.load(p)
    L = Dataset.load(p, pixel_nm=100.0)
    assert torch.allclose(L.pos[0], torch.tensor([1.0, 2.0], dtype=torch.float64)) and L.frame[0] == 0
    assert torch.allclose(L.crlb, torch.full((n, 2), 0.1))


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
    before = ds.copy()
    drift = ds.estimate_drift(device='cpu')
    assert ds.meta == before.meta and all(torch.equal(ds[k], before[k]) for k in ds.keys())
    d, rec = drift.curve, drift.info
    assert float((d - d.mean(0) - (true - true.mean(0))).abs().max()) < 0.05
    assert rec['method'] == 'DME v2' and rec['parameters']['n_rows'] == n
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


def test_concatenate_drift_records():
    ds = make()
    corrected = ds.drift_corrected(Drift([[1., 2.]], {}))
    joined = corrected[:3] + corrected[3:]
    assert joined.meta['drift'] == {} and torch.equal(joined.pos, corrected.pos)
    other = ds.drift_corrected(Drift([[1., 2.]], {'method': 'given'}))
    for first, second in [(ds, corrected), (corrected, ds), (corrected, other)]:
        with pytest.raises(ValueError, match='different drift correction records'):
            Dataset.concatenate([first, second])


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
