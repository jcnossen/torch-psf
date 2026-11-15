"""psf.zip (docs/PSF_FORMAT.md): round trips of every model type, the zip members, the z-stack layout, old json + npz
files, the fit report with and without estimator data, independent channels with a mapping and field dependence."""
import json
import math
import os
import warnings
import zipfile
import numpy as np
import pytest
import tifffile
import torch
from torch_psf import (PSF, VectorialPSF, CubicSplinePSF, GaussianPSF, AstigmaticGaussianPSF, FourPiPSF,
                       FourPiSplinePSF, Camera, ZStackEstimator)
from torch_psf import io
from torch_psf.psfzip import PSFZip, poly_features, read_zip
from torch_psf.spots import Spots

DEV = 'cpu'
POS = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.1, 0.05]])
MEMBERS = {'psf.json', 'zstack.tif', 'preview.png', 'fit_report.pdf'}


def vectorial(R=13, **kw):
    v = VectorialPSF(R, npupil=24, oversample=1, device=DEV, focus_um=0.3, z_range=(0.0, 0.6), **kw)
    v.set_zernike({6: 0.1, 11: 0.02})
    return v


def fourpi():
    p = FourPiPSF(11, pixel_um=0.11, wavelength_um=0.67, NA=1.35, n_imm=1.406, npupil=16, oversample=1, device=DEV,
                  zernike_indices=tuple(range(5, 12)), phi=(0.3, 1.9, 3.4, 5.0), a=0.9, gamma=0.8,
                  d_bot=(0.3, -0.2, 0.02), z_range=(-0.2, 0.2))
    with torch.no_grad():
        p.top.zcoef[1] = 0.03
        p.bot.zcoef[2] = -0.02
    return p


def save(path, psf, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('error')           # no report section may fail
        return io.save_zip(path, psf, zstack_planes=7, **kw)


def models():
    v = vectorial()
    with torch.no_grad():
        v.pupil_phase.copy_(0.03 * v.mask * v.px)
        v.pupil_amp.copy_(1 + 0.1 * v.mask * v.py)
    v.blur = torch.tensor([0.1, 0.05, 0.01])
    v.dipM = torch.diag(torch.tensor([0.3, 0.3, 0.4]))
    s = vectorial().to_spline(11)
    fp = fourpi()
    return dict(vectorial=(v, None), spline=(s, vectorial()), gaussian=(GaussianPSF(11, 1.3, DEV, pixel_um=0.1), None),
                astig_gaussian=(AstigmaticGaussianPSF(11, (1.2, 1.4), 0.2, 0.4, device=DEV, pixel_um=0.1), None),
                fourpi=(fp, None), fourpi_spline=(fp.to_spline(11, oversample=2, dz_um=0.05, zpad=2), fp))


@pytest.mark.parametrize('typ', ['vectorial', 'spline', 'gaussian', 'astig_gaussian', 'fourpi', 'fourpi_spline'])
def test_round_trip(tmp_path, typ):
    psf, src = models()[typ]
    meta = save(tmp_path / 'p.zip', psf, source=src)
    q = PSF.load(tmp_path / 'p.zip', device=DEV)
    assert type(q) is type(psf) and q.type_name == typ
    with torch.no_grad():
        if typ.startswith('fourpi'):
            phi = torch.tensor([0.4, 2.5])
            assert torch.allclose(psf.render(POS, phi=phi), q.render(POS, phi=phi), atol=1e-7)
        else:
            assert torch.allclose(psf.render(POS), q.render(POS), atol=1e-7)
    names = set(zipfile.ZipFile(tmp_path / 'p.zip').namelist())
    assert MEMBERS <= names
    d, arrays = read_zip(tmp_path / 'p.zip')
    assert d['format'] == 'torch-psf' and d['version'] == '0.2.0' and d == json.loads(json.dumps(meta))
    refs = [m for c in d['channels'] for m in (c.get('arrays') or {}).values()]
    refs += [m for c in d['channels'] if c.get('source') for m in c['source']['arrays'].values()]
    refs += list((d.get('source') or {}).get('arrays', {}).values())
    assert set(refs) == {n for n in names if n.endswith('.npy')}
    C = getattr(psf, 'n_channels', 1)
    assert len(d['channels']) == C and d['zstack']['shape'][1] == C
    if typ == 'fourpi_spline':
        assert {c['type'] for c in d['channels']} == {'phase_spline'} and d['source']['type'] == 'fourpi'
        assert d['emitter_params'] == ['x', 'y', 'z', 'I', 'bg', 'phi']
        assert isinstance(q, FourPiSplinePSF) and q.source_state is not None
    if typ == 'fourpi':
        assert {c['type'] for c in d['channels']} == {'source_output'} and d['source']['type'] == 'fourpi'
    if typ == 'spline':
        assert d['channels'][0]['source']['type'] == 'vectorial'
        assert torch.allclose(io.load_spline_source(tmp_path / 'p.zip', device=DEV).zcoef, src.zcoef)
    if typ == 'gaussian':
        assert d['emitter_params'] == ['x', 'y', 'I', 'bg'] and d['zstack']['shape'][0] == 1


def test_zstack_layout(tmp_path):
    g = AstigmaticGaussianPSF(9, (1.2, 1.5), 0.2, 0.4, z_range=(-0.3, 0.3), device=DEV, pixel_um=0.1)
    io.save_zip(tmp_path / 'g.zip', g, zstack_px=0.5, zstack_planes=5)
    with zipfile.ZipFile(tmp_path / 'g.zip') as z:
        meta = json.loads(z.read('psf.json'))
        tif = tifffile.TiffFile(__import__('io').BytesIO(z.read('zstack.tif')))
        st = tif.asarray()
        ij = tif.imagej_metadata
    assert meta['zstack']['axes'] == 'ZCYX' and meta['zstack']['shape'] == [5, 1, 18, 18]
    st = st.reshape(meta['zstack']['shape'])
    assert st.dtype == np.float32 and abs(ij['spacing'] - 0.15) < 1e-6 and ij['unit'] == 'um'
    assert json.loads(ij['Info'])['z_um'] == meta['zstack']['z_um']
    # sample (m, n) = the camera pixel centred at u = (n - (kP - 1) / 2) / k px from the emitter
    k, P, zs = 2, 9, meta['zstack']['z_um']
    for zi, m, n in ((0, 9, 9), (2, 5, 12), (4, 14, 3)):
        u, v = (n - (k * P - 1) / 2) / k, (m - (k * P - 1) / 2) / k
        ref = g.render(torch.tensor([[-u, -v, zs[zi]]]))[0, 4, 4]
        assert abs(float(ref) - float(st[zi, 0, m, n])) < 1e-6


def test_old_json_still_loads_and_save_dispatch(tmp_path):
    v = vectorial()
    v.save(tmp_path / 'v.json')
    assert json.load(open(tmp_path / 'v.json'))['version'] == '0.1.0'
    assert torch.equal(PSF.load(tmp_path / 'v.json', device=DEV).render(POS), v.render(POS))
    io.save(tmp_path / 'v.zip', v)
    assert zipfile.is_zipfile(tmp_path / 'v.zip')
    assert torch.equal(io.load(tmp_path / 'v.zip', device=DEV).render(POS), v.render(POS))


def test_camera_maps(tmp_path):
    cam = Camera(gain_e_per_adu=0.5, offset_adu=np.full((8, 10), 100.0, np.float32), readnoise_e=1.2)
    save(tmp_path / 'g.zip', GaussianPSF(11, 1.3, DEV, pixel_um=0.1), camera=cam)
    c = io.load_camera(tmp_path / 'g.zip')
    assert c.gain_e_per_adu == 0.5 and c.offset_adu.shape == (8, 10) and c.readnoise_e == 1.2


def test_report_model_pages_only(tmp_path):
    assert save(tmp_path / 'v.zip', vectorial())['report']['sections'] == ['summary', 'aberrations', 'pupil', 'psf']
    assert save(tmp_path / 'g.zip', GaussianPSF(11, 1.3, DEV))['report']['sections'] == ['summary', 'psf']
    secs = save(tmp_path / 'f.zip', fourpi())['report']['sections']
    assert secs == ['summary', 'aberrations', 'pupil', 'psf']


def test_report_zstack_estimator(tmp_path):
    vec = VectorialPSF(13, npupil=24, oversample=1, device=DEV).set_zernike({6: 0.08})
    nk, H = 9, 32
    focus = 0.08 * (torch.arange(nk) - (nk - 1) / 2)
    stack = torch.full((nk, H, H), 20.0)
    img = vec.forward_batched(torch.tensor([[0.2, -0.1, 0.0]] * nk), focus)
    for cx, cy in ((9, 10), (22, 21)):
        stack[:, cy - 6:cy + 7, cx - 6:cx + 7] += 20000 * img
    est = ZStackEstimator('vectorial', optics=dict(pixel_um=0.10833, NA=1.49, wavelength_um=0.68), roisize=13,
                          npupil=24, oversample_pupil=1, iters=5, device=DEV)
    psf = est.estimate([torch.poisson(stack).numpy()], dz_um=0.08)
    assert est.data['model'].shape == est.data['rois'].shape and est.data['chi2'].shape == (2, nk)
    secs = save(tmp_path / 'p.zip', psf, report=est)['report']['sections']
    assert secs == ['summary', 'spots', 'quality', 'fit_stats', 'convergence', 'aberrations', 'pupil', 'psf',
                    'examples', 'residuals']
    # the report of a reloaded PSF comes from its provenance (no history / data)
    secs = save(tmp_path / 'q.zip', PSF.load(tmp_path / 'p.zip', device=DEV))['report']['sections']
    assert secs == ['summary', 'spots', 'aberrations', 'pupil', 'psf']


def test_report_insitu_style_data(tmp_path):
    """the InSituEstimator report / history / data keys (spots refitted with data['spline'])"""
    g = AstigmaticGaussianPSF(11, (1.2, 1.4), 0.2, 0.4, z_range=(-0.4, 0.4), device=DEV, pixel_um=0.1)
    n = 60
    gen = torch.Generator().manual_seed(1)
    th = torch.stack([torch.rand(n, generator=gen) - 0.5, torch.rand(n, generator=gen) - 0.5,
                      0.6 * torch.rand(n, generator=gen) - 0.3, torch.full((n,), 3000.0), torch.full((n,), 10.0)], 1)
    rois = torch.poisson(g.expected(th), generator=gen)
    sp = Spots(rois, torch.arange(n), torch.zeros(n, 2, dtype=torch.long), var=torch.tensor(0.0))
    rep = dict(n_spots=n, n_train=40, n_test=20, n_rejected=2, at_limit=0.0, focus_um=0.1, z_scale=1.02,
               deviance=dict(rule='median', k=3.0, median=1.0, scale=0.1, threshold=1.3, rejected_fraction=0.05),
               scan=dict(results=[dict(focus_um=f, wavelength_um=0.68, heldout_nll_per_px=2 + (f - 0.1) ** 2)
                                  for f in (0.0, 0.1, 0.2)], best=dict(focus_um=0.1, wavelength_um=0.68,
                                                                          heldout_nll_per_px=2.0)),
               warnings=['example warning'])
    hist = [dict(stage='vectorial', it=i, value=100.0 / (i + 1), seconds=0.1 * i) for i in range(10)]
    data = dict(spots=sp, train=np.arange(40), test=np.arange(40, 60), spline=g, vectorial_spline=g,
                vectorial=vectorial())
    secs = save(tmp_path / 'p.zip', g, report=dict(report=rep, history=hist, data=data))['report']['sections']
    assert secs == ['summary', 'spots', 'quality', 'fit_stats', 'convergence', 'aberrations', 'pupil', 'psf',
                    'examples', 'residuals']


def insitu_overlaps(n=120, frac=0.1, seed=2):
    """spots of an astigmatic Gaussian, a fraction with a second emitter in the ROI, and the InSituEstimator data
    keys of the first-fit overlap test (deviance, deviance_theta, deviance_psf)"""
    from torch_psf.estimation import spot_deviance
    g = AstigmaticGaussianPSF(11, (1.2, 1.4), 0.2, 0.4, z_range=(-0.4, 0.4), device=DEV, pixel_um=0.1)
    gen = torch.Generator().manual_seed(seed)
    u = lambda: torch.rand(n, generator=gen)
    th = torch.stack([u() - 0.5, u() - 0.5, 0.6 * u() - 0.3, 2000 + 2000 * u(), torch.full((n,), 10.0)], 1)
    th2 = th + torch.tensor([3.0, -2.0, 0.0, 0.0, 0.0])
    th2[:, 4] = 0
    mu = g.expected(th) + (u() < frac)[:, None, None] * g.expected(th2)
    rois = torch.poisson(mu, generator=gen)
    sp = Spots(rois, torch.arange(n), torch.zeros(n, 2, dtype=torch.long), var=torch.tensor(0.0))
    tr = np.arange(n * 2 // 3)
    dv, th1 = spot_deviance(g, rois[:len(tr)])
    return g, sp, tr, dict(spots=sp, train=tr, test=np.arange(len(tr), n), spline=g, vectorial_spline=g,
                           deviance=dv, deviance_theta=th1, deviance_psf=g)


@pytest.mark.parametrize('rejected', [True, False])
def test_report_quality_range(tmp_path, rejected):
    """quality page: q = the stored first-fit deviance and the estimator's threshold; best / last accepted / first
    rejected bands (the last only with rejected spots); models = the first-fit single-emitter fits"""
    from torch_psf import psfreport
    g, sp, tr, data = insitu_overlaps()
    dv = data['deviance']
    thr = float(np.quantile(dv, 0.85)) if rejected else float(dv.max()) + 1
    rep = dict(n_spots=len(sp), n_train=len(tr), n_test=len(sp) - len(tr), n_rejected=int((dv >= thr).sum()),
               deviance=dict(rule='median', k=3.0, median=float(np.median(dv)), scale=0.1, threshold=thr))
    b = dict(report=rep, history=[], data=data)
    ctx = psfreport._Ctx(g, {}, None, None, psfreport.bundle(b, g), None, 1000)
    qd = psfreport._quality(ctx)
    assert qd['kind'] == 'rejection' and qd['cutoff'] == thr and np.array_equal(qd['q'], dv)
    rows = qd['rows']
    assert [r['rank'] for r in rows['best']] == [0, 1, 2]
    assert len(rows['accepted']) == 3 and all(r['q'] < thr for r in rows['accepted'])
    acc_q = np.sort(dv[dv < thr])
    assert [r['q'] for r in rows['accepted']] == pytest.approx(acc_q[-3:].tolist())
    if rejected:
        assert len(rows['rejected']) == 3 and all(r['q'] >= thr for r in rows['rejected'])
        assert [r['q'] for r in rows['rejected']] == pytest.approx(np.sort(dv[dv >= thr])[:3].tolist())
    else:
        assert rows['rejected'] == []
    i = int(np.argmin(dv))                                    # the model is the stored first fit of that spot
    mu = g.expected(data['deviance_theta'][i:i + 1])[0].numpy()
    assert np.allclose(rows['best'][0]['model'], mu) and np.array_equal(rows['best'][0]['data'], sp.rois[i].numpy())
    assert len(psfreport.s_quality(ctx)) == 1
    secs = save(tmp_path / 'p.zip', g, report=b)['report']['sections']
    assert 'quality' in secs and secs.index('quality') == secs.index('spots') + 1


def test_report_quality_zstack_reference():
    """ZStack: q = chi2 per px per bead, no rejection: a labelled reference line and all beads accepted by it or not"""
    from torch_psf import psfreport
    nb, nk, P = 8, 5, 9
    gen = torch.Generator().manual_seed(0)
    mu = 5 + 200 * torch.rand(nb, nk, P, P, generator=gen)
    rois = torch.poisson(mu, generator=gen)
    rois[3] += 300 * (torch.rand(nk, P, P, generator=gen) > 0.9)          # a bead the model does not explain
    c2 = ((rois - mu) ** 2 / mu).mean((2, 3))
    data = dict(rois=rois, model=mu, chi2=c2, photons=torch.full((nb, nk), 1e4), bg=torch.full((nb, nk), 5.0),
                focus_um=torch.linspace(-0.2, 0.2, nk), stack_id=torch.zeros(nb, dtype=torch.long))
    rep = dict(n_beads=nb, n_planes=nk)
    ctx = psfreport._Ctx(None, {}, None, None, psfreport.bundle(dict(report=rep, data=data), None), None, 1000)
    qd = psfreport._quality(ctx)
    q = c2.mean(1).numpy()
    assert qd['kind'] == 'reference' and np.allclose(qd['q'], q)
    assert any('not a rejection' in t for t in qd['notes'])
    assert [r['label'].split(' ')[1] for r in qd['rows']['rejected']] == ['3']          # above the reference line
    assert len(psfreport.s_quality(ctx)) == 1


def simcode_feats(x, y, order, norm):
    """SIMCODE PolyMap.feats, verbatim (simcode/channels/registration.py)"""
    cx, cy, sx, sy = norm
    u, v = (np.asarray(x) - cx) / sx, (np.asarray(y) - cy) / sy
    return np.stack([u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i)], -1)


def test_channels_mapping_and_field(tmp_path):
    v0 = vectorial()
    v1 = vectorial()
    v1.focus_um = 0.6                                     # biplane: the second plane focused deeper
    norm = dict(cx=64.0, cy=60.0, sx=64.0, sy=60.0)
    nf = 6                                                # order 2
    rng = np.random.default_rng(0)
    fwd = np.zeros((nf, 2)); fwd[0] = (130.2, -1.5); fwd[3] = (-64.0, 0.4); fwd[1] = (0.3, 60.0)
    fwd[2] += rng.normal(0, 0.3, 2); fwd[5] += rng.normal(0, 0.3, 2)
    inv = np.zeros((nf, 2)); inv[0] = (130.0, 1.5); inv[3] = (-64.0, -0.4); inv[1] = (-0.3, 60.0)
    polymap = dict(order=2, fwd={'1': fwd.tolist()}, inv={'1': inv.tolist()}, feats_norm=norm, n_pairs=5000)
    noll = [6, 11]
    coef = np.zeros((2, 6))
    coef[:, 0] = [0.1, 0.02]                                 # = the model's Zernikes at the centre (u = v = 0)
    coef[0, 3], coef[0, 1], coef[1, 5] = 0.03, -0.02, 0.01   # u, v, u^2
    field = dict(type='zernike_poly', order=2, zernike_noll=noll, coef=coef, feats_norm=norm)
    crop0, crop1 = dict(x0=0, y0=0, w=128, h=120), dict(x0=128, y0=0, w=128, h=120)
    meta = save(tmp_path / 'bp.zip', [v0, v1], channels=[dict(name='plane0', crop=crop0, field=field),
                                                       dict(name='plane1', crop=crop1, flip='lr', mapping=polymap,
                                                            intensity_ratio=0.9)])
    assert 'field' in meta['report']['sections']
    assert meta['channels'][1]['mapping']['type'] == 'poly' and meta['per_channel_params'] == ['bg']
    with pytest.raises(ValueError):
        PSF.load(tmp_path / 'bp.zip', device=DEV)
    assert torch.equal(io.load(tmp_path / 'bp.zip', device=DEV, channel='plane1').render(POS), v1.render(POS))
    pz = PSFZip.read(tmp_path / 'bp.zip', device=DEV)
    c0, c1 = pz.channels
    assert c1.name == 'plane1' and c1.intensity_ratio == 0.9 and c1.crop == crop1 and c1.entry['flip'] == 'lr'
    x, y = np.array([3.0, 70.5, 120.0]), np.array([5.0, 61.0, 110.0])
    n4 = (64.0, 60.0, 64.0, 60.0)
    assert np.allclose(poly_features(x, y, 2, n4), simcode_feats(x, y, 2, n4))
    F = simcode_feats(x, y, 2, n4)
    assert np.allclose(np.stack(c1.to_ref(x, y), -1), F @ fwd)
    assert np.allclose(np.stack(c1.from_ref(x, y), -1), F @ inv)
    assert np.allclose(np.stack(c0.to_ref(x, y), -1), np.stack([x, y], -1))
    # field dependence: the channel model at a field point = the model with the polynomial Zernikes there
    assert torch.equal(c0.model_at(64.0, 60.0).render(POS), v0.render(POS))
    for fx, fy in ((10.0, 20.0), (64.0, 100.0), (120.0, 5.0)):
        w = simcode_feats(fx, fy, 2, n4) @ coef.T
        ref = v0.resized(13).set_zernike({6: float(w[0]), 11: float(w[1])})
        assert torch.allclose(c0.model_at(fx, fy).render(POS), ref.render(POS), atol=1e-7)
    assert c1.field is None and c1.model_at(1.0, 2.0) is c1.model


def test_field_mismatch_warns(tmp_path):
    field = dict(type='zernike_poly', order=1, zernike_noll=[6], coef=[[0.2, 0.0, 0.01]],
                 feats_norm=dict(cx=0, cy=0, sx=10, sy=10))
    with pytest.warns(UserWarning, match='reference point'):
        io.save_zip(tmp_path / 'v.zip', vectorial(), zstack_planes=3, channels=[dict(field=field)])


def test_grid_mapping_and_names(tmp_path):
    g = GaussianPSF(9, 1.3, DEV, pixel_um=0.1)
    field = np.zeros((2, 20, 30), np.float32)
    field[0] += 2.5                                       # x0 = x + 2.5
    field[1] += np.linspace(0, 1, 20)[:, None]            # y0 = y + y / 19
    save(tmp_path / 'g.zip', [g, g], channels=[dict(name='a'), dict(name='b', mapping=dict(type='grid', fwd=field))])
    names = set(zipfile.ZipFile(tmp_path / 'g.zip').namelist())
    assert 'b/mapping_fwd.npy' in names
    c = PSFZip.read(tmp_path / 'g.zip', device=DEV).channel('b')
    x0, y0 = c.to_ref(np.array([4.0, 10.0]), np.array([0.0, 19.0]))
    assert np.allclose(x0, [6.5, 12.5]) and np.allclose(y0, [0.0, 20.0])
    x, y = c.from_ref(x0, y0)
    assert np.allclose(x, [4.0, 10.0], atol=1e-3) and np.allclose(y, [0.0, 19.0], atol=1e-2)


def test_convert_calibration_lossless(tmp_path):
    """A SIMCODE-style calibration json (PSF keys at the top level, camera, pattern, sample metadata) converts to a
    psf.zip with the same PSF and camera; the other keys go to provenance / patterns / acquisition."""
    import json
    import torch_psf as tp
    from torch_psf import psfzip
    calib = dict(pixelsize_um=0.1083, wavelength_um=0.68, NA=1.49, n_imm=1.518, n_med=1.33,
                 zernike_noll=[5, 6, 7, 8, 11], zernike_waves=[0.02, -0.03, 0.01, 0.0, 0.015], focus_um=0.2,
                 camera=dict(gain_e_per_adu=0.43, offset_adu=104.0, readnoise_e=1.2),
                 pattern=dict(pitch_nm=[224.1, 224.3], depth=[0.92, 0.95]), pattern_note='two angles',
                 tirf_depth_um=0.12, emitter_z_median_um=0.05, emitter_z_range_um=[0.0, 0.4],
                 source='test calibration', psf_v2=dict(script='psf_refit.py'))
    p = tmp_path / 'calib_test.json'
    p.write_text(json.dumps(calib))
    out = tmp_path / 'calib_test.psf.zip'
    meta = psfzip.convert_calibration(p, out, roisize=11, device='cpu')
    assert meta['acquisition'] == dict(tirf_depth_um=0.12, emitter_z_median_um=0.05, emitter_z_range_um=[0.0, 0.4])
    assert meta['patterns']['type'] == 'simcode' and meta['patterns']['pitch_nm'] == [224.1, 224.3]
    assert meta['patterns']['pattern_note'] == 'two angles'
    assert meta['provenance']['source'] == 'test calibration'
    assert meta['provenance']['converted_from']['json'] == calib
    z = psfzip.PSFZip.read(out, roisize=11, device='cpu')
    assert z.acquisition['tirf_depth_um'] == 0.12
    a = tp.PSF.load(str(p), roisize=11, device='cpu')
    th = torch.tensor([[0.3, -0.2, 0.1, 1000.0, 5.0]])
    assert torch.allclose(a.expected(th), z.psf.expected(th), atol=1e-6)
    assert abs(z.camera.gain_e_per_adu - 0.43) < 1e-9 and abs(z.camera.offset_adu - 104.0) < 1e-9
    assert meta['optics']['n_med'] == 1.33 and meta['optics']['NA'] == 1.49     # as in the json, not float32-rounded
    assert meta['channels'][0]['model']['optics']['n_med'] == 1.33


def test_convert_camera_maps_and_gaussian(tmp_path):
    """Camera with maps: the calibrated scalars, free-form keys and all npz entries are kept; a Gaussian calibration
    keeps its optics and puts the unused vectorial keys into provenance; the zip is byte-reproducible."""
    import json
    from torch_psf import psfzip
    rng = np.random.default_rng(0)
    np.savez(tmp_path / 'maps.npz', offset_adu=(100 + rng.random((6, 8))).astype(np.float32),
             readnoise_e=(0.8 + 0.1 * rng.random((6, 8))).astype(np.float32), gain=np.float64(0.56),
             note=np.array('a note'), low5=np.ones((6, 8), np.float32))
    calib = dict(pixelsize_um=0.10833, wavelength_um=0.7, NA=1.4900000095367432, n_imm=1.518, n_med=1.33,
                 zernike_noll=[5, 6], zernike_waves=[0.01, -0.02], focus_um=0.147, psf_gauss_sigma_um=0.140829,
                 camera=dict(gain_e_per_adu=0.4256, offset_adu=103.926, readnoise_e=0.9, maps='maps.npz',
                             note='v2 camera', replaces='old.npz', sensor_shift_vs_0p5nM=[15, -6]),
                 tirf_depth_um=0.082)
    p = tmp_path / 'calib_g.json'
    p.write_text(json.dumps(calib))
    out = tmp_path / 'calib_g.psf.zip'
    meta = psfzip.convert_calibration(p, out, device='cpu')
    assert meta['psf_type'] == 'gaussian'
    assert meta['optics'] == dict(pixel_um=0.10833, NA=1.4900000095367432, wavelength_um=0.7, n_imm=1.518, n_med=1.33)
    assert set(meta['provenance']['unused_model_keys']) == {'zernike_noll', 'zernike_waves', 'focus_um'}
    assert meta['acquisition'] == dict(tirf_depth_um=0.082)
    c = meta['camera']
    assert (c['offset_adu'], c['readnoise_e'], c['gain_e_per_adu']) == (103.926, 0.9, 0.4256)
    assert c['note'] == 'v2 camera' and c['replaces'] == 'old.npz' and c['sensor_shift_vs_0p5nM'] == [15, -6]
    assert set(c['extra_arrays']) == {'gain', 'note', 'low5'}
    z = psfzip.PSFZip.read(out, device='cpu')
    cam = z.camera
    assert cam.extra['note'] == 'v2 camera' and str(cam.map_extra['note']) == 'a note'
    assert float(cam.map_extra['gain']) == 0.56 and cam.scalars['offset_adu'] == 103.926
    assert np.array_equal(cam.offset_adu, np.load(tmp_path / 'maps.npz')['offset_adu'])
    # a re-save of the read camera keeps everything
    out2 = tmp_path / 'resave.psf.zip'
    m2 = psfzip.save_zip(out2, z.psf, camera=cam, zstack_planes=3)
    c2 = m2['camera']
    assert (c2['offset_adu'], c2['readnoise_e'], c2['note']) == (103.926, 0.9, 'v2 camera')
    assert set(c2['extra_arrays']) == {'gain', 'note', 'low5'}
    # byte-reproducible
    out3 = tmp_path / 'again.psf.zip'
    psfzip.convert_calibration(p, out3, device='cpu')
    assert out.read_bytes() == out3.read_bytes()


def test_fourpi_state_exact_and_knot_options(tmp_path):
    """FourPiPSF.state() round-trips bit-exactly (raw parameters); 4Pi spline knots as float16 or rebuilt from the
    source on load; the estimator report is in psf.json."""
    fp = fourpi()
    with torch.no_grad():
        fp.tau_logit.copy_(torch.tensor([0.1, -0.2, 0.05, 0.3]))
    q = FourPiPSF.from_state(json.loads(json.dumps(fp.state())), device=DEV)
    for (k, a), (_, b) in zip(fp.named_parameters(), q.named_parameters()):
        assert torch.equal(a, b), k
    phi = torch.tensor([0.4, 2.5])
    assert torch.equal(fp.render(POS, phi=phi), q.render(POS, phi=phi))
    sp = fp.to_spline(11, oversample=2, dz_um=0.05, zpad=2)
    m32 = save(tmp_path / 's32.zip', sp, source=fp, report=dict(n_spots=12, loss=0.5, big=list(range(20000))))
    assert m32['report']['results'] == dict(n_spots=12, loss=0.5)
    s32 = PSF.load(tmp_path / 's32.zip', device=DEV)
    save(tmp_path / 's16.zip', sp, source=fp, spline_knots='float16')
    s16 = PSF.load(tmp_path / 's16.zip', device=DEV)
    save(tmp_path / 'srb.zip', sp, source=fp, spline_knots='rebuild')
    assert not any(n.endswith('.npy') for n in zipfile.ZipFile(tmp_path / 'srb.zip').namelist())
    srb = PSF.load(tmp_path / 'srb.zip', device=DEV)
    with torch.no_grad():
        ref = sp.render(POS, phi=phi)
        assert torch.equal(s32.render(POS, phi=phi), ref)
        assert torch.allclose(srb.render(POS, phi=phi), ref, atol=1e-6)
        assert (s16.render(POS, phi=phi) - ref).abs().max() < 2e-3 * ref.abs().max()
    assert os.path.getsize(tmp_path / 'srb.zip') < os.path.getsize(tmp_path / 's16.zip') < os.path.getsize(tmp_path / 's32.zip')


def test_zip_reproducible_across_minutes(tmp_path, monkeypatch):
    """no wall-clock time in the zip (the fit-report title once carried one): a clock that changes on every call
    must not change the bytes"""
    import itertools
    import time as _time
    n = itertools.count()
    monkeypatch.setattr(_time, 'strftime', lambda *a: f'2026-01-01 00:{next(n):02d}')
    a, b = tmp_path / 'a.zip', tmp_path / 'b.zip'
    save(a, GaussianPSF(11, 1.3, DEV, pixel_um=0.1))
    save(b, GaussianPSF(11, 1.3, DEV, pixel_um=0.1))
    assert a.read_bytes() == b.read_bytes()


def test_fourpi_arm_terms_round_trip(tmp_path):
    """the per-objective terms of the 4Pi arms (NA, apodization, supercritical-angle terms, diattenuation, pupil maps)
    survive state() / from_state() and a psf.zip"""
    fp = fourpi()
    with torch.no_grad():
        fp.top.NA.fill_(1.33); fp.bot.NA.fill_(1.31)
        fp.top.apod.copy_(torch.tensor([0.1, -0.05, 0.02])); fp.bot.saf.copy_(torch.tensor([0.01, 0.0, 0.05, -0.02]))
        fp.top.pupil_phase.copy_(0.02 * fp.top.mask * fp.top.px); fp.bot.pupil_amp.copy_(1 + 0.1 * fp.bot.mask * fp.bot.py)
    fp.bot.diat = 0.05
    phi = torch.tensor([0.4, 2.5])
    q = FourPiPSF.from_state(json.loads(json.dumps(fp.state())), device=DEV)
    z = save(tmp_path / 'f.zip', fp)
    r = PSF.load(tmp_path / 'f.zip', device=DEV)
    with torch.no_grad():
        ref = fp.render(POS, phi=phi)
        assert torch.equal(q.render(POS, phi=phi), ref)
        assert torch.equal(r.render(POS, phi=phi), ref)
    assert z['source']['model']['fourpi']['arms']['bottom']['diattenuation'] == 0.05
