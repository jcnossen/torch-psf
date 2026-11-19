"""IntegratedSplinePSF (torch_psf.pixint): the integrated cubic B-spline as a file model (psf.json type
'integrated_spline', docs/PSF_FORMAT.md): accuracy against the exact vectorial model, per-angle coefficient sets, PSF
json / psf.zip round trips, and refusal by readers that do not know the type."""
import json
import numpy as np
import pytest
import torch
from torch_psf import PSF, VectorialPSF, CubicSplinePSF, IntegratedSplinePSF, io, psfzip


def _vec(R=11, dip=None):
    v = VectorialPSF(R, npupil=32, oversample=2, device='cpu', focus_um=0.2, z_range=(0.0, 0.3), wavelength_um=0.7,
                     pixel_integration='spectral')
    v.set_zernike({6: 0.05, 11: 0.03})
    if dip is not None:
        v.dipM = torch.tensor(dip, dtype=torch.float32)
    return v


DIP = ([[0.30, 0.0, 0.0], [0.0, 0.42, 0.0], [0.0, 0.0, 0.28]], [[0.42, 0.0, 0.0], [0.0, 0.30, 0.0], [0.0, 0.0, 0.28]])


def _pos(n=40, xy=1.5, z=(0.0, 0.2), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([(torch.rand(n, 2, generator=g) * 2 - 1) * xy, z[0] + torch.rand(n, 1, generator=g) * (z[1] - z[0])], 1)


def _err(a, b):
    a, b = a.detach(), b.detach()
    return float(((a - b).abs() / b.amax((1, 2), keepdim=True)).max())


def test_matches_exact_vectorial():
    """max |spline - VectorialPSF 'spectral'| < 5e-5 of the peak over the z range and offsets <= xy_limit, at S = 4"""
    v = _vec()
    s = IntegratedSplinePSF.from_vectorial(v, (0.0, 0.2), xy_limit=1.5, oversample=4, dz_um=0.01)
    pos = _pos()
    with torch.no_grad():
        assert _err(s.render(pos), v.render(pos)) < 5e-5
    assert s.xy_limit == 1.5 and s.z_limits == (0.0, 0.2) and s.n_angles == 0


def test_gradients_and_chunks():
    v = _vec()
    s = IntegratedSplinePSF.from_vectorial(v, (0.0, 0.2), xy_limit=1.5, oversample=3, dz_um=0.02)
    pos = _pos(6).double().float()
    val, gx, gy, gz = s.render(pos, grad=True)
    e = 1e-3
    for k, g in ((0, gx), (1, gy), (2, gz)):
        d = torch.zeros_like(pos); d[:, k] = e
        num = (s.render(pos + d) - s.render(pos - d)) / (2 * e)
        assert float((num - g).abs().max()) < 2e-3 * float(g.abs().max())
    s.chunk_elems = 4 * s.Lw ** 2 * 2                     # 2 emitters per chunk
    r = s.render(pos, grad=True)
    assert all(torch.equal(a, b) for a, b in zip(r, (val, gx, gy, gz)))


def test_angles_mean_and_selection():
    vs = [_vec(dip=d) for d in DIP]
    s = IntegratedSplinePSF.from_vectorial(vs, (0.0, 0.2), xy_limit=1.0, oversample=3, dz_um=0.02)
    assert s.n_angles == 2 and s.angle is None
    pos = _pos(8, xy=1.0)
    with torch.no_grad():
        a = [s.for_angle(k).render(pos) for k in range(2)]
        assert torch.allclose(s.render(pos), 0.5 * (a[0] + a[1]), atol=1e-7)
        for k in range(2):
            assert _err(a[k], vs[k].render(pos)) < 2e-4
    with pytest.raises(ValueError):
        s.for_angle(2)


def test_resized_shares_coefficients():
    s = IntegratedSplinePSF.from_vectorial(_vec(), (0.0, 0.2), roisize=15, xy_limit=1.0, oversample=3, dz_um=0.02)
    r = s.resized(7)
    assert r.coefs is s.coefs and r.roisize == 7 and r.xy_limit > 4          # a smaller ROI: larger offsets
    pos = _pos(5, xy=1.0)
    with torch.no_grad():
        assert torch.allclose(r.render(pos), s.render(pos)[:, 4:11, 4:11], atol=1e-7)
    assert s.resized(8).render(pos).shape == (5, 8, 8)                          # even ROI sizes too


@pytest.mark.parametrize('kind', ['json', 'zip'])
def test_round_trip(tmp_path, kind):
    vs = [_vec(dip=d) for d in DIP]
    s = IntegratedSplinePSF.from_vectorial(vs, (0.0, 0.2), xy_limit=1.0, oversample=3, dz_um=0.02)
    src = _vec(dip=np.mean(DIP, 0).tolist())
    f = tmp_path / ('m.json' if kind == 'json' else 'm.psf.zip')
    if kind == 'json':
        s.save(f, source=src)
    else:
        io.save_zip(f, s, source=src)
    pos = _pos(6, xy=1.0)
    for a in (None, 0, 1):
        m = PSF.load(f, angle=a, device='cpu')
        assert type(m) is IntegratedSplinePSF and m.angle == a and m.xy_limit == s.xy_limit
        assert torch.equal(m.render(pos), s.for_angle(a).render(pos))
    assert PSF.load(f, roisize=7, device='cpu').roisize == 7
    v = IntegratedSplinePSF.load_source(f, device='cpu')
    assert isinstance(v, VectorialPSF) and torch.allclose(v.zcoef, src.zcoef)
    if kind == 'zip':
        meta, arrays = psfzip.read_zip(f)
        ch = meta['channels'][0]
        assert ch['type'] == 'integrated_spline' and set(ch['arrays']) == {'coefs_angle'}
        assert ch['model']['integrated_spline']['basis'] == 'cubic_bspline' and ch['source']['type'] == 'vectorial'
        assert 'knots' not in ch['arrays'] and 'spline' not in ch['model']
        z = psfzip.PSFZip.read(f, device='cpu')
        assert isinstance(z.channels[0].source, VectorialPSF)


def test_single_model_round_trip(tmp_path):
    s = IntegratedSplinePSF.from_vectorial(_vec(), (0.0, 0.2), xy_limit=1.0, oversample=3, dz_um=0.02)
    f = tmp_path / 'm.psf.zip'
    io.save_zip(f, s)
    meta, _ = psfzip.read_zip(f)
    assert set(meta['channels'][0]['arrays']) == {'coefs'}
    m = PSF.load(f, angle=1, device='cpu')                        # no angle terms: the same model for every angle (as VectorialPSF)
    pos = _pos(4, xy=1.0)
    assert m.angle is None and torch.equal(m.render(pos), s.render(pos))


def test_readers_without_the_type_refuse(tmp_path, monkeypatch):
    """A reader that does not know 'integrated_spline' fails loudly: PSF.load / PSFZip.read raise (the type is not
    registered), and the coefficients are not where a 'spline' reader looks (no 'spline' block, no 'knots' array)"""
    s = IntegratedSplinePSF.from_vectorial(_vec(), (0.0, 0.2), xy_limit=1.0, oversample=3, dz_um=0.02)
    f = tmp_path / 'm.psf.zip'
    io.save_zip(f, s, source=_vec())
    d, arrays, typ = PSF.read_file(f)
    assert typ == 'integrated_spline'
    with pytest.raises(KeyError):
        CubicSplinePSF._from_file(d, arrays)           # a reader that took it for a 'spline' finds no grid / knots
    monkeypatch.delitem(PSF._registry, 'integrated_spline')
    with pytest.raises(ValueError, match='unknown PSF type'):
        PSF.load(f, device='cpu')
    with pytest.raises(KeyError):
        psfzip.PSFZip.read(f)


def test_unknown_basis_or_arrays_refused(tmp_path):
    s = IntegratedSplinePSF.from_vectorial(_vec(), (0.0, 0.2), xy_limit=1.0, oversample=3, dz_um=0.02)
    f = tmp_path / 'm.json'
    d = s.save(f)
    d['integrated_spline']['basis'] = 'quintic_bspline'
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='unknown basis'):
        PSF.load(f, device='cpu')
    d['integrated_spline']['basis'] = 'cubic_bspline'
    f.write_text(json.dumps(d))
    np.savez(tmp_path / 'm.npz', coefs=s.coefs.numpy(), coefs_angle=s.coefs.numpy()[None])
    with pytest.raises(ValueError, match='exactly one'):
        PSF.load(f, device='cpu')


def test_old_spline_files_unchanged(tmp_path):
    """the 'spline' type is written and read as before (CubicSplinePSF, knots)"""
    v = _vec()
    spl = v.to_spline(11, (0.0, 0.2), oversample=2, dz_um=0.05, xy_limit=2.0)
    f = tmp_path / 's.psf.zip'
    io.save_zip(f, spl, source=v)
    meta, _ = psfzip.read_zip(f)
    ch = meta['channels'][0]
    assert ch['type'] == 'spline' and set(ch['arrays']) == {'knots'} and 'basis' not in ch['model']['spline']
    m = PSF.load(f, device='cpu')
    pos = _pos(4, xy=1.0)
    assert type(m) is CubicSplinePSF and torch.equal(m.render(pos), spl.render(pos))


def test_convert_calibration_integrated_spline(tmp_path):
    """convert_calibration(integrated_spline=...): channel 0 = the integrated spline (one coefficient set per pattern
    angle), the vectorial model as its source; the calibration keys stay readable from the source"""
    calib = dict(pixelsize_um=0.1083, wavelength_um=0.7, NA=1.49, n_imm=1.518, n_med=1.33, zernike_noll=[6, 11],
                 zernike_waves=[0.05, 0.03], focus_um=0.2, npupil=32, oversample=2, pixel_integration='spectral',
                 psf_dipole_M_angle=[list(map(list, d)) for d in DIP], tirf_depth_um=0.1)
    p = tmp_path / 'calib.json'
    p.write_text(json.dumps(calib))
    out = tmp_path / 'calib.psf.zip'
    meta = psfzip.convert_calibration(p, out, device='cpu',
                                      integrated_spline=dict(z_range=(0.0, 0.2), roisize=11, xy_limit=1.0, oversample=3,
                                                             dz_um=0.02))
    ch = meta['channels'][0]
    assert ch['type'] == 'integrated_spline' and ch['model']['integrated_spline']['n_angles'] == 2
    assert ch['source']['model']['zernike_waves'] == [0.05, 0.03]
    assert meta['acquisition'] == dict(tirf_depth_um=0.1)
    pos = _pos(5, xy=1.0)
    for k in range(2):
        ref = PSF.load(p, roisize=11, angle=k, device='cpu')
        m = PSF.load(out, angle=k, device='cpu')
        assert _err(m.render(pos), ref.render(pos)) < 2e-4


def test_zip_version(tmp_path, monkeypatch):
    """0.3.0 exactly for the files a 0.2 reader would misread (integrated_spline, pixel_integration != 'box'), refused
    by a 0.2-only reader; 'box' files stay 0.2.0"""
    box = VectorialPSF(11, npupil=32, oversample=2, device='cpu', focus_um=0.2, z_range=(0.0, 0.3))
    cases = dict(box=(box, None, '0.2.0'), spectral=(_vec(), None, '0.3.0'),
                 spline=(IntegratedSplinePSF.from_vectorial(box, (0.0, 0.2), xy_limit=1.0, oversample=3, dz_um=0.02),
                         box, '0.3.0'),
                 box_spline=(box.to_spline(11, (0.0, 0.2), oversample=2, dz_um=0.05), box, '0.2.0'),
                 spectral_source=(box.to_spline(11, (0.0, 0.2), oversample=2, dz_um=0.05), _vec(), '0.3.0'))
    for name, (m, src, ver) in cases.items():
        f = tmp_path / f'{name}.psf.zip'
        meta = io.save_zip(f, m, source=src)
        assert meta['version'] == ver, name
        assert psfzip.read_zip(f)[0]['version'] == ver
        PSF.load(f, device='cpu')
    monkeypatch.setattr(psfzip, 'READ_VERSIONS', ((0, 2),))           # a 0.2 reader
    PSF.load(tmp_path / 'box.psf.zip', device='cpu')
    for name in ('spectral', 'spline', 'spectral_source'):
        with pytest.raises(ValueError, match='version 0.3.0'):
            PSF.load(tmp_path / f'{name}.psf.zip', device='cpu')
