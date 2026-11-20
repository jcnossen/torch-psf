"""The psf.zip files written by the web app (web/test/test_psfzip.ts; saved by `python web/test/run_tests.py` into
web/test/out/) read with torch_psf: the Zernikes and optics, the PSF rendered by torch_psf against the web's
zstack.tif, the ImageJ hyperstack metadata of zstack.tif (tifffile: ZCYX, spacing, unit, resolution, Info), the zip
container (member order, compression, times) and the spline channel (knots against the source tabulated by torch_psf).
Skipped when the web test has not been run.

    python -m pytest tests/test_web_psfzip.py -s       # -s prints the comparison numbers
"""
import io
import json
import os
import zipfile
import numpy as np
import pytest
import tifffile
import torch
from torch_psf import VectorialPSF, SplinePSF
from torch_psf import io as tio
from torch_psf.psfzip import PSFZip, read_zip, render_zstack, zstack_tif_bytes

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web', 'test', 'out')
EXPECTED = os.path.join(OUT, 'web_psfzip_expected.json')
pytestmark = pytest.mark.skipif(not os.path.exists(EXPECTED), reason='run `python web/test/run_tests.py` first (writes web/test/out/web_*.psf.zip)')
NAMES = ['web_vectorial.psf.zip', 'web_spline.psf.zip']


def expected():
    return json.load(open(EXPECTED))


def members(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}, z.infolist()


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / np.abs(b).max())


@pytest.mark.parametrize('name', NAMES)
def test_zip_container(name):
    m, infos = members(os.path.join(OUT, name))
    assert infos[0].filename == 'psf.json'
    assert {'psf.json', 'zstack.tif', 'preview.png', 'fit_report.pdf'} <= set(m)
    assert not any('resid' in n for n in m)
    for i in infos:
        assert i.date_time == (1980, 1, 1, 0, 0, 0), i
        assert i.compress_type == (zipfile.ZIP_STORED if i.filename == 'preview.png' else zipfile.ZIP_DEFLATED), i
    assert m['preview.png'][:8] == b'\x89PNG\r\n\x1a\n'
    assert m['fit_report.pdf'][:5] == b'%PDF-' and b'/CreationDate' not in m['fit_report.pdf']
    meta = json.loads(m['psf.json'])
    assert meta['format'] == 'torch-psf' and meta['version'] == '0.2.0'
    assert set(read_zip(os.path.join(OUT, name))[1]) == {n for n in m if n.endswith('.npy')}


@pytest.mark.parametrize('name', NAMES)
def test_model_zernikes_optics(name):
    e = expected()
    pz = PSFZip.read(os.path.join(OUT, name), device='cpu')
    ch = pz.channels[0]
    vec = ch.model if isinstance(ch.model, VectorialPSF) else ch.source
    assert isinstance(vec, VectorialPSF)
    assert list(vec.zidx) == e['zernike_noll']
    np.testing.assert_array_equal(vec.zcoef.detach().numpy(), np.float32(e['zernike_waves']))
    o = vec.optics
    for k, v in e['optics'].items():
        assert getattr(o, k) == pytest.approx(v, rel=1e-7), k
        assert pz.meta['optics'][k] == v
    assert vec.focus_um == e['focus_um'] and vec.npupil == e['npupil'] and vec.os == e['oversample']
    assert vec.roisize == e['roisize'] and list(vec.z_range) == e['z_range_um']
    assert pz.camera.gain_e_per_adu == e['camera']['gain_e_per_adu'] and pz.camera.offset_adu == e['camera']['offset_adu']
    psf = tio.load(os.path.join(OUT, name), device='cpu')                       # the single-channel reader
    assert type(psf) is (SplinePSF if 'spline' in name else VectorialPSF)
    assert pz.meta['emitter_params'] == ['x', 'y', 'z', 'I', 'bg'] and pz.meta['n_dims'] == 3


@pytest.mark.parametrize('name', NAMES)
def test_render_vs_web_zstack(name):
    path = os.path.join(OUT, name)
    pz = PSFZip.read(path, device='cpu')
    zb = pz.meta['zstack']
    m, _ = members(path)
    web = tifffile.imread(io.BytesIO(m['zstack.tif']), squeeze=False).reshape(zb['shape'])
    py, z = render_zstack(pz.psf, len(zb['z_um']), zb['sample_px'])
    np.testing.assert_array_equal(z, zb['z_um'])
    assert py.shape == web.shape == tuple(zb['shape'])
    e = rel(web, py)
    print(f'\n{name}: torch_psf render vs web zstack.tif {py.shape}: max |diff| / max {e:.2e}, '
          f'sums {py.sum((2, 3)).max():.6f} / {web.sum((2, 3)).max():.6f}')
    assert e < (1e-5 if 'spline' in name else 1e-4)


@pytest.mark.parametrize('name', NAMES)
def test_zstack_tif_metadata(name):
    path = os.path.join(OUT, name)
    m, _ = members(path)
    meta = json.loads(m['psf.json'])
    zb = meta['zstack']
    with tifffile.TiffFile(io.BytesIO(m['zstack.tif'])) as t:
        assert t.is_imagej
        s = t.series[0]
        axes, shape = s.get_axes(False), s.get_shape(False)           # ImageJ order TZCYXS, T = S = 1
        assert axes == 'TZCYXS' and shape[0] == shape[-1] == 1 and tuple(shape[1:5]) == tuple(zb['shape']), (axes, shape)
        assert s.dtype == np.float32
        ij = t.imagej_metadata
        assert ij['unit'] == 'um' and ij['spacing'] == zb['z_um'][1] - zb['z_um'][0]
        info = json.loads(ij['Info'])
        assert info == dict(format='torch-psf', version='0.2.0', psf_type=meta['psf_type'], z_um=zb['z_um'], channels=zb['channels'],
                            sample_px=zb['sample_px'])
        p = t.pages[0]
        xr, yr = p.tags['XResolution'].value, p.tags['YResolution'].value
        assert xr == yr and xr[0] / xr[1] == pytest.approx(1 / zb['sample_um'], rel=1e-9)
        assert zb['sample_um'] == meta['optics']['pixel_um'] * zb['sample_px']
        # the same tags as torch_psf's own writer for this stack
        stack = t.asarray().reshape(zb['shape'])
        with tifffile.TiffFile(io.BytesIO(zstack_tif_bytes(stack, np.asarray(zb['z_um']), zb['sample_um'], info=info))) as tp:
            q = tp.pages[0]
            for tag in ('ImageDescription', 'XResolution', 'YResolution', 'ResolutionUnit', 'BitsPerSample', 'SampleFormat',
                        'IJMetadataByteCounts', 'IJMetadata'):
                assert p.tags[tag].value == q.tags[tag].value, tag
            assert tp.imagej_metadata == ij
            np.testing.assert_array_equal(tp.asarray(), t.asarray())


def test_spline_knots_vs_torch_psf():
    e = expected()
    pz = PSFZip.read(os.path.join(OUT, 'web_spline.psf.zip'), device='cpu')
    spl, src = pz.channels[0].model, pz.channels[0].source
    sp = e['spline']
    ref = src.to_spline(roisize=sp['roisize'], z_range=tuple(sp['z_range']), oversample=sp['S'], dz_um=sp['hz'], zpad=sp['zpad'],
                        xy_limit=sp['xmax'])
    assert spl.to_dict() == ref.to_dict()
    err = rel(spl.knots.numpy(), ref.knots.numpy())
    print(f'\nweb spline knots {tuple(spl.knots.shape)} vs torch_psf sample: max rel err {err:.2e}')
    assert err < 1e-4
