"""File feature declarations, independently of the default for newly sampled tables."""
import json
import zipfile

import pytest
import torch

from torch_psf import ChannelsPSF, FieldPSF, FieldZernikes, FourPiPSF, PSF, VectorialPSF, psfzip, spline
from torch_psf import psf as psf_module


POS = torch.tensor([[.21, -.16, .014], [-.27, .31, .033]])
GRID = dict(dz_um=.02, xy_limit=.5)
NEW = [('catmull', True), ('bspline', False)]


def vectorial(mode='midpoint'):
    return VectorialPSF(5, npupil=16, oversample=2, device='cpu', z_range=(0., .04),
                        pixel_integration=mode)


def equal_render(actual, expected, **options):
    assert torch.equal(actual.render(POS, **options), expected.render(POS, **options))
    for a, b in zip(actual.render(POS, grad=True, **options), expected.render(POS, grad=True, **options)):
        assert torch.equal(a, b)


@pytest.mark.parametrize('kernel,integrated,version', [
    ('catmull', False, '0.2.0'), ('bspline', True, '0.3.0'),
    ('catmull', True, '0.5.0'), ('bspline', False, '0.5.0'),
])
def test_default_round_trip(tmp_path, monkeypatch, kernel, integrated, version):
    monkeypatch.setattr(spline, 'DEFAULT_SPLINE', (kernel, integrated, 2))
    src = vectorial()
    model = src.to_spline(**GRID)
    assert (model.kernel, model.integrated, model.S) == (kernel, integrated, 2)
    # Loading ignores the current builder policy.
    monkeypatch.setattr(spline, 'DEFAULT_SPLINE', ('bspline', True, 4))
    for ext in ('json', 'zip'):
        path = tmp_path / ('model.' + ext)
        meta = (model.save(path, source=src) if ext == 'json' else
                psfzip.save_zip(path, model, source=src, zstack_planes=2))
        assert meta['version'] == ('0.1.0' if ext == 'json' and version == '0.2.0' else version)
        loaded = PSF.load(path, device='cpu')
        assert (loaded.kernel, loaded.integrated, loaded.S, loaded.backend) == (kernel, integrated, 2, model.backend)
        assert torch.equal(loaded.knots, model.knots)
        equal_render(loaded, model)
        if version == '0.5.0':
            block = meta if ext == 'json' else meta['channels'][0]['model']
            assert block['cubic_spline']['kernel'] == kernel
            assert block['cubic_spline']['integrated'] == integrated
            with monkeypatch.context() as old:
                old.setattr(psfzip if ext == 'zip' else psf_module, 'READ_VERSIONS', ((0, 2), (0, 3), (0, 4)))
                with pytest.raises(ValueError, match=r'version 0\.5\.0: this reader supports'):
                    PSF.load(path)


@pytest.mark.parametrize('kernel,integrated', NEW + [('catmull', False)])
def test_angle_component_tables(tmp_path, kernel, integrated):
    models = [vectorial(), vectorial().set_zernike({6: .03})]
    model = spline.to_spline(models, kernel=kernel, integrated=integrated, components='dipole', oversample=2, **GRID)
    path = tmp_path / 'angles.zip'
    meta = psfzip.save_zip(path, model, zstack_planes=2)
    assert meta['version'] == '0.5.0'
    key = 'dipole_coefs_angle' if integrated else 'dipole_knots_angle'
    assert set(meta['channels'][0]['arrays']) == {key}
    for angle in (None, 0, 1):
        loaded = PSF.load(path, device='cpu', angle=angle)
        equal_render(loaded, model.for_angle(angle))
        equal_render(loaded, model.for_angle(angle), dipM=torch.diag(torch.tensor([.2, .5, .3])))


@pytest.mark.parametrize('kernel,integrated', NEW)
def test_phase_and_field_grids(tmp_path, monkeypatch, kernel, integrated):
    # Rebuild evaluates the optical model on the available device. Compare within one arithmetic stack.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    optics = FourPiPSF(5, npupil=16, oversample=2, pol=(0, 1), device='cpu', z_range=(0., .04))
    grid = dict(kernel=kernel, integrated=integrated, oversample=2, **GRID)
    model = optics.to_spline(**grid)
    for storage in ('json', 'float32', 'rebuild'):
        path = tmp_path / ('phase.json' if storage == 'json' else 'phase.zip')
        meta = (model.save(path) if storage == 'json' else
                psfzip.save_zip(path, model, zstack_planes=2, spline_knots=storage))
        assert meta['version'] == '0.5.0'
        loaded = PSF.load(path, device='cpu')
        assert (loaded.spline.kernel, loaded.spline.integrated) == (kernel, integrated)
        assert torch.equal(loaded.knots, model.knots)
        equal_render(loaded, model, phi=.41)

    field = FieldZernikes([6], 1, norm=(10., 10., 10., 10.), coef=[[.03, .05]], device='cpu')
    tiles = FieldPSF(vectorial(), field).to_tiled(tile=10., blend='bilinear', **grid)
    path = tmp_path / 'field.zip'
    assert psfzip.save_zip(path, tiles, zstack_planes=2)['version'] == '0.5.0'
    loaded = PSF.load(path, device='cpu')
    equal_render(loaded, tiles, xy=torch.tensor([[5., 5.], [12., 9.]]))


def test_source_and_channel_maximum(tmp_path):
    src = vectorial('exact')
    table = vectorial().to_spline(**GRID)
    assert table.save(tmp_path / 'model.json', source=src)['version'] == '0.3.0'
    new = vectorial().to_spline(kernel='bspline', integrated=False, **GRID)
    for models in ([table, new], [new, table]):
        assert ChannelsPSF(models).format_version() == (0, 5)
        assert psfzip.save_zip(tmp_path / 'channels.zip', models, zstack_planes=2)['version'] == '0.5.0'

    fourpi = FourPiPSF(5, npupil=16, pol=(0, 1), device='cpu', z_range=(0., .04))
    phase = fourpi.to_spline(kind='cubic', oversample=2, **GRID)
    fourpi.pixel_integration = 'exact'
    for explicit in (fourpi, None):
        if explicit is None:
            phase.source_state = fourpi.state()
        assert phase.format_version(explicit) == (0, 4)
        assert phase.save(tmp_path / 'phase.json', source=explicit)['version'] == '0.4.0'
        assert psfzip.save_zip(tmp_path / 'phase.zip', phase, source=explicit, zstack_planes=2)['version'] == '0.4.0'

    # An explicit box source replaces the embedded spectral state in the file.
    fourpi.pixel_integration = 'midpoint'
    assert psfzip.save_zip(tmp_path / 'phase.zip', phase, source=fourpi, zstack_planes=2)['version'] == '0.2.0'


@pytest.mark.parametrize('integrated,blend,version', [
    (False, 'nearest', '0.2.0'), (False, 'bilinear', '0.4.0'),
    (True, 'nearest', '0.4.0'), (True, 'bilinear', '0.4.0'),
])
def test_explicit_field_metadata(tmp_path, integrated, blend, version):
    src = vectorial()
    field = FieldPSF(src, FieldZernikes([6], 1, norm=(10., 10., 10., 10.), device='cpu'))
    grid = src.to_spline(integrated=integrated, oversample=2, **GRID)
    meta = psfzip.save_zip(tmp_path / 'tiles.zip', [src, grid], source=[None, src], zstack_planes=2,
                           channels=[{}, dict(field=field.field_block())],
                           provenance=dict(field_tiles=[None, dict(tile_px=10., blend=blend)]))
    assert meta['version'] == version


@pytest.mark.parametrize('mode', ['midpoint', 'exact', 'gauss_legendre', 'point'])
def test_vectorial_versions(mode):
    model = vectorial(mode)
    assert model.format_version() == (0, 2 if mode == 'midpoint' else 3)
    assert model.json_format_version() == (0, 1 if mode == 'midpoint' else 3)


def test_channel_extras_cannot_replace_versioned_model(tmp_path):
    with pytest.raises(ValueError, match='conflicts with the serialized PSF'):
        psfzip.save_zip(tmp_path / 'channel.zip', vectorial(), channels=[dict(model=dict(pixel_integration='point'))])


def test_old_single_model_and_future_minor(tmp_path):
    model = vectorial('exact').to_spline(kernel='bspline', integrated=True, oversample=2, **GRID)
    path = tmp_path / 'old.json'
    old = model.save(path)
    old['version'] = '0.1.0'  # All single-model files through stage 3 used this header.
    path.write_text(json.dumps(old))
    equal_render(PSF.load(path, device='cpu'), model)
    meta = dict(format='torch-psf', version='0.7.0', arrays='absent.npz')
    with pytest.raises(ValueError, match=r'psf.json version 0\.7\.0'):
        PSF.load(meta)
    path = tmp_path / 'future.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('psf.json', json.dumps(meta))
        z.writestr('future.npy', b'unknown encoding')
    with pytest.raises(ValueError, match=r'psf.zip version 0\.7\.0'):
        psfzip.PSFZip.read(path)
