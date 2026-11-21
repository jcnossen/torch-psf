"""Model extensions (torch_psf.extensions, format 0.7): entries round-trip through psf.zip and psf.json with the
loader registered, a missing required extension is an error and an optional one a warning, a newer entry version
than the installed loader is refused, entry points are discovered, files without extensions keep their version.
FourPiPSF.channel_field_weights / detection: identity weights equal the default channels, both channel paths agree,
the detection model is stored through its extension (state, psf.zip, the source of a spline, rebuilt knots), and a
spline with stored knots loads without the extension."""
import warnings

import pytest
import torch

from torch_psf import FourPiPSF, GaussianPSF, PSF
from torch_psf import extensions as ext
from torch_psf.io import save_zip
from torch_psf.psfzip import read_zip
from torch_psf.spline import to_spline

POS = torch.tensor([[0.2, -0.3, 0.1], [-0.5, 0.4, -0.2]])


# ------------------------------------------------------------------ a toy extension of a built-in model
class TaggedGaussian(GaussianPSF):
    tag, required, version = 'blue', True, 1

    def file_extensions(self):
        return [ext.entry('tests.tag', {'tag': self.tag}, version=self.version, required=self.required)]


def load_tag(model, state, version):
    model.tag = state['tag']
    return model


@pytest.fixture
def tag_ext():
    ext.register_extension('tests.tag', load_tag, version=1)
    yield
    ext.unregister_extension('tests.tag')


def gaussian(cls=TaggedGaussian):
    return cls(9, 1.3, device='cpu', pixel_um=0.1)


def test_zip_and_json_roundtrip(tmp_path, tag_ext):
    meta = save_zip(tmp_path / 'g.psf.zip', gaussian())
    assert meta['version'] == '0.7.0'
    assert read_zip(tmp_path / 'g.psf.zip')[0]['channels'][0]['model']['extensions'][0]['name'] == 'tests.tag'
    m = PSF.load(tmp_path / 'g.psf.zip', device='cpu')
    assert m.tag == 'blue'
    torch.testing.assert_close(m.render(POS), gaussian().render(POS))
    d = gaussian().save(tmp_path / 'g.json')
    assert d['version'] == '0.7.0' and PSF.load(tmp_path / 'g.json', device='cpu').tag == 'blue'


def test_no_extensions_keep_version(tmp_path):
    meta = save_zip(tmp_path / 'plain.psf.zip', gaussian(GaussianPSF))
    assert meta['version'] != '0.7.0' and 'extensions' not in meta['channels'][0]['model']


def test_missing_required_is_an_error(tmp_path, tag_ext):
    save_zip(tmp_path / 'g.psf.zip', gaussian())
    ext.unregister_extension('tests.tag')
    with pytest.raises(ext.ExtensionError, match="'tests.tag'"):
        PSF.load(tmp_path / 'g.psf.zip', device='cpu')


def test_missing_optional_warns(tmp_path, tag_ext):
    g = gaussian()
    g.required = False
    save_zip(tmp_path / 'g.psf.zip', g)
    ext.unregister_extension('tests.tag')
    with pytest.warns(UserWarning, match='optional extension'):
        m = PSF.load(tmp_path / 'g.psf.zip', device='cpu')
    assert type(m) is GaussianPSF and not hasattr(m, 'tag')


def test_newer_entry_version_is_refused(tmp_path, tag_ext):
    g = gaussian()
    g.version = 2
    save_zip(tmp_path / 'g.psf.zip', g)
    with pytest.raises(ext.ExtensionError, match='version 2'):
        PSF.load(tmp_path / 'g.psf.zip', device='cpu')


def test_entry_point_discovery(tmp_path, monkeypatch, tag_ext):
    save_zip(tmp_path / 'g.psf.zip', gaussian())
    ext.unregister_extension('tests.tag')

    class EP:
        name = 'tests.tag'

        def load(self):
            return load_tag

    import importlib.metadata as md
    monkeypatch.setattr(md, 'entry_points', lambda group=None: [EP()] if group == ext.ENTRY_POINT_GROUP else [])
    assert PSF.load(tmp_path / 'g.psf.zip', device='cpu').tag == 'blue'
    assert ext.registered_extensions()['tests.tag'] == 1


def test_names_are_namespaced():
    with pytest.raises(ValueError):
        ext.register_extension('tag', load_tag)
    with pytest.raises(ValueError, match='reserved'):
        ext.register_extension('torch_psf.tag', load_tag)


# ------------------------------------------------------------------ the 4Pi detection hook
class LinearDetection(torch.nn.Module):
    """Any linear detection of the camera-frame fields: w [2 arm, C, 2] complex (as real and imaginary parts)."""

    def __init__(self, w):
        super().__init__()
        w = torch.as_tensor(w, dtype=torch.complex64)
        self.re, self.im = torch.nn.Parameter(w.real.clone()), torch.nn.Parameter(w.imag.clone())

    def weights(self, pol):
        return torch.complex(self.re, self.im)

    def extension_entry(self):
        return ext.entry('tests.linear_detection', dict(re=self.re.tolist(), im=self.im.tolist()))


def load_detection(model, state, version):
    model.detection = LinearDetection(torch.complex(torch.tensor(state['re']), torch.tensor(state['im'])))
    return model


@pytest.fixture
def det_ext():
    ext.register_extension('tests.linear_detection', load_detection)
    yield
    ext.unregister_extension('tests.linear_detection')


def fourpi():
    torch.manual_seed(0)
    p = FourPiPSF(13, device='cpu', z_range=(-0.4, 0.4))
    with torch.no_grad():
        p.top.zcoef.copy_(0.03 * torch.randn(len(p.zidx)))
        p.bot.zcoef.copy_(0.03 * torch.randn(len(p.zidx)))
    return p


def identity_weights(p):
    w = torch.zeros(2, p.n_channels, 2, dtype=torch.complex64)
    for c, q in enumerate(p.pol.tolist()):
        w[:, c, int(q)] = 1
    return w


def mixing_weights(p):
    torch.manual_seed(3)
    return identity_weights(p) + 0.3 * torch.randn(2, p.n_channels, 2, dtype=torch.complex64)


def offsets(p):
    return torch.zeros(len(POS), p.n_channels, 2)


def test_identity_detection_equals_default():
    p = fourpi()
    ref, ref_c = p.render(POS), p(POS, offsets=offsets(p))
    p.detection = LinearDetection(identity_weights(p))
    with torch.no_grad():
        torch.testing.assert_close(p.render(POS), ref, rtol=0, atol=1e-6 * float(ref.max()))
        torch.testing.assert_close(p(POS, offsets=offsets(p)), ref_c, rtol=0, atol=1e-6 * float(ref.max()))


def test_detection_channel_paths_agree_and_grads():
    p = fourpi()
    ref = p.render(POS).detach()
    p.detection = LinearDetection(mixing_weights(p))
    img = p.render(POS)
    with torch.no_grad():
        torch.testing.assert_close(p(POS, offsets=offsets(p)), img, rtol=0, atol=1e-6 * float(img.max()))
    assert float((img.detach() - ref).abs().max() / ref.max()) > 0.01
    img.sum().backward()
    assert p.detection.re.grad is not None and torch.isfinite(p.detection.re.grad).all()


def test_detection_state_roundtrip(det_ext):
    p = fourpi()
    p.detection = LinearDetection(mixing_weights(p))
    st = p.state()
    assert st['extensions'][0]['name'] == 'tests.linear_detection'
    assert not any(k.startswith('detection.') for k in st['raw'])
    assert p.format_version() == (0, 7)
    q = FourPiPSF.from_state(st, device='cpu', roisize=13, z_range=(-0.4, 0.4))
    with torch.no_grad():
        torch.testing.assert_close(q.render(POS), p.render(POS), rtol=0, atol=1e-7)
    ext.unregister_extension('tests.linear_detection')
    with pytest.raises(ext.ExtensionError, match='FourPiPSF state'):
        FourPiPSF.from_state(st, device='cpu')


def test_detection_without_entry_is_not_stored():
    p = fourpi()

    class Bare(torch.nn.Module):
        def weights(self, pol):
            return identity_weights(p)

    p.detection = Bare()
    with pytest.raises(ValueError, match='extension_entry'):
        p.state()


def test_detection_zip(tmp_path, det_ext):
    p = fourpi()
    p.detection = LinearDetection(mixing_weights(p))
    meta = save_zip(tmp_path / 'd.psf.zip', p)
    assert meta['version'] == '0.7.0'
    q = PSF.load(tmp_path / 'd.psf.zip', device='cpu')
    with torch.no_grad():
        torch.testing.assert_close(q.render(POS), p.render(POS), rtol=0, atol=1e-6)
    ext.unregister_extension('tests.linear_detection')
    with pytest.raises(ext.ExtensionError):
        PSF.load(tmp_path / 'd.psf.zip', device='cpu')


def test_spline_of_detection_model(tmp_path, det_ext):
    p = fourpi()
    p.detection = LinearDetection(mixing_weights(p))
    s = to_spline(p, (-0.3, 0.3), roisize=13, oversample=2, dz_um=0.02)
    assert ext.used(s.source_state) and s.format_version() >= (0, 7)
    pos = POS * torch.tensor([1.0, 1.0, 0.5])
    with torch.no_grad():
        ref = p.render(pos)
        assert float((s.render(pos) - ref).abs().max() / ref.max()) < 5e-3
        save_zip(tmp_path / 'knots.psf.zip', s)
        save_zip(tmp_path / 'rebuild.psf.zip', s, spline_knots='rebuild')
        assert read_zip(tmp_path / 'knots.psf.zip')[0]['version'] == '0.7.0'
        r = PSF.load(tmp_path / 'rebuild.psf.zip', device='cpu')              # knots tabulated from the source
        torch.testing.assert_close(r.render(pos), s.render(pos), rtol=0, atol=1e-5 * float(ref.max()))
        ext.unregister_extension('tests.linear_detection')
        k = PSF.load(tmp_path / 'knots.psf.zip', device='cpu')                 # stored knots: no extension needed
        torch.testing.assert_close(k.render(pos), s.render(pos), rtol=0, atol=1e-6)
        with pytest.raises(ext.ExtensionError):
            PSF.load(tmp_path / 'rebuild.psf.zip', device='cpu')               # rebuilding needs the source model


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_detection_loaded_to_the_read_device(tmp_path, det_ext, device):
    """The loader creates its module on the CPU: reading to any device (and the knot rebuild, which tabulates the
    source on the GPU when there is one) must still render there."""
    p = fourpi()
    p.detection = LinearDetection(mixing_weights(p))
    s = to_spline(p, (-0.3, 0.3), roisize=13, oversample=2, dz_um=0.02)
    pos = (POS * torch.tensor([1.0, 1.0, 0.5])).to(device)
    with torch.no_grad():
        save_zip(tmp_path / 'rebuild.psf.zip', s, spline_knots='rebuild')
        save_zip(tmp_path / 'model.psf.zip', p)
        ref = s.render(pos.cpu())
        r = PSF.load(tmp_path / 'rebuild.psf.zip', device=device)
        torch.testing.assert_close(r.render(pos).cpu(), ref, rtol=0, atol=1e-5 * float(ref.max()))
        q = PSF.load(tmp_path / 'model.psf.zip', device=device)
        assert q.detection.re.device.type == torch.device(device).type
        torch.testing.assert_close(q.render(pos).cpu(), p.render(pos.cpu()), rtol=0, atol=1e-5 * float(ref.max()))
        q2 = FourPiPSF.from_state(p.state(), device=device, roisize=13, z_range=(-0.4, 0.4))
        torch.testing.assert_close(q2.render(pos).cpu(), p.render(pos.cpu()), rtol=0, atol=1e-5 * float(ref.max()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_detection_module_on_another_device():
    p = fourpi().to('cuda')
    ref = p.render(POS.cuda()).detach()
    p.detection = LinearDetection(identity_weights(p).cpu())          # attached by hand, still on the CPU
    with torch.no_grad():
        torch.testing.assert_close(p.render(POS.cuda()), ref, rtol=0, atol=1e-6 * float(ref.max()))
