import pytest
import torch

from torch_psf import ChannelsPSF, FourPiPSF, IntegratedSplinePSF, PSF, VectorialPSF, io, psfzip


def models(kind, mode='box'):
    kw = dict(roisize=7, npupil=16, oversample=3, device='cpu', pixel_integration=mode, z_range=(0., .04))
    offsets = [[.11, -.19], [-.22, .16]]
    if kind == 'fourpi':
        return FourPiPSF(**kw, pol=(0, 1), channel_offsets=offsets, d_bot=(.13, -.2, .02))
    arms = [VectorialPSF(**kw) for _ in offsets]
    for v, diat in zip(arms, [-1., 1.]):
        v.diat = diat
    return ChannelsPSF(arms, channel_offsets=offsets, intensity_ratio=[.8, 1.2])


def emitters():
    pos = torch.tensor([[.17, -.23, .013], [-.32, .24, .027]])
    a = torch.randn(2, 3, 3, generator=torch.Generator().manual_seed(9))
    M = a @ a.transpose(-1, -2)
    return pos, M / M.diagonal(dim1=-1, dim2=-2).sum(-1)[:, None, None]


def combine(basis, M):
    weights = torch.stack([M[:, i, j] for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]], 1)
    return (weights[:, None, :, None, None] * basis).sum(2)


def check_image(actual, ref):
    assert ((actual - ref).abs().amax((-1, -2)) <= 1e-5 * ref.abs().amax((-1, -2))).all()


@pytest.mark.parametrize('kind,mode', [('fourpi', 'box'), ('fourpi', 'spectral'), ('channels', 'spectral'),
                                      ('channels', 'gauss_legendre')])
def test_full_dipoles(kind, mode):
    model = models(kind, mode)
    pos, M = emitters()
    phase = torch.tensor([.75, -1.3])
    kw = dict(phi=phase) if kind == 'fourpi' else {}
    actual = model.render(pos, dipM=M, **kw)
    individual = [model.render(p[None], dipM=m, **{k: v[i:i+1] for k, v in kw.items()})
                  for i, (p, m) in enumerate(zip(pos, M))]
    check_image(actual, torch.cat(individual))
    check_image(model.render(pos, dipM=torch.eye(3)/3, **kw), model.render(pos, **kw))
    check_image(combine(model.dipole_basis(pos, **kw), M), actual)
    pc = model.channel_positions(pos)
    ex = phase[:, None] if kind == 'fourpi' else None
    check_image(combine(model.dipole_basis(pc, **kw), M), model.render_channels(pc, ex, dipM=M))
    forward_kw = dict(phi_extra=phase) if kind == 'fourpi' else {}
    check_image(model.forward(pos, dipM=M, **forward_kw), actual)
    if kind == 'fourpi':
        check_image(model(pos, M=M, **forward_kw), actual)
    M.requires_grad_()
    probe = torch.randn_like(actual)
    loss = lambda m: (probe * model.render(pos, dipM=m, **kw)).sum()
    grad, = torch.autograd.grad(loss(M), M)
    direction = torch.randn_like(M)
    direction = (direction + direction.transpose(-1, -2)) / 2
    eps = .005
    numeric = (loss(M + eps*direction) - loss(M - eps*direction)) / (2*eps)
    torch.testing.assert_close((grad*direction).sum(), numeric, rtol=5e-4, atol=1e-5)


@pytest.mark.parametrize('kind', ['fourpi', 'channels'])
@pytest.mark.parametrize('integrated', [False, True])
def test_spline_dipoles_and_files(kind, integrated, tmp_path, monkeypatch):
    source = models(kind, 'spectral' if integrated else 'box')
    grid = dict(oversample=3, dz_um=.01, xy_limit=.75, dipole_components=True, zpad=3)
    pos, M = emitters()
    if kind == 'fourpi':
        source.dipM = M[0]
    if kind == 'channels' and integrated:
        tables = [IntegratedSplinePSF.from_vectorial(v, v.z_range, **grid) for v in source.models]
        model = ChannelsPSF(tables, source.channel_offsets, source.intensity_ratio)
    else:
        model = source.to_spline(**grid)
    kw = dict(phi=torch.tensor([1.2, -.7])) if kind == 'fourpi' else {}
    val = model.render(pos, dipM=M, **kw)
    check_image(combine(model.dipole_basis(pos, **kw), M), val)
    check_image(model.forward(pos, dipM=M, **kw), val)
    jac = model.render(pos, grad=True, dipM=M, **kw)
    for i in range(len(pos)):
        kw_i = {k: v[i:i+1] for k, v in kw.items()}
        single = model.render(pos[i:i+1], grad=True, dipM=M[i], **kw_i)
        for all_emitters, one in zip(jac, single):
            check_image(all_emitters[i:i+1], one)
    assert (val - source.render(pos, dipM=M, **kw)).abs().max() < .005 * val.max()
    path = tmp_path / 'components.zip'
    if kind == 'channels':
        entries = [dict(intensity_ratio=float(r)) for r in model.intensity_ratio]
        meta = io.save_zip(path, model.models, channels=entries, zstack_planes=3)
    else:
        meta = io.save_zip(path, model, zstack_planes=3)
    assert meta['version'] == '0.4.0'
    loaded = psfzip.PSFZip.read(path, device='cpu').multichannel()
    # Independent channel offsets are acquisition geometry and are supplied separately.
    if kind == 'channels':
        loaded.set_channel_offsets(source.channel_offsets)
    else:
        fixed = source.to_spline(**dict(grid, dipole_components=False)).render(pos, **kw)
        for component_model in (model, loaded):
            torch.testing.assert_close(component_model.render(pos, **kw), fixed, rtol=0, atol=1e-6 * fixed.max().item())
    check_image(loaded.render(pos, dipM=M, **kw), val)
    monkeypatch.setattr(psfzip, 'READ_VERSIONS', ((0, 2), (0, 3)))
    with pytest.raises(ValueError, match='version 0.4.0'):
        PSF.load(path)


def test_integrated_basis_per_angle(tmp_path):
    arms = models('channels').models
    pos, M = emitters()
    arms[0].dipM, arms[1].dipM = M
    arms[1].blur = torch.tensor([.2, .1, .02])
    table = IntegratedSplinePSF.from_vectorial(arms, arms[0].z_range, oversample=2, dz_um=.02,
                                               xy_limit=1, dipole_components=True)
    path = tmp_path / 'angles.zip'
    io.save_zip(path, table, zstack_planes=3)
    by_angle = []
    defaults = []
    for angle in range(2):
        ref = table.for_angle(angle)
        loaded = PSF.load(path, angle=angle, device='cpu')
        check_image(loaded.render(pos, dipM=M), ref.render(pos, dipM=M))
        fixed = IntegratedSplinePSF.from_vectorial(arms[angle], arms[angle].z_range, oversample=2, dz_um=.02,
                                                   xy_limit=1)
        defaults.append(fixed.render(pos))
        torch.testing.assert_close(loaded.render(pos), defaults[-1], rtol=0, atol=1e-6 * defaults[-1].max().item())
        by_angle.append(ref.render(pos, dipM=M))
    check_image(table.render(pos, dipM=M), torch.stack(by_angle).mean(0))
    mean = torch.stack(defaults).mean(0)
    torch.testing.assert_close(PSF.load(path, device='cpu').render(pos), mean,
                               rtol=0, atol=1e-6 * mean.max().item())


@pytest.mark.parametrize('offsets', [False, True])
def test_fourpi_basis_evaluates_each_pupil_once(monkeypatch, offsets):
    model = models('fourpi')
    if not offsets:
        model.set_channel_offsets(torch.zeros_like(model.channel_offsets))
    calls = []
    original = VectorialPSF._emitter_pupil

    def count(arm, *args):
        calls.append(arm)
        return original(arm, *args)

    monkeypatch.setattr(VectorialPSF, '_emitter_pupil', count)
    pos, M = emitters()
    basis = model.dipole_basis(pos, phi=.86)
    assert calls == [model.top, model.bot]
    check_image(combine(basis, M), model.render(pos, phi=.86, dipM=M))


def test_channels_use_native_basis_when_available(monkeypatch):
    model = models('channels')
    class Layered(VectorialPSF):
        type_name = 'test_layered'
    model.models[0] = Layered(7, npupil=16, device='cpu')
    pos, _ = emitters()
    expected = model.dipole_basis(pos)
    calls = []
    for c, channel in enumerate(model.models):
        def native(p, focus, c=c):
            calls.append((c, focus))
            return expected[:, c] / model.intensity_ratio[c]
        monkeypatch.setattr(channel, 'dipole_basis', native, raising=False)
    check_image(model.dipole_basis(pos).flatten(1, 2), expected.flatten(1, 2))
    assert calls == [(c, v.focus_um) for c, v in enumerate(model.models)]


def test_fourpi_dipole_rebuild(tmp_path):
    source = models('fourpi')
    source.dipM = torch.diag(torch.tensor([.2, .7, .1]))
    model = source.to_spline(oversample=2, dz_um=.02, xy_limit=1, dipole_components=True)
    path = tmp_path / 'rebuild.zip'
    io.save_zip(path, model, spline_knots='rebuild', zstack_planes=3)
    loaded = PSF.load(path, device='cpu')
    pos, M = emitters()
    for tensor in (None, M):
        check_image(loaded.render(pos, phi=.3, dipM=tensor), model.render(pos, phi=.3, dipM=tensor))


@pytest.mark.parametrize('kind', ['fourpi', 'channels'])
def test_default_box_spline_keeps_old_layout(kind, tmp_path, monkeypatch):
    source = models(kind)
    model = source.to_spline(oversample=2, dz_um=.02, xy_limit=1)
    path = tmp_path / 'fixed.zip'
    meta = io.save_zip(path, model if kind == 'fourpi' else model.models, zstack_planes=3)
    assert meta['version'] == '0.2.0'
    for channel in meta['channels']:
        assert 'dipole_components' not in channel['model']['spline']
        expected = {'inc', 'cr', 'ci'} if kind == 'fourpi' else {'knots'}
        assert set(channel['arrays']) == expected
    monkeypatch.setattr(psfzip, 'READ_VERSIONS', ((0, 2),))
    loaded = psfzip.PSFZip.read(path, device='cpu').multichannel()
    pos, M = emitters()
    with pytest.raises(ValueError, match='dipole_components=True'):
        loaded.render(pos, dipM=M)


@pytest.mark.parametrize('integrated', [False, True])
def test_component_build_and_default_response(tmp_path, monkeypatch, integrated):
    from torch_psf import CubicSplinePSF, pixint

    v = models('channels').models[0]
    pos, tensors = emitters()
    v.dipM = tensors[0]
    grid = dict(oversample=2, dz_um=.02, xy_limit=1)
    def build(components):
        if integrated:
            return IntegratedSplinePSF.from_vectorial(v, v.z_range, dipole_components=components, **grid)
        return v.to_spline(dipole_components=components, **grid)
    reference = build(False).render(pos)
    owner, name = (pixint, 'point_knots') if integrated else (CubicSplinePSF, 'knots_from_vectorial')
    original = getattr(owner, name)
    calls = []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, name, count)
    model = build(True)
    assert len(calls) == 6
    path = tmp_path / 'components.zip'
    channel = io.save_zip(path, model, zstack_planes=3)['channels'][0]
    assert set(channel['arrays']) == {'dipole_coefs' if integrated else 'dipole_knots'}
    for m in (model, PSF.load(path, device='cpu').resized(v.roisize)):
        torch.testing.assert_close(m.render(pos), reference, rtol=0, atol=1e-6 * reference.max().item())
        assert m._knots is None
    # Pre-0.4 component files had an explicit default (possibly calibrated independently of the basis).
    data, arrays = model._file_dict()
    del data[model.type_name]['default_weights']
    arrays['coefs' if integrated else 'knots'] = (.7 * model.knots).numpy()
    previous = type(model)._from_file(data, arrays, device='cpu')
    torch.testing.assert_close(previous.render(pos), .7 * reference, rtol=0, atol=1e-6 * reference.max().item())


def test_highest_required_version_across_channels(tmp_path):
    v = models('channels', 'spectral').models[0]
    component = v.to_spline(oversample=2, dz_um=.02, dipole_components=True)
    for channels in ([v, component], [component, v]):
        assert io.save_zip(tmp_path / 'mixed.zip', channels, zstack_planes=3)['version'] == '0.4.0'


def test_phase_and_integrated_component_dtype():
    source = models('fourpi')
    pos, M = emitters()
    phase = torch.tensor([[.31], [.72]], dtype=torch.float64)
    spline = source.to_spline(kind='integrated', oversample=2, dz_um=.02, xy_limit=1, dipole_components=True)
    spline.to('cpu', torch.float64)
    for row in spline.luts:
        for lut in row:
            assert lut._knots is None and lut.dipole_knots.dtype == torch.float64
    for model in (source, spline):
        pc = model.channel_positions(pos)
        img = model.render_channels(pc, phase, dipM=M)
        torch.testing.assert_close(img, model.render(pos, phi=phase[:, 0], dipM=M))
    assert img.dtype == torch.float64
    with pytest.raises(ValueError, match='kind'):
        source.to_spline(kind='unknown')
