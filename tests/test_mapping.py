"""torch_psf.mapping (ChannelMap, register_points, channel_rois) and the mapped geometry of MultiChannelPSF /
ChannelsPSF: translation = channel offsets, Jacobians through rotations and flips, fits with mapped ROIs, field
models per channel, the psf.zip round trip. CPU only and small (CI)."""
import warnings
import numpy as np
import pytest
import torch
from torch_psf import ChannelsPSF, AstigmaticGaussianPSF, VectorialPSF, FieldZernikes, FieldPSF, io
from torch_psf.mapping import ChannelMap, register_points, channel_rois, match_pairs
from torch_psf.psfzip import Mapping, PSFZip

DEV = 'cpu'
NORM = (237.5, 200.0, 237.5, 200.0)
POS = torch.tensor([[0.2, -0.4, 0.05], [-0.7, 0.3, -0.12], [0.0, 0.0, 0.2]])


def affine(flip=True, deg=2.0, s=1.01):
    t = np.deg2rad(deg)
    sx = -1.0 if flip else 1.0
    return np.array([[sx * np.cos(t) * s, np.sin(t), 470.0 if flip else 3.5], [np.sin(t), np.cos(t), -8.0]])


def pair():
    return ChannelsPSF([AstigmaticGaussianPSF(13, (1.3, 1.3), gamma_um=0.2, d_um=0.4, z_range=(-0.5, 0.5), device=DEV,
                                              pixel_um=0.1),
                        AstigmaticGaussianPSF(13, (1.4, 1.2), gamma_um=-0.15, d_um=0.35, z_range=(-0.5, 0.5),
                                              device=DEV, pixel_um=0.1)])


# ====================================================================================== ChannelMap
def test_channel_map_affine_poly_and_blocks():
    A = affine()
    m = ChannelMap.from_affine(A, norm=NORM)
    p = np.random.default_rng(0).uniform(0, 400, (50, 2))
    assert np.allclose(m.from_ref(p), p @ A[:, :2].T + A[:, 2], atol=1e-9)
    assert np.allclose(m.to_ref(m.from_ref(p)), p, atol=1e-9)
    J = m.jacobian(torch.as_tensor(p))
    assert np.allclose(J.detach().numpy(), np.broadcast_to(A[:, :2], (50, 2, 2)), atol=1e-12)
    assert m.flip() == 'lr' and abs(abs(m.describe()['rotation_deg']) - 2.0) < 0.05
    # psf.zip affine block (channel -> reference matrix): the numpy reader agrees, and back
    blk = m.to_block(1)
    assert blk['type'] == 'affine'
    rd = Mapping(blk, 1)
    q = m.from_ref(p)
    assert np.allclose(np.stack(rd.to_ref(q[:, 0], q[:, 1]), 1), p, atol=1e-8)
    m2 = ChannelMap.from_block(blk, 1, crop=dict(w=475, h=400))
    assert np.allclose(m2.from_ref(p), q, atol=1e-8)
    # order 3 polynomial: least squares, fwd refitted as the inverse, the poly block round trip
    r = p
    c = r @ A[:, :2].T + A[:, 2] + 1e-4 * (r - 200) ** 2
    m3 = ChannelMap.fit_points(r, c, order=3, norm=NORM)
    assert np.abs(m3.from_ref(r) - c).max() < 1e-6
    assert np.abs(m3.to_ref(c) - r).max() < 1e-6                     # Newton-refined inverse
    blk3 = m3.to_block(1, extent=(0, 0, 474, 399))
    assert blk3['type'] == 'poly' and blk3['roundtrip_px'] < 0.05
    rd3 = Mapping(blk3, 1)
    assert np.allclose(np.stack(rd3.from_ref(r[:, 0], r[:, 1]), 1), c, atol=1e-6)
    assert np.abs(np.stack(rd3.to_ref(c[:, 0], c[:, 1]), 1) - r).max() < 0.05
    m4 = ChannelMap.from_block(blk3, 1)
    assert np.allclose(m4.from_ref(r), c, atol=1e-6)
    # numerical Jacobian of the polynomial
    x = torch.as_tensor(r[:5])
    e = 1e-4
    for k in range(2):
        d = torch.zeros(2, dtype=torch.float64)
        d[k] = e
        fd = (m3.from_ref(x + d) - m3.from_ref(x - d)) / (2 * e)
        assert torch.allclose(fd, m3.jacobian(x)[:, :, k].detach(), atol=1e-7)
    # trainable subset: only the shift gets a gradient
    t = ChannelMap.translation([1.0, 2.0], norm=NORM)
    t.from_ref(torch.as_tensor(p)).sum().backward()
    assert float(t.inv.grad[1:].abs().max()) == 0 and float(t.inv.grad[0].abs().min()) > 0


def test_register_points_flip_rotation_outliers():
    A = affine(flip=True)
    m = ChannelMap.from_affine(A, norm=NORM)
    rng = np.random.default_rng(0)
    r = rng.uniform(20, 380, (80, 2))
    c = m.from_ref(r) + rng.normal(0, 0.2, (80, 2))
    c = np.concatenate([c[:70], rng.uniform(0, 400, (15, 2))])            # 10 beads missing, 15 false detections
    est, (i, j), info = register_points(r, c, (400, 475))
    assert info['flip'] == 'lr' and info['n_pairs'] >= 68 and info['rms_px'] < 0.4
    assert np.abs(est.from_ref(r) - m.from_ref(r)).max() < 0.3
    a, b = match_pairs(r[:5], r[:5] + 0.1, 1.0)
    assert list(a) == list(b) == list(range(5))


# ====================================================================================== mapped geometry
def test_translation_mapping_equals_channel_offsets():
    a = pair()
    a.set_channel_offsets([[0, 0], [0.6, -0.4]])
    b = pair().set_mappings([None, ChannelMap.translation([0.6, -0.4], norm=NORM)])
    cor = torch.tensor([[[10.0, 20.0], [10.0, 20.0]]]).expand(3, 2, 2)
    with torch.no_grad():
        for x, y in zip(a.render(POS, grad=True), b.render(POS, grad=True, corners=cor)):
            assert torch.allclose(x, y, atol=1e-7)
    with pytest.raises(ValueError, match='corners'):
        b.render(POS)


def _mapped_problem(N=120, seed=1):
    m = pair()
    mp = ChannelMap.from_affine(affine(flip=True), norm=NORM)
    m.set_mappings([None, mp])
    g = torch.Generator().manual_seed(seed)
    ref = torch.rand(N, 2, generator=g) * 360 + 20
    xc = mp.from_ref(ref.double()).float()
    h = 6
    cor = torch.stack([torch.round(ref) - h, torch.round(xc) - h], 1)
    pos = torch.cat([ref - cor[:, 0] - h, (torch.rand(N, 1, generator=g) - 0.5) * 0.6], 1)
    th = m.theta(pos, torch.full((N,), 2000.0), torch.tensor([[5.0, 8.0]]).expand(N, 2))
    return m, th, cor, ref, g


def test_mapped_jacobian_and_fit_reach_crlb():
    m, th, cor, ref, g = _mapped_problem()
    mu, J = m.mu_and_jacobian(th[:20], corners=cor[:20])
    for k in range(3):
        e = 1e-3
        tp, tm = th[:20].clone(), th[:20].clone()
        tp[:, k] += e
        tm[:, k] -= e
        with torch.no_grad():
            fd = (m.expected(tp, corners=cor[:20]) - m.expected(tm, corners=cor[:20])) / (2 * e)
        assert float((fd - J[:, k]).abs().max() / J[:, k].abs().max()) < 2e-3
    d = torch.poisson(m.expected(th, corners=cor).detach(), generator=g)
    ds = m.fit(d, corners=cor, zstarts=3)
    err = (ds['theta'] - th).std(0)
    cr = m.crlb(th, corners=cor).mean(0)
    assert ((err / cr)[:4] < 1.35).all(), err / cr
    assert float((ds['pos'][:, :2] - ref.double()).abs().median()) < 0.05      # reference-channel coordinates
    assert not bool(ds['at_limit'].any())


def test_channel_rois_and_field_channels():
    """ROIs cut around mapped positions; a FieldPSF channel evaluated at its ROI centre in channel coordinates"""
    mp = ChannelMap.from_affine(affine(flip=False, deg=1.0), norm=(32.0, 32.0, 32.0, 32.0))
    imgs = [torch.arange(2 * 64 * 64, dtype=torch.float32).reshape(2, 64, 64) for _ in range(2)]
    xy = torch.tensor([[20.3, 30.6], [40.0, 10.2]])
    rois, cor = channel_rois(imgs, torch.tensor([0, 1]), xy, [None, mp], 7)
    assert rois.shape == (2, 2, 7, 7) and cor.shape == (2, 2, 2)
    assert torch.equal(cor[0, 0], torch.tensor([17.0, 28.0]))
    assert torch.equal(rois[1, 1], imgs[1][1, int(cor[1, 1, 1]):int(cor[1, 1, 1]) + 7, int(cor[1, 1, 0]):int(cor[1, 1, 0]) + 7])
    v = VectorialPSF(9, pixel_um=0.1, NA=1.3, npupil=16, oversample=1, device=DEV, zernike_indices=(5, 6),
                     z_range=(-0.3, 0.3))
    f = FieldZernikes([6], 1, norm=(32.0, 32.0, 32.0, 32.0), coef=[[0.0, 0.05]])
    fp = FieldPSF(v, f)
    mc = ChannelsPSF([v, fp], mappings=[None, mp])
    with torch.no_grad():
        out = mc.render(POS[:2], corners=cor)
        ref = fp.render(mc.mapped_positions(POS[:2], corners=cor)[0][:, 1], xy=cor[:, 1] + 4)
    assert torch.allclose(out[:, 1], ref, atol=1e-7)


def test_psfzip_round_trip_mapped_field_channels(tmp_path):
    v0 = VectorialPSF(9, pixel_um=0.1, NA=1.3, npupil=16, oversample=1, device=DEV, zernike_indices=(5, 6, 11),
                      z_range=(-0.3, 0.3))
    v0.set_zernike({6: 0.05})
    v1 = v0.resized(9).set_zernike({5: 0.03})
    v1.focus_um = 0.07                                       # a channel focus offset lives in the channel model
    f0 = FieldZernikes([6], 1, norm=(30.0, 25.0, 30.0, 25.0), coef=[[0.0, 0.02]])
    f1 = FieldZernikes([5, 11], {5: 2, 11: 1}, norm=(31.0, 24.0, 29.0, 25.0), coef=[[0.01, -0.02, 0.0, 0.0, 0.01],
                                                                                     [0.0, 0.01, 0.0, 0.0, 0.0]])
    mp = ChannelMap.fit_points(*(lambda r: (r, r @ affine(False, 1.0)[:, :2].T + [3.0, -2.0] + 1e-4 * r ** 2))(
        np.random.default_rng(0).uniform(0, 60, (40, 2))), order=2, norm=(30.0, 25.0, 30.0, 25.0))
    mc = ChannelsPSF([FieldPSF(v0, f0), FieldPSF(v1, f1)], names=['a', 'b'], intensity_ratio=[1.0, 0.8],
                     mappings=[None, mp])
    chans = [dict(name='a', crop=dict(x0=0, y0=0, w=60, h=50)),
             dict(name='b', crop=dict(x0=60, y0=0, w=60, h=50), mapping=mp.to_block(1, extent=(0, 0, 59, 49)),
                  intensity_ratio=0.8, flip=mp.flip())]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        meta = io.save_zip(tmp_path / 'mc.zip', list(mc.models), channels=chans, zstack_planes=3)
    assert [c['field']['type'] for c in meta['channels']] == ['zernike_poly'] * 2
    assert meta['channels'][1]['mapping']['type'] == 'poly'
    pz = PSFZip.read(tmp_path / 'mc.zip', device=DEV)
    back = pz.multichannel(field=True, mapping=True)
    assert back.names == ['a', 'b'] and back.field_dependent == [True, True] and back.models[1].base.focus_um == 0.07
    cor = torch.tensor([[[10.0, 12.0], [13.0, 10.0]], [[40.0, 30.0], [44.0, 27.0]]])
    with torch.no_grad():
        a, b = mc.render(POS[:2], corners=cor), back.render(POS[:2], corners=cor)
    assert torch.allclose(a, b, atol=1e-6)
    old = pz.multichannel()                                   # default: field-constant models, no mappings
    assert not old.mapped and old.field_dependent == [False, False]
