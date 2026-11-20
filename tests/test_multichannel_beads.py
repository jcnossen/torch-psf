"""Multi-channel bead stacks (torch_psf.estimation.multichannel_beads, MultiChannelBeadEstimator): the data structure
(beads merged across channels, clusters usable per channel, field stops), registration, and the joint fit recovering a
known mapping, per-channel aberrations, field polynomials and the channel focus (GPU)."""
import numpy as np
import pytest
import torch
from conftest import needs_gpu
from torch_psf import VectorialPSF, FieldZernikes, MultiChannelBeadEstimator
from torch_psf.mapping import ChannelMap
from torch_psf.estimation.dense_beads import dense_bead_layout
from torch_psf.estimation.multichannel_beads import (simulate_multichannel_beads, build_multichannel_beads,
                                                     fit_multichannel_beads, detect_channels, register_channels,
                                                     channel_stop_mask, multichannel_nll)

H = W = 128
NORM = (W / 2, H / 2, W / 2, H / 2)
ZID = (5, 6, 7, 8, 11)
A = np.array([[np.cos(np.deg2rad(1.5)) * 1.004, -np.sin(np.deg2rad(1.5)), 3.2],
              [np.sin(np.deg2rad(1.5)), np.cos(np.deg2rad(1.5)) * 0.998, -2.7]])


def make_setup(dev, npupil=32, nk=21, sizes=((1, 1, 1, 2, 2, 1, 1, 2, 1, 1), (1, 1, 2, 1, 1, 1, 2, 1)), photons=20000.0):
    maps = [ChannelMap.identity_map(0, NORM), ChannelMap.from_affine(A, norm=NORM)]

    def vec(z):
        v = VectorialPSF(31, pixel_um=0.1, NA=1.4, wavelength_um=0.67, zernike_indices=ZID, npupil=npupil,
                         oversample=1, device=dev)
        return v.set_zernike(z)
    vecs = [vec({5: -0.02, 6: 0.08, 7: -0.03, 11: 0.01}), vec({5: 0.03, 6: 0.06, 8: 0.02, 11: -0.02})]
    fields = [FieldZernikes([5, 6], 1, norm=NORM, coef=[[0.03, 0.0], [0.0, -0.03]], device=dev),
              FieldZernikes([5, 6], 1, norm=NORM, coef=[[-0.02, 0.0], [0.0, 0.04]], device=dev)]
    focus = torch.linspace(-0.6, 0.6, nk)
    stacks = [[], []]
    for s, sz in enumerate(sizes):
        xy, _ = dense_bead_layout(H, W, list(sz), sep_px=(8, 12), margin=16, gap_px=24, seed=3 + s)
        sim = simulate_multichannel_beads(vecs, maps, focus, xy, (H, W), fields=fields, photons=photons,
                                          ratio=[1.0, 0.8], bg=20.0, channel_focus=[0.0, 0.08], render_size=41, seed=s)
        for c in range(2):
            stacks[c].append(sim[c]['d'])
    return maps, vecs, fields, focus, stacks


def test_data_registration_and_masks():
    maps, vecs, fields, focus, stacks = make_setup('cpu', npupil=16, nk=5)
    dets = detect_channels(stacks)
    reg, info = register_channels(dets, [(H, W)] * 2)
    assert info[1]['flip'] == 'none' and info[1]['n_pairs'] >= 20
    g = np.stack(np.meshgrid(np.linspace(0, 127, 5), np.linspace(0, 127, 5)), -1).reshape(-1, 2)
    assert np.abs(reg[1].from_ref(g) - maps[1].from_ref(g)).max() < 1.0
    # a field stop in channel 1 that cuts off its left part
    mask = np.ones((H, W), bool)
    mask[:, :40] = False
    data = build_multichannel_beads(stacks, reg, 15, masks=[None, mask], dets=dets, render_pad=3)
    s = data.summary()
    assert s['clusters_per_channel'][0] == data.n_clusters and s['clusters_per_channel'][1] < data.n_clusters
    assert all(w[1] == 'field_stop' for w, u in zip(data.why, data.usable) if not u[1])
    for c in range(2):          # every channel bead is the global bead at its mapped position
        d = data.channels[c]
        xy = reg[c].from_ref(data.xy_ref[data.gid[c].numpy()])
        assert np.abs(d.xy_image.numpy() - xy).max() < 1e-4
    sub = data.subset(np.nonzero(data.cl_stack == 1)[0])
    assert sub.n_clusters == int((data.cl_stack == 1).sum()) and (sub.stack == 1).all()
    one = data.channel(1)
    assert one.n_channels == 1 and one.n_clusters == s['clusters_per_channel'][1]
    # field stop of a channel: a bright disc with beads on a dark frame
    img = np.full((5, 80, 90), 1.0, np.float32)
    yy, xx = np.mgrid[:80, :90]
    img[:, (yy - 40) ** 2 + (xx - 45) ** 2 < 30 ** 2] += 20 + 0.1 * xx[(yy - 40) ** 2 + (xx - 45) ** 2 < 30 ** 2]
    img[2, 40:43, 45:48] += 500
    m = channel_stop_mask(img, margin=0)
    disc = (yy - 40) ** 2 + (xx - 45) ** 2 < 30 ** 2
    assert (m == disc).mean() > 0.97
    assert channel_stop_mask(np.full((3, 50, 50), 5.0, np.float32)).all()


@needs_gpu
def test_joint_fit_recovers_mapping_aberrations_fields():
    dev = 'cuda'
    maps, vecs, fields, focus, stacks = make_setup(dev)
    dets = detect_channels(stacks)
    reg, _ = register_channels(dets, [(H, W)] * 2)
    data = build_multichannel_beads(stacks, reg, 17, dets=dets, render_pad=5)
    train = data.cl_stack == 0
    start = [VectorialPSF(31, pixel_um=0.1, NA=1.4, wavelength_um=0.67, zernike_indices=ZID, npupil=32, oversample=1,
                          device=dev) for _ in range(2)]
    fz = [FieldZernikes.from_positions([5, 6], 1, data.channels[c].xy_image.cpu().numpy(), device=dev)
          for c in range(2)]
    r = fit_multichannel_beads(data, focus.to(dev), start, reg, iters=150, lr=5e-3, fit=('zcoef', 'field'),
                               fields=fz, train=train, warmup=20, heldout_iters=100)
    g = np.stack(np.meshgrid(np.linspace(0, 127, 9), np.linspace(0, 127, 9)), -1).reshape(-1, 2)
    err_reg = np.abs(reg[1].from_ref(g) - maps[1].from_ref(g)).max()
    err_fit = np.abs(r['maps'][1].from_ref(g) - maps[1].from_ref(g)).max()
    assert err_fit < 0.05 and err_fit < 0.25 * err_reg, (err_fit, err_reg)
    gt = torch.tensor(np.stack(np.meshgrid(np.linspace(20, 108, 5), np.linspace(20, 108, 5)), -1).reshape(-1, 2),
                      dtype=torch.float32, device=dev)
    for c in range(2):
        zt = vecs[c].zcoef.detach()[None] + fields[c].offsets(gt, list(ZID)).detach()
        ze = r['psfs'][c].zcoef.detach()[None] + r['fields'][c].offsets(gt, list(ZID)).detach()
        assert float((zt - ze).abs().max()) < 0.004, (c, (zt - ze).abs().max())
        assert r['report']['channels'][c]['heldout_deviance_per_px'] < 1.03
    assert abs(float(r['channel_focus'][1]) - 0.08) < 0.005
    # scoring the fitted model on the held-out clusters (shared positions, everything else fixed)
    ho = data.subset(np.nonzero(~train)[0])
    s = multichannel_nll(ho, focus.to(dev), [r['psfs'][c] for c in range(2)], r['maps'], iters=100,
                         channel_focus=r['channel_focus'], fields=r['fields'])
    assert all(ch['deviance_per_px'] < 1.03 for ch in s['report']['channels'])


@needs_gpu
def test_estimator_two_channels_zip(tmp_path):
    """MultiChannelBeadEstimator on side-by-side frames: registration, joint fit, psf.zip with both channels"""
    import warnings
    from torch_psf.psfzip import PSFZip
    dev = 'cuda'
    maps, vecs, fields, focus, stacks = make_setup(dev, nk=11)
    frames = [torch.cat([stacks[0][s], stacks[1][s]], 2).cpu().numpy() for s in range(2)]
    est = MultiChannelBeadEstimator(regions=[(0, 0, W, H), (W, 0, 2 * W, H)], names=['a', 'b'],
                                    optics=dict(pixel_um=0.1, NA=1.4, wavelength_um=0.67), zernike_indices=ZID,
                                    npupil=32, oversample_pupil=1, patch=17, render_pad=5, holdout=1, iters=60,
                                    warmup=10, lr=5e-3, field=dict(modes=[5, 6], order=1), field_stop=False,
                                    device=dev)
    psf = est.estimate(frames, dz_um=0.12)
    assert est.report['registration'][1]['flip'] == 'none' and psf.mapped
    assert abs(est.report['intensity_ratio'][1] - 0.8) < 0.05
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        meta = est.save_zip(tmp_path / 'mc.zip', zstack_planes=3)
    assert meta['channels'][1]['crop']['x0'] == W and meta['channels'][1]['mapping']['type'] == 'affine'
    back = PSFZip.read(tmp_path / 'mc.zip', device=dev).multichannel(field=True, mapping=True)
    cor = torch.tensor([[[30.0, 40.0], [33.0, 38.0]]], device=dev)
    pos = torch.tensor([[0.3, -0.2, 0.0]], device=dev)
    with torch.no_grad():
        assert torch.allclose(psf.render_geometry(pos, corners=cor), back.render_geometry(pos, corners=cor), atol=1e-6)
