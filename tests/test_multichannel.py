"""MultiChannelPSF / ChannelsPSF and the 4Pi models as multi-channel PSFs: reference values of the 4Pi renders, the
channel offsets, the joint Jacobian, fits, the CRLB, to_spline with offsets, state / file round trips.
CPU only and small (CI)."""
import math
import pytest
import torch
from conftest import needs_gpu
from torch_psf import (PSF, MultiChannelPSF, ChannelsPSF, FourPiPSF, FourPiSplinePSF, AstigmaticGaussianPSF,
                       GaussianPSF, VectorialPSF)

DEV = 'cpu'
POS = torch.tensor([[0.2, -0.4, 0.05], [-0.7, 0.3, -0.12], [0.0, 0.0, 0.2]])
PHI = torch.tensor([0.4, 1.1, 2.5])
OFF = [[0.0, 0.0], [0.4, -0.3], [-0.6, 0.2], [0.25, 0.5]]
RAMP = torch.linspace(0, 1, 13)[:, None] + torch.linspace(0, 2, 13)[None, :]       # pixel weights (13 x 13)


def fourpi(asym=False, mirror='y', zmode='emitter', R=13, npupil=32, offsets=None, z_range=(-0.6, 0.6)):
    kw = dict(NA=(1.45, 1.3), n_imm=(1.518, 1.406), n_med=1.33, depth_um=(0.1, 1.0),
              zernike_indices=tuple(range(4, 16))) if asym else dict(NA=1.35, n_imm=1.406, zernike_indices=tuple(range(5, 16)))
    p = FourPiPSF(R, pixel_um=0.11, wavelength_um=0.67, npupil=npupil, oversample=2, mirror=mirror, device=DEV, pixel_integration='midpoint',
                  phi=(0.3, 1.9, 3.4, 5.0), tau=(0.2, 0.3, 0.25, 0.25), a=0.9, gamma=0.8, d_bot=(0.3, -0.2, 0.02),
                  blur_px2=0.12, channel_offsets=offsets, z_range=z_range, **kw)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        p.top.zcoef.copy_(0.04 * (torch.rand(len(p.zidx), generator=g) - 0.5))
        p.bot.zcoef.copy_(0.04 * (torch.rand(len(p.zidx), generator=g) - 0.5))
    from torch_psf import ZMode
    return ZMode(p, zmode)


def gauss_pair(offsets=((0.0, 0.0), (0.6, -0.4)), ratio=(1.0, 0.8), R=13):
    """biplane-like pair of astigmatic Gaussians (different astigmatism per channel)"""
    return ChannelsPSF([AstigmaticGaussianPSF(R, (1.3, 1.3), gamma_um=0.2, d_um=0.4, z_range=(-0.5, 0.5), device=DEV,
                                              pixel_um=0.1),
                        AstigmaticGaussianPSF(R, (1.4, 1.2), gamma_um=-0.15, d_um=0.35, z_range=(-0.5, 0.5),
                                              device=DEV, pixel_um=0.1)],
                       channel_offsets=offsets, intensity_ratio=ratio)


# ====================================================================================== 4Pi reference values
def test_fourpi_reference_values():
    """a few numbers of the 4Pi renders of master 32bc538 (any platform, loose tolerance)"""
    p = fourpi(R=9, npupil=16)
    with torch.no_grad():
        s = p.render(POS, phi=PHI).sum((2, 3))
    ref = torch.tensor([[0.21375, 0.28478, 0.13419, 0.1617], [0.11184, 0.28445, 0.26256, 0.16546],
                        [0.15032, 0.28921, 0.19091, 0.13676]])             # master 32bc538
    assert torch.allclose(s, ref, atol=2e-4), s


# ====================================================================================== channel offsets
@pytest.mark.parametrize('kw', [dict(), dict(asym=True, mirror='x', zmode='stage')])
def test_fourpi_offset_equals_shifted_position(kw):
    p = fourpi(**kw)
    off = torch.tensor(OFF)
    with torch.no_grad():
        ref = torch.stack([p(POS + torch.cat([off[c], torch.zeros(1)])[None], phi=PHI)[:, c] for c in range(4)], 1)
        p.set_channel_offsets(off)
        a, b = p(POS, phi=PHI), p.render(POS, phi=PHI)
        # per-spot offsets add to the channel offsets
        po = torch.tensor([[0.1, -0.2]]).expand(3, 4, 2)
        c = p.render_geometry(POS - torch.tensor([[0.1, -0.2, 0.0]]), phi=PHI, offsets=po)
    assert torch.equal(a, b)
    assert float((a - ref).abs().max() / ref.abs().max()) < 1e-5
    assert float((c - a).abs().max() / a.abs().max()) < 1e-5


def test_spline_offsets_and_knots_override():
    p = fourpi(R=11, npupil=24, z_range=(-0.2, 0.2))
    spl = p.to_spline(11, (-0.2, 0.2), oversample=2, dz_um=0.03, zpad=2)
    off = torch.tensor(OFF)
    with torch.no_grad():
        ref = torch.stack([spl.render(POS + torch.cat([off[c], torch.zeros(1)])[None], phi=PHI)[:, c] for c in range(4)], 1)
        spl.set_channel_offsets(off)
        a = spl.render(POS, phi=PHI)
    assert float((a - ref).abs().max() / ref.abs().max()) < 1e-6
    K = spl.knots.clone().requires_grad_(True)
    v = spl.render_tables(POS, phi=PHI, knots=K)
    assert torch.allclose(v, a, atol=1e-7)
    (v ** 2).sum().backward()
    assert K.grad is not None and float(K.grad.abs().max()) > 0
    with torch.no_grad():
        assert torch.allclose(spl.render_tables(POS, phi=PHI, knots=2 * spl.knots), 2 * a, atol=1e-6)


def test_channels_psf_render_and_offsets():
    m = gauss_pair()
    assert m.n_channels == 2 and m.n_params == 4 + 2 and m.param_names == ['x', 'y', 'z', 'I', 'bg0', 'bg1']
    g = m.render(POS)
    assert g.shape == (3, 2, 13, 13)
    a = m.models[0].render(POS)
    b = m.models[1].render(POS + torch.tensor([[0.6, -0.4, 0.0]])) * 0.8
    assert torch.allclose(g[:, 0], a) and torch.allclose(g[:, 1], b, atol=1e-7)
    with pytest.raises(NotImplementedError):
        m.localize('movie.tif')
    with pytest.raises(ValueError):
        m.expected(torch.zeros(3, 5))


# ====================================================================================== Jacobian
def _fd_check(m, th, offsets=None, tol=2e-2, eps=None):
    mu, J = m.mu_and_jacobian(th, offsets)
    assert J.shape == (len(th), m.n_params) + mu.shape[1:]
    for k in range(m.n_params):
        if m.n_dims == 2 and k == 2:
            continue
        e = (eps or {}).get(k, 1e-3 if k < 3 else (1e-2 if k >= 4 + m.n_channels else 0.1))
        tp, tm = th.clone(), th.clone()
        tp[:, k] += e
        tm[:, k] -= e
        with torch.no_grad():
            fd = (m.expected(tp, offsets=offsets) - m.expected(tm, offsets=offsets)) / (2 * e)
        err = float((fd - J[:, k]).abs().max() / J[:, k].abs().max().clamp_min(1e-12))
        assert err < tol, (m.param_names[k], err)


def test_jacobian_channels_psf():
    m = gauss_pair()
    th = m.theta(POS, torch.tensor([1000.0, 2000.0, 1500.0]), torch.tensor([[3.0, 4.0], [1.0, 2.0], [5.0, 5.0]]))
    po = torch.tensor([[[0.0, 0.0], [0.2, 0.1]]]).expand(3, 2, 2)
    _fd_check(m, th, po, tol=5e-3)


def test_jacobian_fourpi_vectorial_and_spline():
    p = fourpi(offsets=OFF, R=11, npupil=24, z_range=(-0.2, 0.2))
    th = p.theta(POS, torch.tensor([1000.0, 2000.0, 1500.0]), torch.tensor([[3.0, 4.0, 1.0, 2.0]]).expand(3, 4), phi=PHI)
    assert th.shape == (3, 9) and p.param_names[-1] == 'phi'
    _fd_check(p, th, tol=2e-2)
    spl = p.to_spline(11, (-0.2, 0.2), oversample=2, dz_um=0.03, zpad=2)
    spl.set_channel_offsets(OFF)
    _fd_check(spl, th, torch.tensor([[[0.1, 0.0]] * 4]).expand(3, 4, 2), tol=5e-3)


# ====================================================================================== fits and CRLB
def _simulate(m, N, I=2000.0, bg=(5.0, 8.0), zr=0.3, seed=1, **extra):
    g = torch.Generator().manual_seed(seed)
    pos = torch.stack([torch.rand(N, generator=g) - 0.5, torch.rand(N, generator=g) - 0.5,
                       2 * zr * torch.rand(N, generator=g) - zr], 1)
    th = m.theta(pos, torch.full((N,), I), torch.tensor([bg]).expand(N, len(bg)), **extra)
    d = torch.poisson(m.expected(th), generator=g)
    return th, d


@needs_gpu
def test_fit_and_crlb_channels_psf():
    m = gauss_pair()
    th, d = _simulate(m, 1000)
    est, nll = m.fit_theta(d, zstarts=3)
    err = est - th
    cr = m.crlb(th)
    ratio = err.std(0) / cr.mean(0)
    assert ((ratio > 0.85) & (ratio < 1.2)).all(), ratio            # CRLB reached (x, y, z, I, bg0, bg1)
    assert (err.mean(0).abs() < 0.15 * cr.mean(0)).all(), err.mean(0) / cr.mean(0)
    # per-spot offsets (ROIs cut per channel) add to the channel offsets
    po = 0.5 * (torch.rand(300, 2, 2, generator=torch.Generator().manual_seed(7)) - 0.5)
    do = torch.poisson(m.expected(th[:300], offsets=po), generator=torch.Generator().manual_seed(8))
    eo, _ = m.fit_theta(do, offsets=po, zstarts=3)
    ro = (eo - th[:300]).std(0) / cr[:300].mean(0)
    assert (ro < 1.3).all(), ro
    ds = m.fit(d[:50], zstarts=3)
    assert len(ds) == 50 and ds['bg_channels'].shape == (50, 2) and ds['pos'].shape == (50, 3)
    assert torch.allclose(ds['pos'][:, 0].float(), est[:50, 0], atol=1e-5)
    assert torch.allclose(ds['pos'][:, 2].float(), est[:50, 2] / 0.1, atol=1e-4)
    # 2D models: z fixed, NaN in the CRLB
    m2 = ChannelsPSF([GaussianPSF(11, 1.3, device=DEV), GaussianPSF(11, 1.5, device=DEV)], [[0, 0], [-0.5, 0.3]])
    th2, d2 = _simulate(m2, 300, zr=0.0)
    est2, _ = m2.fit_theta(d2)
    assert bool(torch.isnan(m2.crlb(th2)[:, 2]).all()) and float((est2[:, 2]).abs().max()) == 0
    assert float((est2[:, :2] - th2[:, :2]).abs().median()) < 0.05


def test_fit_channels_vectorial():
    vs = [VectorialPSF(11, NA=1.4, npupil=24, oversample=1, device=DEV, focus_um=f, z_range=(0.0, 0.6), pixel_integration='midpoint') for f in (0.2, 0.45)]
    m = ChannelsPSF(vs, [[0, 0], [0.4, -0.3]])
    th, d = _simulate(m, 30, I=3000.0, bg=(4.0, 4.0), zr=0.2)
    th[:, 2] += 0.3
    d = torch.poisson(m.expected(th), generator=torch.Generator().manual_seed(2))
    est, _ = m.fit_theta(d, zstarts=2, iters=20)
    err = (est - th).abs().median(0).values
    cr = m.crlb(th).median(0).values
    assert (err[:3] < 2.5 * cr[:3]).all(), (err, cr)


@pytest.fixture(scope='module')
def spline4pi():
    p = fourpi(R=11, npupil=24, offsets=[[0.0, 0.0], [0.3, -0.2], [-0.25, 0.15], [0.1, 0.3]], z_range=(-0.25, 0.25))
    return p, p.to_spline(11, (-0.25, 0.25), oversample=2, dz_um=0.025, zpad=3)


def _wrap(e):
    return torch.remainder(e + math.pi, 2 * math.pi) - math.pi


@needs_gpu
def test_fourpi_spline_global_fit(spline4pi):
    """multi-start (z x phase) search from the default starting values"""
    _, spl = spline4pi
    N = 30
    g = torch.Generator().manual_seed(4)
    th, d = _simulate(spl, N, I=3000.0, bg=(10.0,) * 4, zr=0.15, seed=4, phi=2 * math.pi * torch.rand(N, generator=g) - math.pi)
    est, _ = spl.fit_theta(d, zstarts=2)
    err = est - th
    err[:, 8] = _wrap(err[:, 8])
    cr = spl.crlb(th)
    assert float(est[:, 8].abs().max()) <= math.pi + 1e-6               # wrapped to (-pi, pi]
    assert (err.abs() < 5 * cr).all(), (err / cr).abs().max(0).values     # no outliers (phase / z ambiguity)
    ds = spl.fit(d[:10], zstarts=2)
    assert set(('phi', 'crlb_phi', 'bg_channels', 'theta')) <= set(ds.keys())


@needs_gpu
def test_fourpi_spline_crlb_phi_free_and_fixed(spline4pi):
    """Monte-Carlo: the spread of the estimates reaches the CRLB, with the phase free and fixed (started near the
    truth: this tests the likelihood and its bound, the global search is test_fourpi_spline_global_fit)"""
    _, spl = spline4pi
    N = 300
    g = torch.Generator().manual_seed(5)
    th, d = _simulate(spl, N, I=3000.0, bg=(10.0,) * 4, zr=0.15, seed=5, phi=2 * math.pi * torch.rand(N, generator=g) - math.pi)
    t0 = th.clone()
    t0[:, :3] += torch.tensor([0.1, -0.1, 0.03])
    t0[:, 3] *= 0.9
    t0[:, 8] += 0.3
    est, _ = spl.fit_theta(d, theta0=t0)
    err = est - th
    err[:, 8] = _wrap(err[:, 8])
    cr = spl.crlb(th)
    ratio = err.std(0) / cr.mean(0)
    assert ((ratio > 0.8) & (ratio < 1.25)).all(), ratio
    # phase fixed at the true value: phi stays put, z and xy at their (smaller) bound
    t0[:, 8] = th[:, 8]
    est_f, _ = spl.fit_theta(d, theta0=t0, fixed=('phi',))
    assert torch.allclose(est_f[:, 8], th[:, 8], atol=1e-6)
    cr_f = spl.crlb(th, fixed=('phi',))
    assert bool(torch.isnan(cr_f[:, 8]).all()) and bool((cr_f[:, 2] <= cr[:, 2] + 1e-6).all())
    rf = (est_f - th).std(0)[:8] / cr_f.mean(0)[:8]
    assert ((rf > 0.8) & (rf < 1.25)).all(), rf


@needs_gpu
def test_fourpi_vectorial_fit_with_offsets(spline4pi):
    p, _ = spline4pi
    N = 6
    th, d = _simulate(p, N, I=4000.0, bg=(5.0,) * 4, zr=0.1, phi=torch.linspace(-2.5, 2.5, N))
    t0 = th.clone()
    t0[:, :3] += torch.tensor([0.15, -0.1, 0.02])
    t0[:, 3] *= 0.8
    est, _ = p.fit_theta(d, theta0=t0, iters=10)
    cr = p.crlb(th)
    assert ((est - th)[:, :3].abs() < 4 * cr[:, :3]).all()
    est_f, _ = p.fit_theta(d, theta0=t0, iters=3, fixed=('phi',))
    assert torch.equal(est_f[:, 8], p.wrap_extra(t0.clone())[:, 8])


# ====================================================================================== spline, state, files
def test_to_spline_bakes_offsets(spline4pi):
    p, spl = spline4pi
    assert not spl.has_offsets()
    pos = torch.tensor([[0.2, -0.3, 0.05], [-0.4, 0.1, -0.13], [0.33, 0.0, 0.17]])
    with torch.no_grad():
        a, b = spl.render(pos, phi=PHI), p.render(pos, phi=PHI)
    assert float((a - b).abs().max() / b.abs().max()) < 1e-2
    zk = float(spl.table.zs[5])
    pk = torch.tensor([[0.0, 0.0, zk], [0.5, -0.5, zk]])
    with torch.no_grad():
        a, b = spl.render(pk, phi=PHI[:2]), p.render(pk, phi=PHI[:2])
    assert float((a - b).abs().max() / b.abs().max()) < 1e-5
    assert spl.source_state['channel_offsets'] == p.channel_offsets.tolist()


def test_state_and_file_round_trip_with_offsets(tmp_path):
    p = fourpi(offsets=OFF)
    s = p.state()
    assert s['channel_offsets'] == torch.tensor(OFF).tolist() and 'channel_offsets' not in s['raw']
    q = FourPiPSF.from_state(s, device=DEV)
    assert torch.equal(q.channel_offsets, p.channel_offsets)
    with torch.no_grad():
        assert torch.allclose(q.render(POS, phi=PHI), p.render(POS, phi=PHI), atol=1e-7)
    # states without offsets: unchanged keys, load as zeros
    s0 = fourpi().state()
    assert 'channel_offsets' not in s0
    assert not FourPiPSF.from_state(s0, device=DEV).has_offsets()
    # a trainable offset is a parameter (gradient flows), still not in raw
    p.set_channel_offsets(OFF, trainable=True)
    assert isinstance(p.channel_offsets, torch.nn.Parameter) and 'channel_offsets' not in p.state()['raw']
    (p.render(POS, phi=PHI) * RAMP).sum().backward()
    assert float(p.channel_offsets.grad.abs().min()) > 0
    # zero trainable offsets also get a gradient (the shared-position shortcut is skipped)
    p.set_channel_offsets(torch.zeros(4, 2), trainable=True)
    (p.render(POS, phi=PHI) * RAMP).sum().backward()
    assert float(p.channel_offsets.grad.abs().min()) > 0
    p.set_channel_offsets(OFF)
    p.save(tmp_path / 'fp.json')
    r = PSF.load(tmp_path / 'fp.json', device=DEV)
    assert torch.equal(r.channel_offsets, torch.tensor(OFF))
    # spline with its own offsets: torch-psf json round trip; psf.zip refuses (no field for them)
    spl = fourpi(R=11, npupil=16).to_spline(11, (-0.1, 0.1), oversample=2, dz_um=0.05, zpad=2)
    spl.set_channel_offsets(OFF)
    spl.save(tmp_path / 'spl.json')
    s2 = PSF.load(tmp_path / 'spl.json', device=DEV)
    assert torch.equal(s2.channel_offsets, spl.channel_offsets)
    assert torch.equal(s2.render(POS[:2] * 0.3, phi=PHI[:2]), spl.render(POS[:2] * 0.3, phi=PHI[:2]))
    from torch_psf import io
    with pytest.raises(ValueError):
        io.save_zip(tmp_path / 'spl.psf.zip', spl)


def test_psfzip_multichannel(tmp_path):
    import warnings
    from torch_psf import io
    from torch_psf.psfzip import PSFZip
    m = gauss_pair(offsets=None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        io.save_zip(tmp_path / 'two.psf.zip', list(m.models),
                    channels=[dict(name='a'), dict(name='b', intensity_ratio=0.7)])
    z = PSFZip.read(tmp_path / 'two.psf.zip', device=DEV)
    assert z.psf is None
    mc = z.multichannel()
    assert isinstance(mc, ChannelsPSF) and mc.names == ['a', 'b']
    assert torch.allclose(mc.intensity_ratio, torch.tensor([1.0, 0.7]))
    with torch.no_grad():
        assert torch.allclose(mc.render(POS)[:, 1], 0.7 * m.models[1].render(POS), atol=1e-7)
