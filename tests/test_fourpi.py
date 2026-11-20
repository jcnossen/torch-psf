"""FourPiPSF / FourPiSplinePSF: shapes, normalization, the analytic phase dependence, derivatives, files, spline table."""
import math
import torch
from torch_psf import PSF, VectorialPSF, FourPiPSF, FourPiSplinePSF

DEV = 'cpu'


def model(asym=False, mirror='y', zmode='emitter', R=13):
    kw = dict(NA=(1.45, 1.3), n_imm=(1.518, 1.406), n_med=1.33, depth_um=(0.1, 1.0),
              zernike_indices=tuple(range(4, 16))) if asym else dict(NA=1.35, n_imm=1.406, zernike_indices=tuple(range(5, 16)))
    p = FourPiPSF(R, pixel_um=0.11, wavelength_um=0.67, npupil=32, oversample=2, mirror=mirror, device=DEV, pixel_integration='midpoint',
                  phi=(0.3, 1.9, 3.4, 5.0), tau=(0.2, 0.3, 0.25, 0.25), a=0.9, gamma=0.8, d_bot=(0.3, -0.2, 0.02),
                  blur_px2=0.12, **kw)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        p.top.zcoef.copy_(0.04 * (torch.rand(len(p.zidx), generator=g) - 0.5))
        p.bot.zcoef.copy_(0.04 * (torch.rand(len(p.zidx), generator=g) - 0.5))
    from torch_psf import ZMode
    return ZMode(p, zmode)


POS = torch.tensor([[0.2, -0.4, 0.05], [-0.7, 0.3, -0.12], [0.0, 0.0, 0.2]])


def test_shape_and_normalization():
    p = FourPiPSF(31, NA=1.35, n_imm=1.406, npupil=48, oversample=2, device=DEV, gamma=0.5, blur_px2=0.01, pixel_integration='midpoint')
    v = VectorialPSF(31, pixel_um=p.pixel_um, NA=1.35, n_imm=1.406, n_med=1.406, wavelength_um=0.68, npupil=48,
                     oversample=2, device=DEV, pixel_integration='midpoint')
    v.blur = torch.tensor([0.01, 0.01, 0.0])
    with torch.no_grad():
        I = p(torch.zeros(1, 3), phi=torch.tensor([0.4]))
        ref = float(v(torch.zeros(1, 3), 0.0).sum())   # one arm, the same ROI (the tails outside it are missing)
    assert I.shape == (1, 4, 31, 31)
    assert ref > 0.9
    # the interference cancels in the sum over the 4 channels: the sum is that of one arm
    assert abs(float(I.sum()) / ref - 1.0) < 1e-4
    # mean over the phase: each channel gets tau_c of the light
    with torch.no_grad():
        Im = sum(p(torch.zeros(1, 3), phi=torch.tensor([ph])) for ph in (0.0, math.pi)) / 2
    tot = Im.sum((2, 3))[0]
    assert torch.allclose(tot / tot.sum(), torch.full((4,), 0.25), atol=1e-3)


def test_more_channels():
    p = FourPiPSF(11, pol=(0, 1, 0, 1, 0, 1), npupil=24, oversample=1, device=DEV, pixel_integration='midpoint')
    with torch.no_grad():
        I = p(POS)
    assert I.shape == (3, 6, 11, 11) and torch.allclose(p.tau, torch.full((6,), 1 / 6))


def test_phase_decomposition_identity():
    for kw in (dict(), dict(mirror='x'), dict(asym=True), dict(asym=True, zmode='stage')):
        p = model(**kw)
        with torch.no_grad():
            g0, gpi, g90 = (p(POS, phi=torch.full((3,), v)) for v in (0.0, math.pi, math.pi / 2))
            inc, cr, ci = 0.5 * (g0 + gpi), 0.5 * (g0 - gpi), 0.5 * (g0 + gpi) - g90
            Phi = torch.tensor([0.7, 2.9, -1.3])
            g = p(POS, phi=Phi)
        pred = inc + torch.cos(Phi)[:, None, None, None] * cr - torch.sin(Phi)[:, None, None, None] * ci
        assert float((g - pred).abs().max() / g.abs().max()) < 1e-5, kw


def test_gradients_match_finite_differences():
    p = model(asym=True)
    Phi = torch.tensor([0.4, 1.1, 2.5])
    out = p.render(POS, grad=True, phi=Phi)
    assert len(out) == 5
    for k in range(4):
        e = 1e-3 if k < 3 else 1e-2
        p1, p2, f1, f2 = POS.clone(), POS.clone(), Phi.clone(), Phi.clone()
        if k < 3:
            p1[:, k] += e; p2[:, k] -= e
        else:
            f1 += e; f2 -= e
        with torch.no_grad():
            fd = (p.render(p1, phi=f1) - p.render(p2, phi=f2)) / (2 * e)
        assert float((fd - out[k + 1]).abs().max() / out[k + 1].abs().max()) < 2e-2, k


def test_parameter_gradients():
    p = model()
    (p(POS, phi=torch.tensor([0.1, 0.2, 0.3])) ** 2).sum().backward()
    for prm in (p.top.zcoef, p.bot.zcoef, p.phi, p.tau_logit, p.a_log, p.gamma_logit, p.d_bot, p.logblur):
        assert prm.grad is not None and torch.isfinite(prm.grad).all() and float(prm.grad.abs().max()) > 0


def test_state_and_file_round_trip(tmp_path):
    for kw in (dict(), dict(mirror='x', asym=True, zmode='stage')):
        p = model(**kw)
        p.dipM = torch.diag(torch.tensor([0.4, 0.4, 0.2]))
        Phi = torch.tensor([0.3, 1.0, 2.0])
        with torch.no_grad():
            ref = p(POS, phi=Phi)
            q = FourPiPSF.from_state(p.state(), device=DEV)
            q.dipM = p.dipM
            assert float((q(POS, phi=Phi) - ref).abs().max() / ref.max()) < 1e-5
            p.save(tmp_path / 'fp.json')
            r = PSF.load(tmp_path / 'fp.json', device=DEV)
            assert isinstance(r, FourPiPSF) and r.asym == p.asym and r.state()["meta"]["zmode"] == p.state()["meta"]["zmode"] and r.mirror == p.mirror
            assert float((r(POS, phi=Phi) - ref).abs().max() / ref.max()) < 1e-5


def test_spline_matches_model_and_phase():
    p = model(asym=True, R=11)
    spl = p.to_spline(11, (-0.2, 0.2), oversample=3, dz_um=0.02, zpad=3)
    assert isinstance(spl, FourPiSplinePSF) and spl.spline.spline_index.shape == (3, 4)
    pos = torch.tensor([[0.2, -0.3, 0.05], [-0.4, 0.1, -0.13], [0.33, 0.0, 0.17]])
    Phi = torch.tensor([0.5, 2.0, 4.0])
    with torch.no_grad():
        a, b = spl.render(pos, phi=Phi), p(pos, phi=Phi)
    assert a.shape == b.shape == (3, 4, 11, 11)
    assert float((a - b).abs().max() / b.abs().max()) < 1e-2
    # at the knots (sub-pixel shift 0, knot z) the table reproduces the model up to float32 noise
    zk = float(spl.spline.zs[5])
    pk = torch.tensor([[0.0, 0.0, zk], [1.0 / 3, -2.0 / 3, zk]])
    with torch.no_grad():
        a, b = spl.render(pk, phi=Phi[:2]), p(pk, phi=Phi[:2])
    assert float((a - b).abs().max() / b.abs().max()) < 1e-5
    # the per-channel path (render_channels) and the component view agree with evaluate
    v = spl.evaluate(pos, Phi)
    assert torch.allclose(spl.render_channels(pos[:, None].expand(3, 4, 3), Phi[:, None]), v, atol=1e-7)
    inc, cr, ci = spl.components(pos)
    assert torch.allclose(inc + torch.cos(Phi)[:, None, None, None] * cr - torch.sin(Phi)[:, None, None, None] * ci, v, atol=1e-7)


def test_spline_gradients_and_file(tmp_path):
    p = model(R=11)
    spl = p.to_spline(11, (-0.2, 0.2), oversample=2, dz_um=0.03, zpad=2)
    pos = torch.tensor([[0.13, -0.27, 0.031], [-0.4, 0.2, -0.12]])
    Phi = torch.tensor([0.8, 3.3])
    out = spl.render(pos, grad=True, phi=Phi)
    for k in range(4):
        e = 1e-3
        p1, p2, f1, f2 = pos.clone(), pos.clone(), Phi.clone(), Phi.clone()
        if k < 3:
            p1[:, k] += e; p2[:, k] -= e
        else:
            f1 += e; f2 -= e
        fd = (spl.render(p1, phi=f1) - spl.render(p2, phi=f2)) / (2 * e)
        assert float((fd - out[k + 1]).abs().max() / out[k + 1].abs().max()) < 5e-3, k
    g = spl.render_channels(pos[:, None].expand(2, 4, 3), Phi[:, None], grad=True)
    for k in range(5):
        assert torch.allclose(g[k], out[k], atol=1e-7)
    spl.save(tmp_path / 'spl.json')
    s2 = PSF.load(tmp_path / 'spl.json', device=DEV)
    assert isinstance(s2, FourPiSplinePSF) and s2.source_state['meta']['R'] == 11
    assert torch.equal(s2.render(pos, phi=Phi), spl.render(pos, phi=Phi))
    assert torch.allclose(s2.tau, spl.tau)
