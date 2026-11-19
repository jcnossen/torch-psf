"""torch_psf.orientation: dipole-orientation model, linear-M basis, excitation factor, CRLB and the MLE fit."""
import math
import torch
from torch_psf import VectorialPSF
from torch_psf.orientation import (angles, from_angles, unit, canonical, cone_to_g, g_to_cone, ensemble_gamma,
                                   dipole_M, M_to_m6, m6_to_M, project_M, excitation_tensor, excitation_factor,
                                   basis_images, solve_M, model_and_jacobian, crlb_orientation, fit_orientation,
                                   angle_error, vortex_phase, tangent_basis)


def small_model(R=11, npupil=32, aberr=True, oversample=2, **kw):
    v = VectorialPSF(R, pixel_um=0.108, wavelength_um=0.7, npupil=npupil, oversample=oversample, device='cpu',
                     focus_um=0.25, **kw)
    if aberr:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            v.zcoef.copy_(torch.randn(17, generator=g) * 0.02)
    for p in v.parameters():
        p.requires_grad_(False)
    return v


def test_angles_roundtrip_and_cone():
    th = torch.tensor([0.0, 20.0, 45.0, 89.0, 60.0])
    ph = torch.tensor([0.0, 300.0, 45.0, 170.0, 200.0])
    mu = from_angles(th, ph)
    t2, p2 = angles(-mu)                       # antipodal: same orientation
    assert torch.allclose(t2, th, atol=1e-3)
    assert torch.allclose(p2[1:], ph[1:], atol=1e-3)
    _, pf = angles(mu, fold=True)
    assert float(pf.max()) < 180
    a = torch.tensor([0.0, 30.0, 60.0, 90.0])
    assert torch.allclose(g_to_cone(cone_to_g(a)), a, atol=1e-2)
    assert abs(float(cone_to_g(0.0)) - 1) < 1e-6 and abs(float(cone_to_g(90.0))) < 1e-6
    e1, e2 = tangent_basis(mu)
    assert float((e1 * mu).sum(-1).abs().max()) < 1e-5 and float((e2 * mu).sum(-1).abs().max()) < 1e-5


def test_dipole_tensor_projection_and_m6():
    g = torch.Generator().manual_seed(1)
    v = torch.randn(20, 3, generator=g)
    gg = torch.rand(20, generator=g) * 0.9 + 0.05
    M = dipole_M(v, gg)
    assert torch.allclose(torch.diagonal(M, dim1=1, dim2=2).sum(1), torch.ones(20), atol=1e-6)
    assert torch.allclose(m6_to_M(M_to_m6(M)), M, atol=1e-7)
    mu, g2 = project_M(M * 3.7)                # scale-free
    assert float(angle_error(mu, unit(v)).max()) < 0.1     # deg (acos near 1 in float32)
    assert torch.allclose(g2, gg, atol=1e-5)


def test_ensemble_photoselection_gamma():
    """<M (3 tr(M K))> over uniformly distributed axes = (1 - 0.4 g^2) I / 3 + 0.4 g^2 K."""
    g = torch.Generator().manual_seed(2)
    mu = unit(torch.randn(200000, 3, generator=g))
    K = excitation_tensor((0.0, 1.0), eps=0.3, k_dir=(1.0, 0.0))
    assert abs(float(torch.trace(K)) - 1) < 1e-6
    for go in (1.0, 0.5):
        M = dipole_M(mu, torch.full((len(mu),), go))
        f = excitation_factor(M, K)
        Me = (M * f[:, None, None]).mean(0) / f.mean()
        gam = float(ensemble_gamma(go))
        assert torch.allclose(Me, (1 - gam) * torch.eye(3) / 3 + gam * K, atol=4e-3)
    assert abs(float(excitation_factor(torch.eye(3) / 3, K)) - 1) < 1e-6


import pytest


@pytest.mark.parametrize('mode,blur', [('box', True), ('spectral', True), ('gauss_legendre', False)])
def test_basis_images_are_linear_in_M(mode, blur):
    v = small_model(pixel_integration=mode) if mode == 'box' else small_model(pixel_integration=mode, oversample=3)
    v.blur = torch.tensor([0.2, 0.1, 0.05]) if blur else None
    pos = torch.tensor([[0.3, -0.2, 0.05], [-0.4, 0.1, 0.15]])
    B = basis_images(v, pos)
    M = dipole_M(torch.tensor([[1.0, 2.0, 0.5], [0.2, -0.3, 1.0]]), torch.tensor([0.8, 0.4]))
    a = v(pos, v.focus_um, dipM=M)
    b = torch.einsum('nk,nkuv->nuv', M_to_m6(M), B)
    assert float((a - b).abs().max() / a.max()) < 1e-5
    iso = v(pos, v.focus_um)
    assert float((iso - B[:, :3].sum(1) / 3).abs().max() / iso.max()) < 1e-5


def test_solve_M_recovers_noise_free_tensor():
    v = small_model()
    pos = torch.tensor([[0.2, 0.1, 0.08]] * 3)
    M = dipole_M(from_angles(torch.tensor([30.0, 70.0, 85.0]), torch.tensor([10.0, 120.0, 250.0])),
                 torch.tensor([1.0, 0.7, 0.5]))
    B = basis_images(v, pos)
    d = 2000 * torch.einsum('nk,nkuv->nuv', M_to_m6(M), B) + 10
    Me, I, bg = solve_M(B, d)
    assert torch.allclose(Me, M, atol=1e-3)
    assert torch.allclose(I, torch.full((3,), 2000.0), rtol=1e-3)
    assert torch.allclose(bg, torch.full((3,), 10.0), rtol=1e-3)


def test_jacobian_matches_finite_differences():
    v = small_model()
    K = torch.stack([excitation_tensor((0.0, 1.0), 0.3, (1.0, 0.0)), excitation_tensor((1.0, 0.0), 0.3, (0.0, 1.0))])
    th = torch.tensor([[0.2, -0.1, 0.06, 1000.0, 10.0, *from_angles(50.0, 30.0).tolist(), 0.7]])
    mu, J = model_and_jacobian(v, th, T=2, K=K)
    for k in range(9):
        e = 1e-3 * (1.0 if k != 3 else 100.0)
        t1, t2 = th.clone(), th.clone()
        t1[0, k] += e; t2[0, k] -= e
        fd = (model_and_jacobian(v, t1, T=2, K=K)[0] - model_and_jacobian(v, t2, T=2, K=K)[0]) / (2 * e)
        assert float((fd - J[:, k]).abs().max()) < 2e-2 * float(J[:, k].abs().max()) + 1e-4, k


def test_crlb_finite_at_pole_and_excitation_helps():
    v = small_model()
    th = torch.tensor([[0.0, 0.0, 0.05, 2000.0, 10.0, 0.0, 0.0, 1.0, 1.0],
                       [0.0, 0.0, 0.05, 2000.0, 10.0, *from_angles(80.0, 45.0).tolist(), 1.0]])
    c = crlb_orientation(v, th)
    assert torch.isfinite(c['theta']).all() and torch.isfinite(c['inplane']).all()
    K = torch.stack([excitation_tensor((0.0, 1.0))] * 3 + [excitation_tensor((1.0, 0.0))] * 3)
    c0 = crlb_orientation(v, th[1:], T=6)
    c1 = crlb_orientation(v, th[1:], T=6, K=K)
    assert float(c1['phi']) < float(c0['phi'])


def test_vortex_phase_makes_a_donut_for_in_plane_dipoles():
    v = small_model(R=15, aberr=False)
    v.focus_um = 0.0
    pos = torch.zeros(1, 3)
    M = dipole_M(from_angles(90.0, 0.0)[None], torch.ones(1))
    a = v(pos, 0.0, dipM=M)
    vortex_phase(v)
    b = v(pos, 0.0, dipM=M)
    c = 7
    assert float(a[0, c, c]) == float(a.max())           # standard PSF: peak in the centre
    assert float(b[0, c, c]) < 0.2 * float(b.max())       # Vortex: dark centre


def test_fit_spots_and_track():
    torch.manual_seed(0)
    v = small_model()
    mus = from_angles(torch.tensor([60.0, 35.0]), torch.tensor([30.0, 200.0]))
    n = 8
    th = torch.cat([torch.tensor([[0.1, -0.2, 0.06, 20000.0, 10.0]]).repeat(2 * n, 1),
                    mus.repeat_interleave(n, 0), torch.full((2 * n, 1), 0.9)], 1)
    mu, _ = model_and_jacobian(v, th)
    d = torch.poisson(mu[:, 0])
    r = fit_orientation(v, d, iters=25, zstarts=2)
    err = angle_error(r['theta'][:, 5:8], th[:, 5:8])
    sd = torch.sqrt(r['crlb']['theta'] ** 2 + r['crlb']['inplane'] ** 2)
    assert float(err.median()) < 3 * float(sd.median()) + 0.5
    assert float((r['theta'][:, 2] - 0.06).abs().median()) < 0.03
    # track: 4 frames, excitation K per frame, a shared orientation
    K = torch.stack([excitation_tensor((0.0, 1.0))] * 2 + [excitation_tensor((1.0, 0.0))] * 2)
    th2 = th[::n].clone(); th2[:, 3] = 5000.0
    mu2, _ = model_and_jacobian(v, th2, T=4, K=K)
    d2 = torch.poisson(mu2)
    r2 = fit_orientation(v, d2, K=K, iters=25, zstarts=2)
    assert float(angle_error(r2['theta'][:, 5:8], th2[:, 5:8]).max()) < 10


def test_orientation_estimator_on_a_small_movie():
    from torch_psf.orientation import OrientationEstimator
    from torch_psf.camera import Camera
    torch.manual_seed(0)
    v = small_model(R=11)
    big = v.resized(17)
    vortex_phase(v, charge=-1); vortex_phase(big, charge=-1)
    cam = Camera(gain_e_per_adu=0.5, offset_adu=100.0)
    mus = from_angles(torch.tensor([50.0, 70.0]), torch.tensor([30.0, 200.0]))
    mov = torch.full((2, 40, 40), 10.0)
    xy = [(12.3, 14.6), (27.8, 25.2)]
    for t in range(2):
        for (x, y), mu in zip(xy, mus):
            ix, iy = int(round(x)), int(round(y))
            with torch.no_grad():
                img = big(torch.tensor([[x - ix, y - iy, 0.05]]), v.focus_um, dipM=dipole_M(mu[None], torch.ones(1)))[0]
            mov[t, iy - 8:iy + 9, ix - 8:ix + 9] += 20000 * img
    adu = torch.poisson(mov) / 0.5 + 100.0
    est = OrientationEstimator(v, camera=cam, threshold=5.0, iters=25, z_limits=(0.0, 0.3), zstarts=2)
    ds = est.estimate(adu.numpy())
    assert len(ds) == 4
    for k in ('theta_deg', 'phi_deg', 'g', 'crlb_theta_deg', 'mu', 'chi2', 'photons'):
        assert k in ds
    for i in range(len(ds)):
        j = 0 if float(ds['pos'][i, 0]) < 20 else 1
        assert abs(float(ds['pos'][i, 0]) - xy[j][0]) < 0.3 and abs(float(ds['pos'][i, 1]) - xy[j][1]) < 0.3
        assert float(angle_error(ds['mu'][i:i + 1], mus[j:j + 1])) < 6
    assert 0.5 < float(ds['chi2'].median()) < 2


def test_vortex_bead_calibration_finds_the_charge():
    from torch_psf.vortex import vortex_psf, calibrate_beads
    from torch_psf.optics import Optics
    torch.manual_seed(0)
    o = Optics(NA=1.4, n_med=1.518, n_imm=1.518, wavelength_um=0.6, pixel_um=0.08)
    truth = vortex_psf(o, roisize=13, charge=-1, npupil=32, oversample=1, device='cpu')
    truth.set_zernike({5: 0.03, 8: -0.02})
    f = 0.15 * (torch.arange(9, dtype=torch.float32) - 4)
    stack = torch.full((9, 40, 40), 5.0)
    for x0, y0 in ((6, 6), (21, 20)):
        with torch.no_grad():
            img = truth(torch.tensor([[0.2, -0.1, 0.0]]).expand(9, 3).contiguous(), f)
        stack[:, y0:y0 + 13, x0:x0 + 13] += 20000 * img
    stack = torch.poisson(stack)
    start = vortex_psf(o, roisize=13, charge=-1, npupil=32, oversample=1, device='cpu')
    psf, rep = calibrate_beads([stack.numpy()], 0.15, start, charges=(-1, 1), iters=80, roisize=13, nbeads=2,
                               drift=False)
    assert rep['charge'] == -1
    assert rep['nll_per_px'][-1] < rep['nll_per_px'][1]
    assert abs(float(psf.zcoef[0]) - 0.03) < 0.01


def test_focus_z_wrapper_matches_defocus():
    from torch_psf.orientation import FocusZ
    v = small_model()
    w = FocusZ(v, height=0.02, z_range=(-0.4, 0.4))
    pos = torch.tensor([[0.1, -0.2, 0.15]])
    M = dipole_M(from_angles(40.0, 20.0)[None], torch.tensor([0.8]))
    a = w(pos, dipM=M)
    b = v(torch.tensor([[0.1, -0.2, 0.02]]), v.focus_um - 0.15, dipM=M)
    assert torch.allclose(a, b)
    B = basis_images(w, pos)
    assert torch.allclose(torch.einsum('nk,nkuv->nuv', M_to_m6(M), B), a, atol=1e-6)
    th = torch.tensor([[0.1, -0.2, 0.15, 2000.0, 10.0, *from_angles(40.0, 20.0).tolist(), 0.8]])
    assert torch.isfinite(crlb_orientation(w, th)['z']).all()


def test_channels_psf_uses_the_vectorial_dipole_basis():
    """ChannelsPSF of vectorial arms: its dipole_basis [N, C, 6, P, P] (VectorialPSF.dipole_basis per channel) combines
    with M_to_m6 to render(dipM); _dipole._m6 is orientation.M_to_m6."""
    from torch_psf import ChannelsPSF
    from torch_psf._dipole import _m6
    kw = dict(roisize=7, npupil=16, oversample=2, device='cpu', z_range=(0.0, 0.3))
    offsets = [[0.11, -0.19], [-0.22, 0.16]]
    arms = [VectorialPSF(**kw) for _ in offsets]
    for v, diat in zip(arms, [-1.0, 1.0]):
        v.diat = diat
    m = ChannelsPSF(arms, channel_offsets=offsets, intensity_ratio=[0.8, 1.2])
    pos = torch.tensor([[0.17, -0.23, 0.05], [-0.32, 0.24, 0.12]])
    M = dipole_M(from_angles(torch.tensor([30.0, 70.0]), torch.tensor([20.0, 200.0])), torch.tensor([0.9, 0.6]))
    assert torch.equal(_m6(M), M_to_m6(M))
    with torch.no_grad():
        B = m.dipole_basis(pos)
        ref = m.render(pos, dipM=M)
    assert B.shape == (2, 2, 6, 7, 7)
    comb = torch.einsum('nk,nckuv->ncuv', M_to_m6(M), B)
    assert float((comb - ref).abs().max()) <= 1e-5 * float(ref.abs().max())


def test_dipole_basis_focus_defaults_to_focus_um_in_channels_psf():
    """VectorialPSF.dipole_basis(pos) uses focus_um (as render), so a VectorialPSF subclass channel whose type_name is
    not 'vectorial' works in ChannelsPSF (which passes focus= only for type_name 'vectorial')"""
    from torch_psf import ChannelsPSF

    class OtherVectorial(VectorialPSF):
        type_name = None

    kw = dict(roisize=7, npupil=16, oversample=2, device='cpu', z_range=(0.0, 0.3), focus_um=0.05)
    arms = [OtherVectorial(**kw), VectorialPSF(**kw)]
    m = ChannelsPSF(arms, channel_offsets=[[0.1, -0.2], [-0.2, 0.1]])
    pos = torch.tensor([[0.17, -0.23, 0.01], [-0.32, 0.24, 0.02]])
    M = dipole_M(from_angles(torch.tensor([30.0, 70.0]), torch.tensor([20.0, 200.0])), torch.tensor([0.9, 0.6]))
    with torch.no_grad():
        assert torch.equal(arms[0].dipole_basis(pos), arms[0].dipole_basis(pos, arms[0].focus_um))
        B = m.dipole_basis(pos)
        ref = m.render(pos, dipM=M)
    comb = torch.einsum('nk,nckuv->ncuv', M_to_m6(M), B)
    assert float((comb - ref).abs().max()) <= 1e-5 * float(ref.abs().max())


def test_composed_vortex_reference():
    """the composed Vortex PSF (plate + Polychromatic + field Zernikes + FocusZ; study/orientation/composed_reference.py)
    reproduces tests/data/orientation_composed_ref.json: images, dipole bases and orientation fits (the reference for
    the psf-unify refactor; image_hash is informative)"""
    import json
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, 'study', 'orientation'))
    import composed_reference as CR
    ref = json.load(open(CR.REF_PATH))
    got = CR.compute()
    for e, r in zip(got['emitters'], ref['emitters']):
        assert abs(e['image_sum'] - r['image_sum']) <= 1e-4 * r['image_sum']
        assert abs(e['image_peak'] - r['image_peak']) <= 1e-4 * r['image_peak']
        for k, v in r['pixels'].items():
            assert abs(e['pixels'][k] - v) <= 1e-4 * r['image_peak']
        assert e['render_vs_basis_max_abs'] <= 1e-3
        for a, b in zip(e['dipole_basis_sums'], r['dipole_basis_sums']):
            assert abs(a - b) <= 1e-4 * max(abs(b), 1e-3)
        for name, tol in (('fit_noise_free', (0.2, 0.01, 0.002)), ('fit_noisy_seed1234', (0.5, 0.02, 0.005))):
            fe, fr = e[name], r[name]
            assert abs(fe['g'] - fr['g']) <= tol[1]
            assert abs(fe['z_um'] - fr['z_um']) <= tol[2]
            if r['emitter']['g'] > 0:
                assert abs(fe['theta_deg'] - fr['theta_deg']) <= tol[0]
                assert abs((fe['phi_deg'] - fr['phi_deg'] + 180) % 360 - 180) <= tol[0]
