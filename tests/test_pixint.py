"""torch_psf.pixint (prototype): exact pixel integration of the vectorial PSF and of a cubic spline."""
import math
import numpy as np
import pytest
import torch
from conftest import DEV, needs_gpu
from torch_psf import VectorialPSF, CubicSplinePSF
from torch_psf.pixint import (ExactVectorialPSF, PixelSplinePSF, build_pixel_spline, bspline_prefilter, _kernel,
                              _dkernel, _qkernel)


def _vec(dev='cpu', R=9, blur=None):
    v = VectorialPSF(R, npupil=32, oversample=2, device=dev, focus_um=0.2, z_range=(0.0, 0.3), wavelength_um=0.7)
    v.set_zernike({6: 0.05, 11: 0.03})
    v.blur = blur
    return v


@pytest.mark.parametrize('kind', ['catmull', 'bspline'])
def test_kernels(kind):
    s = torch.linspace(-3, 3, 6001, dtype=torch.float64)
    K, dK, Q = _kernel(s, kind), _dkernel(s, kind), _qkernel(s, kind)
    assert float(Q[0]) == 0.0 and abs(float(Q[-1]) - 1) < 1e-12
    assert float((torch.gradient(Q, spacing=(s,))[0] - K).abs().max()) < 1e-5
    assert float((torch.gradient(K, spacing=(s,))[0] - dK).abs().max()) < 1e-3
    # partition of unity of the integer translates
    t = torch.rand(50, dtype=torch.float64)
    assert float((sum(_kernel(t - k, kind) for k in range(-2, 4)) - 1).abs().max()) < 1e-12


def test_bspline_prefilter_interpolates():
    x = torch.randn(12, 15, 15, dtype=torch.float64)
    c = bspline_prefilter(x)
    # B-spline value at the knots = (c[k-1] + 4 c[k] + c[k+1]) / 6 per axis (interior)
    f = lambda a, d: (a.narrow(d, 0, a.shape[d] - 2) + 4 * a.narrow(d, 1, a.shape[d] - 2) + a.narrow(d, 2, a.shape[d] - 2)) / 6
    v = f(f(f(c, 0), 1), 2)
    assert float((v - x[1:-1, 1:-1, 1:-1]).abs().max()) < 1e-10


def test_exact_vectorial_agree():
    """spectral (uniform grid + exact sinc) and Gauss-Legendre agree; the midpoint rule converges to them as 1 / s^2"""
    vec = _vec()
    pos = torch.tensor([[0.1, -0.3, 0.05], [0.45, 0.2, 0.2]])
    with torch.no_grad():
        a = ExactVectorialPSF(vec, 'spectral', os=3, pad=4).render(pos)
        b = ExactVectorialPSF(vec, 'gauss', os=6).render(pos)
        pk = a.max()
        assert float((a - b).abs().max() / pk) < 2e-6
        e = [float((ExactVectorialPSF(vec, 'box', os=s).render(pos) - a).abs().max() / pk) for s in (2, 4)]
        assert 3.0 < e[0] / e[1] < 5.0                      # second order
        # 'box' reproduces VectorialPSF
        assert float((ExactVectorialPSF(vec, 'box', os=2).render(pos) - vec.render(pos)).abs().max()) < 1e-7


def test_exact_vectorial_blur():
    vec = _vec(blur=torch.tensor([0.4, 0.2, -0.2]))
    pos = torch.tensor([[0.1, -0.3, 0.05]])
    with torch.no_grad():
        a = ExactVectorialPSF(vec, 'spectral', os=2, pad=3).render(pos)
        b = ExactVectorialPSF(vec, 'spectral', os=4, pad=6).render(pos)
        c = ExactVectorialPSF(vec, 'box', os=8).render(pos)    # midpoint rule, error ~ 1 / 64 of s = 1
    assert float((a - b).abs().max() / b.max()) < 2e-5
    assert float((c - b).abs().max() / b.max()) < 2e-3
    with pytest.raises(ValueError):
        ExactVectorialPSF(vec, 'gauss', os=3)


def _gauss_spline(S, kernel, integrate, sigma=1.1, R=9):
    geo = CubicSplinePSF(R, (0.0, 0.2), oversample=S, dz_um=0.05, zpad=2, xy_limit=2.0, device='cpu')
    u = geo.u0 + torch.arange(geo.nu, dtype=torch.float64) / S
    if integrate:       # density per px^2
        g1 = torch.exp(-u ** 2 / (2 * sigma ** 2)) / math.sqrt(2 * math.pi * sigma ** 2)
    else:               # pixel integral of a pixel centred at -u: knot value g(u) = int over [u - 1/2, u + 1/2]
        Phi = lambda t: 0.5 * (1 + torch.erf(t / (sigma * math.sqrt(2))))
        g1 = Phi(u + 0.5) - Phi(u - 0.5)
    plane = g1[:, None] * g1[None, :]
    samples = plane[None].expand(geo.nz, -1, -1).float().contiguous()
    return PixelSplinePSF(geo, samples, kernel=kernel, integrate=integrate)


def _gauss_pixels(pos, sigma=1.1, R=9):
    o = torch.arange(R, dtype=torch.float64) - (R - 1) / 2
    Phi = lambda t: 0.5 * (1 + torch.erf(t / (sigma * math.sqrt(2))))
    ax = lambda c: Phi(o[None] + 0.5 - c[:, None]) - Phi(o[None] - 0.5 - c[:, None])
    return (ax(pos[:, 1].double())[:, :, None] * ax(pos[:, 0].double())[:, None, :]).float()


@pytest.mark.parametrize('S', [2, 3, 4])
def test_integrated_spline_gaussian(S):
    """integrated spline of a sampled Gaussian density vs the analytic pixel integral (erf)"""
    pos = torch.rand(40, 3) * torch.tensor([1.0, 1.0, 0.2]) - torch.tensor([0.5, 0.5, 0.0])
    ref = _gauss_pixels(pos)
    tol = {2: 3e-3, 3: 6e-4, 4: 2e-4}[S]
    for kernel, f in (('catmull', 1.0), ('bspline', 0.25)):
        spl = _gauss_spline(S, kernel, True)
        err = float((spl.render(pos) - ref).abs().max() / ref.max())
        assert err < tol * f * (4 if kernel == 'catmull' else 1), (kernel, err)


def test_point_spline_matches_cubicsplinepsf():
    spl = _gauss_spline(3, 'catmull', False)
    cs = CubicSplinePSF(9, (0.0, 0.2), oversample=3, dz_um=0.05, zpad=2, xy_limit=spl.xmax, device='cpu',
                        knots=spl.samples, knot_u0_px=spl.geo.u0, knot_z0_um=spl.geo.z0)
    pos = torch.rand(30, 3) * torch.tensor([1.0, 1.0, 0.2]) - torch.tensor([0.5, 0.5, 0.0])
    a, b = spl.render(pos, grad=True), cs.render(pos, grad=True)
    for x, y in zip(a, b):
        assert float((x - y).abs().max()) < 1e-5          # d/dz: float32 cancellation times 1 / hz


@pytest.mark.parametrize('integrate', [True, False])
def test_derivatives_and_backends(integrate):
    spl = _gauss_spline(3, 'bspline', integrate)
    spl.knots = spl.knots * (1 + 0.3 * torch.rand_like(spl.knots))       # make it z dependent
    pos = torch.tensor([[0.13, -0.27, 0.11], [-0.4, 0.2, 0.07], [0.0, 0.0, 0.1]])
    out = spl.render(pos, grad=True)
    for k in range(3):
        e = 1e-3
        p1, p2 = pos.clone(), pos.clone()
        p1[:, k] += e; p2[:, k] -= e
        fd = (spl.render(p1) - spl.render(p2)) / (2 * e)
        assert float((fd - out[k + 1]).abs().max() / out[k + 1].abs().max()) < 5e-3
    spl.backend = 'conv'
    for x, y in zip(out, spl.render(pos, grad=True)):
        assert float((x - y).abs().max() / y.abs().max()) < 1e-5
    # linear in the knots: autograd gradient
    K = spl.knots.clone().requires_grad_(True)
    (spl.render(pos, knots=K) ** 2).sum().backward()
    assert K.grad is not None and float(K.grad.abs().sum()) > 0


@needs_gpu
def test_vectorial_integrated_spline_accuracy():
    """integrated B-spline (S = 3) of the vectorial point density vs the exact pixel integral"""
    torch.backends.cudnn.allow_tf32 = False
    vec = _vec(DEV, R=11)
    spl = build_pixel_spline(vec, 11, (0.0, 0.3), 3, dz_um=0.01, kernel='bspline', integrate=True)
    ref = ExactVectorialPSF(vec, 'spectral', os=3, pad=4)
    pos = torch.rand(100, 3, device=DEV) * torch.tensor([1.0, 1.0, 0.3], device=DEV) - torch.tensor([0.5, 0.5, 0], device=DEV)
    with torch.no_grad():
        a, b = spl.render(pos), ref.render(pos)
    assert float((a - b).abs().max() / b.max()) < 2e-4


# ============================================================================== VectorialPSF.pixel_integration
@pytest.mark.parametrize('blur', [None, (0.4, 0.2, -0.2)])
def test_vectorial_pixel_integration_modes(blur):
    """VectorialPSF pixel_integration 'spectral' (= 'exact') and 'gauss_legendre' = the exact pixel integral (the
    prototype ExactVectorialPSF and a finer spectral reference); 'box' (default) is the midpoint rule, unchanged"""
    vec = _vec(blur=None if blur is None else torch.tensor(blur))
    pos = torch.tensor([[0.1, -0.3, 0.05], [0.45, 0.2, 0.2]])
    with torch.no_grad():
        ref = ExactVectorialPSF(vec, 'spectral', os=4, pad=6).render(pos)
        pk = ref.max()
        ex = vec.resized(vec.roisize, pixel_integration='exact')
        assert ex.pixel_integration == 'spectral' and vec.pixel_integration == 'box'
        a = ex.render(pos)
        assert float((a - ExactVectorialPSF(vec, 'spectral', os=2, pad=3).render(pos)).abs().max() / pk) < 1e-6
        assert float((a - ref).abs().max() / pk) < 1e-5
        box = vec.render(pos)
        assert float((box - ExactVectorialPSF(vec, 'box', os=2).render(pos)).abs().max()) < 1e-7
        assert 3e-3 < float((box - ref).abs().max() / pk) < 3e-2         # the midpoint error at oversample 2
        if blur is None:
            g = vec.resized(vec.roisize, oversample=3, pixel_integration='gauss_legendre').render(pos)
            assert float((g - ref).abs().max() / pk) < 1e-5
        else:
            with pytest.raises(ValueError):
                vec.resized(vec.roisize, oversample=3, pixel_integration='gauss_legendre').render(pos)
    with pytest.raises(ValueError):
        vec.resized(vec.roisize, pixel_integration='trapezoid')
    with pytest.raises(ValueError):
        vec.resized(vec.roisize, oversample=1, pixel_integration='spectral')


def test_vectorial_pixel_integration_grad_and_file(tmp_path):
    """the exact mode is differentiable (Zernike autograd, position jvp vs finite differences) and survives
    resized / save / PSF.load / to_spline"""
    from torch_psf import PSF
    vec = _vec().resized(9, pixel_integration='exact')
    pos = torch.tensor([[0.1, -0.3, 0.05], [0.45, 0.2, 0.2]])
    val, dx, dy, dz = vec.render(pos, grad=True)
    e = 1e-3
    for k, d in enumerate((dx, dy, dz)):
        p1, p2 = pos.clone(), pos.clone()
        p1[:, k] += e; p2[:, k] -= e
        with torch.no_grad():
            fd = (vec.render(p1) - vec.render(p2)) / (2 * e)
        d = d.detach()
        assert float((fd - d).abs().max() / d.abs().max()) < 2e-3
    vec.zcoef.requires_grad_(True)
    vec(pos, 0.2).pow(2).sum().backward()
    assert vec.zcoef.grad is not None and float(vec.zcoef.grad.abs().sum()) > 0
    vec.save(tmp_path / 'v.json')
    w = PSF.load(tmp_path / 'v.json', device='cpu')
    assert w.pixel_integration == 'spectral' and 'pixel_integration' in repr(w)
    assert 'pixel_integration' not in _vec()._file_dict()[0]            # 'box': files unchanged
    with torch.no_grad():
        assert float((w.render(pos) - vec.render(pos)).abs().max()) < 1e-7
        spl = vec.to_spline(z_range=(0.0, 0.3), oversample=4, dz_um=0.01, zpad=3, xy_limit=1.0)
        sb = _vec().resized(9).to_spline(z_range=(0.0, 0.3), oversample=4, dz_um=0.01, zpad=3, xy_limit=1.0)
        r = vec.render(pos)
        assert float((spl.render(pos) - r).abs().max() / r.max()) < float((sb.render(pos) - r).abs().max() / r.max())


def _exact_tracks(R=11, B=80, T=6, photons=4000.0, seed=4):
    from torch_psf.spots import Tracks
    vec = VectorialPSF(R, npupil=32, oversample=2, device=DEV, wavelength_um=0.7).set_zernike({6: 0.04, 11: 0.02})
    ex = vec.resized(R, pixel_integration='exact')
    g = torch.Generator(device=DEV).manual_seed(seed)
    xy = torch.rand(B, 2, generator=g, device=DEV) - 0.5
    z = 0.02 + 0.12 * torch.rand(B, generator=g, device=DEV)
    with torch.no_grad():
        img = ex(torch.cat([xy, z[:, None]], 1), torch.full((B,), 0.1, device=DEV))
        mu = photons * img[:, None].expand(-1, T, -1, -1) + 8.0
    rois = torch.poisson(mu, generator=g)
    tr = Tracks(rois, torch.ones(B, T, dtype=torch.bool, device=DEV), torch.arange(T, device=DEV)[None].expand(B, -1),
                torch.zeros(B, 2, dtype=torch.long, device=DEV), xy + (R - 1) / 2)
    return vec, tr, z


@needs_gpu
def test_fit_and_heldout_tracks_pixel_integration():
    """fit_tracks / heldout_tracks pixel_integration='exact': on data rendered with the exact pixel integral the exact
    model fits better than the midpoint rule (same frozen Zernikes / focus), and the fitted model keeps the setting"""
    from torch_psf.estimation import fit_tracks, heldout_tracks, paired_nll
    vec, tr, z = _exact_tracks()
    vec.focus_um = 0.1
    kw = dict(readnoise_e=1.0, zmax_um=0.2, iters=60, warmup=60, focus_um=0.1, fix_zernike=True, fix_focus=True)
    fe = fit_tracks(tr, vec, pixel_integration='exact', **kw)
    fb = fit_tracks(tr, vec, **kw)
    assert fe['psf'].pixel_integration == 'spectral' and fb['psf'].pixel_integration == 'box'
    assert fe['report']['nll_per_px'] < fb['report']['nll_per_px']
    hk = dict(readnoise_e=1.0, zmax_um=0.2, z_starts_um=(0.03, 0.12), iters=80)
    he = heldout_tracks(tr, vec, pixel_integration='exact', **hk)
    hb = heldout_tracks(tr, vec, **hk)
    p = paired_nll(he['nll'], hb['nll'])
    assert p['mean'] > 0 and p['z'] > 3, p
    assert vec.pixel_integration == 'box'                               # the caller's model is not changed
    assert np.median(np.abs(he['z'] - z.cpu().numpy())) < np.median(np.abs(hb['z'] - z.cpu().numpy())) + 0.005


@needs_gpu
def test_fit_tracks_spline_matches_exact_fit():
    """fit_tracks_spline (samples rendered by the integrated B-spline, knot gradient pulled back to the vectorial
    parameters) follows fit_tracks(pixel_integration='exact'): the same Zernikes, focus, blur and NLL"""
    from torch_psf.estimation import fit_tracks
    from torch_psf.pixint import fit_tracks_spline
    torch.backends.cudnn.allow_tf32 = False
    vec, tr, z = _exact_tracks(B=300)
    kw = dict(readnoise_e=1.0, zmax_um=0.2, iters=80, warmup=20, focus_um=0.1, blur='cov')
    fe = fit_tracks(tr, vec, pixel_integration='exact', **kw)
    fs = fit_tracks_spline(tr, vec, **kw)
    assert fs['psf'].pixel_integration == 'spectral'
    assert float((fs['psf'].zcoef - fe['psf'].zcoef).abs().max()) < 3e-4
    assert abs(fs['report']['focus_um'] - fe['report']['focus_um']) < 1e-3
    assert float((fs['psf'].blur - fe['psf'].blur).abs().max()) < 2e-3
    assert abs(fs['report']['nll_per_px'] - fe['report']['nll_per_px']) < 1e-5
    assert float((fs['z'] - fe['z']).abs().median()) < 1e-3


def test_pixel_integration_refuses_unknown(tmp_path):
    """an unknown pixel_integration (constructor, attribute, resized, a PSF json, a psf.zip) raises ValueError instead
    of rendering 'box'; None / absent = 'box', 'exact' = 'spectral'"""
    import json, zipfile
    from torch_psf import PSF
    from torch_psf.psfzip import save_zip
    kw = dict(npupil=16, oversample=2, device='cpu')
    for bad in ('trapezoid', 'Box', '', 3, ['box']):
        with pytest.raises(ValueError, match='pixel_integration'):
            VectorialPSF(9, pixel_integration=bad, **kw)
    v = VectorialPSF(9, pixel_integration=None, **kw)
    assert v.pixel_integration == 'box'
    assert VectorialPSF(9, pixel_integration='exact', **kw).pixel_integration == 'spectral'
    with pytest.raises(ValueError, match='pixel_integration'):
        v.pixel_integration = 'simpson'
    assert v.pixel_integration == 'box'
    with pytest.raises(ValueError, match='pixel_integration'):
        v.resized(9, pixel_integration='simpson')
    # files: absent = 'box', a known value is read, an unknown one is refused
    d = v.save(tmp_path / 'v.json')
    assert 'pixel_integration' not in d and PSF.load(tmp_path / 'v.json', device='cpu').pixel_integration == 'box'
    for val, ok in (('exact', 'spectral'), ('gauss_legendre', 'gauss_legendre'), ('trapezoid', None), (1, None)):
        json.dump(dict(d, pixel_integration=val), open(tmp_path / 'w.json', 'w'))
        if ok is None:
            with pytest.raises(ValueError, match='PSF file: unknown pixel_integration'):
                PSF.load(tmp_path / 'w.json', device='cpu')
        else:
            assert PSF.load(tmp_path / 'w.json', device='cpu').pixel_integration == ok
    zp = tmp_path / 'v.zip'
    save_zip(zp, VectorialPSF(9, pixel_integration='exact', z_range=(0.0, 0.4), **kw), zstack_planes=3)
    assert PSF.load(zp, device='cpu').pixel_integration == 'spectral'
    with zipfile.ZipFile(zp) as z:
        members = {n: z.read(n) for n in z.namelist()}
    meta = json.loads(members['psf.json'])
    assert meta['channels'][0]['model']['pixel_integration'] == 'spectral'
    meta['channels'][0]['model']['pixel_integration'] = 'trapezoid'
    members['psf.json'] = json.dumps(meta).encode()
    with zipfile.ZipFile(tmp_path / 'bad.zip', 'w') as z:
        for n, b in members.items():
            z.writestr(n, b)
    with pytest.raises(ValueError, match='unknown pixel_integration'):
        PSF.load(tmp_path / 'bad.zip', device='cpu')
