"""Phase diversity: the object-eliminated (Gonsalves) metric, recovery of Zernikes / NA / pupil apodization from a
simulated stack of overlapping beads, a generic differentiable PSF stack, the estimator API, and fit_apod in the bead
fitters. Small enough for a CPU."""
import math
import numpy as np
import torch
from conftest import DEV, needs_gpu
from torch_psf import VectorialPSF, PhaseDiversityEstimator, DenseBeadEstimator, ZStackEstimator
from torch_psf.estimation import stage_focus
from torch_psf.estimation.dense_beads import simulate_dense_bead_stack
from torch_psf.estimation.phase_diversity import (PDPatches, fit_phase_diversity, gonsalves_metric, object_estimate,
                                                  psf_otf, rfft_weights, VectorialStack, CallableStack,
                                                  radial_amplitude)

OPT = dict(pixel_um=0.1, NA=1.49, wavelength_um=0.67)
PUP = dict(npupil=32, oversample=2)
TRUE = {5: 0.04, 6: 0.03, 7: -0.03, 8: 0.02, 11: 0.03}
# 6 beads in a 64 x 64 field: a pair 0.9 um apart, a triplet, one alone (all PSFs overlap when defocused)
XY = np.array([[18.3, 20.6], [26.9, 23.1], [44.2, 18.7], [40.6, 43.4], [49.8, 47.2], [20.4, 46.1]])


def _truth(NA=1.40, apod=(-0.6, 0.0, 0.0), R=31):
    v = VectorialPSF(R, device=DEV, **OPT, **PUP).set_zernike(TRUE)
    with torch.no_grad():
        v.NA.fill_(NA)
        v.apod.copy_(torch.tensor(apod))
    return v


def _sim(truth, nk=21, seed=0, photons=40000.0):
    focus = stage_focus(nk, 1.2 / (nk - 1), 1.0).to(DEV)
    tt = np.linspace(-1, 1, nk)
    sim = simulate_dense_bead_stack(truth, focus, XY, 64, 64, photons=photons, bg=30.0, seed=seed,
                                    drift=np.stack([0.3 * tt, -0.2 * tt ** 2], 1))
    return focus, sim


def _zerr(psf, truth):
    return float((psf.zcoef.detach() - truth.zcoef.detach()).pow(2).mean().sqrt())


def _amp_err(psf, truth):
    """rms difference of the radial pupil transmission over the physical pupil coordinate 0 .. 1.49"""
    s = np.linspace(0, 1.49, 150)
    return float(np.sqrt(np.mean((radial_amplitude(psf, s)[1] - radial_amplitude(truth, s)[1]) ** 2)))


def test_metric_eliminates_the_object():
    """For noise-free data D_k = O H_k the reduced metric is ~0 at the true PSF whatever the object, it equals the
    least-squares error at the closed-form object, and its minimum over a Zernike stays at the truth for different
    objects."""
    truth = _truth(1.49, (0, 0, 0))
    vs = VectorialStack(truth, 31, oversample=2)
    G = 64
    focus = stage_focus(15, 0.1, 1.0).to(DEV)
    wf = rfft_weights(G, DEV)
    g = torch.Generator(device='cpu').manual_seed(1)
    objs = [torch.rand(G, G, generator=g) * (torch.rand(G, G, generator=g) > 0.995),      # sparse points
            torch.nn.functional.avg_pool2d(torch.rand(1, G, G, generator=g), 5, 1, 2)[0]]  # a smooth texture
    with torch.no_grad():
        Ht = psf_otf(vs(focus[None]), G)[0]
        grid = np.linspace(-0.02, 0.02, 9) + TRUE[6]
        for O in objs:
            D = torch.fft.rfft2(O.to(DEV))[None] * Ht
            ref = float((wf * (D.real ** 2 + D.imag ** 2)).sum())
            e_true = float(gonsalves_metric(D, Ht, wf=wf))
            assert e_true < 1e-5 * ref, (e_true, ref)
            # the reduced form = the LS error at the closed-form object (with a regularisation eps)
            eps = 1e-3 * float((Ht[:, 0, 0].abs() ** 2).sum())
            H2 = psf_otf(VectorialStack(_truth(1.49, (0, 0, 0)).set_zernike({6: 0.05}), 31, 2)(focus[None]), G)[0]
            Ob = object_estimate(D, H2, eps=eps)
            ls = float((wf * ((D - H2 * Ob).abs() ** 2).sum(0)).sum() + eps * (wf * Ob.abs() ** 2).sum())
            red = float(gonsalves_metric(D, H2, eps=eps, wf=wf))
            assert abs(ls - red) < 1e-3 * abs(red), (ls, red)
            # scan Z6: the minimum is at the truth for every object
            es = []
            for z6 in grid:
                v = _truth(1.49, (0, 0, 0)).set_zernike({6: float(z6)})
                es.append(float(gonsalves_metric(D, psf_otf(VectorialStack(v, 31, 2)(focus[None]), G)[0], wf=wf)))
            assert int(np.argmin(es)) == 4, es


@needs_gpu
def test_recovers_zernikes_na_and_apodization():
    """Overlapping beads with NA 1.40 (nominal 1.49) and an apodized pupil: phase diversity from the nominal start
    recovers the Zernikes, NA and the pupil transmission; without NA / apod in the fit the Zernikes are biased."""
    truth = _truth()
    focus, sim = _sim(truth)
    pt = PDPatches.from_stacks([sim['d']], 64)
    start = VectorialPSF(21, device=DEV, **OPT, **PUP)
    kw = dict(render_size=31, oversample=2, ls_iters=150, iters=250)
    r = fit_phase_diversity(pt, focus, start, fit=('zcoef', 'NA', 'apod'), **kw)
    p = r['psf']
    e_z, e_na, e_a = _zerr(p, truth), abs(float(p.NA) - 1.40), _amp_err(p, truth)
    assert e_z < 0.003, e_z                                    # a few milli-waves rms
    assert e_na < 0.02, float(p.NA)
    assert e_a < 0.05, (e_a, p.apod)
    assert 0.85 < r['report']['deviance_per_px'] < 1.1, r['report']
    assert abs(float(r['drift'][0, -1, 0] - r['drift'][0, 0, 0]) - 0.6) < 0.1     # the drift is followed
    r0 = fit_phase_diversity(pt, focus, start, fit=('zcoef',), **{**kw, 'ls_iters': 60, 'iters': 60})
    assert r0['report']['deviance_per_px'] > r['report']['deviance_per_px'] + 0.05
    assert _amp_err(r0['psf'], truth) > 3 * e_a


def test_generic_psf_stack():
    """Any differentiable PSF stack: an astigmatic Gaussian whose two focal planes are fitted."""
    R, nk = 21, 21
    focus = stage_focus(nk, 1.2 / (nk - 1), 1.0).to(DEV)
    yy, xx = torch.meshgrid(torch.arange(R, device=DEV) - R // 2, torch.arange(R, device=DEV) - R // 2,
                            indexing='ij')

    def gauss(c):
        def fn(f):
            sx = 1.3 * torch.sqrt(1 + ((f - c[0]) / 0.35) ** 2)
            sy = 1.3 * torch.sqrt(1 + ((f - c[1]) / 0.35) ** 2)
            g = torch.exp(-xx ** 2 / (2 * sx[..., None, None] ** 2) - yy ** 2 / (2 * sy[..., None, None] ** 2))
            return g / g.sum((-2, -1), keepdim=True)
        return fn
    true_c = torch.tensor([-0.2, 0.25], device=DEV)
    h = gauss(true_c)(focus[None])[0]                                    # [nk, R, R]
    O = torch.zeros(64, 64, device=DEV)
    for x, y in XY:
        O[int(round(y)), int(round(x))] = 30000.0
    G = 64 + R - 1
    Hk = psf_otf(h, G)
    img = torch.fft.irfft2(torch.fft.rfft2(torch.nn.functional.pad(O, (R // 2,) * 2 + (R // 2,) * 2))[None] * Hk,
                           s=(G, G))[:, R // 2:R // 2 + 64, R // 2:R // 2 + 64]
    d = torch.poisson(img.clamp_min(0) + 20.0, generator=torch.Generator(device=DEV).manual_seed(0))
    pt = PDPatches(d[None], torch.ones(1, 64, 64, device=DEV), [(0, 0)], [0])
    c = torch.zeros(2, device=DEV)
    model = CallableStack(gauss(c), dict(c=c), R)
    r = fit_phase_diversity(pt, focus, model, fit=('c',), ls_iters=150, iters=100, lr=1e-2, focus_offset=None,
                            drift=None)
    assert torch.allclose(c, true_c, atol=0.02), c


@needs_gpu
def test_estimator_api_heldout_and_field():
    truth = _truth(1.49, (0, 0, 0))
    focus, sim = _sim(truth, nk=15, seed=0)
    _, sim2 = _sim(truth, nk=15, seed=1)
    est = PhaseDiversityEstimator(optics=OPT, patch=32, field_stop=False, holdout=1, ls_iters=40, iters=40,
                                  per_patch=True, patch_iters=10, render_size=25, oversample=2, min_signal=0,
                                  npupil=32, oversample_pupil=2, device=DEV)
    psf = est.estimate([sim['d'], sim2['d']], 1.2 / 14)
    rep = est.report
    assert isinstance(psf, VectorialPSF) and psf.roisize == 21
    assert rep['n_patches'] == 4 and rep['heldout_n_patches'] == 4
    assert 0.8 < rep['heldout_deviance_per_px'] < 1.3, rep
    assert len(est.field) == 4 and {'center', 'zcoef', 'NA', 'apod', 'deviance_per_px'} <= set(est.field[0])
    assert _zerr(psf, truth) < 0.01


def test_fit_apod_in_bead_fitters():
    truth = _truth()
    focus, sim = _sim(truth, nk=15)
    de = DenseBeadEstimator(optics=OPT, patch=15, render_pad=5, iters=40, warmup=5, field_stop=False, fit_apod=True,
                            device=DEV, npupil=32, oversample_pupil=2)
    p = de.estimate([sim['d']], 1.2 / 14)
    assert 'apod' in de.report and float(p.apod.abs().sum()) > 0
    ze = ZStackEstimator('vectorial', optics=OPT, roisize=15, nbeads=2, iters=20, fit_apod=True, device=DEV,
                         npupil=32, oversample_pupil=2)
    p2 = ze.estimate([sim['d'].cpu().numpy()], 1.2 / 14)
    assert 'apod' in ze.report and float(p2.apod.abs().sum()) > 0
