"""Critic test 4: is the y-vs-z tilt of the free pupil on movie patches (doc section 3.4) caused by the +-1.5 px
offset clamp of ConvCubicSplinePSF / localize? The detector centres high-z spots up to 1.7 px (mean) away from the true
emitter (movie_diag.py), beyond the clamp. Free pupil fit (Zernike start = truth, 200 its, as fit_pupil_insitu) on
the scenario A movie patches with the default xmax 1.5 px and with xmax 3.0 px; scored on the 6000 test emitters.
Run from the repo root:  python study/insitu/critic/movie_fits.py MOVIE_A.pt [xmax ...]"""
import sys, time, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf import ConvCubicSplinePSF
from torch_psf.fit import localize

D = torch.load(sys.argv[1], weights_only=False)
rois = D['rois'].to(C.DEV)
xmaxs = [float(a) for a in sys.argv[2:]] or [1.5, 3.0]
zt = C.zc_insitu().to(C.DEV)
test = C.sim_test('A', 6000)


def score(spl, K, nll_true, name):
    r = F.evaluate(spl, K, test, nll_true)
    e = r['th'] - test['th']; z = test['th'][:, 2]
    good = e[:, 2].abs() < 0.15                                           # mask of run_clean / run_movie_clean
    b = torch.linspace(0.05, 0.75, 8, device=C.DEV); ib = torch.bucketize(z, b).clamp(1, 7) - 1
    cy = [float(e[good & (ib == i), 1].mean()) * 108 for i in range(7)]
    cyo = [float(e[r['ok'] & (ib == i), 1].mean()) * 108 for i in range(7)]   # mask of run_study / run_pupil
    print(f'  {name:34s} dnll {r["dnll"]:6.3f}  mean y err per z bin (|dz|<150 nm mask): ' + ' '.join(f'{v:+5.1f}' for v in cy)
          + f'   (ok mask: {min(cyo):+.1f}..{max(cyo):+.1f}; failures {r["outlier"]:.3f})', flush=True)


for xm in xmaxs:
    spl = ConvCubicSplinePSF(C.P, C.ZR, oversample=2, dz_um=0.04, zpad=5, xy_limit=xm, device=C.DEV)
    Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, 'A', 'insitu'), C.insitu_pos)
    spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
    print(f'xmax {xm} px (knot ROI {spl.Pl} px)', flush=True)
    score(spl, Ktrue, nll_true, 'true PSF')
    spl.knots = Ktrue
    th, _ = localize(spl, rois)
    print(f'  movie patches localized with the TRUE PSF: fraction at the xy clamp {((th[:, :2].abs() > xm - 0.06).any(1)).float().mean():.4f}; '
          f'mean y error vs truth for true z > 0.6: {float((th[:, 1] - D["off"][:, 1].to(C.DEV))[D["z"].to(C.DEV) > 0.6].mean()) * 108:+.1f} nm', flush=True)
    t = time.time()
    Kp, prm, _ = F.fit_pupil_insitu(spl, rois, zt, iters=200)
    score(spl, Kp, nll_true, f'free pupil on movie ({time.time() - t:.0f} s)')
