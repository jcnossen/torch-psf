"""Critic test 5: scenario A without the inverse crime. Data (training and test, clean patches) are rendered with a
finer vectorial model (pupil 128, oversampling 4) and optionally a sample index 1.35 instead of 1.33; the fitted model
is the study's (pupil 64, oversampling 3, n 1.33, Noll 5-21). Zernike in-situ fit (route K, 60 its, start = true
Zernikes), scored against the true PSF of the data.
Run from the repo root:  python study/insitu/critic/inverse_crime.py"""
import sys, time, math, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf import localize

NTR, NTE = 8000, 4000
zt = C.zc_insitu().to(C.DEV)
OPT0 = dict(C.OPT)


def sim(N, seed, opt):
    C.OPT.update(opt)
    psf = C.make_vec(C.P, 'A', 'insitu')
    g = torch.Generator(device=C.DEV).manual_seed(900 + seed)
    r = lambda *s: torch.rand(*s, generator=g, device=C.DEV)
    th = torch.stack([r(N) - 0.5, r(N) - 0.5, 0.05 + 0.7 * r(N),
                      torch.exp(math.log(4000.0) + 0.4 * torch.randn(N, generator=g, device=C.DEV)),
                      torch.full((N,), 30.0, device=C.DEV)], 1)
    mu = th[:, 3, None, None] * C.render(psf, th[:, :3], torch.full((N,), C.F0, device=C.DEV), nb=200) + 30.0
    C.OPT.clear(); C.OPT.update(OPT0)
    return dict(d=torch.poisson(mu, generator=g), th=th)


for label, opt in (('control: data 64/3 (inverse crime)', dict()),
                   ('data 128/4', dict(npupil=128, oversample=4)),
                   ('data 128/4, n_med 1.35', dict(npupil=128, oversample=4, n_med=1.35))):
    spl = F.new_spline()
    C.OPT.update(opt)
    Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, 'A', 'insitu'), C.insitu_pos)
    C.OPT.clear(); C.OPT.update(OPT0)
    test, train = sim(NTE, 1, opt), sim(NTR, 2, opt)
    spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
    t = time.time()
    zc, _ = F.fit_vec_insitu(spl, train['d'], zt, iters=60)
    K = F.knots_from(spl, zc)
    for name, KK in (('true PSF of the data', Ktrue), (f'Zernike in situ ({time.time() - t:.0f} s)', K)):
        r = F.evaluate(spl, KK, test, nll_true)
        e = r['th'] - test['th']; z = test['th'][:, 2]; good = e[:, 2].abs() < 0.15
        b = torch.linspace(0.05, 0.75, 8, device=C.DEV); ib = torch.bucketize(z, b).clamp(1, 7) - 1
        cy = [float(e[good & (ib == i), 1].mean()) * 108 for i in range(7)]
        cz = [float(e[good & (ib == i), 2].mean()) * 1000 for i in range(7)]
        print(f'{label:36s} {name:28s} dnll {r["dnll"]:6.3f} failures {r["outlier"]:.3f} z rms {r["z_rmse"]:5.1f} '
              f'y err over z {min(cy):+5.1f}..{max(cy):+5.1f}  z err over z {min(cz):+5.1f}..{max(cz):+5.1f} nm  '
              f'zc rms diff to truth {float((zc - zt).pow(2).mean().sqrt()):.4f} waves', flush=True)
