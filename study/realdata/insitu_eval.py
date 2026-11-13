"""Step 4a: localize the TEST patches (all flags == 0 spots of the odd 1000-frame blocks) with each fitted PSF, same
localizer for all (3 starting heights, Levenberg-Marquardt, +-3 px, z +-0.8 um), and store per spot: theta, Poisson
NLL, chi2/pixel (variance 2 mu), deviance of the halved data (EMCCD), CRLB x sqrt(2).
usage: python insitu_eval.py tag1 tag2 ...   -> insitu/eval_<tag>.npz"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np, torch
from insitu_common import *

spl, _ = movie_spline()
dte = torch.from_numpy(np.load(f'{OUT}/patches_test.npy')).to(DEV)
for tag in sys.argv[1:]:
    t0 = time.time()
    st = torch.load(f'{OUT}/psf_{tag}.pt', weights_only=False)
    assert st['nz'] == spl.nz and st['c'] == spl.c
    spl.knots = st['knots'].to(DEV)
    th, nll = localize_full(spl, dte)
    c2, dev, cr = fit_stats(spl, dte, th)
    lim = limit_fractions(th)
    np.savez(f'{OUT}/eval_{tag}.npz', th=th.cpu().numpy(), nll=nll.cpu().numpy(), chi2=c2.cpu().numpy(),
             dev=dev.cpu().numpy(), crlb=cr.cpu().numpy(), seconds=time.time() - t0)
    print(f'{tag}: {len(dte)} test spots, dev/spot {float(dev.mean()):.4f}, chi2 mean {float(c2.mean()):.4f} median '
          f'{float(c2.median()):.4f}, limits {lim}, {time.time() - t0:.0f}s', flush=True)
