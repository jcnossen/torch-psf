"""Phase diversity on the 2-colour bead stacks (left channel): trained on beads.tif, the PSF scored on the held-out
beads2.tif clusters by dense_nll (the dense-beads procedure; dense fit: 1.441).

  python -I real_pd.py OUT_DIR --variants base fit_na fit_apod fit_na_apod [--patch 64] [--plane-step 1] ...
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from torch_psf import PhaseDiversityEstimator

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--variants', nargs='+', default=['base', 'fit_na', 'fit_apod', 'fit_na_apod'])
ap.add_argument('--patch', type=int, default=64)
ap.add_argument('--step', type=int, default=None)
ap.add_argument('--plane-step', type=int, default=1)
ap.add_argument('--ls-iters', type=int, default=200)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--render', type=int, default=41)
ap.add_argument('--method', default='poisson')
ap.add_argument('--tag', default='')
ap.add_argument('--emitter-z', type=float, default=0.0)
ap.add_argument('--bg', default='const')
ap.add_argument('--min-signal', type=float, default=8.0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
VAR = dict(ls_na_apod=dict(fit_na=True, fit_apod=True, method='ls'), base={}, fit_na=dict(fit_na=True), fit_apod=dict(fit_apod=True), fit_na_apod=dict(fit_na=True, fit_apod=True),
           fit_na_apod_pupil=dict(fit_na=True, fit_apod=True, fit_pupil=True),
           fit_na_apod_dz=dict(fit_na=True, fit_apod=True, fit_dz_scale=True))
st = load()
data, focus, ho = clusters(st)
hod = data.subset(ho)
res_path = os.path.join(args.out, f'real_pd{args.tag}.json')
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
for v in args.variants:
    t0 = time.time()
    est = PhaseDiversityEstimator(optics=OPT, patch=args.patch, step=args.step, holdout=1, plane_step=args.plane_step,
                                  ls_iters=args.ls_iters, iters=args.iters, render_size=args.render,
                                  **({'method': args.method} if 'method' not in VAR[v] else {}),
                                  emitter_z_um=args.emitter_z, bg=args.bg, min_signal=args.min_signal, **VAR[v])
    psf = est.estimate(st, DZ)
    t_fit = time.time() - t0
    psf.save(os.path.join(args.out, f'psf_pd_{v}{args.tag}.json'))
    s, _ = score(hod, focus, psf, emitter_z_um=args.emitter_z)
    rep = {k: x for k, x in est.report.items()}
    res[v] = dict(report=rep, zcoef=psf.zcoef.detach().cpu().tolist(), NA=float(psf.NA),
                  apod=psf.apod.detach().cpu().tolist(), heldout=s, fit_seconds=t_fit,
                  deviance_patch=est.result['deviance'].tolist(), focus_offset=est.result['focus_offset'].cpu().tolist(),
                  corners=est.patches.corner[est.train].tolist())
    print(v, json.dumps(dict(heldout=s, NA=float(psf.NA), apod=res[v]['apod'],
                             z=[round(z * 1e3, 1) for z in res[v]['zcoef'][:13]],
                             pd_dev=rep.get('deviance_per_px'), pd_heldout_dev=rep.get('heldout_deviance_per_px'),
                             n_patches=rep['n_patches'], seconds=t_fit)), flush=True)
    json.dump(res, open(res_path, 'w'), indent=1)
