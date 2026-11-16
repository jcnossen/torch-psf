"""DenseBeadEstimator on the 2-colour bead stacks with NA / apod fitting (settings of study/dense_beads/real_variants.py),
trained on beads.tif; every PSF scored on the held-out beads2 clusters by dense_nll (as real_fit.py: dense 1.441).

  python -I real_dense.py OUT_DIR [--variants fit_na fit_apod fit_na_apod]
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from torch_psf import DenseBeadEstimator, PSF

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--variants', nargs='+', default=['fit_na', 'fit_apod', 'fit_na_apod'])
ap.add_argument('--iters', type=int, default=300)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
VAR = dict(base={}, fit_na=dict(fit_na=True), fit_apod=dict(fit_apod=True), fit_na_apod=dict(fit_na=True, fit_apod=True),
           height50_na_apod=dict(fit_na=True, fit_apod=True, emitter_z_um=0.05))
st = load()
data, focus, ho = clusters(st)
hod = data.subset(ho)
res_path = os.path.join(args.out, 'real_dense.json')
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
if 'base' not in res:
    p = PSF.load(os.path.expanduser('~/densebeads_out/real/psf_dense.json'), device='cuda')
    s, _ = score(hod, focus, p)
    res['base'] = dict(heldout=s, NA=1.49, apod=[0, 0, 0], zcoef=p.zcoef.detach().cpu().tolist(), seconds=330.3,
                       note='psf_dense.json of the dense-beads study')
    print('base', json.dumps(s), flush=True)
for v in args.variants:
    t0 = time.time()
    kw = dict(optics=OPT, iters=args.iters, device='cuda', **{k: x for k, x in DENSE.items()})
    kw.update(VAR[v])
    est = DenseBeadEstimator(**kw)
    psf = est.estimate(st, DZ)
    t_fit = time.time() - t0
    psf.save(os.path.join(args.out, f'psf_dense_{v}.json'))
    s, _ = score(hod, focus, psf, emitter_z_um=kw.get('emitter_z_um', 0.0))
    res[v] = dict(heldout=s, NA=float(psf.NA), apod=psf.apod.detach().cpu().tolist(),
                  zcoef=psf.zcoef.detach().cpu().tolist(), seconds=t_fit,
                  report={k: x for k, x in est.report.items() if not isinstance(x, (list, dict))})
    print(v, json.dumps(dict(heldout=s, NA=res[v]['NA'], apod=res[v]['apod'],
                             z=[round(z * 1e3, 1) for z in res[v]['zcoef'][:13]], seconds=t_fit)), flush=True)
    json.dump(res, open(res_path, 'w'), indent=1)
