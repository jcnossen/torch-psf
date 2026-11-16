"""Field-dependent PSF on the 2-colour bead stacks (left channel): DenseBeadEstimator / PhaseDiversityEstimator with
field= (a polynomial of the field position per Zernike, fitted jointly with the base Zernikes, NA and apod), trained
on beads.tif; the global (field-constant) and the field PSF of the same run scored on the held-out beads2.tif clusters
by dense_nll exactly as study/phase_diversity/common.py (the dense-beads procedure; the global NA + apod dense fit gave
1.3375 deviance / px there, phase diversity 1.3395). Data are read as plain arrays (tifffile).

  python -I beads_field.py OUT_DIR --method dense|pd [--variant mixed|linear|quad] [--prior 0.05] [--iters 300]
Writes OUT_DIR/beads_field.json (one entry per method_variant) and the PSFs (psf.zip).
"""
import os, sys, json, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'phase_diversity'))
from common import *                                   # noqa: F401,F403  (load, clusters, score, OPT, DENSE, DZ)
from torch_psf import DenseBeadEstimator, PhaseDiversityEstimator, PSF, io
from torch_psf.field import FieldPSF

ORDERS = dict(
    mixed={5: 2, 6: 2, 11: 2, 7: 1, 8: 1, 9: 1, 10: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1},
    linear={j: 1 for j in range(5, 22)},
    quad={j: 2 for j in range(5, 22)},
    astig={5: 2, 6: 2, 11: 2})
START = dict(dense='~/phasediv_out/real/psf_dense_fit_na_apod.json', pd='~/phasediv_out/real/psf_pd_fit_na_apod_long.json')

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--method', default='dense', choices=['dense', 'pd'])
ap.add_argument('--variant', default='mixed', choices=sorted(ORDERS))
ap.add_argument('--prior', type=float, default=0.05)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--tag', default='')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
st = load()
data, focus, ho = clusters(st)
hod = data.subset(ho)
spec = dict(order=ORDERS[args.variant], prior_sd=args.prior, iters=args.iters)
start = PSF.load(os.path.expanduser(START[args.method]), device='cuda')
t0 = time.time()
if args.method == 'dense':
    est = DenseBeadEstimator(optics=OPT, start=start, iters=args.iters, fit_na=True, fit_apod=True, field=spec,
                             device='cuda', **DENSE)
else:
    est = PhaseDiversityEstimator(optics=OPT, start=start, patch=64, holdout=1, fit_na=True, fit_apod=True,
                                  ls_iters=0, iters=args.iters, field=spec, device='cuda')
psf = est.estimate(st, DZ)
t_fit = time.time() - t0
if args.method == 'dense':
    glob = est.result['psf'].resized(DENSE['patch']); glob.focus_um = 0.0
    fpsf = FieldPSF(est.result_field['psf'].resized(DENSE['patch']), est.result_field['field'])
else:
    glob = est.result['psf']; glob.focus_um = 0.0
    fpsf = FieldPSF(est.result_field['psf'], est.result_field['field'])
fpsf.base.focus_um = 0.0
s_g, r_g = score(hod, focus, glob)
s_f, r_f = score(hod, focus, fpsf)
npx = np.diff(hod.cl_pix).astype(float) * hod.nk
dnll = (r_f['nll'] - r_g['nll']) / npx                     # per cluster, NLL per px (negative: field better)
cxy = np.array([np.mean(c['xy'], 0) for c in hod.clusters])
key = f'{args.method}_{args.variant}{args.tag}'
path = os.path.join(args.out, 'beads_field.json')
res = json.load(open(path)) if os.path.exists(path) else {}
res[key] = dict(method=args.method, variant=args.variant, orders=ORDERS[args.variant], prior_sd=args.prior,
                iters=args.iters, seconds=t_fit, heldout_global=s_g, heldout_field=s_f,
                dnll_per_px=float(s_f['nll_per_px'] - s_g['nll_per_px']),
                cluster_dnll_per_px=dnll.tolist(), cluster_xy=cxy.tolist(),
                clusters_better=int((dnll < 0).sum()), n_clusters=int(len(dnll)),
                global_psf=dict(NA=float(glob.NA), apod=glob.apod.detach().cpu().tolist(),
                                zcoef=glob.zcoef.detach().cpu().tolist(), noll=list(glob.zidx)),
                field_psf=dict(NA=float(fpsf.base.NA), apod=fpsf.base.apod.detach().cpu().tolist(),
                               zcoef=fpsf.base.zcoef.detach().cpu().tolist(), block=fpsf.field_block()),
                report={k: v for k, v in est.report.items() if k in ('heldout_deviance_per_px', 'heldout_nll_per_px',
                                                                       'deviance_per_px', 'nll_per_px', 'field',
                                                                       'chosen', 'NA', 'apod', 'seconds')})
json.dump(res, open(path, 'w'), indent=1)
io.save_zip(os.path.join(args.out, f'psf_{key}.zip'), fpsf, zstack_planes=25)
print(key, json.dumps(dict(global_=s_g, field=s_f, dnll=res[key]['dnll_per_px'], better=res[key]['clusters_better'],
                           n=len(dnll), NA=[float(glob.NA), float(fpsf.base.NA)], seconds=t_fit)), flush=True)
