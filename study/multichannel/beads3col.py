"""3-colour beads (beads1.tif training, beads2.tif held out), both camera halves: MultiChannelBeadEstimator (no field
stop: the halves are lit by a round illumination spot, not cut by a stop; few beads, so by default no field model).

  python -I beads3col.py [--mode joint|indep] [--iters 300] [--field none|astig] [--zb 0.0]

Writes OUT/3col/<mode>[_field][_zb]/report.json, psf_global.zip (psf_field.zip), fit_*.npz (bead positions, photons
per channel -> per-bead intensity ratios)."""
import os, sys, json, time, argparse, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *                                     # noqa: F401,F403
from torch_psf import MultiChannelBeadEstimator, PSF
from torch_psf.dataset import _plain

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='joint', choices=['indep', 'joint'])
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--field', default='none', choices=['none'] + sorted(ORDERS))
ap.add_argument('--zb', type=float, default=0.0, help='bead centre height (um)')
ap.add_argument('--device', default=None)
ap.add_argument('--max_renders', type=int, default=1500)
ap.add_argument('--rel', type=float, default=0.02, help='detection threshold relative to the brightest bead')
args = ap.parse_args()
dev = args.device or pick_gpu()
tag = args.mode + ('' if args.field == 'none' else '_' + args.field) + ('' if not args.zb else f'_zb{args.zb:g}')
out = os.path.join(OUT, '3col', tag)
os.makedirs(out, exist_ok=True)
st = load(FILES3)
start = PSF.load(START, device=dev)
kw = dict(DENSE, max_members=8, reject=None, max_renders=args.max_renders,
          rel_threshold=args.rel)   # chi2 grows with the photons: the brightest beads would be rejected
est = MultiChannelBeadEstimator(regions=REGIONS, names=NAMES, start=start, optics=OPT, camera=CAM, device=dev,
                                iters=args.iters, fit_na=True, fit_apod=True, joint=args.mode == 'joint',
                                field_stop=False, emitter_z_um=args.zb,
                                field=None if args.field == 'none' else dict(order=ORDERS[args.field], prior_sd=0.05,
                                                                             iters=args.iters),
                                progress=(lambda stage, it, v, s: print(stage, it, round(v, 1), round(s, 1), flush=True)
                                          if it % 50 == 0 else None), **kw)
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    psf = est.estimate(st, DZ)
print('estimate s', time.time() - t0, flush=True)
rep = dict(_plain(est.report), mode=args.mode, field_variant=args.field, iters=args.iters, zb=args.zb)
json.dump(rep, open(os.path.join(out, 'report.json'), 'w'), indent=1, default=str)
d = est.data
np.savez(os.path.join(out, 'data.npz'), xy_ref=d.xy_ref, stack=d.stack, cluster=d.cluster, usable=d.usable,
         cl_stack=d.cl_stack, amp=d.amp, faint=d.faint)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for t, r in (('global', est.result), ('field', est.result_field)):
        if r is None:
            continue
        p = est.model_of(r)
        p.provenance = dict(estimator='MultiChannelBeadEstimator', mode=args.mode, stage=t, report=rep)
        est.save_zip(os.path.join(out, f'psf_{t}.zip'), psf=p, zstack_planes=25)
        ph = np.full((d.n_beads, 2), np.nan)
        for c in range(2):
            if r['photons'][c] is not None:
                ph[r['gid'][c].cpu().numpy(), c] = r['photons'][c].sum(1).cpu().numpy()
        pos = dict(photons=ph, train=r['train'])
        if 'parts' in r:
            pos['xy_channels'] = np.stack(r['xy_channels'])
        else:
            pos['xy_ref'] = r['xy_ref'].cpu().numpy()
            pos['xy_channels'] = np.stack([m.from_ref(pos['xy_ref']) for m in r['maps']])
        np.savez(os.path.join(out, f'fit_{t}.npz'), **pos)
print(json.dumps({k: rep[k] for k in ('registration', 'data', 'global', 'field', 'chosen', 'intensity_ratio') if k in rep},
                 default=str)[:4000])
