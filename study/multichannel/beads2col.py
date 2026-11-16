"""2-colour beads, both channels: MultiChannelBeadEstimator trained on beads.tif, beads2.tif held out.

  python -I beads2col.py --mode indep|joint|joint_shared|joint_poly2 [--iters 300] [--variant mixed] [--dry]

indep: every channel fitted on its own (bead positions free per channel; the mapping from the registration, then
refitted to the fitted bead positions); joint: shared bead positions, affine mapping fitted jointly, NA / apod per
channel; joint_shared: NA and apod one parameter for both channels; joint_poly2: an order-2 polynomial mapping;
joint_fixmap: shared positions with the registration mapping held fixed.
Global fit, then the field fit (field orders `variant`). Writes OUT/2col/<mode>/: report.json, psf_global.zip,
psf_field.zip, fit.npz (bead positions, per channel), and with --dry only the data summary and a detection figure."""
import os, sys, json, time, argparse, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *                                     # noqa: F401,F403
from torch_psf import MultiChannelBeadEstimator, PSF
from torch_psf.dataset import _plain

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='joint', choices=['indep', 'joint', 'joint_shared', 'joint_poly2', 'joint_fixmap'])
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--variant', default='mixed', choices=sorted(ORDERS))
ap.add_argument('--dry', action='store_true')
ap.add_argument('--device', default=None)
args = ap.parse_args()
dev = args.device or pick_gpu()
out = os.path.join(OUT, '2col', args.mode + ('' if args.variant == 'mixed' else '_' + args.variant))
os.makedirs(out, exist_ok=True)
st = load(FILES2)
start = PSF.load(START, device=dev)
kw = dict(DENSE)
est = MultiChannelBeadEstimator(regions=REGIONS, names=NAMES, start=start, optics=OPT, camera=CAM, device=dev,
                                iters=args.iters, fit_na=True, fit_apod=True, joint=args.mode != 'indep',
                                share=('NA', 'apod') if args.mode == 'joint_shared' else (),
                                mapping_order=2 if args.mode == 'joint_poly2' else 1,
                                fit_mapping=args.mode != 'joint_fixmap',
                                field=None if args.dry else dict(order=ORDERS[args.variant], prior_sd=0.05,
                                                                 iters=args.iters),
                                progress=(lambda stage, it, v, s: print(stage, it, round(v, 1), round(s, 1), flush=True)
                                          if it % 50 == 0 else None), **kw)
if args.dry:
    est.iters = 0
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    psf = est.estimate(st, DZ)
print('estimate s', time.time() - t0, flush=True)
rep = dict(_plain(est.report), mode=args.mode, variant=args.variant, iters=args.iters)
json.dump(rep, open(os.path.join(out, 'report.json'), 'w'), indent=1, default=str)
d = est.data
np.savez(os.path.join(out, 'data.npz'), xy_ref=d.xy_ref, stack=d.stack, cluster=d.cluster, usable=d.usable,
         cl_stack=d.cl_stack, amp=d.amp, faint=d.faint,
         masks=np.stack([np.stack(m) for m in est.masks]) if est.masks is not None else np.zeros(0),
         reg_ref=est.detections[0][0]['xy'], reg_res=np.zeros(0))
if args.dry:
    print(json.dumps(rep['data']), json.dumps(rep['registration'][1], default=str))
    sys.exit(0)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for tag, r in (('global', est.result), ('field', est.result_field)):
        if r is None:
            continue
        p = est.model_of(r)
        p.provenance = dict(estimator='MultiChannelBeadEstimator', mode=args.mode, stage=tag, report=rep)
        est.save_zip(os.path.join(out, f'psf_{tag}.zip'), psf=p, zstack_planes=25)
        pos = {}
        if 'parts' in r:
            pos['xy_channels'] = np.stack(r['xy_channels'])
        else:
            pos['xy_ref'] = r['xy_ref'].cpu().numpy()
            pos['xy_channels'] = np.stack([m.from_ref(pos['xy_ref']) for m in r['maps']])
        pos['train'] = r['train']
        np.savez(os.path.join(out, f'fit_{tag}.npz'), **pos)
print(json.dumps({k: rep[k] for k in ('global', 'field', 'chosen', 'intensity_ratio') if k in rep}, default=str)[:3000])
