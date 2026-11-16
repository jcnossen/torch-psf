"""Held-out scores of the 2-colour multi-channel PSFs (the zips of beads2col.py) on the beads2.tif clusters, per
channel, with two refit protocols:
  indep: every channel refitted on its own (bead positions free per channel; PSF fixed);
  joint: both channels together with shared bead positions and focus offsets (PSFs, mapping and channel focus fixed;
         for independently trained models the channel focus is fitted, the mapping is the one refitted to their bead
         positions).
Each cluster's own terms are refitted (200 iterations, as dense_nll). Also the mapping residuals (bead positions of
the independent fits through the registration / fitted maps) and the figures.

  python -I score2col.py [--modes indep,joint,joint_shared,joint_poly2] -> OUT/2col/score2col.json"""
import os, sys, json, time, argparse, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *                                     # noqa: F401,F403
from torch_psf.estimation.multichannel_beads import multichannel_nll

ap = argparse.ArgumentParser()
ap.add_argument('--modes', default='indep,joint,joint_shared,joint_poly2,joint_fixmap')
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--device', default=None)
ap.add_argument('--out', default='score2col.json')
args = ap.parse_args()
dev = args.device or pick_gpu()
base = os.path.join(OUT, '2col')
t0 = time.time()
data, focus, maps_reg, reg, cs, masks = build_data(FILES2, REGIONS, dev)
ho = np.nonzero(data.cl_stack == 1)[0]
dho = data.subset(ho)
print('data', data.summary(), 'held-out', dho.summary(), round(time.time() - t0, 1), flush=True)
res = dict(data=data.summary(), heldout=dho.summary(), registration=reg[1]['rms_px'], models={})


def per_cluster(r, d):
    """per channel: {global held-out cluster id (index into ho): (nll, npx, deviance, chi2)}"""
    out = []
    for c in range(d.n_channels):
        s = r['channel'][c]
        if s is None:
            out.append({})
            continue
        out.append({int(g): (float(s['nll'][k]), float(s['npx'][k]), float(s['deviance'][k]), float(s['chi2'][k]))
                    for k, g in enumerate(d.cl_gid[c])})
    return out


models = {}
for mode in args.modes.split(','):
    for stage in ('global', 'field'):
        p = os.path.join(base, mode, f'psf_{stage}.zip')
        if os.path.exists(p):
            models[f'{mode}/{stage}'] = zip_model(p, dev)
for key, (psfs, maps, cf) in models.items():
    t1 = time.time()
    entry = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # indep protocol: one channel at a time
        pc = [{}, {}]
        for c in range(2):
            dc = dho.channel(c)
            r = multichannel_nll(dc, focus, [psfs[c]], [maps[c]], iters=args.iters)
            loc = per_cluster(r, dc)[0]
            ids = np.nonzero(dho.usable[:, c])[0]                    # dc cluster k = dho cluster ids[k]
            pc[c] = {int(ids[k]): v for k, v in loc.items()}
        entry['indep'] = pc
        # joint protocol
        indep_trained = key.startswith('indep')
        r = multichannel_nll(dho, focus, list(psfs), maps, iters=args.iters, channel_focus=cf,
                             fit_channel_focus=indep_trained)
        entry['joint'] = per_cluster(r, dho)
        entry['joint_channel_focus'] = None if r['channel_focus'] is None else r['channel_focus'].cpu().tolist()
    models[key] = (psfs, maps, cf, entry)
    print(key, 'scored', round(time.time() - t1, 1), flush=True)

# aggregates: held-out clusters with chi2 > 3 x the median in any channel under the indep/global model (as stage 2a)
ref = models.get('indep/global', next(iter(models.values())))[3]['indep']
bad = set()
for c in range(2):
    c2 = np.array([v[3] for v in ref[c].values()])
    med = np.median(c2)
    bad |= {k for k, v in ref[c].items() if v[3] > 3 * med}
res['aggregates'] = sorted(int(b) for b in bad)


def summ(pc, keep):
    nll = sum(pc[k][0] for k in keep)
    npx = sum(pc[k][1] for k in keep)
    dev_ = sum(pc[k][2] * pc[k][1] for k in keep)
    return dict(deviance_per_px=dev_ / npx, nll_per_px=nll / npx, n_clusters=len(keep))


for key, (psfs, maps, cf, entry) in models.items():
    out = {}
    for prot in ('indep', 'joint'):
        for c in range(2):
            pc = entry[prot][c]
            keys_all = sorted(pc)
            keys_clean = [k for k in keys_all if k not in bad]
            out[f'{prot}_ch{c}_all'] = summ(pc, keys_all)
            out[f'{prot}_ch{c}_clean'] = summ(pc, keys_clean)
    out['channel_focus'] = cf
    out['joint_channel_focus'] = entry['joint_channel_focus']
    out['mapping'] = maps[1].describe()
    res['models'][key] = out
# paired differences vs indep/global (clean clusters, NLL per px)
for prot in ('indep', 'joint'):
    for c in range(2):
        r0 = models['indep/global'][3][prot][c] if 'indep/global' in models else None
        if r0 is None:
            continue
        keys = [k for k in sorted(r0) if k not in bad]
        for key, (_, _, _, entry) in models.items():
            pc = entry[prot][c]
            d = np.array([(pc[k][0] - r0[k][0]) / pc[k][1] for k in keys])
            res['models'][key][f'{prot}_ch{c}_vs_indep_global'] = dict(
                dnll_per_px=float(sum(pc[k][0] - r0[k][0] for k in keys) / sum(pc[k][1] for k in keys)),
                better=int((d < 0).sum()), n=len(d))

# mapping residuals: bead positions of the independent fits through the maps
mr = {}
for stage in ('global', 'field'):
    f = os.path.join(base, 'indep', f'fit_{stage}.npz')
    if not os.path.exists(f):
        continue
    z = np.load(f)
    xy = z['xy_channels']                                       # [2, B, 2] channel px
    ok = np.isfinite(xy[0]).all(1) & np.isfinite(xy[1]).all(1)
    trn = np.asarray(z['train'])[data.cluster]
    csize = np.bincount(data.cluster)[data.cluster]
    for name, m in [('registration', maps_reg[1])] + [(k, v[1][1]) for k, v in models.items()]:
        rr = {}
        for sel, msk in (('train', ok & trn), ('heldout', ok & ~trn), ('bright', ok & ~data.faint),
                         ('bright_single', ok & ~data.faint & (csize == 1)), ('faint', ok & data.faint)):
            r = m.from_ref(xy[0][msk]) - xy[1][msk]
            n = np.hypot(r[:, 0], r[:, 1])
            rr[sel] = dict(n=int(msk.sum()), rms_px=float(np.sqrt((n ** 2).mean())), median_px=float(np.median(n)),
                           p90_px=float(np.percentile(n, 90)), mean_px=r.mean(0).tolist(),
                           rms_nm=float(97 * np.sqrt((n ** 2).mean())))
        mr[f'indep_{stage}_positions / {name}'] = rr
res['mapping_residuals'] = mr
res['registration_detail'] = {k: v for k, v in reg[1].items() if k in ('flip', 'n_pairs', 'rms_px', 'describe')}
json.dump(res, open(os.path.join(base, args.out), 'w'), indent=1, default=float)
for key, v in res['models'].items():
    print(key, {k: round(vv['deviance_per_px'], 4) for k, vv in v.items() if k.endswith('_clean')})
for k, v in mr.items():
    print(k, {s: (round(x['rms_px'], 3), round(x['median_px'], 3)) for s, x in v.items()})
print('seconds', time.time() - t0)
