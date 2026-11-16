"""3-colour beads: per-bead intensity ratio right / left from the joint fit (photons summed over the planes), the
fitted per-channel PSFs (Zernikes, NA, apod, channel focus), the mapping vs the 2-colour one and the SMLM affine maps.
  python -I analyze3col.py [run tag, default joint] -> OUT/3col/<tag>/analysis.json, ratios.png"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *                                     # noqa: F401,F403
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else 'joint'
d = os.path.join(OUT, '3col', tag)
rep = json.load(open(os.path.join(d, 'report.json')))
z = np.load(os.path.join(d, 'fit_global.npz'))
dd = np.load(os.path.join(d, 'data.npz'))
ph = z['photons']
both = np.isfinite(ph).all(1) & (ph > 0).all(1)
bright = both & ~dd['faint'].astype(bool)
ratio = ph[:, 1] / ph[:, 0]
out = dict(n_beads=int(len(ph)), n_both=int(both.sum()), n_bright_both=int(bright.sum()))
for name, m in (('all_both', both), ('bright_both', bright)):
    r = ratio[m]
    out[name] = dict(median=float(np.median(r)), mean=float(r.mean()), sd=float(r.std()), cv=float(r.std() / r.mean()),
                     q10_q90=np.percentile(r, [10, 90]).tolist(), values=sorted(float(v) for v in r))
tot = ph[:, 0] + ph[:, 1]
out['beads'] = [dict(xy_left=dd['xy_ref'][i].tolist(), photons_left=float(ph[i, 0]), photons_right=float(ph[i, 1]),
                     ratio=float(ratio[i]), faint=bool(dd['faint'][i]), stack=int(dd['stack'][i]))
                for i in np.argsort(-np.nan_to_num(tot)) if both[i]]
psfs, maps, cf = zip_model(os.path.join(d, 'psf_global.zip'), 'cpu')
out['psf'] = [dict(NA=float(getattr(p, 'base', p).NA), apod=getattr(p, 'base', p).apod.tolist(),
                   zernike_mlambda={int(j): round(1000 * float(v), 1) for j, v in zip(getattr(p, 'base', p).zidx,
                                                                                    getattr(p, 'base', p).zcoef.detach())})
              for p in psfs]
out['channel_focus_um'] = cf
out['mapping'] = maps[1].describe()
r2 = os.path.join(OUT, '2col', 'joint', 'report.json')
if os.path.exists(r2):
    out['mapping_2col'] = json.load(open(r2))['global']['mapping'][1]
out['report_global'] = rep['global']
json.dump(out, open(os.path.join(d, 'analysis.json'), 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(ph[both, 0], ph[both, 1], c=np.where(dd['faint'][both], 'gray', 'k'), s=12)
lim = np.nanmax(ph[both]) * 1.1
ax[0].plot([1, lim], [out['bright_both']['median'], lim * out['bright_both']['median']], 'r-', lw=0.8)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('photons left (all planes)'); ax[0].set_ylabel('photons right')
ax[0].set_title(f"3-colour beads: right / left = {out['bright_both']['median']:.2f} (bright, n={bright.sum()})")
ax[1].hist(ratio[bright], bins=np.linspace(0, 2, 41), color='k'); ax[1].hist(ratio[both & ~bright], bins=np.linspace(0, 2, 41), color='gray', alpha=0.6)
ax[1].set_xlabel('photon ratio right / left per bead'); ax[1].set_title('black: bright beads, gray: faint members')
plt.tight_layout(); plt.savefig(os.path.join(d, 'ratios.png'), dpi=90)
print(json.dumps({k: v for k, v in out.items() if k not in ('beads', 'report_global')}, indent=1)[:4000])
