"""Maps of the bead field fits (beads_field.py) against the per-patch phase-diversity pupils of the phase-diversity
study (~/phasediv_out/real/field_z/field.json: Zernikes per 64 px patch, NA / apod global, both stacks).

    python -I beads_maps.py OUT_DIR RESULT_JSON [RESULT_JSON ...] --patches FIELD_JSON [--keys dense_mixed pd_mixed]
Writes OUT_DIR/figs/beads_field_maps.png, beads_heldout_clusters.png and OUT_DIR/json/beads_maps.json.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
from torch_psf.field import FieldZernikes

MODES = [5, 6, 7, 8, 11, 12, 13, 14]
W, H = 475, 400


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float('nan')


ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('results', nargs='+'); ap.add_argument('--patches', required=True)
ap.add_argument('--keys', nargs='*', default=None)
a = ap.parse_args()
R = {}
for p in a.results:
    R.update(json.load(open(p)))
keys = a.keys or list(R)
pf = json.load(open(a.patches))['patches']
pc = np.array([p['center'] for p in pf]); pz = np.array([p['zcoef'] for p in pf]); ps = np.array([p['stack'] for p in pf])
noll_p = list(range(5, 22))
# stack-to-stack agreement of the per-patch values (same corners)
c0 = {tuple(p['corner']): i for i, p in enumerate(pf) if p['stack'] == 0}
pairs = [(c0[tuple(p['corner'])], i) for i, p in enumerate(pf) if p['stack'] == 1 and tuple(p['corner']) in c0]
out = dict(stack_to_stack={}, fits={})
for j in MODES:
    k = noll_p.index(j)
    i0, i1 = np.array(pairs).T
    out['stack_to_stack'][j] = dict(r=corr(pz[i0, k], pz[i1, k]), med_abs_diff=float(np.median(np.abs(pz[i0, k] - pz[i1, k]))))
X, Y = np.meshgrid(np.linspace(0, W - 1, 96), np.linspace(0, H - 1, 81))
maps = {}
for key in keys:
    r = R[key]
    fz, _ = FieldZernikes.from_block(r['field_psf']['block'])
    base = np.asarray(r['field_psf']['zcoef'])
    noll = r['global_psf']['noll']

    def at(x, y):
        o = np.tile(base, (np.size(x), 1)).reshape(np.shape(x) + (len(base),))
        o[..., [noll.index(j) for j in fz.noll]] += fz.maps(x, y)
        return o
    maps[key] = at(X, Y)
    atp = at(pc[:, 0], pc[:, 1])
    rec = {}
    for j in MODES:
        k, kp = noll.index(j), noll_p.index(j)
        rec[j] = dict(r_stack0=corr(atp[ps == 0, k], pz[ps == 0, kp]), r_stack1=corr(atp[ps == 1, k], pz[ps == 1, kp]),
                      med_abs_diff_stack1=float(np.median(np.abs(atp[ps == 1, k] - pz[ps == 1, kp]))),
                      range=[float(maps[key][..., k].min()), float(maps[key][..., k].max())],
                      global_value=float(r['global_psf']['zcoef'][k]))
    out['fits'][key] = rec
os.makedirs(os.path.join(a.out, 'figs'), exist_ok=True); os.makedirs(os.path.join(a.out, 'json'), exist_ok=True)
json.dump(out, open(os.path.join(a.out, 'json', 'beads_maps.json'), 'w'), indent=1)
print('| mode | stack-to-stack r (per-patch PD) | ' + ' | '.join(f'{k}: r stack0 / stack1, range (mλ)' for k in keys) + ' |')
print('|---|---|' + '---|' * len(keys))
for j in MODES:
    s = out['stack_to_stack'][j]
    cells = [f"{out['fits'][k][j]['r_stack0']:+.2f} / {out['fits'][k][j]['r_stack1']:+.2f}, "
             f"{1e3 * out['fits'][k][j]['range'][0]:.0f}..{1e3 * out['fits'][k][j]['range'][1]:.0f}" for k in keys]
    print(f'| Z{j} | {s["r"]:+.2f} | ' + ' | '.join(cells) + ' |')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(MODES), len(keys), figsize=(3.1 * len(keys), 2.4 * len(MODES)), squeeze=False)
for i, j in enumerate(MODES):
    kp = noll_p.index(j)
    vals = np.concatenate([maps[k][..., R[k]['global_psf']['noll'].index(j)].ravel() for k in keys] + [pz[:, kp]])
    lo, hi = 1e3 * np.percentile(vals, 2), 1e3 * np.percentile(vals, 98)
    for c, key in enumerate(keys):
        ax = axs[i, c]
        k = R[key]['global_psf']['noll'].index(j)
        im = ax.imshow(1e3 * maps[key][..., k], extent=(0, W, H, 0), vmin=lo, vmax=hi, cmap='RdBu_r')
        for st, mk, dx in ((0, 'o', -9), (1, 's', 9)):           # the two stacks side by side at a patch
            m = ps == st
            ax.scatter(pc[m, 0] + dx, pc[m, 1], c=1e3 * pz[m, kp], vmin=lo, vmax=hi, cmap='RdBu_r', marker=mk, s=22,
                       edgecolors='k', linewidths=0.4)
        f = out['fits'][key][j]
        ax.set_title(f'{key} Z{j}: r {f["r_stack0"]:+.2f} / {f["r_stack1"]:+.2f}', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=5)
fig.suptitle('2-colour beads, left channel: field polynomial (mλ) with the per-patch phase-diversity pupils\n'
             '(left circle beads.tif = training stack, right square beads2.tif); r: polynomial vs patches per stack', fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(os.path.join(a.out, 'figs', 'beads_field_maps.png'), dpi=120)
plt.close(fig)
fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False)
for c, key in enumerate(keys):
    r = R[key]
    xy, dn = np.array(r['cluster_xy']), 1e3 * np.array(r['cluster_dnll_per_px'])
    v = np.abs(dn).max()
    ax = axs[0, c]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=dn, cmap='RdBu_r', vmin=-v, vmax=v, s=30, edgecolors='k', linewidths=0.3)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal')
    ax.set_title(f'{key}: field - global, held-out NLL / px x 1e3\n{r["clusters_better"]} / {r["n_clusters"]} clusters '
                 f'better; total {1e3 * r["dnll_per_px"]:+.1f}', fontsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.04)
fig.tight_layout()
fig.savefig(os.path.join(a.out, 'figs', 'beads_heldout_clusters.png'), dpi=120)
