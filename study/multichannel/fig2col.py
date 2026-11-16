"""Figures of the 2-colour multi-channel study: per-channel field maps (independent vs joint field fits), the
mapping (residuals of the independently fitted bead positions through the registration and the jointly fitted maps;
the difference of the maps over the field) and the held-out gain per cluster.
  python -I fig2col.py -> OUT/2col/figs/*.png, field_maps.json"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *                                     # noqa: F401,F403
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

base = os.path.join(OUT, '2col')
figs = os.path.join(base, 'figs'); os.makedirs(figs, exist_ok=True)
dev = 'cpu'
MODES = [5, 6, 7, 8, 11, 12, 13]
runs = [k for k in ('indep', 'joint', 'joint_shared', 'joint_poly2') if os.path.exists(os.path.join(base, k, 'psf_field.zip'))]
models = {k: zip_model(os.path.join(base, k, 'psf_field.zip'), dev) for k in runs}
dd = np.load(os.path.join(base, 'joint', 'data.npz'))
masks = dd['masks'] if dd['masks'].size else None
W, H = 475, 400
gx, gy = np.meshgrid(np.arange(0, W, 5), np.arange(0, H, 5))
g = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
show = [k for k in ('indep', 'joint') if k in models]
fig, ax = plt.subplots(2 * len(show), len(MODES), figsize=(2.6 * len(MODES), 2.3 * 2 * len(show)))
out = {}
for r, (k, c) in enumerate([(k, c) for c in range(2) for k in show]):
    psfs, maps, cf = models[k]
    m = psfs[c]
    z = (m.base.zcoef.detach()[None] + m.field.offsets(g, m.base.zidx).detach()).numpy() * 1000      # mλ
    out[f'{k}_ch{c}'] = {}
    for j, n in enumerate(MODES):
        v = z[:, m.base.zidx.index(n)].reshape(gx.shape)
        if masks is not None:
            v = np.where(masks[c, 0][::5, ::5], v, np.nan)
        out[f'{k}_ch{c}'][n] = dict(min=float(np.nanmin(v)), max=float(np.nanmax(v)), mean=float(np.nanmean(v)))
        a = ax[r, j]
        im = a.imshow(v, extent=(0, W, H, 0), cmap='coolwarm')
        a.set_title(f'{NAMES[c]} {k}: Z{n}\n{np.nanmin(v):.0f} .. {np.nanmax(v):.0f} mλ', fontsize=8)
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.04)
plt.tight_layout(); plt.savefig(os.path.join(figs, 'field_maps.png'), dpi=80); plt.close()
# correlation of the two channels' maps (joint) and of indep vs joint per channel
corr = {}
for n in MODES:
    vals = {}
    for k in show:
        for c in range(2):
            m = models[k][0][c]
            vals[(k, c)] = m.field.offsets(g, m.base.zidx).detach().numpy()[:, m.base.zidx.index(n)]
    if len(show) == 2:
        corr[n] = dict(indep_vs_joint_left=float(np.corrcoef(vals[('indep', 0)], vals[('joint', 0)])[0, 1]),
                       indep_vs_joint_right=float(np.corrcoef(vals[('indep', 1)], vals[('joint', 1)])[0, 1]),
                       left_vs_right_joint=float(np.corrcoef(vals[('joint', 0)], vals[('joint', 1)])[0, 1]))
json.dump(dict(maps=out, corr=corr), open(os.path.join(base, 'field_maps.json'), 'w'), indent=1)

# mapping: residuals of the independent bead positions
z = np.load(os.path.join(base, 'indep', 'fit_field.npz'))
xy = z['xy_channels']
ok = np.isfinite(xy[0]).all(1) & np.isfinite(xy[1]).all(1) & ~dd['faint'].astype(bool)     # bright beads
data_reg = json.load(open(os.path.join(base, 'joint', 'report.json')))['registration'][1]['describe']
A, b = np.array(data_reg['A']), np.array(data_reg['b'])
reg = lambda p: p @ A.T + b
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, (name, f) in zip(ax[:2], [('registration (detections)', reg), ('joint fit', models['joint'][1][1].from_ref)]):
    r = f(xy[0][ok]) - xy[1][ok]
    q = a.quiver(xy[0][ok, 0], xy[0][ok, 1], r[:, 0], -r[:, 1], angles='xy', scale_units='xy', scale=0.01, color='k')
    a.set_xlim(0, W); a.set_ylim(H, 0); a.set_aspect('equal')
    a.set_title(f'{name}: map(left) - right, bright beads, rms {np.sqrt((r ** 2).sum(1).mean()):.3f} px, median {np.median(np.hypot(*r.T)):.3f} px (x 100)')
d = models['joint'][1][1].from_ref(np.stack([gx.ravel(), gy.ravel()], 1).astype(float)) - reg(np.stack([gx.ravel(), gy.ravel()], 1).astype(float))
s = (slice(None, None, 6), slice(None, None, 6))
ax[2].quiver(gx[s], gy[s], d[:, 0].reshape(gx.shape)[s], -d[:, 1].reshape(gx.shape)[s], angles='xy', scale_units='xy', scale=0.005)
ax[2].set_xlim(0, W); ax[2].set_ylim(H, 0); ax[2].set_aspect('equal')
ax[2].set_title(f'joint map - registration map (x 200), max {np.abs(d).max():.2f} px')
plt.tight_layout(); plt.savefig(os.path.join(figs, 'mapping.png'), dpi=80); plt.close()

# held-out gain per cluster (joint protocol, field joint vs global indep)
sc = os.path.join(base, 'score2col.json')
if os.path.exists(sc):
    S = json.load(open(sc))
    print(json.dumps({k: {kk: round(vv['deviance_per_px'], 4) for kk, vv in v.items() if kk.endswith('clean')}
                      for k, v in S['models'].items()}, indent=1))
print(json.dumps(corr, indent=1))
