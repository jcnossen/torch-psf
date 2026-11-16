"""Field map by phase diversity on the 2-colour bead stacks (left channel): one pupil (Zernikes, NA, apod) per patch,
started from a global PD fit, in both stacks (the same camera field, different bead fields: the per-patch values of
the two stacks are an independent check of each other). Held-out score: every beads2 cluster is scored (dense_nll,
the dense-beads procedure) with the PSF of the beads.tif patch that holds its centre, and with the global PSF.

  python -I real_field.py OUT_DIR START_PSF_JSON [--fit zcoef NA apod] [--patch 64] [--iters 200]
Writes OUT_DIR/field.json, field_map.png, field_stacks.png.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from torch_psf import PSF
from torch_psf.estimation.phase_diversity import PDPatches, fit_phase_diversity, radial_amplitude
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('start')
ap.add_argument('--fit', nargs='+', default=['zcoef', 'NA', 'apod'])
ap.add_argument('--patch', type=int, default=64)
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--min-signal', type=float, default=8.0)
ap.add_argument('--min-valid', type=float, default=0.6)
ap.add_argument('--warmup', type=int, default=40, help='first iterations with the PSF fixed (focus offset, drift settle)')
ap.add_argument('--skip-fit', action='store_true', help='reuse field.json, redo the scoring / figures')
args = ap.parse_args()
st = load()
data, focus, ho = clusters(st)
hod = data.subset(ho)
g = PSF.load(os.path.expanduser(args.start), device='cuda')
nk = st[0].shape[0]
focus_all = stage_focus(nk, DZ, 1.0).cuda()
path = os.path.join(args.out, 'field.json')
masks = [field_stop_mask(s.cpu().numpy()) for s in st]
pt = PDPatches.from_stacks(st, args.patch, masks=masks, min_valid=args.min_valid, min_signal=args.min_signal)
print('patches', pt.n, 'per stack', np.bincount(pt.stack.cpu().numpy()).tolist(), flush=True)
if args.skip_fit:
    field = json.load(open(path))['patches']
else:
    field = []
    t0 = time.time()
    for i in range(pt.n):
        ti = time.time()
        r = fit_phase_diversity(pt.subset([i]), focus_all, g, fit=tuple(args.fit), ls_iters=0, iters=args.iters,
                                warmup=args.warmup)
        p = r['psf']
        field.append(dict(index=i, stack=int(pt.stack[i]), corner=pt.corner[i].tolist(), center=pt.centers[i].tolist(),
                          zcoef=p.zcoef.detach().cpu().tolist(), NA=float(p.NA), apod=p.apod.detach().cpu().tolist(),
                          focus_offset_um=float(r['focus_offset'][0]), deviance_per_px=r['report']['deviance_per_px'],
                          valid=float(pt.valid[i].mean()), seconds=time.time() - ti))
        print(i, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in field[-1].items()
                             if k not in ('zcoef', 'apod')}), [round(z * 1e3, 1) for z in field[-1]['zcoef'][:8]],
              flush=True)
    json.dump(dict(start=args.start, fit=args.fit, patch=args.patch, patches=field, seconds=time.time() - t0),
              open(path, 'w'), indent=1)

# ---- held-out score: beads2 clusters with the PSF of the beads.tif patch at their centre vs the global PSF
tr = [f for f in field if f['stack'] == 0]
cc = np.array([f['center'] for f in tr])
groups = {}
for j, c in enumerate(ho):
    x0, y0, x1, y1 = data.clusters[c]['box']
    ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    groups.setdefault(int(np.argmin(((cc - ctr) ** 2).sum(1))), []).append(j)
sc = dict(local=dict(dev=0.0, nll=0.0, npx=0.0), global_=dict(dev=0.0, nll=0.0, npx=0.0))
per_group = []
for gi, js in groups.items():
    sub = hod.subset(js)
    f = tr[gi]
    p = g.resized(g.roisize)
    with torch.no_grad():
        p.zcoef.copy_(torch.tensor(f['zcoef'])); p.NA.fill_(f['NA']); p.apod.copy_(torch.tensor(f['apod']))
    npx = float(sub.n_px * sub.nk)
    row = dict(patch=f['index'], center=f['center'], n_clusters=len(js))
    for name, psf in (('local', p), ('global_', g)):
        s, _ = score(sub, focus, psf)
        sc[name]['dev'] += s['deviance_per_px'] * npx
        sc[name]['nll'] += s['nll_per_px'] * npx
        sc[name]['npx'] += npx
        row[name] = s['deviance_per_px']
    per_group.append(row)
    print('heldout group', json.dumps(row), flush=True)
heldout = {k: dict(deviance_per_px=v['dev'] / v['npx'], nll_per_px=v['nll'] / v['npx']) for k, v in sc.items()}
print('heldout', json.dumps(heldout), flush=True)
J = json.load(open(path))
J.update(heldout=heldout, heldout_groups=per_group)
json.dump(J, open(path, 'w'), indent=1)

# ---- figures
def amp_at(f, s):
    """radial pupil transmission exp(sum apod_n rho^2n) at the pupil coordinate s = NA rho (0 beyond NA)"""
    rho = s / f['NA']
    a = f['apod']
    return float(np.exp(a[0] * rho ** 2 + a[1] * rho ** 4 + a[2] * rho ** 6)) if rho <= 1 else 0.0


zidx = list(g.zidx)
qty = [('Z5', lambda f: 1e3 * f['zcoef'][zidx.index(5)]), ('Z6', lambda f: 1e3 * f['zcoef'][zidx.index(6)]),
       ('Z7', lambda f: 1e3 * f['zcoef'][zidx.index(7)]), ('Z8', lambda f: 1e3 * f['zcoef'][zidx.index(8)]),
       ('Z11', lambda f: 1e3 * f['zcoef'][zidx.index(11)]), ('Z12', lambda f: 1e3 * f['zcoef'][zidx.index(12)]),
       ('NA', lambda f: f['NA']),
       ('A(1.33)', lambda f: amp_at(f, 1.33)),
       ('focus offset (um)', lambda f: f['focus_offset_um']), ('PD deviance / px', lambda f: f['deviance_per_px'])]
H, W = st[0].shape[-2:]
fig, axs = plt.subplots(2, len(qty), figsize=(2.6 * len(qty), 5.2))
for s_ in (0, 1):
    fs = [f for f in field if f['stack'] == s_]
    for j, (name, fn) in enumerate(qty):
        ax = axs[s_, j]
        v = np.array([fn(f) for f in fs])
        both = np.array([fn(f) for f in field])
        lo, hi = np.nanpercentile(both, [2, 98])
        img = np.full((H, W), np.nan)
        for f, x in zip(fs, v):
            x0, y0 = f['corner']
            img[y0:y0 + args.patch, x0:x0 + args.patch] = x
        im = ax.imshow(img, vmin=lo, vmax=hi, cmap='viridis')
        ax.contour(masks[s_], [0.5], colors='w', linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        if s_ == 0:
            ax.set_title(name, fontsize=9)
        if j == 0:
            ax.set_ylabel(['beads.tif', 'beads2.tif'][s_])
        plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('phase diversity per 64 px patch (6.2 um): Zernikes in mλ, NA, pupil transmission at NA 1.33 (rel. to the centre)')
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'field_map.png'), dpi=120)
# the two stacks against each other (patches at the same field position)
c0 = {tuple(f['corner']): f for f in field if f['stack'] == 0}
pairs = [(c0[tuple(f['corner'])], f) for f in field if f['stack'] == 1 and tuple(f['corner']) in c0]
fig, axs = plt.subplots(1, 7, figsize=(19, 3))
for j, (name, fn) in enumerate(qty[:7]):
    a = np.array([fn(p[0]) for p in pairs]); b = np.array([fn(p[1]) for p in pairs])
    axs[j].plot(a, b, '.')
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    axs[j].plot([lo, hi], [lo, hi], 'k:')
    r = np.corrcoef(a, b)[0, 1] if len(a) > 2 else np.nan
    axs[j].set_title(f'{name} (r = {r:.2f}, n = {len(a)})', fontsize=9)
    axs[j].set_xlabel('beads.tif')
    if j == 0:
        axs[j].set_ylabel('beads2.tif (same field position)')
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'field_stacks.png'), dpi=120)
print('done', flush=True)
