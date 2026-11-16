"""Does fitting NA / apod remove the defocused-ring mismatch? Every PSF is scored on the held-out beads2 clusters by
dense_nll (the dense-beads procedure) and its refitted model is compared with the data: deviance per plane over the
focus, radial profiles of the single-bead clusters at defocused planes, and data / model / residual images.

  python -I fig_ring.py OUT_DIR label=psf.json [label=psf.json ...]
Writes OUT_DIR/ring_profiles.png, ring_images.png, deviance_focus.png, ring.json.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from torch_psf import PSF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1]
items = [a.split('=', 1) for a in sys.argv[2:]]
st = load()
data, focus, ho = clusters(st)
hod = data.subset(ho)
del st
fk = focus.cpu().numpy()
singles = [c for c in range(hod.n_clusters) if hod.cl_bead[c + 1] - hod.cl_bead[c] == 1]
PLANES = [float(f) for f in (-0.74, -0.5, -0.25, 0.25, 0.5, 0.73)]
kp = [int(np.argmin(np.abs(fk - f))) for f in PLANES]
rb = np.arange(0, 16.5, 0.5)
res, prof, imgs = {}, {}, {}
d = hod.d


def radial(img, cx, cy):
    """mean of img [nk, h, w] in radius bins around (cx, cy) -> [nk, nbins] (NaN-aware)"""
    h, w = img.shape[-2:]
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(xx - cx, yy - cy)
    out_ = np.full((img.shape[0], len(rb) - 1), np.nan)
    for i in range(len(rb) - 1):
        m = (r >= rb[i]) & (r < rb[i + 1])
        if m.any():
            out_[:, i] = np.nanmean(img[:, m], 1)
    return out_


for label, path in items:
    psf = PSF.load(os.path.expanduser(path), device='cuda')
    s, r = score(hod, focus, psf)
    mu = r['mu']
    with torch.no_grad():
        dv = 2 * (torch.where(d > 0, d * torch.log(d.clamp_min(1e-9) / mu.clamp_min(1e-6)), torch.zeros_like(d)) - (d - mu))
        dev_plane = dv.mean(1).cpu().numpy()
    res[label] = dict(path=path, heldout=s, NA=float(psf.NA), apod=psf.apod.detach().cpu().tolist(),
                      deviance_per_plane=dev_plane.tolist())
    pd_, pm_ = [], []
    xy, ph, bgc = r['xy'].cpu().numpy(), r['photons'].cpu().numpy(), r['bg'].cpu().numpy()
    for c in singles:
        b = hod.cl_bead[c]
        x0, y0 = hod.clusters[c]['box'][:2]
        cx, cy = xy[b, 0] - x0, xy[b, 1] - y0
        D = hod.unflatten(d, c).cpu().numpy()
        M = hod.unflatten(mu, c).cpu().numpy()
        bgv = bgc[c][:, None, None]
        norm = ph[b][:, None]
        pd_.append(radial(D - bgv, cx, cy) / norm)
        pm_.append(radial(M - bgv, cx, cy) / norm)
        if c in singles[:3]:
            imgs.setdefault(c, {})['data'] = D
            imgs[c][label] = M
    prof[label] = (np.nanmean(pd_, 0), np.nanmean(pm_, 0))
    print(label, json.dumps(s), flush=True)
json.dump(dict(focus_um=fk.tolist(), planes=PLANES, singles=len(singles), psfs=res), open(os.path.join(out, 'ring.json'), 'w'),
          indent=1)

labels = [l for l, _ in items]
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
# deviance per plane
fig, ax = plt.subplots(figsize=(7, 4))
for i, l in enumerate(labels):
    ax.plot(fk, res[l]['deviance_per_plane'], color=cols[i % 10], lw=1.4,
            label=f"{l} ({res[l]['heldout']['deviance_per_px']:.3f})")
ax.set_xlabel('focus (um, stage)')
ax.set_ylabel('held-out deviance / px')
ax.set_title('held-out beads2 clusters: deviance per plane (overall in brackets)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, 'deviance_focus.png'), dpi=130)
# radial profiles at the defocused planes
fig, axs = plt.subplots(1, len(kp), figsize=(3.0 * len(kp), 3.2), sharey=False)
rc = 0.5 * (rb[1:] + rb[:-1]) * OPT['pixel_um']
for j, k in enumerate(kp):
    ax = axs[j]
    ax.plot(rc, prof[labels[0]][0][k], 'k.-', lw=2, label='data')
    for i, l in enumerate(labels):
        ax.plot(rc, prof[l][1][k], color=cols[i % 10], lw=1.2, label=l)
    ax.set_title(f'focus {fk[k]:+.2f} um')
    ax.set_xlabel('radius (um)')
    if j == 0:
        ax.set_ylabel('(image - bg) / photons per px')
axs[-1].legend(fontsize=7)
fig.suptitle(f'radial profiles, mean of {len(singles)} single-bead held-out clusters')
fig.tight_layout()
fig.savefig(os.path.join(out, 'ring_profiles.png'), dpi=130)
# images of one bead: data / models / residuals at the planes
c = list(imgs)[0]
rows = ['data'] + labels + [f'res {l}' for l in labels]
fig, axs = plt.subplots(len(rows), len(kp), figsize=(1.9 * len(kp), 1.9 * len(rows)))
for j, k in enumerate(kp):
    D = imgs[c]['data'][k]
    vmax = np.nanpercentile(D, 99.5)
    for i, row in enumerate(rows):
        ax = axs[i, j]
        if row.startswith('res '):
            M = imgs[c][row[4:]][k]
            im = (D - M) / np.sqrt(np.maximum(M, 1))
            ax.imshow(im, cmap='RdBu_r', vmin=-5, vmax=5)
        else:
            ax.imshow(imgs[c][row][k], cmap='magma', vmin=np.nanpercentile(D, 1), vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(row, fontsize=7)
        if i == 0:
            ax.set_title(f'{fk[k]:+.2f} um', fontsize=8)
fig.suptitle('held-out bead: data, models, residuals (d - mu) / sqrt(mu) in [-5, 5]', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out, 'ring_images.png'), dpi=130)
print('figures in', out)
