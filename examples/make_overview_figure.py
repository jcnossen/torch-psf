"""Overview figure of torch-psf (docs/overview.png / .svg / .pdf), from generated data only.

    python examples/make_overview_figure.py [--out docs/overview]      (GPU recommended: about a minute)

Panels: (a) data in: a bead z-stack and an SMLM movie frame with detected spots; (b) spots / ROIs collected;
(c) vectorial model: pupil aberration phase and the 3D PSF (isosurfaces) of an astigmatic and a tetrapod-like PSF;
(d) xz sections; (e) the convolutional spline: knots on an oversampled grid, one strided conv2d per batch;
(f) fitting: data / model / residual of one spot; (g) the in-situ Zernike estimate vs the truth; (h) CRLB vs z;
(i) outputs: the localizations (rendered, coloured by z) and the PSF file.
"""
import argparse, json, os
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import torch_psf as tp
from torch_psf.simulate import simulate_movie, simulate_bead_stack
from torch_psf.insitu import fit_zernike_knots
from torch_psf.zernike import zernike

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'overview'))
a = ap.parse_args()
torch.manual_seed(0)
CM, DIV = 'magma', 'RdBu_r'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'font.family': 'DejaVu Sans'})

# ------------------------------------------------------------------------------------------------ models
opt = dict(pixel_um=0.108, NA=1.49, wavelength_um=0.68)
truth = {5: 0.02, 6: 0.10, 7: 0.03, 8: -0.02, 11: 0.04}
astig = tp.VectorialPSF(23, focus_um=0.4, z_range=(0.0, 0.8), **opt).set_zernike(truth)
tetra = tp.VectorialPSF(31, focus_um=0.0, z_range=(-1.0, 1.0), zernike_indices=tuple(range(5, 23)), **opt)
tetra.set_zernike({6: 0.2, 12: -0.2})
dev = astig.dev


def volume(vec, zs, R):
    pos = torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), torch.zeros_like(zs)], 1)
    with torch.no_grad():
        if vec is tetra:      # tetrapod-like: emitter at the coverslip, focus scanned
            return vec.forward_batched(pos, -zs).cpu().numpy()
        pos[:, 2] = zs
        return vec.forward_batched(pos, torch.full_like(zs, vec.focus_um)).cpu().numpy()


# ------------------------------------------------------------------------------------------------ data
g = torch.Generator(device=dev).manual_seed(1)
f0 = lambda p: astig.forward_batched(p, torch.full((len(p),), 0.4, device=dev))
movie, events = simulate_movie(f0, 23, T=1500, H=96, W=96, nsites=350, p_on=0.004, mean_on=3.0, bg=30.0,
                               photons=4000.0, z_range=(0.0, 0.8), generator=g, device=dev)
focus = 0.4 - 0.04 * (torch.arange(21, device=dev) - 10)
beads = simulate_bead_stack(lambda p, f: astig.forward_batched(p, f), focus, nbeads=1, photons=30000.0, bg=30.0,
                            generator=g, device=dev)['d'][0].cpu().numpy()
spots = tp.extract_spots(movie, 15, device=dev, threshold=6.0, isolation=12)
spline = tp.ConvSplinePSF(15, (0.0, 0.8), dz_um=0.04, zpad=5, xy_limit=3.0, device=dev)
start = tp.VectorialPSF(15, focus_um=0.4, **opt).set_zernike({6: 0.08})
est = fit_zernike_knots(spline, spots.rois, start, iters=60)
fitted = tp.ConvSplinePSF.from_vectorial(est.psf, 15, (0.0, 0.8))
theta, _ = tp.localize(fitted, spots.rois)
xy = spots.image_xy(theta).cpu().numpy()
theta = theta.cpu()

# ------------------------------------------------------------------------------------------------ figure
fig = plt.figure(figsize=(13.5, 8.6))
outer = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.28, left=0.04, right=0.985, top=0.90, bottom=0.06)


def letter(ax, s, dx=-0.08, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')


def img(ax, im, title=None, cmap=CM, log=False, sym=False, scalebar=None):
    im = np.asarray(im)
    if log:
        im = np.log10(np.clip(im / im.max(), 1e-3, None))
    if sym:
        v = np.abs(im).max(); ax.imshow(im, cmap=cmap, vmin=-v, vmax=v, interpolation='nearest', rasterized=True)
    else:
        ax.imshow(im, cmap=cmap, interpolation='nearest', rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, pad=2)
    if scalebar:
        n, lab = scalebar
        H, W = im.shape
        ax.plot([W - 2 - n, W - 2], [H - 2.5, H - 2.5], color='w', lw=2)
        ax.text(W - 2 - n / 2, H - 4, lab, color='w', ha='center', va='bottom', fontsize=7)


# (a) data in: bead stack montage + movie frame with detections
sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0, 0], hspace=0.15, wspace=0.05)
for k, i in enumerate([2, 7, 13, 18]):
    ax = fig.add_subplot(sub[0, k]); img(ax, beads[i, 3:-3, 3:-3], f'f {float(focus[i]):.2f} um')
    if k == 0:
        letter(ax, 'a', dx=-0.35, dy=1.25)
        ax.text(-0.15, 0.5, 'bead\nz-stack', transform=ax.transAxes, ha='right', va='center')
ax = fig.add_subplot(sub[1, :])
fr = int(spots.frame[torch.bincount(spots.frame).argmax()])
mov = movie[fr].float().numpy()
ax.imshow(mov[24:72, :], cmap='gray', interpolation='nearest', rasterized=True)
for (x0, y0), f in zip(spots.corner.cpu().numpy(), spots.frame.cpu().numpy()):
    if f == fr and 24 <= y0 + 7 < 72:
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 24 - 0.5), 15, 15, fill=False, ec='#ffb000', lw=1))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'SMLM movie frame {fr}: detected spots', pad=2)

# (b) spots / ROIs collected
sub = gridspec.GridSpecFromSubplotSpec(4, 6, subplot_spec=outer[0, 1], hspace=0.06, wspace=0.06)
order = torch.argsort(theta[:, 2])
pick = order[torch.linspace(0, len(order) - 1, 24).long()]
for k, i in enumerate(pick):
    ax = fig.add_subplot(sub[k // 6, k % 6]); img(ax, spots.rois[i].cpu().numpy())
    if k == 0:
        letter(ax, 'b', dx=-0.5, dy=1.15)
        ax.set_title(f'{len(spots)} spot ROIs (15 x 15 px), by fitted z (nm)', loc='left', pad=3)
    ax.text(0.03, 0.03, f'{1000 * float(theta[i, 2]):.0f}', color='w', fontsize=6, transform=ax.transAxes)

# (c) vectorial model: pupil phase + 3D isosurfaces
sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 2:], wspace=0.15, width_ratios=[0.8, 1, 1])
ax = fig.add_subplot(sub[0])
W = (torch.einsum('j,jnm->nm', astig.zcoef.detach(), astig.Z) * astig.mask).cpu().numpy()
W[astig.mask.cpu().numpy() == 0] = np.nan
v = np.nanmax(np.abs(W)); im = ax.imshow(W, cmap=DIV, vmin=-v, vmax=v, rasterized=True)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('pupil phase (waves)\nNoll 5, 6, 7, 8, 11', pad=2)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format='%.2f'); cb.ax.tick_params(labelsize=6)
letter(ax, 'c', dx=-0.2, dy=1.12)
try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None
for k, (vec, name, zr, R) in enumerate([(astig, 'astigmatic', (0.0, 0.8), 23), (tetra, 'tetrapod-like', (-1.0, 1.0), 31)]):
    ax = fig.add_subplot(sub[k + 1], projection='3d')
    zs = torch.linspace(zr[0], zr[1], 41, device=dev)
    V = volume(vec, zs, R)
    V = V / V.max()
    if marching_cubes is not None:
        for lv, col, al in ((0.5, '#ffb000', 0.9), (0.12, '#e8590c', 0.35), (0.03, '#3b6fd8', 0.15)):
            verts, faces, _, _ = marching_cubes(V, lv)
            z_um = zr[0] + verts[:, 0] * (zr[1] - zr[0]) / 40
            xs = (verts[:, 2] - (R - 1) / 2) * 0.108
            ys = (verts[:, 1] - (R - 1) / 2) * 0.108
            ax.plot_trisurf(xs, ys, faces, z_um, color=col, alpha=al, lw=0, shade=True, rasterized=True)
    ax.set_xlabel('x (um)', labelpad=-8); ax.set_ylabel('y (um)', labelpad=-8); ax.set_zlabel('z (um)', labelpad=-6)
    ax.tick_params(pad=-3, labelsize=6)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title(f'{name} PSF: isosurfaces\n50 / 12 / 3 % of the peak', pad=0)
    ax.view_init(elev=18, azim=-60)

# (d) xz sections and xy slices over z
sub = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[1, 0:2], hspace=0.25, wspace=0.08)
zs = torch.linspace(0.0, 0.8, 41, device=dev)
Va = volume(astig, zs, 23)
for k, i in enumerate([0, 10, 20, 30, 40]):
    ax = fig.add_subplot(sub[0, k]); img(ax, Va[i, 4:-4, 4:-4], f'z {float(zs[i]):.1f} um', scalebar=(5, '0.5 um') if k == 4 else None)
    if k == 0:
        letter(ax, 'd', dx=-0.25, dy=1.15)
ax = fig.add_subplot(sub[1, 0:3])
ax.imshow(np.log10(np.clip(Va[:, 11, :] / Va.max(), 1e-3, None)), cmap=CM, aspect='auto', extent=[-11.5 * 0.108, 11.5 * 0.108, 0.8, 0.0], rasterized=True)
ax.set_xlabel('x (um)'); ax.set_ylabel('z (um)'); ax.set_title('astigmatic: xz section (log)', pad=2)
ax = fig.add_subplot(sub[1, 3:5])
Vt = volume(tetra, torch.linspace(-1, 1, 41, device=dev), 31)
ax.imshow(np.log10(np.clip(Vt[:, 15, :] / Vt.max(), 1e-3, None)), cmap=CM, aspect='auto', extent=[-15.5 * 0.108, 15.5 * 0.108, 1, -1], rasterized=True)
ax.set_xlabel('x (um)'); ax.set_title('tetrapod-like (log)', pad=2); ax.set_yticks([])

# (e) convolutional spline: knots and a strided conv
sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 2:], wspace=0.25, width_ratios=[1, 1.15])
ax = fig.add_subplot(sub[0])
K = fitted.knots.cpu().numpy()
kz = int(np.argmin(np.abs(fitted.zs.cpu().numpy() - 0.4)))
c0 = (K.shape[1] - 30) // 2
ax.imshow(K[kz, c0:c0 + 30, c0:c0 + 30], cmap=CM, interpolation='nearest', rasterized=True)
for t in np.arange(-0.5, 30, 2):
    ax.axhline(t, color='w', lw=0.3, alpha=0.5); ax.axvline(t, color='w', lw=0.3, alpha=0.5)
ax.add_patch(Rectangle((7.5, 7.5), 4, 4, fill=False, ec='#3bd1ff', lw=1.5))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('ConvSplinePSF knots at z 0.4 um\n(1/2 px grid; one 4x4x4 Catmull-Rom kernel)', pad=2)
letter(ax, 'e', dx=-0.3, dy=1.08)
ax = fig.add_subplot(sub[1]); ax.axis('off')
ax.text(0, 0.95, 'one grouped conv2d per batch of emitters', fontweight='bold', va='top')
ax.text(0, 0.80, 'knot block around each emitter (index_select)\n'
                 '  -> 4x4x4 weights: value, d/dx, d/dy, d/dz\n'
                 '  -> conv2d(stride = oversample, groups = N)\n'
                 '  -> N x P x P patches and their derivatives', va='top', family='monospace', fontsize=7.5)
ax.text(0, 0.38, 'differentiable in the knots:\n  free-knot fits, and the gradient on the knots\n'
                 '  passed to the pupil parameters with one\n  vector-Jacobian product (in-situ "knot route")',
        va='top', fontsize=7.5)
sp = spline.Pl
ax.text(0, 0.02, f'knot array {tuple(fitted.knots.shape)} = {fitted.knots.numel() * 4 / 1e6:.2f} MB', fontsize=7, color='0.35')

# (f) fit: data / model / residual
sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2, 0], wspace=0.05)
i = int(pick[16])
mu = theta[i, 3] * fitted.render(theta[i:i + 1, :3].to(dev)).cpu()[0] + theta[i, 4]
d = spots.rois[i].cpu()
vmax = float(d.max())
for k, (im, t) in enumerate([(d, 'data'), (mu, 'model'), ((d - mu) / mu.sqrt(), 'residual / sd')]):
    ax = fig.add_subplot(sub[k])
    if k < 2:
        ax.imshow(im, cmap=CM, vmin=0, vmax=vmax, interpolation='nearest', rasterized=True)
    else:
        ax.imshow(im, cmap=DIV, vmin=-3, vmax=3, interpolation='nearest', rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, pad=2)
    if k == 0:
        letter(ax, 'f', dx=-0.25, dy=1.12)
        ax.set_xlabel(f'MLE: z {1000 * float(theta[i, 2]):.0f} nm, {float(theta[i, 3]):.0f} photons', loc='left')

# (g) Zernike estimate
ax = fig.add_subplot(outer[2, 1])
nl = list(range(5, 22))
tv = np.array([truth.get(j, 0.0) for j in nl])
sv = start.zcoef.detach().cpu().numpy()
ev = est.psf.zcoef.detach().cpu().numpy()
x = np.arange(len(nl))
ax.bar(x - 0.27, sv, 0.27, color='0.7', label='start')
ax.bar(x, tv, 0.27, color='#3b6fd8', label='truth')
ax.bar(x + 0.27, ev, 0.27, color='#e8590c', label='in situ (knot route)')
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(nl, fontsize=6); ax.set_xlabel('Zernike (Noll index)'); ax.set_ylabel('waves rms')
ax.legend(frameon=False, fontsize=7)
err = float(np.sqrt(np.mean((ev - tv) ** 2)))
ax.set_title(f'in-situ estimate from {len(spots)} spots: rms error {1000 * err:.1f} mwaves', pad=2)
letter(ax, 'g')

# (h) CRLB vs z
ax = fig.add_subplot(outer[2, 2])
zz = torch.linspace(0.02, 0.78, 39, device=dev)
th = torch.stack([torch.zeros_like(zz), torch.zeros_like(zz), zz, torch.full_like(zz, 4000.0), torch.full_like(zz, 30.0)], 1)
cr = tp.crlb(fitted, th).cpu().numpy()
z_ = zz.cpu().numpy() * 1000
ax.plot(z_, cr[:, 0] * 108, color='#ffb000', label='x')
ax.plot(z_, cr[:, 1] * 108, color='#3b6fd8', label='y')
ax.plot(z_, cr[:, 2] * 1000, color='#d6336c', label='z')
ax.set_xlabel('emitter height z (nm)'); ax.set_ylabel('CRLB (nm)'); ax.set_ylim(0, None)
ax.legend(frameon=False); ax.set_title('Cramer-Rao bound, 4000 photons, 30 bg / px', pad=2)
letter(ax, 'h')

# (i) outputs
sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, 3], wspace=0.08, width_ratios=[1, 0.9])
ax = fig.add_subplot(sub[0])
L = tp.localize_movie(movie.numpy(), fitted, device=dev, isolation=None)
ax.scatter(L['x'], L['y'], c=L['z'], s=2, cmap='viridis', vmin=0, vmax=0.8, rasterized=True)
ax.set_xlim(0, 96); ax.set_ylim(96, 0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'{len(L)} localizations\n(colour: z 0-800 nm)', pad=2)
letter(ax, 'i', dx=-0.12)
ax = fig.add_subplot(sub[1]); ax.axis('off')
txt = ('psf.json\n{"format":"torch-psf",\n "type": "vectorial",\n "optics": {...},\n "zernike_noll": [...],\n'
       ' "zernike_waves": [...],\n "focus_um": 0.4, ...}\n\nlocalizations:\nframe, x, y, z,\nphotons, bg,\ncrlb_*, loglik,\n'
       'chi2, at_limit')
ax.text(0.02, 0.98, txt, va='top', family='monospace', fontsize=6.5,
        bbox=dict(boxstyle='round', fc='#f4f4f4', ec='0.8'))
fig.suptitle('torch-psf: data -> spots -> PSF model (vectorial / convolutional spline) -> fit -> PSF file and localizations',
             fontsize=11, x=0.02, y=0.985, ha='left')

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out + '.png', dpi=110)
fig.savefig(a.out + '.svg', dpi=150)
fig.savefig(a.out + '.pdf', dpi=150)
try:   # smaller png (palette) when Pillow is available
    from PIL import Image
    im = Image.open(a.out + '.png').convert('RGB').quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    im.save(a.out + '.png', optimize=True)
except Exception:
    pass
print('written', a.out + '.png / .svg / .pdf', os.path.getsize(a.out + '.png') // 1000, 'kB')
