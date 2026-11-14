"""Overview figure of torch-psf (docs/overview.png), from generated data only.

    python examples/make_overview_figure.py [--out docs/overview]      (GPU recommended: about a minute)

Top row, the processing flow: (a) an SMLM movie frame with detected spots -> (b) spot ROIs -> (c) the vectorial model
(pupil aberration phase) -> (d) a maximum-likelihood fit (data, model, residual) -> (e) the localizations rendered as
a super-resolution image, coloured by z. Below, the details: (f) sections of an astigmatic and a tetrapod-like PSF,
(g) the convolutional spline, (h) the in-situ Zernike estimate against the truth, (i) the Cramer-Rao bound.
"""
import argparse, os
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter
import torch_psf as tp
from torch_psf.simulate import simulate_movie, filament_sites
from torch_psf.insitu import fit_zernike_knots

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'overview'))
a = ap.parse_args()
torch.manual_seed(0)
CM, DIV, ZCM = 'magma', 'RdBu_r', 'turbo'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'font.family': 'DejaVu Sans',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
PX = 0.108                                                      # um

# ------------------------------------------------------------------------------------------------ models and data
opt = dict(pixel_um=PX, NA=1.49, wavelength_um=0.68)
truth = {5: 0.02, 6: 0.10, 7: 0.03, 8: -0.02, 11: 0.04}
astig = tp.VectorialPSF(23, focus_um=0.4, z_range=(0.0, 0.8), **opt).set_zernike(truth)
tetra = tp.VectorialPSF(31, focus_um=1.0, z_range=(0.0, 2.0), zernike_indices=tuple(range(5, 23)), **opt)
tetra.set_zernike({6: 0.2, 12: -0.2})
dev = astig.dev
g = torch.Generator(device=dev).manual_seed(1)
H = W = 96
sites = filament_sites(n_filaments=7, sites_per_um=70.0, H=H, W=W, pixel_um=PX, z_range=(0.05, 0.75), generator=g,
                       device=dev)
render_true = lambda p: astig.forward_batched(p, torch.full((len(p),), 0.4, device=dev))
movie, events = simulate_movie(render_true, 23, T=4000, H=H, W=W, p_on=0.0006, mean_on=3.0, bg=30.0, photons=4000.0,
                               photons_logsd=0.3, generator=g, device=dev, sites=sites)
spots = tp.extract_spots(movie, 15, device=dev, threshold=6.0, isolation=16)
start = tp.VectorialPSF(15, focus_um=0.4, **opt).set_zernike({6: 0.08})
spline = tp.ConvSplinePSF(15, (0.0, 0.8), dz_um=0.04, zpad=5, xy_limit=3.0, device=dev)
est = fit_zernike_knots(spline, spots.rois, start, iters=200)
fitted = tp.ConvSplinePSF.from_vectorial(est.psf, 15, (0.0, 0.8))
locs = tp.localize_movie(movie.numpy(), fitted, device=dev)
locs = locs[~locs['at_limit'] & (locs['crlb_x'] < 0.25) & (locs['z'] > 0.0) & (locs['z'] < 0.8)]
theta, _ = tp.localize(fitted, spots.rois[:2000])
theta = theta.cpu()


def sections(vec, zs):
    pos = torch.zeros(len(zs), 3, device=dev)
    pos[:, 2] = zs
    with torch.no_grad():
        return vec.forward_batched(pos, torch.full_like(zs, vec.focus_um)).cpu().numpy()


# ------------------------------------------------------------------------------------------------ layout
fig = plt.figure(figsize=(7.2, 7.0))
G = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.25, 1.05], hspace=0.36, left=0.08, right=0.985,
                      top=0.955, bottom=0.065)


LETTERS = []


def letter(ax, s, row):
    """bold panel letter; placed after the layout: left of the panel, at the top of its row (titles included)"""
    LETTERS.append((ax, s, row))


def place_letters():
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    top = {}
    for ax, s, row in LETTERS:
        bb = inv.transform(ax.get_tightbbox(r))
        top[row] = max(top.get(row, 0), bb[1, 1])
    for ax, s, row in LETTERS:
        x0 = min(ax.get_position().x0, inv.transform(ax.get_tightbbox(r))[0, 0])
        fig.text(max(x0 - 0.012, 0.022), top[row], s, fontsize=10, fontweight='bold', ha='right', va='top')


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def scalebar(ax, length_px, label, shape, color='w'):
    h, w = shape
    x1 = w - 0.06 * w; x0 = x1 - length_px
    ax.plot([x0, x1], [h - 2, h - 2], color=color, lw=1.5, solid_capstyle='butt')
    ax.text((x0 + x1) / 2, h - 3, label, color=color, ha='center', va='bottom', fontsize=8)


def arrow(ax0, ax1):
    """thin arrow between two panels of the flow row (figure coordinates)"""
    b0, b1 = ax0.get_position(), ax1.get_position()
    y = (b0.y0 + b0.y1) / 2
    fig.add_artist(FancyArrowPatch((b0.x1 + 0.004, y), (b1.x0 - 0.004, y), transform=fig.transFigure,
                                   arrowstyle='-|>', mutation_scale=9, lw=0.8, color='0.35'))


# ---- row 1: the flow
r1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=G[0], wspace=0.32)
# (a) data
ax_a = fig.add_subplot(r1[0])
fr = int(torch.bincount(spots.frame).argmax())
ax_a.imshow(movie[fr].float().numpy()[18:78, 18:78], cmap='gray', interpolation='nearest')
for (x0, y0), f in zip(spots.corner.cpu().numpy(), spots.frame.cpu().numpy()):
    if f == fr and 18 <= x0 and x0 + 15 <= 78 and 18 <= y0 and y0 + 15 <= 78:
        ax_a.add_patch(Rectangle((x0 - 18 - 0.5, y0 - 18 - 0.5), 15, 15, fill=False, ec='#ffb000', lw=0.9))
clean(ax_a); scalebar(ax_a, 2 / PX, '2 µm', (60, 60))
ax_a.set_title('movie frame\n+ detected spots')
letter(ax_a, 'a', 0)
# (b) spots
ax_b = fig.add_subplot(r1[1])
order = torch.argsort(theta[:, 2])
pick = order[torch.linspace(0, len(order) - 1, 9).long()]
mos = np.full((3 * 16 - 1, 3 * 16 - 1), np.nan)
for k, i in enumerate(pick):
    r_ = spots.rois[i].cpu().numpy()
    mos[(k // 3) * 16:(k // 3) * 16 + 15, (k % 3) * 16:(k % 3) * 16 + 15] = (r_ - r_.min()) / (r_.max() - r_.min())
cmap_b = plt.get_cmap(CM).copy(); cmap_b.set_bad('white')
ax_b.imshow(mos, cmap=cmap_b, interpolation='nearest'); ax_b.axis('off')
ax_b.set_title(f'{len(spots)} spots\n(ROIs, 15 × 15 px)')
letter(ax_b, 'b', 0)
# (c) model: pupil phase
ax_c = fig.add_subplot(r1[2])
Wp = (torch.einsum('j,jnm->nm', astig.zcoef.detach(), astig.Z)).cpu().numpy()
Wp[astig.mask.cpu().numpy() == 0] = np.nan
v = np.nanmax(np.abs(Wp))
ax_c.imshow(Wp, cmap=DIV, vmin=-v, vmax=v)
ax_c.axis('off')
ax_c.set_title('vectorial model:\npupil phase')
ax_c.text(0.5, -0.08, f'Zernikes, ±{v:.2f} λ', transform=ax_c.transAxes, ha='center', va='top')
letter(ax_c, 'c', 0)
# (d) fit
i = int(pick[5])
mu = (theta[i, 3] * fitted.render(theta[i:i + 1, :3].to(dev)).cpu()[0] + theta[i, 4]).numpy()
d = spots.rois[i].cpu().numpy()
sub3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=r1[3], wspace=0.05)
axs_d = [fig.add_subplot(sub3[k]) for k in range(3)]
for ax, im, t in zip(axs_d, (d, mu, (d - mu) / np.sqrt(mu)), ('data', 'model', 'resid.')):
    if t == 'resid.':
        ax.imshow(im, cmap=DIV, vmin=-3, vmax=3, interpolation='nearest')
    else:
        ax.imshow(im, cmap=CM, vmin=0, vmax=d.max(), interpolation='nearest')
    clean(ax); ax.set_xlabel(t, labelpad=2)
axs_d[1].set_title('MLE fit\n(x, y, z, photons, bg)')
letter(axs_d[0], 'd', 0)
# (e) output: super-resolution image coloured by z
ax_e = fig.add_subplot(r1[4])
sr_px = 0.02                                                     # um per render pixel
n_sr = int(W * PX / sr_px)
zb = np.linspace(0, 0.8, 17)
img = np.zeros((n_sr, n_sr, 3))
cm = plt.get_cmap(ZCM)
for k in range(16):
    m = (locs['z'] >= zb[k]) & (locs['z'] < zb[k + 1])
    h2, _, _ = np.histogram2d(locs['y'][m] * PX / sr_px, locs['x'][m] * PX / sr_px, bins=n_sr,
                              range=[[0, n_sr], [0, n_sr]])
    img += gaussian_filter(h2, 1.0)[..., None] * np.array(cm((k + 0.5) / 16)[:3])[None, None]
img = np.clip(img / np.percentile(img.sum(-1), 99.5) * 1.3, 0, 1) ** 0.7
ax_e.imshow(img, interpolation='bilinear')
clean(ax_e); scalebar(ax_e, 2 / sr_px, '2 µm', img.shape[:2])
ax_e.set_title(f'output: {len(locs)}\nlocalizations')
cax = ax_e.inset_axes([0.05, -0.13, 0.9, 0.06])
plt.colorbar(plt.cm.ScalarMappable(Normalize(0, 800), ZCM), cax=cax, orientation='horizontal')
cax.set_xticks([0, 400, 800]); cax.set_xlabel('z (nm)', labelpad=1)
letter(ax_e, 'e', 0)

# ---- row 2: PSF sections (f) and the spline (g)
r2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1], width_ratios=[1.75, 1], wspace=0.18)
sf = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=r2[0], width_ratios=[1, 1, 1, 0.62, 1.8, 0.08],
                                      wspace=0.08, hspace=0.35)
norm = LogNorm(1e-3, 1)
for row, (vec, name, zr) in enumerate([(astig, 'astigmatic', (0.0, 0.8)), (tetra, 'tetrapod-like', (0.0, 2.0))]):
    zs3 = torch.tensor([zr[0] + 0.15 * (zr[1] - zr[0]), (zr[0] + zr[1]) / 2, zr[1] - 0.15 * (zr[1] - zr[0])],
                       device=dev)
    S = sections(vec, zs3)
    vmax = S.max()
    c = vec.roisize // 2
    hw = 8 if vec is astig else 12
    for k in range(3):
        ax = fig.add_subplot(sf[row, k])
        ax.imshow(np.clip(S[k, c - hw:c + hw + 1, c - hw:c + hw + 1] / vmax, 1e-3, None), cmap=CM, norm=norm,
                  interpolation='nearest')
        clean(ax)
        ax.set_title(f'z {float(zs3[k]):.2f} µm', pad=2)
        if k == 0:
            ax.set_ylabel(name, labelpad=3)
            if row == 0:
                letter(ax, 'f', 1)
        if k == 2:
            scalebar(ax, 1 / PX, '1 µm', (2 * hw + 1, 2 * hw + 1))
    zz = torch.linspace(zr[0], zr[1], 61, device=dev)
    V = sections(vec, zz)
    ax = fig.add_subplot(sf[row, 4])
    im = ax.imshow(np.clip(V[:, c, :] / V.max(), 1e-3, None), cmap=CM, norm=norm, aspect='auto',
                   extent=[-(c + 0.5) * PX, (c + 0.5) * PX, zr[1], zr[0]])
    ax.set_title('xz section', pad=2); ax.set_ylabel('z (µm)', labelpad=1)
    if row == 1:
        ax.set_xlabel('x (µm)', labelpad=1)
    else:
        ax.set_xticklabels([])
cb = fig.colorbar(im, cax=fig.add_subplot(sf[:, 5]))
cb.set_label('PSF / peak (log)', labelpad=2)
# (g) the convolutional spline: knots + compact schematic
sg = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=r2[1], height_ratios=[1.35, 1], hspace=0.12)
ax = fig.add_subplot(sg[0])
K = fitted.knots.cpu().numpy()
kz = int(np.argmin(np.abs(fitted.zs.cpu().numpy() - 0.4)))
c0 = (K.shape[1] - 28) // 2
ax.imshow(K[kz, c0:c0 + 28, c0:c0 + 28], cmap=CM, interpolation='nearest')
for t in np.arange(-0.5, 28, 2):
    ax.axhline(t, color='w', lw=0.25, alpha=0.45); ax.axvline(t, color='w', lw=0.25, alpha=0.45)
ax.add_patch(Rectangle((6.5, 6.5), 4, 4, fill=False, ec='#38c6ff', lw=1.2))
clean(ax)
ax.set_title('ConvSplinePSF: knots at z = 400 nm\n(2 per px; blue: one 4 × 4 × 4 kernel)', pad=2)
letter(ax, 'g', 1)
ax = fig.add_subplot(sg[1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for x, t in [(0.0, 'knot block\n(gather)'), (0.35, 'conv2d\nstride 2'), (0.70, 'ROIs and\nderivatives')]:
    ax.add_patch(FancyBboxPatch((x + 0.01, 0.42), 0.28, 0.5, boxstyle='round,pad=0.01', fc='#f2f2f2', ec='0.6',
                                lw=0.6))
    ax.text(x + 0.15, 0.67, t, ha='center', va='center', fontsize=8)
for x in (0.29, 0.64):
    ax.annotate('', xy=(x + 0.07, 0.67), xytext=(x + 0.005, 0.67),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='0.35'))
ax.text(0.5, 0.18, 'one call per batch of emitters; autograd\ngives the gradient on the knots',
        ha='center', va='center', fontsize=8, color='0.25')

# ---- row 3: in-situ estimate (h) and CRLB (i)
r3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[2], width_ratios=[1.35, 1], wspace=0.3)
ax = fig.add_subplot(r3[0])
nl = list(range(5, 22))
tv = np.array([truth.get(j, 0.0) for j in nl])
sv = start.zcoef.detach().cpu().numpy()
ev = est.psf.zcoef.detach().cpu().numpy()
x = np.arange(len(nl))
ax.bar(x - 0.28, 1000 * sv, 0.28, color='0.72', label='start')
ax.bar(x, 1000 * tv, 0.28, color='#3b6fd8', label='truth')
ax.bar(x + 0.28, 1000 * ev, 0.28, color='#e8590c', label='in situ')
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(nl); ax.set_xlabel('Zernike mode (Noll index)')
ax.set_ylabel('aberration (mλ rms)')
ax.legend(frameon=False, ncol=3, loc='upper right')
err = float(np.sqrt(np.mean((ev - tv) ** 2)))
ax.set_title(f'in-situ estimate from {len(spots)} spots: rms error {1000 * err:.1f} mλ', pad=3)
ax.spines[['top', 'right']].set_visible(False)
letter(ax, 'h', 2)
ax = fig.add_subplot(r3[1])
zz = torch.linspace(0.02, 0.78, 39, device=dev)
th = torch.stack([torch.zeros_like(zz), torch.zeros_like(zz), zz, torch.full_like(zz, 4000.0),
                  torch.full_like(zz, 30.0)], 1)
cr = tp.crlb(fitted, th).cpu().numpy()
z_ = zz.cpu().numpy() * 1000
ax.plot(z_, cr[:, 0] * PX * 1000, color='#e8a000', label='x')
ax.plot(z_, cr[:, 1] * PX * 1000, color='#3b6fd8', label='y')
ax.plot(z_, cr[:, 2] * 1000, color='#c2255c', label='z')
ax.set_xlabel('emitter height z (nm)'); ax.set_ylabel('CRLB (nm)'); ax.set_ylim(0, None); ax.set_xlim(0, 800)
ax.legend(frameon=False, loc='upper left')
ax.set_title('Cramér-Rao bound (4000 photons, 30 bg / px)', pad=3)
ax.spines[['top', 'right']].set_visible(False)
letter(ax, 'i', 2)

# arrows of the flow row (after the layout is fixed)
fig.canvas.draw()
arrow(ax_a, ax_b); arrow(ax_b, ax_c); arrow(ax_c, axs_d[0]); arrow(axs_d[2], ax_e)
place_letters()

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out + '.png', dpi=200)
try:   # palette png (smaller) when Pillow is available
    from PIL import Image
    im = Image.open(a.out + '.png').convert('RGB').quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    im.save(a.out + '.png', optimize=True)
except Exception:
    pass
print('written', a.out + '.png', os.path.getsize(a.out + '.png') // 1000, 'kB')
