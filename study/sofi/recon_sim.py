"""PSF-aware SOFI reconstruction vs standard SOFI2 on a simulated line-pair sample (known truth).
Pairs of parallel vertical lines with separations 60 .. 200 nm, dense blinking emitters on them, aberrated PSF.
Compared: the mean image, standard cross-cumulant SOFI2 (with / without Fourier reweighting), and the PSF-aware
reconstruction (torch_psf.sofi.reconstruct) with the true PSF and with an aberration-free PSF. Output: a figure and
line-pair contrast per method: 1 - (profile midway) / (mean profile at the two lines); > 0 = resolved dip.

python recon_sim.py   -> ~/sofi_psf_out/recon_sim.png, results/recon_sim.json
"""
from common import *
from torch_psf.sofi.reconstruct import reconstruct, sofi2_xc
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

torch.manual_seed(0)
H, W, T, Q = 48, 80, 8000, 15
SEPS_NM = [60, 90, 120, 160, 200]
zern = random_aberrations(0.07, seed=3)
psf = make_psf(Q, zern)
spec = DenseMovieSpec(H=H, W=W, T=T, density_um2=1e-6, photons=400, photons_logsd=0.3, bg=60,
                      blink=BlinkModel(duty=0.15, t_on=2.0), Q=Q, seed=1)
mv = DenseMovie(psf, spec)
g = torch.Generator(device=DEV).manual_seed(5)
xs, ys = [], []
lin_density = 40.0 / psf.pixel_um / 1e0 * 0.108          # ~40 emitters per um per line
for i, s_nm in enumerate(SEPS_NM):
    xc = 10 + 15 * i
    sp = s_nm / 1e3 / psf.pixel_um
    for off in (-sp / 2, sp / 2):
        n = int(40 * (H - 8) * psf.pixel_um)
        ys.append(4 + (H - 8) * torch.rand(n, generator=g, device=DEV))
        xs.append(torch.full((n,), xc + off, device=DEV))
mv.x, mv.y = torch.cat(xs), torch.cat(ys)
mv.n = len(mv.x)
mv.z = torch.zeros(mv.n, device=DEV)
mv.eps = spec.photons * torch.exp(spec.photons_logsd * torch.randn(mv.n, generator=g, device=DEV))
acc = CumulantAccumulator(H, W, R=5, block=500)
for t0, adu in mv.frames():
    acc.update(mv.to_photons(adu))
acc.finish()
res = acc.result()
var = sofi.fit.smooth_variance(acc.bootstrap(100).var(0))
flat = make_psf(Q, None)
sig = 0.21 * psf.wavelength_um / float(psf.NA) / psf.pixel_um
out = {'mean': (res['mean'], 1), 'SOFI2 (xc)': (sofi2_xc(res['G'], res['mask'], res['D'], sig), 2),
       'SOFI2 + reweighting': (sofi2_xc(res['G'], res['mask'], res['D'], sig, reweight=0.02), 2)}
for name, p, l1, sm in [('PSF-aware, true PSF', psf, 0.0, 0.0), ('PSF-aware, true PSF, l1', psf, 0.02, 0.0),
                        ('PSF-aware, no-aberration PSF', flat, 0.0, 0.0)]:
    t0 = time.time()
    r = reconstruct(res['G'], res['mask'], p, var, os=4, Q=Q, D=res['D'], l1=l1, smooth=sm, iters=800)
    print(f'{name}: {time.time() - t0:.0f}s loss {r["loss"]:.4e}', flush=True)
    out[name] = (r['image'][0], 4)


def contrast(img, up):
    """per line pair: 1 - (min between the lines) / (mean of the two maxima) of the profile averaged along y."""
    prof = img[int(8 * up):int((H - 8) * up)].mean(0).cpu().numpy()
    cs = []
    for i, s_nm in enumerate(SEPS_NM):
        xc, sp = 10 + 15 * i, s_nm / 1e3 / psf.pixel_um
        X = np.arange(len(prof)) / up                 # fine index j = frame position j / up (pixel centres)
        a = np.interp([xc - sp / 2, xc, xc + sp / 2], X, prof)
        cs.append(float(1 - a[1] / (0.5 * (a[0] + a[2]))))
    return cs, prof


summary = {}
fig, ax = plt.subplots(len(out), 2, figsize=(13, 2.3 * len(out)), gridspec_kw=dict(width_ratios=[2.2, 1]))
for k, (name, (img, up)) in enumerate(out.items()):
    cs, prof = contrast(img, up)
    summary[name] = dict(zip(map(str, SEPS_NM), cs))
    print(f'{name:32s} ' + ' '.join(f'{s}nm:{c:+.2f}' for s, c in zip(SEPS_NM, cs)), flush=True)
    a = img.cpu().numpy()
    ax[k, 0].imshow(a, cmap='magma', vmin=0, vmax=np.percentile(a, 99.7), interpolation='nearest')
    ax[k, 0].set_title(name, fontsize=9); ax[k, 0].axis('off')
    X = np.arange(len(prof)) / up
    ax[k, 1].plot(X, prof / prof.max(), lw=1)
    for i, s_nm in enumerate(SEPS_NM):
        sp = s_nm / 1e3 / psf.pixel_um
        for o in (-sp / 2, sp / 2):
            ax[k, 1].axvline(10 + 15 * i + o, color='0.7', lw=0.5)
    ax[k, 1].set_xlim(0, W); ax[k, 1].set_yticks([])
ax[-1, 1].set_xlabel('x (px); line pairs ' + ', '.join(f'{s}' for s in SEPS_NM) + ' nm')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'recon_sim.png'), dpi=110)
save_json(dict(seps_nm=SEPS_NM, contrast=summary, aberration_rms=float(np.sqrt((zern ** 2).sum()))), 'recon_sim.json')
