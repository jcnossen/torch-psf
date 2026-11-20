"""
Quick start for torch_psf: build a spline PSF from a vectorial model, localize simulated spots on it, and
pass a likelihood gradient from the spline knots back to the Zernike coefficients of the vectorial model.

    python examples/quickstart.py        (uses CUDA when available, otherwise the CPU)
"""
import time
import torch
from torch_psf import SplinePSF
from torch_psf.fit import localize, crlb
from torch_psf.fit import poisson_nll
from torch_psf import VectorialPSF

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# 1. A spline PSF for 15 x 15 px patches, emitters 0 - 0.8 um above the coverslip.
#    Knots every 1/2 px laterally (S=2) and every 40 nm in z (hz=0.04 um).
spl = SplinePSF(roisize=15, z_range=(0.0, 0.8), oversample=2, dz_um=0.04, xy_limit=3.0, device=dev)

# 2. Fill the knots from a vectorial pupil model with some astigmatism (Noll 6, 0.1 waves rms).
#    The model is evaluated once per knot image, not once per emitter.
vec = VectorialPSF(spl.Pl, pixel_um=0.108, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68, device=dev)
with torch.no_grad():
    vec.zcoef[vec.zidx.index(6)] = 0.1
focus = 0.45                                                     # um, focal plane during the acquisition
pos_fn = lambda z: (z, torch.full_like(z, focus))                # knot z -> (emitter height, focus)
spl.knots = spl.sample(vec, pos_fn)
print(f'knot array {tuple(spl.knots.shape)}, {spl.knots.numel() * 4 / 1e6:.1f} MB')

# 3. Simulate 20,000 spots with Poisson noise on the spline.
N = 20000
th_true = torch.stack([torch.rand(N, device=dev) - 0.5,          # x offset from the patch centre (px)
                       torch.rand(N, device=dev) - 0.5,          # y offset (px)
                       0.05 + 0.7 * torch.rand(N, device=dev),   # z (um)
                       torch.full((N,), 2000., device=dev),      # photons
                       torch.full((N,), 10., device=dev)], 1)    # background photons / px
with torch.no_grad():
    mu = th_true[:, 3, None, None] * spl.render_batched(th_true[:, :3]) + th_true[:, 4, None, None]
patches = torch.poisson(mu)

# 4. Maximum-likelihood localization on the spline (Levenberg-Marquardt, value and x/y/z derivatives from one conv2d).
t = time.time()
th, nll = localize(spl, patches)
if dev == 'cuda':
    torch.cuda.synchronize()
err = (th - th_true)[:, :3] * torch.tensor([108., 108., 1000.], device=dev)     # nm
cr = crlb(spl, th_true)[:, :3] * torch.tensor([108., 108., 1000.], device=dev)
print(f'localized {N} spots in {time.time() - t:.2f} s')
print('rms error x/y/z (nm):', [round(v, 1) for v in err.pow(2).mean(0).sqrt().tolist()])
print('mean CRLB x/y/z (nm):', [round(v, 1) for v in cr.mean(0).tolist()])

# 5. Gradient of the total negative log-likelihood with respect to the Zernike coefficients, through the knots:
#    (a) d NLL / d knots by autograd through the spline, summed over all spots,
#    (b) one vector-Jacobian product through the vectorial model that built the knots.
#    Evaluated at a PSF model that is slightly off (as during a fit): +0.02 waves on Noll 5 and 11.
with torch.no_grad():
    vec.zcoef[vec.zidx.index(5)] += 0.02
    vec.zcoef[vec.zidx.index(11)] += 0.02
spl.knots = spl.sample(vec, pos_fn)
t = time.time()
K = spl.knots.clone().requires_grad_(True)
for s in range(0, N, 4000):
    sl = slice(s, s + 4000)
    m = th[sl, 3, None, None] * spl.render(th[sl, :3], knots=K) + th[sl, 4, None, None]
    poisson_nll(patches[sl], m).sum().backward()
g_knots = spl.vjp_vectorial(vec, pos_fn, K.grad)['zcoef']
if dev == 'cuda':
    torch.cuda.synchronize()
print(f'Zernike gradient over {N} spots through the knots: {time.time() - t:.3f} s')

# The same gradient with the vectorial model evaluated for every spot (the slow route), on the first 2000 spots.
n = 2000
small = VectorialPSF(spl.roisize, pixel_um=0.108, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68, device=dev)
with torch.no_grad():
    small.zcoef.copy_(vec.zcoef)
K2 = spl.knots.clone().requires_grad_(True)
m = th[:n, 3, None, None] * spl.render(th[:n, :3], knots=K2) + th[:n, 4, None, None]
poisson_nll(patches[:n], m).sum().backward()
g_spline = spl.vjp_vectorial(vec, pos_fn, K2.grad)['zcoef']
m = small(th[:n, :3], torch.full((n,), focus, device=dev), intensity=th[:n, 3], bg=th[:n, 4])
g_direct, = torch.autograd.grad(poisson_nll(patches[:n], m).sum(), small.zcoef)
cos = torch.nn.functional.cosine_similarity(g_spline, g_direct, dim=0)
print(f'knot route vs per-spot vectorial route ({n} spots): cosine similarity {cos:.6f}')
