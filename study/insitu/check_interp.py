"""Interpolation error of the knot spline against the vectorial model, for several knot spacings."""
import sys, time, torch
from torch_psf import VectorialPSF
from torch_psf import ConvCubicSplinePSF
dev = 'cuda'
P, F0 = 15, 0.45
kw = dict(pixel_um=0.108, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68, npupil=64, oversample=3, device=dev)
small = VectorialPSF(P, **kw)
zc = torch.zeros(17); zc[1] = 0.10; zc[6] = 0.03
torch.manual_seed(0)
N = 2000
off = torch.cat([torch.rand(N, 2) - 0.5, torch.rand(N, 1) * 0.8], 1).to(dev)
with torch.no_grad():
    small.zcoef.copy_(zc.to(dev))
    ref = torch.cat([small(off[s:s + 250], torch.full((len(off[s:s+250]),), F0, device=dev)) for s in range(0, N, 250)])
for S in (1, 2, 3, 4):
    for hz in (0.08, 0.04, 0.02):
        spl = ConvCubicSplinePSF(P, (0.0, 0.8), oversample=S, dz_um=hz, xy_limit=1.5, device=dev)
        big = VectorialPSF(spl.Pl, **kw)
        with torch.no_grad(): big.zcoef.copy_(zc.to(dev))
        t = time.time()
        spl.knots = spl.knots_from_vectorial(big, lambda z: (z, torch.full_like(z, F0)))
        torch.cuda.synchronize(); tb = time.time() - t
        g = spl.render_batched(off)
        err = (g - ref).flatten(1)
        rel = err.norm(dim=1) / ref.flatten(1).norm(dim=1)
        # likelihood cost of the interpolation error for a 4000-photon emitter on 30 photons/px background
        mu = 4000 * ref + 30; chi = ((4000 * err.view_as(ref)) ** 2 / mu).flatten(1).sum(1) / 2
        print(f'S {S} hz {hz:.2f} knots {spl.knots.numel():7d} build {tb:5.2f}s  rel L2 err median {rel.median():.2e} max {rel.max():.2e}  '
              f'nll cost @4000ph {chi.median():.3f}')
