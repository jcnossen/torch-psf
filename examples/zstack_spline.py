"""A spline PSF directly from a measured (here: generated) z-stack: ConvCubicSplinePSF.from_zstack (oversample 1, the
stack planes are the knots), then localization and the CRLB on it.

    python examples/zstack_spline.py
"""
import numpy as np, torch
import torch_psf as tp

# an astigmatic Gaussian "bead stack": 41 planes, 50 nm apart, 25 x 25 px, emitter on the centre pixel
gauss = tp.AstigmaticGaussianPSF(25, sigma_px=1.4, gamma_um=0.25, d_um=0.4, z_range=(-1, 1))
zs = torch.linspace(-1.0, 1.0, 41, device=gauss.dev)
stack = gauss.render(torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], 1))

spline = tp.ConvCubicSplinePSF.from_zstack(stack, z0_um=-1.0, dz_um=0.05, roisize=13)
print(spline)
th = torch.tensor([[0.3, -0.2, 0.25, 2000.0, 10.0]], device=gauss.dev).repeat(2000, 1)
mu = th[:, 3, None, None] * spline.render(th[:, :3]) + th[:, 4, None, None]
L = spline.fit(torch.poisson(mu))                   # a Dataset; this spline has no pixel size, so z stays in um
fit = torch.cat([L.pos.float(), L.photons[:, None], L.bg[:, None]], 1)
sd = (fit - th.cpu()).std(0)
cr = spline.crlb(th[:1])[0].cpu()
for k, n in enumerate(['x (px)', 'y (px)', 'z (um)', 'photons', 'bg']):
    print(f'{n:8s} std {float(sd[k]):.4f}  CRLB {float(cr[k]):.4f}')
