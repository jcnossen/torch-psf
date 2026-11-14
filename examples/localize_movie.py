"""Localize a movie with a given PSF: psf.localize(movie) -> a localization table (numpy structured array).

A movie is simulated with an astigmatic vectorial PSF; the same PSF, converted to a ConvCubicSplinePSF for speed,
localizes it. The errors are compared with the Cramer-Rao lower bound.

    python examples/localize_movie.py           (command line: python -m torch_psf localize movie.tif --psf psf.json -o locs.csv)
"""
import numpy as np, torch
import torch_psf as tp
from torch_psf.simulate import simulate_movie

vec = tp.VectorialPSF(23, pixel_um=0.108, NA=1.49, wavelength_um=0.68, focus_um=0.3, z_range=(0.0, 0.6))
vec.set_zernike({6: 0.1})
g = torch.Generator(device=vec.dev).manual_seed(1)
movie, events = simulate_movie(lambda p: vec.forward_batched(p, torch.full((len(p),), 0.3, device=vec.dev)), 23,
                               T=2000, H=96, W=96, nsites=300, p_on=0.003, mean_on=2.0, bg=20.0, photons=3000.0,
                               z_range=(0.05, 0.55), generator=g, device=vec.dev)
spline = vec.to_spline(roisize=13)                                  # same PSF, fast: one conv2d per batch
locs = spline.localize(movie.numpy(), isolation=8)
print(f'{len(locs)} localizations of {len(events)} events; columns: {locs.dtype.names}')

ev = events.cpu().numpy()
err = []
for r in locs:
    e = ev[ev[:, 0] == r['frame']]
    d = np.hypot(e[:, 1] - r['x'], e[:, 2] - r['y'])
    if len(d) and d.min() < 1:
        k = d.argmin()
        err.append((r['x'] - e[k, 1], r['y'] - e[k, 2], r['z'] - e[k, 3], r['crlb_x'], r['crlb_y'], r['crlb_z']))
err = np.array(err)
nm = np.array([108, 108, 1000])
print(f'matched {len(err)}; rms error x / y / z (nm): {np.sqrt((err[:, :3] ** 2).mean(0)) * nm}')
print(f'rms CRLB      x / y / z (nm): {np.sqrt((err[:, 3:] ** 2).mean(0)) * nm}')
print(f"fraction at the offset limit: {locs['at_limit'].mean():.4f}")
np.save('locs.npy', locs)
