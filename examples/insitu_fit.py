"""In-situ PSF estimation from a (simulated) SMLM movie: torch_psf.InSituEstimator (knot route).

The movie is simulated with a known aberration set; the fit starts from a nominal astigmatic model (as a bead
calibration made on another day would be) and recovers the aberrations from the blinking emitters themselves.

    python examples/insitu_fit.py           (a GPU makes this take seconds instead of minutes)
"""
import torch
import torch_psf as tp
from torch_psf.simulate import simulate_movie

truth = {5: 0.02, 6: 0.10, 7: 0.03, 8: -0.02, 11: 0.04}
true = tp.VectorialPSF(23, pixel_um=0.108, NA=1.49, wavelength_um=0.68, focus_um=0.45, z_range=(0.0, 0.8))
true.set_zernike(truth)
g = torch.Generator(device=true.dev).manual_seed(0)
f0 = torch.full((1,), 0.45, device=true.dev)
movie, events = simulate_movie(lambda p: true.forward_batched(p, f0.expand(len(p))), 23, T=3000, H=128, W=128,
                               nsites=700, p_on=0.002, mean_on=3.0, bg=30.0, photons=4000.0, z_range=(0.0, 0.8),
                               generator=g, device=true.dev)
print(f'movie {tuple(movie.shape)}, {len(events)} emitter events')

start = tp.VectorialPSF(15, pixel_um=0.108, NA=1.49, wavelength_um=0.68, focus_um=0.45).set_zernike({6: 0.08})
est = tp.InSituEstimator(start=start, roisize=15, z_range=(0.0, 0.8), nspots=6000, nheldout=3000, isolation=14,
                         iters=60, refine=True,
                         progress=lambda st, it, v, sec: it % 20 == 0 and print(f'  {st}: it {it}  {v:.1f}  {sec:.1f} s'))
psf = est.estimate(movie)
print(psf)
vec = psf if isinstance(psf, tp.VectorialPSF) else est.data['vectorial']
for j in sorted(set(truth) | {9, 10, 12}):
    i = vec.zidx.index(j)
    print(f'  Noll {j:2d}: true {truth.get(j, 0.0):+.3f}  start {float(start.zcoef[i]):+.3f}  fitted {float(vec.zcoef[i].detach()):+.3f} waves')
print('held-out NLL per pixel (lower is better):',
      {k: round(v['nll_per_px'], 5) for k, v in est.report['heldout'].items()})
print(f"fits at the offset limit: {est.report['at_limit']:.4f}; z scale vs the start: {est.report['z_scale']:.3f}")
print('chosen:', est.report['chosen'], est.report.get('refine'), est.report['warnings'])
psf.save('insitu_psf.json')
