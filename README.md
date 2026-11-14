# torch-spline-3d (`torch_convspline`)

A 3D cubic-spline point spread function (PSF) for single-molecule localization microscopy, written in plain
PyTorch. A batch of spline-PSF patches is evaluated with a single grouped, strided `conv2d` call. The spline is
differentiable with respect to the emitter positions and also with respect to the PSF itself.

The main use is **in-situ PSF estimation**: fitting a vectorial pupil model to the blinking data itself rather than
to a bead stack. The pupil model is evaluated only on the spline knots. All emitters are evaluated on the spline,
the likelihood gradient is summed on the knots, and one backward pass through the pupil model turns that sum into
the gradient for the pupil parameters. The cost of the pupil model then does not grow with the number of emitters.

The method and the results are in the paper draft
[docs/paper/insitu_spline_psf.pdf](docs/paper/insitu_spline_psf.pdf): *Differentiable spline for efficient in-situ
estimation of point spread functions* (J. Cnossen, draft).

## Why use it

- **The PSF values are the parameters.** The spline is stored as its knot values, one number per knot. Every
  update of the knots is again a continuous spline, so the PSF can be optimized directly or back-propagated
  through. The conventional spline PSF stores 64 polynomial coefficients per voxel. That form is differentiable
  in the emitter position only: gradient descent on the coefficients does not give a continuous function.
- **No custom GPU kernel.** One evaluation is an indexed gather of knot windows plus one call of `conv2d` with
  groups and stride. The value and the x, y, z derivatives come out of the same call. The gradient on the knots
  comes from autograd, so it runs on any device PyTorch supports, and the spline can sit inside any other
  differentiable model.
- **Fast.** On an RTX 3090 with 15×15 patches, the spline runs at 7.1 million spots per second for the value,
  4.1 million with the x/y/z derivatives, and 2.6 million with the gradient on the PSF values. That is 1.2 to
  1.9 times faster than the same spline as a coefficient table in PyTorch. The stored PSF is 51 times smaller
  (0.3 MB against 16 MB).
- **In-situ fits at the cost of a spline.** For a pupil model whose own numerical error matches that of the
  spline (128-pixel pupil, 4× oversampling, 0.15 % error), the gradient for the Zernike coefficients is 21 times
  cheaper at 10⁴ emitters and 40 times cheaper at 10⁵ emitters. A whole iteration, gradient plus
  re-localization, is about 6 times cheaper. Below about 270 emitters there is no gain. Away from the optimum the
  knot-route gradient equals the per-emitter gradient to 0.2 % (cosine similarity 1.000000).
- **Beyond Zernike terms.** The same route fits a free pixel map of pupil phase and amplitude. It also fits free
  knots on top of a pupil model, or free knots on their own. The paper shows when each of these helps and when it
  does not.

The numbers above are from the paper's simulation study (`study/insitu/`, 3 seeds). The paper's limitations
section lists what has not been tested, for example field-dependent PSFs and a comparison with uiPSF or other
in-situ codes on the same data.

## Install

Requires Python 3, PyTorch (CUDA optional), numpy, matplotlib and tqdm.

```bash
git clone https://gitlab.com/jcnossen/torch-spline-3d.git
```

```bash
cd torch-spline-3d && pip install -e .
```

The code is also mirrored at https://github.com/jcnossen/torch-psf. The `insitu-study` branch has the current
paper draft, the clean benchmarks and the real-data scripts. `master` has the library and the first part of the
simulation study.

## Quick start

[examples/quickstart.py](examples/quickstart.py) runs in a few seconds on a GPU and in about a minute on the CPU:

```bash
python examples/quickstart.py
```

It builds a spline from a vectorial model, simulates 20,000 noisy spots, localizes them, and computes the Zernike
gradient through the knots. It then checks that gradient against evaluating the vectorial model for every spot.
The core of it:

```python
import torch
from torch_convspline.knotpsf import KnotPSF, localize, crlb, poisson_nll
from torch_convspline.vectorial import VectorialPSF

# spline for 15x15 px patches, z from 0 to 0.8 um, knots every 1/2 px (S=2) and every 40 nm (hz=0.04 um)
spl = KnotPSF(P=15, zlo=0.0, zhi=0.8, S=2, hz=0.04, xmax=3.0, device='cuda')

# fill the knots from a vectorial pupil model (one evaluation per knot image, not per emitter)
vec = VectorialPSF(spl.Pl, pixelsize=0.108, NA=1.49, n_imm=1.518, n_med=1.33, wavelength=0.68)
pos_fn = lambda z: (z, torch.full_like(z, 0.45))       # knot z -> (emitter height, focus), um
spl.knots = spl.stack_from_vectorial(vec, pos_fn)

# evaluate: off = [N, 3] (x px, y px, z um) relative to the patch centre -> [N, 15, 15], normalized PSF
psf = spl.evaluate(off)
psf, dx, dy, dz = spl.evaluate(off, grad=True)         # value and derivatives from one conv2d

# maximum-likelihood fit of (x, y, z, photons, bg) for patches [N, 15, 15], and the CRLB
theta, nll = localize(spl, patches)
sd = crlb(spl, theta)

# Zernike gradient for all spots: autograd onto the knots, then one vector-Jacobian product
K = spl.knots.clone().requires_grad_(True)
mu = theta[:, 3, None, None] * spl.evaluate(theta[:, :3], knots=K) + theta[:, 4, None, None]
poisson_nll(patches, mu).sum().backward()
g_zernike = spl.vjp_vectorial(vec, pos_fn, K.grad)['zcoef']
```

Output on a laptop RTX 4060:

```
localized 20000 spots in 6.62 s
rms error x/y/z (nm): [10.1, 15.9, 32.3]
mean CRLB x/y/z (nm): [9.6, 13.2, 27.5]
Zernike gradient over 20000 spots through the knots: 1.317 s
knot route vs per-spot vectorial route (2000 spots): cosine similarity 1.000000
```

The timings include the first-call overhead of PyTorch. The benchmarks in `study/insitu/bench_*.py` give
steady-state numbers.

### Main pieces

| | |
|---|---|
| `torch_convspline/knotpsf.py` | `KnotPSF`: knots on an oversampled grid, evaluation by strided grouped `conv2d`, knots from a vectorial model (`stack_from_vectorial`), and the vector-Jacobian product back to its parameters (`vjp_vectorial`). Also `localize` (Levenberg-Marquardt on the spline), `crlb` and `poisson_nll`. |
| `torch_convspline/vectorial.py` | `VectorialPSF`: batched, differentiable vectorial pupil model. It has Zernike aberrations (Noll), a free pupil phase map and a free pupil amplitude map, supercritical-angle fluorescence, and optionally fitted NA and medium index. |
| `torch_convspline/conv_spline.py`, `psf.py` | The original interface: `SplinePSF(zstack, roi_shape)` turns a measured z-stack (for example beads) directly into a spline PSF with `forward`, `forward_with_deriv` and `crlb`. `examples/simple.py` uses it; its viewer at the end needs `smlmtorch`. |

### Things to know

- **Offset limit `xmax`.** `KnotPSF` stores the PSF on a support that allows emitter offsets up to `xmax` pixels
  from the patch centre, and `localize` clamps at that limit. For defocused, aberrated spots the detection peak can
  lie up to 3 px from the emitter. With a limit that is too small, a flexible PSF model absorbs the clamped spots
  as a z-dependent position error. Use `xmax=3` or more for patches centred on detections, and check how many fits
  sit at the limit. The paper describes this pitfall in detail.
- **What blinking data cannot determine.** A lateral shift of the PSF with z, the z scale, and the absolute
  brightness are not fixed by blinking data alone. Free knots keep whatever their starting model had. A pupil
  model ties these to the PSF shape, and a second measurement fixes them, for example a structured-illumination
  pattern (`study/insitu/run_pattern.py`). See the paper's section on this.
- **Spline accuracy.** With S=2 and 40 nm in z, the spline differs from the pupil model by 0.19 % (relative L2);
  with S=1 the difference is about 2 %. Near the optimum this is visible in the gradient. It starts to matter at
  about 10⁶ emitters, where a finer spline (S=3) or a final step with the per-emitter route would be needed.

## Reproducing the study

Simulation study (`study/insitu/`, run from the repository root, GPU required):

```bash
python study/insitu/run_all.py 0
```

That command runs one seed of all accuracy results and writes `study/insitu/out/all_seed0.json`. The paper uses
seeds 0, 1 and 2. The speed benchmarks are `bench_speed3.py`, `bench_conventional.py` and `bench_split.py`. The
figures and LaTeX tables are made by `make_figs.py` and `make_paper.py`. `docs/insitu_spline_psf.md` is the first
version of the study and is kept as a record. `docs/CRITIC_INSITU_1.md` is the independent review that led to the
current version.

Real data (`study/realdata/`): nuclear pores from the EPFL SMLM challenge 2016 (NPC-A647-3D) and a microtubule
data set. The data are not included. `README_NPC.md` and `README_Tubulin.md` there describe the inputs, the steps
and the output files.

## License

MIT, see [LICENSE](LICENSE).
