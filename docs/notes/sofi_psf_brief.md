# Brief: PSF estimation from dense blinking movies, simulation study (branch `sofi-psf`)

Written 2026-11-15 by the laptop session that wrote `docs/notes/psf_estimation_theory.md`. Read that note first,
especially §2 (symmetries), §5 (cumulants) and §6 (proposed estimator). This brief is the task for a session on
the TU Delft server.

## Goal

Find out, on simulations, whether the aberrations (Noll 5-21) of a vectorial PSF can be estimated from a movie of
blinking emitters that are too dense to localize, using second-order cross-cumulants. Then build a prototype
estimator. It has to work for permanently dense data (SOFI: fluorescent proteins, quantum dots) as well as for
dense SMLM movies. The only requirement on the data is independent blinking. Real data (the EPFL challenge
SOFI and SIM-SOFI sets) comes later, not in this task.

## Setup and rules

* Work in a separate worktree, because `~/torch-spline-3d` has `master` checked out and other workers may have
  uncommitted changes there:
  `cd ~/torch-spline-3d && git worktree add ~/torch-spline-3d-sofi sofi-psf`
* Commit to `sofi-psf` only. Do not merge into `master`. The laptop relays pushes to GitLab/GitHub when asked.
* `simcode2` owns `torch_psf` on `master`. To keep merges easy, put new code in new files
  (for example `torch_psf/fluct/` for library code, `study/sofi/` for study scripts). Change existing modules only
  where unavoidable, and say so in the commit message.
* Nothing generated goes into the repo: no simulated movies, no reference outputs, no large binaries. Scripts
  regenerate everything.
* Run on the GPU. Long runs go in the background.

## Steps

1. **Simulator** for dense blinking movies with a known PSF (reuse `torch_psf.simulate` where possible):
   * emitter density (thin 2D sample at one z, plus an option for a z range);
   * blinking as a two- or three-state Markov chain, including blinking faster than the frame time (integrate
     within the frame), and bleaching;
   * camera: per-pixel offset, gain and read noise (sCMOS), an optional row/column-correlated read noise, an
     EMCCD option; optional drift;
   * PSF: vectorial, with Noll 5-21 aberrations and an optional astigmatic bias.
2. **Cumulant accumulator**, streaming and per block of frames: the mean and the zero-lag covariances
   $G(\mathbf m, \mathbf d)$ for all separations $0 < \lvert\mathbf d\rvert \le R$, plus an optional lag-1
   covariance. Check it against the analytic expectation from the simulator truth,
   $G = \rho * h_{\mathbf d}$ with pixel-integrated PSFs (theory note §5.2).
3. **Fisher information / CRLB** for the Zernikes from $G$, with $\rho$ profiled out (theory note §3 and §6).
   Get the noise covariance of the cumulant estimates from a block bootstrap on simulated movies. Sweep density,
   number of frames, z spread and astigmatic bias. Report which modes are determinable, and how well.
   **Do this before step 4.** If it says the method cannot work, stop and report.
4. **Prototype estimator**: the closed-form elimination of $\rho$ across separations (theory note §6, step 3),
   then a non-negative least-squares refinement of $\rho$. Test recovery against the truth and compare the
   errors with the CRLB of step 3. Also test the confounders one at a time: bleaching, drift, correlated read
   noise.
5. **Optional, if there is time:** the sparse-tail cross-check. Simulate a movie that is dense early and sparse
   late, run the new estimator on the dense part and the existing `InSituEstimator` on the sparse part, and
   compare.

## Done means

* Code and scripts committed on `sofi-psf`.
* A results note `docs/notes/sofi_psf_study.md` with:
  * the numbers (CRLB per mode and recovered error per scenario);
  * what works and what does not;
  * the open questions.
* Tell the user what is ready to push, and to which remotes.
