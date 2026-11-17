"""SOFIEstimator: the PSF (Zernike aberrations) from a dense blinking movie, by its second-order cross-cumulants.

For SOFI data (permanently dense, blinking fluorophores: fluorescent proteins, quantum dots) and for SMLM movies too
dense to localize; the only requirement on the data is that the emitters blink independently. The movie is read
once, in chunks; per block of frames (bleaching) the zero-lag covariances G(r, d) of all pixel pairs 0 < |d| <= R are
accumulated (torch_psf.sofi.cumulants), weighted by their block-bootstrap variances, and fitted by the model
G = sum_p rho_p (*) h_p h_p(. + d) with a non-negative density rho on a sub-pixel grid and the vectorial PSF
(torch_psf.sofi.fit.CumulantFit). Shot noise, uncorrelated read noise and the background do not enter G for d != 0;
row / column-correlated read noise does: give a dark movie (`dark`) to measure and subtract it.

Symmetries (docs/notes/psf_estimation_theory.md section 2): tip, tilt and defocus are gauge modes (Noll 5-21 are
fitted, plus a common z shift of the sample); for a thin, index-matched sample in focus the even modes have no
first-order signal and their sign is ambiguous unless the start carries a known astigmatic bias (or the sample is
defocused / at a refractive-index interface, whose supercritical light breaks the symmetry).
"""
import time
import torch
from torch_psf.estimators import Estimator, _camera, _start_vectorial
from torch_psf.movie import open_movie
from torch_psf.sofi.cumulants import CumulantAccumulator
from torch_psf.sofi.fit import CumulantFit, smooth_variance


class SOFIEstimator(Estimator):
    """PSF from a dense blinking movie (SOFI or dense SMLM).

    start (VectorialPSF or PSF file, e.g. a bead fit, carrying any known bias such as the astigmatism of a cylindrical
    lens) or optics: the starting model (zernike_indices, npupil, oversample_pupil for a start built from optics).
    camera: Camera / calibration json / None (movie in photons). focus_um: focal plane during the acquisition
    (default: the start's). z_layers_um: heights of the sample layers modelled (default one layer at z = 0, a thin
    sample on the coverslip). roi = (y0, x0, H, W): the part of the frame used (default: the whole frame; the PSF is
    assumed constant over it, use patches of ~10-20 um for field-dependent aberrations). R: largest pixel separation
    (px); block: frames per cumulant block; oversample: density grid points per px per side; roisize: PSF patch (px);
    nboot: bootstrap replicates for the weights; iters: outer iterations; dark: a dark movie (same camera, no light)
    whose cross-covariances are subtracted.

    estimate(movie, frames=None) -> VectorialPSF. report: n_frames, n_blocks, roi, zernike (dict noll -> waves),
    dz_um (fitted common z shift of the sample), loss, seconds. data: cumulants (dict n, mean, G, mask, D), var,
    rho (fitted density), fit (CumulantFit)."""

    def __init__(self, start=None, optics=None, camera=None, focus_um=None, z_layers_um=(0.0,), roi=None, R=5,
                 block=500, oversample=2, roisize=13, nboot=200, iters=40, chunk=500, dark=None,
                 zernike_indices=tuple(range(5, 22)), npupil=64, oversample_pupil=3, device=None, progress=None,
                 stop=None):
        super().__init__(device, progress, stop)
        self.start, self.optics, self.camera, self.focus_um = start, optics, _camera(camera), focus_um
        self.z_layers_um, self.roi, self.R, self.block = tuple(z_layers_um), roi, int(R), int(block)
        self.oversample, self.roisize, self.nboot, self.iters, self.chunk = oversample, roisize, nboot, iters, chunk
        self.dark = dark
        self.zernike_indices, self.npupil, self.oversample_pupil = tuple(zernike_indices), npupil, oversample_pupil
        self.data = {}

    def _photons(self, a):
        x = torch.as_tensor(a, device=self.dev).float()
        return x if self.camera is None else self.camera.to_photons(x).to(self.dev)

    def accumulate(self, movie, frames=None):
        """Stream the movie (or its `frames` = (f0, f1) range) into a CumulantAccumulator."""
        mv = open_movie(movie)
        f0, f1 = (0, mv.n) if frames is None else (int(frames[0]), int(frames[1]))
        Hf, Wf = mv.shape
        y0, x0, H, W = self.roi if self.roi is not None else (0, 0, Hf, Wf)
        acc = CumulantAccumulator(H, W, self.R, block=self.block, device=self.dev)
        for t in range(f0, f1, self.chunk):
            a = mv.read(t, min(f1, t + self.chunk))[:, y0:y0 + H, x0:x0 + W]
            acc.update(self._photons(a))
            if self._stopped():
                break
        return acc.finish()

    def estimate(self, movie, frames=None):
        self._reset()
        t0 = time.time()
        vec0 = _start_vectorial(self.start, self.optics, self.roisize, self.dev, self.zernike_indices, self.npupil,
                                self.oversample_pupil)
        if self.focus_um is not None:
            vec0.focus_um = float(self.focus_um)
        acc = self.accumulate(movie, frames)
        res = acc.result()
        G = res['G']
        if self.dark is not None:
            dk = self.accumulate(self.dark).result()
            G = G - dk['G']
        var = smooth_variance(acc.bootstrap(self.nboot).var(0)) if len(acc.blocks) > 2 else None
        fit = CumulantFit(vec0, Q=self.roisize, os=self.oversample, z_layers=self.z_layers_um, focus=vec0.focus_um,
                          device=self.dev)
        out = fit.fit(G, res['mask'], var, n_outer=self.iters, D=acc.D, log=self._log('cumulant fit'),
                      stop=self.stop)
        psf = vec0.resized(vec0.roisize)
        with torch.no_grad():
            psf.zcoef.copy_(out['zcoef'].to(psf.zcoef.dtype))
        self.report.update(n_frames=res['n'], n_blocks=len(acc.blocks), roi=self.roi, dz_um=out['dz'],
                           loss=out['loss'], zernike={int(j): float(c) for j, c in zip(psf.zidx, psf.zcoef.detach())},
                           seconds=time.time() - t0)
        self.data.update(cumulants=res, var=var, rho=out['rho'], fit=fit, psf=psf)
        return psf

    def reconstruct(self, os=4, l1=0.0, smooth=0.0, iters=500, psf=None, standard=True):
        """SOFI image from the cumulants of the last estimate() with the estimated PSF (or `psf`), on an os x finer
        grid (torch_psf.sofi.reconstruct.reconstruct; l1 / smooth: priors). standard: also the standard cross-cumulant
        SOFI2 image (2x grid) with and without Fourier reweighting. Returns dict(image, sofi2, sofi2_rw, mean)."""
        from torch_psf.sofi.reconstruct import reconstruct, sofi2_xc
        if 'cumulants' not in self.data:
            raise RuntimeError('run estimate() first')
        res, psf = self.data['cumulants'], psf or self.data['psf']
        r = reconstruct(res['G'], res['mask'], psf, self.data['var'], os=os, Q=self.roisize, z_layers=self.z_layers_um,
                        focus=psf.focus_um, dz=self.report.get('dz_um', 0.0), D=res['D'], l1=l1, smooth=smooth,
                        iters=iters)
        out = dict(image=r['image'], mean=res['mean'])
        if standard:
            sig = 0.21 * psf.wavelength_um / float(psf.NA) / psf.pixel_um
            out['sofi2'] = sofi2_xc(res['G'], res['mask'], res['D'], sig)
            out['sofi2_rw'] = sofi2_xc(res['G'], res['mask'], res['D'], sig, reweight=0.02)
        return out
