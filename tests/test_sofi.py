"""SOFIEstimator on a small simulated dense blinking movie (GPU; a few minutes on CPU)."""
import torch
import torch_psf as tp
from torch_psf.sofi import BlinkModel, CameraModel, DenseMovieSpec, DenseMovie


def test_sofi_estimator_recovers_coma():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    psf = tp.VectorialPSF(13, pixel_um=0.108, NA=1.49, wavelength_um=0.68, device=dev)
    psf.set_zernike({7: 0.04, 8: -0.03, 6: 0.02})
    spec = DenseMovieSpec(H=40, W=40, T=4000, density_um2=50.0, photons=300.0, bg=60.0,
                          blink=BlinkModel(duty=0.3, t_on=2.0), camera=CameraModel(offset_adu=100.0, readnoise_e=1.5),
                          Q=13, grid_os=2, chunk=1000, seed=1)
    mv = DenseMovie(psf, spec, device=dev)
    movie = torch.cat([a for _, a in mv.frames()]).cpu().numpy()
    est = tp.SOFIEstimator(optics=tp.Optics(pixel_um=0.108, NA=1.49, wavelength_um=0.68),
                           camera=tp.Camera(gain_e_per_adu=1.0, offset_adu=100.0, readnoise_e=1.5), iters=20,
                           nboot=50, device=dev)
    out = est.estimate(movie)
    z = est.report['zernike']
    assert abs(z[7] - 0.04) < 0.004 and abs(z[8] + 0.03) < 0.004, z      # odd modes: first-order signal in focus
    assert abs(z[6] - 0.02) < 0.01, z                                    # even mode in focus: weak (no bias)
    assert isinstance(out, tp.VectorialPSF) and est.report['n_blocks'] == 8
