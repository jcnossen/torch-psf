"""analyze_movie: band limit and mode of a movie in one call."""
import dataclasses
import numpy as np

from torch_psf.movie import open_movie
from torch_psf.auto.spectrum import power_spectrum, fit_band_limit, frame_blocks, read_frames
from torch_psf.auto.classify import movie_statistics, classify_movie


@dataclasses.dataclass
class AutoResult:
    band: object            # BandLimit (spectrum frames)
    spectrum: object        # Spectrum
    mode: object            # ModeResult
    spectrum_frames: np.ndarray
    class_frames: np.ndarray

    def summary(self):
        return (f'{self.band.summary()}\n  {self.band.message}\nmode: {self.mode.mode} '
                f'(confidence {self.mode.confidence:.2f})\n  {self.mode.reason}\n  {self.mode.advice}')

    def route(self, **kw):
        """The PSF-estimation route for this movie (torch_psf.auto.route.plan_route; kw: pixel_um, wavelength_um,
        NA, dz_um, meta)."""
        from torch_psf.auto.route import plan_route
        return plan_route(self, **kw)

    def to_dict(self):
        return dict(band=self.band.to_dict(), mode=self.mode.to_dict(),
                    spectrum_frames=[int(self.spectrum_frames[0]), int(self.spectrum_frames[-1]),
                                     len(self.spectrum_frames)],
                    class_frames=[int(self.class_frames[0]), int(self.class_frames[-1]), len(self.class_frames)])


def choose_frames(n, nblocks=8, block_len=32, class_frames=400, class_start=0.4, skip_start=0.05, frame_px=None,
                  pixel_budget=64e6):
    """Frame indices for the spectrum (nblocks blocks of block_len frames spread over the movie after the first
    skip_start fraction, e.g. the dense start of a dSTORM movie; with frame_px = H x W given, block_len grows so that
    about pixel_budget pixels are used, up to 4096 frames: small frames need more of them) and for the classifier
    (one contiguous block of class_frames frames starting at class_start; the whole movie if it is shorter)."""
    if frame_px:
        block_len = int(np.clip(pixel_budget / frame_px / nblocks, block_len, 4096 // nblocks))
    si = frame_blocks(n, nblocks, block_len, skip_start if n > 4 * nblocks * block_len else 0.0)
    if n <= class_frames:
        ci = np.arange(n)
    else:
        c0 = min(int(class_start * n), n - class_frames)
        ci = np.arange(c0, c0 + class_frames)
    return si, ci


def analyze_frames(spec_frames, class_frames, pixel_um=None, wavelength_um=None, n_boot=100, device=None,
                   spectrum_kw=None, si=None, ci=None):
    """analyze_movie on frames already read: spec_frames [Ts, H, W] for the band limit, class_frames [Tc, H, W]
    (contiguous) for the mode (may be the same array)."""
    spec = power_spectrum(spec_frames, device=device, **(spectrum_kw or {}))
    band = fit_band_limit(spec, pixel_um=pixel_um, wavelength_um=wavelength_um, n_boot=n_boot)
    b2 = fit_band_limit(spec if class_frames is spec_frames else
                        power_spectrum(class_frames, device=device, **(spectrum_kw or {})), n_boot=0,
                        anisotropy=False)
    if band.status in ('ok', 'weak', 'no_floor'):
        b2 = dataclasses.replace(b2, fc_px=band.fc_px, status=band.status)
    st = movie_statistics(class_frames, band=b2, device=device)
    si = np.arange(len(spec_frames)) if si is None else si
    ci = np.arange(len(class_frames)) if ci is None else ci
    return AutoResult(band=band, spectrum=spec, mode=classify_movie(st), spectrum_frames=si, class_frames=ci)


def analyze_movie(movie, pixel_um=None, wavelength_um=None, nblocks=8, block_len=32, class_frames=400,
                  class_start=0.4, skip_start=0.05, n_boot=100, device=None):
    """Band limit (fit_band_limit) and mode (classify_movie) of a movie; frames chosen by choose_frames.
    movie: a path (.tif / .npy), an array [T, H, W] or a movie reader (torch_psf.movie interface).
    Returns an AutoResult (band, spectrum, mode; .summary(), .to_dict())."""
    m = open_movie(movie)
    si, ci = choose_frames(m.n, nblocks, block_len, class_frames, class_start, skip_start,
                           frame_px=m.shape[0] * m.shape[1])
    fr = read_frames(m, si)
    cf = fr if np.array_equal(ci, si) else read_frames(m, ci)
    return analyze_frames(fr, cf, pixel_um, wavelength_um, n_boot, device, si=si, ci=ci)
