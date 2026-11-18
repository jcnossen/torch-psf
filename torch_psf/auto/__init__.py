"""Automatic analysis of a microscopy movie before any PSF fit ("drop in a movie, get the PSF", first steps).

  spectrum   the optical band limit f_c = 2 NA / lambda from the radially averaged power spectrum (NA / lambda with
             the pixel size, NA with the wavelength), with uncertainty and a statement of what can be estimated
  classify   which PSF-estimation route applies: smlm | sofi | zstack | static | moving | scene | no_signal, with a
             confidence, the key statistics and a plain-language reason
  motion     global frame-to-frame motion and natural-scene statistics (inputs of the classifier)
  route      mode -> estimator, the inputs it still needs, the steps (plan_route, make_estimator)
  psfgrab    reader for PSF grab recordings (phone camera zips)
  synth      cheap synthetic movies with an exactly known band limit (tests, calibration)
  analyze    analyze_movie(movie, ...) runs both on a movie reader / array / path

Layer: after optics, movie and detect (uses torch_psf.detect.blur); nothing in torch_psf imports it.
"""
from torch_psf.auto.spectrum import (Spectrum, BandLimit, power_spectrum, fit_band_limit, band_limit, otf_circ,
                                     frame_blocks, read_frames)
from torch_psf.auto.classify import movie_statistics, classify_movie, ModeResult
from torch_psf.auto.analyze import analyze_movie
from torch_psf.auto.synth import synthetic_movie, pupil_otf, scene_movie, moving_movie
from torch_psf.auto.motion import global_motion, motion_summary, scene_statistics
from torch_psf.auto.route import Route, plan_route, make_estimator
from torch_psf.auto.psfgrab import read_psfgrab

__all__ = ['Spectrum', 'BandLimit', 'power_spectrum', 'fit_band_limit', 'band_limit', 'otf_circ', 'frame_blocks',
           'read_frames', 'movie_statistics', 'classify_movie', 'ModeResult', 'analyze_movie', 'synthetic_movie',
           'pupil_otf', 'scene_movie', 'moving_movie', 'global_motion', 'motion_summary', 'scene_statistics', 'Route',
           'plan_route', 'make_estimator', 'read_psfgrab']
