"""torch_psf.nn: a neural-network localizer for SMLM movies (docs/nn.md).

The detection network of SIMCODE v1 in its plain-SMLM form, a DECODE-style network (Speiser et al., Nature Methods
2021): a window of 3 frames -> per-pixel detection probability, x, y, z, background and photons with uncertainties
(optionally a full x, y, z covariance), trained with a Gaussian-mixture likelihood on live simulated scenes.

  SMLMNet                   the network (model.py); xyz_cholesky: covariance from its sigma / correlation outputs
  GMMLoss, targets_tensor   the training loss (loss.py)
  detect, nms, grid         network output -> emitters (merge.py: merge.merge, the offset-aware merge, is the default)
  Scenes, scene_config      the training-scene simulator and its labels (scenes.py)
  train_model, load_config, convert_simcode_config  training (train.py)
  bn_recalibrate, fit_calibration, apply_calibration     BatchNorm and sigma calibrations (calibrate.py)
  localize_movie            movie -> Dataset (localize.py)
  load_model, save_model, convert_simcode_model     the model directory (files.py)
"""
from torch_psf.nn.model import SMLMNet, xyz_cholesky, correlations
from torch_psf.nn.loss import GMMLoss, targets_tensor
from torch_psf.nn.merge import detect, nms, grid, MERGE_DEFAULTS, P_MIN
from torch_psf.nn.scenes import Scenes, scene_config, RANGES, SCENE_DEFAULTS
from torch_psf.nn.train import train_model, load_config, convert_simcode_config, Lion
from torch_psf.nn.calibrate import bn_recalibrate, recalibrate_bn, fit_calibration
from torch_psf.nn.calibrate import apply as apply_calibration
from torch_psf.nn.localize import localize_movie
from torch_psf.nn.files import load_model, save_model, convert_simcode_model

__all__ = ['SMLMNet', 'xyz_cholesky', 'correlations', 'GMMLoss', 'targets_tensor', 'detect', 'nms', 'grid',
           'MERGE_DEFAULTS', 'P_MIN', 'Scenes', 'scene_config', 'RANGES', 'SCENE_DEFAULTS', 'train_model', 'load_config',
           'convert_simcode_config', 'Lion', 'bn_recalibrate', 'recalibrate_bn', 'fit_calibration',
           'apply_calibration', 'localize_movie', 'load_model', 'save_model', 'convert_simcode_model']
