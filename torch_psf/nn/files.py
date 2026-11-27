"""Model files of torch_psf.nn: a directory with

  weights.pt         the state_dict of the network (torch.save; loaded with weights_only=True)
  model.json         format 'torch-psf nn model' (version 1): network (class name), architecture (the network options),
                     scaling (input and output scaling), pixel_nm (x pixel, also the unit of z in the localization
                     tables) and pixel_nm_y (non-square pixels), psf (the PSF files of the training scenes with their
                     sha256), camera (the camera the training frames were simulated for, torch_psf Camera dict),
                     units, training (steps, seed, config), source (provenance), calibration (file name or null)
  calibration.json   optional sigma calibration (torch_psf.nn.calibrate; format 'torch-psf nn calibration')

A training run directory (torch_psf.nn.train) is such a directory too.
"""
import hashlib
import json
import os
import numpy as np
import torch
from torch_psf.nn.model import SMLMNet

FORMAT, VERSION = 'torch-psf nn model', 1
ARCH_KEYS = ('window', 'z', 'cov', 'frame_features', 'window_features', 'window_out', 'head_features', 'batch_norm')
SCALING_KEYS = ('input_scale', 'input_offset', 'max_photons', 'max_bg', 'xyz_scale')
UNITS = dict(x='px (image column, pixel centres at integers)', y='px (image row)', z='um (the z of the PSF)',
             bg='photoelectrons / px / frame', photons='photoelectrons per frame',
             input='photoelectrons / px / frame (camera ADU converted with the gain and offset, no excess-noise factor)')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def camera_dict(cam):
    """Scene camera (adu_per_e, offset_adu, readnoise_adu) -> torch_psf Camera dict (an EMCCD, excess noise 2)."""
    g = 1.0 / float(cam['adu_per_e'])
    return dict(gain_e_per_adu=g, offset_adu=float(cam['offset_adu']), readnoise_e=float(cam['readnoise_adu']) * g,
                type='emccd', excess_noise=2.0)


def model_meta(net, pixel_nm, pixel_nm_y=None, psf=(), camera=None, training=None, source=None, calibration=None):
    """The model.json dict of a network."""
    o = net.options()
    psf_files = [str(p) for p in psf]
    return dict(format=FORMAT, version=VERSION, network=type(net).__name__,
                architecture={k: o[k] for k in ARCH_KEYS}, scaling={k: o[k] for k in SCALING_KEYS},
                pixel_nm=float(pixel_nm), pixel_nm_y=None if pixel_nm_y is None else float(pixel_nm_y),
                psf=dict(files=psf_files, sha256=[sha256(p) if os.path.isfile(p) else None for p in psf_files]),
                camera=camera, units=UNITS, weights='weights.pt', calibration=calibration,
                training=training or {}, source=source or {})


def save_model(folder, net, meta, calibration=None):
    """Write weights.pt, model.json (meta: model_meta) and, if given, calibration.json (a dict)."""
    os.makedirs(folder, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(folder, 'weights.pt'))
    meta = dict(meta, weights='weights.pt')
    if calibration is not None:
        with open(os.path.join(folder, 'calibration.json'), 'w') as f:
            json.dump(calibration, f, indent=1)
        meta['calibration'] = 'calibration.json'
    with open(os.path.join(folder, 'model.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return folder


def read_meta(folder):
    with open(os.path.join(folder, 'model.json')) as f:
        meta = json.load(f)
    if meta.get('format') != FORMAT:
        raise ValueError(f'{folder}: not a {FORMAT!r} directory')
    if int(meta.get('version', 0)) > VERSION:
        raise ValueError(f'{folder}: model.json version {meta["version"]} is newer than this reader ({VERSION})')
    return meta


def build(meta):
    """The (untrained) network of a model.json dict."""
    if meta.get('network', 'SMLMNet') != 'SMLMNet':
        raise ValueError(f"unknown network {meta.get('network')!r}")
    return SMLMNet(**meta['architecture'], **meta['scaling'])


def load_model(folder, device=None, weights=None):
    """-> (network in eval mode on device, model.json dict, calibration dict or None). weights: another state_dict
    file of the same architecture (e.g. a training checkpoint weights.step10000.pt)."""
    from torch_psf.optics import resolve_device
    dev = resolve_device(device)
    meta = read_meta(folder)
    net = build(meta).to(dev)
    sd = torch.load(weights or os.path.join(folder, meta.get('weights', 'weights.pt')), map_location=dev,
                    weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    cal = None
    if meta.get('calibration'):
        with open(os.path.join(folder, meta['calibration'])) as f:
            cal = json.load(f)
    return net, meta, cal


# ---------------------------------------------------------------------------------------------- SIMCODE models
_SIMCODE_PREFIX = (('unet_shared.', 'frame_unet.'), ('unet_combiner.', 'window_unet.'),
                   ('output_heads_layer_main.', 'head.'), ('cov_head.', 'corr_head.'))
_SIMCODE_PART = (('.inc.', '.stem.'), ('.outc.', '.out.'))


def simcode_state_dict(sd):
    """A SIMCODE V1Net state_dict -> the SMLMNet state_dict (renamed keys, same tensors)."""
    import re
    out = {}
    for k, v in sd.items():
        n = k
        for a, b in _SIMCODE_PREFIX:
            if n.startswith(a):
                n = b + n[len(a):]
                break
        else:
            raise ValueError(f'unexpected V1Net key {k!r}')
        for a, b in _SIMCODE_PART:
            n = n.replace(a, b)
        n = re.sub(r'\.encoders\.(\d+)\.conv\.', r'.down.\1.convs.', n)
        n = re.sub(r'\.decoders\.(\d+)\.conv\.', r'.up.\1.convs.', n)
        n = re.sub(r'\.encoders\.(\d+)\.pool\.', r'.down.\1.pool.', n)
        out[n] = v
    return out


def simcode_calibration(S):
    """A SIMCODE sigma_scale json (dict) -> the calibration.json dict (torch_psf.nn.calibrate)."""
    from torch_psf.nn.calibrate import FORMAT as CFORMAT, VERSION as CVERSION
    if S.get('bin', 'N') != 'N':
        raise ValueError(f"sigma_scale bin {S.get('bin')!r}: only 'N' (the network's photons)")
    if any(np.ndim(S.get(k, 0)) == 2 for k in ('sx', 'sy', 'sz')) or 'crowd' in S:
        raise ValueError('crowding-binned sigma_scale factors are not part of torch_psf.nn')
    if 'sNi' in S:
        raise ValueError('sNi (per-frame intensities) belongs to the SIM network')
    C = dict(format=CFORMAT, version=CVERSION, bin='photons', edges=[float(e) for e in S['edges']])
    if 'xyz_matrix' in S:
        C['xyz_matrix'] = S['xyz_matrix']
        C['method'] = (S.get('check') or {}).get('method', (S.get('fit') or {}).get('matrix_fit', 'rms'))
    else:
        C['factors'] = {k[1]: S[k] for k in ('sx', 'sy', 'sz') if k in S}
        C['method'] = 'axis_q68'
    for k in ('check', 'fit', 'seeds', 'counts_fit', 'counts_check'):
        if k in S:
            C.setdefault('source', {})[k] = S[k]
    return C


def convert_simcode_model(weights, out, meta=None, sigma_scale=None, config=None, psf=None):
    """SIMCODE plain-SMLM V1Net (v1net.pt + v1net.json; optional sigma_scale json; the run's torch_psf.nn config, e.g.
    from train.convert_simcode_config, for pixel size, camera and PSF) -> a torch_psf.nn model directory `out`.
    psf: the PSF file(s) to reference (default: those of the config)."""
    from torch_psf.nn.train import simcode_model_options, load_config
    meta = meta or os.path.join(os.path.dirname(os.path.abspath(weights)), 'v1net.json')
    with open(meta) as f:
        M = json.load(f)
    if M.get('stage', 'base') != 'base' or M.get('scene', 'plain') != 'plain':
        raise ValueError(f"{meta}: stage {M.get('stage')!r} / scene {M.get('scene')!r}: only base plain-SMLM V1Nets")
    net = SMLMNet(**simcode_model_options(M['model']))
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    net.load_state_dict(simcode_state_dict(sd), strict=True)
    cfg = load_config(config) if config is not None else None
    sc = cfg['scenes'] if cfg else {}
    psf_files = [psf] if isinstance(psf, (str, os.PathLike)) else list(psf or sc.get('psf', []))
    cal = None
    if sigma_scale is not None:
        with open(sigma_scale) as f:
            cal = simcode_calibration(json.load(f))
    m = model_meta(net, sc.get('pixel_nm', 117.6), sc.get('pixel_nm_y'), psf_files,
                   camera_dict(sc['camera']) if sc else None,
                   training=dict(steps=M.get('steps'), seed=cfg['seed'] if cfg else None),
                   source=dict(tool='torch_psf.nn.files.convert_simcode_model', weights=os.path.abspath(weights),
                               weights_sha256=sha256(weights), meta=os.path.abspath(meta), simcode_meta=M,
                               sigma_scale=None if sigma_scale is None else os.path.abspath(sigma_scale)))
    if cfg is not None:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=1)
        m['training']['config'] = 'config.json'
    return save_model(out, net, m, cal)
