"""Localization of a movie with a trained SMLMNet -> torch_psf Dataset.

    python -m torch_psf nn localize movie.tif --model MODEL_DIR [--gain 0.065 --offset 394.1 | --camera model] -o locs.hdf5

Every frame t is the centre of the window t - T // 2 .. t + T // 2 (frames beyond the movie are replaced by its
first / last frame). The frames are converted to photoelectrons, (ADU - offset) x gain (the excess-noise factor of an
EMCCD is not applied: the network is trained on photoelectrons), padded (replicate) to a multiple of the network's
size multiple, and the output is decoded by the offset-aware merge (torch_psf.nn.merge; defaults cand_th 0.05,
r0 0.6, p >= 0.2) and, if the model has one and calibrate=True, the sigma calibration (torch_psf.nn.calibrate).

Dataset fields (docs/DATASET_FORMAT.md): frame, pos [n, 3] px (x, y; z in px of the x pixel size,
meta['pixel_nm'], z = 0 at the PSF's z = 0), crlb [n, 3] px (the calibrated sigmas; the raw ones without a
calibration), photons (photoelectrons), bg (photoelectrons / px / frame), score (the merged detection probability p),
sigma [n, 3] px (the network's sigmas), s_photons (photoelectrons); full_xyz models also cov [n, 3] px^2 (the
calibrated xy, xz, yz covariances) and rho [n, 3] (the network's xy, xz, yz correlations). 2D models: pos, crlb,
sigma [n, 2]. meta: pixel_nm, pixel_nm_y, image_size, n_frames, method, model, calibrated, photon_unit.
"""
import dataclasses
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_psf.camera import Camera
from torch_psf.dataset import Dataset
from torch_psf.movie import open_movie
from torch_psf.nn import calibrate as C
from torch_psf.nn.files import load_model
from torch_psf.nn.merge import detect, P_MIN
from torch_psf.nn.model import correlations


def photoelectrons(camera, adu):
    """(adu - offset) x gain for frames [T, H, W] (camera None: the frames are already photoelectrons)."""
    x = torch.as_tensor(adu).float()
    if camera is None:
        return x
    return dataclasses.replace(camera, excess_noise=1.0).to_photons(x)


def windows(f0, f1, n, T):
    """Indices [f1 - f0, T] of the frames of the windows centred at f0 .. f1 - 1 (clamped to 0 .. n - 1)."""
    h = T // 2
    t = torch.arange(f0, f1)[:, None] + torch.arange(-h, h + 1)[None]
    return t.clamp(0, n - 1)


@torch.no_grad()
def run_network(net, frames):
    """frames [B, T, H, W] photoelectrons -> network output [B, C, H, W] (padded to the size multiple and cropped)."""
    m = net.size_multiple
    H, W = frames.shape[-2:]
    ph, pw = (-H) % m, (-W) % m
    x = F.pad(frames, (0, pw, 0, ph), mode='replicate') if ph or pw else frames
    return net(x)[..., :H, :W]


def rows_to_dataset(rows, frame, net, meta, calibrated):
    """Merged rows (+ the frame of each row) -> Dataset (module docstring)."""
    px = float(meta['pixel_nm'])
    zs = 1000.0 / px                                         # um -> px of the x pixel
    axes = 'xyz' if net.z else 'xy'
    sc = torch.tensor([1., 1., zs][:len(axes)], dtype=torch.float64)
    get = lambda k: rows[k].double().cpu()
    pos = torch.stack([get(k) for k in axes], 1) * sc
    crlb = torch.stack([get('s' + k) for k in axes], 1) * sc
    F_ = dict(frame=frame, pos=pos, crlb=crlb.float(), photons=get('photons').float(), bg=get('bg').float(),
              score=get('p').float(), s_photons=get('s_photons').float(),
              sigma=(torch.stack([get('raw_s' + k) for k in axes], 1) * sc).float())
    if 'cov' in rows:
        cov = rows['cov'].double().cpu() * (sc[:, None] * sc[None])
        F_['cov'] = cov[:, [0, 0, 1], [1, 2, 2]].float()
        F_['rho'] = correlations(rows['raw_cov'].double().cpu()).float()
    m = dict(pixel_nm=px, method='torch_psf.nn SMLMNet', calibrated=bool(calibrated), photon_unit='photoelectrons',
             z_unit='px of pixel_nm (z_um x 1000 / pixel_nm)')
    if meta.get('pixel_nm_y') is not None and float(meta['pixel_nm_y']) != px:
        m['pixel_nm_y'] = float(meta['pixel_nm_y'])
    return Dataset(F_, m)


@torch.no_grad()
def localize_movie(movie, model, camera=None, frames=None, batch=8, device=None, rule='merge', p_min=P_MIN,
             calibrate=True, progress=None, **rule_kw):
    """Localize a movie (module docstring). movie: path / array / reader (ADU, or photoelectrons with camera None);
    model: a model directory or a (net, meta, calibration) tuple of files.load_model; camera: Camera, a calibration
    json path, 'model' (the camera of the training scenes in model.json) or None; frames: (f0, f1); batch: windows
    per forward; rule / rule_kw: torch_psf.nn.merge.detect; progress(done, total, n_locs)."""
    net, meta, cal = load_model(model, device) if isinstance(model, (str, os.PathLike)) else model
    dev = next(net.parameters()).device
    if isinstance(camera, str):
        camera = Camera.from_dict(meta['camera']) if camera == 'model' else Camera.from_json(camera)
    mv = open_movie(movie)
    n = mv.n
    f0, f1 = (0, n) if frames is None else (int(frames[0]), int(frames[1]))
    T = net.window
    use_cal = bool(calibrate and cal)
    parts, fr = [], []
    for b0 in range(f0, f1, batch):
        b1 = min(b0 + batch, f1)
        idx = windows(b0, b1, n, T)
        lo, hi = int(idx.min()), int(idx.max()) + 1
        img = photoelectrons(camera, mv.read(lo, hi)).to(dev)
        x = img[(idx - lo).to(dev)]                          # [b, T, H, W]
        r = detect(run_network(net, x), rule, p_min=p_min, n_params=net.n_params, cov=net.cov, **rule_kw)
        for k in 'xyz':
            r['raw_s' + k] = r['s' + k]
        if 'cov' in r:
            r['raw_cov'] = r['cov']
        if use_cal:
            r = C.apply(cal, r)
        parts.append({k: v.cpu() for k, v in r.items()})
        fr.append((r['b'] + b0).cpu())
        if progress is not None:
            progress(b1 - f0, f1 - f0, sum(len(f) for f in fr))
    rows = {k: torch.cat([p[k] for p in parts]) for k in parts[0]} if parts else None
    if rows is None:
        raise ValueError('no frames to localize')
    ds = rows_to_dataset(rows, torch.cat(fr).int(), net, meta, use_cal)
    ds.meta.update(image_size=[int(v) for v in mv.shape[-2:]], n_frames=int(f1),
                   model=os.path.abspath(model) if isinstance(model, (str, os.PathLike)) else None)
    return ds
