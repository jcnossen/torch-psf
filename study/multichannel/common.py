"""Shared settings of the multi-channel bead study (2- and 3-colour stacks of the user's setup): data paths, the
assumptions of docs/notes/dense_beads.md (97 nm px, 670 nm, gain 0.597 e-/ADU, offset 100.6 ADU), the dense-bead
settings of the stage-2a runs, and the field orders. The data are read as plain arrays (tifffile); nothing in the data
folders is executed or unpickled."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import tifffile
import torch
from torch_psf import Camera

D2 = '/mnt/data/multicolor/june/astig/2col/beads100nm_10nm_steps'
D3 = '/mnt/data/multicolor/june/astig/3col/beads'
FILES2 = [os.path.join(D2, f) for f in ('beads.tif', 'beads2.tif')]
FILES3 = [os.path.join(D3, f) for f in ('beads1.tif', 'beads2.tif')]
OPT = dict(pixel_um=0.097, NA=1.49, wavelength_um=0.67, n_imm=1.518, n_med=1.33)
CAM = Camera(gain_e_per_adu=0.597, offset_adu=100.6, readnoise_e=0.0)
REGIONS = [(0, 0, 475, 400), (475, 0, 950, 400)]
NAMES = ['left', 'right']
DZ = 0.010
START = os.path.expanduser('~/phasediv_out/real/psf_dense_fit_na_apod.json')   # left channel, global NA + apod (stage 2a)
DENSE = dict(patch=21, plane_step=2, max_box=84, max_members=10, threshold=6.0, faint=0.1, focus_offset='bead',
             bg='linear', reject=3.0, holdout=1)
ORDERS = dict(
    mixed={5: 2, 6: 2, 11: 2, 7: 1, 8: 1, 9: 1, 10: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1},
    linear={j: 1 for j in range(5, 22)},
    astig={5: 2, 6: 2, 11: 2})
OUT = os.path.expanduser('~/mcfield_out')


def load(files, plane_step=1):
    """raw frames [nk, H, W] float32 (ADU) of each file"""
    return [tifffile.imread(f)[::plane_step].astype(np.float32) for f in files]


def pick_gpu():
    """the least loaded GPU (memory used), as a device string"""
    if not torch.cuda.is_available():
        return 'cpu'
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f'cuda:{int(np.argmax(free))}'


def build_data(files, regions, dev, plane_step=2, field_stop=True, patch=21, max_members=10, max_box=84,
               threshold=6.0, faint=0.1, merge_px=2.0, mapping_order=1, rel_threshold=0.0):
    """The MultiChannelBeads of MultiChannelBeadEstimator.estimate with the DENSE settings (the same detections,
    registration and clusters as the estimator runs; deterministic): (data, focus_um, maps_registered, registration
    info, channel stacks, masks)"""
    from torch_psf.estimation import stage_focus
    from torch_psf.estimation.multichannel_beads import (channel_stop_mask, detect_channels, register_channels,
                                                         build_multichannel_beads)
    raw = load(files)
    nk = raw[0].shape[0]
    cs = [[] for _ in regions]
    for a in raw:
        t = CAM.to_photons(torch.as_tensor(a, device=dev))[::plane_step]
        for c, (x0, y0, x1, y1) in enumerate(regions):
            cs[c].append(t[:, y0:y1, x0:x1].contiguous())
    focus = stage_focus(nk, DZ, 1.0).to(dev)[::plane_step]
    S = len(files)
    masks = [[channel_stop_mask(cs[c][s]) for s in range(S)] for c in range(len(regions))] if field_stop else None
    dets = detect_channels(cs, threshold, 3, faint, rel_threshold=rel_threshold)
    maps, reg = register_channels(dets, [tuple(cs[c][0].shape[-2:]) for c in range(len(regions))], order=mapping_order)
    data = build_multichannel_beads(cs, maps, patch, masks, dets=dets, merge_px=merge_px, max_members=max_members,
                                    max_box=max_box, render_pad=patch // 3, device=dev)
    return data, focus, maps, reg, cs, masks


def zip_model(path, dev):
    """(per-channel psfs (VectorialPSF or FieldPSF), maps, channel focus [C]) of a multi-channel psf.zip"""
    from torch_psf.psfzip import PSFZip
    mc = PSFZip.read(path, device=dev).multichannel(field=True, mapping=True)
    cf = [float(getattr(m, 'base', m).focus_um) for m in mc.models]
    return mc.models, mc.mappings, cf
