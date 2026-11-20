"""Non-square pixels on real beads: the SMLM-challenge Tubulin-A647-3D bead stack (10 nm z steps), vectorial bead fit
with square pixels and with a fitted pixel aspect py / px (x = image column). Camera gain / offset from the localization
run's camera_noise.json; starting optics from an earlier in-situ bead fit (vecbeads_n36_NAfree.json). The bead height is
fitted too: with astigmatism a fixed, wrong focus offset is absorbed by the aspect.

python study/nonsquare/smlm_beads.py [--data /mnt/data/SMLM_challenge/Tubulin-A647-3D] [--out results.json]
Result on qnano1 (2026-11-20): square NLL -8106420, aspect 1.0724 (0.1065 x 0.1142 um), NLL -8107942.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch_psf import VectorialPSF
from torch_psf.estimation.beads import fit_beads, stage_focus
from torch_psf.estimators import find_beads

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data', type=Path, default=Path('/mnt/data/SMLM_challenge/Tubulin-A647-3D'))
ap.add_argument('--out', type=Path, default=None)
ap.add_argument('--roisize', type=int, default=23)
ap.add_argument('--nbeads', type=int, default=8)
ap.add_argument('--iters', type=int, default=2000)
A = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cam = json.loads((A.data / 'locs/camera_noise.json').read_text())['median']
prev = json.loads((A.data / 'insitu/vecbeads_n36_NAfree.json').read_text())
stack = tifffile.imread(A.data / 'Tubulin-A647-3D-BEADS-as-stacks.tif').astype(np.float32)   # [planes, row, col] ADU
ph = np.clip((stack - cam['offset_ADU']) / cam['g_ADU_per_e'], 0, None)
R, STEP, HALF = A.roisize, 4, 0.8                       # every 4th plane (40 nm), +-0.8 um around the focus
beads = find_beads(ph, R, nmax=A.nbeads)
ii = np.arange(R)
rois = np.stack([ph[:, y0 + ii[:, None], x0 + ii[None, :]] for x0, y0 in beads])
flat = rois.reshape(len(rois), len(ph), -1)
k0 = int(np.argmax(np.convolve((flat.max(2) / flat.sum(2)).mean(0), np.ones(9) / 9, 'same')))   # most compact plane
n = int(round(HALF / 0.01 / STEP))
ks = k0 + STEP * np.arange(-n, n + 1)
ks = ks[(ks >= 0) & (ks < len(ph))]
d = torch.as_tensor(rois[:, ks], device=dev)
focus = stage_focus(len(ks), 0.01 * STEP, float(prev['s'])).to(dev)
p0 = math.sqrt(prev['pxy'][0] * prev['pxy'][1])
print(f'{len(beads)} beads, focus plane {k0}, {len(ks)} planes, geometric-mean pixel {p0:.4f} um', flush=True)


def start():
    v = VectorialPSF(R, pixel_um=p0, NA=prev['NA'], n_imm=prev['n_imm'], n_med=prev['n_med'], wavelength_um=prev['lam'],
                     zernike_indices=tuple(prev['noll']), npupil=64, oversample=3, device=dev, z_range=(-HALF, HALF))
    with torch.no_grad():
        v.zcoef.copy_(torch.tensor(prev['zcoef'], device=dev))
        v.apod.copy_(torch.tensor(prev['apod'], device=dev))
    return v


out = dict(beads=[list(map(int, b)) for b in beads], focus_plane=k0, planes=ks.tolist())
for name, kw in (('square', {}), ('aspect', dict(fit_aspect=True))):
    r = fit_beads(d, focus, start(), iters=A.iters, fit=('zcoef', 'NA', 'apod'), drift=True, drift_order=1,
                  emitter_z_um=0.05, fit_height=True, **kw)
    rep = r['report']
    out[name] = dict(nll=rep['nll_per_px'] * d.numel(), NA=float(r['psf'].NA), height_um=rep['height_um'],
                     pixel_aspect=rep.get('pixel_aspect', 1.0), pixel_um=rep.get('pixel_um', p0))
    print(name, json.dumps(out[name]), flush=True)
if A.out:
    A.out.write_text(json.dumps(out, indent=1))
