"""Held-out score (dense_nll, beads2 clusters) of saved PSFs with and without a lateral bead blur: a 100 nm bead with
a stained surface projects onto the image plane with variance r^2 / 3 per axis (r = 50 nm: 0.0886 px^2 at 97 nm).
Phase diversity puts the lateral bead size into the object (its PSF is that of a point emitter); the bead fitters
put it into their PSF.

  python -I score_blur.py OUT_JSON label=psf.json ...
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from torch_psf import PSF
st = load()
data, focus, ho = clusters(st)
hod = data.subset(ho)
del st
v = (0.05 ** 2 / 3) / OPT['pixel_um'] ** 2
res = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
for a in sys.argv[2:]:
    label, path = a.split('=', 1)
    p = PSF.load(os.path.expanduser(path), device='cuda')
    s0, _ = score(hod, focus, p)
    p.blur = torch.tensor([v, v, 0.0], device='cuda')
    s1, _ = score(hod, focus, p)
    res[label] = dict(path=path, plain=s0['deviance_per_px'], bead_blur=s1['deviance_per_px'],
                      plain_nll=s0['nll_per_px'], bead_blur_nll=s1['nll_per_px'])
    print(label, json.dumps(res[label]), flush=True)
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
