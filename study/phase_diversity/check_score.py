"""Reproduce the dense-beads held-out score (1.4409 for psf_dense.json) with study/phase_diversity/common.py."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from torch_psf import PSF
st = load()
data, focus, ho = clusters(st)
print('clusters', data.n_clusters, 'heldout', len(ho), flush=True)
hod = data.subset(ho)
p = PSF.load(os.path.expanduser('~/densebeads_out/real/psf_dense.json'), device='cuda')
s, _ = score(hod, focus, p)
print(json.dumps(s))
