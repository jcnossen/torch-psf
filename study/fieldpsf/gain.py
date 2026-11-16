"""Camera gain of the FD-DeepLoc example movie by photon transfer (torch_psf.photon_transfer).

    python -I gain.py MOVIE OUT_JSON [--offset 100]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # the worktree (python -I ignores PYTHONPATH)
import torch_psf

ap = argparse.ArgumentParser()
ap.add_argument('movie'); ap.add_argument('out')
ap.add_argument('--offset', type=float, default=None)
ap.add_argument('--frames', type=int, nargs=2, default=None)
a = ap.parse_args()
cams = {}
for off in ([None] if a.offset is None else [None, a.offset]):
    cam, rep = torch_psf.photon_transfer(a.movie, frames=a.frames, offset_adu=off, nblocks=6, block=100)
    cams['free_offset' if off is None else 'offset_%g' % off] = dict(camera=cam.to_dict(), report={k: v for k, v in rep.items() if k != 'per_block'}, per_block=rep.get('per_block'))
    print(off, cam, {k: v for k, v in rep.items() if k != 'per_block'})
json.dump(cams, open(a.out, 'w'), indent=1, default=float)
