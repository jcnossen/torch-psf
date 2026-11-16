"""Model variants of the dense fit on the real stacks (left channel): trained on beads.tif, scored on the held-out
beads2.tif clusters (DenseBeadEstimator holdout=1). Same assumptions as real_fit.py.

  python -I real_variants.py DATA_DIR OUT_JSON [--plane-step 2] [--iters 300] [--variants base fit_na ...] [--exclude 48 49 59]
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import tifffile
import torch
from torch_psf import Camera, DenseBeadEstimator

ap = argparse.ArgumentParser()
ap.add_argument('data')
ap.add_argument('out')
ap.add_argument('--plane-step', type=int, default=2)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--variants', nargs='+', default=['base', 'fit_na', 'bg_const', 'tilt', 'height50', 'no_pad'])
ap.add_argument('--exclude', type=int, nargs='*', default=[48, 49, 59],
                help='held-out clusters not scored (real_fit.py: chi2 above 3 x the training median; aggregates)')
args = ap.parse_args()
dev = 'cuda'
OPT = dict(pixel_um=0.097, NA=1.49, wavelength_um=0.67, n_imm=1.518, n_med=1.33)
CAM = Camera(gain_e_per_adu=0.597, offset_adu=100.6, readnoise_e=0.0)
x0, y0, x1, y1 = 0, 0, 475, 400
st = [CAM.to_photons(torch.as_tensor(tifffile.imread(os.path.join(args.data, f))[:, y0:y1, x0:x1].astype(np.float32),
                                     device=dev)) for f in ('beads.tif', 'beads2.tif')]
BASE = dict(optics=OPT, patch=21, iters=args.iters, holdout=1, reject=3.0, plane_step=args.plane_step, max_box=84,
            max_members=10, focus_offset='bead', bg='linear', device=dev)
VAR = dict(base={}, fit_na=dict(fit_na=True), bg_const=dict(bg='const'), tilt=dict(focus_offset='tilt'),
           height50=dict(emitter_z_um=0.05), no_pad=dict(render_pad=0))
out = {}
if os.path.exists(args.out):
    out = json.load(open(args.out))
for v in args.variants:
    t0 = time.time()
    est = DenseBeadEstimator(**{**BASE, **VAR[v]})
    psf = est.estimate(st, 0.010)
    r = est.result
    # the same held-out clusters for every variant (detection and grouping do not depend on the variant)
    ho = np.nonzero(r['heldout'])[0]
    ok = ho[~np.isin(ho, args.exclude)]
    npx = np.diff(est.data.cl_pix)[ok].astype(float)
    rep = est.report
    out[v] = dict(zcoef=psf.zcoef.detach().cpu().tolist(), NA=float(psf.NA), n_beads=rep['n_beads'],
                  deviance_per_px=rep['deviance_per_px'], heldout_deviance_all=rep.get('heldout_deviance_per_px'),
                  heldout_deviance_per_px=float((r['deviance'][ok] * npx).sum() / npx.sum()),
                  heldout_nll_per_px=float(r['nll'][ok].sum() / (npx.sum() * est.data.nk)), n_heldout=len(ok),
                  seconds=time.time() - t0)
    print(v, json.dumps({k: x for k, x in out[v].items() if k != 'zcoef'}), flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)
