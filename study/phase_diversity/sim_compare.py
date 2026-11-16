"""Phase diversity vs DenseBeadEstimator on simulated dense bead stacks, each with and without NA / apod fitting.

Truth: 9 Zernikes up to 60 mλ (as study/dense_beads/sim_compare.py), NA `--na` (default 1.40; the fits start at the
nominal 1.49) and a radial pupil amplitude apod `--apod` (default exp(-0.5 rho^2)), 320 x 320 px of 0.1 um, 51
planes over 1.5 um, beads in clusters ('dense': neighbours 1-3 um apart; 'packed': 0.8-1.7 um, two isolated beads),
log-normal photons, background 40, a coverslip tilt (0.1 um across x) and a lateral drift. A second stack (another
layout and noise) is held out: every PSF (and the truth) is scored on its clusters by dense_nll (each cluster's own
terms refitted with the PSF fixed).

Fits: pd / pd_na_apod (PhaseDiversityEstimator, 64 px patches, shared pupil), dense / dense_na_apod
(DenseBeadEstimator, all clusters of the training stack). Errors: Zernike rms (mλ), NA, pupil transmission (rms
difference of the radial amplitude over the physical pupil coordinate 0 .. 1.49, the NA cut-off included).

  python -I sim_compare.py OUT_JSON [--layout dense|packed] [--seeds 0 1 2] [--na 1.40] [--apod -0.5 0 0]
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from torch_psf import VectorialPSF, DenseBeadEstimator, PhaseDiversityEstimator
from torch_psf.estimation import stage_focus
from torch_psf.estimation.dense_beads import dense_bead_layout, simulate_dense_bead_stack, dense_nll
from torch_psf.estimation.phase_diversity import radial_amplitude

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
ap.add_argument('--layout', default='dense', choices=['dense', 'packed'])
ap.add_argument('--na', type=float, default=1.40)
ap.add_argument('--apod', type=float, nargs=3, default=[-0.5, 0.0, 0.0])
ap.add_argument('--photons', type=float, default=30000.0)
ap.add_argument('--tilt', type=float, default=0.1)
ap.add_argument('--nk', type=int, default=51)
ap.add_argument('--size', type=int, default=320)
ap.add_argument('--iters', type=int, default=300, help='dense iterations')
ap.add_argument('--pd-ls', type=int, default=200)
ap.add_argument('--pd-iters', type=int, default=400)
ap.add_argument('--methods', nargs='+', default=['pd', 'pd_na_apod', 'dense', 'dense_na_apod'])
args = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
OPT = dict(pixel_um=0.1, NA=1.49, wavelength_um=0.67)
TRUE = {5: 0.06, 6: 0.02, 7: -0.03, 8: 0.025, 9: 0.015, 10: -0.01, 11: 0.04, 12: 0.01, 15: -0.01}
H = W = args.size
dz = 1.5 / (args.nk - 1)
focus = stage_focus(args.nk, dz, 1.0).to(dev)
truth = VectorialPSF(41, npupil=64, oversample=3, device=dev, **OPT).set_zernike(TRUE)
with torch.no_grad():
    truth.NA.fill_(args.na)
    truth.apod.copy_(torch.tensor(args.apod))
zt = truth.zcoef.detach().cpu().numpy()
sizes = {'dense': [1] * 2 + [2] * 5 + [3] * 5 + [4] * 4 + [5] * 2,
         'packed': [1] * 2 + [2] * 6 + [3] * 6 + [4] * 4}[args.layout]
SEP = (8, 17) if args.layout == 'packed' else (10, 30)
S_GRID = np.linspace(0, 1.49, 300)


def errors(psf):
    z = psf.zcoef.detach().cpu().numpy()
    a = radial_amplitude(psf, S_GRID)[1] - radial_amplitude(truth, S_GRID)[1]
    return dict(rms_mwaves=float(np.sqrt(((z - zt) ** 2).mean()) * 1e3), zcoef=z.tolist(), NA=float(psf.NA),
                NA_err=float(psf.NA) - args.na, apod=psf.apod.detach().cpu().tolist(),
                pupil_amp_rms=float(np.sqrt((a ** 2).mean())))


def stack(seed):
    xy, cid = dense_bead_layout(H, W, sizes, sep_px=SEP, margin=16, gap_px=34, seed=seed)
    rng = np.random.default_rng(100 + seed)
    ph = np.exp(np.log(args.photons) + 0.3 * rng.standard_normal(len(xy)))
    fo = args.tilt * (xy[:, 0] - W / 2) / W
    tt = np.linspace(-1, 1, args.nk)
    drift = np.stack([0.4 * tt, -0.3 * tt ** 2], 1)
    return simulate_dense_bead_stack(truth, focus, xy, H, W, photons=ph, bg=40.0, focus_offset=fo, drift=drift,
                                     seed=seed)['d']


results = []
for seed in args.seeds:
    st_train, st_test = stack(seed), stack(1000 + seed)
    rec = dict(seed=seed)
    # held-out clusters of the test stack (detection / grouping by the dense estimator, no fit)
    probe = DenseBeadEstimator(optics=OPT, iters=0, field_stop=False, device=dev, holdout_fraction=0.0)
    probe.estimate([st_test], dz)
    ho_data, fit_kw = probe.data, probe.fit_kw
    psfs = {}
    for m in args.methods:
        t0 = time.time()
        if m.startswith('pd'):
            est = PhaseDiversityEstimator(optics=OPT, patch=64, field_stop=False, min_signal=8.0, ls_iters=args.pd_ls,
                                          iters=args.pd_iters, fit_na=m.endswith('na_apod'),
                                          fit_apod=m.endswith('na_apod'), device=dev)
        else:
            est = DenseBeadEstimator(optics=OPT, iters=args.iters, field_stop=False, device=dev,
                                     fit_na=m.endswith('na_apod'), fit_apod=m.endswith('na_apod'))
        p = est.estimate([st_train], dz)
        sec = time.time() - t0
        psfs[m] = p
        rec[m] = dict(seconds=sec, **errors(p), n=est.report.get('n_beads', est.report.get('n_patches')))
        print(seed, m, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec[m].items()
                                   if k not in ('zcoef',)}), flush=True)
    sc = {}
    for name, p in [('truth', truth)] + list(psfs.items()):
        r = dense_nll(ho_data, focus, p, iters=200, **fit_kw)
        sc[name] = r['report']['deviance_per_px']
    rec['heldout_deviance'] = sc
    print(seed, 'heldout', json.dumps({k: round(v - sc['truth'], 5) for k, v in sc.items()}), flush=True)
    results.append(rec)
    json.dump(dict(args=vars(args), truth=TRUE, results=results), open(args.out, 'w'), indent=1)
