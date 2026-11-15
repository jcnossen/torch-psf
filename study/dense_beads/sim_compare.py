"""Dense vs isolated bead PSF estimation on simulated z-stacks with known Zernikes.

A field of beads in clusters (pairs, triplets, chains; 1-3 um between neighbours) plus a few isolated beads, a
coverslip tilt and a small lateral drift. Fits:
  iso      ZStackEstimator (existing): the brightest 'isolated' beads by its find_beads rule (ROIs may still hold
           parts of neighbours)
  single   the dense fitter on the strictly isolated beads only (clusters of one)
  dense    DenseBeadEstimator on all clusters (a held-out quarter of the clusters is scored)
and the held-out NLL / deviance per pixel of every PSF (and of the truth) on the held-out clusters (each cluster's own
terms refitted with the PSF fixed).

usage: python sim_compare.py OUT_JSON [--seeds 0 1 2] [--tilt 0.1] [--layout mixed|dense|packed] [--photons 30000]
"""
import argparse, json, time
import numpy as np
import torch
from torch_psf import VectorialPSF, ZStackEstimator, DenseBeadEstimator
from torch_psf.estimation import stage_focus
from torch_psf.estimation.dense_beads import (dense_bead_layout, simulate_dense_bead_stack, fit_dense_beads,
                                              dense_nll)

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
ap.add_argument('--tilt', type=float, default=0.1, help='focus change (um) across the field (x)')
ap.add_argument('--nk', type=int, default=51)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--size', type=int, default=320)
ap.add_argument('--layout', default='mixed', choices=['mixed', 'dense', 'packed'])
ap.add_argument('--photons', type=float, default=30000.0)
args = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
OPT = dict(pixel_um=0.1, NA=1.49, wavelength_um=0.67)
TRUE = {5: 0.06, 6: 0.02, 7: -0.03, 8: 0.025, 9: 0.015, 10: -0.01, 11: 0.04, 12: 0.01, 15: -0.01}
H = W = args.size
dz = 1.5 / (args.nk - 1)
focus = stage_focus(args.nk, dz, 1.0).to(dev)
truth = VectorialPSF(41, npupil=64, oversample=3, device=dev, **OPT).set_zernike(TRUE)
zt = truth.zcoef.detach().cpu().numpy()
# mixed / dense: neighbours 1-3 um apart (those beyond the patch become separate clusters); packed: 0.8-1.7 um (all
# within the patch), only two isolated beads
sizes = {'mixed': [1] * 5 + [2] * 8 + [3] * 5 + [4] * 3, 'dense': [1] * 2 + [2] * 5 + [3] * 5 + [4] * 4 + [5] * 2,
         'packed': [1] * 2 + [2] * 6 + [3] * 6 + [4] * 4}[args.layout]
SEP = (8, 17) if args.layout == 'packed' else (10, 30)
results = []


def zerr(psf):
    z = psf.zcoef.detach().cpu().numpy()
    return dict(rms_mwaves=float(np.sqrt(((z - zt) ** 2).mean()) * 1e3), max_mwaves=float(np.abs(z - zt).max() * 1e3),
                zcoef=z.tolist())


for seed in args.seeds:
    t0 = time.time()
    xy, cid = dense_bead_layout(H, W, sizes, sep_px=SEP, margin=16, gap_px=34, seed=seed)
    rng = np.random.default_rng(100 + seed)
    ph = np.exp(np.log(args.photons) + 0.3 * rng.standard_normal(len(xy)))
    fo = args.tilt * (xy[:, 0] - W / 2) / W                       # a tilted coverslip
    tt = np.linspace(-1, 1, args.nk)
    drift = np.stack([0.4 * tt, -0.3 * tt ** 2], 1)
    sim = simulate_dense_bead_stack(truth, focus, xy, H, W, photons=ph, bg=40.0, focus_offset=fo, drift=drift,
                                    seed=seed)
    stack = sim['d']
    rec = dict(seed=seed, n_beads_true=len(xy), cluster_sizes_true=sizes)
    # (b) dense, a quarter of the clusters held out
    de = DenseBeadEstimator(optics=OPT, iters=args.iters, holdout_fraction=0.25, holdout_seed=seed, field_stop=False,
                            device=dev)
    pd = de.estimate([stack], dz)
    rec['dense'] = dict(report={k: v for k, v in de.report.items() if not isinstance(v, (list,)) or k == 'rejected_clusters'},
                        **zerr(pd))
    data, ho = de.data, de.result['heldout']
    P = data.patch
    ho_data = data.subset(np.nonzero(ho)[0])
    tr_ids = np.nonzero(de.result['train'])[0]
    # (a) existing ZStackEstimator (all beads its find_beads rule accepts)
    ie = ZStackEstimator('vectorial', optics=OPT, roisize=P, nbeads=64, iters=args.iters, device=dev)
    pi = ie.estimate([stack.cpu().numpy()], dz)
    rec['iso'] = dict(n_beads=ie.report['n_beads'], nll_per_px=ie.report['nll_per_px'], **zerr(pi))
    # (a2) strictly isolated beads (training clusters of one bead) with the dense fitter
    single = [c for c in tr_ids if data.cl_bead[c + 1] - data.cl_bead[c] == 1]
    rs = fit_dense_beads(data.subset(single), focus, VectorialPSF(P, npupil=64, oversample=3, device=dev, **OPT),
                         iters=args.iters, **{k: v for k, v in de.fit_kw.items()})
    rec['single'] = dict(n_beads=len(single), **zerr(rs['psf']))
    # held-out scoring of every PSF on the held-out clusters
    sc = {}
    for name, p in [('truth', truth), ('dense', pd), ('iso', pi), ('single', rs['psf']),
                    ('start', VectorialPSF(P, npupil=64, oversample=3, device=dev, **OPT))]:
        r = dense_nll(ho_data, focus, p, iters=200, **de.fit_kw)
        sc[name] = dict(nll_per_px=r['report']['nll_per_px'], deviance_per_px=r['report']['deviance_per_px'])
    rec['heldout'] = sc
    rec['seconds'] = time.time() - t0
    results.append(rec)
    print(json.dumps(dict(seed=seed, dense=rec['dense']['rms_mwaves'], iso=rec['iso']['rms_mwaves'],
                          iso_n=rec['iso']['n_beads'], single=rec['single']['rms_mwaves'], single_n=len(single),
                          dense_n=de.report['n_beads'], heldout={k: v['deviance_per_px'] for k, v in sc.items()},
                          seconds=rec['seconds'])), flush=True)
json.dump(dict(args=vars(args), truth=TRUE, results=results), open(args.out, 'w'), indent=1)
