"""Dense vs isolated bead PSF estimation on the 2-colour astigmatic bead stacks (left channel).

  python -I real_fit.py DATA_DIR OUT_DIR [--detect-only] [--iters 300] [--plane-step 1] [--patch P]

DATA_DIR holds beads.tif and beads2.tif (150 planes, 10 nm steps, 400 x 950 px, two channels side by side). The
data are read as plain arrays (tifffile); nothing in DATA_DIR is executed. Run with python -I (the repo path is added
below from this file's location).

Fits (trained on beads.tif, scored on the clusters of beads2.tif):
  (a)  ZStackEstimator: isolated beads by its find_beads rule (nbeads=8, the default, and nbeads=40)
  (a2) the dense fitter on the strictly isolated beads (clusters of one)
  (b)  DenseBeadEstimator: all clusters
Then every PSF is scored on the held-out beads2 clusters (own terms refitted, PSF fixed), and the field dependence is
probed by fitting the clusters of field regions (both stacks) separately.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import tifffile
import torch
from torch_psf import Camera, VectorialPSF, ZStackEstimator, DenseBeadEstimator
from torch_psf.estimation import stage_focus
from torch_psf.estimation.dense_beads import (field_stop_mask, find_all_beads, find_bead_clusters, psf_extent,
                                              fit_dense_beads, dense_nll)

ap = argparse.ArgumentParser()
ap.add_argument('data')
ap.add_argument('out')
ap.add_argument('--detect-only', action='store_true')
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--plane-step', type=int, default=1)
ap.add_argument('--patch', type=int, default=None)
ap.add_argument('--threshold', type=float, default=6.0)
ap.add_argument('--focus-offset', default='bead')
ap.add_argument('--reject', type=float, default=3.0)
ap.add_argument('--regions', type=int, default=3, help='n x n field regions for the regional fits')
ap.add_argument('--skip-regions', action='store_true')
ap.add_argument('--faint', type=float, default=0.1)
ap.add_argument('--max-members', type=int, default=8)
ap.add_argument('--max-box', type=int, default=None)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- assumptions (see docs/notes/dense_beads.md): pixel size from the DECODE configs of the same setup (97 nm);
# NA 1.49 oil (n 1.518); emission ~670 nm (far-red 2-colour set; not in the metadata); beads in water (n_med 1.33);
# camera from the photon transfer of the stack itself (adjacent-plane differences): offset 100.6 ADU, 0.597 e-/ADU.
OPT = dict(pixel_um=0.097, NA=1.49, wavelength_um=0.67, n_imm=1.518, n_med=1.33)
CAM = Camera(gain_e_per_adu=0.597, offset_adu=100.6, readnoise_e=0.0)
REGION = (0, 0, 475, 400)           # left channel (x < 475)
DZ = 0.010
files = [os.path.join(args.data, f) for f in ('beads.tif', 'beads2.tif')]
raw = [tifffile.imread(f) for f in files]
print('stacks', [r.shape for r in raw], flush=True)
nk = raw[0].shape[0]
focus_all = stage_focus(nk, DZ, 1.0).to(dev)


def to_photons(a):
    x0, y0, x1, y1 = REGION
    return CAM.to_photons(torch.as_tensor(a[:, y0:y1, x0:x1].astype(np.float32), device=dev))


start = VectorialPSF(21, npupil=64, oversample=3, device=dev, **OPT)
P = args.patch or psf_extent(start, focus_all, 0.0)
print('patch', P, flush=True)

if args.detect_only:
    det = []
    for i, a in enumerate(raw):
        ph = to_photons(a).cpu().numpy()
        mask = field_stop_mask(ph)
        xy, amp = find_all_beads(ph, threshold=args.threshold)
        kept, dropped = find_bead_clusters(xy, amp, P, ph.shape[1:], mask=mask, faint=args.faint,
                                           max_members=args.max_members, max_box=args.max_box)
        sizes = np.array([len(c['xy']) for c in kept])
        why = {}
        for c in dropped:
            why[c['why']] = why.get(c['why'], 0) + len(c['xy'])
        nf = sum(int(np.sum(c['faint'])) for c in kept)
        print(f'stack {i}: detected {len(xy)}, clusters kept {len(kept)} with {sizes.sum()} beads ({nf} faint) '
              f'(sizes {np.bincount(sizes).tolist()}), dropped beads by reason {why}', flush=True)
        det.append(dict(mp=ph.max(0), mask=mask, xy=xy, amp=amp, kept=[c['box'] for c in kept],
                        dropped=[(c['box'], c['why']) for c in dropped]))
    np.save(os.path.join(args.out, 'detections.npy'), np.array(det, dtype=object), allow_pickle=True)
    sys.exit()

st = [to_photons(a) for a in raw]
del raw
res = dict(optics=OPT, camera=dict(gain_e_per_adu=CAM.gain_e_per_adu, offset_adu=CAM.offset_adu), patch=P,
           args=vars(args))
fit_kw = dict(focus_offset=args.focus_offset, bg='linear')

# ---- (b) dense, trained on beads.tif, beads2.tif held out
t0 = time.time()
de = DenseBeadEstimator(optics=OPT, patch=P, iters=args.iters, holdout=1, reject=args.reject, plane_step=args.plane_step,
                        threshold=args.threshold, faint=args.faint, max_members=args.max_members, max_box=args.max_box,
                        device=dev, **fit_kw)
pd = de.estimate(st, DZ)
print('dense', json.dumps({k: v for k, v in de.report.items()}), f'{time.time() - t0:.0f}s', flush=True)
data, r = de.data, de.result
focus = de.focus_um
pd.save(os.path.join(args.out, 'psf_dense.json'))
res['dense'] = dict(report=de.report, zcoef=pd.zcoef.detach().cpu().tolist(), zidx=pd.zidx)
fkw = {k: v for k, v in de.fit_kw.items()}
tr_ids = np.nonzero(r['train'])[0]
# held-out clusters scored: those the dense fit explains about as well as the training ones (drops aggregates / debris
# by the same rule as the training rejection; the same subset is used for every PSF)
lim = args.reject * np.median(r['chi2'][tr_ids])
ho_all = np.nonzero(r['heldout'])[0]
ho_ids = ho_all[r['chi2'][ho_all] <= lim]
res['heldout_clusters'] = dict(all=len(ho_all), used=len(ho_ids), chi2_limit=float(lim),
                               dropped=[int(c) for c in ho_all if r['chi2'][c] > lim])
print('heldout clusters', res['heldout_clusters'], flush=True)
ho = data.subset(ho_ids)

# ---- (a) ZStackEstimator on beads.tif (its own held-out: the beads it picks in beads2.tif)
for nb in (8, 40):
    t0 = time.time()
    ie = ZStackEstimator('vectorial', optics=OPT, roisize=P, nbeads=nb, iters=args.iters, holdout=1, device=dev)
    pi = ie.estimate([s[::args.plane_step].cpu().numpy() for s in st], DZ * args.plane_step)
    print(f'iso{nb}', json.dumps(ie.report), f'{time.time() - t0:.0f}s', flush=True)
    pi.save(os.path.join(args.out, f'psf_iso{nb}.json'))
    res[f'iso{nb}'] = dict(report=ie.report, zcoef=pi.zcoef.detach().cpu().tolist())
    if nb == 8:
        p_iso8 = pi
    else:
        p_iso40 = pi
# ---- (a2) strictly isolated beads of beads.tif (training clusters of one) with the dense fitter
single = [c for c in tr_ids if data.cl_bead[c + 1] - data.cl_bead[c] == 1]
rs = fit_dense_beads(data.subset(single), focus, start.resized(P), iters=args.iters, **fkw)
p_single = rs['psf']
p_single.save(os.path.join(args.out, 'psf_single.json'))
res['single'] = dict(n_beads=len(single), report=rs['report'], zcoef=p_single.zcoef.detach().cpu().tolist())
print('single', len(single), json.dumps(rs['report']), flush=True)

# ---- scoring on the held-out beads2 clusters
sc = {}
for name, p in [('dense', pd), ('iso8', p_iso8), ('iso40', p_iso40), ('single', p_single), ('start', start)]:
    rr = dense_nll(ho, focus, p, iters=200, **fkw)
    sc[name] = dict(nll_per_px=rr['report']['nll_per_px'], deviance_per_px=rr['report']['deviance_per_px'],
                    chi2_median=rr['report']['chi2_median'], n_clusters=ho.n_clusters, n_beads=ho.n_beads)
    print('heldout', name, json.dumps(sc[name]), flush=True)
res['heldout'] = sc

# ---- per-cluster outputs for the figures (data / model of the dense fit; positions, deviance)
cl = []
for c in range(data.n_clusters):
    b0, b1 = data.cl_bead[c], data.cl_bead[c + 1]
    cl.append(dict(stack=int(data.clusters[c]['stack']), box=list(map(int, data.clusters[c]['box'])),
                   n=int(b1 - b0), train=bool(r['train'][c]), heldout=bool(r['heldout'][c]),
                   chi2=float(r['chi2'][c]), deviance=float(r['deviance'][c]),
                   xy=r['xy'][b0:b1].cpu().tolist(), focus_offset=r['focus_offset'][b0:b1].cpu().tolist(),
                   photons=r['photons'][b0:b1].mean(1).cpu().tolist()))
res['clusters'] = cl
# a few dense clusters: data, model, at all planes (for the figure)
big = sorted([c for c in tr_ids if data.cl_bead[c + 1] - data.cl_bead[c] >= 2],
             key=lambda c: -(data.cl_bead[c + 1] - data.cl_bead[c]) - 0.01 * np.random.default_rng(c).random())[:6]
fig = {}
for c in big:
    fig[f'data_{c}'] = data.unflatten(data.d, c).cpu().numpy()
    fig[f'model_{c}'] = data.unflatten(r['mu'], c).cpu().numpy()
np.savez_compressed(os.path.join(args.out, 'clusters_fig.npz'), focus=focus.cpu().numpy(), ids=np.array(big), **fig)
det = [dict(mp=s.max(0).values.cpu().numpy(), **{k: v for k, v in d.items() if k in ('xy', 'amp', 'mask')},
            kept=[c['box'] for c in d['kept']], dropped=[(c['box'], c['why']) for c in d['dropped']])
       for s, d in zip(st, de.detections)]
np.save(os.path.join(args.out, 'detections.npy'), np.array(det, dtype=object), allow_pickle=True)
json.dump(res, open(os.path.join(args.out, 'real_results.json'), 'w'), indent=1)

# ---- field dependence: the clusters of n x n regions of the field (both stacks), each fitted separately
if not args.skip_regions:
    n = args.regions
    x0, y0, x1, y1 = REGION
    cen = np.array([[np.mean(np.asarray(c['xy'])[:, 0]), np.mean(np.asarray(c['xy'])[:, 1])] for c in data.clusters])
    reg = []
    excl = list(de.report.get('rejected_clusters', [])) + res['heldout_clusters']['dropped']   # misfits, aggregates
    for iy in range(n):
        for ix in range(n):
            sel = np.nonzero((cen[:, 0] >= (x1 - x0) * ix / n) & (cen[:, 0] < (x1 - x0) * (ix + 1) / n) &
                             (cen[:, 1] >= (y1 - y0) * iy / n) & (cen[:, 1] < (y1 - y0) * (iy + 1) / n) &
                             ~np.isin(np.arange(data.n_clusters), excl))[0]
            nb = int(sum(data.cl_bead[c + 1] - data.cl_bead[c] for c in sel))
            if nb < 4:
                reg.append(dict(ix=ix, iy=iy, n_beads=nb, n_clusters=len(sel)))
                continue
            t0 = time.time()
            rr = fit_dense_beads(data.subset(sel), focus, pd, iters=args.iters, warmup=30, **fkw)
            # the global dense PSF scored on the same clusters (own terms refitted) for the NLL gain of a local PSF
            rg = dense_nll(data.subset(sel), focus, pd, iters=200, **fkw)
            reg.append(dict(ix=ix, iy=iy, n_beads=nb, n_clusters=len(sel), zcoef=rr['psf'].zcoef.detach().cpu().tolist(),
                            deviance_local=rr['report']['deviance_per_px'], deviance_global=rg['report']['deviance_per_px'],
                            nll_local=rr['report']['nll_per_px'], nll_global=rg['report']['nll_per_px'],
                            centre=cen[sel].mean(0).tolist()))
            print('region', ix, iy, nb, f'{time.time() - t0:.0f}s',
                  json.dumps({k: v for k, v in reg[-1].items() if k != 'zcoef'}), flush=True)
    res['regions'] = reg
    json.dump(res, open(os.path.join(args.out, 'real_results.json'), 'w'), indent=1)
print('done', flush=True)
