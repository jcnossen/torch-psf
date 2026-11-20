"""Reference data for the SMLM-movie mode of the web version, computed with torch_psf itself (the web code is a port
of these functions and must agree with them). Written to web/test/.cache/smlm/ (git-ignored; nothing generated is
tracked).

    python tests/make_web_smlm_reference.py [--torch-psf DIR] [--device cuda]

Cases:
  movie.tif + movie.json       a seeded movie (torch_psf.simulate.simulate_movie, astigmatic vectorial PSF via its
                               SplinePSF), its optics and truth
  detect.json                  torch_psf.spots.extract_spots on it (chunk = 100 frames, threshold 6, border h + 2):
                               frame, corner, score per spot, and the noise / median of each chunk
  localize.npz / .json         the spline knots (VectorialPSF.to_spline), the ROIs of the first 600 detections,
                               torch_psf.fit.localize (theta, nll) and crlb, and the table columns
                               (theta, nll, crlb, chi2 as the localization table of PSF.fit)
  pt_movie.tif +               torch_psf.camera.photon_transfer on a movie with a background varying over the field
  photon_transfer.json         (gain 0.5 e-/ADU, offset 100 ADU, read noise 1.5 e-): defaults, offset_adu given,
                               excess_noise 2, and non-default keywords (frames, nblocks, block, mask_radius,
                               threshold, nbins); per block also the number of detect_spots detections
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--torch-psf', default=ROOT, help='directory that contains the torch_psf package')
ap.add_argument('--device', default='cuda')
ap.add_argument('--out', default=os.path.join(ROOT, 'web', 'test', '.cache', 'smlm'))
a = ap.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, a.torch_psf)
import torch, tifffile
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import SplinePSF
from torch_psf.simulate import simulate_movie
from torch_psf.spots import extract_spots
from torch_psf.detect import filter_frames
from torch_psf.fit import localize, crlb, model_and_jacobian, at_limit

os.makedirs(a.out, exist_ok=True)
dev = a.device
OPT = dict(pixel_um=0.1, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68)
NOLL = list(range(5, 22))
ZC = {6: 0.1, 11: 0.02}
P, ZR, FOCUS, CHUNK = 15, (0.0, 0.8), 0.4, 100
GRID = dict(oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0)

vec = VectorialPSF(P, zernike_indices=NOLL, npupil=64, oversample=2, device=dev, focus_um=FOCUS, z_range=ZR, **OPT)
with torch.no_grad():
    for j, v in ZC.items():
        vec.zcoef[NOLL.index(j)] = v
spl = vec.to_spline(P, ZR, **GRID)

movie, truth = simulate_movie(spl.render, P, T=300, H=96, W=112, nsites=120, p_on=0.01, mean_on=3.0, bg=20.0,
                              photons=3000.0, z_range=(0.1, 0.7), seed=3, device=dev)
tifffile.imwrite(os.path.join(a.out, 'movie.tif'), movie.numpy())
json.dump(dict(optics=OPT, noll=NOLL, zernike=ZC, roisize=P, z_range=ZR, focus_um=FOCUS, npupil=64, oversample=2,
               grid=GRID, chunk=CHUNK, T=300, H=96, W=112, n_events=len(truth)),
          open(os.path.join(a.out, 'movie.json'), 'w'), indent=1)

# detection (extract_spots) and the per-chunk noise / median as computed by detect_spots
sp = extract_spots(movie.numpy(), P, None, chunk=CHUNK, device=dev, threshold=6.0)
chunks = []
for t0 in range(0, movie.shape[0], CHUNK):
    x = torch.as_tensor(movie[t0:t0 + CHUNK].numpy()).to(dev).float()
    f = filter_frames(x)
    med = f.median()
    chunks.append(dict(t0=t0, median=float(med), noise=float(1.4826 * (f - med).abs().median())))
json.dump(dict(frame=sp.frame.tolist(), corner=sp.corner.tolist(), score=sp.score.tolist(), chunks=chunks),
          open(os.path.join(a.out, 'detect.json'), 'w'))
print('detections', len(sp), 'events', len(truth))

# localization of the first n spots on the spline
n = min(600, len(sp))
s = sp.subset(slice(0, n))
th, nll = localize(spl, s.rois)
cr = crlb(spl, th)
mu, _ = model_and_jacobian(spl, th)
h = (P - 1) // 2
tab = dict(frame=s.frame.cpu().numpy(), x=(s.corner[:, 0] + h + th[:, 0]).cpu().numpy(), y=(s.corner[:, 1] + h + th[:, 1]).cpu().numpy(),
           z=th[:, 2].cpu().numpy(), chi2=((s.rois - mu) ** 2 / mu.clamp_min(1e-6)).flatten(1).mean(1).cpu().numpy(),
           at_limit=at_limit(th, spl).cpu().numpy().astype(np.uint8))
np.savez(os.path.join(a.out, 'localize.npz'), knots=spl.knots.cpu().numpy(), rois=s.rois.cpu().numpy(),
         theta=th.cpu().numpy(), nll=nll.cpu().numpy(), crlb=cr.cpu().numpy(),
         **{'tab_' + k: v for k, v in tab.items()})
json.dump(dict(n=n, z0=spl.z0, hz=spl.hz, S=spl.S, nz=spl.nz, nv=spl.nv, nu=spl.nu, xmax=spl.xmax, c=spl.c,
               z_limits=spl.z_limits), open(os.path.join(a.out, 'localize.json'), 'w'), indent=1)
print('localized', n, 'median |x - truth| etc. not checked here')

# ---------------------------------------------------------------------------------------------- in-situ fit
# a reference movie (simulate_reference_movie, astigmatic preset: Zernikes {5: 0.02, 6: 0.10, 11: 0.03}, focus 0.4 um,
# z 0 .. 0.8 um, camera gain 0.5 e-/ADU, offset 100, read noise 1.5 e-) and InSituEstimator from a start PSF with
# only Noll 6 = 0.07 (defaults: overlap rejection on, deviance_k 3, second pass with 3/4 of the iterations)
from torch_psf.simulate import simulate_reference_movie
from torch_psf.estimators import InSituEstimator
from torch_psf.estimation import knot_gradient, split_checkerboard, fit_zernike_knots
from torch_psf.composition import ZMode
INS = dict(T=1500, H=96, W=96, emitters_per_frame=8.0, seed=5)
r = simulate_reference_movie('astigmatic', device=dev, **INS)
tifffile.imwrite(os.path.join(a.out, 'insitu_movie.tif'), r['movie_adu'])
truth_vec = r['psf']
start = VectorialPSF(15, zernike_indices=NOLL, npupil=64, oversample=3, device=dev, focus_um=0.4, z_range=(0.0, 0.8),
                     pixel_um=0.108, NA=1.49, wavelength_um=0.68, n_imm=1.518, n_med=1.33)
with torch.no_grad():
    start.zcoef[NOLL.index(6)] = 0.07
estm = InSituEstimator(start=start, camera=r['camera'], roisize=15, z_range=(0.0, 0.8), nspots=3000, nheldout=1500, iters=40,
                       device=dev)
psf_fit = estm.estimate(r['movie_adu'])
rep = estm.report
json.dump(dict(params=r['params'], optics=dict(pixel_um=0.108, NA=1.49, wavelength_um=0.68, n_imm=1.518, n_med=1.33),
               noll=NOLL, truth_zernike={str(j): float(truth_vec.zcoef[truth_vec.zidx.index(j)]) for j in NOLL if j in truth_vec.zidx},
               start_zernike={'6': 0.07}, focus_um=0.4, z_range=[0.0, 0.8], roisize=15, nspots=3000, nheldout=1500, iters=40,
               final_zernike=psf_fit.zcoef.tolist(),
               report={k: v for k, v in rep.items() if isinstance(v, (int, float, str, dict, list)) or v is None}),
          open(os.path.join(a.out, 'insitu.json'), 'w'), indent=1, default=float)
print('insitu: spots', rep.get('n_spots'), 'train', rep.get('n_train'), 'rejected', rep.get('n_rejected'), 'final', np.round(psf_fit.zcoef.tolist(), 4).tolist())
print('heldout', rep.get('heldout'), 'z_scale', rep.get('z_scale'), 'at_limit', rep.get('at_limit'))

# the gradient at the start: spline grid of InSituEstimator (zpad 5), theta by localize, knot gradient, VJP
spl5 = SplinePSF(15, (0.0, 0.8), oversample=2, dz_um=0.04, zpad=5, xy_limit=3.0, device=dev)
big = start.resized(spl5.Pl)
pf = ZMode(big, 'emitter', focus=.4).coordinates
spl5.knots = spl5.sample(big, pf, chunk=spl5.nz)
sp = extract_spots(r['movie_adu'], 15, r['camera'], device=dev, max_spots=2 * 4500, threshold=6.0)
par = split_checkerboard(sp.corner[:, 0] + 7, sp.corner[:, 1] + 7, 16)
tr = np.flatnonzero(par == 0)[:3000]
d = sp.rois[tr]
th, _ = localize(spl5, d)
gK, nll0 = knot_gradient(spl5, d, th)
gz = spl5.vjp_vectorial(big, pf, gK, chunk=spl5.nz)['zcoef']
np.savez(os.path.join(a.out, 'insitu_grad.npz'), knots=spl5.knots[0].cpu().numpy(), rois=d.cpu().numpy(),
         theta=th.cpu().numpy(), gK=gK.cpu().numpy(), gz=gz.cpu().numpy(), nll=np.float64(nll0),
         train_frame=sp.frame[tr].cpu().numpy(), train_corner=sp.corner[tr].cpu().numpy())
print('gradient: nll', nll0, 'gz', np.round(gz.cpu().numpy(), 2).tolist())

# fit_zernike_knots alone on these training ROIs (no overlap rejection), with the Zernikes of every iteration
hist = []
spl5b = SplinePSF(15, (0.0, 0.8), oversample=2, dz_um=0.04, zpad=5, xy_limit=3.0, device=dev)
fz = fit_zernike_knots(spl5b, d, start, iters=40, log=lambda it, nll, sec, zc: hist.append(dict(it=it, nll=nll, zcoef=zc.tolist())))
json.dump(dict(final_zernike=fz['psf'].zcoef.tolist(), history=hist), open(os.path.join(a.out, 'insitu_fit.json'), 'w'), indent=1)
print('fit_zernike_knots only:', np.round(fz['psf'].zcoef.tolist(), 4).tolist())

# the spot deviance of the overlap rejection (estimation.spot_deviance) on these ROIs with the start spline
from torch_psf.estimation import spot_deviance
dv, thd = spot_deviance(spl5, d)
np.savez(os.path.join(a.out, 'insitu_deviance.npz'), deviance=dv.astype(np.float32), theta=thd.cpu().numpy())
print('deviance: median', float(np.median(dv)), 'above 1.283:', int((dv > 1 + 3 * np.sqrt(2 / 225)).sum()))

# free-knot refinement (fit_free_knots, as InSituEstimator(refine=True)): 3 outer iterations from the vectorial spline
# of fit_zernike_knots on the training ROIs, regularised towards it (lam_ref 3000, lam_s 1000)
from torch_psf.estimation import fit_free_knots
s_ref = fz['psf'].to_spline(15, (0.0, 0.8), oversample=2, dz_um=0.04, zpad=5, xy_limit=3.0)
K_vec = s_ref.knots[0].clone()
hist_r = []
rr = fit_free_knots(s_ref, K_vec, rois=d, ref=K_vec, lam_ref=3000.0, lam_s=1000.0, outer=3,
                    log=lambda o, l, sec, K: hist_r.append(dict(it=o, loss=float(l))))
np.savez(os.path.join(a.out, 'insitu_refine.npz'), K_vec=K_vec.cpu().numpy(), knots=rr['psf'].knots[0].cpu().numpy())
json.dump(dict(history=hist_r, outer=3, lam_ref=3000.0, lam_s=1000.0), open(os.path.join(a.out, 'insitu_refine.json'), 'w'), indent=1)
print('refine: loss per outer', [round(h['loss'], 2) for h in hist_r])

# the whole estimator with refine=True
estr = InSituEstimator(start=start, camera=r['camera'], roisize=15, z_range=(0.0, 0.8), nspots=3000, nheldout=1500, iters=40,
                       refine=True, refine_outer=4, device=dev)
psf_r = estr.estimate(r['movie_adu'])
rep_r = estr.report
json.dump(dict(chosen=rep_r['chosen'], refine=rep_r.get('refine'), heldout=rep_r.get('heldout'), refine_outer=4),
          open(os.path.join(a.out, 'insitu_refine_est.json'), 'w'), indent=1, default=float)
print('estimator refine:', rep_r['chosen'], rep_r.get('refine'), {k: v['nll_per_px'] for k, v in rep_r['heldout'].items()})

# localization files for the viewer, written by torch_psf.Dataset (HDF5, Picasso HDF5 + yaml, ThunderSTORM csv)
from torch_psf.dataset import Dataset
g = torch.Generator().manual_seed(7)
nloc = 2000
pos = torch.stack([torch.rand(nloc, generator=g) * 90 + 3, torch.rand(nloc, generator=g) * 70 + 3, torch.rand(nloc, generator=g) * 6], 1).double()
ds = Dataset(dict(frame=torch.randint(0, 500, (nloc,), generator=g).int(), pos=pos, crlb=torch.rand(nloc, 3, generator=g) * 0.2 + 0.05,
                  photons=torch.rand(nloc, generator=g) * 3000 + 500, bg=torch.rand(nloc, generator=g) * 20 + 5),
             dict(pixel_nm=108.0, image_size=[80, 100]))
ds.save(os.path.join(a.out, 'locs_dataset.hdf5'))
ds.save_picasso(os.path.join(a.out, 'locs_picasso.hdf5'))
ds.save_csv(os.path.join(a.out, 'locs_ts.csv'))
F = ds.numpy()
np.savez(os.path.join(a.out, 'locs_ref.npz'), frame=F['frame'].astype(np.int32), pos=F['pos'].astype(np.float64), crlb=F['crlb'].astype(np.float32),
         photons=F['photons'].astype(np.float32))
print('viewer files: dataset / picasso / csv,', nloc, 'localizations;', sorted(os.listdir(a.out))[:0])

# camera calibration from the movie (torch_psf.camera.photon_transfer). The movies above have a uniform background (no
# lever arm for the variance-vs-mean line), so a movie as tests/test_u373fixes.py::test_photon_transfer: background
# 20 .. 50 photons varying over the field, blinking Gaussian spots, gain 0.5 e-/ADU, offset 100 ADU, read noise
# 1.5 e-, rounded to uint16 (cpu, seeded). Cases: defaults, offset_adu given, excess_noise 2, other keywords.
from torch_psf.camera import photon_transfer
from torch_psf.detect import detect_spots
from torch_psf.gaussian import GaussianPSF
g = torch.Generator().manual_seed(11)
T, H, W = 1200, 64, 80
bgm = 20.0 + 30.0 * torch.rand(1, H, W, generator=g)
spots, _ = simulate_movie(GaussianPSF(11, sigma_px=1.3, device='cpu').render, 11, T=T, H=H, W=W, nsites=40, p_on=0.03,
                          mean_on=2.0, bg=0.0, photons=2000.0, photons_logsd=0.2, z_range=(0.0, 0.0), margin=6,
                          generator=g, device='cpu')
e = torch.poisson(bgm.expand(T, H, W), generator=g) + spots.float()
adu = (e / 0.5 + 100.0 + (1.5 / 0.5) * torch.randn(e.shape, generator=g)).round().clamp(0, 65535).numpy().astype('uint16')
tifffile.imwrite(os.path.join(a.out, 'pt_movie.tif'), adu)
PT = [dict(), dict(offset_adu=100.0), dict(excess_noise=2.0),
      dict(frames=[50, 1150], nblocks=4, block=150, mask_radius=3, threshold=5.0, nbins=16)]
pt = []
for kw in PT:
    cam, rep = photon_transfer(adu, device=dev, **kw)
    th = kw.get('threshold', 4.0)
    nd = [int(len(detect_spots(torch.as_tensor(adu[b['frames'][0]:b['frames'][1]], device=dev).float(), threshold=th)['frame']))
          for b in rep['per_block']]
    pt.append(dict(kwargs=kw, camera=cam.to_dict(), report=rep, n_spots=nd))
    print('photon_transfer', kw, 'gain', round(cam.gain_e_per_adu, 4), 'offset', round(cam.offset_adu, 3),
          'readnoise', round(cam.readnoise_e, 3), 'blocks', len(rep['per_block']), 'spots', nd)
json.dump(dict(T=T, H=H, W=W, gain_e_per_adu=0.5, offset_adu=100.0, readnoise_e=1.5, cases=pt),
          open(os.path.join(a.out, 'photon_transfer.json'), 'w'), indent=1, default=float)
