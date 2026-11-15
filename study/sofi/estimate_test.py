"""Step 4: recover Noll 5-21 from simulated dense blinking movies with the prototype estimator
(torch_psf.sofi.fit.CumulantFit) and compare with the truth and the CRLB.

python estimate_test.py --case base --seeds 0 1 2
cases: see CASES (thin sample in water at the coverslip, random aberrations 0.05 waves rms, optional astigmatic
bias, defocus, confounders: bleaching, drift, row-correlated read noise, off-grid emitters).
"""
import argparse
from common import *
from torch_psf.sofi.fit import CumulantFit, smooth_variance

ap = argparse.ArgumentParser()
ap.add_argument('--case', nargs='*', default=['base'])
ap.add_argument('--seeds', type=int, nargs='*', default=[0])
ap.add_argument('--T', type=int, default=10000)
ap.add_argument('--H', type=int, default=64)
ap.add_argument('--n-outer', type=int, default=None)
ap.add_argument('--method', default='lm')
ap.add_argument('--tag', default='')
a = ap.parse_args()

R, Q, OS = 5, 13, 2
BASE = dict(blink=BlinkModel(duty=0.3, t_on=2.0), photons=300.0, bg=60.0, density=50.0, focus=0.0, astig=0.0,
            grid=True, bleach=0.0, drift=None, row_noise=0.0, n_med=None, aberr_seed=1)
CASES = {
    'base': {},
    'astig0.10': dict(astig=0.10),
    'defocus0.4': dict(focus=0.4),
    'offgrid': dict(grid=False),
    'offgrid_astig0.10': dict(grid=False, astig=0.10),
    'bleach50': dict(bleach=0.693 / 10000 / 0.3),         # ~50% of the emitters bleached at the end (on-time rate)
    'drift2px': dict(drift=(2.0, 1.0)),
    'rownoise3e': dict(row_noise=3.0),
    'smlm_dense': dict(blink=BlinkModel(duty=0.02, t_on=3.0, k_dark=0.05, t_dark=200.0), photons=3000.0, bg=40.0,
                       density=400.0),
}


def run(name, seed):
    c = {**BASE, **CASES[name]}
    truth = random_aberrations(0.05, c['aberr_seed'] + 100 * seed)
    psf = make_psf(Q, truth, astig=c['astig'], focus_um=c['focus'], n_med=c['n_med'])
    blink = BlinkModel(**{**c['blink'].to_dict(), 'k_bleach': c['bleach']})
    spec = DenseMovieSpec(H=a.H, W=a.H, T=a.T, density_um2=c['density'], photons=c['photons'], bg=c['bg'],
                          blink=blink, camera=CameraModel(readnoise_e=1.5, row_noise_e=c['row_noise']),
                          drift_px=c['drift'], Q=Q, grid_os=OS if c['grid'] else None, chunk=1000, seed=seed)
    mv = DenseMovie(psf, spec)
    acc = CumulantAccumulator(a.H, a.H, R, block=500)
    t0 = time.time()
    for _, adu in mv.frames():
        acc.update(mv.to_photons(adu))
    acc.finish()
    t_sim = time.time() - t0
    res = acc.result()
    boot = acc.bootstrap(200)
    var = smooth_variance(boot.var(0))
    start = make_psf(Q, None, astig=c['astig'], focus_um=c['focus'], n_med=c['n_med'])   # known bias only
    est = CumulantFit(start, Q=Q, os=OS, z_layers=(0.0,), focus=c['focus'])
    t0 = time.time()
    out = est.fit(res['G'], res['mask'], var, n_outer=a.n_outer, D=acc.D, method=a.method)
    t_fit = time.time() - t0
    zt = psf.zcoef.detach().cpu().numpy()
    ze = out['zcoef'].cpu().numpy()
    err = (ze - zt) * 1000
    print(f"{name:14s} seed {seed}: n={mv.n} sim {t_sim:.0f}s fit {t_fit:.0f}s dz {out['dz'] * 1000:.1f} nm  "
          f"rms err {np.sqrt((err ** 2).mean()):.2f} mwaves (truth rms {np.sqrt((zt ** 2).mean()) * 1000:.1f})", flush=True)
    print('   err  ' + ' '.join(f'{e:6.2f}' for e in err), flush=True)
    return dict(case=name, seed=seed, n=mv.n, truth=zt, est=ze, err_mw=err, dz_nm=out['dz'] * 1000,
                t_sim=t_sim, t_fit=t_fit, area_um2=a.H * a.H * psf.pixel_um ** 2, T=a.T,
                hist=[h['loss'] for h in est.history])


rows = [run(c, s) for c in a.case for s in a.seeds]
save_json(rows, f"estimate_{'_'.join(a.case)[:50]}{a.tag}.json")
