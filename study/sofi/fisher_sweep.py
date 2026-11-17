"""Step 3: Cramer-Rao bounds for Noll 5-21 from the second-order cross-cumulants (0 < |d| <= R) of a dense blinking
movie, with the density rho profiled out (torch_psf.sofi.fisher). Two nuisance models bracket the answer:
  arbitrary  rho free at every sub-pixel grid point (and depth layer): "any object" (positivity ignored),
  support    rho free only where emitters are (support known): optimistic.
A common z shift of the sample (= defocus gauge) is always a nuisance parameter.
The Fisher matrix is computed on P x P patches and extrapolated to the bulk information per px^2 from two patch
sizes (J(P) = a P^2 + b P), then reported for a reference field of 10 x 10 um and 10^4 frames (CRLB ~ 1 / sqrt(T A)).

python fisher_sweep.py [--quick] [--only name ...]
"""
import argparse
from common import *
from torch_psf.sofi.fisher import fisher, crlb
from torch_psf.sofi.model import GridPSF

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--only', nargs='*', default=None)
ap.add_argument('--sizes', type=int, nargs=2, default=[12, 16])
ap.add_argument('--check-sizes', action='store_true', help='J vs patch size (linearity check) for the base case')
a = ap.parse_args()

R = 5
A_REF_UM2, T_REF = 100.0, 10000
BLINK = {
    'sofi': dict(blink=BlinkModel(duty=0.3, t_on=2.0), eps=300.0, bg=60.0, rn=1.5, F2=1.0),
    'sofi_fast': dict(blink=BlinkModel(duty=0.3, t_on=0.3), eps=300.0, bg=60.0, rn=1.5, F2=1.0),
    'smlm_dense': dict(blink=BlinkModel(duty=0.02, t_on=3.0, k_dark=0.05, t_dark=200.0), eps=3000.0, bg=40.0,
                       rn=1.5, F2=1.0),
}
_stats = {}


def stats(name):
    if name not in _stats:
        _stats[name] = sofi.trace_stats(BLINK[name]['blink'], n=2000, T=8000)
    return _stats[name]


def config(**kw):
    c = dict(blink='sofi', density=50.0, z0=0.0, focus=0.0, z_range=None, nz=1, astig=0.0, aberr_rms=0.05,
             aberr_seed=1, os=2, Q=13, n_med=None)
    c.update(kw)
    return c


def run(c, sizes):
    sc, st = BLINK[c['blink']], stats(c['blink'])
    zl = [c['z0']] if c['z_range'] is None else list(np.linspace(c['z_range'][0], c['z_range'][1], c['nz']))
    psf = make_psf(c['Q'], random_aberrations(c['aberr_rms'], c['aberr_seed']), astig=c['astig'], focus_um=c['focus'],
                   n_med=c['n_med'])
    gp = GridPSF(psf, c['Q'], c['os'], zl, focus=c['focus'])
    h, dh = gp.jac()
    h, dh = h.double(), dh.double()
    g = torch.Generator(device=DEV).manual_seed(7)
    eps = sc['eps']
    cell_um2 = psf.pixel_um ** 2 / c['os'] ** 2 / len(zl)
    Js, Jsup, info = [], [], {}
    Pmax = max(sizes) + c['Q'] - 1
    cnt_all = torch.poisson(torch.full((gp.Np, Pmax, Pmax), c['density'] * cell_um2, device=DEV, dtype=torch.float64),
                            generator=g)                       # one sample; the smaller patch is a crop of it
    for P in sizes:
        Pm = P + c['Q'] - 1
        cnt = cnt_all[:, :Pm, :Pm].contiguous()
        rho, k3, c4 = cnt * eps ** 2 * st['var'], cnt * eps ** 3 * st['k3'], cnt * eps ** 4 * st['c4']
        # mean image -> noise variance (photons^2): F2 * mean + read variance
        from torch_psf.sofi.fisher import patch_matrix
        Hm = patch_matrix(h, P)
        mean = sc['bg'] + Hm @ (cnt.reshape(-1) * eps * st['mean'])
        Nvar = (sc['F2'] * mean + sc['rn'] ** 2).reshape(P, P)
        t0 = time.time()
        r1 = fisher(h, dh, rho, P, R, T_REF, st['alpha'], Nvar, k3, c4, sc['F2'])
        r2 = fisher(h, dh, rho, P, R, T_REF, st['alpha'], Nvar, k3, c4, sc['F2'], support=cnt > 0)
        Js.append(r1['J']); Jsup.append(r2['J'])
        info[P] = dict(M=r1['M'], Ng=r1['Ng'], rank=r1['rank'], Ng_support=r2['Ng'], rank_support=r2['rank'],
                       time_s=round(time.time() - t0, 1))
        del Hm
        torch.cuda.empty_cache()
    # bulk information per px^2 from two sizes: J = a P^2 + b P
    P1, P2 = sizes
    bulk = lambda J1, J2: (J2 / P2 - J1 / P1) / (P2 - P1)
    A_px = A_REF_UM2 / psf.pixel_um ** 2
    out = dict(config=c, patch=info, stats={k: v for k, v in st.items() if k != 'r'})
    names = [f'Z{j}' for j in NOLL] + ['dz']
    for tag, (J1, J2) in (('arbitrary', Js), ('support', Jsup)):
        Jb = bulk(J1, J2) * A_px
        cr = crlb(Jb)
        out[tag] = {n: float(v) * 1000 for n, v in zip(names, cr)}             # mwaves (dz: nm)
        out[tag + '_small_patch_P%d' % P2] = {n: float(v) * 1000 for n, v in zip(names, crlb(J2 * (A_px / P2 ** 2)))}
    return out


def fmt(r, key='arbitrary'):
    v = r[key]
    return ' '.join(f"{v[f'Z{j}']:7.2f}" if np.isfinite(v[f'Z{j}']) else '    inf' for j in NOLL)


CASES = {
    'perfect_thin_focus0': config(aberr_rms=0.0),
    'perfect_thin_focus0_astig0.10': config(aberr_rms=0.0, astig=0.10),
    'perfect_thin_focus0_matched': config(aberr_rms=0.0, n_med=1.518),
    'base_thin_focus0_matched': config(n_med=1.518),
    'thin_defocus0.4_matched': config(focus=0.4, n_med=1.518),
    'base_thin_focus0': config(),
    'thin_focus0_astig0.05': config(astig=0.05),
    'thin_focus0_astig0.10': config(astig=0.10),
    'thin_focus0_astig0.15': config(astig=0.15),
    'thin_defocus0.2': config(focus=0.2),
    'thin_defocus0.4': config(focus=0.4),
    'thin_defocus0.4_astig0.10': config(focus=0.4, astig=0.10),
    'density10': config(density=10.0),
    'density30': config(density=30.0),
    'density150': config(density=150.0),
    'density500': config(density=500.0),
    'sofi_fast_blink': config(blink='sofi_fast'),
    'smlm_dense_400': config(blink='smlm_dense', density=400.0),
    'smlm_dense_400_astig0.10': config(blink='smlm_dense', density=400.0, astig=0.10),
    'z0-0.4_3layers': config(z_range=(0.0, 0.4), nz=3, Q=15),
    'z0-0.8_5layers': config(z_range=(0.0, 0.8), nz=5, Q=17),
    'z0-0.8_5layers_astig0.10': config(z_range=(0.0, 0.8), nz=5, Q=17, astig=0.10),
}

if __name__ == '__main__':
    if a.check_sizes:
        c = config()
        rows = {}
        for P in (8, 12, 16, 20):
            rr = run(c, [P, P + 4])
            rows[P] = rr
            print(P, rr['patch'], '\n  arb', fmt(rr), '\n  sup', fmt(rr, 'support'), flush=True)
        save_json(rows, 'fisher_sizecheck.json')
        sys.exit()
    sel = a.only or (list(CASES)[:3] if a.quick else list(CASES))
    res = {}
    print('CRLB (mwaves rms) per Noll mode, 10x10 um, 1e4 frames:   ' + ' '.join(f'{"Z%d" % j:>7s}' for j in NOLL))
    for name in sel:
        t0 = time.time()
        r = run(CASES[name], a.sizes)
        res[name] = r
        print(f"{name:28s} arb {fmt(r)}  dz {r['arbitrary']['dz']:.1f} nm  ({time.time() - t0:.0f}s)", flush=True)
        print(f"{'':28s} sup {fmt(r, 'support')}", flush=True)
        save_json(res, 'fisher_sweep.json' if not a.only else 'fisher_sweep_' + '_'.join(a.only)[:60] + '.json')
