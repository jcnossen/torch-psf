"""Reference data for the web tests (web/test/), GENERATED on the fly into web/test/.cache/ (not tracked).
web/test/run_tests.py runs this script when the cache is missing or this file has changed.

The reference is torch_psf itself (torch_psf.vectorial.VectorialPSF, torch_psf.spline.SplinePSF), imported from
this repository; no other package and no calibration / fit result is used: all test PSFs are generated from known
parameters (the web app's tetrapod-like and astigmatic presets).

    python tests/make_web_reference.py [--out web/test/.cache] [--torch-psf DIR] [--device cuda]

Outputs:
  forward.json / forward.npz   forward cases: PSF images [B, R, R] and the Jacobian [B, 21, R, R] (x px, y px, focus um,
                               z_e um, Zernike Noll 5..21 waves; forward-mode AD)
  knots.json / knots.npz       spline knots of torch_psf SplinePSF (sample) for the astigmatic preset
  tiff/*.tif, tiff.json/.npz   TIFF reader cases (compressions, BigTIFF, tiles, dtypes, byte order, multi-part)
  stack_<name>.tif, <name>.json  simulated bead z-stacks (3 beads, Poisson noise) of the presets 'astig' and 'tetrapod'
  fit_ref.json                 the Python fit of the web's bead model (web/src/fit/beads.ts) on both simulated stacks and
                               on the tracked one-bead real crop tests/data/web/epfl_npc_bead.tif (public data, see its json)
"""
import argparse, json, math, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap.add_argument('--torch-psf', default=ROOT, help='directory that contains the torch_psf package')
ap.add_argument('--device', default='cuda')
ap.add_argument('--out', default=os.path.join(ROOT, 'web', 'test', '.cache'))
ap.add_argument('--real', default=os.path.join(ROOT, 'tests', 'data', 'web', 'epfl_npc_bead.tif'))
ap.add_argument('--skip-fit', action='store_true')
ap.add_argument('--only', default='', help='comma list of parts: forward,tiff,knots,sim,fit')
a = ap.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, a.torch_psf)
import torch, tifffile
from scipy import ndimage as ndi
from torch_psf.vectorial import VectorialPSF
from torch.func import jvp, functional_call

dev = a.device
os.makedirs(a.out, exist_ok=True)
os.makedirs(os.path.join(a.out, 'tiff'), exist_ok=True)
ZIDX = list(range(5, 22))


def vec(o):
    return VectorialPSF(o['roisize'], pixel_um=o['pixel_um'], NA=o['NA'], n_imm=o['n_imm'], n_med=o['n_med'],
                        wavelength_um=o['wavelength_um'], zernike_indices=ZIDX, npupil=o['npupil'],
                        oversample=o['oversample'], device=dev)


# ---------------------------------------------------------------- forward cases
TETRAPOD_Z = {6: 0.2, 12: -0.2}                   # the web app's presets (web/src/ui/app.ts)
ASTIG_Z = {6: 0.1}
CASES = [
    dict(roisize=16, pixel_um=0.1, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68, npupil=64, oversample=2),
    dict(roisize=20, pixel_um=0.065, NA=1.4, n_imm=1.518, n_med=1.45, wavelength_um=0.6, npupil=48, oversample=3),
    dict(roisize=26, pixel_um=0.10833, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.7, npupil=64, oversample=2),
    dict(roisize=15, pixel_um=0.12, NA=1.2, n_imm=1.33, n_med=1.33, wavelength_um=0.52, npupil=32, oversample=1),
]


def forward_case(o, rng, zern=None):
    m = vec(o)
    B = 5
    zc = torch.tensor(rng.normal(0, 0.04, len(ZIDX)) if zern is None else [zern.get(j, 0.0) for j in ZIDX],
                      dtype=torch.float32, device=dev)
    with torch.no_grad():
        m.zcoef.copy_(zc)
    pos = torch.tensor(np.c_[rng.uniform(-1.5, 1.5, (B, 2)), rng.uniform(0, 0.5, B)], dtype=torch.float32, device=dev)
    focus = torch.tensor(rng.uniform(-0.5, 0.5, B), dtype=torch.float32, device=dev)
    params = {k: v.detach() for k, v in m.named_parameters()}

    def f(pos_, focus_, zc_):
        p = dict(params); p['zcoef'] = zc_
        return functional_call(m, p, (pos_, focus_))
    with torch.no_grad():
        img = f(pos, focus, zc)
    jac = []
    zp, zf, zz = torch.zeros_like(pos), torch.zeros_like(focus), torch.zeros_like(zc)
    for d in range(4 + len(ZIDX)):
        tp, tf, tz = zp.clone(), zf.clone(), zz.clone()
        if d < 2: tp[:, d] = 1
        elif d == 2: tf[:] = 1
        elif d == 3: tp[:, 2] = 1
        else: tz[d - 4] = 1
        _, t = jvp(f, (pos, focus, zc), (tp, tf, tz))
        jac.append(t.detach())
    jac = torch.stack(jac, 1)
    return dict(pos=pos.cpu().numpy(), focus=focus.cpu().numpy(), zcoef=zc.cpu().numpy(), psf=img.cpu().numpy(),
                jac=jac.cpu().numpy())


def make_forward():
    rng = np.random.default_rng(1)
    arrs, meta = {}, []
    for i, o in enumerate(CASES):
        r = forward_case(o, rng, [TETRAPOD_Z, ASTIG_Z, None, None][i])
        for k, v in r.items():
            arrs[f'c{i}_{k}'] = v.astype(np.float32)
        meta.append(dict(optics=o, B=int(r['pos'].shape[0])))
    np.savez(os.path.join(a.out, 'forward.npz'), **arrs)
    json.dump(dict(cases=meta, zernike_noll=ZIDX,
                   jac_directions=['x_px', 'y_px', 'focus_um', 'z_um'] + [f'noll{j}' for j in ZIDX]),
              open(os.path.join(a.out, 'forward.json'), 'w'), indent=1)
    print('forward cases written')


# ---------------------------------------------------------------- the web bead model (same as web/src/fit/beads.ts)
def find_beads(stack, R, min_rel=0.3):
    """Beads = isolated maxima of the mean projection over z (the sum projection is peaked at the bead centre also for
    wide 3D PSFs such as the tetrapod, whose max projection has one maximum per lobe): local maximum over an R x R
    neighbourhood (size R rounded up to odd), above median + min_rel (max - median) of the projection; plateau
    duplicates removed; ROI fully inside the frame. Returns ROI corners (y0, x0): y0 = y - R // 2. Same rule as
    web/src/fit/beads.ts findBeads."""
    mp = stack.astype(np.float64).mean(0).astype(np.float32)
    size = R | 1
    med = np.median(mp)
    loc = (mp == ndi.maximum_filter(mp, size, mode='nearest')) & (mp > med + min_rel * (mp.max() - med))
    pts = sorted(zip(*np.nonzero(loc)), key=lambda p: (-mp[p], p[0], p[1]))
    out = []
    for y, x in pts:
        if not (R // 2 <= y < stack.shape[1] - R // 2 and R // 2 <= x < stack.shape[2] - R // 2):
            continue
        if any(max(abs(y - q[0]), abs(x - q[1])) <= size // 2 for q in out):
            continue
        out.append((int(y), int(x)))
    out.sort()
    return [(y - R // 2, x - R // 2) for y, x in out]


def fit_beads_web(rois, opt, holdout, iters_adam=1500, iters_lbfgs=500):
    """rois [nb, nz, R, R] photons (float32). opt: optics + sign, dz_um, ze_um, fit_dz, fit_drift.
    Model: mu = I_bj psf(x_b + dx(j), y_b + dy(j), ze, f_bj) + bg_bj, f_bj = sign dz s (j - (nz-1)/2) - zoff_b + ze,
    s = exp(logdz); drift dx, dy = sum_d c_d tt^d (d = 1..4, tt = linspace(-1, 1, nz)), shared by all beads (one stack).
    Global: Zernikes 5..21 (+ logdz, drift). Fitted on the beads != holdout; the held-out bead is then refitted with the
    global parameters frozen. NLL per pixel includes lgamma(d + 1)."""
    nb, nz, R, _ = rois.shape
    o = dict(opt); o['roisize'] = R
    m = vec(o)
    d = torch.tensor(rois, device=dev)
    lg = torch.lgamma(d + 1)
    j = torch.arange(nz, device=dev, dtype=torch.float32)
    tt = torch.linspace(-1, 1, nz, device=dev)
    basis = torch.stack([tt, tt ** 2, tt ** 3, tt ** 4])
    peak = d.amax((2, 3)).argmax(1).float()
    bg0 = torch.stack([torch.cat([d[b, :, 0], d[b, :, -1], d[b, :, :, 0], d[b, :, :, -1]], 1).median(1).values
                       for b in range(nb)])
    I0 = (d - bg0[:, :, None, None]).sum((2, 3)).clamp(min=10)
    p = dict(xy=torch.zeros(nb, 2, device=dev),
             zoff=(peak - (nz - 1) / 2) * opt['dz_um'] * opt['sign'],
             logI=torch.log(I0), bg=bg0.clamp(min=0.1).clone(),
             logdz=torch.zeros((), device=dev), drift=torch.zeros(2, 4, device=dev))
    for v in p.values(): v.requires_grad_(True)

    def mu_of(idx):
        dxy = torch.einsum('cd,dz->zc', p['drift'], basis) if opt['fit_drift'] else torch.zeros(nz, 2, device=dev)
        xy = p['xy'][idx, None, :] + dxy[None]
        sc = torch.exp(p['logdz']) if opt['fit_dz'] else torch.ones((), device=dev)
        f = opt['sign'] * opt['dz_um'] * sc * (j - (nz - 1) / 2)[None] - p['zoff'][idx, None] + opt['ze_um']
        pos = torch.cat([xy.reshape(-1, 2), torch.full((len(idx) * nz, 1), opt['ze_um'], device=dev)], 1)
        psf = m(pos, f.reshape(-1)).reshape(len(idx), nz, R, R)
        return torch.exp(p['logI'][idx])[..., None, None] * psf + p['bg'][idx].clamp(min=1e-3)[..., None, None]

    def nll(idx):
        mu = mu_of(idx).clamp(min=1e-4).double()      # float64 sum: the L-BFGS line search needs the resolution
        return (mu - d[idx] * torch.log(mu) + lg[idx]).sum()

    tr = torch.tensor([b for b in range(nb) if b != holdout], device=dev)
    glob = [m.zcoef] + ([p['logdz']] if opt['fit_dz'] else []) + ([p['drift']] if opt['fit_drift'] else [])
    loc = [p['xy'], p['zoff'], p['logI'], p['bg']]

    def run(idx, params, n_adam, n_lbfgs):
        ad = torch.optim.Adam([{'params': [q for q in params if q is not m.zcoef], 'lr': 0.02}]
                              + ([{'params': [m.zcoef], 'lr': 0.005}] if any(q is m.zcoef for q in params) else []))
        for it in range(n_adam):
            ad.zero_grad(); l = nll(idx); l.backward(); ad.step()
        lb = torch.optim.LBFGS(params, lr=1, max_iter=n_lbfgs, history_size=50, line_search_fn='strong_wolfe',
                               tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            lb.zero_grad(); l = nll(idx); l.backward(); return l
        lb.step(closure)
        with torch.no_grad():
            return float(nll(idx))

    l_tr = run(tr, glob + loc, iters_adam, iters_lbfgs)
    for _ in range(20):                       # L-BFGS restarts until the NLL stops improving
        l_new = run(tr, glob + loc, 0, iters_lbfgs)
        if l_tr - l_new < 1e-7 * abs(l_tr): l_tr = min(l_tr, l_new); break
        l_tr = l_new
    out = dict(zernike_noll=ZIDX, zernike_waves=[float(v) for v in m.zcoef.detach().cpu()],
               dz_scale=float(torch.exp(p['logdz'].detach())), drift=p['drift'].detach().cpu().numpy().tolist(),
               nll_train_px=l_tr / (len(tr) * nz * R * R))
    if holdout is not None and holdout >= 0:
        ho = torch.tensor([holdout], device=dev)
        m.zcoef.requires_grad_(False)
        l_ho = run(ho, loc, iters_adam, iters_lbfgs)
        for _ in range(10):
            l_new = run(ho, loc, 0, iters_lbfgs)
            if l_ho - l_new < 1e-7 * abs(l_ho): l_ho = min(l_ho, l_new); break
            l_ho = l_new
        out['nll_heldout_px'] = l_ho / (nz * R * R)
        m.zcoef.requires_grad_(True)
    out['beads'] = [dict(x=float(p['xy'][b, 0]), y=float(p['xy'][b, 1]), zoff=float(p['zoff'][b])) for b in range(nb)]
    return out


# ---------------------------------------------------------------- simulated stacks (the presets)
OIL = dict(pixel_um=0.1, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68, npupil=64, oversample=2)
SIMS = {
    # astigmatic preset (Noll 6 = 0.10) plus small coma / trefoil / spherical, so that several coefficients are tested
    'astig': dict(OIL, sign=1.0, dz_um=0.02, nz=61, ze_um=0.05, H=80, W=120, R=26, Rb=41, gain=0.5, offset=100.0,
                  bg=15.0, I=20000.0, zernike={6: 0.1, 7: 0.02, 8: -0.015, 9: 0.01, 11: -0.03},
                  beads=[dict(x=25.3, y=30.6, zoff=0.0), dict(x=60.7, y=45.2, zoff=0.1), dict(x=95.1, y=35.8, zoff=-0.08)],
                  fit=dict(fit_dz=True, fit_drift=False)),
    # tetrapod-like preset: Noll 6 = +0.20, Noll 12 = -0.20 (z range 3 um, ROI 40 px)
    'tetrapod': dict(OIL, sign=-1.0, dz_um=0.05, nz=61, ze_um=0.05, H=100, W=170, R=40, Rb=61, gain=0.5, offset=100.0,
                     bg=10.0, I=60000.0, zernike=dict(TETRAPOD_Z),
                     beads=[dict(x=35.3, y=50.4, zoff=0.0), dict(x=85.6, y=45.7, zoff=0.1),
                            dict(x=135.2, y=55.1, zoff=-0.1)],
                     fit=dict(fit_dz=False, fit_drift=False)),
}


def make_sim(name):
    s = SIMS[name]
    Rb = s['Rb']
    o = dict(s); o['roisize'] = Rb
    m = vec(o)
    with torch.no_grad():
        m.zcoef.copy_(torch.tensor([s['zernike'].get(j, 0.0) for j in ZIDX], device=dev))
    nz = s['nz']
    rng = np.random.default_rng(7)
    img = np.full((nz, s['H'], s['W']), s['bg'], np.float64)
    img *= (1 + 0.1 * np.sin(np.arange(nz) / 7))[:, None, None]        # background drifts a little per plane
    for b in s['beads']:
        xi, yi = int(round(b['x'])), int(round(b['y']))
        j = np.arange(nz)
        f = s['sign'] * s['dz_um'] * (j - (nz - 1) / 2) - b['zoff'] + s['ze_um']
        pos = torch.tensor(np.c_[np.full(nz, b['x'] - xi), np.full(nz, b['y'] - yi), np.full(nz, s['ze_um'])],
                           dtype=torch.float32, device=dev)
        with torch.no_grad():
            psf = m(pos, torch.tensor(f, dtype=torch.float32, device=dev)).cpu().numpy()
        I = s['I'] * (1 - 0.2 * j / nz)                                 # bleaching
        h = Rb // 2
        img[:, yi - h:yi + h + 1, xi - h:xi + h + 1] += I[:, None, None] * psf
    adu = np.round(rng.poisson(img) / s['gain'] + s['offset']).astype(np.uint16)
    tifffile.imwrite(os.path.join(a.out, f'stack_{name}.tif'), adu, imagej=True, metadata={'axes': 'ZYX'})
    truth = dict(s); truth['zernike'] = {str(k): v for k, v in s['zernike'].items()}
    json.dump(truth, open(os.path.join(a.out, f'{name}.json'), 'w'), indent=1)
    print('sim stack written', name, adu.shape)
    return adu


# ---------------------------------------------------------------- spline knots
def make_knots():
    from torch_psf.spline import SplinePSF, emitter_height_pos_fn
    o = dict(roisize=13, pixel_um=0.1, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.68, npupil=64, oversample=3)
    zc = np.array([ASTIG_Z.get(j, 0.0) for j in ZIDX], np.float32)
    sp = SplinePSF(o['roisize'], (0.0, 0.4), oversample=2, dz_um=0.04, zpad=3, xy_limit=3.0, device=dev)
    m = vec(dict(o, roisize=sp.Pl))
    with torch.no_grad():
        m.zcoef.copy_(torch.tensor(zc, device=dev))
    knots = sp.sample(m, emitter_height_pos_fn(0.1))
    np.savez(os.path.join(a.out, 'knots.npz'), knots=knots.cpu().numpy().astype(np.float32), zcoef=zc)
    json.dump(dict(optics=o, focus_um=0.1, zmax=0.4, S=2, hz=0.04, zpad=3, xmax=3.0, z0=sp.z0, u0=sp.u0,
                   shape=list(knots.shape)), open(os.path.join(a.out, 'knots.json'), 'w'), indent=1)
    print('knots written', tuple(knots.shape))


# ---------------------------------------------------------------- TIFF reader cases
def make_tiff_cases():
    rng = np.random.default_rng(3)
    arr16 = rng.integers(0, 4000, (5, 37, 29)).astype(np.uint16)
    arr16[:, 3, :] = 7                          # runs, so PackBits / LZW have something to compress
    t = os.path.join(a.out, 'tiff')
    cases = []

    def w(name, arr, **kw):
        tifffile.imwrite(os.path.join(t, name), arr, **kw); cases.append(dict(file=name, key=name.split('.')[0]))
    w('u16_le.tif', arr16)
    w('u16_be.tif', arr16, byteorder='>')
    w('u16_deflate.tif', arr16, compression='zlib')
    w('u16_lzw.tif', arr16, compression='lzw')
    w('u16_lzw_pred.tif', arr16, compression='lzw', predictor=True)
    w('u16_packbits.tif', arr16, compression='packbits')
    w('u16_bigtiff.tif', arr16, bigtiff=True)
    w('u16_tiled.tif', arr16, tile=(16, 16))
    w('u16_strips.tif', arr16, rowsperstrip=5)
    w('u16_imagej.tif', arr16, imagej=True, metadata={'axes': 'ZYX'})
    w('u16_ome.tif', arr16, metadata={'axes': 'ZYX'}, ome=True)
    arr8 = (arr16 % 251).astype(np.uint8)
    w('u8.tif', arr8)
    arrf = (arr16.astype(np.float32) / 7.0 - 100).astype(np.float32)
    w('f32.tif', arrf)
    w('f32_deflate_be.tif', arrf, compression='zlib', byteorder='>')
    arri = (arr16.astype(np.int32) - 2000).astype(np.int16)
    w('i16.tif', arri)
    w('u32.tif', arr16.astype(np.uint32) * 1000)
    # multi-part movie: movie.tif (frames 0-2) + movie_X1.tif (frames 3-4)
    tifffile.imwrite(os.path.join(t, 'movie.tif'), arr16[:3], photometric='minisblack')
    tifffile.imwrite(os.path.join(t, 'movie_X1.tif'), arr16[3:], photometric='minisblack')
    np.savez(os.path.join(a.out, 'tiff.npz'), u16=arr16, u8=arr8, f32=arrf, i16=arri, u32=arr16.astype(np.uint32) * 1000)
    json.dump(dict(cases=cases, multipart=['movie_X1.tif', 'movie.tif']), open(os.path.join(a.out, 'tiff.json'), 'w'),
              indent=1)
    print('tiff cases written')


def rois_of(stack_adu, gain, offset, R, corners):
    ph = np.clip((stack_adu.astype(np.float32) - offset) * gain, 0, None)
    return np.stack([ph[:, y0:y0 + R, x0:x0 + R] for y0, x0 in corners])


if __name__ == '__main__':
    torch.manual_seed(0)
    only = set(a.only.split(',')) if a.only else None
    want = lambda k: only is None or k in only
    if want('forward'): make_forward()
    if want('tiff'): make_tiff_cases()
    if want('knots'): make_knots()
    stacks = {n: make_sim(n) for n in SIMS} if want('sim') or want('fit') else {}
    if not a.skip_fit and want('fit'):
        res = {}
        real = json.load(open(os.path.splitext(a.real)[0] + '.json'))
        real.update(npupil=64, oversample=2, ze_um=0.05, R=26, fit=dict(fit_dz=False, fit_drift=False))
        cases = [(n, stacks[n], SIMS[n], True) for n in SIMS] + [('real', tifffile.imread(a.real), real, False)]
        for name, stack, o, ho in cases:
            R = o['R']
            corners = find_beads(stack.astype(np.float32), R)
            rois = rois_of(stack, o['gain'], o['offset'], R, corners)
            opt = dict(pixel_um=o['pixel_um'], NA=o['NA'], n_imm=o['n_imm'], n_med=o['n_med'],
                       wavelength_um=o['wavelength_um'], npupil=o['npupil'], oversample=o['oversample'],
                       sign=o['sign'], dz_um=o['dz_um'], ze_um=o['ze_um'], **o['fit'])
            holdout = len(corners) - 1 if ho and len(corners) > 1 else -1
            r = fit_beads_web(rois, opt, holdout)
            r.update(roisize=R, corners_yx=[list(c) for c in corners], holdout=holdout, options=opt,
                     gain=o['gain'], offset=o['offset'])
            res[name] = r
            print(name, 'beads', corners, 'nll train/ho', r['nll_train_px'], r.get('nll_heldout_px'),
                  'zern', np.round(r['zernike_waves'], 4).tolist(), 'dz', r['dz_scale'])
        json.dump(res, open(os.path.join(a.out, 'fit_ref.json'), 'w'), indent=1)
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(a.out) for f in fs)
    print(f'total {sz / 1e6:.1f} MB in {a.out}')
