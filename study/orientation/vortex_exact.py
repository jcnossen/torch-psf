"""Their model (vecfitcpu_vortex, Hulleman et al. 2021) expressed EXACTLY in our VectorialPSF, checked against
simcode2's PyTorch port of their code (~/vortex_ref/vecfitcpu_vortex-pytorch, validated against Octave), and the
per-spot comparison with their Octave tables (~/vortex_ref/out/tables).

The mapping (study/orientation/vortex_crosscheck.py found it; max relative image difference 0.2-0.9 %, the rest is
their two-interface Fresnel through the 1.523 coverslip vs our index-matched coverslip):

  their image(x, y, z, mu, W)  =  rot180[ our image(-x', -y', z, mu', -W) ]

  * arrays: their pupil and image x run along the first array index (image row), y along the column, so their pupil
    maps W (vortex plate DOE + Wfocus = immersion mismatch + nominal focus, and the per-spot NAT Zernikes) are our
    pupil maps array by array, with the sign flipped (their exp(+i 2 pi W / lambda) with cos_med = -i|.| beyond the
    critical angle is the complex conjugate of our convention);
  * positions: x' = their y (column), y' = their x (row); the dipole mu' = (mu_y, mu_x, mu_z) of theirs, unchanged
    otherwise; z the same (their stage z);
  * the image is rotated by 180 deg: we fit rot180(data).
  * their g2 / N from our g / I through the pupil energies (vortex_baseline.to_their_g2).

Stages: check (the image identity on random spots), dna (their Octave lambda-DNA table: the same ROIs, our fit from
their initial values and from z-grid starts; per-spot differences), pmma (PMMA per slice with their model, n_med 1.0,
z grid / z0 = 0).
usage: CUDA_VISIBLE_DEVICES=2 python -I study/orientation/vortex_exact.py check|dna|pmma|insitu_dna [refmed=1.0] [zgrid]
       CUDA_VISIBLE_DEVICES=2 python -I study/orientation/vortex_exact.py acceptance     (the reference numbers, ~12 min;
       needs ~/vortex_ref (simcode2's port and tables) and ~/data/vortex_hulleman2021)
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
REF = os.environ.get('VORTEX_REF', os.path.expanduser('~/vortex_ref'))
sys.path.insert(0, os.path.join(REF, 'vecfitcpu_vortex-pytorch', 'python'))
import numpy as np                                                                     # noqa: E402
import torch                                                                           # noqa: E402
import vortex_real as VR                                                              # noqa: E402
import vortex_baseline as VB                                                          # noqa: E402
from torch_psf import VectorialPSF                                                     # noqa: E402
from torch_psf.zernike import zernike                                                  # noqa: E402
from torch_psf.orientation import FocusZ, fit_orientation, model_and_jacobian, unit    # noqa: E402
from vecfit.set_parameters_vortex_lambdaDNA import set_parameters_vortex_lambdaDNA    # noqa: E402
from vecfit.funfit.get_pupil_matrix import pupil_static                               # noqa: E402
from vecfit.pipeline import nat_predictions                                           # noqa: E402

DEV = VR.DEV
OUT = VR.OUT
NOLL = tuple(range(4, 23))


def log(*a):
    print(*a, flush=True)


class Exact:
    """our model of their model for a parameter set (their port's set_parameters_vortex_lambdaDNA overrides)"""

    def __init__(self, **over):
        self.p = p = set_parameters_vortex_lambdaDNA(**over)
        self.st = st = pupil_static(p)
        lam = p['lambda']
        v = VectorialPSF(p['Mx'], pixel_um=p['pixelsize'] / 1e3, NA=p['NA'], n_imm=p['refimm'], n_med=p['refmed'],
                         wavelength_um=lam / 1e3, npupil=p['Npupil'], oversample=1, device=DEV,
                         zernike_indices=NOLL, focus_um=0.0, z_range=(-1.1, 1.1))
        for q in v.parameters():
            q.requires_grad_(False)
        W = (np.real(st['W0']) + np.real(st['Wfocus'])) / lam * st['ApertureMask']
        with torch.no_grad():
            v.pupil_phase.copy_(-torch.as_tensor(W, dtype=torch.float32, device=DEV))
        self.v = v
        self.psf = FocusZ(v, height=0.0, z_range=(-1.1, 1.1))
        # their Zernike maps (nm per nm of coefficient) -> our Noll coefficients (waves), sign flipped
        Zt = st['allzernikes'] * st['normfac'][None, None]                     # [N, N, 12]
        m = v.mask.bool().cpu().numpy()
        Zo = np.stack([zernike(j, v.rho.clamp(max=1), v.phi).cpu().numpy() for j in NOLL], -1)   # [N, N, 18]
        A, res, *_ = np.linalg.lstsq(Zo[m], Zt[m], rcond=None)                 # Zt = Zo A
        self.proj = torch.as_tensor(-A.T / lam, dtype=torch.float32, device=DEV)   # [12, 18]: nm -> our waves
        self.proj_residual = float(np.abs(Zo[m] @ A - Zt[m]).max())

    def offsets(self, nat_nm):
        return torch.as_tensor(nat_nm, dtype=torch.float32, device=DEV) @ self.proj

    def nat(self, roixy, fov):
        """their NAT prediction (nm) for roixy [S, 2] = (column, row), 1-based, frame width fov"""
        return nat_predictions(self.p, np.asarray(roixy, float), fov)[0]

    # theirs <-> ours parameters (ours in the rot180 frame of the ROI)
    @staticmethod
    def their_to_ours(th_t, pixel_nm=65.0):
        """their theta [S, 8] (x row nm, y column nm, z nm, N, bg, azim, pola, g2) -> our theta0 [S, 9] (I = N,
        g = g2: starting values)"""
        th_t = torch.as_tensor(th_t, dtype=torch.float32, device=DEV)
        a, p = th_t[:, 5], th_t[:, 6]
        mu_t = torch.stack([torch.sin(p) * torch.cos(a), torch.sin(p) * torch.sin(a), torch.cos(p)], 1)
        th = torch.zeros(len(th_t), 9, device=DEV)
        th[:, 0] = -th_t[:, 1] / pixel_nm; th[:, 1] = -th_t[:, 0] / pixel_nm; th[:, 2] = th_t[:, 2] / 1e3
        th[:, 3] = th_t[:, 3]; th[:, 4] = th_t[:, 4]
        th[:, 5:8] = mu_t[:, [1, 0, 2]]; th[:, 8] = th_t[:, 7]
        return th

    @staticmethod
    def ours_to_their_axis(mu):
        """our mu [S, 3] (rot180 frame) -> their axis (mu_x row, mu_y column, mu_z)"""
        return mu[:, [1, 0, 2]]

    def check(self, n=6, seed=0):
        """image identity with their port on random spots"""
        from vecfit.funfit.get_pupil_matrix import get_pupil_matrix
        from vecfit.funfit.poissonrate import poissonrate
        rng = np.random.default_rng(seed)
        roixy = np.stack([rng.uniform(50, 1000, n), rng.uniform(50, 1000, n)], 1)
        nat = self.nat(roixy, 1024)
        th_t = np.zeros((n, 8))
        th_t[:, 0] = rng.uniform(-80, 80, n); th_t[:, 1] = rng.uniform(-80, 80, n); th_t[:, 2] = rng.uniform(-300, 300, n)
        th_t[:, 3] = 5000; th_t[:, 4] = 0; th_t[:, 5] = rng.uniform(0, 2 * np.pi, n); th_t[:, 6] = rng.uniform(0.2, 2.9, n)
        th_t[:, 7] = 1.0
        wv, wvz, PM = get_pupil_matrix(self.p, self.st, torch.as_tensor(nat, dtype=torch.float64), 'cpu')
        mu_t, _ = poissonrate(self.p, torch.as_tensor(th_t), PM, wv, wvz, torch.as_tensor(self.p['Cx']),
                              torch.as_tensor(self.p['Cy']))
        th = self.their_to_ours(th_t)
        th[:, 4] = 0; th[:, 3] = 1.0
        with torch.no_grad():
            mu, _ = model_and_jacobian(self.psf, th, zcoef_offset=self.offsets(nat))
        a = mu_t.numpy(); a = a / a.sum((1, 2), keepdims=True)
        b = torch.flip(mu[:, 0], (-1, -2)).cpu().numpy(); b = b / b.sum((1, 2), keepdims=True)
        return (np.abs(a - b).max((1, 2)) / a.max((1, 2))).tolist()


def axis_angle(u, w):
    return np.degrees(np.arccos(np.clip(np.abs((u * w).sum(1)), 0, 1)))


def run_check():
    out = {}
    for refmed in (1.333, 1.0):
        E = Exact(refmed=refmed)
        out[refmed] = E.check()
        log(f'refmed {refmed}: Zernike projection residual {E.proj_residual:.2e} nm; image max rel diff vs their '
            f'port: {np.round(out[refmed], 4).tolist()}')
    return out


def run_dna(zgrid=None, nmax=None):
    """their Octave lambda-DNA table (stk_0001, frames 0-600, crop): the same ROIs, our fit of their model from their
    initial values (one start) or from a z grid; per-spot differences to theirs"""
    t0 = time.time()
    import csv
    rows = list(csv.DictReader(l for l in open(os.path.join(REF, 'out', 'tables', 'octave_dna_crop_f0-600.csv'))
                               if not l.startswith('#')))
    if nmax:
        rows = rows[:nmax]
    f = lambda k: np.array([float(r[k]) for r in rows])
    fr = f('frame').astype(int); rx = f('roix').astype(int); ry = f('roiy').astype(int)
    mov = VR._movie_photons(VR.DNA_MOVIE, frames=(0, int(fr.max()) + 1)).numpy()
    mov = np.where(mov <= 0, 1e-3, mov)
    rois = np.stack([mov[t, r - 1 - 7:r - 1 + 8, c - 1 - 7:c - 1 + 8] for t, c, r in zip(fr, rx, ry)])
    E = Exact(refmed=1.333)
    nat = E.nat(np.stack([rx, ry], 1), 1024)
    zo = E.offsets(nat)
    d = torch.flip(torch.as_tensor(rois, dtype=torch.float32, device=DEV), (-1, -2))
    th_init = np.stack([f('init_dx_roi_nm'), f('init_dy_roi_nm'), f('init_z_nm'), f('init_N'), f('init_bg'),
                        f('init_azim_rad'), f('init_pola_rad'), f('init_g2')], 1)
    th_their = np.stack([f('dx_roi_nm'), f('dy_roi_nm'), f('z_nm'), f('N'), f('bg'), f('azim_rad'), f('pola_rad'),
                         f('g2')], 1)
    t0i = E.their_to_ours(th_init)
    res = {}
    for name, starts in (('their_init', [t0i]), ('zgrid', [t0i] + [t0i.clone().index_fill_(1, torch.tensor([2], device=DEV), z)
                                                                   for z in (-0.6, -0.3, 0.3, 0.6)])):
        if name == 'zgrid' and not zgrid:
            continue
        thl, nll = [], []
        for c0 in range(0, len(d), 800):
            sl_ = slice(c0, c0 + 800)
            b_ = None
            for st in starts:
                r = fit_orientation(E.psf, d[sl_], theta0=st[sl_], iters=50, z_limits=(-1.1, 1.1), g_limits=(0.0, 1.0),
                                    batch=800, zcoef_offset=zo[sl_])
                if b_ is None:
                    b_ = r
                else:
                    b = r['nll'] < b_['nll']
                    b_['theta'][b] = r['theta'][b]; b_['nll'][b] = r['nll'][b]
            thl.append(b_['theta']); nll.append(b_['nll'])
        best = dict(theta=torch.cat(thl), nll=torch.cat(nll))
        th = best['theta']
        g2, Nf = VB.to_their_g2(E.v, th[:, 5:8], th[:, 8])
        mu_o = E.ours_to_their_axis(unit(th[:, 5:8])).cpu().numpy()
        a, p = th_their[:, 5], th_their[:, 6]
        mu_t = np.stack([np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)], 1)
        ang = axis_angle(mu_o, mu_t)
        x_o = -th[:, 1].cpu().numpy() * 65; y_o = -th[:, 0].cpu().numpy() * 65          # their x (row), y (column) nm
        dz = th[:, 2].cpu().numpy() * 1e3 - th_their[:, 2]
        dg = g2.cpu().numpy() - th_their[:, 7]
        dN = (th[:, 3] * Nf).cpu().numpy() / th_their[:, 3]
        ok = f('outlier') == 0
        res[name] = dict(n=int(len(rows)), n_their_ok=int(ok.sum()),
                         axis_angle_median=float(np.median(ang[ok])), axis_angle_p90=float(np.percentile(ang[ok], 90)),
                         frac_axis_lt_2deg=float((ang[ok] < 2).mean()),
                         dxy_median_nm=float(np.median(np.hypot(x_o - th_their[:, 0], y_o - th_their[:, 1])[ok])),
                         dz_median_nm=float(np.median(dz[ok])), dz_mad_nm=float(np.median(np.abs(dz[ok] - np.median(dz[ok])))),
                         dg2_median=float(np.median(dg[ok])), dg2_mad=float(np.median(np.abs(dg[ok] - np.median(dg[ok])))),
                         N_ratio_median=float(np.median(dN[ok])),
                         g2_ours_median=float(np.median(g2.cpu().numpy()[ok])), g2_theirs_median=float(np.median(th_their[ok, 7])),
                         seconds=time.time() - t0)
        log(name, json.dumps(res[name]))
        # per-spot table (their FOV coordinates: x = column, y = row, 1-based as their x_px / y_px; their axis)
        np.savez(os.path.join(OUT, f'exact_dna_{name}.npz'), frame=fr, roix=rx, roiy=ry,
                 x_px=rx - th[:, 0].cpu().numpy(), y_px=ry - th[:, 1].cpu().numpy(), z_nm=th[:, 2].cpu().numpy() * 1e3,
                 N=(th[:, 3] * Nf).cpu().numpy(), bg=th[:, 4].cpu().numpy(), g2=g2.cpu().numpy(), g=th[:, 8].cpu().numpy(),
                 mu_their=mu_o, nll=best['nll'].cpu().numpy(), their_outlier=~ok, their_chi2=f('chi2'),
                 their_x_px=f('x_px'), their_y_px=f('y_px'), their_azim=th_their[:, 5], their_pola=th_their[:, 6],
                 their_g2=th_their[:, 7], their_z_nm=th_their[:, 2], their_N=th_their[:, 3])
    json.dump(res, open(os.path.join(OUT, 'exact_dna.json'), 'w'), indent=1)
    return res


def _their_loglik(d, mu):
    """their likelihood.m: sum n log mu - mu - gammln(max(n, 0.1) + 1) per ROI"""
    n = d.double(); m = mu.double().clamp_min(1e-12)
    return (n * torch.log(m) - m - torch.lgamma(n.clamp_min(0.1) + 1)).flatten(1).sum(1)


def run_pmma(refmed=1.0, zgrid=False, tag=None):
    """their PMMA per-slice ROIs (~/vortex_ref/out/tables/octave_pmma_n149_slices.npz: molecule, slice, ROI centre,
    their initial values), our fit of their model with n_med = refmed, one start (their initial values) or + a
    z grid; exported in their table layout and summarised with their py/pmma_summary.py (40 brightest molecules,
    11 slices around focus)."""
    t0 = time.time()
    T = dict(np.load(os.path.join(REF, 'out', 'tables', 'octave_pmma_n149_slices.npz'), allow_pickle=True))
    S = len(T['mol'])
    stack = np.asarray(__import__('tifffile').imread(os.path.join(VR.DATA, VR.PMMA_V)), np.float32)
    stack = np.maximum((stack - 106.0) * 0.33, 1e-3)
    rx, ry, sl = T['roix'].astype(int), T['roiy'].astype(int), T['slice'].astype(int)
    rois = np.stack([stack[k, r - 1 - 7:r - 1 + 8, c - 1 - 7:c - 1 + 8] for k, c, r in zip(sl, rx, ry)])
    E = Exact(refmed=refmed)
    nat = E.nat(np.stack([rx, ry], 1), 2048)
    zo = E.offsets(nat)
    draw = torch.as_tensor(rois, dtype=torch.float32, device=DEV)
    d = torch.flip(draw, (-1, -2))
    th_init = np.stack([T['init_dx_roi_nm'], T['init_dy_roi_nm'], T['init_z_nm'], T['init_N'], T['init_bg'],
                        T['init_azim_rad'], T['init_pola_rad'], T['init_g2']], 1)
    t0i = E.their_to_ours(th_init)
    starts = [t0i] + ([t0i.clone().index_fill_(1, torch.tensor([2], device=DEV), z) for z in (-0.8, -0.4, 0.4, 0.8)]
                      if zgrid else [])
    from torch_psf.orientation import crlb_orientation
    ths, crs = [], []
    for c0 in range(0, S, 800):                 # chunks (fit_orientation's CRLB is not batched)
        sl_ = slice(c0, c0 + 800)
        best = None
        for st in starts:
            r = fit_orientation(E.psf, d[sl_], theta0=st[sl_], iters=50, z_limits=(-1.1, 1.1), g_limits=(0.0, 1.0),
                                batch=800, zcoef_offset=zo[sl_])
            if best is None:
                best = r
            else:
                b = r['nll'] < best['nll']
                best['theta'][b] = r['theta'][b]; best['nll'][b] = r['nll'][b]
        ths.append(best['theta'])
        crs.append(crlb_orientation(E.psf, best['theta'], zcoef_offset=zo[sl_]))
    th = torch.cat(ths)
    cr = {k: torch.cat([c_[k] for c_ in crs]) for k in ('x', 'y', 'z', 'theta', 'phi', 'g')}
    with torch.no_grad():
        mu = torch.cat([model_and_jacobian(E.psf, th[s:s + 1000], zcoef_offset=zo[s:s + 1000])[0][:, 0]
                        for s in range(0, S, 1000)])
    chi2 = ((d - mu) ** 2 / mu.clamp_min(1e-12)).flatten(1).sum(1) / 225
    ll = _their_loglik(d, mu)
    g = th[:, 8]
    g2, Nf = VB.to_their_g2(E.v, th[:, 5:8], g)
    T2 = VB.pupil_energy(E.v).to(th.dtype)
    nfix = torch.einsum('ni,ij,nj->n', unit(th[:, 5:8]), T2, unit(th[:, 5:8]))
    dg2 = nfix / (g * nfix + 1 - g) ** 2
    mu_t = E.ours_to_their_axis(unit(th[:, 5:8]))
    az = torch.atan2(mu_t[:, 1], mu_t[:, 0]); po = torch.acos(mu_t[:, 2].clamp(-1, 1))
    c = lambda t: t.detach().cpu().numpy().astype(np.float64)
    x_off, y_off = c(th[:, 0]), c(th[:, 1])
    out = dict(T)
    out.update(z_nm=c(th[:, 2]) * 1e3, N=c(th[:, 3] * Nf), bg=c(th[:, 4]), azim_rad=c(az), pola_rad=c(po), g2=c(g2),
               chi2=c(chi2), loglik=c(ll), x_px=rx - x_off, y_px=ry - y_off,
               crlb_azim_rad=np.radians(c(cr['phi'])), crlb_pola_rad=np.radians(c(cr['theta'])),
               crlb_g2=c(cr['g'] * dg2), crlb_dx_roi_nm=c(cr['y']) * 65, crlb_dy_roi_nm=c(cr['x']) * 65,
               crlb_z_nm=c(cr['z']) * 1e3, numiters=np.zeros(S))
    # their outlier rule: |x|, |y| > 3 px from the ROI centre, loglik < mean - 3 sd (per call: here per slice)
    outl = (np.abs(x_off) > 3) | (np.abs(y_off) > 3)
    for k in np.unique(sl):
        m = sl == k
        outl[m] |= out['loglik'][m] < out['loglik'][m].mean() - 3 * out['loglik'][m].std(ddof=1)
    out['outlier'] = outl
    tag = tag or f'n{refmed:g}' + ('_zgrid' if zgrid else '_z0')
    np.savez(os.path.join(OUT, f'exact_pmma_{tag}.npz'), **out)
    sys.path.insert(0, os.path.join(REF, 'py'))
    import pmma_summary as PS
    res = {}
    for name, kw in (('their_outliers', dict(chi2_filter=False)), ('their_outliers+chi2', dict(chi2_filter=True))):
        res[name] = PS.summarize(out, top=40, **kw).get('summary')
    res['seconds'] = time.time() - t0
    log(tag, json.dumps({k: {q: round(v, 3) if isinstance(v, float) else v for q, v in (s_ or {}).items()}
                         if isinstance(s_, dict) else s_ for k, s_ in res.items()}))
    json.dump(res, open(os.path.join(OUT, f'exact_pmma_{tag}.json'), 'w'), indent=1)
    return res


def run_insitu_dna():
    """our own in-situ calibrated model (vortex_real.py plate2: plate centre, Zernikes, apodization; n_med 1.333 for
    water, FocusZ at the coverslip) on the same lambda-DNA ROIs as their Octave table, our default starts + z grid;
    compared per spot with their fits and with our exact-model fits (exact_dna_zgrid.npz)"""
    t0 = time.time()
    import csv
    rows = list(csv.DictReader(l for l in open(os.path.join(REF, 'out', 'tables', 'octave_dna_crop_f0-600.csv'))
                               if not l.startswith('#')))
    f = lambda k: np.array([float(r[k]) for r in rows])
    fr = f('frame').astype(int); rx = f('roix').astype(int); ry = f('roiy').astype(int)
    mov = VR._movie_photons(VR.DNA_MOVIE, frames=(0, int(fr.max()) + 1)).numpy()
    mov = np.where(mov <= 0, 1e-3, mov)
    d = torch.as_tensor(np.stack([mov[t, r - 1 - 7:r - 1 + 8, c - 1 - 7:c - 1 + 8] for t, c, r in zip(fr, rx, ry)]),
                        dtype=torch.float32, device=DEV)
    psf = VR.sm_focusz(n_med=1.333, height=0.0, z_range=(-1.1, 1.1))
    ths, nll = [], []
    for c0 in range(0, len(d), 800):
        r = fit_orientation(psf, d[c0:c0 + 800], iters=40, z_limits=(-1.1, 1.1), batch=800,
                            zgrid=(-0.6, -0.3, 0.0, 0.3, 0.6))
        ths.append(r['theta']); nll.append(r['nll'])
    th = torch.cat(ths)
    g2, Nf = VB.to_their_g2(psf.psf, th[:, 5:8], th[:, 8])
    mu = unit(th[:, 5:8]).cpu().numpy()
    mu_o = mu[:, [1, 0, 2]]                                  # our (column, row, z) -> their (row, column, z)
    a, p = f('azim_rad'), f('pola_rad')
    mu_t = np.stack([np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)], 1)
    ok = f('outlier') == 0
    ang = axis_angle(mu_o, mu_t)
    E = np.load(os.path.join(OUT, 'exact_dna_zgrid.npz'))
    ang_e = axis_angle(mu_o, E['mu_their'])
    x = rx - 1 + th[:, 0].cpu().numpy(); y = ry - 1 + th[:, 1].cpu().numpy()
    dxy = np.hypot(x - (f('x_px') - 1), y - (f('y_px') - 1)) * 65
    dz = th[:, 2].cpu().numpy() * 1e3 - f('z_nm')
    mad = lambda v: float(np.median(np.abs(v - np.median(v))))
    res = dict(n=len(rows), axis_angle_vs_theirs_median=float(np.median(ang[ok])),
               axis_angle_vs_theirs_p90=float(np.percentile(ang[ok], 90)),
               frac_lt_5deg=float((ang[ok] < 5).mean()), axis_angle_vs_exact_median=float(np.median(ang_e[ok])),
               dxy_median_nm=float(np.median(dxy[ok])), dz_median_nm=float(np.median(dz[ok])), dz_mad_nm=mad(dz[ok]),
               g2_insitu_median=float(np.median(g2.cpu().numpy()[ok])), g2_theirs_median=float(np.median(f('g2')[ok])),
               N_ratio_median=float(np.median((th[:, 3] * Nf).cpu().numpy()[ok] / f('N')[ok])),
               nll_insitu_minus_exact_median=float(np.median(torch.cat(nll).cpu().numpy()[ok] - E['nll'][ok])),
               seconds=time.time() - t0)
    log('insitu', json.dumps(res))
    np.savez(os.path.join(OUT, 'insitu_dna.npz'), mu_their=mu_o, g2=g2.cpu().numpy(), x=x, y=y,
             z_nm=th[:, 2].cpu().numpy() * 1e3, nll=torch.cat(nll).cpu().numpy())
    json.dump(res, open(os.path.join(OUT, 'insitu_dna.json'), 'w'), indent=1)


# acceptance test (the PSF refactor must keep these): expected values and tolerances
EXPECTED = dict(
    image_max_rel_diff=(0.0, 0.02),                       # vs their port, 6 random spots, n_med 1.333 and 1.0
    dna_axis_angle_median_deg=(0.145, 0.06),              # lambda-DNA crop, 15065 spots, our fit (z grid) vs theirs
    dna_frac_axis_lt_2deg=(0.915, 0.02),
    dna_dxy_median_nm=(0.235, 0.1),
    dna_dz_median_nm=(0.14, 0.3),
    dna_dg2_median=(0.0003, 0.002),
    dna_N_ratio_median=(0.998, 0.003),
    pmma_z_slope_median=(-0.984, 0.02),                   # PMMA, n_med 1.0, z grid, their summary (top 40)
    pmma_sd_azim_deg_median=(3.48, 0.4),
    pmma_sd_pola_deg_median=(2.84, 0.3),
    pmma_crlb_azim_deg_median=(2.60, 0.1),
    pmma_crlb_pola_deg_median=(2.20, 0.1),
    pmma_g2_mean_median=(0.839, 0.02))


def run_acceptance():
    """one command for the reference numbers: image identity with their port, the lambda-DNA per-spot match on the
    whole crop, the PMMA table (n_med 1.0, z grid); writes results/acceptance.json with PASS / FAIL per value"""
    t0 = time.time()
    got = {}
    chk = run_check()
    got['image_max_rel_diff'] = float(max(max(v) for v in chk.values()))
    d = run_dna(zgrid=True)['zgrid']
    got.update(dna_axis_angle_median_deg=d['axis_angle_median'], dna_frac_axis_lt_2deg=d['frac_axis_lt_2deg'],
               dna_dxy_median_nm=d['dxy_median_nm'], dna_dz_median_nm=d['dz_median_nm'], dna_dg2_median=d['dg2_median'],
               dna_N_ratio_median=d['N_ratio_median'])
    pm = run_pmma(1.0, zgrid=True)['their_outliers']
    got.update(pmma_z_slope_median=pm['z_slope_median'], pmma_sd_azim_deg_median=pm['sd_azim_deg_median'],
               pmma_sd_pola_deg_median=pm['sd_pola_deg_median'], pmma_crlb_azim_deg_median=pm['crlb_azim_deg_median'],
               pmma_crlb_pola_deg_median=pm['crlb_pola_deg_median'], pmma_g2_mean_median=pm['g2_mean_median'])
    rep = {}
    for k, (ref, tol) in EXPECTED.items():
        v = got[k]
        ok = (v <= tol) if k == 'image_max_rel_diff' else abs(v - ref) <= tol
        rep[k] = dict(value=v, expected=ref, tol=tol, ok=bool(ok))
        log(f"{'PASS' if ok else 'FAIL'} {k}: {v:.4f} (expected {ref} +- {tol})")
    rep['all_pass'] = all(r['ok'] for r in rep.values() if isinstance(r, dict))
    rep['seconds'] = time.time() - t0
    log('ACCEPTANCE', 'PASS' if rep['all_pass'] else 'FAIL', f"({rep['seconds']:.0f} s)")
    json.dump(rep, open(os.path.join(OUT, 'acceptance.json'), 'w'), indent=1)


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'acceptance':
        run_acceptance()
    elif stage == 'check':
        run_check()
    elif stage == 'insitu_dna':
        run_insitu_dna()
    elif stage == 'pmma':
        rm = [float(a.split('=')[1]) for a in sys.argv if a.startswith('refmed=')]
        run_pmma(rm[0] if rm else 1.0, zgrid='zgrid' in sys.argv)
    elif stage == 'dna':
        run_dna(zgrid='zgrid' in sys.argv, nmax=int(os.environ.get('NMAX', 0)) or None)
