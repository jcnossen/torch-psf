"""Vortex orientation pipeline on simulations that mimic the experiments of Hulleman et al. (Nat. Commun. 2021):

  beads   bead z-stacks (vortex plate, charge -1, a small plate offset, 64-level spiral, random aberrations, beads in
          oil) -> torch_psf.vortex.calibrate_beads (charge choice, Zernikes, plate centre)
  zstack  ATTO 565 in PMMA through focus (11 slices, 100 nm steps, fixed dipoles, g = 0.9, 5e3-4e4 photons, bg 20)
          -> OrientationEstimator on the movie with the calibrated PSF -> per molecule the spread of phi / theta over
          the slices vs the CRLB, and the z vs focus slope (paper: (2.3, 1.8) deg vs CRLB (1.8, 1.5); slope -0.99)
  dna     lambda-DNA-like BALM: lines on the coverslip in water, intercalator dipoles perpendicular to the line
          (mostly in-plane), g ~ 0.8, ~4600 photons, bg 10 -> OrientationEstimator -> the histograms of
          Delta phi (dipole azimuth - line direction; paper 82 +- 17 deg) and g (paper peak 0.8)

Optics as their lambda-DNA parameters (vecfitcpu_vortex set_parameters_vortex_lambdaDNA.m): NA 1.47, 597.5 nm,
n_imm 1.518 (coverslip index-matched here; theirs 1.523), pixel 65 nm, ROI 15; camera offset 106 ADU, 0.33 e / ADU.

usage: CUDA_VISIBLE_DEVICES=2 python study/orientation/vortex_pipeline_sim.py [beads] [zstack] [dna]
(writes results/pipeline_*.json, results/vortex_calibrated_psf.json and fig_pipeline_*.png)
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, style, COLORS, INK2                                          # noqa: E402
from torch_psf.optics import Optics                                                    # noqa: E402
from torch_psf.camera import Camera                                                    # noqa: E402
from torch_psf.vortex import vortex_psf, calibrate_beads                               # noqa: E402
from torch_psf.orientation import (from_angles, dipole_M, OrientationEstimator, angle_error, canonical,
                                   unit)                                               # noqa: E402

OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OPT = dict(NA=1.47, wavelength_um=0.5975, pixel_um=0.065, n_imm=1.518)
CAM = Camera(gain_e_per_adu=0.33, offset_adu=106.0)
DEPTH = 593.0 / 597.5
TRUE_VORTEX = dict(charge=-1, center=(0.03, -0.02), levels=64, depth=DEPTH)
NOLL = tuple(range(5, 22))


def true_zernikes(seed=3):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(len(NOLL), generator=g) * 0.008
    for j, v in {5: 0.025, 6: -0.015, 7: 0.012, 8: -0.010, 11: 0.020}.items():   # astigmatism, coma, spherical
        z[NOLL.index(j)] = v
    return z


def model(n_med, roisize=15, vortex=TRUE_VORTEX, zern=None, focus=0.0, z_range=(-0.6, 0.6), oversample=2):
    v = vortex_psf(Optics(n_med=n_med, **OPT), roisize=roisize, device=DEV, oversample=oversample, focus_um=focus,
                   z_range=z_range, **vortex)
    if zern is not None:
        v.set_zernike(zern)
    for p in v.parameters():
        p.requires_grad_(False)
    return v


def render_movie(psf_big, frames, emitters, H, W, bg, rng):
    """frames: list of focus values; emitters: list of dict(t (frame or None = all), x, y, z, I, M). Returns the
    movie in ADU (CAM), [T, H, W] float32, and the noise-free photon movie."""
    T, R = len(frames), psf_big.roisize
    h = R // 2
    mov = torch.full((T, H, W), float(bg), device=DEV)
    for e in emitters:
        ts = range(T) if e['t'] is None else [e['t']]
        ix, iy = int(round(e['x'])), int(round(e['y']))
        for t in ts:
            pos = torch.tensor([[e['x'] - ix, e['y'] - iy, e['z']]], device=DEV)
            with torch.no_grad():
                img = psf_big(pos, float(frames[t]), dipM=e['M'][None].to(DEV))[0] * e['I'] * e.get('scale', [1] * T)[t]
            y0, x0 = iy - h, ix - h
            ys, xs = slice(max(y0, 0), min(y0 + R, H)), slice(max(x0, 0), min(x0 + R, W))
            mov[t, ys, xs] += img[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    ph = torch.poisson(mov, generator=rng)
    adu = ph / CAM.gain_e_per_adu + CAM.offset_adu
    return adu.cpu().numpy().astype(np.float32), mov


# ------------------------------------------------------------------ beads
def run_beads():
    t0 = time.time()
    zt = true_zernikes()
    truth = model(1.518, roisize=21, zern=zt.tolist(), oversample=2)
    nk, dz = 21, 0.1
    focus = dz * (torch.arange(nk, dtype=torch.float32) - (nk - 1) / 2)
    rng = torch.Generator(device=DEV).manual_seed(0)
    stacks = []
    gen = np.random.default_rng(0)
    for s in range(2):
        H = W = 96
        em = []
        for b in range(6):
            em.append(dict(t=None, x=float(14 + (b % 3) * 30 + gen.uniform(-1, 1)), y=float(20 + (b // 3) * 50 + gen.uniform(-1, 1)),
                           z=0.09, I=float(gen.uniform(2e4, 4e4)), M=torch.eye(3) / 3))
        big = truth.resized(31)
        adu, _ = render_movie(big, focus.tolist(), em, H, W, 20.0, rng)
        stacks.append(adu)
    start = model(1.518, roisize=21, vortex=dict(charge=-1, center=(0.0, 0.0), levels=64, depth=DEPTH),
                  oversample=2)
    psf, rep = calibrate_beads(stacks, dz, start, camera=CAM, charges=(-1, 1), fit_center=True, rounds=2,
                               iters=300, roisize=21, nbeads=6, emitter_z_um=0.09, drift=False)
    zf = torch.tensor(rep['zernike_waves'])
    err = (zf - zt)
    res = dict(true_zernike_waves=zt.tolist(), fitted_zernike_waves=zf.tolist(), zernike_rms_error=float(err.norm()),
               zernike_true_rms=float(zt.norm()), max_abs_error=float(err.abs().max()),
               true_charge=TRUE_VORTEX['charge'], fitted_charge=rep['charge'], nll_per_px=rep['nll_per_px'],
               true_center=list(TRUE_VORTEX['center']), fitted_center=rep['center'], rounds=rep['rounds'],
               seconds=time.time() - t0)
    print(json.dumps({k: v for k, v in res.items() if 'waves' not in k}, indent=1), flush=True)
    psf.save(os.path.join(OUT, 'vortex_calibrated_psf.json'), provenance=dict(vortex=psf.vortex, sim=True))
    json.dump(res, open(os.path.join(OUT, 'pipeline_beads.json'), 'w'), indent=1)
    return psf


def calibrated(n_med, roisize=15, focus=0.0, z_range=(-0.6, 0.6)):
    """The calibrated Zernikes + plate (from the beads run) on a model with another medium index."""
    r = json.load(open(os.path.join(OUT, 'pipeline_beads.json')))
    vx = dict(charge=r['fitted_charge'], center=tuple(r['fitted_center']), levels=64, depth=DEPTH)
    return model(n_med, roisize=roisize, vortex=vx, zern=r['fitted_zernike_waves'], focus=focus, z_range=z_range)


# ------------------------------------------------------------------ PMMA z-stack
def run_zstack(nmol=20, seed=1):
    t0 = time.time()
    gen = np.random.default_rng(seed)
    zt = true_zernikes()
    truth = model(1.49, roisize=31, zern=zt.tolist())
    focus = [round(0.1 * k, 3) for k in range(-5, 6)]          # stage: focal plane relative to the coverslip
    H = W = 128
    em, mols = [], []
    pts = []
    while len(pts) < nmol:
        p = gen.uniform(12, H - 12, 2)
        if all(np.abs(p - q).max() > 16 for q in pts):
            pts.append(p)
    for i, (x, y) in enumerate(pts):
        th, ph = math.degrees(math.acos(1 - 2 * gen.uniform())), gen.uniform(0, 360)
        mu = canonical(from_angles(th, ph))
        g = 0.9
        I = float(np.exp(gen.uniform(np.log(5e3), np.log(4e4))))
        em.append(dict(t=None, x=float(x), y=float(y), z=0.03, I=I, M=dipole_M(mu, torch.tensor(g))))
        mols.append(dict(x=float(x), y=float(y), mu=mu.tolist(), g=g, I=I))
    rng = torch.Generator(device=DEV).manual_seed(seed)
    adu, _ = render_movie(truth, focus, em, H, W, 20.0, rng)
    psf = calibrated(1.49, z_range=(-0.7, 0.7))
    est = OrientationEstimator(psf, camera=CAM, threshold=8.0, iters=40, z_limits=(-0.7, 0.7), batch=400)
    ds = est.estimate(adu)
    print('zstack: %d spots in %.0f s' % (len(ds), est.report['seconds']), flush=True)
    pos = ds['pos'].numpy()
    rows = []
    for i, m in enumerate(mols):
        sel = np.nonzero((np.abs(pos[:, 0] - m['x']) < 3) & (np.abs(pos[:, 1] - m['y']) < 3))[0]
        if len(sel) < 4:
            continue
        mu_t = torch.tensor(m['mu'])
        mu_e = ds['mu'][sel]
        mu_e = mu_e * torch.sign((mu_e * mu_t).sum(1, keepdim=True))
        err = angle_error(mu_e, mu_t[None]).numpy()
        fr = ds['frame'][sel].numpy()
        f = np.array([focus[k] for k in fr])
        z = ds['pos'][sel, 2].numpy() * psf.pixel_um
        th_t, ph_t = [float(a) for a in __import__('torch_psf.orientation', fromlist=['angles']).angles(mu_t)]
        inner = np.abs(f) <= 0.3 + 1e-6
        slope = float(np.polyfit(f, z, 1)[0])
        rows.append(dict(mol=i, n=int(len(sel)), theta_true=th_t, phi_true=ph_t, I=m['I'],
                         theta_est=ds['theta_deg'][sel].numpy().tolist(), phi_est=ds['phi_deg'][sel].numpy().tolist(),
                         g_est=ds['g'][sel].numpy().tolist(), axis_err_deg=err.tolist(), focus=f.tolist(),
                         z_um=z.tolist(), z_slope=slope,
                         crlb_theta=ds['crlb_theta_deg'][sel].numpy().tolist(),
                         crlb_inplane=ds['crlb_inplane_deg'][sel].numpy().tolist(),
                         crlb_phi=ds['crlb_phi_deg'][sel].numpy().tolist(), inner=inner.tolist(),
                         chi2=ds['chi2'][sel].numpy().tolist()))
    summ = summarize_zstack(rows)
    summ['seconds'] = time.time() - t0
    print(json.dumps(summ, indent=1), flush=True)
    json.dump(dict(summary=summ, molecules=rows), open(os.path.join(OUT, 'pipeline_zstack.json'), 'w'), indent=1)
    return rows


def _circ_sd_deg(a, period=360.0):
    r = np.exp(2j * np.pi * np.asarray(a) / period)
    m = np.abs(r.mean())
    return float(np.sqrt(-2 * np.log(max(m, 1e-12))) * period / (2 * np.pi))


def summarize_zstack(rows):
    """Per molecule (|focus| <= 300 nm slices, the Vortex PSF's working range, and all slices): the sd of the
    estimated theta and azimuth over the slices, their mean CRLB, the axis error of the mean, and the z slope."""
    out = dict(n_molecules=len(rows))
    for key, use in (('inner', lambda r: np.array(r['inner'])), ('all', lambda r: np.ones(len(r['inner']), bool))):
        sd_t, sd_p, cr_t, cr_p, bias_t, bias_p, ax, nout = [], [], [], [], [], [], [], 0
        for r in rows:
            u = use(r)
            ok = u & (np.array(r['axis_err_deg']) < 30)
            nout += int((u & ~ok).sum())
            if ok.sum() < 3:
                continue
            t = np.array(r['theta_est'])[ok]
            p = np.array(r['phi_est'])[ok]
            sd_t.append(float(t.std())); sd_p.append(_circ_sd_deg(p))
            cr_t.append(float(np.mean(np.array(r['crlb_theta'])[ok])))
            cr_p.append(float(np.mean(np.array(r['crlb_phi'])[ok])))
            bias_t.append(float(t.mean() - r['theta_true']))
            dp = (np.angle(np.exp(1j * np.deg2rad(p - r['phi_true'])).mean(), deg=True))
            bias_p.append(float(dp))
            ax.append(float(np.median(np.array(r['axis_err_deg'])[ok])))
        out[key] = dict(sd_theta_deg=float(np.mean(sd_t)), crlb_theta_deg=float(np.mean(cr_t)),
                        sd_phi_deg=float(np.mean(sd_p)), crlb_phi_deg=float(np.mean(cr_p)),
                        mean_bias_theta_deg=float(np.mean(bias_t)), rms_bias_theta_deg=float(np.sqrt(np.mean(np.square(bias_t)))),
                        mean_bias_phi_deg=float(np.mean(bias_p)), rms_bias_phi_deg=float(np.sqrt(np.mean(np.square(bias_p)))),
                        median_axis_err_deg=float(np.median(ax)), outliers=nout,
                        n_fits=int(sum(use(r).sum() for r in rows)))
    sl = np.array([r['z_slope'] for r in rows])
    out['z_slope_mean'], out['z_slope_sd'] = float(sl.mean()), float(sl.std())
    dzs = []
    for r in rows:
        f, z = np.array(r['focus']), np.array(r['z_um'])
        fu = np.unique(f)
        zu = np.array([z[f == v].mean() for v in fu])
        dzs += list(np.diff(zu) / np.diff(fu) * 0.1 * 1e3)
    out['dz_per_100nm_step_nm'] = [float(np.mean(dzs)), float(np.std(dzs))]
    return out


# ------------------------------------------------------------------ lambda-DNA
def run_dna(nlines=10, T=400, per_frame=8, seed=2):
    t0 = time.time()
    gen = np.random.default_rng(seed)
    zt = true_zernikes()
    truth = model(1.333, roisize=31, zern=zt.tolist(), z_range=(-0.1, 0.5))
    H = W = 160
    lines = []
    for _ in range(nlines):
        a = gen.uniform(0, np.pi)
        c = gen.uniform(30, H - 30, 2)
        lines.append((a, c))
    em, truth_rows = [], []
    for t in range(T):
        for _ in range(gen.poisson(per_frame)):
            a, c = lines[gen.integers(nlines)]
            s = gen.uniform(-60, 60)
            x, y = c[0] + s * np.cos(a), c[1] + s * np.sin(a)
            if not (10 < x < W - 10 and 10 < y < H - 10):
                continue
            psi = np.deg2rad(gen.normal(0, 25))              # tilt out of the plane, around the DNA axis
            dperp = np.deg2rad(gen.normal(0, 8))              # in-plane deviation from perpendicular
            b = a + np.pi / 2 + dperp
            mu = torch.tensor([np.cos(psi) * np.cos(b), np.cos(psi) * np.sin(b), np.sin(psi)], dtype=torch.float32)
            g = float(np.clip(gen.beta(16, 4), 0.3, 0.98))    # mean 0.8
            I = float(4600 * np.exp(gen.normal(0, 0.35)))
            z = float(gen.uniform(0.0, 0.03))
            em.append(dict(t=t, x=float(x), y=float(y), z=z, I=I, M=dipole_M(mu, torch.tensor(g))))
            truth_rows.append(dict(t=t, x=float(x), y=float(y), z=z, line_angle=float(np.degrees(a)), mu=mu.tolist(),
                                   g=g, I=I))
    rng = torch.Generator(device=DEV).manual_seed(seed)
    adu, _ = render_movie(truth, [0.0] * T, em, H, W, 10.0, rng)
    print(f'dna: {len(em)} emitters rendered ({time.time() - t0:.0f} s)', flush=True)
    psf = calibrated(1.333, z_range=(-0.1, 0.5))
    est = OrientationEstimator(psf, camera=CAM, threshold=6.0, iters=40, z_limits=(-0.1, 0.5), batch=500,
                               isolation=10)
    ds = est.estimate(adu)
    print('dna: %d spots fitted in %.0f s' % (len(ds), est.report['seconds']), flush=True)
    # match to the truth (same frame, within 2 px)
    tr_t = np.array([r['t'] for r in truth_rows]); tr_xy = np.array([[r['x'], r['y']] for r in truth_rows])
    pos = ds['pos'].numpy(); fr = ds['frame'].numpy()
    match = -np.ones(len(ds), int)
    for i in range(len(ds)):
        c = np.nonzero(tr_t == fr[i])[0]
        if len(c):
            d = np.abs(tr_xy[c] - pos[i, :2]).max(1)
            if d.min() < 2:
                match[i] = c[np.argmin(d)]
    ok = match >= 0
    chi2 = ds['chi2'].numpy()
    keep = ok & (chi2 > 0.75) & (chi2 < 3)                    # their chi2 filter
    la = np.array([truth_rows[j]['line_angle'] for j in match[keep]])
    dphi = (ds['phi_deg'].numpy()[keep] - la) % 180.0         # azimuth relative to the DNA axis (axis: mod 180)
    th_est = ds['theta_deg'].numpy()[keep]
    g_est = ds['g'].numpy()[keep]
    mu_t = torch.tensor([truth_rows[j]['mu'] for j in match[keep]])
    ax = angle_error(ds['mu'][keep], mu_t).numpy()
    g_t = np.array([truth_rows[j]['g'] for j in match[keep]])
    xy_err = (pos[keep, :2] - tr_xy[match[keep]]) * 65.0
    z_err = pos[keep, 2] * 65.0 - np.array([truth_rows[j]['z'] for j in match[keep]]) * 1e3
    # true Delta phi of the same spots
    mt = canonical(mu_t)
    ph_t = np.degrees(np.arctan2(mt[:, 1].numpy(), mt[:, 0].numpy()))
    dphi_t = (ph_t - la) % 180.0
    hist_g, _ = np.histogram(g_est, bins=20, range=(0, 1))
    summ = dict(n_emitters=len(em), n_fits=len(ds), n_matched=int(ok.sum()), n_chi2_kept=int(keep.sum()),
                dphi_median=float(np.median(dphi)), dphi_mad=float(np.median(np.abs(dphi - np.median(dphi)))),
                dphi_true_median=float(np.median(dphi_t)),
                dphi_true_mad=float(np.median(np.abs(dphi_t - np.median(dphi_t)))),
                g_peak=float((np.argmax(hist_g) + 0.5) / 20), g_median=float(np.median(g_est)),
                g_true_median=float(np.median(g_t)), g_bias_median=float(np.median(g_est - g_t)),
                axis_err_median_deg=float(np.median(ax)), axis_err_p90_deg=float(np.percentile(ax, 90)),
                xy_err_rms_nm=float(np.sqrt((xy_err ** 2).mean())), z_err_median_nm=float(np.median(z_err)),
                z_err_sd_nm=float(z_err.std()),
                crlb_inplane_median_deg=float(np.median(ds['crlb_inplane_deg'].numpy()[keep])),
                crlb_theta_median_deg=float(np.median(ds['crlb_theta_deg'].numpy()[keep])),
                crlb_g_median=float(np.median(ds['crlb_g'].numpy()[keep])),
                crlb_xy_median_nm=float(np.median(ds['crlb'][keep, :2].numpy()) * 65),
                crlb_z_median_nm=float(np.median(ds['crlb'][keep, 2].numpy()) * 65),
                photons_median=float(np.median(ds['photons'].numpy()[keep])),
                chi2_median=float(np.median(chi2[ok])), seconds=time.time() - t0)
    print(json.dumps(summ, indent=1), flush=True)
    json.dump(dict(summary=summ, dphi=dphi.tolist(), dphi_true=dphi_t.tolist(), g=g_est.tolist(),
                   g_true=g_t.tolist(), theta=th_est.tolist()),
              open(os.path.join(OUT, 'pipeline_dna.json'), 'w'), indent=1)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    pz = os.path.join(OUT, 'pipeline_zstack.json'); pd = os.path.join(OUT, 'pipeline_dna.json')
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
    if os.path.exists(pz):
        rows = json.load(open(pz))['molecules']
        for r in rows:
            o = np.argsort(r['focus'])
            ax[0].plot(np.array(r['focus'])[o] * 1e3, np.array(r['z_um'])[o] * 1e3, color=COLORS[0], lw=0.8, alpha=0.6)
            ax[1].plot(np.array(r['focus'])[o] * 1e3, (np.array(r['theta_est']) - r['theta_true'])[o], color=COLORS[1],
                       lw=0.8, alpha=0.6)
        ax[0].set_xlabel('focus (nm)'); ax[0].set_ylabel('fitted z (nm)')
        ax[0].set_title('PMMA z-stack: z vs focus', fontsize=8.5)
        ax[1].set_xlabel('focus (nm)'); ax[1].set_ylabel('theta error (deg)'); ax[1].set_ylim(-15, 15)
        ax[1].set_title('theta error per slice', fontsize=8.5)
    if os.path.exists(pd):
        r = json.load(open(pd))
        ax[2].hist(r['dphi'], bins=36, range=(0, 180), color=COLORS[2], alpha=0.85, label='estimated')
        ax[2].hist(r['dphi_true'], bins=36, range=(0, 180), histtype='step', color=INK2, label='truth')
        ax[2].set_xlabel('Delta phi (deg)'); ax[2].legend(fontsize=7)
        ax[2].set_title('lambda-DNA: azimuth - DNA axis', fontsize=8.5)
        ax[3].hist(r['g'], bins=20, range=(0, 1), color=COLORS[3], alpha=0.85, label='estimated')
        ax[3].hist(r['g_true'], bins=20, range=(0, 1), histtype='step', color=INK2, label='truth')
        ax[3].set_xlabel('g (g2)'); ax[3].legend(fontsize=7)
        ax[3].set_title('lambda-DNA: g', fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_pipeline_sim.png')); plt.close(fig)


if __name__ == '__main__':
    what = sys.argv[1:] or ['beads', 'zstack', 'dna']
    if 'beads' in what:
        run_beads()
    if 'zstack' in what:
        run_zstack()
    if 'dna' in what:
        run_dna()
    figures()
