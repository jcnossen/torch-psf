"""Reproduction (simulation part) of Hulleman, Thorsen, Kim, Dekker, Stallinga, Rieger, "Simultaneous orientation and
3D localization microscopy with a Vortex point spread function", Nat. Commun. 12, 5934 (2021), with torch_psf.

Their simulation setup (Methods): NA 1.45, wavelength 597.5 nm, n_med 1.33, coverslip 1.523, immersion 1.518 (our
model: coverslip index-matched to the immersion -> 1.518 throughout), pixel 65 nm, ROI 15 x 15, 4000 signal photons
(captured by the NA, over the whole image plane), 10 background photons / px, Poisson noise. Vortex PSF: a charge-1
spiral phase plate in the emission pupil (orientation.vortex_phase). Emitter at the coverslip, in focus, unless stated.

  (i)   Fig. 1b/c: standard and Vortex PSF images, theta = 10 / 45 / 80 deg, phi = 45 deg, g2 = 1, in focus
  (ii)  Fig. 1e/f: sqrt(CRLB) of azimuth / polar angle vs theta (0..180) at phi = 45 deg, g2 = 1
  (iii) text numbers: dipoles uniform on the sphere, g2 = 0.75: sigma_xy, z, phi, theta, g2; sigma_z at g2 = 1 / 0 and
        sigma_xy Vortex vs standard for free rotators
  (iv)  MLE fits (torch_psf.orientation.fit_orientation) of simulated Vortex spots vs the CRLB

g2 of the paper = our g. Photon convention: N = photons over the whole image plane for that orientation; our model's
intensity I is relative to the isotropic emitter at z = 0, so I = N / S(M, z) with S the total of the PSF over a large
ROI (orientation and height dependent). The angle CRLBs do not depend on how the nuisance intensity is parametrised.

usage: CUDA_VISIBLE_DEVICES=2 python study/orientation/vortex_repro.py   (writes results/vortex_*.json, fig_vortex_*.png)
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, style, COLORS, INK2                               # noqa: E402
from torch_psf import VectorialPSF                                         # noqa: E402
from torch_psf.orientation import (from_angles, basis_images, crlb_orientation, dipole_M, M_to_m6, vortex_phase,
                                   model_and_jacobian, fit_orientation, tangent_basis, canonical, unit,
                                   angles)                                 # noqa: E402

OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
OPT = dict(pixel_um=0.065, NA=1.45, n_imm=1.518, n_med=1.33, wavelength_um=0.5975, npupil=64, oversample=3)
NPH, BG = 4000.0, 10.0
N_EFF = 1700.0    # photon count at which our curves match the level of their Fig. 1e/f (see the note)


def make(vortex, R=15, focus=0.0, charge=1, dev='cpu', oversample=3):
    v = VectorialPSF(R, device=dev, focus_um=focus, z_range=(-0.3, 0.6), **dict(OPT, oversample=oversample))
    for p in v.parameters():
        p.requires_grad_(False)
    if vortex:
        vortex_phase(v, charge=charge)
    return v


def totals(z, dev, R=121):
    """Total over a large ROI (121 px = 7.9 um) of the six basis images at emitter height z [n] -> [n, 6]: the
    photons over the image plane of M are I * (m . totals)."""
    big = make(False, R=R, dev=dev, oversample=1)
    pos = torch.zeros(len(z), 3, device=dev); pos[:, 2] = z
    return torch.cat([basis_images(big, pos[s:s + 64]).sum((-1, -2)) for s in range(0, len(z), 64)])


def theta_vec(mu, g, z, tot, x=None, y=None, nph=NPH):
    """theta [n, 9] with I = nph / S(M, z) (tot: [n, 6] totals at z)."""
    n = len(mu)
    M = dipole_M(mu, g)
    S = (M_to_m6(M) * tot).sum(1)
    th = torch.zeros(n, 9, device=mu.device)
    if x is not None:
        th[:, 0], th[:, 1] = x, y
    th[:, 2] = z; th[:, 3] = nph / S; th[:, 4] = BG; th[:, 5:8] = mu; th[:, 8] = g
    return th


def crlb_all(psf, th, chunk=256):
    out = {}
    for s in range(0, len(th), chunk):
        c = crlb_orientation(psf, th[s:s + chunk])
        for k, v in c.items():
            out.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def fig1_images(dev, plt, charge=-1, origin='upper'):
    """Fig. 1b/c. Their images match charge -1 in our pupil coordinates shown with row 0 at the top (MATLAB imagesc);
    equivalently charge +1 with the image y axis mirrored (a handedness convention; CRLBs are identical)."""
    std, vor = make(False, dev=dev), make(True, charge=charge, dev=dev)
    mu = from_angles(torch.tensor([10.0, 45.0, 80.0]), torch.tensor([45.0] * 3)).to(dev)
    g = torch.ones(3, device=dev)
    pos = torch.zeros(3, 3, device=dev)
    tot = totals(pos[:, 2], dev)
    th = theta_vec(mu, g, pos[:, 2], tot)
    with torch.no_grad():
        a = std(pos, 0.0, dipM=dipole_M(mu, g)) * th[:, 3, None, None]
        b = vor(pos, 0.0, dipM=dipole_M(mu, g)) * th[:, 3, None, None]
    ref = float(a[2].max())
    fig, ax = plt.subplots(2, 3, figsize=(6.6, 4.6))
    for i, t in enumerate((10, 45, 80)):
        ax[0, i].imshow((a[i] / ref).cpu(), origin=origin, cmap='hot', vmin=0, vmax=1)
        ax[1, i].imshow((b[i] / ref).cpu(), origin=origin, cmap='hot', vmin=0, vmax=0.35)
        ax[0, i].set_title(f'standard, theta = {t}', fontsize=8)
        ax[1, i].set_title(f'Vortex, theta = {t}', fontsize=8)
        for r in (0, 1):
            ax[r, i].set_xticks([]); ax[r, i].set_yticks([]); ax[r, i].grid(False)
    fig.suptitle(f'phi = 45 deg, g2 = 1, in focus; row 0 {"top" if origin == "upper" else "bottom"}; standard 0..1, Vortex 0..0.35 of the standard '
                 f'theta = 80 peak (charge {charge:+d})', fontsize=7.5, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_vortex_images.png'))
    plt.close(fig)
    return dict(vortex_peak_over_standard_peak=[float(b[i].max() / ref) for i in range(3)],
                standard_peak=[float(a[i].max() / ref) for i in range(3)],
                roi_fraction_standard=[float(a[i].sum() / NPH) for i in range(3)],
                roi_fraction_vortex=[float(b[i].sum() / NPH) for i in range(3)])


# approximate values read off the paper's Fig. 1e/f (theta deg, sqrt(CRLB) deg)
PAPER = dict(standard_phi=[(35, 15), (90, 4.7), (145, 15)],
             vortex_phi=[(10, 15), (50, 4.3), (90, 6.3), (130, 4.3), (170, 15)],
             standard_theta=[(5, 15), (52, 2.4), (90, 8.0), (128, 2.4), (175, 15)],
             vortex_theta=[(0, 2.6), (35, 3.5), (90, 2.2), (145, 3.5), (180, 2.6)])


def fig1_crlb(dev, plt):
    thetas = torch.arange(1.0, 180.0, 2.0)
    n = len(thetas)
    mu = from_angles(thetas, torch.full((n,), 45.0)).to(dev)
    z = torch.zeros(n, device=dev)
    tot = totals(z, dev)
    res = {}
    for nph in (NPH, N_EFF):
        th = theta_vec(mu, torch.ones(n, device=dev), z, tot, nph=nph)
        for name, vort in (('standard', False), ('vortex', True)):
            c = crlb_all(make(vort, dev=dev), th)
            res[f'{name}_N{nph:.0f}'] = dict(phi=c['phi'].tolist(), theta=c['theta'].tolist(),
                                             x_nm=(c['x'] * 65).tolist(), z_nm=(c['z'] * 1e3).tolist())
    style(plt)
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for name, col in (('standard', COLORS[0]), ('vortex', COLORS[1])):
        for nph, lst in ((NPH, '-'), (N_EFF, '--')):
            r = res[f'{name}_N{nph:.0f}']
            ax[0].plot(thetas, r['phi'], lst, color=col, label=f'{name}, ours, N = {nph:.0f}')
            ax[1].plot(thetas, r['theta'], lst, color=col, label=f'{name}, ours, N = {nph:.0f}')
        for a, k in ((ax[0], 'phi'), (ax[1], 'theta')):
            pts = np.array(PAPER[f'{name}_{k}'])
            a.plot(pts[:, 0], pts[:, 1], 'o', ms=5, mfc='none', mec=col, mew=1.2, label=f'{name}, paper (read off)')
    ax[0].set_ylabel('azimuth sqrt(CRLB) (deg)'); ax[1].set_ylabel('polar sqrt(CRLB) (deg)')
    for a in ax:
        a.set_ylim(0, 15); a.set_xlim(0, 180); a.set_xlabel('polar angle theta (deg)'); a.set_xticks([0, 45, 90, 135, 180])
    ax[1].legend(fontsize=6.5, loc='upper center')
    fig.suptitle('Fig. 1e/f reproduction: bg 10 / px, g2 = 1, phi = 45, in focus at the coverslip; '
                 'N = photons over the whole image plane', fontsize=8.5, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_vortex_crlb.png')); plt.close(fig)
    res['theta_deg'] = thetas.tolist()
    return res


# geometries: emitter height (um), focal plane ('best': maximum peak of the isotropic standard PSF; a number: that
# focal-plane position in the sample coordinate of VectorialPSF)
GEOM = {'coverslip': (0.0, 0.0), 'depth500_nominal': (0.5, 0.5), 'depth500_best': (0.5, 'best')}


def best_focus(depth, dev):
    """Focal-plane position (sample coordinate) maximising the peak of the isotropic standard PSF of an emitter at
    `depth` (index mismatch n_med 1.33 / n_imm 1.518 shifts and blurs the focus)."""
    if depth == 0:
        return 0.0
    psf = make(False, dev=dev)
    fs = torch.linspace(depth * 0.8, depth * 2.0, 121)
    pk = [float(psf(torch.tensor([[0.0, 0.0, depth]], device=dev), float(f)).max()) for f in fs]
    return float(fs[int(np.argmax(pk))])


def fisher_both_z(psf, th, focus, g_free):
    """sd of (x, y, z, I, bg[, a, b, g]) with z = emitter height and with z = -(focal-plane position) (the emitter
    displacement relative to the focal plane as a stage / focus scan sees it). focus [n]."""
    m, J = model_and_jacobian(psf, th, focus=focus)
    e = 1e-3
    Jf = -(model_and_jacobian(psf, th, focus=focus + e)[0] - model_and_jacobian(psf, th, focus=focus - e)[0]) / (2 * e)
    e_th, e_ph = tangent_basis(unit(th[:, 5:8]))
    Ja = torch.einsum('nc...,nc->n...', J[:, 5:8], e_th)
    Jb = torch.einsum('nc...,nc->n...', J[:, 5:8], e_ph)
    out = {}
    for zname, Jz in (('emitter', J[:, 2]), ('focus', Jf)):
        cols = [J[:, 0], J[:, 1], Jz, J[:, 3], J[:, 4]] + ([] if g_free else [Ja, Jb, J[:, 8]])
        Jt = torch.stack(cols, 1).flatten(2)
        F = torch.einsum('nap,nbp->nab', Jt / m.flatten(1)[:, None], Jt).double()
        out[zname] = torch.sqrt(torch.diagonal(torch.linalg.inv(F), dim1=1, dim2=2)).float()
    return out


def uniform_numbers(dev, n=1000, seed=0, chunk=200):
    """CRLBs over n instances: x, y uniform in +-1 px, focal plane uniform within +-65 nm of the emitter, orientations
    uniform on the sphere (phi = pi u, theta = arccos(1 - 2 u)); geometries GEOM (emitter at the coverslip, with SAF;
    emitter 500 nm deep, no SAF, index-mismatch aberration, focal plane at 500 nm or at the best focus); g2 = 1, 0.75 and 0 (free rotator: orientation not
    estimated). sigma_xy = sqrt((sd_x^2 + sd_y^2) / 2); means over the instances, sigma_phi also as the median and the
    mean over 20 < theta < 160 (it diverges at the poles)."""
    gen = torch.Generator().manual_seed(seed)
    u = lambda: torch.rand(n, generator=gen)
    x, y, df = 2 * u() - 1, 2 * u() - 1, 0.065 * (2 * u() - 1)
    phi = 180 * u(); theta = torch.rad2deg(torch.acos(1 - 2 * u()))
    mu = from_angles(theta, phi).to(dev)
    mid = (theta > 20) & (theta < 160)
    out = {}
    for gname, (depth, fg) in GEOM.items():
        f0 = best_focus(depth, dev) if fg == 'best' else fg
        ze = torch.full((n,), depth, device=dev)
        tot = totals(ze[:1], dev).expand(n, 6)
        foc = (f0 + df).to(dev)
        for name, vort in (('standard', False), ('vortex', True)):
            psf = make(vort, dev=dev)
            for g in (1.0, 0.75, 0.0):
                gg = torch.full((n,), max(g, 1e-3), device=dev)
                th = theta_vec(mu, gg, ze, tot, x.to(dev), y.to(dev))
                parts = [fisher_both_z(psf, th[s:s + chunk], foc[s:s + chunk], g == 0.0) for s in range(0, n, chunk)]
                sd = {k: torch.cat([p_[k] for p_ in parts]).cpu() for k in ('emitter', 'focus')}
                e = sd['emitter']
                r = dict(focus_um=f0, xy_nm=float(torch.sqrt((e[:, 0] ** 2 + e[:, 1] ** 2) / 2).mean() * 65),
                         z_emitter_nm=float(e[:, 2].mean() * 1e3), z_focus_nm=float(sd['focus'][:, 2].mean() * 1e3))
                if g > 0:
                    th_r = torch.deg2rad(theta)
                    sphi = torch.rad2deg(e[:, 6] / torch.sin(th_r).clamp_min(1e-6))
                    r.update(phi_median=float(sphi.median()), phi_mean_20_160=float(sphi[mid].mean()),
                             theta_mean=float(torch.rad2deg(e[:, 5]).mean()), theta_median=float(torch.rad2deg(e[:, 5]).median()),
                             g_mean=float(e[:, 7].mean()))
                out[f'{gname}_{name}_g{g}'] = r
                print(gname, name, g, {k: round(v, 3) for k, v in r.items()}, flush=True)
    return out


def fit_check(dev, n=300, seed=1):
    """MLE fits of simulated Vortex spots (fixed orientations, g2 = 0.75 and 1) vs the CRLB. Errors of the axis in the
    tangent coordinates of the true axis (a: polar, b: in-plane), so pole / wrap issues do not enter."""
    torch.manual_seed(seed)
    psf = make(True, focus=0.0, dev=dev)
    rows = []
    for (t, p, g) in ((30.0, 45.0, 0.75), (60.0, 120.0, 0.75), (85.0, 20.0, 0.75), (60.0, 120.0, 1.0),
                      (15.0, 80.0, 1.0)):
        mu = from_angles(torch.full((n,), t), torch.full((n,), p)).to(dev)
        z = torch.full((n,), 0.03, device=dev)
        x = (torch.rand(n, device=dev) - 0.5); y = (torch.rand(n, device=dev) - 0.5)
        th = theta_vec(mu, torch.full((n,), g, device=dev), z, totals(z[:1], dev).expand(n, 6), x, y)
        with torch.no_grad():
            m, _ = model_and_jacobian(psf, th)
        d = torch.poisson(m[:, 0])
        t0 = time.time()
        r = fit_orientation(psf, d, z_limits=(0.0, 0.3), iters=50)
        el = time.time() - t0
        est = r['theta']
        mt = canonical(mu)
        me = est[:, 5:8] * torch.sign((est[:, 5:8] * mt).sum(1, keepdim=True))
        e_th, e_ph = tangent_basis(mt)
        a = torch.rad2deg((me * e_th).sum(1)); b = torch.rad2deg((me * e_ph).sum(1))
        cr = crlb_orientation(psf, th[:1])
        mirror = float(((me * mt).sum(1) < math.cos(math.radians(30))).float().mean())
        ok = (me * mt).sum(1) > math.cos(math.radians(30))
        row = dict(theta=t, phi=p, g=g, n=n, fit_s=el, frac_outliers_gt30deg=mirror,
                   bias_a_deg=float(a[ok].mean()), sd_a_deg=float(a[ok].std()), crlb_theta_deg=float(cr['theta'][0]),
                   bias_b_deg=float(b[ok].mean()), sd_b_deg=float(b[ok].std()), crlb_inplane_deg=float(cr['inplane'][0]),
                   bias_g=float((est[ok, 8] - g).mean()), sd_g=float(est[ok, 8].std()), crlb_g=float(cr['g'][0]),
                   bias_x_nm=float((est[ok, 0] - x[ok]).mean() * 65), sd_x_nm=float((est[ok, 0] - x[ok]).std() * 65),
                   crlb_x_nm=float(cr['x'][0] * 65),
                   bias_z_nm=float((est[ok, 2] - 0.03).mean() * 1e3), sd_z_nm=float(est[ok, 2].std() * 1e3),
                   crlb_z_nm=float(cr['z'][0] * 1e3))
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    what = sys.argv[1:] or ['images', 'crlb', 'uniform', 'fit']
    path = os.path.join(OUT, 'vortex_repro.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    if 'images' in what:
        res['images'] = fig1_images(dev, plt)
        res.pop('images_neg', None)
    if 'crlb' in what:
        res['fig1ef'] = fig1_crlb(dev, plt)
    if 'uniform' in what:
        res['uniform'] = uniform_numbers(dev)
    if 'fit' in what:
        res['fit'] = fit_check(dev)
    json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
