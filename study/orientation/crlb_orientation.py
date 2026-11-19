"""Orientation CRLB / fittability study for the SIMCODE geometry (docs/notes/orientation_psf.md).

Fisher matrices (Poisson, + bg) of (x, y, z, I, bg, a, b, g), (a, b) tangent coordinates of the dipole axis mu, for
  models: aberration-free at the two in-situ foci, the two in-situ PSFs (m1809: Zernikes, focus 308 nm; m25: Zernikes,
          focus 101 nm, general blur);
  cases:  single frame; track of 6 frames (3 per pattern angle) without excitation term (free intensity) and with the
          excitation factor 3 tr(M K_a) (K from the in-situ per-angle ensemble tensors, and pure s-polarised);
  sweep:  theta 0..90, phi 0..315 (8), z 0 / 50 / 100 / 200 nm, g 1 .. 0.1, photons 500 / 1000 / 2000 per frame,
          bg 10 (and 30) photons / px;
plus the bias of the isotropic-PSF fit of fixed-dipole spots (noise-free expected images), and the angle-brightness
contrast between the two pattern angles (single dipoles, ensembles, its CRLB per track).

usage: CUDA_VISIBLE_DEVICES=2 python study/orientation/crlb_orientation.py   (writes results/ and figures here)
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, models, ensemble_to_K, style, COLORS, INK2       # noqa: E402
from torch_psf.orientation import (from_angles, basis_images, crlb_orientation, dipole_M, excitation_factor,
                                   M_to_m6, unit)                         # noqa: E402
from torch_psf.fit import localize, fisher as fisher5                     # noqa: E402

OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
THETA = np.arange(0, 91, 10.0)
PHI = np.arange(0, 360, 45.0)
G = np.array([1.0, 0.7, 0.4, 0.3, 0.2, 0.1])
Z = np.array([0.0, 0.05, 0.10, 0.20])
PHOT = [500.0, 1000.0, 2000.0]
BGS = [10.0, 30.0]
T = 6
KEYS = ('theta', 'phi', 'inplane', 'g', 'z', 'x', 'y')
CORR = dict(c_az=(5, 2), c_ax=(5, 0), c_bx=(6, 0), c_by=(6, 1), c_gz=(7, 2), c_gI=(7, 3))


def grid_theta(z, phot, bg, dev):
    """theta [n, 9] over (THETA, PHI, G) at (0, 0, z)."""
    tt, pp, gg = np.meshgrid(THETA, PHI, G, indexing='ij')
    n = tt.size
    mu = from_angles(torch.tensor(tt.ravel()), torch.tensor(pp.ravel())).to(dev)
    th = torch.zeros(n, 9, device=dev)
    th[:, 2] = z; th[:, 3] = phot; th[:, 4] = bg
    th[:, 5:8] = mu; th[:, 8] = torch.tensor(gg.ravel(), dtype=torch.float32, device=dev)
    return th, tt.shape


def crlb_sweep(dev):
    ms = models(dev)
    exc = {n: ensemble_to_K(n) for n in ('m1809', 'm25')}
    Kmain = exc['m1809']['K'].to(dev)        # SIMCODE-consistent excitation (eps fitted on 1809; see the note)
    Ks = exc['m1809']['K_s'].to(dev)         # pure s-polarised
    Kseq = lambda K: torch.stack([K[0]] * 3 + [K[1]] * 3)
    cases = {'single': (1, None), 'track6': (T, None), 'track6_K': (T, Kseq(Kmain)), 'track6_Ks': (T, Kseq(Ks))}
    res = {}
    for mname, (psf, desc) in ms.items():
        for zi, z in enumerate(Z):
            pos = torch.tensor([[0.0, 0.0, float(z)]], device=dev)
            Bb = basis_images(psf, pos, grad=True)
            for phot in PHOT:
                for bg in BGS:
                    if bg != 10.0 and phot != 1000.0:
                        continue
                    th, shp = grid_theta(float(z), phot, bg, dev)
                    n = len(th)
                    basis = tuple(b.expand(n, *b.shape[1:]) for b in Bb)
                    for cname, (Tc, K) in cases.items():
                        out = {}
                        for s in range(0, n, 160):
                            sl = slice(s, s + 160)
                            bs = tuple(b[sl] for b in basis)
                            c = crlb_orientation(psf, th[sl], T=Tc, K=K, basis=bs)
                            for k in KEYS:
                                out.setdefault(k, []).append(c[k].cpu())
                            for k, (i, j) in CORR.items():
                                out.setdefault(k, []).append(c['corr'][:, i, j].cpu())
                        key = (mname, cname, float(z), phot, bg)
                        res[key] = {k: torch.cat(v).reshape(shp).numpy() for k, v in out.items()}
            print(f'{mname} z={z}: done', flush=True)
    return res, {n: {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in e.items()} for n, e in exc.items()}, \
        {k: v[1] for k, v in ms.items()}


def iso_bias(dev, phot=1000.0, bg=10.0):
    """Fit noise-free expected images of dipoles (g = 1, 0.7; phi = 0) with the isotropic model (x, y, z, I, bg);
    z allowed in [-0.1, 0.5] um. Returns {model: dict(theta, z, g -> dx nm, dz nm)}."""
    ms = models(dev)
    out = {}
    for mname, (psf, _) in ms.items():
        zr0 = psf.z_range
        psf.z_range = (-0.1, 0.5)
        rows = []
        for z in Z:
            for g in (1.0, 0.7):
                for phi in (0.0, 90.0):
                    mu = from_angles(torch.tensor(THETA), torch.full((len(THETA),), phi)).to(dev)
                    n = len(mu)
                    pos = torch.tensor([[0.0, 0.0, float(z)]], device=dev).expand(n, 3).contiguous()
                    with torch.no_grad():
                        img = psf(pos, psf.focus_um, dipM=dipole_M(mu, torch.full((n,), g, device=dev)))
                    d = phot * img + bg
                    th0 = torch.stack([pos[:, 0], pos[:, 1], pos[:, 2], phot * img.sum((1, 2)), torch.full((n,), bg, device=dev)], 1)
                    t1, n1 = localize(psf, d, theta0=th0, iters=60)
                    t2, n2 = localize(psf, d, iters=60, zstarts=6)
                    t = torch.where((n2 < n1)[:, None], t2, t1)
                    for i in range(n):
                        rows.append(dict(z=float(z), g=g, phi=phi, theta=float(THETA[i]),
                                         dx_nm=float(t[i, 0]) * psf.pixel_um * 1e3, dy_nm=float(t[i, 1]) * psf.pixel_um * 1e3,
                                         dz_nm=(float(t[i, 2]) - float(z)) * 1e3, I_ratio=float(t[i, 3] / th0[i, 3])))
        psf.z_range = zr0
        out[mname] = rows
        print(f'bias {mname}: done', flush=True)
    return out


def contrast_study(dev, phot=1000.0, bg=10.0):
    """Angle-brightness contrast c = (F_1 - F_0) / (F_1 + F_0), F_a = 3 tr(M K_a) (per-angle mean over its frames):
    max |c| over orientations and its sd over uniformly distributed axes vs g, for the fitted and pure-s K; and the CRLB
    of c for a 6-frame track (3 frames per angle, per-angle intensities free, isotropic PSF of each model, z = 50 nm)."""
    exc = {n: ensemble_to_K(n) for n in ('m1809', 'm25')}
    gen = torch.Generator().manual_seed(0)
    mu = unit(torch.randn(100000, 3, generator=gen))
    rows = []
    for n, e in exc.items():
        for kname in ('K', 'K_s'):
            K = e[kname]
            for g in (1.0, 0.7, 0.55, 0.4, 0.3, 0.2, 0.1):
                M = dipole_M(mu, torch.full((len(mu),), g))
                F = excitation_factor(M[:, None], K[None])
                c = (F[:, 1] - F[:, 0]) / (F[:, 1] + F[:, 0])
                rows.append(dict(K=f'{n}:{kname}', eps=0.0 if kname == 'K_s' else e['eps'], g=g,
                                 c_max=float(c.abs().max()), c_sd=float(c.std())))
    # CRLB of c: per-angle intensities I_0, I_1 (3 frames each) with a shared position / bg
    crl = {}
    for mname, (psf, _) in models(dev).items():
        pos = torch.tensor([[0.0, 0.0, 0.05]], device=dev)
        th = torch.tensor([[0.0, 0.0, 0.05, phot, bg]], device=dev)
        F1 = fisher5(psf, th)[0].double()                # (x, y, z, I, bg) of one frame
        # 6 frames: params (x, y, z, I0, I1, bg); frames of angle a carry I_a
        Fm = torch.zeros(6, 6, dtype=torch.float64, device=dev)
        idx = [[0, 1, 2, 3, 5], [0, 1, 2, 4, 5]]
        for a in range(2):
            ii = torch.tensor(idx[a], device=dev)
            Fm[ii[:, None], ii[None, :]] += 3 * F1
        C = torch.linalg.inv(Fm)
        # c = (I1 - I0) / (I1 + I0) at I0 = I1 = I: grad = (-1, 1) / (2 I)
        gvec = torch.zeros(6, dtype=torch.float64, device=dev); gvec[3] = -1 / (2 * phot); gvec[4] = 1 / (2 * phot)
        crl[mname] = float(torch.sqrt(gvec @ C @ gvec))
    return rows, crl


def summarize(res):
    """Headline table: 1000 photons, bg 10, z = 50 nm; median over phi; theta = 60 deg (and median over 30..80)."""
    lines = []
    ti = list(THETA).index(60.0)
    sel = [i for i, t in enumerate(THETA) if 30 <= t <= 80]
    for (m, c, z, p, b), r in sorted(res.items()):
        if z != 0.05 or p != 1000.0 or b != 10.0:
            continue
        for gi, g in enumerate(G):
            med = lambda k, ix: float(np.median(r[k][ix][:, :, gi] if isinstance(ix, list) else r[k][ix, :, gi]))
            lines.append(dict(model=m, case=c, g=float(g), phi60=med('phi', ti), theta60=med('theta', ti),
                              g60=med('g', ti), z60_nm=1e3 * med('z', ti), x60_nm=1e3 * PIXEL_UM * med('x', ti),
                              phi_30_80=med('phi', sel), theta_30_80=med('theta', sel), g_30_80=med('g', sel)))
    return lines


PIXEL_UM = 0.10833


def figures(res, bias, plt):
    style(plt)
    mnames = ['free_f0.10', 'm25', 'free_f0.31', 'm1809']
    lab = {'free_f0.10': 'aberration-free, focus 101 nm', 'm25': '2.5 nM in-situ PSF (focus 101 nm)',
           'free_f0.31': 'aberration-free, focus 308 nm', 'm1809': '1809 in-situ PSF (focus 308 nm)'}
    ls = {'free_f0.10': '--', 'm25': '-', 'free_f0.31': '--', 'm1809': '-'}
    col = {'free_f0.10': COLORS[0], 'm25': COLORS[0], 'free_f0.31': COLORS[1], 'm1809': COLORS[1]}
    # fig 1: single frame, 1000 photons, z 50 nm, g = 1: sd vs theta (median over phi)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for m in mnames:
        r = res[(m, 'single', 0.05, 1000.0, 10.0)]
        for a, k in zip(ax, ('phi', 'theta', 'z')):
            y = np.median(r[k][:, :, 0], 1) * (1e3 if k == 'z' else 1)
            a.plot(THETA, y, ls[m], color=col[m], label=lab[m])
    ax[0].set_ylabel('sd azimuth phi (deg)'); ax[1].set_ylabel('sd polar theta (deg)'); ax[2].set_ylabel('sd z (nm)')
    ax[0].set_ylim(0, 60); ax[1].set_ylim(0, 40)
    for a in ax:
        a.set_xlabel('polar angle theta (deg)')
    ax[0].legend(fontsize=7.5)
    fig.suptitle('CRLB, single frame: 1000 photons, bg 10 / px, z = 50 nm, fixed dipole (g = 1), median over phi',
                 fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_crlb_vs_theta.png')); plt.close(fig)
    # fig 2: vs g, theta = 60, m1809 and m25: single / track6 / track6_K
    ti = list(THETA).index(60.0)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    cc = {'single': COLORS[0], 'track6': COLORS[1], 'track6_K': COLORS[2], 'track6_Ks': COLORS[3]}
    cl = {'single': 'single frame', 'track6': 'track, 6 frames', 'track6_K': 'track + excitation (in-situ K)',
          'track6_Ks': 'track + excitation (pure s)'}
    for m, lst in (('m1809', '-'), ('m25', ':')):
        for c in cc:
            r = res[(m, c, 0.05, 1000.0, 10.0)]
            for a, k in zip(ax, ('phi', 'theta', 'g')):
                a.plot(G, np.median(r[k][ti], 0), lst, marker='o', ms=3, color=cc[c],
                       label=f'{cl[c]} ({m})')
    ax[0].set_ylabel('sd azimuth phi (deg)'); ax[1].set_ylabel('sd polar theta (deg)'); ax[2].set_ylabel('sd g')
    ax[0].set_ylim(0, 60); ax[1].set_ylim(0, 40); ax[2].set_ylim(0, 0.6)
    for a in ax:
        a.set_xlabel('order parameter g'); a.invert_xaxis()
    ax[0].legend(fontsize=6.5, ncol=1)
    fig.suptitle('CRLB vs g: theta = 60 deg, 1000 photons / frame, bg 10, z = 50 nm, median over phi '
                 '(solid: 1809 PSF, dotted: 2.5 nM PSF)', fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_crlb_vs_g.png')); plt.close(fig)
    # fig 3: isotropic-model bias, z = 50 nm, g = 1, phi = 0
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for m in mnames:
        rows = [r for r in bias[m] if r['z'] == 0.05 and r['g'] == 1.0 and r['phi'] == 0.0]
        t = [r['theta'] for r in rows]
        ax[0].plot(t, [r['dz_nm'] for r in rows], ls[m], color=col[m], label=lab[m])
        ax[1].plot(t, [r['dx_nm'] for r in rows], ls[m], color=col[m])
    ax[0].set_ylabel('z bias (nm)'); ax[1].set_ylabel('x bias (nm), dipole tilted along x')
    for a in ax:
        a.set_xlabel('polar angle theta (deg)')
    ax[0].legend(fontsize=7)
    fig.suptitle('Isotropic-PSF fit of fixed dipoles (g = 1), z = 50 nm, 1000 photons, bg 10, noise-free',
                 fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_iso_bias.png')); plt.close(fig)
    # fig 4: theta-z correlation and the effect of z
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for m in ('m1809', 'm25'):
        r = res[(m, 'single', 0.05, 1000.0, 10.0)]
        ax[0].plot(THETA, np.median(np.abs(r['c_az'][:, :, 0]), 1), ls[m], color=col[m], label=lab[m])
        ax[0].plot(THETA, np.median(np.abs(r['c_bx'][:, :, 0]), 1), ls[m], color=col[m], alpha=0.45)
        sd = [np.median(res[(m, 'single', float(z), 1000.0, 10.0)]['theta'][list(THETA).index(60.0), :, 0]) for z in Z]
        ax[1].plot(Z * 1e3, sd, ls[m], marker='o', ms=3, color=col[m], label=lab[m])
    ax[0].set_ylabel('|correlation|'); ax[0].set_xlabel('polar angle theta (deg)')
    ax[0].text(5, 0.92, 'dark: theta-z, light: phi-x', fontsize=7, color=INK2)
    ax[0].set_ylim(0, 1)
    ax[1].set_xlabel('emitter height z (nm)'); ax[1].set_ylabel('sd theta at theta = 60 (deg)')
    ax[0].legend(fontsize=7, loc='lower right')
    fig.suptitle('Single frame, g = 1, 1000 photons, bg 10 (median over phi)', fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_corr_z.png')); plt.close(fig)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    res, exc, desc = crlb_sweep(dev)
    np.savez_compressed(os.path.join(OUT, 'crlb_sweep.npz'),
                        **{'|'.join(map(str, k)) + '|' + q: v for k, r in res.items() for q, v in r.items()})
    bias = iso_bias(dev)
    crows, ccrlb = contrast_study(dev)
    summ = summarize(res)
    json.dump(dict(axes=dict(theta=THETA.tolist(), phi=PHI.tolist(), g=G.tolist(), z=Z.tolist()), excitation=exc,
                   models=desc, summary=summ, iso_bias=bias, contrast=crows, contrast_crlb_track6=ccrlb),
              open(os.path.join(OUT, 'crlb_summary.json'), 'w'), indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures(res, bias, plt)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
