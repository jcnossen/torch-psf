"""Vortex orientation pipeline on the real data of Hulleman et al. (Nat. Commun. 2021; 4TU 10.4121/13031837), stored
read-only in ~/data/vortex_hulleman2021 (untrusted input: read with tifffile / h5py / yaml.safe_load only; run this
script with python -I).

Stages (argv):
  beads    the 180 nm bead z-stacks in oil (no vortex plate: a standard PSF; their code also calibrates without it)
           -> ZStackEstimator (Zernikes 5-21, constant over the field) -> results/real_beads_psf.json
  plate    the vortex plate on the PMMA single-molecule z-stack: charge (+-1) and centre offset, chosen by the
           total likelihood of orientation fits of the brightest in-focus spots -> results/real_plate.json
  pmma     the PMMA z-stacks with and without the vortex plate: OrientationEstimator per slice; per molecule the
           spread of phi / theta / g over the slices vs the CRLB and the z vs stage slope (paper: (2.3, 1.8) deg vs
           CRLB (1.8, 1.5), slope -0.99); the standard-PSF stack gives an independent orientation per molecule
  reorient the re-orienting ATTO 565 molecules in air (900 ms frames)
  dna      the first lambda-DNA movie (stk_0001): Delta phi to the local DNA axis and the g histogram, compared with
           their processed localisations (2500_2_X1_10.hdf5)

Settings from their MATLAB code (vecfitcpu_vortex): NA 1.47, n_imm 1.518 (coverslip 1.523 there; index-matched
here), pixel 65 nm; beads: 596 nm, n_med 1.518, offset 104.2 ADU, 0.52 e / ADU, bead radius 90 nm; single
molecules: 597.5 nm, offset 106 ADU, 0.33 e / ADU (their lambda-DNA settings; used for all single-molecule data),
vortex 64 levels, phase depth 593 nm, charge -1 in our pupil coordinates (their doevortexflip = 1 with the pupil x
along the image rows). PMMA n = 1.49, air n = 1.0, water n = 1.333.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np                                                                     # noqa: E402
import tifffile                                                                        # noqa: E402
import torch                                                                           # noqa: E402
from torch_psf.optics import Optics                                                    # noqa: E402
from torch_psf.camera import Camera                                                    # noqa: E402
from torch_psf.psf import PSF                                                          # noqa: E402
from torch_psf.estimators import ZStackEstimator, find_beads                           # noqa: E402
from torch_psf.vortex import vortex_psf                                                # noqa: E402
from torch_psf.orientation import OrientationEstimator, fit_orientation, FocusZ       # noqa: E402

DATA = os.path.expanduser('~/data/vortex_hulleman2021')
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CAM_BEAD = Camera(gain_e_per_adu=0.52, offset_adu=104.2)
CAM_SM = Camera(gain_e_per_adu=0.33, offset_adu=106.0)
OPT = dict(NA=1.47, n_imm=1.518, pixel_um=0.065)
DEPTH = 593.0 / 597.5


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ beads
def run_beads(stacks=(1, 2, 3, 4, 5, 6), planes=(7, 38), nbeads=8, roisize=25, sat_adu=60000, bead_blur=True,
              crop=(512, 1536), variants=None):
    """planes: the in-focus plane of the beads is ~22 of 51 (not the stack centre 25), so the 31 planes around it
    are used (the stack centre is focus 0 for ZStackEstimator; Zernike defocus is not fitted). Saturated beads
    (any pixel >= sat_adu) are blanked before the bead search."""
    variants = variants or {
        'sign+1': dict(sign=1.0), 'sign-1': dict(sign=-1.0), 'sign+1_dz': dict(sign=1.0, fit_dz_scale=True),
        'sign+1_na': dict(sign=1.0, fit_na=True), 'sign+1_apod': dict(sign=1.0, fit_apod=True),
        'sign+1_dz_apod': dict(sign=1.0, fit_dz_scale=True, fit_apod=True)}
    t0 = time.time()
    arr = []
    for i in stacks:
        a = np.asarray(tifffile.imread(os.path.join(DATA, f'oil_3perc_300ms_100nmsteps_EPI_{i}.tif')), np.float32)
        sat = np.argwhere(a.max(0) >= sat_adu)
        med = np.median(a[:, ::16, ::16], axis=(1, 2))
        for y, x in sat:
            a[:, max(y - 20, 0):y + 21, max(x - 20, 0):x + 21] = med[:, None, None]
        arr.append(a[planes[0]:planes[1], crop[0]:crop[1], crop[0]:crop[1]])   # central field (the SM data)
    res = {}
    best = None
    for name, var in variants.items():
        start = PSF._registry['vectorial'].from_optics(Optics(n_med=1.518, wavelength_um=0.596, **OPT), roisize,
                                                        npupil=64, oversample=2, device=DEV)
        # the 180 nm bead (uniformly stained sphere) as a lateral Gaussian blur: sd = r / sqrt(5) = 40 nm = 0.62 px
        start.blur = torch.tensor([0.383, 0.383, 0.0], device=DEV) if bead_blur else None
        est = ZStackEstimator(model='vectorial', start=start, camera=CAM_BEAD, roisize=roisize, nbeads=nbeads,
                              iters=400, emitter_z_um=0.09, drift=True, drift_order=2, holdout=len(stacks) - 1,
                              device=DEV, **var)
        psf = est.estimate(arr, 0.1)
        r = {k: est.report.get(k) for k in ('nll_per_px', 'heldout_nll_per_px', 'chi2_median', 'n_beads', 'NA',
                                            'dz_scale', 'apod')}
        r['settings'] = var
        r['zernike_waves'] = psf.zcoef.detach().cpu().tolist()
        res[name] = r
        log(f'beads {name}: {r}')
        if best is None or r['heldout_nll_per_px'] < best[0]:      # the held-out stack decides
            best = (r['heldout_nll_per_px'], var.get('sign', 1.0), psf, name)
    res['chosen'] = best[3]
    _, sign, psf, _ = best
    psf.blur = None                    # the bead size, not part of the single-molecule PSF
    psf.save(os.path.join(OUT, 'real_beads_psf.json'), provenance=dict(source='vortex_real.py beads', sign=sign))
    res.update(sign=sign, zernike_noll=list(psf.zidx), NA=float(psf.NA), apod=psf.apod.tolist(),
               seconds=time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'real_beads.json'), 'w'), indent=1)
    log(json.dumps({k: v for k, v in res.items() if not k.startswith('sign')}, indent=1))


def sm_model(n_med, roisize=15, charge=-1, center=(0.0, 0.0), focus=0.0, z_range=(-0.6, 0.6), vortex=True):
    """Calibrated Zernikes (beads) with the vortex plate (or without), for single molecules in a medium n_med."""
    b = PSF.load(os.path.join(OUT, 'real_beads_psf.json'), device=DEV)
    o = Optics(n_med=n_med, wavelength_um=0.5975, **OPT)
    if vortex:
        v = vortex_psf(o, roisize=roisize, charge=charge, center=center, levels=0, depth=DEPTH, device=DEV,
                       oversample=2, focus_um=focus, z_range=z_range)
    else:
        v = PSF._registry['vectorial'].from_optics(o, roisize, npupil=64, oversample=2, device=DEV, focus_um=focus,
                                                    z_range=z_range)
    zw = b.zcoef.detach().cpu().tolist()
    pf = os.path.join(OUT, 'real_plate.json')
    if vortex and os.path.exists(pf):
        zw = json.load(open(pf)).get('zernike_waves', zw)      # in-situ refined (plate2)
    v.set_zernike(zw)
    with torch.no_grad():
        v.apod.copy_(b.apod)           # objective / relay transmission (fitted on the beads)
        v.NA.copy_(b.NA)
    for p in v.parameters():
        p.requires_grad_(False)
    return v


def plate():
    return json.load(open(os.path.join(OUT, 'real_plate.json')))


def sm_focusz(n_med=None, vortex=True, z_range=(-0.9, 0.9), height=None):
    """The calibrated single-molecule model with z = position relative to the focal plane (FocusZ): plate charge /
    centre / Zernikes from real_plate.json; n_med and the emitter height from it unless given."""
    pl = plate()
    n = pl.get('n_med', 1.0) if n_med is None else n_med
    h = pl.get('height_um', 0.0) if height is None else height
    v = sm_model(n, charge=pl['charge'], center=pl['center'], z_range=z_range, vortex=vortex)
    return FocusZ(v, height=h, z_range=z_range)


# ------------------------------------------------------------------ PMMA z-stacks
PMMA_V = 'atto565_zstack_20perc_600ms_YESvortex_YESqwp_3.tif'
PMMA_S = 'atto565_zstack_20perc_600ms_NOvortex_YESqwp_3.tif'


def stack_rois(name, centers, roisize=15, camera=CAM_SM):
    """ROIs [n_mol, nk, P, P] (photons) of a z-stack around fixed centres (x, y) and the corners [n_mol, 2]."""
    a = np.asarray(tifffile.imread(os.path.join(DATA, name)), np.float32)
    h = roisize // 2
    rois, corners = [], []
    for x, y in centers:
        x0, y0 = int(x) - h, int(y) - h
        rois.append(a[:, y0:y0 + roisize, x0:x0 + roisize])
        corners.append((x0, y0))
    r = torch.as_tensor(np.stack(rois), device=DEV)
    return camera.to_photons(r), np.array(corners)


def pmma_molecules(nmax=40, roisize=15):
    """Molecule centres from the vortex stack (bright isolated spots), matched to the standard stack (within 6 px:
    the two stacks image the same field)."""
    v = np.asarray(tifffile.imread(os.path.join(DATA, PMMA_V)), np.float32)
    s = np.asarray(tifffile.imread(os.path.join(DATA, PMMA_S)), np.float32)
    cv = find_beads(v, 21, nmax=nmax, min_sep=21) + 10
    cs = find_beads(s, 21, nmax=3 * nmax, min_sep=21) + 10
    out = []
    for p in cv:
        d = np.abs(cs - p).max(1)
        if d.min() <= 6:
            out.append((p.tolist(), cs[d.argmin()].tolist()))
    return out


def fit_stack(psf, rois, iters=40, z_limits=(-0.9, 0.9), mirror=True, zgrid=(-0.6, -0.3, 0.0, 0.3, 0.6)):
    """Orientation fits of every plane of every molecule: rois [n_mol, nk, P, P] -> dict of [n_mol, nk] arrays."""
    nm, nk, P = rois.shape[:3]
    r = fit_orientation(psf, rois.reshape(-1, P, P), iters=iters, z_limits=z_limits, batch=400, mirror=mirror,
                        zgrid=zgrid)
    out = dict(theta=r['theta'].reshape(nm, nk, -1).cpu().numpy(), nll=r['nll'].reshape(nm, nk).cpu().numpy(),
               theta_deg=r['theta_deg'].reshape(nm, nk).cpu().numpy(), phi_deg=r['phi_deg'].reshape(nm, nk).cpu().numpy())
    for k in ('theta', 'phi', 'inplane', 'g', 'z', 'x', 'y', 'I'):
        out['crlb_' + k] = r['crlb'][k].reshape(nm, nk).cpu().numpy()
    return out


def run_plate(nmol=16, planes=range(8, 15)):
    """Charge and plate centre from the likelihood of the vortex PMMA spots (planes near focus, brightest molecules)."""
    t0 = time.time()
    mols = pmma_molecules()
    rois, _ = stack_rois(PMMA_V, [m[0] for m in mols])
    sig = (rois - rois.flatten(2).median(2).values[..., None, None]).clamp_min(0).flatten(2).sum(2).mean(1)
    idx = torch.argsort(sig, descending=True)[:nmol]
    d = rois[idx][:, list(planes)]
    res = dict(n_mol=len(idx), planes=list(planes), grid=[])

    def total(charge, center):
        psf = sm_model(1.49, charge=charge, center=center, z_range=(-0.9, 0.9))
        f = fit_stack(psf, d, iters=30)
        return float(f['nll'].sum())
    nll = {ch: total(ch, (0.0, 0.0)) for ch in (-1, 1)}
    res['nll_charge'] = nll
    ch = min(nll, key=nll.get)
    log(f'plate: charge NLL {nll} -> {ch} ({time.time() - t0:.0f} s)')
    best = ((0.0, 0.0), nll[ch])
    for step in (0.04, 0.02):
        c0 = best[0]
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                c = (round(c0[0] + dx, 4), round(c0[1] + dy, 4))
                if c == c0:
                    continue
                l = total(ch, c)
                res['grid'].append(dict(center=list(c), nll=l))
                if l < best[1]:
                    best = (c, l)
        log(f'plate: step {step}: best centre {best[0]} dNLL {best[1] - nll[ch]:.1f} ({time.time() - t0:.0f} s)')
    res.update(charge=ch, center=list(best[0]), nll_best=best[1], nll_center0=nll[ch], seconds=time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'real_plate.json'), 'w'), indent=1)
    log(json.dumps({k: v for k, v in res.items() if k != 'grid'}, indent=1))


def run_plate2(nmol=16, planes=range(4, 21), kc=12, iters=300, n_med=1.0):
    """In-situ plate calibration on the PMMA vortex z-stack (fit_dipole_stacks): charge x stage-sign by the NLL with
    the plate centre fitted (Zernikes from the beads), then the Zernikes, the emitter height and the medium index
    refined too. n_med = 1.0: the molecules sit in a thin PMMA film with air above, and the emission pattern is that
    of a glass / air interface (n_med 1.0 fits the stacks far better than 1.49; see the note)."""
    from torch_psf.vortex import fit_dipole_stacks
    t0 = time.time()
    mols = pmma_molecules()
    rois, _ = stack_rois(PMMA_V, [m[0] for m in mols])
    sig = (rois - rois.flatten(2).median(2).values[..., None, None]).clamp_min(0).flatten(2).sum(2).mean(1)
    idx = torch.argsort(sig, descending=True)[:nmol]
    d = rois[idx][:, list(planes)].contiguous()
    res = dict(molecules=idx.tolist(), planes=list(planes), kc=kc, runs={})
    best = None
    for ch in (1, -1):
        base = sm_model(n_med, charge=ch, z_range=(-0.9, 0.9))
        r0 = fit_orientation(base, rois[idx][:, kc], iters=40, z_limits=(0.0, 0.3))
        for sgn in (1.0, -1.0):
            f = sgn * 0.1 * (torch.tensor(list(planes), dtype=torch.float32) - kc)
            r = fit_dipole_stacks(base, d, f, ch, depth=DEPTH, mu0=r0['theta'][:, 5:8], xy0=r0['theta'][:, :2],
                                  iters=iters, fit_center=True, fit_zernike=False)
            key = f'charge{ch:+d}_sign{sgn:+.0f}'
            res['runs'][key] = dict(nll=r['nll'], nll_per_px=r['nll_per_px'], center=r['center'])
            log(f'plate2 {key}: nll/px {r["nll_per_px"]:.4f} centre {r["center"]} ({time.time() - t0:.0f} s)')
            if best is None or r['nll'] < best[0]:
                best = (r['nll'], ch, sgn, r)
    _, ch, sgn, r = best
    f = sgn * 0.1 * (torch.tensor(list(planes), dtype=torch.float32) - kc)
    r2 = fit_dipole_stacks(r['psf'], d, f, ch, center=r['center'], depth=DEPTH, mu0=r['mu'], xy0=r['xy'],
                           iters=2 * iters, fit_center=True, fit_zernike=True, height=0.01, fit_height=True,
                           fit_nmed=True)
    log(f'plate2 refined (Zernikes): nll/px {r2["nll_per_px"]:.4f} centre {r2["center"]}')
    chi2 = ((d - r2['model']) ** 2 / r2['model'].clamp_min(1e-6)).flatten(2).mean(2)
    res.update(charge=ch, stage_sign=sgn, center=r2['center'], nll_per_px=r2['nll_per_px'], n_med=r2['n_med'],
               height_um=r2['height'],
               zernike_waves=r2['psf'].zcoef.tolist(), chi2_median=float(chi2.median()),
               drift_px=r2['drift'].tolist(),
               chi2_per_plane=chi2.median(0).values.tolist(), g=r2['g'].tolist(), offsets_um=r2['offset'].tolist(),
               seconds=time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'real_plate.json'), 'w'), indent=1)
    log(json.dumps({k: v for k, v in res.items() if k not in ('runs',)}, indent=1))


def _circ_mean_sd(a, period=360.0, w=None):
    z = np.exp(2j * np.pi * np.asarray(a) / period)
    m = np.average(z, weights=w)
    return float(np.angle(m) * period / (2 * np.pi) % period), float(np.sqrt(-2 * np.log(max(abs(m), 1e-12))) * period / (2 * np.pi))


def _axis(th, ph):
    t, p = np.radians(th), np.radians(ph)
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)


def _mean_axis(mu):
    w, v = np.linalg.eigh((mu[:, :, None] * mu[:, None, :]).sum(0))
    a = v[:, 2]
    return a if a[2] >= 0 else -a


def _tangent_dev(mu, a):
    """deviations (deg) of axes mu [n, 3] from the axis a in its tangent coordinates (polar, in-plane)"""
    th = np.arccos(np.clip(a[2], -1, 1)); ph = np.arctan2(a[1], a[0])
    e1 = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    e2 = np.array([-np.sin(ph), np.cos(ph), 0.0])
    m = mu * np.sign((mu @ a))[:, None]
    return np.degrees(m @ e1), np.degrees(m @ e2), np.degrees(np.arccos(np.clip(np.abs(mu @ a), 0, 1))), th


def summarize_pmma(rows, kmin=3, kmax=20, zr=0.3, out_deg=30.0, step_um=0.1):
    """Per molecule: a robust line z(stage) (planes kmin..kmax; residuals > 150 nm dropped iteratively); the
    'inner' planes are those with |z| <= zr on that line (the Vortex PSF's working range) and a residual < 150 nm.
    Orientation spread over the inner planes in the tangent coordinates of the molecule's mean axis (outliers: axis
    > out_deg from it, counted separately), vs the median CRLB; the standard-PSF axis (defocused planes,
    0.25 < |z| < 0.8) vs the vortex axis."""
    res = dict(per_molecule=[])
    for r in rows:
        p = r['per_plane']
        k = np.arange(len(p['z_vortex']))
        z = np.array(p['z_vortex'])
        sel = (k >= kmin) & (k <= kmax)
        for _ in range(4):
            if sel.sum() < 4:
                break
            b, a0 = np.polyfit(k[sel] * step_um, z[sel], 1)
            resid = z - (a0 + b * k * step_um)
            sel = (k >= kmin) & (k <= kmax) & (np.abs(resid) < 0.15)
        if sel.sum() < 4:
            continue
        zl = a0 + b * k * step_um
        inner = sel & (np.abs(zl) <= zr)
        if inner.sum() < 3:
            continue
        mu = _axis(np.array(p['theta_vortex']), np.array(p['phi_vortex']))
        a = _mean_axis(mu[inner])
        da, db, err, th_a = _tangent_dev(mu[inner], a)
        ok = err < out_deg
        if ok.sum() >= 3:
            a = _mean_axis(mu[inner][ok]); da, db, err, th_a = _tangent_dev(mu[inner], a); ok = err < out_deg
        cr_t = float(np.median(np.array(p['crlb_theta'])[inner][ok])); cr_p = float(np.median(np.array(p['crlb_phi'])[inner][ok]))
        g = np.array(p['g_vortex'])[inner][ok]
        zs = np.array(p['z_std'])
        dfc = (np.abs(zs) > 0.25) & (np.abs(zs) < 0.8) & (k >= kmin) & (k <= kmax)
        mus = _axis(np.array(p['theta_std']), np.array(p['phi_std']))
        cmp = None
        if dfc.sum() >= 3:
            a_s = _mean_axis(mus[dfc])
            _, _, err_s, _ = _tangent_dev(mus[dfc], a_s)
            if (err_s < out_deg).sum() >= 3:
                a_s = _mean_axis(mus[dfc][err_s < out_deg])
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(a @ a_s))))))
            th_v, th_s = np.degrees(np.arccos(abs(a[2]))), np.degrees(np.arccos(abs(a_s[2])))
            ph_v, ph_s = np.degrees(np.arctan2(a[1], a[0])) % 180, np.degrees(np.arctan2(a_s[1], a_s[0])) % 180
            cmp = dict(axis_angle_deg=ang, dtheta=float(th_v - th_s), dphi_mod180=float((ph_v - ph_s + 90) % 180 - 90))
        res['per_molecule'].append(dict(mol=r['mol'], z_slope=float(b), dz_step_nm=float(b * step_um * 1e3),
                                        n_inner=int(inner.sum()), n_out=int((~ok).sum()),
                                        sd_theta=float(da[ok].std()), sd_phi=float(db[ok].std() / max(np.sin(th_a), 1e-3)),
                                        sd_inplane=float(db[ok].std()), crlb_theta=cr_t, crlb_phi=cr_p,
                                        theta=float(np.degrees(th_a)), g=float(g.mean()), sd_g=float(g.std()),
                                        vs_standard=cmp))
    pm = res['per_molecule']
    mean = lambda key: float(np.mean([q[key] for q in pm]))
    med = lambda key: float(np.median([q[key] for q in pm]))
    cm = [q['vs_standard'] for q in pm if q['vs_standard'] is not None]
    res.update(n_molecules=len(pm), z_slope_mean=mean('z_slope'), z_slope_sd=float(np.std([q['z_slope'] for q in pm])),
               dz_step_nm_mean=mean('dz_step_nm'), sd_theta_mean=mean('sd_theta'), sd_theta_median=med('sd_theta'),
               crlb_theta_mean=mean('crlb_theta'), sd_phi_mean=mean('sd_phi'), sd_phi_median=med('sd_phi'),
               crlb_phi_mean=mean('crlb_phi'), g_mean=mean('g'), g_sd_between=float(np.std([q['g'] for q in pm])),
               sd_g_mean=mean('sd_g'), outlier_planes=int(sum(q['n_out'] for q in pm)),
               inner_planes=int(sum(q['n_inner'] for q in pm)),
               vs_standard=dict(n=len(cm), axis_angle_median=float(np.median([c['axis_angle_deg'] for c in cm])),
                                dtheta_mean=float(np.mean([c['dtheta'] for c in cm])),
                                dtheta_sd=float(np.std([c['dtheta'] for c in cm])),
                                dphi_mean=float(np.mean([c['dphi_mod180'] for c in cm])),
                                dphi_sd=float(np.std([c['dphi_mod180'] for c in cm]))) if cm else None)
    return res


def run_pmma():
    t0 = time.time()
    pl = json.load(open(os.path.join(OUT, 'real_plate.json')))
    mols = pmma_molecules()
    rv, cv = stack_rois(PMMA_V, [m[0] for m in mols])
    rs, cs = stack_rois(PMMA_S, [m[1] for m in mols])
    pv = sm_focusz()
    ps = sm_focusz(vortex=False)
    fv = fit_stack(pv, rv)
    fs = fit_stack(ps, rs)
    log(f'pmma: fitted {rv.shape[0]} molecules x {rv.shape[1]} planes, both stacks ({time.time() - t0:.0f} s)')
    nk = rv.shape[1]
    rows = []
    for i in range(len(mols)):
        zv = fv['theta'][i, :, 2]
        k = np.arange(nk)
        inr = np.abs(zv) <= 0.3                      # the Vortex PSF's working range (fitted z)
        if inr.sum() < 3:
            continue
        sl = np.polyfit(k[inr] * 0.1, zv[inr], 1)[0]
        # the axis of the vortex fits: canonical mu; azimuth of the axis is phi (0..360 with mu_z >= 0)
        tv, phv, gv = fv['theta_deg'][i], fv['phi_deg'][i], fv['theta'][i, :, 8]
        ts, phs, gs = fs['theta_deg'][i], fs['phi_deg'][i], fs['theta'][i, :, 8]
        zs = fs['theta'][i, :, 2]
        outs = (np.abs(zs) > 0.25) & (np.abs(zs) < 0.8)   # standard PSF: defocused slices carry the orientation
        I = fv['theta'][i, inr, 3]
        row = dict(mol=i, xy=mols[i][0], n_inner=int(inr.sum()), z_slope=float(sl),
                   vortex=dict(theta_mean=float(tv[inr].mean()), theta_sd=float(tv[inr].std()),
                               phi_mean_sd=_circ_mean_sd(phv[inr]), g_mean=float(gv[inr].mean()), g_sd=float(gv[inr].std()),
                               crlb_theta=float(fv['crlb_theta'][i, inr].mean()), crlb_phi=float(np.median(fv['crlb_phi'][i, inr])),
                               crlb_g=float(fv['crlb_g'][i, inr].mean()), intensity=float(I.mean()),
                               bg=float(fv['theta'][i, inr, 4].mean())),
                   standard=dict(n=int(outs.sum()), theta_mean=float(ts[outs].mean()) if outs.any() else None,
                                 phi_mean_sd=_circ_mean_sd(phs[outs]) if outs.any() else None,
                                 g_mean=float(gs[outs].mean()) if outs.any() else None),
                   per_plane=dict(z_vortex=zv.tolist(), theta_vortex=tv.tolist(), phi_vortex=phv.tolist(),
                                  g_vortex=gv.tolist(), z_std=zs.tolist(), theta_std=ts.tolist(), phi_std=phs.tolist(),
                                  g_std=gs.tolist(), crlb_theta=fv['crlb_theta'][i].tolist(),
                                  crlb_phi=fv['crlb_phi'][i].tolist()))
        rows.append(row)
    summ = summarize_pmma(rows)
    summ.update(charge=pl['charge'], center=pl['center'], n_med=pl.get('n_med'), height_um=pl.get('height_um'))
    summ['seconds'] = time.time() - t0
    log(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, molecules=rows), open(os.path.join(OUT, 'real_pmma.json'), 'w'), indent=1)


# ------------------------------------------------------------------ focus of a single-frame data set
def best_focus(n_med, rois, charge, center, foci, z_limits=(0.0, 0.4), iters=30):
    """The model focal plane (um above the coverslip) for emitters on / near the coverslip: total NLL of the
    orientation fits of rois [N, P, P] over a grid of foci."""
    out = {}
    for f in foci:
        psf = sm_model(n_med, charge=charge, center=center, focus=float(f), z_range=z_limits)
        r = fit_orientation(psf, rois, iters=iters, z_limits=z_limits, batch=400)
        out[float(f)] = float(r['nll'].sum())
    return min(out, key=out.get), out


def _movie_photons(name, crop=None, frames=None, camera=CAM_SM):
    with tifffile.TiffFile(os.path.join(DATA, name)) as t:
        n = len(t.pages)
        f0, f1 = frames or (0, n)
        out = []
        for k in range(f0, f1):
            a = t.pages[k].asarray()
            if crop is not None:
                x0, y0, w, h = crop
                a = a[y0:y0 + h, x0:x0 + w]
            out.append(np.asarray(a, np.float32))
    return camera.to_photons(torch.as_tensor(np.stack(out)))


def _bright_spots(mov, n=200, roisize=15):
    from torch_psf.spots import extract_spots
    sp = extract_spots(mov.numpy(), roisize, None, None, device=DEV, threshold=8.0)
    sig = (sp.rois - sp.rois.flatten(1).median(1).values[:, None, None]).clamp_min(0).flatten(1).sum(1)
    return sp.rois[torch.argsort(sig, descending=True)[:n]]


# ------------------------------------------------------------------ re-orientation (ATTO 565 in air)
def run_reorient():
    t0 = time.time()
    mov = _movie_photons('reorientation_ATTO_565_tirf_qwp_vortex_900ms.tif')
    psf = sm_focusz(n_med=1.0, height=0.0)                 # molecules on glass in air
    f0 = None
    est = OrientationEstimator(psf, threshold=8.0, iters=40, z_limits=(-0.9, 0.9), batch=400, isolation=8,
                               zgrid=(-0.6, -0.3, 0.0, 0.3, 0.6))
    ds = est.estimate(mov.numpy())
    log(f'reorient: {len(ds)} spots ({time.time() - t0:.0f} s)')
    # link spots of the same molecule over frames (within 2 px of the molecule's first position)
    pos, fr = ds['pos'].numpy(), ds['frame'].numpy()
    mols = []
    for i in np.argsort(fr):
        for m in mols:
            if np.abs(m['xy'] - pos[i, :2]).max() < 2.0 and fr[i] not in m['frames']:
                m['idx'].append(i); m['frames'].add(fr[i]); break
        else:
            mols.append(dict(xy=pos[i, :2].copy(), idx=[i], frames={fr[i]}))
    rows = []
    for m in mols:
        if len(m['idx']) < 10:
            continue
        ii = np.array(m['idx'])[np.argsort(fr[m['idx']])]
        mu = ds['mu'][ii].numpy()
        # consecutive-frame axis changes; a jump > 20 deg is a re-orientation
        c = np.abs((mu[1:] * mu[:-1]).sum(1)).clip(max=1)
        step = np.degrees(np.arccos(c))
        cr = np.sqrt(ds['crlb_theta_deg'][ii].numpy() ** 2 + ds['crlb_inplane_deg'][ii].numpy() ** 2)
        rows.append(dict(xy=m['xy'].tolist(), n=len(ii), frames=fr[ii].tolist(), theta=ds['theta_deg'][ii].tolist(),
                         phi=ds['phi_deg'][ii].tolist(), g=ds['g'][ii].tolist(), z_nm=(ds['pos'][ii, 2] * 65).tolist(),
                         photons=ds['photons'][ii].tolist(), bg=ds['bg'][ii].tolist(), chi2=ds['chi2'][ii].tolist(),
                         step_deg=step.tolist(), crlb_axis_deg=cr.tolist(),
                         crlb_theta=ds['crlb_theta_deg'][ii].tolist(), crlb_inplane=ds['crlb_inplane_deg'][ii].tolist(),
                         crlb_g=ds['crlb_g'][ii].tolist(), n_jumps=int((step > 20).sum())))
    # precision on the stationary stretches: frame-to-frame axis change vs the CRLB (for two independent estimates
    # of the same axis the change ~ sqrt(2) x the per-frame error)
    st = [s for r in rows for s in r['step_deg'] if s < 20]
    cr = [c for r in rows for c in r['crlb_axis_deg']]
    summ = dict(focus_um=f0, n_spots=len(ds), n_molecules=len(rows), n_reorienting=int(sum(r['n_jumps'] > 0 for r in rows)),
                median_step_deg=float(np.median(st)), rms_step_deg=float(np.sqrt(np.mean(np.square(st)))),
                median_crlb_axis_deg=float(np.median(cr)),
                expected_rms_step_deg=float(np.sqrt(2) * np.sqrt(np.mean(np.square(cr)))),
                g_median=float(np.median([g for r in rows for g in r['g']])),
                photons_median=float(np.median([p for r in rows for p in r['photons']])),
                bg_median=float(np.median([b for r in rows for b in r['bg']])),
                chi2_median=float(np.median([c for r in rows for c in r['chi2']])), seconds=time.time() - t0)
    log(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, molecules=rows), open(os.path.join(OUT, 'real_reorient.json'), 'w'), indent=1)


# ------------------------------------------------------------------ lambda-DNA
DNA_MOVIE = 'stk_0001_2500_2_middle_tryrefocus.tif'


def _their_locs(frames=(0, 1980), crop=None):
    import h5py
    with h5py.File(os.path.join(DATA, '2500_2_X1_10.hdf5'), 'r') as f:
        L = f['locs'][:]
    m = (L['frame'] >= frames[0]) & (L['frame'] < frames[1])
    if crop is not None:
        x0, y0, w, h = crop
        m &= (L['x'] >= x0) & (L['x'] < x0 + w) & (L['y'] >= y0) & (L['y'] < y0 + h)
    return L[m]


def local_axis(xy, radius=3.0, min_n=8, min_aniso=4.0):
    """Local line direction (deg, mod 180) at each point from the PCA of its neighbours within radius (px); NaN if
    too few neighbours or not elongated (eigenvalue ratio < min_aniso)."""
    from scipy.spatial import cKDTree
    tr = cKDTree(xy)
    out = np.full(len(xy), np.nan)
    for i, nb in enumerate(tr.query_ball_point(xy, radius)):
        if len(nb) < min_n:
            continue
        p = xy[nb] - xy[nb].mean(0)
        w, v = np.linalg.eigh(p.T @ p)
        if w[1] < min_aniso * max(w[0], 1e-9):
            continue
        out[i] = np.degrees(np.arctan2(v[1, 1], v[0, 1])) % 180
    return out


def run_dna(crop_size=256, frames=(0, 600)):
    t0 = time.time()
    pl = json.load(open(os.path.join(OUT, 'real_plate.json')))
    L = _their_locs(frames)
    H, _, _ = np.histogram2d(L['y'], L['x'], bins=1024 // 32, range=[[0, 1024], [0, 1024]])
    by, bx = np.unravel_index(np.argmax(H), H.shape)              # the densest 32 px block: centre the crop there
    x0 = int(np.clip(bx * 32 + 16 - crop_size // 2, 0, 1024 - crop_size))
    y0 = int(np.clip(by * 32 + 16 - crop_size // 2, 0, 1024 - crop_size))
    crop = (x0, y0, crop_size, crop_size)
    mov = _movie_photons(DNA_MOVIE, crop=crop, frames=frames)
    log(f'dna: crop {crop}, {mov.shape[0]} frames read ({time.time() - t0:.0f} s)')
    f0 = None
    psf = sm_focusz(n_med=1.333, height=0.0)               # lambda-DNA on the coverslip in water; z = defocus
    est = OrientationEstimator(psf, threshold=6.0, iters=40, z_limits=(-0.6, 0.6), batch=500,
                               zgrid=(-0.2, 0.2))
    ds = est.estimate(mov.numpy())
    log(f'dna: {len(ds)} spots fitted ({time.time() - t0:.0f} s)')
    chi2 = ds['chi2'].numpy()
    keep = (chi2 > 0.75) & (chi2 < 3)
    xy = ds['pos'].numpy()[:, :2]
    ax = local_axis(xy[keep])
    ph = ds['phi_deg'].numpy()[keep] % 180
    dphi = (ph - ax) % 180
    ok = np.isfinite(dphi)
    g = ds['g'].numpy()[keep]
    hg, _ = np.histogram(g, bins=20, range=(0, 1))
    # theirs, same crop and frames
    Lc = _their_locs(frames, crop)
    txy = np.stack([Lc['x'] - x0, Lc['y'] - y0], 1)
    tax = local_axis(txy)
    tdphi = (Lc['sx'] % 180 - tax) % 180
    tok = np.isfinite(tdphi)
    # per-localisation match (same frame, < 1 px): our vs their azimuth axis, polar angle, g, z
    from scipy.spatial import cKDTree
    fr = ds['frame'].numpy()
    pairs = []
    for t in np.unique(fr):
        a = np.nonzero(fr == t)[0]; b = np.nonzero(Lc['frame'] - frames[0] == t)[0]
        if len(a) == 0 or len(b) == 0:
            continue
        d, j = cKDTree(txy[b]).query(xy[a])
        for ia, dd, jj in zip(a, d, j):
            if dd < 1.0:
                pairs.append((ia, b[jj]))
    pa = np.array(pairs)
    mad = lambda v: float(np.median(np.abs(v - np.median(v))))
    summ = dict(crop=list(crop), frames=list(frames), focus_um=f0, n_spots=len(ds), n_chi2_kept=int(keep.sum()),
                n_axis=int(ok.sum()), dphi_median=float(np.median(dphi[ok])), dphi_mad=mad(dphi[ok]),
                g_peak=float((np.argmax(hg) + 0.5) / 20), g_median=float(np.median(g)),
                theta_median=float(np.median(ds['theta_deg'].numpy()[keep])),
                z_median_nm=float(np.median(ds['pos'].numpy()[keep, 2] * 65)),
                photons_median=float(np.median(ds['photons'].numpy()[keep])), bg_median=float(np.median(ds['bg'].numpy()[keep])),
                crlb_xy_median_nm=float(np.median(ds['crlb'].numpy()[keep, :2]) * 65),
                crlb_inplane_median_deg=float(np.median(ds['crlb_inplane_deg'].numpy()[keep])),
                crlb_theta_median_deg=float(np.median(ds['crlb_theta_deg'].numpy()[keep])),
                crlb_g_median=float(np.median(ds['crlb_g'].numpy()[keep])), chi2_median=float(np.median(chi2)),
                theirs=dict(n=len(Lc), n_axis=int(tok.sum()), dphi_median=float(np.median(tdphi[tok])),
                            dphi_mad=mad(tdphi[tok]), g_median=float(np.median(Lc['ellipticity'])),
                            photons_median=float(np.median(Lc['photons'])), bg_median=float(np.median(Lc['bg'])),
                            z_median_nm=float(np.median(Lc['net_gradient']))),
                seconds=time.time() - t0)
    if len(pa):
        ia, ib = pa[:, 0], pa[:, 1]
        dax = ((ds['phi_deg'].numpy()[ia] - Lc['sx'][ib] + 90) % 180) - 90
        summ['matched'] = dict(n=len(pa), dphi_axis_median=float(np.median(dax)), dphi_axis_mad=mad(dax),
                               dg_median=float(np.median(ds['g'].numpy()[ia] - Lc['ellipticity'][ib])),
                               dz_median_nm=float(np.median(ds['pos'].numpy()[ia, 2] * 65 - Lc['net_gradient'][ib])),
                               dxy_median_nm=float(np.median(np.abs(xy[ia] - txy[ib])) * 65),
                               photons_ratio_median=float(np.median(ds['photons'].numpy()[ia] / Lc['photons'][ib])))
    log(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, dphi=dphi[ok].tolist(), g=g.tolist(), dphi_theirs=tdphi[tok].tolist(),
                   g_theirs=Lc['ellipticity'].tolist()), open(os.path.join(OUT, 'real_dna.json'), 'w'))
    ds.save(os.path.join(OUT, 'real_dna_locs.hdf5'))


if __name__ == '__main__':
    what = sys.argv[1:]
    if 'beads' in what:
        run_beads()
    if 'plate' in what:
        run_plate()
    if 'plate2' in what:
        run_plate2()
    if 'pmma' in what:
        run_pmma()
    if 'pmma_summary' in what:
        d = json.load(open(os.path.join(OUT, 'real_pmma.json')))
        sm = summarize_pmma(d['molecules'])
        d['summary'].update(sm)
        json.dump(d, open(os.path.join(OUT, 'real_pmma.json'), 'w'), indent=1)
        log(json.dumps({k: v for k, v in sm.items() if k != 'per_molecule'}, indent=1))
    if 'reorient' in what:
        run_reorient()
    if 'dna' in what:
        run_dna()
