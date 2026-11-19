"""Vortex PSF (Hulleman et al., Nat. Commun. 12, 5934 (2021)): a VectorialPSF with a spiral phase plate in the
emission pupil, and its calibration from bead z-stacks.

The plate is a fixed pupil phase (orientation.vortex_phase written to VectorialPSF.pupil_phase); the system
aberrations are Zernikes on top of it. Beads are isotropic emitters (g = 0), so the bead fit is the usual
ZStackEstimator / estimation.fit_beads with the vortex start model (fit_beads copies pupil_phase and only fits the
Zernikes). calibrate_beads adds what the vortex needs: the handedness (charge +-1, chosen by the likelihood) and
optionally a lateral offset of the plate centre (fitted with the Zernikes fixed, alternating with the Zernike fit).

    psf = vortex_psf(Optics(NA=1.47, n_med=1.49, wavelength_um=0.5975, pixel_um=0.065), roisize=15, charge=-1)
    psf, rep = calibrate_beads(stacks, dz_um=0.1, start=psf, charges=(-1, 1), fit_center=True)

Field dependence (their Nodal Aberration Theory map): torch_psf.field.FieldPSF / FieldZernikes can hold Zernikes that
vary over the field on top of the same fixed pupil_phase; not done here yet (constant aberrations).
"""
import copy
import time
import torch
from torch_psf.optics import Optics
from torch_psf.vectorial import VectorialPSF
from torch_psf.fit import poisson_nll
from torch_psf.orientation import vortex_phase


def vortex_psf(optics=None, roisize=15, charge=-1, center=(0.0, 0.0), levels=0, depth=1.0, start=None,
               zernike_indices=tuple(range(5, 22)), npupil=64, oversample=3, device=None, focus_um=0.0,
               z_range=(-0.5, 0.5)):
    """A VectorialPSF with the vortex phase (orientation.vortex_phase arguments charge, center, levels, depth).
    start: a VectorialPSF to copy (its Zernikes etc.; resized to roisize), else optics (Optics / dict / None)."""
    if start is not None:
        v = start.resized(roisize) if start.roisize != roisize else copy.deepcopy(start)
    else:
        o = optics if isinstance(optics, Optics) else Optics.from_dict(optics or {})
        v = VectorialPSF.from_optics(o, roisize, zernike_indices=zernike_indices, npupil=npupil,
                                     oversample=oversample, device=device, focus_um=focus_um, z_range=z_range)
    vortex_phase(v, charge=charge, center=center, levels=levels, depth=depth)
    v.vortex = dict(charge=int(charge), center=[float(c) for c in center], levels=int(levels), depth=float(depth))
    return v


def _fit_center(psf, d, focus, xy, photons, bg, vortex, emitter_z_um=0.0, iters=150, lr=2e-3):
    """Plate centre (pupil radii) with the Zernikes fixed; per-bead xy and per-plane photons / bg refined too.
    d [nb, nk, P, P] photons, focus [nk]. Returns (center [2], nll)."""
    nb, nk, P = d.shape[0], d.shape[1], d.shape[-1]
    dev = d.device
    c = torch.tensor(vortex['center'], device=dev, dtype=torch.float32, requires_grad=True)
    xy = xy.clone().requires_grad_(True)
    logI = torch.log(photons.clamp_min(1.0)).clone().requires_grad_(True)
    bg = bg.clone().requires_grad_(True)
    opt = torch.optim.Adam([dict(params=[c], lr=lr), dict(params=[xy], lr=1e-2), dict(params=[logI], lr=1e-2),
                            dict(params=[bg], lr=0.3)])
    f = focus[None].expand(nb, -1).reshape(-1)
    pp0 = psf.pupil_phase.clone()
    l = None
    for _ in range(iters):
        opt.zero_grad()
        psf.pupil_phase = vortex_phase(psf, vortex['charge'], center=c, levels=0, depth=vortex['depth'], apply=False)
        pos = torch.cat([xy[:, None].expand(-1, nk, -1), torch.full((nb, nk, 1), emitter_z_um, device=dev)], -1)
        mu = torch.exp(logI).reshape(-1, 1, 1) * psf(pos.reshape(-1, 3), f) + bg.reshape(-1, 1, 1)
        l = poisson_nll(d.reshape(-1, P, P), mu).sum()
        l.backward()
        opt.step()
    psf.pupil_phase = pp0
    return c.detach(), float(l.detach())


def calibrate_beads(stacks, dz_um, start, camera=None, charges=(-1, 1), fit_center=False, rounds=2, center_iters=150,
                    iters=300, **zstack_kw):
    """Vortex PSF from bead z-stacks: ZStackEstimator (model 'vectorial', Zernikes on top of the fixed vortex phase)
    for each charge in `charges` (the handedness; the best NLL wins), then optionally `rounds` alternations of the
    plate-centre fit (_fit_center, Zernikes fixed) and the Zernike fit. start: vortex_psf(...) (its .vortex dict gives
    the plate settings) or a VectorialPSF (treated as charge -1, centre 0). zstack_kw: ZStackEstimator options
    (roisize, nbeads, sign, drift, bead_diameter_um, ...). Returns (psf, report) with report = dict(charge,
    nll_per_px per charge, center, zernike_waves, zstack report)."""
    from torch_psf.estimators import ZStackEstimator
    vx = dict(getattr(start, 'vortex', None) or dict(charge=-1, center=[0.0, 0.0], levels=0, depth=1.0))
    rep = dict(nll_per_px={}, rounds=[])
    t0 = time.time()
    best = None
    for ch in charges:
        s = vortex_psf(start=start, roisize=start.roisize, charge=ch, center=vx['center'], levels=vx['levels'],
                       depth=vx['depth'])
        est = ZStackEstimator(model='vectorial', start=s, camera=camera, iters=iters, device=start.dev, **zstack_kw)
        psf = est.estimate(stacks, dz_um)
        nll = est.report['nll_per_px']
        rep['nll_per_px'][int(ch)] = nll
        if best is None or nll < best[0]:
            best = (nll, ch, psf, est)
    nll, ch, psf, est = best
    vx['charge'] = int(ch)
    psf.vortex = dict(vx)
    if fit_center:
        for r in range(rounds):
            dd = est.data
            c, l = _fit_center(psf, dd['rois'], dd['focus_um'], dd['xy'], dd['photons'], dd['bg'], vx,
                               emitter_z_um=est.emitter_z_um, iters=center_iters)
            vx['center'] = [float(c[0]), float(c[1])]
            s = vortex_psf(start=psf, roisize=psf.roisize, **vx)
            est = ZStackEstimator(model='vectorial', start=s, camera=camera, iters=iters, device=start.dev, **zstack_kw)
            psf = est.estimate(stacks, dz_um)
            psf.vortex = dict(vx)
            rep['rounds'].append(dict(center=list(vx['center']), nll_per_px=est.report['nll_per_px']))
    rep.update(charge=vx['charge'], center=vx['center'], zernike_noll=list(psf.zidx),
               zernike_waves=psf.zcoef.detach().cpu().tolist(), zstack=est.report, seconds=time.time() - t0)
    return psf, rep
