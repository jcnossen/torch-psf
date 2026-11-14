"""PSF estimation recovers known Zernike sets on simulated data: bead stacks and in situ (knot route)."""
import torch
from conftest import DEV, needs_gpu
from torch_psf import VectorialPSF, CubicSplinePSF
from torch_psf.insitu import fit_beads, fit_zernike_knots, stage_focus, z_scale
from torch_psf.simulate import simulate_bead_stack, simulate_emitters

TRUE = {5: 0.02, 6: 0.10, 7: -0.03, 8: 0.02, 11: 0.04}


def true_model(R, **kw):
    v = VectorialPSF(R, npupil=48, oversample=2, device=DEV, **kw)
    return v.set_zernike(TRUE)


@needs_gpu
def test_bead_fit_recovers_zernikes():
    vec = true_model(15)
    focus = stage_focus(26, 0.04, 1.0).to(DEV)
    sim = simulate_bead_stack(vec.forward_batched, focus, nbeads=4, photons=40000.0, bg=30.0, device=DEV,
                              generator=torch.Generator(device=DEV).manual_seed(2))
    start = VectorialPSF(15, npupil=48, oversample=2, device=DEV)
    est = fit_beads(sim['d'], focus, start, iters=300)
    err = (est['psf'].zcoef.detach() - vec.zcoef.detach()).pow(2).mean().sqrt()
    assert float(err) < 0.005, float(err)
    assert est['report']['method'] == 'beads'


@needs_gpu
def test_insitu_zernike_fit_recovers_known_set():
    vec = true_model(15, focus_um=0.4, z_range=(0.0, 0.8))
    g = torch.Generator(device=DEV).manual_seed(4)
    sim = simulate_emitters(vec.render, 4000, bg=20.0, photons=4000.0, z_range=(0.05, 0.75), generator=g, device=DEV)
    start = VectorialPSF(15, npupil=48, oversample=2, device=DEV, focus_um=0.4, z_range=(0.0, 0.8))
    start.set_zernike({6: 0.08})                          # an approximate start (e.g. a bead calibration)
    spl = CubicSplinePSF(15, (0.0, 0.8), dz_um=0.04, zpad=5, xy_limit=3.0, device=DEV)
    est = fit_zernike_knots(spl, sim['d'], start, iters=80)
    err = (est['psf'].zcoef.detach() - vec.zcoef.detach()).pow(2).mean().sqrt()
    assert float(err) < 0.006, float(err)
    assert est['report']['at_limit'] < 0.01
    assert 'z_scale' in est['report']
    sc, off = z_scale(est['psf'].to_spline(15, dz_um=0.04), vec.to_spline(15, dz_um=0.04))
    assert abs(sc - 1) < 0.03 and abs(off) < 0.02, (sc, off)


@needs_gpu
def test_bead_fit_dz_scale_height_and_heldout_refit():
    vec = true_model(15)
    focus_true = stage_focus(26, 0.04, 1.0).to(DEV) * 0.8          # the stage moves 20 % less than nominal
    sim = simulate_bead_stack(vec.forward_batched, focus_true, nbeads=5, photons=40000.0, bg=30.0, device=DEV,
                              emitter_z=0.1, generator=torch.Generator(device=DEV).manual_seed(5))
    start = VectorialPSF(15, npupil=48, oversample=2, device=DEV)
    w = torch.tensor([1.0, 1, 1, 1, 0], device=DEV)
    est = fit_beads(sim['d'], stage_focus(26, 0.04, 1.0).to(DEV), start, iters=400, fit_dz_scale=True,
                    fit_height=True, emitter_z_um=0.05, weights=w)
    rep = est['report']
    assert abs(rep['dz_scale'] - 0.8) < 0.03 and abs(rep['height_um'] - 0.1) < 0.03, rep
    assert 'heldout_nll_per_px' in rep and rep['heldout_nll_per_px'] < rep['nll_per_px'] + 0.5
