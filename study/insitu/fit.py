"""PSF estimators of the in-situ study: the torch_psf estimators with the study's settings (common.py), and the
study's scoring.

  fit_vec_stack     vectorial Zernike model fitted to the bead stack (torch_psf.insitu.fit_beads)
  fit_knots         free spline knots from bead stack and / or SMLM patches (torch_psf.insitu.fit_free_knots)
  fit_vec_insitu    vectorial Zernike model fitted to SMLM patches through the knot route (route K) or with the
                    vectorial model per emitter (route V) (torch_psf.insitu.fit_zernike_knots)
  fit_pupil_insitu  Zernikes + free pupil maps (torch_psf.insitu.fit_pupil_knots)
"""
import math, time
import torch
import common as C
from torch_psf import ConvCubicSplinePSF
from torch_psf.fit import localize
from torch_psf.insitu import (fit_beads, fit_free_knots, fit_zernike_knots, fit_pupil_knots, bead_theta,
                              knot_gradient, vectorial_gradient, limit_fraction)


def new_spline(S=2, hz=0.04, xmax=3.0):
    # zpad 5: knots cover the bead stack range. xmax 3 px: the detection peak of a defocused, aberrated spot lies up
    # to 3 px from the emitter; with the old default (1.5 px) the offset clamp biased flexible PSF fits (CRITIC C-01).
    return ConvCubicSplinePSF(C.P, C.ZR, oversample=S, dz_um=hz, zpad=5, xy_limit=xmax, device=C.DEV)


def clamp_fraction(spl, th):
    """Fraction of emitters whose fitted x or y sits at the offset clamp of the spline."""
    return limit_fraction(spl, th)


def knots_from(spl, zc, cond='insitu'):
    big = C.model_vec(spl.Pl, zc)
    return spl.knots_from_vectorial(big, C.insitu_pos if cond == 'insitu' else C.bead_pos)


def fit_vec_stack(stack, iters=300, lr=3e-3):
    focus = C.F0 - stack['z_nom'] / C.ALPHA
    return fit_beads(stack['d'], focus, C.model_vec(C.P), iters=iters, lr=lr)['psf'].zcoef.detach().clone()


def bead_params(spl, stack, th=None):
    """Per bead frame (x, y, z, I, bg): z is the known nominal plane height, x and y are shared by the planes of a
    bead (mean over the planes with z_nom in 0.1 - 0.7 um), I and bg per frame."""
    return bead_theta(spl, stack['d'], stack['z_nom'], th, xy_zrange=(0.1, 0.7))


def fit_knots(spl, K0, insitu=None, stack=None, ref=None, lam_ref=0.0, lam_s=0.0, w_stack=1.0, outer=12, inner=8,
              th_ins=None, log=None, loc=None):
    r = fit_free_knots(spl, K0, insitu, stack, ref, lam_ref, lam_s, w_stack, outer, inner, th_ins, log, loc,
                       xy_zrange=(0.1, 0.7))
    return r['knots'], r['theta']


def knot_grad(spl, d, th, nb=4000):
    return knot_gradient(spl, d, th, nb)


def vec_grad_direct(psf, d, th, nb=1000):
    """d NLL / d zcoef with the vectorial model evaluated for every emitter (route V)."""
    g, tot = vectorial_gradient(psf, d, th, C.F0, nb)
    return g['zcoef'], tot


def fit_vec_insitu(spl, d, zc0, iters=80, lr=2e-3, route='K', loc_iters=3, log=None, th=None, loc=None):
    r = fit_zernike_knots(spl, d, C.model_vec(C.P, zc0), C.insitu_pos, iters, lr, route, loc_iters, log, th, loc)
    return r['zcoef'], r['theta']


def fit_pupil_insitu(spl, d, zc0, iters=200, lr_z=1e-3, lr_p=2e-3, lr_a=1e-2, lam=0.0, th=None, log=None, loc=None):
    r = fit_pupil_knots(spl, d, C.model_vec(C.P, zc0), C.insitu_pos, iters, lr_z, lr_p, lr_a, lam, th, log, loc)
    return r['knots'], r, r['theta']


# ------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(spl, knots, test, nll_true=None):
    """Localize the test emitters with the PSF `knots`; errors against the truth and the likelihood deficit."""
    spl.knots = knots
    th, nll = localize(spl, test['d'])
    e = th - test['th']
    z = test['th'][:, 2]
    ok = e[:, 2].abs() < 0.15       # wrong z branch only; lateral bias must stay in the statistics (CRITIC C-02)
    # z error after the best straight line (offset and scale of the z axis are reported separately)
    A = torch.stack([torch.ones_like(z), z], 1)[ok]
    co = torch.linalg.lstsq(A.double(), th[ok, 2:3].double()).solution[:, 0]
    zres = th[:, 2] - (co[0] + co[1] * z)
    out = dict(outlier=float(1 - ok.float().mean()),
               x_rmse=float(e[ok, 0].pow(2).mean().sqrt()) * 108, y_rmse=float(e[ok, 1].pow(2).mean().sqrt()) * 108,
               z_rmse=float(e[ok, 2].pow(2).mean().sqrt()) * 1000, z_rmse_lin=float(zres[ok].pow(2).mean().sqrt()) * 1000,
               z_scale=float(co[1]), z_off=float(co[0]) * 1000,
               x_bias=float(e[ok, 0].mean()) * 108, y_bias=float(e[ok, 1].mean()) * 108,
               I_ratio=float((th[ok, 3] / test['th'][ok, 3]).median()), th=th, ok=ok)
    if nll_true is not None:
        out['dnll'] = float((nll - nll_true).mean())
    out['nll'] = nll
    return out


def psf_rel_err(spl, knots, knots_true):
    """Relative L2 difference of the PSF planes within the emitter z range."""
    zs = spl.zs
    m = (zs >= C.ZR[0]) & (zs <= C.ZR[1])
    return float((knots[m] - knots_true[m]).norm() / knots_true[m].norm())
