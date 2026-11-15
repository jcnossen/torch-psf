"""In-situ Zernike fits per field tile of the FD-DeepLoc TOM20 tetrapod movie, against one global in-situ PSF.

Spots: detect_tetrapod.py (matched filter; isolated spots, ROIs centred on the emitters). The in-situ steps are those of
torch_psf.InSituEstimator, called here through the low-level functions because its built-in peak detection finds
tetrapod lobes, not emitters: fit_zernike_knots (knot route), overlap rejection by spot_deviance (median + k robust
scale), a refit, and held-out NLL. The localizer's first, multi-start z search uses 9 starts instead of 5 over the
6 um range (a wrapper set here; 5 starts leave ~2-5 % of tetrapod spots in a wrong z minimum, detect_check.py).

Split: a checkerboard of 32 px squares in absolute image coordinates (parity 0 = training, 1 = held out), the same
for the global fit and every tile, so no held-out spot of a tile was used by the global fit either.
Per tile, candidates scored on the tile's held-out spots (per-spot NLL after a full refit of each spot):
  global   the global in-situ PSF (the start of the tile fit)
  tile     the tile fit
  aber_map FD-DeepLoc's aber_map averaged over the tile, converted to our convention (common.their_to_ours_matrix)
  beads    the bead-mean start PSF

    python -I insitu_tiles.py SPOTS_NPZ START_JSON ABER_MAP OUT_JSON [--grids 4 6] [--pixel_um 0.110]
START_JSON: {"zcoef": [21 waves, Noll THEIR_NOLL order], "optics": {...}} (insitu_start.py).
"""
import argparse
import dataclasses
import functools
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np
import torch
from common import OPTICS, THEIR_NOLL, load_aber_map, setup_path, theirs_to_ours

setup_path()
import torch_psf.estimation.knots as _knots              # noqa: E402
import torch_psf.estimation.heldout as _heldout          # noqa: E402
from torch_psf.fit import localize as _localize          # noqa: E402
from torch_psf.vectorial import VectorialPSF             # noqa: E402
from torch_psf.spline import CubicSplinePSF              # noqa: E402
from torch_psf.estimation import fit_zernike_knots, spot_deviance, split_checkerboard   # noqa: E402

ZSTARTS = 9
_knots.localize = functools.partial(_localize, zstarts=ZSTARTS)
_heldout.localize = functools.partial(_localize, zstarts=ZSTARTS)


def per_spot_nll(spline, rois, var):
    """NLL of every ROI after its own MLE fit (multi-start z) [N] (numpy), and the at-limit fraction"""
    with torch.no_grad():
        th, nll = _localize(spline, rois, var=var, zstarts=ZSTARTS)
        lim = float(_knots.at_limit(th, spline).float().mean())
    return nll.cpu().numpy().astype(np.float64), lim, th


class Fitter:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        self.grid = dict(oversample=2, dz_um=a.dz_um, zpad=5, xy_limit=a.xy_limit)
        self.zr = (0.0, 2 * a.focus)

    def vec(self, zcoef):
        o = dataclasses.replace(OPTICS, pixel_um=self.a.pixel_um, wavelength_um=self.a.wavelength_um)
        v = VectorialPSF.from_optics(o, self.a.roisize, zernike_indices=tuple(THEIR_NOLL), npupil=64, oversample=3,
                                     device=self.dev, focus_um=self.a.focus, z_range=self.zr)
        return v.set_zernike(list(map(float, zcoef)))

    def spline(self, v):
        return v.to_spline(self.a.roisize, self.zr, **self.grid)

    def fit(self, v0, rois, var, iters):
        """fit_zernike_knots, overlap rejection, refit; returns (vectorial, report)"""
        spl = CubicSplinePSF(self.a.roisize, self.zr, device=self.dev, **self.grid)
        r = fit_zernike_knots(spl, rois, v0, iters=iters, lr=self.a.lr, var=var)
        v = r['psf']
        dv, _ = spot_deviance(self.spline(v), rois, var)
        med = float(np.median(dv)); mad = 1.4826 * float(np.median(np.abs(dv - med)))
        thr = med + self.a.deviance_k * max(mad, med * np.sqrt(2.0 / self.a.roisize ** 2))
        keep = torch.as_tensor(dv < thr, device=self.dev)
        r2 = fit_zernike_knots(spl, rois[keep], v, iters=max(iters * 3 // 4, 1), lr=self.a.lr, var=var)
        rep = dict(n_train=int(len(rois)), n_rejected=int((~keep).sum()), deviance_median=med, deviance_threshold=thr,
                   at_limit=r2['report']['at_limit'], nll_first=r['history'][0]['nll'] if r['history'] else None,
                   nll_last=r2['history'][-1]['nll'] if r2['history'] else None)
        return r2['psf'], rep


def heldout(fitter, cands, rois, var):
    """per-spot held-out NLL of each candidate (dict name -> VectorialPSF); differences to 'global' per spot"""
    out, nl = {}, {}
    P2 = rois[0].numel()
    for k, v in cands.items():
        nll, lim, th = per_spot_nll(fitter.spline(v), rois, var)
        nl[k] = nll
        out[k] = dict(nll_per_px=float(nll.mean() / P2), at_limit=lim)
    for k in cands:
        d = nl[k] - nl['global']
        out[k].update(dnll_per_spot_vs_global=float(d.mean()), dnll_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                      dnll_total_vs_global=float(d.sum()), frac_spots_better=float((d < 0).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spots'); ap.add_argument('start'); ap.add_argument('aber_map'); ap.add_argument('out')
    ap.add_argument('--grids', type=int, nargs='*', default=[4, 6])
    ap.add_argument('--tiles', type=int, nargs='*', default=None, help='only these tile indices (row-major)')
    ap.add_argument('--size', type=int, default=1608)
    ap.add_argument('--pixel_um', type=float, default=0.110); ap.add_argument('--wavelength_um', type=float, default=0.68)
    ap.add_argument('--focus', type=float, default=3.0)
    ap.add_argument('--readnoise_e', type=float, default=1.6)
    ap.add_argument('--dz_um', type=float, default=0.04); ap.add_argument('--xy_limit', type=float, default=3.0)
    ap.add_argument('--iters', type=int, default=80); ap.add_argument('--tile_iters', type=int, default=60)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--deviance_k', type=float, default=3.0)
    ap.add_argument('--nglobal', type=int, default=12000); ap.add_argument('--nglobal_test', type=int, default=6000)
    ap.add_argument('--ntile', type=int, default=8000); ap.add_argument('--ntile_test', type=int, default=4000)
    ap.add_argument('--checker', type=int, default=32)
    ap.add_argument('--global_json', default=None, help='reuse the global fit of an earlier run (its out json)')
    ap.add_argument('--cache_dir', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda')
    t0 = time.time()
    S = np.load(a.spots)
    a.roisize = int(S['roisize'])
    h = a.roisize // 2
    cx = S['corners'][:, 0] + h; cy = S['corners'][:, 1] + h
    rois = (torch.as_tensor(S['rois'].astype(np.float32), device=dev) - float(S['offset'])) * float(S['gain'])
    var = torch.tensor(a.readnoise_e ** 2, device=dev)
    par = split_checkerboard(cx, cy, a.checker)
    rng = np.random.default_rng(0)
    st = json.load(open(a.start))
    F = Fitter(a, dev)
    v_beads = F.vec(st['zcoef'])
    A = load_aber_map(a.aber_map, cache_dir=a.cache_dir)
    res = dict(args=vars(a), n_spots=int(len(cx)), start=st, tiles={})
    print(f'{len(cx)} spots, roisize {a.roisize} ({time.time() - t0:.0f} s)', flush=True)

    def pick(mask, n):
        i = np.flatnonzero(mask)
        return np.sort(rng.choice(i, n, replace=False)) if len(i) > n else i

    if a.global_json:
        g = json.load(open(a.global_json))['global']
        v_glob = F.vec(g['zcoef']); res['global'] = g
    else:
        tr, te = pick(par == 0, a.nglobal), pick(par == 1, a.nglobal_test)
        v_glob, rep = F.fit(v_beads, rois[torch.as_tensor(tr, device=dev)], var, a.iters)
        ho = heldout(F, {'global': v_glob, 'beads': v_beads}, rois[torch.as_tensor(te, device=dev)], var)
        res['global'] = dict(zcoef=v_glob.zcoef.detach().cpu().tolist(), report=rep, heldout=ho, n_test=len(te))
        print(f'global: {rep} heldout {ho} ({time.time() - t0:.0f} s)', flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
    for g in a.grids:
        T = a.size / g
        tx, ty = np.clip((cx // T).astype(int), 0, g - 1), np.clip((cy // T).astype(int), 0, g - 1)
        for iy in range(g):
            for ix in range(g):
                ti = iy * g + ix
                if a.tiles is not None and ti not in a.tiles:
                    continue
                inn = (tx == ix) & (ty == iy)
                tr, te = pick(inn & (par == 0), a.ntile), pick(inn & (par == 1), a.ntile_test)
                y0, y1, x0, x1 = int(round(iy * T)), int(round((iy + 1) * T)), int(round(ix * T)), int(round((ix + 1) * T))
                mv = theirs_to_ours(np.asarray(A[y0:y1, x0:x1, :21], np.float64).reshape(-1, 21).mean(0))
                rec = dict(grid=g, ix=ix, iy=iy, box=[x0, y0, x1, y1], n_train_avail=int((inn & (par == 0)).sum()),
                           n_test_avail=int((inn & (par == 1)).sum()), aber_map_mean=mv.tolist())
                if len(tr) < 200 or len(te) < 100:
                    rec['skipped'] = 'too few spots'
                    res['tiles'][f'g{g}_{ti}'] = rec
                    print(f'grid {g} tile {ti}: too few spots ({len(tr)} / {len(te)})', flush=True)
                    continue
                v_t, rep = F.fit(v_glob, rois[torch.as_tensor(tr, device=dev)], var, a.tile_iters)
                cands = {'global': v_glob, 'tile': v_t, 'aber_map': F.vec(mv), 'beads': v_beads}
                ho = heldout(F, cands, rois[torch.as_tensor(te, device=dev)], var)
                rec.update(zcoef=v_t.zcoef.detach().cpu().tolist(), report=rep, heldout=ho, n_test=int(len(te)))
                res['tiles'][f'g{g}_{ti}'] = rec
                print(f'grid {g} tile {ti} ({ix},{iy}): train {len(tr)} (rej {rep["n_rejected"]}) test {len(te)}; '
                      + ', '.join(f'{k} {v["dnll_per_spot_vs_global"]:+.2f}+-{v["dnll_se"]:.2f}' for k, v in ho.items())
                      + f' ({time.time() - t0:.0f} s)', flush=True)
                json.dump(res, open(a.out, 'w'), indent=1)
    res['seconds'] = time.time() - t0
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
