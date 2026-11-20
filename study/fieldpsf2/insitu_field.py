"""Field-dependent in-situ PSF on the FD-DeepLoc TOM20 tetrapod movie (stage 2a): the base Zernikes and a polynomial
of the field position per mode (torch_psf.field.FieldZernikes) fitted jointly by the knot route per field tile
(torch_psf.estimation.fit_field_knots), started from the stage-1 global in-situ fit, and scored on held-out spots
against the stage-1 candidates with the stage-1 scoring (per-spot NLL after a full refit of every held-out spot,
9 z starts; differences to the global PSF per spot).

Spots, split, optics, camera: as stage 1 (study/fieldpsf/insitu_tiles.py): the matched-filter ROIs (41 px), a
checkerboard of 32 px squares of the spot positions (parity 0 training, 1 held out; the stage-1 global and tile fits
used parity-0 spots only), pixel 0.110 um, gain 0.818, offset 100, read noise 1.6 e-, focus 3 um, z 0..6 um, 21 modes.
Training spots: a random subset of the parity-0 spots of the whole field, overlaps rejected with the global PSF
(deviance > median + 3 robust scales, as stage 1).

Candidates on the held-out spots:
  global        the stage-1 global in-situ PSF (the start)
  tile4 / tile6 the stage-1 per-tile PSFs (4 x 4 / 6 x 6 tiles), each spot with the PSF of its tile
  aber4 / aber6 FD-DeepLoc's aber_map averaged over the stage-1 tiles (converted to our convention)
  field<T>      the field model as a tiled spline with T px tiles (TiledSplinePSF; T = 268 is the 6 x 6 layout)

    python -I insitu_field.py SPOTS_NPZ STAGE1_G4_JSON STAGE1_G6_JSON OUT_JSON [--orders mixed] [--ntrain 40000]
                              [--tile 268] [--iters 60] [--score_tiles 268 134] [--ntest 1500] [--fit_json F]
"""
import argparse
import json
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fieldpsf'))
import numpy as np
import torch
from common import OPTICS, THEIR_NOLL, setup_path   # noqa: E402  (study/fieldpsf/common.py)

setup_path()
import dataclasses                                                    # noqa: E402
from torch_psf.vectorial import VectorialPSF                          # noqa: E402
from torch_psf.spline import SplinePSF                           # noqa: E402
from torch_psf.field import FieldZernikes, FieldPSF                   # noqa: E402
from torch_psf.fit import localize                                    # noqa: E402
from torch_psf.estimation import fit_field_knots, spot_deviance, split_checkerboard   # noqa: E402

ZSTARTS = 9
SECOND = (5, 6, 11, 12, 13, 22, 23, 24)            # astigmatism / spherical families: quadratic (stage 1, section 6)
ORDERS = dict(mixed={j: (2 if j in SECOND else 1) for j in THEIR_NOLL}, linear={j: 1 for j in THEIR_NOLL},
              quad={j: 2 for j in THEIR_NOLL})


class Models:
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


@torch.no_grad()
def nll_groups(models, groups, rois, var):
    """per-spot NLL [N] (numpy) after a full refit (9 z starts), every group of spots with its own spline"""
    out = np.full(len(rois), np.nan)
    for spl, idx in groups:
        it = torch.as_tensor(idx, device=rois.device)
        _, nll = localize(spl, rois[it], var=var, zstarts=ZSTARTS)
        out[idx] = nll.double().cpu().numpy()
    return out


def tile_of(cx, cy, g, size):
    T = size / g
    return np.clip((cy // T).astype(int), 0, g - 1) * g + np.clip((cx // T).astype(int), 0, g - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spots'); ap.add_argument('stage1_g4'); ap.add_argument('stage1_g6'); ap.add_argument('out')
    ap.add_argument('--orders', default='mixed', choices=sorted(ORDERS))
    ap.add_argument('--prior_sd', type=float, default=0.05)
    ap.add_argument('--ntrain', type=int, default=40000)
    ap.add_argument('--tile', type=float, default=268.0)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--score_tiles', type=float, nargs='*', default=[268.0, 134.0])
    ap.add_argument('--ntest', type=int, default=1500, help='held-out spots per 6 x 6 tile')
    ap.add_argument('--fit_json', default=None, help='reuse the field fit of an earlier run (its out json)')
    ap.add_argument('--size', type=int, default=1608)
    ap.add_argument('--pixel_um', type=float, default=0.110); ap.add_argument('--wavelength_um', type=float, default=0.68)
    ap.add_argument('--focus', type=float, default=3.0); ap.add_argument('--readnoise_e', type=float, default=1.6)
    ap.add_argument('--dz_um', type=float, default=0.04); ap.add_argument('--xy_limit', type=float, default=3.0)
    ap.add_argument('--deviance_k', type=float, default=3.0); ap.add_argument('--checker', type=int, default=32)
    a = ap.parse_args()
    dev = torch.device('cuda')
    t0 = time.time()
    S = np.load(a.spots)
    a.roisize = int(S['roisize'])
    h = a.roisize // 2
    cx = (S['corners'][:, 0] + h).astype(np.float64); cy = (S['corners'][:, 1] + h).astype(np.float64)
    rois = (torch.as_tensor(S['rois'].astype(np.float32), device=dev) - float(S['offset'])) * float(S['gain'])
    var = torch.tensor(a.readnoise_e ** 2, device=dev)
    par = split_checkerboard(cx, cy, a.checker)
    g4, g6 = json.load(open(a.stage1_g4)), json.load(open(a.stage1_g6))
    M = Models(a, dev)
    v_glob = M.vec(g4['global']['zcoef'])
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res.update(args=vars(a), n_spots=int(len(cx)))
    xy_all = torch.as_tensor(np.stack([cx, cy], 1), dtype=torch.float32, device=dev)
    print(f'{len(cx)} spots ({time.time() - t0:.0f} s)', flush=True)

    # ---- the field fit
    if a.fit_json:
        fr = json.load(open(a.fit_json))['fit']
        res['fit'] = fr
    else:
        rng = np.random.default_rng(1)
        i0 = np.flatnonzero(par == 0)
        tr = np.sort(rng.choice(i0, min(a.ntrain, len(i0)), replace=False))
        itr = torch.as_tensor(tr, device=dev)
        dv, _ = spot_deviance(M.spline(v_glob), rois[itr], var, zstarts=ZSTARTS)
        med = float(np.median(dv)); mad = 1.4826 * float(np.median(np.abs(dv - med)))
        thr = med + a.deviance_k * max(mad, med * np.sqrt(2.0 / a.roisize ** 2))
        tr = tr[dv < thr]
        itr = torch.as_tensor(tr, device=dev)
        print(f'training {len(tr)} spots ({int((dv >= thr).sum())} rejected; {time.time() - t0:.0f} s)', flush=True)
        fz = FieldZernikes.from_positions(THEIR_NOLL, ORDERS[a.orders], xy_all[itr], prior_sd=a.prior_sd, device=dev)
        spl = SplinePSF(a.roisize, M.zr, device=dev, **M.grid)

        def log(it, nll, sec, zc, coef):
            print(f'  it {it} nll {nll:.1f} ({sec:.0f} s) |coef| max {float(coef.abs().max()):.4f}', flush=True)
        r = fit_field_knots(spl, rois[itr], xy_all[itr], v_glob, fz, tile=a.tile, extent=(0, 0, a.size, a.size),
                            iters=a.iters, lr=a.lr, zstarts=ZSTARTS, var=var, log=log)
        fp = r['psf']
        fr = dict(report=r['report'], history=r['history'], block=fp.field_block(),
                  zcoef=fp.base.zcoef.detach().cpu().tolist(), orders=fz.orders, noll=fz.noll, n_train=int(len(tr)),
                  n_rejected=int((dv >= thr).sum()), tile_counts=r['tile_counts'], centers=r['centers'].tolist())
        res['fit'] = fr
        json.dump(res, open(a.out, 'w'), indent=1)
        print(f'field fit: {r["report"]} ({time.time() - t0:.0f} s)', flush=True)
    fz, _ = FieldZernikes.from_block(fr['block'], device=dev)
    fp = FieldPSF(M.vec(fr['zcoef']), fz)

    # ---- held-out spots: up to ntest per 6 x 6 tile
    rng = np.random.default_rng(2)
    t6, t4 = tile_of(cx, cy, 6, a.size), tile_of(cx, cy, 4, a.size)
    te = np.sort(np.concatenate([(lambda i: rng.choice(i, min(a.ntest, len(i)), replace=False))(
        np.flatnonzero((par == 1) & (t6 == t))) for t in range(36)]))
    ite = torch.as_tensor(te, device=dev)
    rte = rois[ite]
    print(f'{len(te)} held-out spots ({time.time() - t0:.0f} s)', flush=True)
    nll = {}
    nll['global'] = nll_groups(M, [(M.spline(v_glob), np.arange(len(te)))], rte, var)
    for name, js, g, tt in (('tile4', g4, 4, t4), ('tile6', g6, 6, t6)):
        tiles = {k: v for k, v in js['tiles'].items() if v['grid'] == g}
        for kind in ('tile', 'aber'):
            groups = []
            for k, v in tiles.items():
                ti = v['iy'] * g + v['ix']
                idx = np.flatnonzero(tt[te] == ti)
                if not len(idx):
                    continue
                z = v['zcoef'] if kind == 'tile' else v['aber_map_mean']     # stage 1 stored it converted
                groups.append((M.spline(M.vec(z)), idx))
            nm = name if kind == 'tile' else name.replace('tile', 'aber')
            nll[nm] = nll_groups(M, groups, rte, var)
            print(f'{nm}: {np.nanmean(nll[nm] - nll["global"]):+.2f} per spot ({time.time() - t0:.0f} s)', flush=True)
    for T in a.score_tiles:
        tp = fp.to_tiled(tile=T, extent=(0, 0, a.size, a.size), roisize=a.roisize, z_range=M.zr, **M.grid)
        ti = tp.tile_index(xy_all[ite]).cpu().numpy()
        groups = [(tp.spline(int(t)), np.flatnonzero(ti == t)) for t in np.unique(ti)]
        nm = f'field{int(T)}'
        nll[nm] = nll_groups(M, groups, rte, var)
        print(f'{nm}: {np.nanmean(nll[nm] - nll["global"]):+.2f} per spot ({time.time() - t0:.0f} s)', flush=True)
        del tp, groups
        torch.cuda.empty_cache()
    np.savez(os.path.splitext(a.out)[0] + '_nll.npz', te=te, cx=cx[te], cy=cy[te], **nll)
    # ---- summaries: overall and per stage-1 tile
    summ = {}
    for k, v in nll.items():
        d = v - nll['global']
        summ[k] = dict(dnll_per_spot=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                       frac_better=float((d < 0).mean()), nll_per_px=float(v.mean() / a.roisize ** 2))
    per = {}
    for g, tt in ((4, t4), (6, t6)):
        rows = []
        for ti in range(g * g):
            m = tt[te] == ti
            row = dict(tile=ti, ix=ti % g, iy=ti // g, n=int(m.sum()))
            for k, v in nll.items():
                d = v[m] - nll['global'][m]
                row[k] = float(d.mean()); row[k + '_se'] = float(d.std(ddof=1) / np.sqrt(max(m.sum(), 2)))
            rows.append(row)
        per[f'g{g}'] = rows
    res.update(n_test=int(len(te)), heldout=summ, per_tile=per, seconds=time.time() - t0)
    json.dump(res, open(a.out, 'w'), indent=1)
    for k, v in summ.items():
        print(f'{k:10s} {v["dnll_per_spot"]:+8.2f} +- {v["se"]:.2f}  better {v["frac_better"]:.2f}')
    print(f'done ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
