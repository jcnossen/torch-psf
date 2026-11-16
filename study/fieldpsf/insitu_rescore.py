"""Control for the tile fits of insitu_tiles.py: is the tile's gain over the global PSF field dependence, or only the
extra iterations / spots of the tile fit?

'control' = the global PSF refitted like a tile (same number of training spots and iterations, started from the
global fit) but with training spots drawn from the whole field. Every tile's held-out spots (parity 1, a fresh
subsample) are then scored with global, control, tile (zcoef from the tile json) and aber_map (tile mean). If the tile
beats the control, the gain is spatial.

    python -I insitu_rescore.py SPOTS_NPZ ABER_MAP OUT_JSON --insitu I1.json [I2.json ...] [--control_n 4000]
"""
import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np
import torch
from common import load_aber_map
from insitu_tiles import Fitter, heldout, split_checkerboard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spots'); ap.add_argument('aber_map'); ap.add_argument('out')
    ap.add_argument('--insitu', nargs='+', required=True)
    ap.add_argument('--control_n', type=int, default=4000); ap.add_argument('--control_iters', type=int, default=40)
    ap.add_argument('--ntest', type=int, default=3000); ap.add_argument('--readnoise_e', type=float, default=1.6)
    ap.add_argument('--cache_dir', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda')
    t0 = time.time()
    runs = [json.load(open(p)) for p in a.insitu]
    args = argparse.Namespace(**runs[0]['args'])
    S = np.load(a.spots)
    h = int(S['roisize']) // 2
    cx = S['corners'][:, 0] + h; cy = S['corners'][:, 1] + h
    rois = (torch.as_tensor(S['rois'].astype(np.float32), device=dev) - float(S['offset'])) * float(S['gain'])
    var = torch.tensor(a.readnoise_e ** 2, device=dev)
    par = split_checkerboard(cx, cy, args.checker)
    F = Fitter(args, dev)
    v_glob = F.vec(runs[0]['global']['zcoef'])
    rng = np.random.default_rng(123)
    tr = np.sort(rng.choice(np.flatnonzero(par == 0), a.control_n, replace=False))
    v_ctl, rep = F.fit(v_glob, rois[torch.as_tensor(tr, device=dev)], var, a.control_iters)
    print(f'control fit: {rep} ({time.time() - t0:.0f} s)', flush=True)
    res = dict(control=dict(zcoef=v_ctl.zcoef.detach().cpu().tolist(), report=rep, n=a.control_n,
                            iters=a.control_iters), tiles={})
    for R in runs:
        for key, t in R['tiles'].items():
            if 'zcoef' not in t:
                continue
            x0, y0, x1, y1 = t['box']
            inn = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1) & (par == 1)
            i = np.flatnonzero(inn)
            te = np.sort(rng.choice(i, a.ntest, replace=False)) if len(i) > a.ntest else i
            cands = {'global': v_glob, 'control': v_ctl, 'tile': F.vec(t['zcoef']), 'aber_map': F.vec(t['aber_map_mean'])}
            ho, nl = heldout(F, cands, rois[torch.as_tensor(te, device=dev)], var, raw=True)
            d = {k: ho[k]['dnll_per_spot_vs_global'] - ho['control']['dnll_per_spot_vs_global'] for k in ho}
            dd = nl['tile'] - nl['control']
            d['tile_se'] = float(dd.std(ddof=1) / np.sqrt(len(dd)))
            d['tile_frac_better'] = float((dd < 0).mean())
            res['tiles'][key] = dict(grid=t['grid'], ix=t['ix'], iy=t['iy'], box=t['box'], n_test=int(len(te)), heldout=ho,
                                     dnll_vs_control=d)
            print(f'{key}: n_test {len(te)}; vs global: ' + ', '.join(
                f'{k} {v["dnll_per_spot_vs_global"]:+.2f}+-{v["dnll_se"]:.2f}' for k, v in ho.items())
                  + f'; tile - control {d["tile"]:+.2f}+-{d["tile_se"]:.2f} ({time.time() - t0:.0f} s)', flush=True)
            json.dump(res, open(a.out, 'w'), indent=1)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
