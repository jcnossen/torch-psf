"""How much of the field dependence a low-order 2D polynomial captures, per Zernike mode (input for the library stage).

  - aber_map(s) (converted to our convention, subsampled): field std per mode and the rms residual after a least-squares
    polynomial of total order 0..5 in normalised field coordinates (u, v in [-1, 1]);
  - our bead fits (scattered points): leave-one-out rms prediction error for orders 0..3;
  - our in-situ tile values (insitu json, one grid): leave-one-out rms for orders 0..2.

    python -I field_poly.py OUT_JSON --maps NAME=PATH.mat ... [--beads B.json ...] [--insitu I.json --grid 6]
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np
from common import THEIR_NOLL, fmt_table, load_aber_map, mode_label, their_to_ours_matrix


def design(u, v, order):
    return np.stack([u ** i * v ** (k - i) for k in range(order + 1) for i in range(k + 1)], 1)


def loo_rms(u, v, y, order):
    """leave-one-out rms prediction error of a polynomial fit (hat-matrix shortcut), y [n, m]"""
    X = design(u, v, order)
    if X.shape[1] >= len(u) - 1:
        return np.full(y.shape[1], np.nan)
    H = X @ np.linalg.pinv(X)
    r = y - H @ y
    e = r / (1 - np.diag(H))[:, None]
    return np.sqrt((e ** 2).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--maps', nargs='*', default=[]); ap.add_argument('--beads', nargs='*', default=[])
    ap.add_argument('--insitu', nargs='*', default=[]); ap.add_argument('--grid', type=int, default=6)
    ap.add_argument('--size', type=float, default=1608.0); ap.add_argument('--cache_dir', default=None)
    a = ap.parse_args()
    M = their_to_ours_matrix()
    res = {}
    nrm = lambda x: 2 * np.asarray(x, np.float64) / a.size - 1
    for spec in a.maps:
        name, path = spec.split('=', 1)
        A = load_aber_map(path, cache_dir=a.cache_dir)
        s = 24
        Y = np.asarray(A[s // 2::s, s // 2::s, :21], np.float64)
        yy, xx = np.mgrid[s // 2:A.shape[0]:s, s // 2:A.shape[1]:s]
        Y = Y.reshape(-1, 21) @ M.T
        u, v = nrm(xx.ravel()), nrm(yy.ravel())
        rows = []
        for k, j in enumerate(THEIR_NOLL):
            r = dict(label=mode_label(j), field_std=float(Y[:, k].std()), p2p=float(np.ptp(Y[:, k])))
            for o in range(6):
                X = design(u, v, o)
                c, *_ = np.linalg.lstsq(X, Y[:, k], rcond=None)
                r[f'res_o{o}'] = float(np.sqrt(((Y[:, k] - X @ c) ** 2).mean()))
            rows.append(r)
        res[f'map_{name}'] = rows
        print(f'\n### aber_map "{name}": field std and rms residual (waves) after a polynomial of order n\n')
        print(fmt_table(rows, ['label', 'field_std', 'p2p'] + [f'res_o{o}' for o in range(6)]))
    if a.beads:
        bs = [b for p in a.beads for b in json.load(open(p))['beads']]
        u = nrm([b['x'] for b in bs]); v = nrm([b['y'] for b in bs]); Y = np.array([b['zcoef'] for b in bs])
        L = {o: loo_rms(u, v, Y, o) for o in range(4)}
        rows = [dict(label=mode_label(j), std=float(Y[:, k].std()), **{f'loo_o{o}': float(L[o][k]) for o in L})
                for k, j in enumerate(THEIR_NOLL)]
        res['beads'] = dict(n=len(bs), rows=rows)
        print(f'\n### our bead fits ({len(bs)} beads): std and leave-one-out rms (waves) of a polynomial of order n\n')
        print(fmt_table(rows, ['label', 'std'] + [f'loo_o{o}' for o in L]))
    for p in a.insitu:
        R = json.load(open(p))
        T = [t for t in R['tiles'].values() if t['grid'] == a.grid and 'zcoef' in t]
        cx = [(t['box'][0] + t['box'][2]) / 2 for t in T]; cy = [(t['box'][1] + t['box'][3]) / 2 for t in T]
        u, v = nrm(cx), nrm(cy); Y = np.array([t['zcoef'] for t in T])
        L = {o: loo_rms(u, v, Y, o) for o in range(3)}
        rows = [dict(label=mode_label(j), std=float(Y[:, k].std()), **{f'loo_o{o}': float(L[o][k]) for o in L})
                for k, j in enumerate(THEIR_NOLL)]
        res[f'insitu_g{a.grid}'] = dict(n=len(T), rows=rows)
        print(f'\n### in situ, grid {a.grid} ({len(T)} tiles): std and leave-one-out rms (waves) of a polynomial of order n\n')
        print(fmt_table(rows, ['label', 'std'] + [f'loo_o{o}' for o in L]))
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
