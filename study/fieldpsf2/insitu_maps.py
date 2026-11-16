"""Figures and tables of the in-situ field fit (insitu_field.py) against the stage-1 6 x 6 tile fits and
FD-DeepLoc's aber_map (converted to our convention).

    python -I insitu_maps.py FIELD_JSON STAGE1_G6_JSON ABER_MAP OUT_DIR [--tag mixed] [--cache_dir DIR]
Writes OUT_DIR/figs/field_maps_<tag>.png (per mode: the polynomial over the field with the stage-1 tile values as
squares, and the aber_map, one colour scale per mode), field_vs_tiles_<tag>.png (scatter per mode), heldout_<tag>.png
(held-out NLL per spot vs global per 6 x 6 tile) and OUT_DIR/json/field_maps_<tag>.json (correlations).
"""
import argparse
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fieldpsf'))
import numpy as np
from common import THEIR_NOLL, load_aber_map, mode_label, their_to_ours_matrix, setup_path   # noqa: E402

setup_path()
import torch                                         # noqa: E402
from torch_psf.field import FieldZernikes           # noqa: E402


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('field'); ap.add_argument('stage1_g6'); ap.add_argument('aber_map'); ap.add_argument('out')
    ap.add_argument('--tag', default='mixed'); ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--size', type=int, default=1608)
    a = ap.parse_args()
    R = json.load(open(a.field)); S1 = json.load(open(a.stage1_g6))
    fr = R['fit']
    fz, _ = FieldZernikes.from_block(fr['block'])
    base = np.asarray(fr['zcoef'])
    noll = list(THEIR_NOLL)
    tiles = sorted([t for t in S1['tiles'].values() if t['grid'] == 6 and 'zcoef' in t], key=lambda t: (t['iy'], t['ix']))
    tz = np.array([t['zcoef'] for t in tiles]); am = np.array([t['aber_map_mean'] for t in tiles])
    tc = np.array([[(t['box'][0] + t['box'][2]) / 2, (t['box'][1] + t['box'][3]) / 2] for t in tiles])
    # the polynomial at the tile centres (absolute waves)
    col = [noll.index(j) for j in fz.noll]
    def poly_at(x, y):
        out = np.tile(base, (np.size(x), 1)).reshape(np.shape(x) + (len(base),))
        out[..., col] += fz.maps(x, y)
        return out
    pt = poly_at(tc[:, 0], tc[:, 1])
    A = load_aber_map(a.aber_map, cache_dir=a.cache_dir)
    sub = 16
    Asub = np.asarray(A[::sub, ::sub, :21], np.float64) @ their_to_ours_matrix().T       # [n, n, 21] ours
    g = (np.arange(Asub.shape[0]) * sub).astype(float)
    X, Y = np.meshgrid(g, g)
    P = poly_at(X, Y)
    rows = []
    for k, j in enumerate(noll):
        rows.append(dict(noll=j, label=mode_label(j), order=fz.orders[fz.noll.index(j)] if j in fz.noll else 0,
                         base=float(base[k]), r_field_tile=corr(pt[:, k], tz[:, k]), r_field_aber=corr(pt[:, k], am[:, k]),
                         r_tile_aber=corr(tz[:, k], am[:, k]), std_field=float(pt[:, k].std()),
                         std_tile=float(tz[:, k].std()), std_aber=float(am[:, k].std()),
                         rms_field_minus_tile=float(np.sqrt(((pt[:, k] - tz[:, k]) ** 2).mean())),
                         field_range=[float(P[..., k].min()), float(P[..., k].max())]))
    os.makedirs(os.path.join(a.out, 'figs'), exist_ok=True); os.makedirs(os.path.join(a.out, 'json'), exist_ok=True)
    json.dump(dict(modes=rows, tile_centers=tc.tolist(), field_at_tiles=pt.tolist()),
              open(os.path.join(a.out, 'json', f'field_maps_{a.tag}.json'), 'w'), indent=1)
    print('| mode | order | r(field, tiles) | r(field, aber_map) | r(tiles, aber_map) [stage 1] | std field / tiles / map (mλ) | rms field - tiles (mλ) |')
    print('|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['label']} | {r['order']} | {r['r_field_tile']:+.2f} | {r['r_field_aber']:+.2f} | {r['r_tile_aber']:+.2f} | "
              f"{1e3 * r['std_field']:.0f} / {1e3 * r['std_tile']:.0f} / {1e3 * r['std_aber']:.0f} | {1e3 * r['rms_field_minus_tile']:.0f} |")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # ---- maps: per mode, polynomial + tile squares | aber_map
    nc = 3
    nr = int(np.ceil(len(noll) / nc))
    fig, axs = plt.subplots(nr, 2 * nc, figsize=(2.3 * 2 * nc, 2.25 * nr), gridspec_kw=dict(hspace=0.35, wspace=0.08))
    ext = (0, a.size, a.size, 0)
    for k, j in enumerate(noll):
        r_, c_ = k // nc, 2 * (k % nc)
        m_f, m_a = P[..., k].mean(), Asub[..., k].mean()                # each map minus its own field mean
        Pk, Tk, Ak = P[..., k] - m_f, tz[:, k] - m_f, Asub[..., k] - m_a
        lim = 1.15 * np.percentile(np.abs(np.concatenate([Pk.ravel(), Tk])), 99)
        lim = max(lim, 5e-3)
        ax = axs[r_, c_]
        im = ax.imshow(1e3 * Pk, extent=ext, vmin=-1e3 * lim, vmax=1e3 * lim, cmap='RdBu_r')
        ax.scatter(tc[:, 0], tc[:, 1], c=1e3 * Tk, vmin=-1e3 * lim, vmax=1e3 * lim, cmap='RdBu_r', marker='s', s=55,
                   edgecolors='k', linewidths=0.4)
        ax.set_title(f'{mode_label(j)} (mean {1e3 * m_f:.0f})\nfield poly (o{rows[k]["order"]}) + tiles, '
                     f'r {rows[k]["r_field_tile"]:+.2f}', fontsize=6.5)
        ax2 = axs[r_, c_ + 1]
        ax2.imshow(1e3 * Ak, extent=ext, vmin=-1e3 * lim, vmax=1e3 * lim, cmap='RdBu_r')
        ax2.set_title(f'aber_map (mean {1e3 * m_a:.0f}), r(field) {rows[k]["r_field_aber"]:+.2f}', fontsize=6.5)
        cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=5)
        for x in (ax, ax2):
            x.set_xticks([]); x.set_yticks([])
    for k in range(len(noll), nr * nc):
        axs[k // nc, 2 * (k % nc)].axis('off'); axs[k // nc, 2 * (k % nc) + 1].axis('off')
    fig.suptitle('FD-DeepLoc movie, in situ: Zernikes (mλ rms) over the field. Left of each pair: the field polynomial '
                 '(background) and the stage-1 6 x 6 tile fits (squares); right: FD-DeepLoc aber_map; each '
                 'minus its own field mean, one scale per mode (aber_map clipped)', fontsize=8, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.95, bottom=0.02)
    fig.savefig(os.path.join(a.out, 'figs', f'field_maps_{a.tag}.png'), dpi=130)
    plt.close(fig)
    # ---- scatter
    fig, axs = plt.subplots(nr, nc, figsize=(3 * nc, 2.6 * nr))
    for k, j in enumerate(noll):
        ax = axs[k // nc, k % nc]
        ax.scatter(1e3 * tz[:, k], 1e3 * pt[:, k], s=10, label='field poly vs tiles')
        ax.scatter(1e3 * tz[:, k], 1e3 * am[:, k], s=10, marker='x', label='aber_map vs tiles')
        lo = 1e3 * min(tz[:, k].min(), pt[:, k].min(), am[:, k].min()); hi = 1e3 * max(tz[:, k].max(), pt[:, k].max(), am[:, k].max())
        ax.plot([lo, hi], [lo, hi], 'k-', lw=0.5)
        ax.set_title(f'{mode_label(j)}: r {rows[k]["r_field_tile"]:+.2f} / {rows[k]["r_tile_aber"]:+.2f}', fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=5)
    for k in range(len(noll), nr * nc):
        axs[k // nc, k % nc].axis('off')
    fig.supxlabel('stage-1 tile fit (mλ)', fontsize=8); fig.supylabel('field polynomial / aber_map at the tile (mλ)', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'figs', f'field_vs_tiles_{a.tag}.png'), dpi=120)
    plt.close(fig)
    # ---- held-out per tile
    if 'per_tile' in R:
        cands = [k for k in R['heldout'] if k != 'global' and k != 'tile4' and k != 'aber4']
        rows6 = R['per_tile']['g6']
        fig, axs = plt.subplots(1, len(cands), figsize=(2.6 * len(cands), 2.7))
        vmax = max(abs(r[k]) for r in rows6 for k in cands)
        for ax, k in zip(np.atleast_1d(axs), cands):
            M = np.zeros((6, 6))
            for r in rows6:
                M[r['iy'], r['ix']] = r[k]
            im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            for r in rows6:
                ax.text(r['ix'], r['iy'], f'{r[k]:.0f}', ha='center', va='center', fontsize=5.5)
            ax.set_title(f'{k}: mean {R["heldout"][k]["dnll_per_spot"]:+.1f}', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=list(np.atleast_1d(axs)), fraction=0.02)
        fig.suptitle('held-out NLL per spot minus the global in-situ PSF, per 6 x 6 tile (negative = better)', fontsize=8)
        fig.savefig(os.path.join(a.out, 'figs', f'heldout_{a.tag}.png'), dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
