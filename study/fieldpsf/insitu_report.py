"""Tables and figures of the in-situ tile fits (insitu_tiles.py) against FD-DeepLoc's aber_map and our bead fits.

    python -I insitu_report.py INSITU_JSON ABER_MAP OUT_DIR --beads B1.json [B2.json ...] [--tag r41]

Writes OUT_DIR/json/insitu_report_<tag>.json, OUT_DIR/figs/zernike_maps_g<grid>_<tag>.png (per mode: their map in our
convention with our tile values and bead fits overlaid), OUT_DIR/figs/heldout_g<grid>_<tag>.png and prints markdown.
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np
from common import THEIR_NOLL, compare_modes, fmt_table, load_aber_map, mode_label, their_to_ours_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('insitu'); ap.add_argument('aber_map'); ap.add_argument('out')
    ap.add_argument('--beads', nargs='*', default=[]); ap.add_argument('--tag', default='')
    ap.add_argument('--min_beads', type=int, default=1)
    a = ap.parse_args()
    R = json.load(open(a.insitu))
    A = load_aber_map(a.aber_map, cache_dir=os.path.join(a.out, 'inspect'))
    M = their_to_ours_matrix()
    beads = []
    for p in a.beads:
        for b in json.load(open(p))['beads']:
            beads.append(b)
    bxy = np.array([[b['x'], b['y']] for b in beads]) if beads else np.zeros((0, 2))
    bz = np.array([b['zcoef'] for b in beads]) if beads else np.zeros((0, 21))
    rep = dict(global_zcoef=R['global']['zcoef'], global_heldout=R['global'].get('heldout'))
    tag = ('_' + a.tag) if a.tag else ''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(a.out, 'figs'), exist_ok=True)
    grids = sorted({t['grid'] for t in R['tiles'].values()})
    for g in grids:
        T = [t for t in R['tiles'].values() if t['grid'] == g]
        ok = [t for t in T if 'zcoef' in t]
        ours = np.array([t['zcoef'] for t in ok]); mp = np.array([t['aber_map_mean'] for t in ok])
        rows = []
        for t in T:
            r = dict(tile=f'({t["ix"]},{t["iy"]})', train=t['n_train_avail'], test=t['n_test_avail'])
            if 'zcoef' in t:
                ho = t['heldout']
                r.update(used=t['report']['n_train'], rejected=t['report']['n_rejected'], n_test=t['n_test'],
                         tile=f'({t["ix"]},{t["iy"]})')
                for k in ('tile', 'aber_map', 'beads'):
                    r[k + '_dnll'] = f'{ho[k]["dnll_per_spot_vs_global"]:+.2f} ± {ho[k]["dnll_se"]:.2f}'
                r['tile_better_frac'] = f'{ho["tile"]["frac_spots_better"]:.2f}'
                r['at_limit'] = f'{ho["tile"]["at_limit"]:.3f}'
            else:
                r.update(used='-', rejected='-', n_test='-', tile_dnll='skipped', aber_map_dnll='-', beads_dnll='-',
                         tile_better_frac='-', at_limit='-')
            rows.append(r)
        print(f'\n### grid {g} x {g}: held-out NLL per spot relative to the global in-situ PSF (negative = better)\n')
        print(fmt_table(rows, ['tile', 'train', 'test', 'used', 'rejected', 'n_test', 'tile_dnll', 'aber_map_dnll',
                               'beads_dnll', 'tile_better_frac', 'at_limit']))
        # summary over tiles
        d = {k: np.array([t['heldout'][k]['dnll_per_spot_vs_global'] for t in ok]) for k in ('tile', 'aber_map', 'beads')}
        tot = {k: float(sum(t['heldout'][k]['dnll_total_vs_global'] for t in ok)) for k in d}
        nte = int(sum(t['n_test'] for t in ok))
        summ = dict(n_tiles=len(ok), n_test=nte, mean_dnll_per_spot={k: float(v.mean()) for k, v in d.items()},
                    total_dnll={k: tot[k] for k in tot}, tiles_where_tile_beats_global=int((d['tile'] < 0).sum()),
                    tiles_where_tile_beats_map=int((d['tile'] < d['aber_map']).sum()))
        print('summary', json.dumps(summ))
        per = compare_modes(ours, mp)
        print(f'\n### grid {g}: tile Zernikes (ours, in situ) vs aber_map tile mean (converted), over {len(ok)} tiles\n')
        print(fmt_table(per, ['label', 'mean_ours', 'mean_theirs', 'mean_diff', 'rms_diff', 'std_ours', 'std_theirs', 'r', 'slope']))
        # vs bead fits inside the tiles
        bt = []
        if len(bxy):
            for t in ok:
                x0, y0, x1, y1 = t['box']
                m = (bxy[:, 0] >= x0) & (bxy[:, 0] < x1) & (bxy[:, 1] >= y0) & (bxy[:, 1] < y1)
                if m.sum() >= a.min_beads:
                    bt.append((np.array(t['zcoef']), bz[m].mean(0), np.array(t['aber_map_mean']), int(m.sum())))
        per_b = per_bm = None
        if len(bt) >= 3:
            o, b_, m_ = (np.array([q[i] for q in bt]) for i in range(3))
            per_b = compare_modes(o, b_)
            per_bm = compare_modes(b_, m_)
            print(f'\n### grid {g}: tile Zernikes (in situ) vs our bead fits in the tile ({len(bt)} tiles with beads)\n')
            print(fmt_table(per_b, ['label', 'mean_ours', 'mean_theirs', 'mean_diff', 'rms_diff', 'std_ours', 'std_theirs', 'r']))
        rep[f'grid{g}'] = dict(rows=rows, summary=summ, vs_map=per, vs_beads=per_b, beads_vs_map=per_bm,
                               n_tiles_with_beads=len(bt))
        # figure 1: per mode, their map (converted) with our tile values as squares and bead fits as dots
        nm = 21
        fig, axs = plt.subplots(3, 7, figsize=(24, 11))
        step = 8
        Ad = np.asarray(A[::step, ::step, :21], np.float64) @ M.T        # converted map, downsampled
        for k, ax in enumerate(axs.flat[:nm]):
            vals = np.concatenate([Ad[..., k].ravel(), ours[:, k], bz[:, k] if len(bz) else []])
            lim = np.percentile(np.abs(vals - np.median(vals)), 99) + 1e-4
            vmin, vmax = np.median(vals) - lim, np.median(vals) + lim
            ax.imshow(Ad[..., k], extent=(0, 1608, 1608, 0), cmap='RdBu_r', vmin=vmin, vmax=vmax)
            for t in ok:
                x0, y0, x1, y1 = t['box']; w = (x1 - x0) * 0.42
                cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
                ax.add_patch(plt.Rectangle((cxm - w / 2, cym - w / 2), w, w, facecolor=plt.cm.RdBu_r((t['zcoef'][k] - vmin) / (vmax - vmin)),
                                           edgecolor='k', lw=0.6))
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#555', lw=0.3))
            if len(bxy):
                ax.scatter(bxy[:, 0], bxy[:, 1], c=bz[:, k], cmap='RdBu_r', vmin=vmin, vmax=vmax, s=16, edgecolors='k', linewidths=0.4)
            ax.set_title(f'{mode_label(THEIR_NOLL[k])}\n[{vmin:+.3f}, {vmax:+.3f}] waves', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'Background: FD-DeepLoc aber_map (our convention). Squares: our in-situ fit per tile ({g}x{g}). '
                     f'Dots: our bead fits. Same colour scale per mode (RdBu, waves rms).', fontsize=11)
        fig.tight_layout()
        fp = os.path.join(a.out, 'figs', f'zernike_maps_g{g}{tag}.png'); fig.savefig(fp, dpi=100); plt.close(fig)
        print('figure', fp)
        # figure 2: per-mode scatter tile vs map tile mean
        fig, axs = plt.subplots(3, 7, figsize=(21, 9))
        for k, ax in enumerate(axs.flat):
            ax.scatter(mp[:, k], ours[:, k], s=14, c='#2a6fdb')
            lo, hi = min(mp[:, k].min(), ours[:, k].min()), max(mp[:, k].max(), ours[:, k].max())
            ax.plot([lo, hi], [lo, hi], color='#888', lw=0.8)
            ax.set_title(f'{mode_label(THEIR_NOLL[k])}  r={per[k]["r"]:.2f}', fontsize=8); ax.tick_params(labelsize=7)
        fig.supxlabel('aber_map tile mean (waves rms, our convention)'); fig.supylabel('in situ, per tile (waves rms)')
        fig.tight_layout()
        fp = os.path.join(a.out, 'figs', f'tiles_vs_map_g{g}{tag}.png'); fig.savefig(fp, dpi=100); plt.close(fig)
        # figure 3: held-out dNLL per tile
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, k in zip(axs, ('tile', 'aber_map', 'beads')):
            im = np.full((g, g), np.nan)
            for t in ok:
                im[t['iy'], t['ix']] = t['heldout'][k]['dnll_per_spot_vs_global']
            lim = np.nanmax(np.abs(im))
            h = ax.imshow(im, cmap='RdBu_r', vmin=-lim, vmax=lim)
            for t in ok:
                ax.text(t['ix'], t['iy'], f'{im[t["iy"], t["ix"]]:+.1f}', ha='center', va='center', fontsize=8)
            ax.set_title(f'{k} - global: held-out NLL per spot')
            plt.colorbar(h, ax=ax, fraction=0.046)
        fig.tight_layout()
        fp = os.path.join(a.out, 'figs', f'heldout_g{g}{tag}.png'); fig.savefig(fp, dpi=100); plt.close(fig)
    os.makedirs(os.path.join(a.out, 'json'), exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, 'json', f'insitu_report{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
