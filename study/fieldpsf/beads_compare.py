"""Compare our per-bead Zernike fits (beads_fit.py) with FD-DeepLoc's: their per-bead fits of the same beads and their
aber_map(s) at the bead positions, after converting theirs to our convention (common.their_to_ours_matrix).

    python -I beads_compare.py OUT_DIR --beads B1.json [B2.json ...] --maps NAME=PATH.mat [...] \
        [--modesets noll5_21=a.json theirs=b.json noll5_37=c.json] [--signneg s20_signneg.json --signpos s20.json]

Writes OUT_DIR/json/beads_compare.json, OUT_DIR/figs/beads_vs_theirs.png and prints markdown tables.
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np
from common import (THEIR_NOLL, compare_modes, fmt_table, load_aber_map, map_at, mode_label, their_to_ours_matrix,
                    theirs_to_ours, zflip_signs)


def load(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        stack = os.path.basename(p).split('_')[1]
        for b in d['beads']:
            b = dict(b, stack=stack)
            rows.append(b)
    return rows, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--beads', nargs='+', required=True)
    ap.add_argument('--maps', nargs='*', default=[])
    ap.add_argument('--modesets', nargs='*', default=[])
    ap.add_argument('--signpos'); ap.add_argument('--signneg')
    ap.add_argument('--px110'); ap.add_argument('--chi2_max', type=float, default=None)
    a = ap.parse_args()
    res = {}
    rows, d0 = load(a.beads)
    assert d0['noll'] == THEIR_NOLL
    ours = np.array([r['zcoef'] for r in rows])
    th_raw = np.array([r['their_coef'] for r in rows])
    xy = np.array([[r['x'], r['y']] for r in rows])
    print(f'{len(rows)} beads from {sorted(set(r["stack"] for r in rows))}')
    # 1) their per-bead fits under the candidate conventions: which mapping explains our numbers
    conv = {}
    for name, kw in (('transpose', dict(transpose=True)), ('no_transpose', dict(transpose=False)),
                     ('transpose_zflip', dict(transpose=True, zflip=True)),
                     ('no_transpose_zflip', dict(transpose=False, zflip=True))):
        th = theirs_to_ours(th_raw, **kw)
        conv[name] = float(np.sqrt(((ours - th) ** 2).mean()))
    print('rms difference ours - theirs (waves) per convention:', {k: round(v, 4) for k, v in conv.items()})
    res['convention_rms'] = conv
    th = theirs_to_ours(th_raw, transpose=True)
    per = compare_modes(ours, th)
    res['vs_their_beads'] = per
    print('\n### ours vs their per-bead fits (same beads), waves rms\n')
    print(fmt_table(per, ['label', 'mean_ours', 'mean_theirs', 'mean_diff', 'rms_diff', 'std_ours', 'std_theirs', 'r']))
    # spatial variation explained: rms of the field variation in either, and the residual
    dv = (ours - ours.mean(0)) - (th - th.mean(0))
    res['field_variation'] = dict(rms_ours=float(np.sqrt(((ours - ours.mean(0)) ** 2).mean())),
                                  rms_theirs=float(np.sqrt(((th - th.mean(0)) ** 2).mean())),
                                  rms_diff_of_variation=float(np.sqrt((dv ** 2).mean())))
    print('field variation (rms over beads and modes, waves):', res['field_variation'])
    # 2) the aber_maps at our bead positions
    for spec in a.maps:
        name, path = spec.split('=', 1)
        A = load_aber_map(path, cache_dir=os.path.join(a.out, 'inspect'))
        mv = theirs_to_ours(np.array([map_at(A, x, y) for x, y in xy]), transpose=True)
        per_m = compare_modes(ours, mv)
        res[f'vs_map_{name}'] = per_m
        print(f'\n### ours vs aber_map "{name}" at the bead positions\n')
        print(fmt_table(per_m, ['label', 'mean_ours', 'mean_theirs', 'mean_diff', 'rms_diff', 'std_ours', 'std_theirs',
                                'r']))
        print('overall rms diff', float(np.sqrt(((ours - mv) ** 2).mean())),
              'after removing the mean offset per mode', float(np.sqrt((((ours - ours.mean(0)) - (mv - mv.mean(0))) ** 2).mean())))
        res[f'vs_map_{name}_overall'] = dict(rms=float(np.sqrt(((ours - mv) ** 2).mean())),
                                             rms_centered=float(np.sqrt((((ours - ours.mean(0)) - (mv - mv.mean(0))) ** 2).mean())))
        res.setdefault('map_values', {})[name] = mv.tolist()
    # 3) mode sets: NLL per bead
    if a.modesets:
        ms = {}
        for spec in a.modesets:
            name, path = spec.split('=', 1)
            d = json.load(open(path))
            ms[name] = dict(n_modes=len(d['noll']), nll={b['index']: b['nll_per_px'] for b in d['beads']},
                            chi2=float(np.median([b['chi2_median'] for b in d['beads']])), npx=d['roisize'] ** 2 * 61)
        names = list(ms)
        ref = names[0]
        print('\n### mode sets (per bead: 2 x NLL difference to', ref, 'over 61 x P^2 px)\n')
        for n in names:
            dl = [2 * (ms[n]['nll'][i] - ms[ref]['nll'][i]) * ms[n]['npx'] for i in ms[ref]['nll']]
            ms[n]['two_dnll_median'] = float(np.median(dl))
            ms[n]['two_dnll_range'] = [float(np.min(dl)), float(np.max(dl))]
            print(f'{n}: {ms[n]["n_modes"]} modes, median chi2 {ms[n]["chi2"]:.3f}, 2 dNLL vs {ref}: median '
                  f'{np.median(dl):.0f} (range {np.min(dl):.0f} .. {np.max(dl):.0f})')
        res['modesets'] = ms
    # 4) stage sign: the flipped scan must give the same NLL and c -> -(-1)^m c
    if a.signpos and a.signneg:
        p, n = json.load(open(a.signpos)), json.load(open(a.signneg))
        cp = np.array([b['zcoef'] for b in p['beads']]); cn = np.array([b['zcoef'] for b in n['beads']])
        lp = np.array([b['nll_per_px'] for b in p['beads']]); ln = np.array([b['nll_per_px'] for b in n['beads']])
        s = zflip_signs()
        res['sign'] = dict(rms_cneg_minus_zflip_cpos=float(np.sqrt(((cn - cp * s) ** 2).mean())),
                           rms_cneg_minus_cpos=float(np.sqrt(((cn - cp) ** 2).mean())),
                           nll_per_px_diff_median=float(np.median(ln - lp)),
                           nll_per_px_diff_max_abs=float(np.abs(ln - lp).max()))
        print('\nstage sign test:', res['sign'])
    if a.px110:
        q = json.load(open(a.px110)); p = json.load(open(a.beads[0]))
        l1 = np.array([b['nll_per_px'] for b in p['beads']]); l2 = np.array([b['nll_per_px'] for b in q['beads']])
        res['pixel_110_vs_108'] = dict(nll_per_px_diff_median=float(np.median(l2 - l1)),
                                       two_dnll_per_bead_median=float(np.median(2 * (l2 - l1) * 61 * p['roisize'] ** 2)),
                                       joint_108=p['joint'], joint_110=q['joint'])
        print('pixel 110 vs 108 nm:', {k: v for k, v in res['pixel_110_vs_108'].items() if 'joint' not in k},
              'joint nll/px', p['joint']['nll_per_px'], q['joint']['nll_per_px'])
    os.makedirs(os.path.join(a.out, 'json'), exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'json', 'beads_compare.json'), 'w'), indent=1)
    # figure: per-mode scatter ours vs theirs (per-bead fits and maps)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 7, figsize=(21, 9.5))
    mapv = {k: np.array(v) for k, v in res.get('map_values', {}).items()}
    for k, ax in enumerate(axs.flat):
        ax.scatter(ours[:, k], th[:, k], s=14, c='#2a6fdb', label='their per-bead fit')
        for (nm, v), c in zip(mapv.items(), ['#e07b00', '#3a9a4a', '#a33']):
            ax.scatter(ours[:, k], v[:, k], s=10, marker='x', c=c, label=f'aber_map {nm}')
        lo = min(ours[:, k].min(), th[:, k].min()); hi = max(ours[:, k].max(), th[:, k].max())
        pad = 0.1 * (hi - lo + 1e-3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#888', lw=0.8)
        ax.set_title(mode_label(THEIR_NOLL[k]), fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6)
    fig.supxlabel('ours (waves rms)'); fig.supylabel('FD-DeepLoc, converted to our convention (waves rms)')
    fig.tight_layout()
    os.makedirs(os.path.join(a.out, 'figs'), exist_ok=True)
    fig.savefig(os.path.join(a.out, 'figs', 'beads_vs_theirs.png'), dpi=110)
    print('figure', os.path.join(a.out, 'figs', 'beads_vs_theirs.png'))


if __name__ == '__main__':
    main()
