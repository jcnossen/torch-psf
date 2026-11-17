"""Tables and the classifier-statistics overview from the outputs of run_real.py / run_synth.py (~/autopsf_out).
Writes real_table.md, synth_table.md and figs/classifier_stats.png."""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import SETS                                       # noqa: E402

OUT = os.path.expanduser('~/autopsf_out')
COL = dict(smlm='tab:blue', sofi='tab:orange', zstack='tab:green', static='tab:purple', no_signal='0.5')


def fmt(v, n=3):
    return '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{n}f}'


def real_table():
    rows, recs = [], []
    hdr = ('| data set | pixel (um) | NA nominal | lambda (um) | expected f_c (c/px): 2NA/lambda; 2 min(NA, n_med)/lambda '
           '| fitted f_c (c/px) | f_detect | status | NA/lambda (1/um) | NA_eff | true mode | estimated mode (conf.) |')
    rows += [hdr, '|' + '---|' * 12]
    for name in SETS:
        p = f'{OUT}/json/{name}.json'
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        e, b, m = d['meta'], d['result']['band'], d['result']['mode']
        exp = d['expected_fc_px']
        exps = '; '.join(fmt(x[0]) for x in exp) if exp else '-'
        lower = b['status'] in ('edge_unresolved', 'no_signal') or (b['status'] == 'weak' and (b['dchi2'] or 0) < 50) \
            or b['fc_px_err'] is None
        fc = f"> {fmt(b['f_detect_px'])}" if lower else f"{fmt(b['fc_px'])} +- {fmt(b['fc_px_err'])}"
        nal = (f"> {fmt(b['na_over_lambda_lower'], 2)}" if lower else
               f"{fmt(b['na_over_lambda'], 2)} +- {fmt(b['na_over_lambda_err'], 2)}") if b.get('pixel_um') else '-'
        na = (f"> {fmt(b['NA_lower'], 2)}" if lower else f"{fmt(b['NA'], 2)} +- {fmt(b['NA_err'], 2)}") \
            if b.get('NA') is not None or b.get('NA_lower') is not None else '-'
        ok = 'ok' if m['mode'] in e['mode'].split('|') else '**wrong**'
        rows.append(f"| {name} | {e.get('pixel_um') or '?'} | {e.get('NA') or '?'} | {e.get('wavelength_um') or '?'} | "
                    f"{exps} | {fc} | {fmt(b['f_detect_px'])} | {b['status']} | {nal} | {na} | {e['mode']} | "
                    f"{m['mode']} ({m['confidence']:.2f}) {ok} |")
        recs.append((name, e, b, m, d))
    with open(f'{OUT}/real_table.md', 'w') as f:
        f.write('\n'.join(rows) + '\n\n')
        for name, e, b, m, d in recs:
            f.write(f"- **{name}** ({e['note']}; source: {e['src']}): {b['message']}. Mode: {m['reason']}\n")
    return recs


def stats_figure(recs, synth):
    pts = [(n, m['key'], e['mode'], 'o') for n, e, b, m, d in recs] + \
          [(r['case'], r['key'], r['truth'], '^') for r in synth]
    pairs = [('dyn_snr', 'kurtosis', 'log', 'symlog'), ('spots_per_psf', 'isolated', 'log', 'linear'),
             ('trend_r2', 'trend_range', 'linear', 'log'), ('persist_long', 'persist_wide_long', 'linear', 'linear'),
             ('band_snr', 'static_snr', 'log', 'log'), ('rho_1', 'rho_far', 'linear', 'linear')]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for a, (kx, ky, sx, sy) in zip(ax.flat, pairs):
        for n, k, mode, mk in pts:
            x, y = k.get(kx), k.get(ky)
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                continue
            if sx == 'log':
                x = max(x, 1e-3)
            if sy == 'log':
                y = max(y, 1e-3)
            a.scatter(x, y, c=COL.get(mode.split('|')[0], 'k'), marker=mk, s=40, alpha=0.8, edgecolors='k', linewidths=0.3)
            if mk == 'o':
                a.annotate(n, (x, y), fontsize=6, alpha=0.8)
        a.set_xscale(sx); a.set_yscale(sy); a.set_xlabel(kx); a.set_ylabel(ky)
    for mode, c in COL.items():
        ax[0, 0].scatter([], [], c=c, label=f'true {mode}')
    ax[0, 0].scatter([], [], c='w', edgecolors='k', marker='o', label='real'); ax[0, 0].scatter([], [], c='w', edgecolors='k', marker='^', label='synthetic')
    ax[0, 0].legend(fontsize=7)
    fig.suptitle('classifier statistics (colour = true mode; circles real, triangles synthetic)')
    fig.tight_layout(); fig.savefig(f'{OUT}/figs/classifier_stats.png', dpi=90); plt.close(fig)


def synth_tables():
    out = []
    pa = f'{OUT}/synth/part_A.json'
    if os.path.exists(pa):
        rows = json.load(open(pa))
        out += ['### A. Band limit with vectorial PSFs (simulate_movie, 400 frames 128 x 128, px 0.1 um, 600 nm)', '',
                '| PSF | NA | n_med | emitter z offset (um) | photons | f_c true 2NA/lambda | f_c n_med-limited | '
                'fit f_c | f_detect | fit / n_med-limited | status |', '|' + '---|' * 11]
        for r in rows:
            out.append(f"| {r['psf']} | {r['NA']} | {r['n_med']} | {r['z_offset_um']} | {r['photons']} | "
                       f"{fmt(r['fc_true'])} | {fmt(r['fc_nmed'])} | {fmt(r['fc_fit'])} +- {fmt(r['fc_err'])} | "
                       f"{fmt(r['f_detect'])} | {fmt(r['ratio'])} | {r['status']} |")
        out.append('')
    pb = f'{OUT}/synth/part_B.json'
    if os.path.exists(pb):
        out += ['### B. Sampling and signal (scalar synthetic SMLM movies)', '',
                '| case | f_c true | fit f_c | f_detect | status |', '|---|---|---|---|---|']
        for r in json.load(open(pb)):
            out.append(f"| {r['case']} | {fmt(r['fc_true'])} | {fmt(r['fc_fit'])} +- {fmt(r['fc_err'])} | "
                       f"{fmt(r['f_detect'])} | {r['status']} |")
        out.append('')
    synth = []
    pc = f'{OUT}/synth/part_C.json'
    if os.path.exists(pc):
        synth = json.load(open(pc))
        out += ['### C. Mode classifier on synthetic movies', '', '| case | true mode | estimated (conf.) | key |',
                '|---|---|---|---|']
        for r in synth:
            k = r['key']
            out.append(f"| {r['case']} | {r['truth']} | {r['mode']} ({r['confidence']:.2f}) | dyn_snr "
                       f"{fmt(k['dyn_snr'], 2)}, kurt {fmt(k['kurtosis'], 1)}, spots/PSF {fmt(k['spots_per_psf'])}, "
                       f"isolated {fmt(k['isolated'], 2)}, trend R2 {fmt(k['trend_r2'], 2)} |")
        out.append('')
    pd = f'{OUT}/synth/part_D.json'
    if os.path.exists(pd):
        out += ['### D. Astigmatism in the time-averaged spectrum of SMLM movies', '',
                '| PSF | z distribution | contrast m2 | se | angle (deg) | m2 block sd | per-frame anisotropy rms | '
                'support ellipse ratio |', '|---|---|---|---|---|---|---|---|']
        for r in json.load(open(pd)):
            out.append(f"| {r['psf']} | {r['z']} | {fmt(r['contrast_m2'])} | {fmt(r['contrast_m2_se'])} | "
                       f"{fmt(r['contrast_angle'], 0)} | {fmt(r['m2_block_sd'])} | {fmt(r.get('aniso_frame_rms'))} | "
                       f"{fmt(r['ellipse_ratio'])} |")
    with open(f'{OUT}/synth_table.md', 'w') as f:
        f.write('\n'.join(out) + '\n')
    return synth


if __name__ == '__main__':
    recs = real_table()
    synth = synth_tables()
    stats_figure(recs, synth)
    print(open(f'{OUT}/real_table.md').read())
    print(open(f'{OUT}/synth_table.md').read())
