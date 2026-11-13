"""Tables (LaTeX) and figures of the paper from study/insitu/out/all_seed*.json, speed3.json, pattern_M*.json."""
import json, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O, D = 'study/insitu/out/server/', 'docs/paper/'
runs = [json.load(open(f)) for f in sorted(glob.glob(O + 'all_seed*.json'))]
ns = len(runs)


def ms(vals, f='{:.1f}'):
    v = np.array(vals, float)
    return f.format(v.mean()) + r'\,$\pm$\,' + f.format(v.std(ddof=1) if len(v) > 1 else 0)


rows = [('true PSF', 'true PSF'), ('beads: Zernike', 'beads only, Zernike'), ('beads: spline', 'beads only, spline'),
        ('movie: shared spline', 'spline shared by beads and movie'), ('movie: Zernike', 'Zernike in situ'),
        ('movie: Zernike + knots', 'Zernike + free knots'), ('movie: free pupil', 'Zernike + free pupil map'),
        None, ('clean: Zernike', 'Zernike in situ'), ('clean: Zernike + knots', 'Zernike + free knots'),
        ('clean: free pupil', 'Zernike + free pupil map')]
out = []
for scn in 'AB':
    out.append(r'\begin{tabular}{lrrrrrrr}\toprule' + '\n' + r'PSF estimate & deficit & wrong $z$ (\%) & $y$ rms & $z$ rms & $y$ tilt & $z$ tilt & $I$ ratio\\'
               + '\n' + r' & (nats) & & (nm) & (nm) & (nm) & (nm) & \\ \midrule' + '\n' + r'\multicolumn{8}{l}{\emph{patches detected in the movie}}\\')
    for row in rows:
        if row is None:
            out.append(r'\multicolumn{8}{l}{\emph{clean patches (no detection step)}}\\'); continue
        k, lab = row
        g = lambda key: [r[scn]['methods'][k][key] for r in runs]
        out.append(f"{lab} & {ms(g('dnll'), '{:.2f}')} & {ms([100 * v for v in g('outlier')])} & {ms(g('y_rmse'))} & {ms(g('z_rmse'))} & "
                   f"{ms(g('y_tilt'))} & {ms(g('z_tilt'))} & {ms(g('I_ratio'), '{:.2f}')}\\\\")
    out.append(r'\bottomrule\end{tabular}')
    open(D + f'tab_acc{scn}.tex', 'w').write('\n'.join(out)); out = []

# bias curves
zc = np.linspace(0.05, 0.75, 8); zc = (zc[:-1] + zc[1:]) / 2 * 1000
sel = [('true PSF', 'k', 'true PSF'), ('movie: Zernike', 'C0', 'Zernike in situ'), ('movie: Zernike + knots', 'C1', '+ free knots'),
       ('movie: free pupil', 'C2', '+ free pupil map'), ('movie: shared spline', 'C4', 'shared spline')]
fig, ax = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True)
for i, scn in enumerate('AB'):
    for j, (c, lab) in enumerate((('y', 'mean y error (nm)'), ('z', 'mean z error (nm)'))):
        for k, col, name in sel:
            if k == 'movie: shared spline' and c == 'z':
                continue
            a = np.array([r[scn]['methods'][k]['curve'][c] for r in runs])
            ax[i, j].errorbar(zc, a.mean(0), a.std(0, ddof=1) if ns > 1 else None, color=col, label=name, marker='o', ms=3, lw=1, capsize=2)
        ax[i, j].axhline(0, color='gray', lw=0.5); ax[i, j].set_ylabel(lab); ax[i, j].set_title(f'scenario {scn}', fontsize=9)
ax[1, 0].set_xlabel('true emitter height (nm)'); ax[1, 1].set_xlabel('true emitter height (nm)'); ax[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(D + 'fig_bias.pdf'); plt.close(fig)

# training z histogram
fig, ax = plt.subplots(figsize=(4, 2.6))
zb = np.arange(8) * 100 + 50
ev = np.mean([r['A']['zhist_events'] for r in runs], 0); de = np.mean([r['A']['zhist_det'] for r in runs], 0)
ax.bar(zb, ev, 90, color='0.8', label='on-events'); ax.bar(zb, de, 90, color='C0', label='detected patches')
ax.set_xlabel('emitter height (nm)'); ax.set_ylabel('count'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(D + 'fig_zhist.pdf'); plt.close(fig)

# speed
S = json.load(open(O + 'speed3.json'))
t = [r'\begin{tabular}{lrrrrrrrr}\toprule', r'pupil grid, & model & \multicolumn{3}{c}{thousand emitters per second} & fixed & break- & \multicolumn{2}{c}{speed-up at $10^4$ / $10^5$ / $10^6$ emitters}\\',
     r'oversampling & error & vectorial & spline & localize & (s) & even & gradient & whole iteration\\ \midrule']
for r in S['table']:
    fx = np.nanmin([r['fixed_two_builds'], r['fixed_one_pass']])
    t.append(f"{r['npupil']}, {r['oversample']} & {100 * r['vec_err']:.2f}\\,\\% & {1e3 / r['us_V']:.0f} & {1e3 / r['us_K']:.0f} & {1e3 / r['us_loc']:.0f} & {fx:.3f} & {r['breakeven']:.0f} & "
             f"{r['grad_10000']:.1f} / {r['grad_100000']:.1f} / {r['grad_1e+06']:.0f} & {r['iter_10000']:.1f} / {r['iter_100000']:.1f} / {r['iter_1e+06']:.1f}\\\\")
t.append(r'\bottomrule\end{tabular}')
open(D + 'tab_speed.tex', 'w').write('\n'.join(t))
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
N = np.logspace(2.5, 6, 50)
for r, col in zip(S['table'], ('C0', 'C1', 'C3')):
    fx = np.nanmin([r['fixed_two_builds'], r['fixed_one_pass']])
    ax[0].loglog(N, N * r['us_V'] * 1e-6, col, label=f"vectorial per emitter, {r['npupil']}/{r['oversample']}")
    ax[0].loglog(N, N * r['us_K'] * 1e-6 + fx, col + '--', label=f"through the knots, {r['npupil']}/{r['oversample']}")
ax[0].set_xlabel('emitters'); ax[0].set_ylabel('seconds per gradient'); ax[0].legend(fontsize=6)
for route, col, lab in (('K', 'C0', 'through the knots'), ('V', 'C3', 'vectorial per emitter')):
    L = np.array(S['trace'][route]); ax[1].semilogy(L[:, 1], L[:, 3], col, label=lab)
ax[1].set_xlabel('seconds'); ax[1].set_ylabel('Zernike error (waves rms)'); ax[1].legend(fontsize=7)
ax[1].set_title(f"same fit, {S['N_fit']} emitters, {S['gpu']}", fontsize=8)
fig.tight_layout(); fig.savefig(D + 'fig_speed.pdf'); plt.close(fig)

# pattern
t = [r'\begin{tabular}{llrrrrrr}\toprule', r' & & \multicolumn{3}{c}{with pattern} & \multicolumn{3}{c}{same frames, no pattern}\\',
     r'scenario & PSF model & deficit & $y$ range & $z$ range & deficit & $y$ range & $z$ range\\ \midrule']
P1, P0 = json.load(open(O + 'pattern_M0.9.json')), json.load(open(O + 'pattern_M0.json'))
f = lambda v: f'{min(v):+.0f} .. {max(v):+.0f}'
for scn in 'AB':
    for m, lab in (('Zernike', 'Zernike'), ('free pupil', 'free pupil map'), ('free knots', 'free knots')):
        a, b = P1[f'{scn}/{m}'], P0[f'{scn}/{m}']
        t.append(f"{scn} & {lab} & {a['dnll']:.2f} & {f(a['curve']['y'])} & {f(a['curve']['z'])} & {b['dnll']:.2f} & {f(b['curve']['y'])} & {f(b['curve']['z'])}\\\\")
t.append(r'\bottomrule\end{tabular}')
open(D + 'tab_pattern.tex', 'w').write('\n'.join(t))
for scn in 'AB':
    r0 = runs[0][scn]
    print(scn, 'rois', [r[scn]['n_rois'] for r in runs], 'events', [r[scn]['n_events'] for r in runs], 'sites', [r[scn]['n_sites'] for r in runs],
          'zc err movie', [round(r[scn]['zc_err_movie'], 5) for r in runs], 'clamp', {k: round(v.get('clamp', -1), 4) for k, v in r0['methods'].items() if 'clamp' in v},
          'time', {k: round(np.mean([r[scn]['methods'][k]['sec'] for r in runs]), 1) for k in r0['methods']})
