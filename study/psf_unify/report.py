"""Render the stage-0 PDF from benchmark JSON; no model, CUDA or retiming required."""
import argparse
from collections import Counter
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent / 'out'
LABEL = dict(cr_int='Integrated Catmull–Rom', bs_int='Integrated B-spline', cr_pix='Exact-pixel Catmull–Rom',
             box_os2='Legacy midpoint Catmull–Rom', box='Vectorial box', spectral='Vectorial spectral',
             gauss_legendre='Vectorial Gauss–Legendre', fourpi='Spectral 4Pi', channels='Two spectral channels')
TABLES = list(LABEL)[:4]
COLORS = ['#176878', '#c25c32', '#6955a1', '#526579', '#9c7233']
OPS = ['forward', 'grad', 'backward']
BATCHES = [1000, 10000, 100000]
INK, MUTED = '#1c3444', '#495c68'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
                     'axes.labelcolor': INK, 'axes.edgecolor': '#8798a0', 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def primary(row):
    return not row['tf32'] and row['backend'] == ('sep' if row['name'] in TABLES[:2] else
                                                 'conv' if row['name'] in TABLES else 'native')


def page(pdf, title, subtitle):
    f = plt.figure(figsize=(11.69, 8.27))
    f.text(.065, .92, title, fontsize=22, weight='bold')
    f.text(.065, .878, subtitle, fontsize=10, color=MUTED)
    f.add_artist(plt.Line2D([.065, .935], [.854, .854], transform=f.transFigure, color='#bed1d5', lw=1))
    f.text(.065, .032, 'torch-psf   /   PSF unification   /   Stage 0', color=MUTED, fontsize=8)
    f.text(.935, .032, str(pdf.get_pagecount() + 1), ha='right', color=MUTED, fontsize=8)
    return f


def finish(pdf, f):
    pdf.savefig(f)
    plt.close(f)


def heading(f, y, title):
    f.text(.065, y, title, fontsize=13, weight='bold')


def para(f, y, message, width=138):
    f.text(.065, y, textwrap.fill(message, width), va='top', linespacing=1.45, fontsize=9.5)


def table(f, bottom, height, columns, rows, widths=None, size=9):
    ax = f.add_axes([.065, bottom, .87, height])
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=columns, colWidths=widths, loc='center',
                 bbox=[0, 0, 1, 1], cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#245361' if i == 0 else '#eaf1f2' if i % 2 else '#f7f9fa')
        if i == 0:
            cell.set_text_props(color='white', weight='bold')
        if j == 0:
            cell.set_text_props(ha='left')
    return t


def rate(row):
    if not row or 'spots_s' not in row:
        return '—'
    return f"{row['spots_s']/1000:,.1f}" + ('' if row['idle'] else '*')


def selected_timing(data, name, op, n=10000, s=2):
    return next((r for r in data['timing'] if r['name'] == name and r['S'] == s and r['op'] == op
                 and r['batch'] == n and primary(r) and not r['cudnn_benchmark']), None)


def headline_timing(data, name, op):
    for sample in data.get('paired', []):
        if sample.get('idle'):
            return dict(next(r for r in sample['rows'] if r['name'] == name and r['op'] == op), idle=True)
    return selected_timing(data, name, op)


def decision(data):
    # The default sep backend runs full precision. Compare a matched 10k batch for
    # all three requested operations, equally weighting throughput ratios.
    ratios = []
    for op in OPS:
        a, b = [headline_timing(data, name, op) for name in TABLES[:2]]
        if not all(r and r.get('idle') and 'spots_s' in r for r in (a, b)):
            return 'Integrated Catmull–Rom at S = 2 (provisional)', (
                'The agreed rule retains Catmull–Rom until a clean paired timing establishes that B-spline is faster.')
        ratios.append(b['spots_s'] / a['spots_s'])
    ratio = np.exp(np.mean(np.log(ratios)))
    name = TABLES[int(ratio > 1)]
    return LABEL[name] + ' at S = 2', (
        f'The geometric mean of the B-spline / Catmull–Rom throughput ratios is {ratio:.3f} '
        'across forward, value plus gradient, and backward at 10,000 spots (sep, autotune off).')


def summary(pdf, data, acc):
    f = page(pdf, 'Choosing a default PSF table', 'm1809 in-situ calibration   /   Spectral reference   /   2 knots per pixel')
    title, why = decision(data)
    heading(f, .795, title)
    para(f, .76, why)
    rows = []
    for r in (r for r in acc if r['S'] == 2):
        b = r['bias_nm']['5000']['rms']
        rows.append([LABEL[r['name']], f"{r['value']['max']:.3g} / {r['value']['median']:.3g}",
                     ' / '.join(f'{x:.3g}' for x in b),
                     ' / '.join(rate(headline_timing(data, r['name'], op)) for op in OPS)])
    table(f, .48, .20, ['Table', 'Error / peak · max / median', 'RMS bias x / y / z (nm)', 'Fwd / grad / back (kspots/s)'],
          rows, [.245, .225, .25, .28])
    para(f, .455, 'Bias at 5,000 photons and background 10 photons/pixel. Throughput uses 10,000 spots, '
         'autotune off, TF32 off, sep for integrated tables and conv for pixel tables. Clean interleaved measurements take '
         'precedence when available. An asterisk marks background GPU activity.')
    heading(f, .337, 'Measurement conditions')
    med = data['crlb_nm']['5000']
    para(f, .30, f"{data['N']:,} uniform sub-pixel positions in ±1.9 px and 0–0.4 µm, 13 × 13 ROI, 0.01 µm axial knots. "
         f"NA {data['NA']:.2f}, {data['pixel_um']*1000:.2f} nm pixels, {data['npupil']} pupil samples per axis, "
         f"optical oversampling {data['optical_os']} (legacy midpoint table 2). "
         f"The exact-model median CRLB at 5,000 photons is {med[0]:.2f} / {med[1]:.2f} / {med[2]:.2f} nm (x / y / z).")
    counted = Counter('idle' if r.get('idle') else 'desktop' for r in data['timing'] if 'spots_s' in r)
    initial = data.get('initial_gpu_observation')
    first = f"Initial GPU utilization was {initial['util_percent']:.0f}%. " if initial else ''
    para(f, .18, f"{data['gpu']} · PyTorch {data['torch']} · CUDA {data['cuda']}. {first}Of {len(data['timing'])} timing cases, "
         f"{counted['idle']} passed the idle screen, {counted['desktop']} ran with background activity, and "
         f"{len(data['timing']) - sum(counted.values())} have no rate. The timing pages give coverage and reasons.")
    finish(pdf, f)


def accuracy_plots(pdf, acc):
    f = page(pdf, 'Effect of lateral knot spacing', 'Full-precision primary backend   /   Spot-wise maximum pixel error   /   Noise-free fits')
    plots = [(.09, .51, 'Worst image error', lambda r: r['value']['max'], 'fraction of reference peak'),
             (.56, .51, 'Worst axial derivative error', lambda r: r['dz']['max'], 'fraction of derivative peak'),
             (.09, .15, 'Lateral localization bias', lambda r: r['bias_nm']['5000']['rms'][0], 'RMS x bias (nm)'),
             (.56, .15, 'Axial localization bias', lambda r: r['bias_nm']['5000']['rms'][2], 'RMS z bias (nm)')]
    for x, y, title, metric, unit in plots:
        ax = f.add_axes([x, y, .36, .245])
        for i, name in enumerate(TABLES):
            rows = sorted((r for r in acc if r['name'] == name), key=lambda r: r['S'])
            ax.semilogy([r['S'] for r in rows], [metric(r) for r in rows], marker='o', ms=4,
                        color=COLORS[i], label=LABEL[name])
        ax.set(xticks=[1, 2, 3, 4], xlabel='Knots / pixel', ylabel=unit)
        ax.set_title(title, loc='left', fontsize=11, weight='bold')
        ax.grid(alpha=.2)
    f.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .061), ncol=4,
             frameon=False, fontsize=8)
    finish(pdf, f)


def candidate(pdf, data, name):
    descriptions = {
        'cr_int': 'PixelSplinePSF   /   Point-density knots   /   Exact pixel integral of Catmull–Rom interpolation',
        'bs_int': 'PixelSplinePSF   /   Prefiltered point-density knots   /   Exact pixel integral of cubic B-spline interpolation',
        'cr_pix': 'Spectrally integrated pixel values   /   Catmull–Rom interpolation at sub-pixel shifts',
        'box_os2': '2 × 2 midpoint quadrature   /   Catmull–Rom interpolation of pixel values   /   S = 2 only',
    }
    f = page(pdf, LABEL[name], descriptions[name])
    rows = sorted((r for r in data['accuracy'] if r['name'] == name and primary(r)), key=lambda r: r['S'])
    heading(f, .796, 'Peak-normalized error and excess χ² per pixel')
    def error(r, k):
        return f"{r[k]['max']:.2g} / {r[k]['median']:.2g}"
    table(f, .593, .16, ['S', 'Value', 'dx', 'dy', 'dz', '50k photons', '200k photons'],
          [[r['S'], *[error(r, k) for k in ('value', 'dx', 'dy', 'dz')],
            *[f"{r['chi2'][p]['mean_pixel']:.3g}" for p in ('50000', '200000')]] for r in rows],
          [.045, .17, .17, .17, .17, .135, .14])
    para(f, .572, 'Each error cell gives maximum / median over 3,000 positions. Each spot uses its own reference '
         'absolute peak for that quantity. χ² is the ROI sum of squared photon error divided pixelwise by exact '
         'expected counts including background, averaged over 169 pixels and all positions.')
    heading(f, .455, 'Localization bias and median exact-model CRLB (nm)')
    b = []
    for r in rows:
        for p in ('1000', '5000', '20000'):
            fmt = lambda a: ' / '.join(f'{x:.3g}' for x in a)
            b.append([r['S'], f'{int(p):,}', fmt(r['bias_nm'][p]['max']), fmt(r['bias_nm'][p]['rms']),
                      fmt(data['crlb_nm'][p])])
    table(f, .142, .28, ['S', 'Photons', 'Maximum |bias|   x / y / z', 'RMS bias   x / y / z', 'CRLB   x / y / z'], b,
          [.05, .11, .30, .30, .24], size=8.4)
    para(f, .116, f"{data['n_fit']:,} fixed emitters. A joint Poisson fit of x, y, z, intensity and background uses an offset start and "
         '25 damped iterations. The JSON includes every conv/TF32 variant, ROI χ² and worst per-spot mean χ².')
    finish(pdf, f)


def speeds(pdf, data, names, title):
    f = page(pdf, title, 'Median spots per second   /   Autotune off   /   Full precision')
    for j, op in enumerate(OPS):
        ax = f.add_axes([.09 + j*.307, .506, .246, .245])
        for i, name in enumerate(names):
            rows = [selected_timing(data, name, op, n=n, s=2 if name in TABLES else None) for n in BATCHES]
            ys = [r['spots_s'] if r and 'spots_s' in r else np.nan for r in rows]
            ax.plot(BATCHES, ys, color=COLORS[i], label=LABEL[name])
            for n, r in zip(BATCHES, rows):
                if r and 'spots_s' in r:
                    ax.vlines(n, r['q1'], r['q3'], color=COLORS[i])
                    ax.plot(n, r['spots_s'], marker='o', ms=4, mec=COLORS[i], mfc=COLORS[i] if r['idle'] else 'white')
        ax.set(xscale='log', yscale='log', xticks=BATCHES, xticklabels=['1k', '10k', '100k'], xlabel='Spots / batch')
        if j == 0:
            ax.set_ylabel('Spots / s')
        ax.set_title(['Forward', 'Value + derivatives', 'Forward + backward'][j], loc='left', fontsize=11, weight='bold')
        ax.grid(alpha=.2)
    f.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.5, .413), ncol=len(names), frameon=False,
             fontsize=8)
    rows = []
    if names == TABLES:
        timings = {(r['name'], r['backend'], r['tf32'], r['op'], r['cudnn_benchmark']): r
                   for r in data['timing'] if r['S'] == 2 and r['batch'] == 10000}
        for r in (r for r in data['accuracy'] if r['S'] == 2):
            key = (r['name'], r['backend'], r['tf32'])
            rates = [' / '.join(rate(timings[key + (op, tune)]) for tune in (False, True)) for op in OPS]
            rows.append([LABEL[r['name']], r['backend'] + (' + TF32' if r['tf32'] else ''),
                         f"{r['value']['max']:.3g}", *rates])
        table(f, .163, .22, ['10,000 spots', 'Backend', 'Max error', 'Forward (k/s)', 'Grad (k/s)', 'Backward (k/s)'],
              rows, [.23, .13, .095, .18, .18, .185], size=8)
        para(f, .143, 'Table entries give autotune off / on. Max error is the worst value error / reference peak. '
             'Whiskers show the interquartile range over 10 synchronized repeats. Open points and starred rates had '
             'background GPU activity. The JSON gives peak memory and quartiles for every measured case.')
        para(f, .071, 'Backward includes rendering and differentiating the sum of squared pixels with respect to the stored knots or coefficients.')
    else:
        for name in names:
            r = [selected_timing(data, name, op, s=None) for op in OPS]
            rows.append([LABEL[name], *[rate(a) for a in r], ' / '.join(f"{a['peak_MiB']:.0f}" if a and 'peak_MiB' in a
                                                                        else '—' for a in r)])
        table(f, .196, .17, ['10,000 spots', 'Forward (k/s)', 'Grad (k/s)', 'Backward (k/s)', 'Peak MiB, same order'],
              rows, [.28, .16, .16, .16, .24])
        para(f, .167, 'Whiskers give the interquartile range across 10 synchronized repeats after 2 warm-ups. Open points '
             'and starred rates had background GPU activity. Gaps indicate unavailable timings. Peak memory is the '
             'maximum live PyTorch allocation. Backward includes rendering and differentiating the sum of squared '
             'pixels with respect to each arm’s Zernikes. The native 4Pi gradient includes phase.')
        para(f, .086, '4Pi uses four channels and the symmetric 32-sample pupil arms from the test fixture. The second '
             'm1809 channel has +0.12 µm focus offset and (+0.2, −0.1) px lateral offset.')
    finish(pdf, f)


def status(r):
    if 'spots_s' in r:
        return 'I' if r.get('idle') else 'D'
    reason = r.get('unavailable', '')
    if reason.startswith('Preflight'):
        return 'B'
    if 'memory' in reason.lower() or 'allocate' in reason.lower():
        return 'M'
    return 'E'


def coverage(pdf, data):
    # Each cell describes all nine batch/operation combinations for a named setting.
    groups = {}
    for r in data['timing']:
        key = (r['name'], r['S'], r['backend'], r['tf32'])
        groups.setdefault(key, {})[(r['cudnn_benchmark'], r['batch'], r['op'])] = r
    items = list(groups.items())
    for start in range(0, len(items), 20):
        f = page(pdf, 'Timing coverage by configuration', 'Within each three-letter group the order is forward, value + derivatives, backward.')
        rows = []
        for (name, s, be, tf), rr in items[start:start+20]:
            cells = ['   '.join(''.join(status(rr[t, n, op]) for op in OPS) for n in BATCHES) for t in [False, True]]
            rows.append([LABEL[name], s or '—', be, 'on' if tf else 'off', *cells])
        table(f, .278, .51, ['Model', 'S', 'Backend', 'TF32', 'Autotune off · 1k / 10k / 100k',
                              'Autotune on · 1k / 10k / 100k'], rows, [.25, .045, .085, .07, .275, .275], size=8.3)
        para(f, .247, 'I = idle-screened, D = desktop-active (clean rate unavailable), B = busy ≥40% or missing '
             'nvidia-smi telemetry, M = memory allocation failed, E = other runtime error (verbatim in JSON). '
             'Three nvidia-smi samples bracket each model, density and autotune group. Idle requires median utilization '
             '≤5% and maximum ≤10% at both ends. Display activity can occur between samples.')
        budgets = data['allocator_budget_MiB']
        para(f, .122, f"Each batch uses one native render call. Autotune runs in fresh processes with PyTorch allocator "
             f"budgets of {budgets['False']:,.0f} / {budgets['True']:,.0f} MiB (off / on). The JSON records every "
             "repeat, quartiles, peak memory, GPU samples and individual failure messages.")
        finish(pdf, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    latest = OUT / 'psf_bench.json'
    parser.add_argument('--input', type=Path, default=latest if latest.exists() else OUT.parent / 'baseline.json')
    parser.add_argument('--output', type=Path, default=OUT/'psf_bench.pdf')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if len(data.get('accuracy', [])) != 34:
        raise ValueError('Complete the precision sweep before generating the report')
    if not data.get('timing') or any('spots_s' not in r and 'unavailable' not in r for r in data['timing']):
        raise ValueError('Complete the timing sweep before generating the report')
    acc = [r for r in data['accuracy'] if primary(r)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output, metadata={'Author': 'Jelmer Cnossen', 'Title': 'PSF unification baseline'}) as pdf:
        summary(pdf, data, acc)
        accuracy_plots(pdf, acc)
        for name in TABLES:
            candidate(pdf, data, name)
        speeds(pdf, data, TABLES, 'Table throughput at two knots per pixel')
        speeds(pdf, data, list(LABEL)[4:], 'Optical model throughput')
        coverage(pdf, data)


if __name__ == '__main__':
    main()
