"""Render the single-PSF comparison PDF from benchmark JSON; no model, CUDA or retiming required."""
import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent / 'out'
LABEL = dict(spectral='Vectorial, exact (spectral)', box='Vectorial, midpoint', gauss_legendre='Vectorial, Gauss–Legendre',
             astig='Astigmatic Gaussian', gauss2d='2D Gaussian (in focus)',
             cr_int='Integrated Catmull–Rom', bs_int='Integrated B-spline', cr_pix='Catmull–Rom')
OPTICS = ['spectral', 'gauss_legendre', 'box']
GAUSS = ['astig', 'gauss2d']
TABLES = ['cr_int', 'bs_int', 'cr_pix']     # spline definitions
COLORS = ['#176878', '#c25c32', '#6955a1', '#526579', '#9c7233']
OPS = ['forward', 'grad', 'backward']
BATCHES = [1000, 10000, 100000]
INK, MUTED = '#1c3444', '#495c68'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
                     'axes.labelcolor': INK, 'axes.edgecolor': '#8798a0', 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
FOOTER = 'torch-psf   /   single-PSF models'


def label(name, s=None):
    return 'Point sampling (pixel centre)' if name == 'box' and s == 1 else LABEL[name]


def primary(row):
    return not row['tf32'] and row['backend'] == ('sep' if row['name'] in TABLES else 'native')


def page(pdf, title, subtitle):
    f = plt.figure(figsize=(11.69, 8.27))
    f.text(.065, .92, title, fontsize=22, weight='bold')
    f.text(.065, .878, subtitle, fontsize=10, color=MUTED)
    f.add_artist(plt.Line2D([.065, .935], [.854, .854], transform=f.transFigure, color='#bed1d5', lw=1))
    f.text(.065, .032, FOOTER, color=MUTED, fontsize=8)
    f.text(.935, .032, str(pdf.get_pagecount() + 1), ha='right', color=MUTED, fontsize=8)
    return f


def finish(pdf, f):
    pdf.savefig(f)
    plt.close(f)


def heading(f, y, title):
    f.text(.065, y, title, fontsize=13, weight='bold')


def para(f, y, message, width=140):
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


def fmt3(a, digits=3):
    return ' / '.join('—' if x is None or not np.isfinite(x) else f'{x:.{digits}g}' for x in a)


def rate(row):
    if not row or 'spots_s' not in row:
        return '—'
    noisy = row.get('iqr', 0) > .1 * row['spots_s']      # repeats spread > 10 %: disturbed window
    return f"{row['spots_s']/1000:,.1f}" + ('' if row['idle'] else '*') + ('~' if noisy else '')


def selected_timing(data, name, op, n=10000, s=None):
    return next((r for r in data['timing'] if r['name'] == name and r['S'] == s and r['op'] == op
                 and r['batch'] == n and primary(r) and not r['cudnn_benchmark']), None)


def headline_timing(data, name, op):
    """Spline rates from the clean interleaved run when there is one."""
    for sample in data.get('paired', []):
        if sample.get('idle'):
            row = next((r for r in sample['rows'] if r['name'] == name and r['op'] == op), None)
            if row:
                return dict(row, idle=True)
    return selected_timing(data, name, op, s=2)


def speed_cells(data, name, s, n=10000, ops=OPS):
    """forward / grad / backward rates (k spots/s) at n spots; a missing rate falls back to 1,000 spots (marked †)."""
    out = []
    for op in ops:
        r = headline_timing(data, name, op) if name in TABLES and n == 10000 else selected_timing(data, name, op, n, s)
        if (not r or 'spots_s' not in r) and n != 1000:
            r1 = selected_timing(data, name, op, 1000, s)
            out.append(rate(r1) + '†' if r1 and 'spots_s' in r1 else '—')
        else:
            out.append(rate(r))
    return ' / '.join(out)


def acc_row(data, name, s):
    return next((r for r in data['accuracy'] if r['name'] == name and r['S'] == s and primary(r)), None)


def bias(r, photons='5000'):
    b = list(r['bias_nm'][photons]['rms'])
    if r['name'] == 'gauss2d':
        b[2] = None
    return b


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
    return LABEL[name] + ' at 2 knots / pixel', (
        f'Of the two integrated splines the B-spline is {ratio:.3f}× as fast (geometric mean of forward, value plus '
        'gradient and backward at 10,000 spots, sep backend, interleaved timing) and about eight times as accurate.')


def missing(data):
    """One sentence on the timings without a rate."""
    rows = [r for r in data['timing'] if 'spots_s' not in r]
    if not rows:
        return 'Every timing case has a rate.'
    oom = [r for r in rows if 'memory' in r.get('unavailable', '').lower()]
    cap = data.get('allocator_budget_MiB', {}).get('False')
    text = (f"{len(oom)} timing cases ran out of memory under the {cap / 1024:.0f} GB PyTorch memory cap (one call "
            "renders the whole batch)." if cap else f'{len(oom)} timing cases ran out of GPU memory.')
    other = len(rows) - len(oom)
    return text + (f' {other} failed otherwise (see the JSON).' if other else '')


# ---------------------------------------------------------------------------------------------------------- pages
def at_a_glance(pdf, data):
    os_ = data['optical_os']
    f = page(pdf, 'Single-PSF models in torch-psf', f"At a glance   /   {data['model']}   /   {data['gpu']}")
    crlb = data['crlb_nm']['5000']
    heading(f, .80, f"10,000 spots; bias at 5,000 photons (CRLB {fmt3(crlb)} nm in x / y / z)")
    picks = [('spectral', os_, 'Vectorial, exact', f'{os_} samples / px'),
             ('gauss_legendre', os_, 'Vectorial, Gauss–Legendre', f'{os_} nodes / px'),
             ('box', os_, 'Vectorial, midpoint', f'{os_} samples / px'),
             ('box', 1, 'Point sampling (what most codes do)', '1 sample / px'),
             ('bs_int', 2, 'Integrated B-spline (default)', '2 knots / px'),
             ('cr_pix', 2, 'Catmull–Rom spline', '2 knots / px'),
             ('astig', None, 'Astigmatic Gaussian', 'fitted')]
    rows, got = [], {}
    for name, s, title, setting in picks:
        r = acc_row(data, name, s)
        if not r:
            continue
        got[(name, s)] = r
        z = r['bias_nm']['5000']['rms'][2]
        err = 'reference' if name == 'spectral' else '—' if name == 'astig' else f"{r['value']['max']:.2g}"
        rows.append([title, setting, err, '≈ 0' if z < .005 else f'{z:.3g} nm' if z >= 1 else f'{z:.2g} nm',
                     speed_cells(data, name, s, ops=OPS[:2])])
    table(f, .44, .32, ['Model', 'Setting', 'Max error / peak', 'RMS z bias', 'Forward / value + derivatives (k spots/s)'],
          rows, [.27, .13, .14, .13, .33], size=9.5)
    para(f, .425, 'Max error: worst |model − exact model| relative to the spot\'s peak (— for the Gaussian: model mismatch, '
         'see its page). † = 1,000 spots; ~ = disturbed timing window. Spline rates from the interleaved run.')
    def sp(name, s, op):
        r = headline_timing(data, name, op) if name in TABLES else selected_timing(data, name, op, 10000, s)
        return r['spots_s'] if r and 'spots_s' in r else None
    notes = []
    pt, bs, ex, ast = got.get(('box', 1)), got.get(('bs_int', 2)), got.get(('spectral', os_)), got.get(('astig', None))
    if pt:
        notes.append(f"Point sampling, the usual choice elsewhere, biases z by {pt['bias_nm']['5000']['rms'][2]:.2g} nm RMS, "
                     f"{pt['bias_nm']['5000']['rms'][2] / crlb[2]:.0%} of the CRLB: integrating over the pixel matters.")
    if bs and ex and sp('bs_int', 2, 'forward') and sp('spectral', os_, 'forward') and sp('spectral', os_, 'grad'):
        notes.append(f"The integrated B-spline is {sp('bs_int', 2, 'forward') / sp('spectral', os_, 'forward'):.0f}× "
                     f"faster than the exact model forward and {sp('bs_int', 2, 'grad') / sp('spectral', os_, 'grad'):.0f}× "
                     f"for value plus derivatives (what fitting needs), with its z bias "
                     f"{crlb[2] / max(bs['bias_nm']['5000']['rms'][2], 1e-9):,.0f}× below the CRLB.")
    notes.append(f'Gauss–Legendre at {os_} nodes per pixel is as accurate as exact integration and needs no FFT; midpoint '
                 'converges slowly (error ∝ 1/S²).')
    if ast:
        notes.append(f"A fitted astigmatic Gaussian biases z by {ast['bias_nm']['5000']['rms'][2]:.0f} nm RMS, "
                     f"{ast['bias_nm']['5000']['rms'][2] / crlb[2]:.0f}× the CRLB, on this calibration.")
    y = .34
    for n in notes:
        para(f, y, '•  ' + n, width=150)
        y -= .05 if len(n) < 150 else .07
    finish(pdf, f)


def overview(pdf, data):
    f = page(pdf, 'Models and method', f"{data['model']}   /   {data['gpu']}   /   one emitter per "
             f"{data['roisize']} × {data['roisize']} ROI")
    heading(f, .80, 'Models compared')
    rows = [['Vectorial PSF', 'Pupil model; pixel = point sample or midpoint / exact / Gauss–Legendre integral',
             '1–4 samples / pixel'],
            ['Astigmatic Gaussian', 'σx(z), σy(z) curves fitted to the vectorial PSF through focus', 'fitted'],
            ['2D Gaussian', 'Fixed width fitted at the in-focus plane; x, y only', 'fitted'],
            ['Catmull–Rom spline', 'Tabulated vectorial PSF: interpolates pixel values', '1–4 knots / pixel'],
            ['Integrated splines', 'Tabulated point density (Catmull–Rom or cubic B-spline) integrated exactly over '
             'each pixel', '1–4 knots / pixel']]
    table(f, .555, .22, ['Model', 'Definition', 'Compared at'], rows, [.18, .62, .2], size=9)
    heading(f, .48, 'How they are compared')
    chk = data.get('reference_check')
    ref = ('The reference is the vectorial PSF with exact (spectral) pixel integration at '
           f"{data['optical_os']} samples / pixel" +
           (f"; an independent Gauss–Legendre quadrature (6 nodes per pixel) agrees with it to {chk['value_max']:.1e} "
            "of the peak. "
            if chk else '. '))
    para(f, .445, ref + f"Errors are the worst |model − reference| over {data['N']:,} random positions (±1.9 px, "
         f"{data['z_range'][0]}–{data['z_range'][1]} µm), relative to each spot's reference peak. Bias comes from "
         "Poisson fits (x, y, z, photons, background) of noise-free reference images at 1,000 / 5,000 / 20,000 photons "
         f"and {data['bg']:g} background photons / pixel; compare it with the Cramér–Rao bound (CRLB) at 5,000 photons, "
         f"{fmt3(data['crlb_nm']['5000'])} nm (x / y / z). Speed is the median of 10 repeats in spots / s on one GPU, "
         "TF32 and cuDNN autotune off.")
    heading(f, .275, 'Contents')
    para(f, .24, '1. Vectorial PSF: pixel integration.   2. Gaussian approximations.   3. Splines: knot spacing, the '
         'three spline definitions and their throughput.   4. Summary of all models and the default spline.')
    para(f, .17, f"{data['gpu']} · PyTorch {data['torch']} · CUDA {data['cuda']} · revision "
         f"{(data.get('revision') or '?')[:7]}. {missing(data)}")
    finish(pdf, f)


def vectorial(pdf, data):
    f = page(pdf, 'Vectorial PSF: pixel integration', 'Same optics, three ways to integrate the light over each '
             'camera pixel')
    para(f, .83, 'Midpoint averages point samples of the image on a sub-pixel grid (approximate; error ∝ 1/S²). Exact '
         '(spectral) integration multiplies the band-limited image spectrum by the pixel transfer function; it needs '
         'alias-free sub-pixel sampling. Gauss–Legendre evaluates the image at quadrature nodes inside each pixel and '
         'sums them with weights (no FFT, no image-plane blur). Point sampling, midpoint at one sample per pixel, takes '
         'the PSF value at the pixel centre, as most vectorial PSF codes do. Exact at '
         f"{data['optical_os']} samples / pixel is the reference itself.")
    plots = [(.09, .41, 'Worst image error', lambda r: r['value']['max'], 'fraction of reference peak'),
             (.56, .41, 'Axial localization bias', lambda r: r['bias_nm']['5000']['rms'][2], 'RMS z bias (nm), 5,000 ph')]
    for x, y, title, metric, unit in plots:
        ax = f.add_axes([x, y, .36, .24])
        for i, name in enumerate(OPTICS):
            rows = sorted((r for r in data['accuracy'] if r['name'] == name and primary(r)), key=lambda r: r['S'])
            pts = [(r['S'], metric(r)) for r in rows if not (name == 'spectral' and r['S'] == data['optical_os'])]
            if pts:
                ax.semilogy(*zip(*[(s, max(v, 1e-9)) for s, v in pts]), marker='o', ms=4, color=COLORS[i],
                            label=LABEL[name])
        ax.set(xticks=[1, 2, 3, 4], xlabel='Samples / pixel', ylabel=unit)
        ax.set_title(title, loc='left', fontsize=11, weight='bold')
        ax.grid(alpha=.2)
    handles, names = ax.get_legend_handles_labels()
    names = [n + ' (1 sample = point sampling)' if n == LABEL['box'] else n for n in names]
    f.legend(handles, names, loc='center', bbox_to_anchor=(.5, .345), ncol=3, frameon=False, fontsize=8)
    rows = []
    for name in OPTICS:
        for s in (1, 2, 3, 4):
            r = acc_row(data, name, s)
            if r:
                ref = name == 'spectral' and s == data['optical_os']
                rows.append([label(name, s), s, 'reference' if ref else f"{r['value']['max']:.2g}", fmt3(bias(r)),
                             speed_cells(data, name, s, n=1000)])
    table(f, .065, .25, ['Model', 'Samples / px', 'Max error', 'RMS bias 5k ph, x / y / z (nm)',
                          'Fwd / grad / back, 1,000 spots (k/s)'], rows, [.25, .1, .12, .26, .27], size=8)
    finish(pdf, f)


def gaussians(pdf, data):
    g = data['gaussian_fit']
    a, b = g['astig'], g['gauss2d']
    f = page(pdf, 'Gaussian approximations', 'Fitted to the vectorial PSF: what a Gaussian model costs')
    para(f, .815, 'Both Gaussians are pixel-integrated and fitted by least squares to the reference images of a '
         'centred emitter. Astigmatic: σx0 / σy0 = '
         f"{a['sigma_px'][0]:.3f} / {a['sigma_px'][1]:.3f} px, x and y foci at ±{1000 * a['gamma_um']:.0f} nm around "
         f"z0 = {a['z0_um']:.3f} µm, depth of focus {a['d_um']:.3f} µm. 2D: σ = {b['sigma_px'][0]:.3f} / "
         f"{b['sigma_px'][1]:.3f} px at the in-focus plane z = {b['z_um']:.3f} µm, where it is evaluated (z fixed). "
         'The error below is model mismatch, not numerical error: a Gaussian has no rings or tails, so its image error '
         'is of order the peak and the bias is what matters.')
    rows = []
    for name, crlb in (('astig', data['crlb_nm']), ('gauss2d', data.get('crlb_nm_focus', data['crlb_nm']))):
        r = acc_row(data, name, None)
        for p in ('1000', '5000', '20000'):
            mx, rms, cr = list(r['bias_nm'][p]['max']), bias(r, p), list(crlb[p])
            if name == 'gauss2d':
                mx[2] = cr[2] = None
            rows.append([LABEL[name], f'{int(p):,}', fmt3(mx), fmt3(rms), fmt3(cr)])
    table(f, .37, .3, ['Model', 'Photons', 'Maximum |bias|  x / y / z (nm)', 'RMS bias  x / y / z (nm)',
                       'CRLB  x / y / z (nm)'], rows, [.22, .1, .24, .22, .22], size=8.4)
    para(f, .345, 'The CRLB of the 2D rows is the exact model\'s at the in-focus plane. Bias grows with the photon '
         'count relative to the CRLB because the mismatch is systematic while the noise shrinks.')
    rows = [[LABEL[n], f"{acc_row(data, n, None)['value']['max']:.2g}", speed_cells(data, n, None, ops=OPS[:2])]
            for n in GAUSS]
    table(f, .14, .1, ['Model', 'Max image error', 'Fwd / grad, 10,000 spots (k/s)'], rows, [.3, .3, .4])
    para(f, .12, 'torch-psf stores the Gaussian shape parameters (σx0, σy0, and γ, d for the astigmatic model) as plain numbers, '
         'not tensors, so there is no gradient with respect to them yet and no backward timing.')
    finish(pdf, f)


def accuracy_plots(pdf, data):
    f = page(pdf, 'Splines: effect of knot spacing', 'sep backend   /   Spot-wise maximum pixel error   /   '
             'Noise-free fits')
    acc = [r for r in data['accuracy'] if primary(r) and r['name'] in TABLES]
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
        'cr_int': 'SplinePSF   /   Knots sample the PSF density   /   Interpolant integrated exactly over each pixel',
        'bs_int': 'SplinePSF   /   Knots are prefiltered B-spline coefficients of the PSF density   /   Interpolant integrated exactly over each pixel',
        'cr_pix': 'SplinePSF   /   Knots are pixel values of the exactly integrated model   /   Interpolated, not integrated',
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
    para(f, .572, f"Each error cell gives maximum / median over {data['N']:,} positions. Each spot uses its own "
         'reference absolute peak for that quantity. χ² is the ROI sum of squared photon error divided pixelwise by '
         'exact expected counts including background, averaged over the pixels and all positions.')
    heading(f, .455, 'Localization bias and median exact-model CRLB (nm)')
    b = []
    for r in rows:
        for p in ('1000', '5000', '20000'):
            b.append([r['S'], f'{int(p):,}', fmt3(r['bias_nm'][p]['max']), fmt3(r['bias_nm'][p]['rms']),
                      fmt3(data['crlb_nm'][p])])
    table(f, .142, .28, ['S', 'Photons', 'Maximum |bias|   x / y / z', 'RMS bias   x / y / z', 'CRLB   x / y / z'], b,
          [.05, .11, .30, .30, .24], size=8.4)
    para(f, .116, f"{data['n_fit']:,} fixed emitters. A joint Poisson fit of x, y, z, intensity and background uses an "
         'offset start and 25 damped iterations.')
    finish(pdf, f)


def spline_speeds(pdf, data):
    f = page(pdf, 'Spline throughput at two knots per pixel', 'Median spots per second   /   Autotune off   /   '
             'Full precision')
    for j, op in enumerate(OPS):
        ax = f.add_axes([.09 + j*.307, .506, .246, .245])
        for i, name in enumerate(TABLES):
            rows = [selected_timing(data, name, op, n=n, s=2) for n in BATCHES]
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
    f.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(.5, .413), ncol=3, frameon=False,
             fontsize=8)
    timings = {(r['name'], r['backend'], r['tf32'], r['op'], r['cudnn_benchmark']): r
               for r in data['timing'] if r['S'] == 2 and r['batch'] == 10000}
    rows = []
    for r in (r for r in data['accuracy'] if r['S'] == 2 and r['name'] in TABLES):
        key = (r['name'], r['backend'], r['tf32'])
        rates = [' / '.join(rate(timings.get(key + (op, tune))) for tune in (False, True)) for op in OPS]
        rows.append([LABEL[r['name']], r['backend'] + (' + TF32' if r['tf32'] else ''), f"{r['value']['max']:.3g}",
                     *rates])
    table(f, .163, .22, ['10,000 spots', 'GPU backend', 'Max error', 'Forward (k/s)', 'Grad (k/s)', 'Backward (k/s)'],
          rows, [.23, .13, .095, .18, .18, .185], size=8)
    para(f, .148, 'Entries: cuDNN autotune off / on; max error = worst value error / reference peak; whiskers = '
         'interquartile range of 10 repeats; open points = background GPU activity; ~ = repeats spread > 10 % (a disturbed '
         'window: sep does not use cuDNN, so its off / on rates should agree). sep (separable) contracts each spot\'s knot '
         'block axis by axis with batched matrix products (cuBLAS); conv runs one grouped 2-D convolution per batch '
         '(cuDNN, needs autotune).', width=150)
    finish(pdf, f)


def summary(pdf, data):
    f = page(pdf, 'Summary and default spline', 'All single-PSF models at their usual settings   /   '
             'bias at 5,000 photons')
    os_ = data['optical_os']
    rows = []
    for name, s, setting in [('spectral', os_, f'{os_} samples / px'), ('gauss_legendre', os_, f'{os_} samples / px'),
                             ('box', os_, f'{os_} samples / px'), ('box', 1, '1 sample / px'),
                             ('bs_int', 2, '2 knots / px'),
                             ('cr_int', 2, '2 knots / px'), ('cr_pix', 2, '2 knots / px'),
                             ('astig', None, 'fitted'), ('gauss2d', None, 'fitted, z fixed')]:
        r = acc_row(data, name, s)
        if not r:
            continue
        err = 'reference' if name == 'spectral' else f"{r['value']['max']:.2g}"
        rows.append([label(name, s), setting, err, fmt3(bias(r)), speed_cells(data, name, s)])
    table(f, .44, .33, ['Model', 'Setting', 'Max error / peak', 'RMS bias x / y / z (nm)',
                        'Fwd / grad / back, 10,000 spots (k/s)'], rows, [.24, .14, .13, .22, .27], size=8.6)
    para(f, .425, f"CRLB at 5,000 photons: {fmt3(data['crlb_nm']['5000'])} nm (x / y / z). † = 1,000 spots (10,000 did "
         'not fit in memory); * = background GPU activity; ~ = repeats spread > 10 %; — = not applicable or no rate. Spline rates come from the '
         'interleaved sep run.')
    title, why = decision(data)
    heading(f, .335, 'Default spline: ' + title)
    para(f, .30, why + ' to_spline builds this spline by default.')
    heading(f, .225, 'Which model when')
    para(f, .19, 'The vectorial model with exact (spectral) integration is the reference and the one to fit optical '
         'parameters with; Gauss–Legendre matches it without FFTs, midpoint is approximate and point sampling, the usual '
         'choice elsewhere, ignores the pixel area. For localization the integrated '
         'B-spline gives nearly the vectorial accuracy many times faster. The plain Catmull–Rom spline is the fastest '
         'and still far below the CRLB, at the cost of larger model error. The Gaussians are fast but biased far beyond '
         'the splines: only for quick estimates or when no calibration exists.')
    finish(pdf, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    latest = OUT / 'psf_bench.json'
    parser.add_argument('--input', type=Path, default=latest)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if not data.get('accuracy') or 'gaussian_fit' not in data:
        raise ValueError('Complete the precision sweep (bench.py of the single-PSF comparison) before the report')
    if not data.get('timing') or any('spots_s' not in r and 'unavailable' not in r for r in data['timing']):
        raise ValueError('Complete the timing sweep before generating the report')
    output = args.output or args.input.with_suffix('.pdf')
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output, metadata={'Author': 'Jelmer Cnossen', 'Title': 'Single-PSF models in torch-psf'}) as pdf:
        at_a_glance(pdf, data)
        overview(pdf, data)
        vectorial(pdf, data)
        gaussians(pdf, data)
        accuracy_plots(pdf, data)
        for name in TABLES:
            candidate(pdf, data, name)
        spline_speeds(pdf, data)
        summary(pdf, data)


if __name__ == '__main__':
    main()
