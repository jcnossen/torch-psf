"""fit_report.pdf of a psf.zip: an overview of the PSF estimation (never every ROI), written with matplotlib
(PdfPages). Every section is written only when its data is available, so a PSF saved without estimator data gets the
model pages only (summary, aberrations / pupil when the model has them, PSF views, field dependence).

Sections (psf.json report.sections lists the ones written):
  summary      model, optics, camera, estimator, settings, results (scalars of the report), warnings
  spots        ROIs found / used / held out / rejected and why (overlap deviance histogram with its threshold)
  quality      the range of ROI quality: histogram of a per-ROI score q (higher = worse) with the cutoff, and data /
               model / difference of the 3 best ROIs, the 3 last accepted (just below the cutoff) and the 3 first
               rejected (just above). InSitu: q = Poisson deviance per px of the first fit, cutoff = the overlap
               rejection threshold; ZStack: q = chi2 per px of a bead (mean over its planes), no rejection, a
               reference line median + 3 x 1.4826 MAD
  fit_stats    histograms of chi2, photons, background, z of refitted spots (InSitu: held-out spots refitted with the
               final spline; ZStack: every bead plane with the fitted model)
  convergence  loss per iteration of each estimation stage; the focus / wavelength scan
  aberrations  Zernike bar chart (error bars when the report has zernike_std), focus / z offset
  pupil        pupil phase and amplitude, Zernike-only and with the free pupil maps (4Pi: both arms)
  psf          xy slices through z and xz sections
  examples     a few ROIs: data / model / difference
  residuals    mean residual (data - model) / photons, binned by z
  field        Zernike maps over the field (zernike_poly field blocks) and the non-affine part of poly mappings

Inputs: report = an estimator (its report, history, data), a dict {report, history, data, settings} or an estimator
report dict; without one, provenance['report'] / ['settings'] of the PSF (the estimators put them there).
"""
import io as _io
import os
import time
import warnings
import numpy as np
import torch

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
A4 = (8.27, 11.69)
A4L = (11.69, 8.27)


def _fig(size=A4):
    from matplotlib.figure import Figure
    return Figure(figsize=size)


def _style(ax, title=None):
    ax.tick_params(labelsize=7, colors=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=8.5, color=INK, loc='left')


def _header(fig, text):
    fig.text(0.06, 0.965, text, fontsize=12, color=INK, weight='bold')


def _np(t):
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ====================================================================================================== inputs
def _attr_settings(est):
    out = {}
    for k, v in vars(est).items():
        if k.startswith('_') or k in ('report', 'history', 'data', 'progress', 'stop', 'dev'):
            continue
        if k == 'camera' and v is not None:
            v = v.to_dict()
        elif not isinstance(v, (int, float, str, bool, tuple, list, dict, type(None))):
            v = repr(v)
        out[k] = v
    return out


def bundle(report, psf):
    """{report, history, data, settings, estimator} from an estimator / dict / None (then the PSF provenance)."""
    m0 = psf[0] if isinstance(psf, (list, tuple)) else psf
    prov = getattr(m0, 'provenance', None) or {}
    b = dict(report={}, history=[], data={}, settings=None, estimator=prov.get('estimator'))
    if report is not None and hasattr(report, 'report') and hasattr(report, 'history'):
        b.update(report=dict(report.report or {}), history=list(report.history or []),
                 data=dict(getattr(report, 'data', None) or {}), estimator=type(report).__name__)
        b['settings'] = report._settings() if hasattr(report, '_settings') else _attr_settings(report)
    elif isinstance(report, dict):
        if report and set(report) <= {'report', 'history', 'data', 'settings', 'estimator'}:
            b.update({k: v for k, v in report.items() if v is not None})
        else:
            b['report'] = dict(report)
    if not b['report']:
        b['report'] = dict(prov.get('report') or {})
    if b['settings'] is None:
        b['settings'] = prov.get('settings') or {k: v for k, v in prov.items() if k not in ('report', 'settings')}
    return b


class _Ctx:
    def __init__(self, psf, meta, stack, z_um, b, source, max_spots):
        self.psf, self.meta, self.stack, self.z_um, self.source, self.max_spots = psf, meta, stack, z_um, source, max_spots
        self.rep, self.history, self.data, self.settings, self.estimator = (b['report'], b['history'], b['data'],
                                                                           b['settings'] or {}, b['estimator'])
        self._fit = False

    @property
    def models(self):
        return list(self.psf) if isinstance(self.psf, (list, tuple)) else [self.psf]

    def fit(self):
        """refitted spots / planes: dict(rois, mu [N, P, P], chi2, photons, bg [N], z [N] or None, kind, label), or
        None (cached)"""
        if self._fit is False:
            self._fit = _fit_data(self)
        return self._fit


@torch.no_grad()
def _fit_data(ctx):
    d = ctx.data
    if d.get('rois') is not None and d.get('model') is not None:            # ZStackEstimator
        rois, mu = _np(d['rois']), _np(d['model'])
        nb, nk, P = rois.shape[0], rois.shape[1], rois.shape[-1]
        ph = _np(d['photons']).reshape(nb, -1) * np.ones((1, nk)) if d.get('photons') is not None else None
        bg = _np(d['bg']).reshape(nb, -1) * np.ones((1, nk)) if d.get('bg') is not None else None
        c2 = _np(d['chi2']) if d.get('chi2') is not None else ((rois - mu) ** 2 / np.maximum(mu, 1e-6)).mean((2, 3))
        z = np.broadcast_to(-_np(d['focus_um'])[None], (nb, nk))
        r = lambda a: None if a is None else np.asarray(a).reshape(nb * nk, *np.shape(a)[2:])
        return dict(rois=r(rois), mu=r(mu), chi2=r(c2), photons=r(ph), bg=r(bg), z=r(z), kind='zstack',
                    bead=np.repeat(np.arange(nb), nk),
                    label=f'{nb} beads x {nk} planes, the fitted model; z = -focus (um)')
    sp, spl = d.get('spots'), d.get('spline')
    if sp is not None and spl is not None:
        from torch_psf.fit import localize
        te = np.asarray(d.get('test', []))
        which, idx = ('held-out', te) if len(te) else ('training', np.asarray(d.get('train', np.arange(len(sp)))))
        idx = idx[:ctx.max_spots]
        if not len(idx):
            return None
        ii = torch.as_tensor(idx, device=sp.rois.device)
        rois = sp.rois[ii].to(spl.dev)
        var = sp.var
        if var is not None and var.dim() > 0:
            var = var[ii].to(spl.dev)
        elif var is not None:
            var = var.to(spl.dev)
            var = None if float(var) == 0 else var
        th, _ = localize(spl, rois, var=var)
        mu = spl.expected(th)
        vv = 0.0 if var is None else var
        c2 = ((rois - mu) ** 2 / (mu + vv).clamp_min(1e-6)).flatten(1).mean(1)
        th = th.cpu().numpy()
        return dict(rois=rois.cpu().numpy(), mu=mu.cpu().numpy(), chi2=c2.cpu().numpy(), photons=th[:, 3],
                    bg=th[:, 4], z=th[:, 2] if spl.z_range is not None else None, kind='spots',
                    label=f'{len(idx)} {which} spots refitted with the final spline (z = fitted, um)')
    return None


def _vectorials(ctx):
    """[(label, VectorialPSF)] for the Zernike / pupil pages"""
    out = []
    reg = __import__('torch_psf.psf', fromlist=['PSF']).PSF._registry
    srcs = ctx.source if isinstance(ctx.source, (list, tuple)) else [ctx.source] * len(ctx.models)
    for c, (m, s) in enumerate(zip(ctx.models, srcs)):
        tag = f'ch{c} ' if len(ctx.models) > 1 else ''
        t = m.type_name
        if t == 'vectorial':
            out.append((tag + 'model', m))
        elif t == 'fourpi':
            out += [('top arm', m.top), ('bottom arm', m.bot)]
        elif t == 'fourpi_spline':
            st = s.state() if s is not None else m.source_state
            if st is not None:
                fp = reg['fourpi'].from_state(st, device='cpu')
                out += [('top arm (source)', fp.top), ('bottom arm (source)', fp.bot)]
        elif s is not None and getattr(s, 'type_name', None) == 'vectorial':
            out.append((tag + 'source', s))
    if not out and getattr(ctx.data.get('vectorial'), 'type_name', None) == 'vectorial':
        out.append(('fitted vectorial', ctx.data['vectorial']))
    return out


# ====================================================================================================== sections
def _flat(d, prefix='', depth=0):
    out = []
    for k, v in d.items():
        name = f'{prefix}{k}'
        if isinstance(v, dict) and depth < 2:
            out += _flat(v, name + '.', depth + 1)
        elif isinstance(v, (list, tuple)) and len(v) > 8:
            out.append((name, f'[{len(v)} values]'))
        elif isinstance(v, float):
            out.append((name, f'{v:.6g}'))
        else:
            out.append((name, str(v)))
    return out


def _text_pages(title, blocks):
    """figures of monospace text: blocks = [(heading, [lines])]"""
    lines = []
    for h, ls in blocks:
        if not ls:
            continue
        lines.append(('h', h))
        for l in ls:
            while len(l) > 110:
                lines.append(('t', l[:110]))
                l = '    ' + l[110:]
            lines.append(('t', l))
        lines.append(('t', ''))
    per = 78
    figs = []
    for p in range(0, max(len(lines), 1), per):
        fig = _fig()
        _header(fig, title if p == 0 else title + ' (continued)')
        y = 0.93
        for kind, l in lines[p:p + per]:
            if kind == 'h':
                fig.text(0.06, y, l, fontsize=9, color=INK, weight='bold')
            else:
                fig.text(0.06, y, l, fontsize=6.8, color=INK2, family='monospace')
            y -= 0.0115
        figs.append(fig)
    return figs


def _archive_description(model):
    """Historical labels embedded in the reproducible psf.zip PDF."""
    if model.type_name == 'spline':
        return (f'CubicSplinePSF(roisize={model.roisize}, z_range=({model.zlo}, {model.zhi}), '
                f'oversample={model.S}, dz_um={model.hz}, zpad={model.zpad}, xy_limit={model.xmax}, '
                f'knots {(model.nz, model.nv, model.nu)})')
    if model.type_name == 'integrated_spline':
        return '<torch_psf.pixint.IntegratedSplinePSF object at 0x00000000>'
    return repr(model)


def s_summary(ctx):
    m = ctx.meta
    chans = m.get('channels', [])
    model = [f'type {m.get("psf_type")}, {len(chans)} channel(s): ' + ', '.join(f'{c["name"]} ({c["type"]})' for c in chans),
             f'emitter parameters: {", ".join(m.get("emitter_params", []))}']
    model += [f'{c}: {_archive_description(x)}' for c, x in enumerate(ctx.models)]
    st = ctx.meta.get('zstack', {})
    model.append(f'zstack.tif: {st.get("shape")} (ZCYX), {len(st.get("z_um", []))} planes, sample {st.get("sample_px")} px')
    blocks = [('Model', model)]
    if m.get('optics'):
        blocks.append(('Optics', [f'{k} = {v}' for k, v in _flat(m['optics'])]))
    if m.get('camera'):
        blocks.append(('Camera', [f'{k} = {v}' for k, v in _flat(m['camera']) if not k.startswith('arrays')] +
                       ([f'per-pixel maps: {", ".join(m["camera"]["arrays"])}'] if m['camera'].get('arrays') else [])))
    for c in chans:
        mp = c.get('mapping') or {}
        extra = [f'mapping: {mp.get("type", "identity")}'] + [f'{k}: {c[k]}' for k in ('crop', 'flip', 'intensity_ratio')
                                                             if k in c]
        if c.get('field'):
            extra.append(f'field: {c["field"].get("type")}')
        if len(chans) > 1 or len(extra) > 2:
            blocks.append((f'Channel {c["name"]}', extra))
    blocks.append(('Estimation', [f'estimator: {ctx.estimator or "(none recorded)"}']))
    if ctx.settings:
        blocks.append(('Settings', [f'{k} = {v}' for k, v in _flat(ctx.settings)]))
    res = {k: v for k, v in ctx.rep.items() if k != 'warnings'}
    if res:
        blocks.append(('Results', [f'{k} = {v}' for k, v in _flat(res)]))
    if ctx.rep.get('warnings'):
        blocks.append(('Warnings', [f'- {w}' for w in ctx.rep['warnings']]))
    if not ctx.rep and not ctx.history and not ctx.data:
        blocks.append(('Note', ['No estimator data: this report has the model pages only.']))
    sde = os.environ.get('SOURCE_DATE_EPOCH')       # no wall-clock time: the zip stays byte-reproducible
    title = 'PSF report' + (f'  ({time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(sde)))} UTC)' if sde else '')
    return _text_pages(title, blocks)


def s_spots(ctx):
    r = ctx.rep
    keys = [k for k in ('n_spots', 'n_train', 'n_test', 'n_rejected', 'n_stacks', 'n_beads', 'n_planes') if k in r]
    if not keys:
        return []
    fig = _fig()
    _header(fig, 'ROIs')
    ax = fig.add_axes([0.25, 0.70, 0.65, 0.2])
    labels = {'n_spots': 'detected spots', 'n_train': 'training spots', 'n_test': 'held-out spots',
              'n_rejected': 'rejected (overlap)', 'n_stacks': 'stacks', 'n_beads': 'beads', 'n_planes': 'planes per stack'}
    vals = [float(r[k]) for k in keys]
    ax.barh(range(len(keys)), vals, color=[SERIES[1] if k == 'n_rejected' else SERIES[0] for k in keys], height=0.6)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([labels[k] for k in keys], fontsize=7.5)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f' {v:g}', va='center', fontsize=7, color=INK2)
    _style(ax, 'Counts')
    notes = []
    dv = r.get('deviance')
    if dv:
        notes.append(f'Rejected: Poisson deviance per pixel above {dv["threshold"]:.3f} (rule {dv.get("rule")}, '
                     f'k = {dv.get("k")}; median {dv.get("median", float("nan")):.3f}): overlapping emitters or spots '
                     f'the single-emitter model does not explain. Rejected fraction {100 * dv.get("rejected_fraction", 0):.1f} %.')
    if 'at_limit' in r:
        notes.append(f'At the xy offset limit after the fit: {100 * float(r["at_limit"]):.1f} % of the spots.')
    if r.get('frames_read') is not None:
        notes.append(f'Frames read: {r["frames_read"]} (requested {r.get("frames_requested")}, span {r.get("frame_span")}).')
    if 'n_beads' in r and ctx.settings.get('holdout') is not None:
        notes.append(f'Stack {ctx.settings["holdout"]} held out of the fit.')
    y = 0.64
    for n in notes:
        import textwrap
        for l in textwrap.wrap(n, 120):
            fig.text(0.06, y, l, fontsize=7.5, color=INK2)
            y -= 0.016
        y -= 0.006
    d = ctx.data
    if dv and d.get('deviance') is not None:                 # the estimator's first-fit values
        dev, idx = np.asarray(d['deviance']), np.asarray(d['train'])
    elif dv and d.get('spots') is not None and d.get('vectorial_spline') is not None and len(d.get('train', [])):
        sp = d['spots']
        idx = np.asarray(d['train'])[:ctx.max_spots]
        ii = torch.as_tensor(idx, device=sp.rois.device)
        var = sp.var[ii] if (sp.var is not None and sp.var.dim() > 0) else None
        dev, _ = spot_dev(d['vectorial_spline'], sp.rois[ii], var)
    else:
        dev = None
    if dev is not None:
        ax = fig.add_axes([0.1, 0.12, 0.8, 0.3])
        hi = np.quantile(dev, 0.995)
        ax.hist(np.clip(dev, None, hi), bins=60, color=SERIES[0])
        ax.axvline(dv['threshold'], color=SERIES[1], lw=1.5)
        ax.text(dv['threshold'], ax.get_ylim()[1] * 0.95, ' threshold', color=INK2, fontsize=7, va='top')
        ax.set_xlabel('Poisson deviance per px (first fit)', fontsize=7.5)
        _style(ax, f'Overlap deviance, {len(idx)} training spots')
    return [fig]


def _hist(ax, v, label, color=SERIES[0]):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    lo, hi = np.quantile(v, [0.005, 0.995]) if len(v) > 20 else (v.min(), v.max())
    ax.hist(np.clip(v, lo, hi), bins=50, color=color)
    ax.axvline(np.median(v), color=INK, lw=1)
    _style(ax, f'{label}: median {np.median(v):.4g}')


def s_fit_stats(ctx):
    f = ctx.fit()
    if f is None:
        return []
    fig = _fig()
    _header(fig, 'Fit statistics')
    fig.text(0.06, 0.935, f['label'], fontsize=7.5, color=INK2)
    axs = [fig.add_axes([0.08 + 0.47 * (i % 2), 0.53 - 0.42 * (i // 2), 0.38, 0.33]) for i in range(4)]
    _hist(axs[0], f['chi2'], 'chi2 per px')
    if f['photons'] is not None:
        _hist(axs[1], f['photons'], 'photons')
    if f['bg'] is not None:
        _hist(axs[2], f['bg'], 'background (photons / px)')
    if f['z'] is not None and f['kind'] == 'spots':
        _hist(axs[3], f['z'], 'z (um)')
    elif f['z'] is not None:
        zu = np.unique(f['z'])
        med = [np.median(f['chi2'][f['z'] == z]) for z in zu]
        axs[3].plot(zu, med, color=SERIES[0], lw=2)
        axs[3].set_xlabel('z (um)', fontsize=7.5)
        _style(axs[3], 'median chi2 per plane')
    for ax in axs:
        if not ax.has_data():
            ax.set_visible(False)
    return [fig]


def s_convergence(ctx):
    h = [r for r in ctx.history if 'value' in r]
    scan = (ctx.rep.get('scan') or {}).get('results')
    if not h and not scan:
        return []
    fig = _fig()
    _header(fig, 'Convergence')
    stages = list(dict.fromkeys(r['stage'] for r in h))[:6]
    n = len(stages) + (1 if scan else 0)
    H = 0.82 / max(n, 1)
    for i, s in enumerate(stages):
        ax = fig.add_axes([0.1, 0.92 - H * (i + 1) + 0.04, 0.82, H - 0.07])
        v = np.array([r['value'] for r in h if r['stage'] == s])
        it = np.arange(len(v))
        ax.plot(it, v, color=SERIES[0], lw=1.5)
        if (v > 0).all() and v.max() / max(v.min(), 1e-30) > 30:
            ax.set_yscale('log')
        ax.set_xlabel('iteration', fontsize=7)
        _style(ax, f'{s}: loss {v[0]:.6g} -> {v[-1]:.6g}')
    if scan:
        ax = fig.add_axes([0.1, 0.92 - H * n + 0.04, 0.82, H - 0.07])
        fs = np.array([q['focus_um'] for q in scan])
        ws = np.array([q['wavelength_um'] for q in scan])
        nl = np.array([q['heldout_nll_per_px'] for q in scan])
        for j, w in enumerate(np.unique(ws)):
            s_ = ws == w
            o = np.argsort(fs[s_])
            ax.plot(fs[s_][o], nl[s_][o], '-o', ms=4, lw=1.5, color=SERIES[j % len(SERIES)], label=f'{w:g} um')
        b = ctx.rep['scan']['best']
        ax.plot([b['focus_um']], [b['heldout_nll_per_px']], 'o', ms=9, mfc='none', mec=INK)
        if len(np.unique(ws)) > 1:
            ax.legend(fontsize=7, frameon=False, title='wavelength', title_fontsize=7)
        ax.set_xlabel('focus (um)', fontsize=7)
        _style(ax, 'focus / wavelength scan: held-out NLL per px')
    return [fig]


def s_aberrations(ctx):
    from torch_psf.zernike import NOLL_NAMES
    vs = _vectorials(ctx)
    if not vs:
        return []
    fig = _fig()
    _header(fig, 'Aberrations')
    ax = fig.add_axes([0.1, 0.52, 0.85, 0.36])
    noll = sorted({j for _, v in vs for j in v.zidx})
    w = 0.8 / len(vs)
    err = ctx.rep.get('zernike_std')
    for i, (lab, v) in enumerate(vs[:4]):
        zc = dict(zip(v.zidx, _np(v.zcoef).tolist()))
        y = np.array([1000 * zc.get(j, np.nan) for j in noll])
        e = None
        if err is not None and len(vs) == 1 and len(err) == len(v.zidx):
            ed = dict(zip(v.zidx, err))
            e = np.array([1000 * float(ed.get(j, np.nan)) for j in noll])
        ax.bar(np.arange(len(noll)) + (i - (len(vs) - 1) / 2) * w, y, w * 0.9, yerr=e, color=SERIES[i], label=lab,
               error_kw=dict(lw=0.8, ecolor=INK, capsize=2))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(len(noll)))
    ax.set_xticklabels([f'{j} {NOLL_NAMES.get(j, "")}' for j in noll], rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel('m-waves rms', fontsize=7.5)
    if len(vs) > 1:
        ax.legend(fontsize=7, frameon=False)
    _style(ax, 'Zernike coefficients (Noll)' + (' with standard errors' if err is not None and len(vs) == 1 else ''))
    lines = []
    r = ctx.rep
    for lab, v in vs:
        if hasattr(v, 'focus_um'):
            lines.append(f'{lab}: focus_um = {v.focus_um:.4g}, NA {float(v.NA):.4g}, n_med {float(v.n_med):.4g}, '
                         f'wavelength {v.wavelength_um:g} um')
    for k, lab in (('focus_um', 'estimated focus (um)'), ('z_offset_um', 'z offset vs the start model (um)'),
                   ('z_scale', 'z scale vs the start model'), ('dz_scale', 'stage dz scale'),
                   ('height_um', 'bead height (um)')):
        if r.get(k) is not None:
            lines.append(f'{lab}: {float(r[k]):.4g}')
    y = 0.36
    for l in lines:
        fig.text(0.08, y, l, fontsize=7.5, color=INK2)
        y -= 0.018
    return [fig]


def _pupil_maps(v):
    with torch.no_grad():
        mask = _np(v.mask) > 0
        Z = torch.einsum('j,jnm->nm', v.zcoef.detach(), v.Z)
        r2 = v.rho ** 2
        a = v.apod.detach()
        amp0 = torch.exp(a[0] * r2 + a[1] * r2 ** 2 + a[2] * r2 ** 3)
        out = dict(phase_z=_np(Z), phase=_np(Z + v.pupil_phase), amp_z=_np(amp0), amp=_np(amp0 * v.pupil_amp))
    for k in out:
        out[k] = np.where(mask, out[k], np.nan)
    has_maps = bool((_np(v.pupil_phase) != 0).any() or (_np(v.pupil_amp) != 1).any())
    return out, has_maps


def s_pupil(ctx):
    vs = _vectorials(ctx)[:4]
    if not vs:
        return []
    fig = _fig()
    _header(fig, 'Pupil')
    H = 0.22
    for i, (lab, v) in enumerate(vs):
        maps, has = _pupil_maps(v)
        ph = np.nanmax(np.abs(maps['phase'])) or 1e-3
        am = (0.0, float(np.nanmax(maps['amp'])))
        panels = [('phase_z', 'phase, Zernike only (waves)'), ('phase', 'phase with pupil map (waves)'),
                  ('amp_z', 'amplitude, apodization only'), ('amp', 'amplitude with pupil map')]
        for j, (k, t) in enumerate(panels):
            ax = fig.add_axes([0.04 + 0.24 * j, 0.92 - H * (i + 1) + 0.03, 0.2, H - 0.06])
            if k.startswith('phase'):
                im = ax.imshow(maps[k], cmap='RdBu_r', vmin=-ph, vmax=ph)
            else:
                im = ax.imshow(maps[k], cmap='gray', vmin=am[0], vmax=am[1] or 1.0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(t, fontsize=6.5, color=INK2)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5.5)
        fig.text(0.04, 0.925 - H * i, lab + ('' if has else '  (no free pupil maps)'), fontsize=8, color=INK)
    return [fig]


def s_psf(ctx):
    from torch_psf.psfzip import psf_figure
    st = ctx.meta.get('zstack', {})
    names = [c['name'] for c in ctx.meta.get('channels', [])] or None
    C = ctx.stack.shape[1]
    fig = _fig(A4L)
    _header(fig, 'PSF: xy slices through z and xz section (normalized per panel)')
    psf_figure(ctx.stack, ctx.z_um, names, st.get('sample_px', 1.0), fig=fig,
               rect=(0.06, max(0.06, 0.9 - 0.21 * C), 0.97, 0.9))
    return [fig]


def _pick(f, n=4):
    if f['kind'] == 'zstack':
        b = int(np.bincount(f['bead'], weights=np.nan_to_num(f['photons']) if f['photons'] is not None else None).argmax())
        idx = np.flatnonzero(f['bead'] == b)
        return idx[np.round(np.linspace(0, len(idx) - 1, n)).astype(int)]
    o = np.argsort(f['z']) if f['z'] is not None else np.argsort(-f['photons'])
    return o[np.round(np.linspace(0, len(o) - 1, min(n, len(o)))).astype(int)]


def s_examples(ctx):
    f = ctx.fit()
    if f is None or not len(f['rois']):
        return []
    idx = _pick(f)
    fig = _fig()
    _header(fig, 'Example ROIs: data / model / data - model')
    n = len(idx)
    H = 0.86 / max(n, 1)
    for i, k in enumerate(idx):
        d, m = f['rois'][k], f['mu'][k]
        lo, hi = float(min(d.min(), m.min())), float(max(d.max(), m.max()))
        r = d - m
        rl = float(np.abs(r).max()) or 1.0
        for j, (img, cm, vmin, vmax, t) in enumerate(((d, 'gray', lo, hi, 'data'), (m, 'gray', lo, hi, 'model'),
                                                      (r, 'RdBu_r', -rl, rl, 'data - model'))):
            ax = fig.add_axes([0.06 + 0.3 * j, 0.92 - H * (i + 1) + 0.02, 0.24, H - 0.06])
            im = ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5.5)
            if j == 0:
                z = '' if f['z'] is None else f'z {f["z"][k]:.3f} um, '
                ph = '' if f['photons'] is None else f'{f["photons"][k]:.0f} photons, '
                ax.set_title(f'{z}{ph}chi2 {f["chi2"][k]:.2f}', fontsize=7, color=INK2, loc='left')
            else:
                ax.set_title(t, fontsize=7, color=INK2)
    return [fig]


QBANDS = (('best', 'Best: the lowest q', SERIES[2]),
          ('accepted', 'Last accepted: just below the cutoff', SERIES[0]),
          ('rejected', 'First rejected: just above the cutoff', SERIES[1]))


def _bands(q, accepted, n=3):
    """indices of the n lowest q, the n accepted ones with the highest q (not already among the best) and the n
    not-accepted ones with the lowest q"""
    o = np.argsort(q, kind='stable')
    best = o[:n]
    acc = o[accepted[o] & ~np.isin(o, best)][-n:]
    rej = o[~accepted[o] & ~np.isin(o, best)][:n]
    return dict(best=best, accepted=acc, rejected=rej)


def _ref_cutoff(q):
    """reference line median + 3 x 1.4826 MAD (no estimator rejection)"""
    med = float(np.median(q))
    return med + 3 * 1.4826 * float(np.median(np.abs(q - med)))


@torch.no_grad()
def _quality(ctx):
    """per-ROI quality score q (higher = worse), the cutoff and the ROIs of the three bands, or None:
    dict(q, accepted [N] bool, cutoff, kind ('rejection' | 'reference' | None), qlabel, notes, rows {band: [row]}),
    row = dict(data, model [P, P], q, rank, photons, bg, z, label)"""
    d, rep = ctx.data, ctx.rep
    if d.get('rois') is not None and d.get('model') is not None and d.get('chi2') is not None:   # ZStackEstimator
        rois, mu, c2 = _np(d['rois']), _np(d['model']), _np(d['chi2'])
        nb, nk = c2.shape
        q = c2.mean(1)
        cut = _ref_cutoff(q) if nb >= 3 else None
        acc = np.ones(nb, bool) if cut is None else q < cut
        sid = _np(d.get('stack_id'))
        ho = ctx.settings.get('holdout')
        foc = _np(d.get('focus_um'))
        per_plane = lambda a: None if a is None else np.reshape(a, (nb, -1))[:, np.arange(nk) % np.size(a[0])]
        ph, bg = per_plane(_np(d.get('photons'))), per_plane(_np(d.get('bg')))
        o = np.argsort(q, kind='stable')
        rank = np.empty(nb, int)
        rank[o] = np.arange(nb)

        def row(b):
            k = int(mu[b].reshape(nk, -1).max(1).argmax())             # the plane with the sharpest model
            tag = '' if sid is None else f' (stack {int(sid[b])}{", held out" if ho is not None and int(sid[b]) == ho else ""})'
            return dict(data=rois[b, k], model=mu[b, k], q=float(q[b]), rank=int(rank[b]),
                        photons=None if ph is None else float(ph[b, k]), bg=None if bg is None else float(bg[b, k]),
                        z=None if foc is None else 0.0 - float(foc[k]), label=f'bead {b}{tag}, plane {k + 1} of {nk} (sharpest)')
        notes = [f'q = chi2 per px of a bead, (data - model)^2 / model averaged over its {nk} planes and the pixels '
                 f'(about 1 when the model explains the data); images: the plane with the sharpest model.']
        if cut is not None:
            notes.append(f'The ZStackEstimator rejects no beads. Reference line (not a rejection): median + 3 x 1.4826 '
                         f'MAD = {cut:.3f}; {int((~acc).sum())} of {nb} beads lie above it (none rejected: the estimator rejects no beads).')
        bands = _bands(q, acc)
        return dict(q=q, accepted=acc, cutoff=cut, kind=None if cut is None else 'reference', notes=notes,
                    qlabel='chi2 per px of a bead (mean over its planes)', unit='beads',
                    rows={b: [row(int(i)) for i in ix] for b, ix in bands.items()})
    sp, dvr = d.get('spots'), rep.get('deviance')
    tr = np.asarray(d.get('train', []))
    if sp is None or not dvr or not len(tr):
        return None
    if d.get('deviance') is None or d.get('deviance_psf') is None or d.get('deviance_theta') is None:
        return None
    q, psf, th = np.asarray(d['deviance'], float), d['deviance_psf'], d['deviance_theta']
    ii = torch.as_tensor(tr, device=sp.rois.device)
    thr = float(dvr['threshold'])
    acc = q < thr
    o = np.argsort(q, kind='stable')
    rank = np.empty(len(q), int)
    rank[o] = np.arange(len(q))
    bands = _bands(q, acc)
    sel = np.concatenate(list(bands.values())).astype(int)
    st = torch.as_tensor(sel, device=ii.device)
    th_sel = torch.as_tensor(_np(th)[sel])
    dev_ = getattr(psf, 'dev', None) or th_sel.device
    th_sel = th_sel.to(dev_)
    mu = _np(psf.expected(th_sel))
    thn = _np(th_sel)
    data = _np(sp.rois[ii[st]])
    fr = _np(getattr(sp, 'frame', None))
    has_z = getattr(psf, 'z_range', None) is not None
    pos = {int(i): j for j, i in enumerate(sel)}

    def row(i):
        j = pos[int(i)]
        return dict(data=data[j], model=mu[j], q=float(q[i]), rank=int(rank[i]), photons=float(thn[j, 3]),
                    bg=float(thn[j, 4]), z=float(thn[j, 2]) if has_z else None,
                    label=f'training spot {int(i)}' + ('' if fr is None else f', frame {int(fr[tr[i]])}'))
    rule = (f'median + {dvr.get("k")} x max(1.4826 MAD, median sqrt(2 / P^2))' if dvr.get('rule') == 'median' else
            f'rule {dvr.get("rule")}')
    notes = [f'q = Poisson deviance per px of a training spot after its single-emitter fit with the first-fit PSF (the '
             f'values the estimator used; about 1 when the model explains the spot; overlapping emitters give more). '
             f'Model images: that fit.',
             f'Rejection (InSituEstimator overlap rule {rule}): q >= {thr:.3f}; {int((~acc).sum())} of {len(q)} '
             f'spots rejected, the PSF was refitted without them.']
    return dict(q=q, accepted=acc, cutoff=thr, kind='rejection', notes=notes, unit='training spots',
                qlabel='Poisson deviance per px (first fit)', rows={b: [row(i) for i in ix] for b, ix in bands.items()})


def spot_dev(psf, rois, var):
    from torch_psf.estimation import spot_deviance
    return spot_deviance(psf, rois, var)


def s_quality(ctx):
    qd = _quality(ctx)
    if qd is None or not len(qd['q']):
        return []
    import textwrap
    from matplotlib.transforms import blended_transform_factory
    q, acc, cut, kind = np.asarray(qd['q'], np.float64), qd['accepted'], qd['cutoff'], qd['kind']
    n = len(q)
    fig = _fig()
    _header(fig, 'ROI quality range: best, last accepted, first rejected')
    y = 0.945
    for t in qd['notes']:
        for l in textwrap.wrap(t, 130):
            fig.text(0.06, y, l, fontsize=6.8, color=INK2)
            y -= 0.0125
    titles = dict(reference=dict(accepted='Just below the reference line',
                                 rejected='Just above the reference line (kept: the estimator rejects nothing)'),
                  none=dict(accepted='Worst')).get(kind or 'none', {})
    bands = [(b, titles.get(b, t), SERIES[3] if b == 'rejected' and kind != 'rejection' else c) for b, t, c in QBANDS]
    # histogram of q, the cutoff, ticks at the shown ROIs
    lo = float(q.min())
    hi = float(np.quantile(q, 0.99)) if n > 20 else float(q.max())       # long tails (overlaps) in the last bin
    if cut is not None:
        hi = max(float(np.quantile(q, 0.95)) if n > 20 else -np.inf, cut + 0.6 * (cut - lo))
    hi = hi if hi > lo else lo + 1.0
    bins = np.linspace(lo, hi, int(np.clip(np.sqrt(n) * 2, 30, 70)) + 1)
    ax = fig.add_axes([0.1, y - 0.205, 0.84, 0.17])
    qc = np.clip(q, lo, hi)
    ax.hist([qc[acc], qc[~acc]] if (~acc).any() else [qc], bins=bins, stacked=True,
            color=[SERIES[0], SERIES[1] if kind == 'rejection' else SERIES[3]][:2 if (~acc).any() else 1],
            label=(['accepted', 'rejected'] if kind == 'rejection' else ['below the reference', 'above the reference (kept)'])
            [:2 if (~acc).any() else 1])
    if n > 100:
        ax.set_yscale('log')
    tr_ = blended_transform_factory(ax.transData, ax.transAxes)
    if cut is not None:
        ax.axvline(cut, color=SERIES[1] if kind == 'rejection' else INK2, lw=1.4, ls='-' if kind == 'rejection' else '--')
        ax.text(cut, 0.97, f' rejection threshold {cut:.3f}' if kind == 'rejection' else f' reference {cut:.3f} (no rejection)',
                transform=tr_, fontsize=7, color=INK, va='top')
    for b, col, ym in ((b, c, -0.2 if b == 'rejected' else -0.15) for b, _, c in bands):   # two rows of marks
        xs = np.asarray([r['q'] for r in qd['rows'].get(b, [])])
        if len(xs):
            ax.plot(np.clip(xs[xs <= hi], lo, hi), [ym] * int((xs <= hi).sum()), '|', ms=8, mew=1.6, color=col,
                    transform=tr_, clip_on=False)
            if (xs > hi).any():                                          # beyond the axis: an arrow at the edge
                ax.plot([hi + 0.012 * (hi - lo)], [ym], '>', ms=5, color=col, transform=tr_, clip_on=False)
    if (q > hi).any():
        m = int((q > hi).sum())
        ax.text(1.0, -0.36, f'{m} value{"s" if m > 1 else ""} above {hi:.3g} counted in the last bin',
                transform=ax.transAxes, fontsize=6.5, color=INK2, ha='right', va='top')
    ax.set_xlim(lo - 0.01 * (hi - lo), hi + 0.01 * (hi - lo))
    ax.set_xlabel(qd['qlabel'], fontsize=7.5, labelpad=22)
    if (~acc).any():
        ax.legend(fontsize=6.5, frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
    _style(ax, f'q of {n} {qd["unit"]}; marks below the axis: the ROIs shown')
    # bands of rows: caption | data | model | data - model
    rows = [(b, t, c, qd['rows'].get(b, [])) for b, t, c in bands]
    rows = [(b, t, c, rr) for b, t, c, rr in rows if rr]
    nrow = sum(len(rr) for *_, rr in rows)
    top = y - 0.29
    H = min(0.066, (top - 0.035 - 0.012 * len(rows)) / (nrow + 0.45 * len(rows)))
    yy = top
    for b, t, c, rr in rows:
        fig.text(0.06, yy - 0.018, t, fontsize=8.5, color=c, weight='bold')
        yy -= 0.45 * H
        for r in rr:
            yy -= H
            dd, m = r['data'], r['model']
            lo_, hi_ = float(min(dd.min(), m.min())), float(max(dd.max(), m.max()))
            res = dd - m
            rl = float(np.abs(res).max()) or 1.0
            pc = 100 * (r['rank'] + 1) / n
            cap = [f'q {r["q"]:.3f}   rank {r["rank"] + 1} of {n} ({pc:.2f} %)' if pc < 1 else
                   f'q {r["q"]:.3f}   rank {r["rank"] + 1} of {n} ({pc:.1f} %)',
                   ', '.join(s for s in (None if r['photons'] is None else f'{r["photons"]:.0f} photons',
                                         None if r['bg'] is None else f'bg {r["bg"]:.1f}',
                                         None if r['z'] is None else f'z {r["z"]:.3f} um') if s),
                   r['label']]
            for k, l in enumerate(cap):
                fig.text(0.075, yy + H * (0.68 - 0.24 * k), l, fontsize=7 if k == 0 else 6.5,
                         color=INK if k == 0 else INK2, weight='bold' if k == 0 else 'normal')
            fig.patches.append(__import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle(
                (0.06, yy + 0.1 * H), 0.006, 0.8 * H, transform=fig.transFigure, color=c))
            w = H * A4[1] / A4[0] * 0.92                                       # square panels
            for j, (img, cm, v0, v1, tt) in enumerate(((dd, 'gray', lo_, hi_, 'data'), (m, 'gray', lo_, hi_, 'model'),
                                                       (res, 'RdBu_r', -rl, rl, 'data - model'))):
                x0 = 0.42 + j * (w + 0.035) + (0.02 if j == 2 else 0)
                a = fig.add_axes([x0, yy + 0.04 * H, w, 0.92 * H])
                im = a.imshow(img, cmap=cm, vmin=v0, vmax=v1, interpolation='nearest')
                a.set_xticks([]); a.set_yticks([])
                if r is rows[0][3][0]:
                    a.set_title(tt, fontsize=7, color=INK2)
                if j >= 1:
                    cax = fig.add_axes([x0 + w + 0.004, yy + 0.04 * H, 0.008, 0.92 * H])
                    cb = fig.colorbar(im, cax=cax)
                    cb.ax.tick_params(labelsize=5, length=2, pad=1)
                    cb.outline.set_linewidth(0.4)
        yy -= 0.012
    fig.text(0.06, 0.018, 'data and model on a shared linear scale (photons) per ROI (colour bar next to the model); '
             'data - model on a symmetric scale', fontsize=6.5, color=INK2)
    return [fig]


def s_residuals(ctx):
    f = ctx.fit()
    if f is None or f['z'] is None or f['photons'] is None or len(f['rois']) < 4:
        return []
    z = f['z']
    zu = np.unique(z)
    nb = int(min(6, len(zu)))
    if f['kind'] == 'zstack':
        groups = np.array_split(zu, nb)
        sel = [np.isin(z, g) for g in groups]
    else:
        e = np.quantile(z, np.linspace(0, 1, nb + 1))
        sel = [(z >= e[i]) & (z <= e[i + 1]) for i in range(nb)]
    res = (f['rois'] - f['mu']) / np.maximum(f['photons'], 1.0)[:, None, None]
    means = [res[s].mean(0) for s in sel if s.any()]
    zr = [(z[s].min(), z[s].max(), int(s.sum())) for s in sel if s.any()]
    lim = max(float(np.abs(m).max()) for m in means) or 1e-6
    fig = _fig()
    _header(fig, 'Mean residuals by z: (data - model) / photons')
    fig.text(0.06, 0.935, f['label'], fontsize=7.5, color=INK2)
    for i, (m, (a, b, n)) in enumerate(zip(means, zr)):
        ax = fig.add_axes([0.05 + 0.31 * (i % 3), 0.62 - 0.3 * (i // 3), 0.26, 0.26])
        im = ax.imshow(m, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'z {a:.2f} .. {b:.2f} um (n = {n})', fontsize=7, color=INK2)
    cax = fig.add_axes([0.25, 0.04, 0.5, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=6)
    cb.set_label('fraction of the photons per px', fontsize=7)
    return [fig]


def _field_grid(fb, crop, n=25):
    if crop and crop.get('w') and crop.get('h'):
        x = np.linspace(0, float(crop['w']) - 1, n)
        y = np.linspace(0, float(crop['h']) - 1, n)
    else:
        cx, cy, sx, sy = fb.norm
        x, y = np.linspace(cx - sx, cx + sx, n), np.linspace(cy - sy, cy + sy, n)
    return np.meshgrid(x, y)


def s_field(ctx):
    from torch_psf.psfzip import FieldZernikePoly, Mapping, poly_features
    from torch_psf.zernike import NOLL_NAMES
    figs = []
    for i, c in enumerate(ctx.meta.get('channels', [])):
        fd = c.get('field')
        if fd and fd.get('type') == 'zernike_poly':
            fb = FieldZernikePoly(fd, {}, c.get('crop'))
            X, Y = _field_grid(fb, c.get('crop'))
            W = fb.zernike_at(X, Y)                                   # [n, n, nz]
            ref = fb.zernike_at(*fb.ref)
            var = [k for k in range(len(fb.noll)) if np.abs(fb.coef[k, 1:]).max() > 0][:12]
            if var:
                fig = _fig()
                _header(fig, f'Field dependence, channel {c["name"]}: Zernikes over the field')
                fig.text(0.06, 0.935, f'm-waves rms minus the value at the reference point ({fb.ref[0]:g}, '
                         f'{fb.ref[1]:g}) px; channel coordinates', fontsize=7.5, color=INK2)
                for p, k in enumerate(var):
                    dW = 1000 * (W[..., k] - ref[k])
                    lim = float(np.abs(dW).max()) or 1e-6
                    ax = fig.add_axes([0.05 + 0.235 * (p % 4), 0.72 - 0.22 * (p // 4), 0.19, 0.17])
                    im = ax.imshow(dW, cmap='RdBu_r', vmin=-lim, vmax=lim, origin='upper',
                                   extent=(X.min(), X.max(), Y.max(), Y.min()))
                    ax.tick_params(labelsize=5.5)
                    j = fb.noll[k]
                    ax.set_title(f'Z{j} {NOLL_NAMES.get(j, "")}\nref {1000 * ref[k]:.1f}', fontsize=6.5, color=INK2)
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cb.ax.tick_params(labelsize=5)
                figs.append(fig)
        mp = c.get('mapping') or {}
        if mp.get('type') == 'poly' and i > 0:
            m = Mapping(mp, i, {}, c.get('crop'))
            if m.type != 'poly':
                continue
            crop = c.get('crop')
            n = 15
            if crop and crop.get('w'):
                xs, ys = np.linspace(0, crop['w'] - 1, n), np.linspace(0, crop['h'] - 1, n)
            else:
                cx, cy, sx, sy = m.norm
                xs, ys = np.linspace(cx - sx, cx + sx, n), np.linspace(cy - sy, cy + sy, n)
            X, Y = np.meshgrid(xs, ys)
            x0, y0 = m.to_ref(X, Y)
            A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
            Mx, *_ = np.linalg.lstsq(A, x0.ravel(), rcond=None)
            My, *_ = np.linalg.lstsq(A, y0.ravel(), rcond=None)
            dx, dy = x0.ravel() - A @ Mx, y0.ravel() - A @ My
            back = m.from_ref(x0, y0)
            rt = np.sqrt(((back[0] - X) ** 2 + (back[1] - Y) ** 2).mean())
            fig = _fig()
            _header(fig, f'Mapping, channel {c["name"]} -> {ctx.meta["channels"][0]["name"]}: non-affine part')
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.58])
            mag = np.hypot(dx, dy)
            ax.quiver(X.ravel(), Y.ravel(), dx, dy, color=SERIES[0], angles='xy')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xlabel('x (channel px)', fontsize=7.5); ax.set_ylabel('y (channel px)', fontsize=7.5)
            _style(ax, f'distortion after the best affine fit: max {mag.max():.3f} px, rms {np.sqrt((mag ** 2).mean()):.3f} px')
            lines = [f'affine part: x0 = {Mx[0]:.5f} x + {Mx[1]:.5f} y + {Mx[2]:.3f}; '
                     f'y0 = {My[0]:.5f} x + {My[1]:.5f} y + {My[2]:.3f}',
                     f'polynomial order {m.order}; round trip inv(fwd(p)) - p: rms {rt:.4f} px']
            lines += [f'{k} = {v}' for k, v in _flat({k: v for k, v in mp.items()
                                                       if k not in ('fwd', 'inv', 'order', 'feats_norm', 'type')})][:20]
            y = 0.24
            for l in lines:
                fig.text(0.08, y, l[:130], fontsize=7, color=INK2)
                y -= 0.016
            figs.append(fig)
    return figs


SECTIONS = (('summary', s_summary), ('spots', s_spots), ('quality', s_quality), ('fit_stats', s_fit_stats), ('convergence', s_convergence),
            ('aberrations', s_aberrations), ('pupil', s_pupil), ('psf', s_psf), ('examples', s_examples),
            ('residuals', s_residuals), ('field', s_field))


def write_report(psf, meta, stack, z_um, report=None, source=None, max_spots=1000):
    """(pdf bytes, names of the sections written). A section that fails is skipped with a warning (the PSF file is
    still written)."""
    from matplotlib.backends.backend_pdf import PdfPages
    ctx = _Ctx(psf, meta, stack, z_um, bundle(report, psf), source, max_spots)
    bio = _io.BytesIO()
    written = []
    with PdfPages(bio, metadata=dict(Title='torch-psf fit report', Creator='torch_psf.psfreport', CreationDate=None)) as pdf:
        for name, fn in SECTIONS:
            try:
                figs = fn(ctx)
            except Exception as e:      # noqa: BLE001  (the report must not stop the PSF export)
                warnings.warn(f'fit report: section {name!r} skipped ({type(e).__name__}: {e})')
                continue
            for f in figs:
                pdf.savefig(f)
            if figs:
                written.append(name)
    return bio.getvalue(), written
