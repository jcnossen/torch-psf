"""Figures of the auto analysis: radial spectrum with the fitted cutoff, and the classifier statistics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_result(r, path, title='', expected_fc=None, expected_label=None):
    """r: AutoResult. expected_fc: list of (fc_px, label) reference cutoffs."""
    b, sp, m = r.band, r.spectrum, r.mode
    st = m.stats
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    a.semilogy(sp.f, sp.p, '.', ms=3, label='data')
    a.semilogy(sp.f, b.model(sp.f), '-', lw=1, label='fit')
    a.axhline(b.floor, color='0.5', ls=':', lw=1, label='floor')
    a.set_xlabel('f (cycles / px)'); a.set_ylabel('power (data units^2 / px)'); a.set_title('radial power spectrum')
    a = ax[0, 1]
    ex = (sp.p - b.floor) / b.floor
    a.semilogy(sp.f, np.where(ex > 0, ex, np.nan), '.', ms=3, label='(p - floor) / floor')
    a.semilogy(sp.f, sp.se / b.floor, '-', color='0.6', lw=1, label='1 se')
    a.semilogy(sp.f, np.clip((b.model(sp.f) - b.floor) / b.floor, 1e-12, None), '-', lw=1, label='fit')
    lo = max(np.nanmin(sp.se / b.floor) / 3, 1e-6)
    a.set_ylim(lo, max(np.nanmax(ex) * 2, 10 * lo))
    a.set_xlabel('f (cycles / px)'); a.set_title('excess over the noise floor')
    for aa in (ax[0, 0], ax[0, 1]):
        aa.axvline(b.fc_px, color='r', lw=1.5, label=f'fit f_c {b.fc_px:.3f}')
        aa.axvspan(b.fc_px - b.fc_px_err, b.fc_px + b.fc_px_err, color='r', alpha=0.15)
        aa.axvline(b.f_detect_px, color='orange', ls='--', lw=1, label=f'f_detect {b.f_detect_px:.3f}')
        for k, (fe, lab) in enumerate(expected_fc or []):
            aa.axvline(fe, color=['g', 'c', 'm'][k % 3], ls='-.', lw=1, label=f'{lab} {fe:.3f}')
        aa.axvline(0.5, color='k', lw=0.5)
    ax[0, 1].legend(fontsize=7, loc='upper right')
    a = ax[0, 2]
    S = sp.tile
    p2 = np.log10(np.maximum(sp.p2d, 1e-12))
    ext = [-0.5, 0.5 - 1 / S, 0.5 - 1 / S, -0.5]
    a.imshow(p2, extent=ext, cmap='magma', vmin=np.percentile(p2, 1), vmax=np.percentile(p2, 99.7))
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(b.fc_px * np.cos(t), b.fc_px * np.sin(t), 'c-', lw=0.8)
    for k, (fe, lab) in enumerate(expected_fc or []):
        a.plot(fe * np.cos(t), fe * np.sin(t), ['g', 'c', 'm'][k % 3] + '-.', lw=0.8)
    a.set_title('log10 2D spectrum (cyan: fitted f_c)'); a.set_xlabel('fx (cycles / px)'); a.set_ylabel('fy')
    a = ax[1, 0]
    if 'sharpness' in st:
        x = np.arange(len(st['sharpness']))
        a.plot(x, st['sharpness'], '.', ms=2, label='high-band fraction')
        a.plot(x, st['sharpness_smooth'], '-', label='running mean')
        if 'astig_signed' in st:
            a.plot(x, st['astig_signed'], '-', label='signed anisotropy (cos 2 theta)')
        a.legend(fontsize=7)
    a.set_title(f"per frame: trend R2 {st.get('trend_r2', np.nan):.2f}, range {st.get('trend_range', np.nan):.2f}, "
                f"astig swing {st.get('astig_swing', np.nan):.2f}", fontsize=9)
    a.set_xlabel('frame (classifier block)')
    a = ax[1, 1]
    rho = st.get('rho', {}); pers = st.get('persist', {})
    if rho:
        a.semilogx(list(map(int, rho)), list(rho.values()), 'o-', label='rho(L), fluctuations')
    if pers:
        a.semilogx(list(map(int, pers)), list(pers.values()), 's-', label='spot persistence(L)')
    a.axhline(0, color='0.5', lw=0.5); a.set_ylim(-0.6, 1.05); a.set_xlabel('lag L (frames)'); a.legend(fontsize=7)
    a.set_title('temporal correlation')
    a = ax[1, 2]
    a.axis('off')
    k = m.key
    fmt = lambda v: 'n/a' if v is None or (isinstance(v, float) and not np.isfinite(v)) else (
        f'{v:.3g}' if isinstance(v, float) else str(v))
    lines = [f'mode: {m.mode}  (confidence {m.confidence:.2f})',
             'P: ' + ', '.join(f'{c} {v:.2f}' for c, v in m.probs.items()), '',
             f'band: {b.status}; f_c = {b.fc_px:.4f} +- {b.fc_px_err:.4f} c/px']
    if b.na_over_lambda is not None:
        lines.append(f'NA/lambda = {b.na_over_lambda:.3f} +- {b.na_over_lambda_err:.3f} /um')
    if b.NA is not None:
        lines.append(f'NA_eff = {b.NA:.3f} +- {b.NA_err:.3f}')
    if b.anisotropy and 'ellipse_ratio' in b.anisotropy:
        an = b.anisotropy
        lines.append(f"support ellipse {an['ellipse_ratio']:.3f} (long {an['ellipse_long_axis_deg']:.0f} deg); "
                     f"contrast m2 {an['contrast_m2']:.3f} +- {an['contrast_m2_se']:.3f}")
    lines.append('')
    lines += [f'{kk}: {fmt(v)}' for kk, v in k.items()]
    a.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=7.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
