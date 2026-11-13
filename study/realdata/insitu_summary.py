"""Collect all numbers of the real-data in-situ test into insitu/insitu_summary.json (CPU)."""
import sys, json, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
from insitu_common import OUT, jdump

S = dict(readme='Real-data test of in-situ vectorial PSF refinement through the spline knots (NPC-A647-3D). '
                'Scripts: study/realdata/insitu_*.py. Test set: all flags == 0 spots of the odd 1000-frame blocks; '
                'training: 150,000 spots of the even blocks; validation: 30,000 further even-block spots.')
keys = ['tag', 'noll', 'NA', 'pxy', 'lam', 'f0', 's', 'apod', 'bead_height', 'blur_sigma', 'zscale', 'poly', 'fixed',
        'nll', 'nll_spline', 'n_pix', 'chi2_mean', 'chi2_bead_median', 'chi2_spline_mean', 'chi2_spline_bead_median',
        'chi2_per_z_vec_spline', 'seconds', 'n_params_global', 'pupil_phase_rms', 'pupil_amp_rms', 'zcoef']
S['bead_fits'] = {}
for f in sorted(glob.glob(f'{OUT}/vecbeads_*.json')):
    if os.path.islink(f) or 'test' in f or '_s1_' in f or '_s-1_' in f:
        continue
    r = json.load(open(f))
    S['bead_fits'][r['tag']] = {k: r[k] for k in keys if k in r}
S['bead_fit_used_for_b'] = 'n36_pri'
S['in_situ_fits'] = {}
for f in sorted(glob.glob(f'{OUT}/fit_*.json')):
    r = json.load(open(f))
    tag = os.path.basename(f)[4:-5]
    S['in_situ_fits'][tag] = r
for f in sorted(glob.glob(f'{OUT}/state_*.json')):
    r = json.load(open(f))
    tag = os.path.basename(f)[6:-5]
    S['in_situ_fits'].setdefault(tag, {})['final_state'] = {k: v for k, v in r.items()
                                                             if k not in ('pupil_phase', 'pupil_amp', 'bead_xy', 'bead_dz',
                                                                          'chi2_bead', 'chi2_per_z_vec_spline')}
for t in ('main', 'win3000'):
    if os.path.exists(f'{OUT}/results_{t}.json'):
        S[f'test_results_{t}'] = json.load(open(f'{OUT}/results_{t}.json'))
if os.path.exists(f'{OUT}/timing.json'):
    S['timing_clean'] = json.load(open(f'{OUT}/timing.json'))
# profile likelihood of the optical constants (bead fits started from n36_pri with one constant fixed, same budget)
prof = {k: v for k, v in S['bead_fits'].items() if k.startswith('prof_')}
if 'prof_ref' in prof:
    ref = prof['prof_ref']['nll']
    S['profiles'] = {k: dict(dnll=v['nll'] - ref, chi2_mean=v['chi2_mean'], NA=v['NA'], pxy=v['pxy'], lam=v['lam'],
                             bead_height=v['bead_height']) for k, v in prof.items()}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for i, (pre, key, lab) in enumerate((('prof_NA', 'NA', 'NA'), ('prof_lam', 'lam', 'wavelength (um)'),
                                         ('prof_ratio', None, 'px / py'), ('prof_he', 'bead_height', 'bead centre height (um)'))):
        pts = [(v['pxy'][0] / v['pxy'][1] if key is None else v[key], v['dnll']) for k, v in S['profiles'].items() if k.startswith(pre)]
        r0 = S['profiles']['prof_ref']
        pts.append((r0['pxy'][0] / r0['pxy'][1] if key is None else r0[key], 0.0))
        pts.sort()
        ax[i].plot([p[0] for p in pts], [p[1] for p in pts], 'o-'); ax[i].set_xlabel(lab); ax[i].axhline(0, color='k', lw=0.5)
    ax[0].set_ylabel('NLL - NLL(ref)  (bead frames)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_bead_profiles.png', dpi=120); plt.close(fig)
jdump(S, f'{OUT}/insitu_summary.json')
print('written', f'{OUT}/insitu_summary.json', len(json.dumps(S)) // 1000, 'kB')
