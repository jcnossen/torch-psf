"""Collect all numbers of the tubulin in-situ test into insitu/insitu_summary.json (CPU).
usage: INSITU_DS=tubulin python insitu_tub_summary.py [results_tag=main] [mt_tag=main] [kappa_tag=main]"""
import sys, json, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
from insitu_common import OUT, jdump

rt = sys.argv[1] if len(sys.argv) > 1 else 'main'
S = dict(readme='Real-data test of in-situ vectorial PSF refinement through the spline knots, Tubulin-A647-3D. '
                'Scripts: study/realdata/insitu_*.py with INSITU_DS=tubulin, plus insitu_tub_analyze.py (metrics), '
                'insitu_tub_mt.py (microtubule axial ruler), insitu_tub_kappa.py (predicted depth factor). Test set: all '
                'flags == 0 spots of the odd 1000-frame blocks; training: 150,000 spots of the even blocks; validation: '
                '30,000 further even-block spots. z in nominal nm (10 nm per bead-stack plane assumed), no index factor.')
keys = ['tag', 'noll', 'NA', 'pxy', 'lam', 'f0', 's', 'apod', 'bead_height', 'blur_sigma', 'zscale', 'poly', 'fixed', 'maps',
        'nll', 'nll_spline', 'n_pix', 'chi2_mean', 'chi2_bead_median', 'chi2_bead', 'chi2_spline_mean', 'chi2_spline_bead_median',
        'chi2_per_z_vec_spline', 'seconds', 'n_params_global', 'pupil_phase_rms', 'pupil_amp_rms', 'zcoef']
S['bead_fits'] = {}
for f in sorted(glob.glob(f'{OUT}/vecbeads_*.json')):
    r = json.load(open(f))
    S['bead_fits'][r['tag']] = {k: r[k] for k in keys if k in r}
S['bead_fit_used_for_b'] = 'n36_pri'
S['in_situ_fits'] = {}
for f in sorted(glob.glob(f'{OUT}/fit_*.json')):
    S['in_situ_fits'][os.path.basename(f)[4:-5]] = json.load(open(f))
for f in sorted(glob.glob(f'{OUT}/state_*.json')):
    r = json.load(open(f))
    S['in_situ_fits'].setdefault(os.path.basename(f)[6:-5], {})['final_state'] = {
        k: v for k, v in r.items() if k not in ('pupil_phase', 'pupil_amp', 'bead_xy', 'bead_dz', 'chi2_bead', 'chi2_per_z_vec_spline')}
for pre, key in (('results', 'test_results'), ('mt', 'microtubule_ruler'), ('kappa', 'kappa_prediction'), ('check', 'zbin_repeat_check')):
    for f in sorted(glob.glob(f'{OUT}/{pre}_*.json')):
        t = os.path.basename(f)[len(pre) + 1:-5]
        r = json.load(open(f))
        if pre == 'kappa':
            r.pop('curves', None)
        S[f'{key}_{t}'] = r
if os.path.exists(f'{OUT}/timing.json'):
    S['timing_clean'] = json.load(open(f'{OUT}/timing.json'))
try:
    npc = json.load(open('/mnt/data/SMLM_challenge/NPC-A647-3D/insitu/results_main.json'))
    S['npc_comparison'] = {l: dict(ddev_vs_ref=m['ddev_vs_ref'], chi2_mean=m['chi2_mean'], limits=m['limits'],
                                   z_slope=m['z_vs_ref']['slope'], repeat_z_a_units=m['local_scale_corrected']['repeat_z_all'])
                           for l, m in npc['methods'].items()}
except Exception as ex:
    S['npc_comparison'] = str(ex)
jdump(S, f'{OUT}/insitu_summary.json')
print('written', f'{OUT}/insitu_summary.json', len(json.dumps(S)) // 1000, 'kB')
