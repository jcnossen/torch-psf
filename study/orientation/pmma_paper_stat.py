"""The paper's PMMA statistic on our exact-model fits (vortex_exact.py pmma) and on their code's table: per molecule the
sd of azim / pola over the 11 slices around focus, MEAN over a 21-molecule selection (the paper: 21 molecules, signal
4.4e3-45e3 photons, mean 17e3; their selection is not in the release): (a) the 21 brightest, (b) molecules whose median
N over the window is within 4.4e3-45e3, the 21 brightest of those.
usage: python -I study/orientation/pmma_paper_stat.py results/exact_pmma_n1_zgrid.npz [their_table.npz]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.environ.get('VORTEX_REF', os.path.expanduser('~/vortex_ref'))
sys.path.insert(0, os.path.join(REF, 'py'))
import numpy as np                                                                     # noqa: E402
import pmma_summary as PS                                                              # noqa: E402

KEYS = ('n_molecules', 'sd_azim_deg_mean', 'sd_pola_deg_mean', 'crlb_azim_deg_mean', 'crlb_pola_deg_mean',
        'z_slope_mean', 'z_slope_sd', 'g2_mean_median', 'N_median')


def stats(rows):
    R = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != 'slices'}
    return dict(n_molecules=len(rows), sd_azim_deg_mean=float(R['sd_azim_deg'].mean()),
                sd_pola_deg_mean=float(R['sd_pola_deg'].mean()), crlb_azim_deg_mean=float(R['crlb_azim_deg'].mean()),
                crlb_pola_deg_mean=float(R['crlb_pola_deg'].mean()),
                ratio_azim=float(R['sd_azim_deg'].mean() / R['crlb_azim_deg'].mean()),
                ratio_pola=float(R['sd_pola_deg'].mean() / R['crlb_pola_deg'].mean()),
                z_slope_mean=float(R['z_slope'].mean()), z_slope_sd=float(R['z_slope'].std()),
                g2_mean=float(R['g2_mean'].mean()), g2_sd_between=float(R['g2_mean'].std()),
                N_mean=float(R['N_median'].mean()), N_range=[float(R['N_median'].min()), float(R['N_median'].max())])


def run(path):
    T = dict(np.load(path, allow_pickle=True))
    allrows = PS.summarize(T, top=0)['molecules']
    bright = sorted(allrows, key=lambda r: -r['N_median'])
    out = dict(all=stats(allrows), brightest21=stats(bright[:21]))
    inr = [r for r in bright if 4.4e3 <= r['N_median'] <= 45e3]
    out['N_4.4-45e3_brightest21'] = stats(inr[:21])
    out['N_4.4-45e3_all'] = stats(inr)
    return out


if __name__ == '__main__':
    res = {os.path.basename(p): run(p) for p in sys.argv[1:]}
    for k, v in res.items():
        for sel, s in v.items():
            print(f"{k:32s} {sel:24s} n {s['n_molecules']:2d}  sd az/po {s['sd_azim_deg_mean']:.2f}/{s['sd_pola_deg_mean']:.2f}"
                  f"  crlb {s['crlb_azim_deg_mean']:.2f}/{s['crlb_pola_deg_mean']:.2f}  ratio {s['ratio_azim']:.2f}/"
                  f"{s['ratio_pola']:.2f}  slope {s['z_slope_mean']:.3f}+-{s['z_slope_sd']:.3f}  g2 {s['g2_mean']:.3f}"
                  f"+-{s['g2_sd_between']:.3f}  N {s['N_mean']:.0f} [{s['N_range'][0]:.0f}, {s['N_range'][1]:.0f}]")
    json.dump(res, open(os.path.join(HERE, 'results', 'pmma_paper_stat.json'), 'w'), indent=1)
