"""Table of the simulation study (sim_compare.py outputs): mean (per seed) Zernike rms error, NA error, pupil
transmission error, held-out deviance minus the truth's, runtime.

  python -I sim_summary.py label=sim.json [label=sim.json ...]   (several files with one label are merged by seed)
"""
import sys, json
import numpy as np

rows = {}
for a in sys.argv[1:]:
    label, path = a.split('=', 1)
    J = json.load(open(path))
    for r in J['results']:
        rows.setdefault(label, {})[r['seed']] = {**rows.get(label, {}).get(r['seed'], {}), **r}
print('| scenario | method | Zernike rms error mλ (mean, per seed) | NA (mean) | pupil transmission rms | '
      'held-out deviance / px - truth | seconds |')
print('|---|---|---|---|---|---|---|')
for label, seeds in rows.items():
    methods = [m for m in ('pd', 'pd_na_apod', 'dense', 'dense_na_apod') if any(m in r for r in seeds.values())]
    for m in methods:
        rs = [r for r in seeds.values() if m in r and np.isfinite(r[m]['rms_mwaves'])]
        z = [r[m]['rms_mwaves'] for r in rs]
        ho = [r['heldout_deviance'][m] - r['heldout_deviance']['truth'] for r in rs if 'heldout_deviance' in r
              and m in r['heldout_deviance']]
        print(f"| {label} | {m} | {np.mean(z):.2f} ({', '.join(f'{v:.2f}' for v in z)}) | "
              f"{np.mean([r[m]['NA'] for r in rs]):.4f} | {np.mean([r[m]['pupil_amp_rms'] for r in rs]):.3f} | "
              f"{np.mean(ho):+.4f} | {np.mean([r[m]['seconds'] for r in rs]):.0f} |")
