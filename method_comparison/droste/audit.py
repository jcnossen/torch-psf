"""Reproduce one legacy calibration and one legacy localization line-search stop."""
import copy
import csv
import numpy as np
from simulate import HERE, datasets, fit, nat_matrix, tensor


def audit():
    rows = []
    for seed, scenario, psf, baseline, _, design, cal, _, _, test_design, test in datasets([1100,1102]):
        if scenario != 'NAT field':
            continue
        model = 'Independent + ridge' if seed==1100 else 'NAT'
        basis = tensor(np.eye(25) if seed==1100 else nat_matrix())
        base = tensor(baseline)
        _, field, original = fit(psf,cal,design,basis,base,ridge=seed==1100,audit=seed==1100)
        if seed==1100:
            args=(cal,design,basis,base)
            stage='calibration'
        else:
            args=(test,test_design,tensor(np.empty((25,0))),field)
            _, _, original=fit(psf,*args,audit=True)
            stage='localization'
        _, _, refined=fit(copy.deepcopy(psf).double(),*args,ridge=seed==1100,warm=original,audit=True)
        print(seed,stage,original.status,original.message,original.pg,refined.pg,flush=True)
        for result in (original,refined):
            for record in result.audit:
                rows.append(dict(seed=seed,model=model,stage=stage,status=result.status,
                                 projected_gradient=result.pg,**record))
    with (HERE/'line_search_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__=='__main__':
    audit()
