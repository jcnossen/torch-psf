"""Paired Monte Carlo comparison. Run python simulate.py, then python compare.py.

Outputs per-seed metrics and pooled summaries for the adjacent PDF. Set
PYTHONPATH to a torch-psf checkout if the package is not installed.
"""
import csv
import copy
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from compare import MODES, SLOTS, VectorialPSF, field_design, nat_matrix

HERE = Path(__file__).parent
SEEDS = range(1100, 1112)
MODELS = ['NAT', 'Independent', 'Independent + ridge']
SCENARIOS = {'NAT field': 0., 'Cross-coma field': .030}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


def geometry(rng, count):
    uv = rng.uniform(-1, 1, (count, 2))
    design = tensor(np.stack([field_design(*p) for p in uv]))
    xyz = np.column_stack([rng.uniform(-.35,.35,(count,2)),rng.uniform(.055,.245,count)])
    return uv, design, tensor(xyz)


def fit(psf, counts, design, basis, start_field, ridge=False, *, warm=None, audit=False):
    n = len(counts)
    dtype = psf.zcoef.dtype
    def cast(x):
        return torch.as_tensor(x, dtype=dtype, device=DEVICE)
    design, basis, start_field = map(cast, (design, basis, start_field))
    k = basis.shape[1]
    # Scale fitted coordinates to comparable pixel likelihood curvature.
    local_scale = cast([.05,.05,.025,.04,.04])
    field_scale = .012 / np.sqrt(n)
    initial = np.zeros(k + 5*n)
    bounds = [(None,None)] * k + [(-25,25),(-25,25),(-10,14),(-20,20),(-35,35)]*n
    prior = cast([sum(p)>0 for _,p in SLOTS])
    # Double-count-free Poisson deviance relative to the starting prediction.
    with torch.no_grad():
        reference = 3000*psf(cast([[0.,0.,.15]]).expand(n,-1),.15,
                             zcoef_offset=design@start_field) + 8
    def unpack(x):
        local = x[k:].reshape(n,5)*local_scale
        xyz = local[:,:3] + cast([0.,0.,.15])
        slots = start_field + basis@(x[:k]*field_scale)
        return xyz, slots, local
    def objective(x):
        pars = cast(x).requires_grad_()
        xyz, slots, local = unpack(pars)
        image = psf(xyz,.15,zcoef_offset=design@slots)
        mu = image * (3000*local[:,3].exp())[:,None,None] + (8*local[:,4].exp())[:,None,None]
        # Accumulate in float64 to resolve small likelihood changes on a GPU.
        ratio = (mu.double()-reference)/reference
        loss = ((mu.double()-reference)-counts*torch.log1p(ratio)).sum()
        if ridge:
            loss = loss + .5*((slots*prior/.05)**2).sum()
        grad, = torch.autograd.grad(loss,pars)
        return loss.item(), grad.detach().cpu().numpy().astype(float)
    options = {'maxiter':250,'ftol':1e-10,'gtol':.003,'maxls':30}
    if warm is not None:
        initial = warm.x
        options.update(maxiter=1200,ftol=0.)
    sol = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                   options=options)
    lower = np.array([-np.inf if lo is None else lo for lo,_ in bounds])
    upper = np.array([np.inf if hi is None else hi for _,hi in bounds])
    def projected(x,g):
        return x - np.clip(x-g,lower,upper)
    sol.pg = np.max(np.abs(projected(sol.x,sol.jac)))
    if warm is not None:
        if not sol.success or not np.isfinite(sol.pg) or sol.pg > .00301:
            raise RuntimeError(f'Unresolved fit: {sol.message}, projected gradient {sol.pg:g}')
        sol.improvement = objective(warm.x)[0]-sol.fun
    if audit:
        # Compare centered objective differences at identical parameters/directions.
        point = sol.x if warm is None else warm.x
        _, grad = objective(point)
        direction = -projected(point,grad) if warm is None else warm.direction
        if warm is None:
            direction /= np.linalg.norm(direction)
        sol.direction = direction
        sol.audit = []
        for step in [1.e-2,1.e-3,1.e-4]:
            fd = (objective(point+step*direction)[0]-objective(point-step*direction)[0])/(2*step)
            sol.audit.append(dict(precision=str(dtype),step=step,finite_difference=fd,autograd=grad@direction))
    xyz, slots, _ = unpack(cast(sol.x))
    return xyz.detach().cpu().numpy(), slots.detach(), sol


def datasets(seeds):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    psf = VectorialPSF(15,pixel_um=.108,NA=1.4,n_imm=1.518,n_med=1.33,wavelength_um=.68,
                       zernike_indices=MODES,npupil=32,oversample=2,device=DEVICE,pixel_integration='exact')

    baseline = np.zeros(len(SLOTS))
    baseline[SLOTS.index((6,(0,0)))] = .08
    shared = nat_matrix() @ np.array([0,.004,-.003,.004,0,.005,.007,-.004,.006,.003,-.004,.008,.003])
    test_uv, test_design, test_xyz = geometry(np.random.default_rng(41082),128)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        _, cal_design, cal_xyz = geometry(rng,96)
        for scenario, magnitude in SCENARIOS.items():
            truth = baseline + shared
            truth[SLOTS.index((8,(0,1)))] += magnitude
            # Independent photon realizations between scenarios and train/test sets.
            def expose(design,xyz):
                with torch.no_grad():
                    mu = 3000*psf(xyz,.15,zcoef_offset=design@tensor(truth))+8
                return tensor(rng.poisson(mu.cpu().numpy()))
            cal = expose(cal_design,cal_xyz)
            test = expose(test_design,test_xyz)
            yield seed,scenario,psf,baseline,truth,cal_design,cal,test_uv,test_xyz,test_design,test


def experiment():
    for seed,scenario,psf,baseline,truth,cal_design,cal,test_uv,test_xyz,test_design,test in datasets(SEEDS):
        precise_psf = copy.deepcopy(psf).double()
        for model in MODELS:
            basis = nat_matrix() if model=='NAT' else np.eye(len(SLOTS))

            _, _, cal_warm = fit(psf,cal,cal_design,tensor(basis),tensor(baseline),
                                     ridge=model.endswith('ridge'))
            _, fitted, cal_sol = fit(precise_psf,cal,cal_design,tensor(basis),tensor(baseline),
                                     ridge=model.endswith('ridge'),warm=cal_warm)
            empty = tensor(np.empty((25,0)))
            _, _, test_warm = fit(psf,test,test_design,empty,fitted)
            xyz, _, test_sol = fit(precise_psf,test,test_design,empty,fitted,warm=test_warm)
            err = (xyz-test_xyz.cpu().numpy()) * [108,108,1000]
            field_error = (test_design.double()@(fitted-tensor(truth).double())).cpu().numpy()*1000
            # Regression of signed x error against v detects cancellation across the field.
            slope = np.polyfit(test_uv[:,1],err[:,0],1)[0]
            row = dict(seed=seed,scenario=scenario,model=model,
                field_rmse=np.sqrt(np.mean(field_error**2)),xy_rmse=np.sqrt(np.mean(err[:,:2]**2)),
                z_rmse=np.sqrt(np.mean(err[:,2]**2)),x_slope=slope,z_bias=err[:,2].mean(),
                cross_coma=1000*fitted[SLOTS.index((8,(0,1)))].item(),
                cal_iterations=cal_sol.nit,test_iterations=test_sol.nit,
                cal_status=cal_sol.status,test_status=test_sol.status,
                cal_warm_status=cal_warm.status,test_warm_status=test_warm.status,
                cal_pg=cal_sol.pg,test_pg=test_sol.pg,cal_warm_pg=cal_warm.pg,test_warm_pg=test_warm.pg,
                cal_gain=cal_sol.improvement,test_gain=test_sol.improvement)
            print(seed,scenario,model,cal_sol.nit,test_sol.nit,cal_sol.pg,test_sol.pg,flush=True)
            yield row,err


def run():
    records,errors=[],{}
    for row,err in experiment():
        records.append(row)
        errors.setdefault((row['scenario'],row['model']),[]).append(err)
    def save(name,rows):
        with (HERE/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    save('simulation_runs.csv',records)
    summary=[]
    for (scenario,model),e in errors.items():
        subset=[r for r in records if (r['scenario'],r['model'])==(scenario,model)]
        e=np.asarray(e)
        sd=e.std(axis=0,ddof=1)
        summary.append(dict(scenario=scenario,model=model,
            field_rmse=np.sqrt(np.mean([r['field_rmse']**2 for r in subset])),
            xy_sd=np.sqrt(np.mean(sd[:,:2]**2)),z_sd=np.sqrt(np.mean(sd[:,2]**2)),
            xy_rmse=np.sqrt(np.mean(e[:,:,:2]**2)),z_rmse=np.sqrt(np.mean(e[:,:,2]**2)),
            x_slope=np.mean([r['x_slope'] for r in subset]),cross_coma=np.mean([r['cross_coma'] for r in subset]),
            z_bias=e[:,:,2].mean()))
    save('simulation_summary.csv',summary)


if __name__=='__main__':
    run()
