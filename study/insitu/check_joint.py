import time, torch, sys
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
scn = sys.argv[1]
D = torch.load(f'study/insitu/out_dev_{scn}.pt')
rois = D['rois'].to(C.DEV); stack = {k: (v.to(C.DEV) if torch.is_tensor(v) else v) for k, v in D['stack'].items()}
spl = F.new_spline()
Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
test = C.sim_test(scn, 3000)
spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
def show(name, K):
    r = F.evaluate(spl, K, test, nll_true)
    print(f"{name:26s} psf err {F.psf_rel_err(spl, K, Ktrue):.4f} dnll {r['dnll']:8.2f} outl {r['outlier']:.3f} xy rmse {r['x_rmse']:.1f} {r['y_rmse']:.1f} "
          f"z rmse {r['z_rmse']:.1f} lin {r['z_rmse_lin']:.1f} zscale {r['z_scale']:.3f} zoff {r['z_off']:.1f} I {r['I_ratio']:.3f}", flush=True)
zc_i, th = F.fit_vec_insitu(spl, rois, D['zc_s'].to(C.DEV), iters=60)
K_vi = F.knots_from(spl, zc_i); show('V-insitu', K_vi)
K_ss = D['K_ss'].to(C.DEV)
for cfg in sys.argv[2:]:
    kind, a, b = cfg.split(',')
    t = time.time()
    if kind == 'joint':
        K, _ = F.fit_knots(spl, K_ss, insitu=rois, stack=stack, w_stack=float(a), lam_s=float(b))
    else:
        K, _ = F.fit_knots(spl, K_vi, insitu=rois, ref=K_vi, lam_ref=float(a), lam_s=float(b), th_ins=th)
    show(f'{cfg} {time.time()-t:.0f}s', K)
