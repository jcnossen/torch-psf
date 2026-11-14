import time, torch, sys
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
scn = sys.argv[1] if len(sys.argv) > 1 else 'A'
spl = F.new_spline()
Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
test = C.sim_test(scn, 3000)
ev = lambda K: F.evaluate(spl, K, test, nll_true)
spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
def show(name, K):
    r = ev(K)
    print(f"{name:14s} psf err {F.psf_rel_err(spl, K, Ktrue):.4f} dnll {r['dnll']:8.2f} outl {r['outlier']:.3f} xy rmse {r['x_rmse']:.1f} {r['y_rmse']:.1f} "
          f"z rmse {r['z_rmse']:.1f} lin {r['z_rmse_lin']:.1f} zscale {r['z_scale']:.3f} zoff {r['z_off']:.1f} I {r['I_ratio']:.3f}", flush=True)
show('truth', Ktrue)
stack = C.sim_stack(scn)
t = time.time(); zc_s = F.fit_vec_stack(stack); print('vec stack fit', f'{time.time()-t:.1f}s', 'zc err vs bead truth', (zc_s.cpu() - C.zc_bead()).abs().max().item())
K_vs = F.knots_from(spl, zc_s, 'insitu'); show('V-stack', K_vs)
# spline from stack only (init: vectorial stack fit on the stack axis)
t = time.time(); K_ss, _ = F.fit_knots(spl, F.knots_from(spl, zc_s, 'bead'), stack=stack, lam_s=float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
print('spline stack fit', f'{time.time()-t:.1f}s'); show('S-stack', K_ss)
movie, truth = C.sim_movie(scn, T=6000)
rois, fr, cor = C.detect_crop(movie); print('rois', len(rois))
torch.save(dict(rois=rois.cpu(), zc_s=zc_s.cpu(), K_ss=K_ss.cpu(), stack={k: (v.cpu() if torch.is_tensor(v) else v) for k, v in stack.items()}), f'study/insitu/out_dev_{scn}.pt')
log = lambda it, tot, t, zc: print(it, f'{tot:.1f} {t:.1f}s zc err {(zc.cpu()-C.zc_insitu()).abs().max():.4f}', flush=True) if it % 10 == 0 else None
t = time.time(); zc_i, th = F.fit_vec_insitu(spl, rois, zc_s, iters=60, log=log); print('V-insitu K', f'{time.time()-t:.1f}s')
show('V-insitu', F.knots_from(spl, zc_i))
