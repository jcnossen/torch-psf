"""The in-situ PSF study: simulate (bead stack + SMLM movie), estimate the PSF five ways, score on test emitters."""
import time, json, sys, torch, numpy as np
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf import localize
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = 'study/insitu/out/'
res, curves, gal = {}, {}, {}
for scn in 'AB':
    spl = F.new_spline()
    Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
    test = C.sim_test(scn, 6000)
    spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
    stack = C.sim_stack(scn)
    movie, truth = C.sim_movie(scn, T=6000)
    rois, fr, cor = C.detect_crop(movie)
    if scn == 'A':
        import tifffile; tifffile.imwrite(OUT + 'movie_A_first200.tif', movie[:200].numpy())
        gal['frame'] = movie[10].numpy(); gal['stack'] = stack['d'][0, ::5].cpu().numpy()
    R = {}; tms = {}
    def rec(name, K, sec):
        r = F.evaluate(spl, K, test, nll_true)
        th, ok = r.pop('th'), r.pop('ok'); r.pop('nll')
        r['psf_err'] = F.psf_rel_err(spl, K, Ktrue); r['sec'] = sec
        e = (th - test['th'])[ok]; z = test['th'][ok, 2]
        b = torch.linspace(0.05, 0.75, 15, device=C.DEV); ib = torch.bucketize(z, b).clamp(1, 14) - 1
        cur = [[float(e[ib == i, k].mean()) for i in range(14)] for k in range(3)]
        curves[f'{scn}/{name}'] = dict(z=((b[:-1] + b[1:]) / 2).tolist(), x=cur[0], y=cur[1], zz=cur[2])
        R[name] = r
        print(f"{scn} {name:22s} dnll {r['dnll']:7.2f} outl {r['outlier']:.3f} x {r['x_rmse']:.1f} y {r['y_rmse']:.1f} z {r['z_rmse']:.1f} "
              f"(lin {r['z_rmse_lin']:.1f}, scale {r['z_scale']:.3f}) {sec:.0f}s", flush=True)
    rec('true PSF', Ktrue, 0)
    t = time.time(); zc_s = F.fit_vec_stack(stack); ts = time.time() - t
    rec('Zernike, beads only', F.knots_from(spl, zc_s), ts)
    t = time.time(); K_ss, _ = F.fit_knots(spl, F.knots_from(spl, zc_s, 'bead'), stack=stack)
    rec('spline, beads only', K_ss, time.time() - t)
    t = time.time(); K_j, _ = F.fit_knots(spl, K_ss, insitu=rois, stack=stack, w_stack=0.1, lam_s=1000.)
    rec('spline, beads + SMLM', K_j, time.time() - t)
    t = time.time(); zc_i, th = F.fit_vec_insitu(spl, rois, zc_s, iters=60); tv = time.time() - t
    K_vi = F.knots_from(spl, zc_i); rec('Zernike, beads -> SMLM', K_vi, tv)
    t = time.time(); K_h, _ = F.fit_knots(spl, K_vi, insitu=rois, ref=K_vi, lam_ref=3000., lam_s=1000., th_ins=th)
    rec('Zernike + free knots', K_h, tv + time.time() - t)
    res[scn] = dict(methods=R, n_rois=len(rois), n_events=len(truth), zc_err_insitu=float((zc_i.cpu() - C.zc_insitu()).pow(2).mean().sqrt()))
    zi = [int(round((z - spl.z0) / spl.hz)) for z in (0.1, 0.4, 0.7)]
    gal[scn] = {k: v[zi][:, ::2, ::2].cpu().numpy() for k, v in dict(true=Ktrue, zern=K_vi, hyb=K_h).items()}
json.dump(dict(res=res, curves=curves), open(OUT + 'results.json', 'w'), indent=1)
np.savez(OUT + 'gallery.npz', **{f'{a}_{b}': v for a, d in gal.items() if isinstance(d, dict) for b, v in d.items()}, frame=gal['frame'], stack=gal['stack'])
