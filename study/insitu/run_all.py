"""All accuracy results of the study for one seed (after CRITIC_INSITU_1): python study/insitu/run_all.py SEED [scns]

Changes against run_study / run_pupil / run_clean: offset clamp 3 px, one scoring rule (wrong-z mask only), data
simulated with a finer pupil grid than the fitted model (common.SIM_OPT), seeds for beads / movie / test / clean.
Writes study/insitu/out/all_seed{SEED}.json.
"""
import sys, time, json, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
scns = sys.argv[2] if len(sys.argv) > 2 else 'AB'
res = {}


def score(spl, K, test, nll_true):
    r = F.evaluate(spl, K, test, nll_true)
    th, ok = r.pop('th'), r.pop('ok'); r.pop('nll')
    e, z = (th - test['th'])[ok], test['th'][ok, 2]
    b = torch.linspace(0.05, 0.75, 8, device=C.DEV)
    ib = torch.bucketize(z, b).clamp(1, 7) - 1
    sc = (108, 108, 1000)
    r['curve'] = {n: [float(e[ib == i, k].mean()) * sc[k] for i in range(7)] for k, n in enumerate('xyz')}
    # straight line through the error against the true height: slope x 0.7 um = change of the mean error over the
    # scored z range ("tilt", nm); for z this is the z-scale error
    A = torch.stack([torch.ones_like(z), z - 0.4], 1).double()
    for k, n in enumerate('xyz'):
        co = torch.linalg.lstsq(A, e[:, k:k + 1].double()).solution[:, 0]
        r[f'{n}_tilt'] = float(co[1]) * 0.7 * sc[k]
        r[f'{n}_mean'] = float(co[0]) * sc[k]
    return r


for scn in scns:
    spl = F.new_spline()
    Ktrue = spl.sample(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
    test = C.sim_test(scn, 6000, seed=1000 + seed)
    spl.knots = Ktrue
    _, nll_true = localize(spl, test['d'])
    stack = C.sim_stack(scn, seed=seed)
    movie, truth = C.sim_movie(scn, T=6000, seed=seed)
    rois, fr, cor = C.detect_crop(movie)
    clean = C.sim_test(scn, 11000, seed=2000 + seed)['d']
    R = {}

    def rec(name, K, sec, th=None):
        r = score(spl, K, test, nll_true)
        r['sec'] = sec
        r['psf_err'] = F.psf_rel_err(spl, K, Ktrue)
        if th is not None:
            r['clamp'] = F.clamp_fraction(spl, th)
        R[name] = r
        print(f"s{seed} {scn} {name:28s} dnll {r['dnll']:6.2f} wrongz {r['outlier']:.3f} rms {r['x_rmse']:5.1f} {r['y_rmse']:5.1f} {r['z_rmse']:5.1f} "
              f"tilt x {r['x_tilt']:6.1f} y {r['y_tilt']:6.1f} z {r['z_tilt']:6.1f}  mean z {r['z_mean']:6.1f}  I {r['I_ratio']:.3f} "
              f"clamp {r.get('clamp', float('nan')):.4f} {sec:.0f}s", flush=True)

    rec('true PSF', Ktrue, 0)
    t = time.time(); zc_s = F.fit_vec_stack(stack); ts = time.time() - t
    rec('beads: Zernike', F.knots_from(spl, zc_s), ts)
    t = time.time(); K_ss, _ = F.fit_knots(spl, F.knots_from(spl, zc_s, 'bead'), stack=stack)
    rec('beads: spline', K_ss, time.time() - t)
    t = time.time(); K_j, thj = F.fit_knots(spl, K_ss, insitu=rois, stack=stack, w_stack=0.1, lam_s=1000.)
    rec('movie: shared spline', K_j, time.time() - t, thj)
    t = time.time(); zc_i, th = F.fit_vec_insitu(spl, rois, zc_s, iters=60); tv = time.time() - t
    K_vi = F.knots_from(spl, zc_i); rec('movie: Zernike', K_vi, tv, th)
    t = time.time(); K_h, thh = F.fit_knots(spl, K_vi, insitu=rois, ref=K_vi, lam_ref=3000., lam_s=1000., th_ins=th.clone())
    rec('movie: Zernike + knots', K_h, tv + time.time() - t, thh)
    t = time.time(); K_p, prm, thp = F.fit_pupil_insitu(spl, rois, zc_i, iters=200, th=th.clone())
    rec('movie: free pupil', K_p, tv + time.time() - t, thp)
    t = time.time(); zc_c, thc = F.fit_vec_insitu(spl, clean, zc_s, iters=60); tv = time.time() - t
    K_vc = F.knots_from(spl, zc_c); rec('clean: Zernike', K_vc, tv, thc)
    t = time.time(); K_hc, _ = F.fit_knots(spl, K_vc, insitu=clean, ref=K_vc, lam_ref=3000., lam_s=1000., th_ins=thc.clone())
    rec('clean: Zernike + knots', K_hc, tv + time.time() - t)
    t = time.time(); K_pc, _, thpc = F.fit_pupil_insitu(spl, clean, zc_c, iters=200, th=thc.clone())
    rec('clean: free pupil', K_pc, tv + time.time() - t, thpc)
    # training z histogram of the movie patches (fitted z with the true PSF), and how many events were detected
    spl.knots = Ktrue
    thm, _ = localize(spl, rois)
    hb = torch.linspace(0, 0.8, 9, device=C.DEV)
    res[scn] = dict(methods=R, n_rois=len(rois), n_events=len(truth), n_sites=int(truth[:, 5].unique().numel()),
                    zhist_det=torch.histc(thm[:, 2], 8, 0, 0.8).tolist(), zhist_events=torch.histc(truth[:, 3], 8, 0, 0.8).tolist(),
                    zc_err_movie=float((zc_i.cpu() - C.zc_insitu()).pow(2).mean().sqrt()),
                    zc_err_clean=float((zc_c.cpu() - C.zc_insitu()).pow(2).mean().sqrt()))
    json.dump(res, open(f'study/insitu/out/all_seed{seed}.json', 'w'), indent=1)
