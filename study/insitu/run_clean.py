"""Control for section 3.3: the free pupil fit on CLEAN single-frame patches (isolated emitters simulated directly,
uniform sub-pixel offsets, no detection step), same photons and number of emitters as the movie."""
import sys, time, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
for scn in 'AB':
    spl = F.new_spline()
    Ktrue = spl.sample(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
    test = C.sim_test(scn, 6000)
    spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
    rois = C.sim_test(scn, 11000, seed=7)['d']
    zc_s = F.fit_vec_stack(C.sim_stack(scn))

    def rec(name, K):
        r = F.evaluate(spl, K, test, nll_true)
        e = r['th'] - test['th']; z = test['th'][:, 2]; good = e[:, 2].abs() < 0.15
        b = torch.linspace(0.05, 0.75, 8, device=C.DEV); ib = torch.bucketize(z, b).clamp(1, 7) - 1
        cy = [float(e[good & (ib == i), 1].mean()) * 108 for i in range(7)]; cz = [float(e[good & (ib == i), 2].mean()) * 1000 for i in range(7)]
        print(f"clean {scn} {name:12s} dnll {r['dnll']:6.2f}  mean err over z: y {min(cy):6.1f}..{max(cy):5.1f}  z {min(cz):6.1f}..{max(cz):5.1f} nm", flush=True)

    zc, th = F.fit_vec_insitu(spl, rois, zc_s, iters=60); rec('Zernike', F.knots_from(spl, zc))
    Kz = F.knots_from(spl, zc)
    Kp, _, _ = F.fit_pupil_insitu(spl, rois, zc, iters=200, th=th); rec('free pupil', Kp)
    Kk, _ = F.fit_knots(spl, Kz, insitu=rois, ref=Kz, lam_ref=3000., lam_s=1000., th_ins=th); rec('free knots', Kk)
