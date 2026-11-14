"""Free pupil map (phase + amplitude pixels) fitted in situ through the knot route, scenarios A and B."""
import time, json, sys, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf import localize
OUT = 'study/insitu/out/'
lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
lr_a = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2
res = {}
for scn in (sys.argv[3] if len(sys.argv) > 3 else 'AB'):
    spl = F.new_spline()
    Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
    test = C.sim_test(scn, 6000)
    spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
    D = torch.load(f'study/insitu/out_dev_{scn}.pt', weights_only=False)
    rois = D['rois'].to(C.DEV)
    def rec(name, K, sec):
        r = F.evaluate(spl, K, test, nll_true); th, ok = r.pop('th'), r.pop('ok'); r.pop('nll'); r['sec'] = sec
        e = (th - test['th'])[ok]; z = test['th'][ok, 2]
        b = torch.linspace(0.05, 0.75, 15, device=C.DEV); ib = torch.bucketize(z, b).clamp(1, 14) - 1
        r['curve'] = dict(z=((b[:-1] + b[1:]) / 2).tolist(), **{n: [float(e[ib == i, k].mean()) for i in range(14)] for k, n in enumerate(('x', 'y', 'zz'))})
        res[f'{scn}/{name}'] = r
        yb = [v * 108 for v in r['curve']['y']]; zb = [v * 1000 for v in r['curve']['zz']]
        print(f"{scn} {name:24s} dnll {r['dnll']:6.2f} outl {r['outlier']:.3f} x {r['x_rmse']:.1f} y {r['y_rmse']:.1f} z {r['z_rmse']:.1f} zscale {r['z_scale']:.3f} "
              f"y bias {min(yb):.1f}..{max(yb):.1f} z bias {min(zb):.1f}..{max(zb):.1f}  {sec:.0f}s", flush=True)
    t = time.time(); zc_i, th = F.fit_vec_insitu(spl, rois, D['zc_s'].to(C.DEV), iters=60); tz = time.time() - t
    rec('Zernike', F.knots_from(spl, zc_i), tz)
    log = lambda it, tot, t, p: print(f'   it {it} nll {tot:.0f} {t:.0f}s phase rms {p["phase"].std():.4f}', flush=True) if it % 50 == 0 else None
    t = time.time(); K, prm, _ = F.fit_pupil_insitu(spl, rois, zc_i, iters=iters, lam=lam, lr_a=lr_a, th=th, log=log)
    rec('free pupil map', K, tz + time.time() - t)
    torch.save({k: v.cpu() for k, v in prm.items()}, OUT + f'pupil_{scn}.pt')
json.dump(res, open(OUT + f'pupil_lam{lam:g}_lra{lr_a:g}.json', 'w'), indent=1)
