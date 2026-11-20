"""Per-emitter cost of the two gradient routes as the vectorial model gets more expensive (pupil grid, oversampling)."""
import time, json, torch, sys
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
D = torch.load('study/insitu/out_dev_A.pt', weights_only=False)
rois = D['rois'].to(C.DEV); zc0 = D['zc_s'].to(C.DEV)
def tm(fn, rep=2):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep): r = fn()
    torch.cuda.synchronize(); return (time.time() - t) / rep, r
out = []
for npupil, osamp in ((64, 3), (128, 4), (256, 6)):
    C.OPT.update(npupil=npupil, oversample=osamp)
    spl = F.new_spline()
    big, small = C.model_vec(spl.Pl, zc0), C.model_vec(C.P, zc0)
    ch = 4 if npupil < 256 else 1
    t_build, K = tm(lambda: spl.sample(big, C.insitu_pos, chunk=ch)); spl.knots = K
    th, _ = localize(spl, rois[:4000])
    N = 4000; d, t = rois[:N], th
    tk, (gK, _) = tm(lambda: F.knot_grad(spl, d, t))
    tv, gz = tm(lambda: spl.vjp_vectorial(big, C.insitu_pos, gK, chunk=ch)['zcoef'])
    nV = 500
    tV, (gV, _) = tm(lambda: F.vec_grad_direct(small, d[:nV], t[:nV], nb=250 if npupil < 256 else 50), rep=1)
    perK, perV, fixed = tk / N, tV / nV, tv + t_build
    r = dict(npupil=npupil, oversample=osamp, us_per_emitter_K=perK * 1e6, us_per_emitter_V=perV * 1e6, fixed_K_s=fixed,
             breakeven_N=fixed / (perV - perK), speedup_1e4=1e4 * perV / (1e4 * perK + fixed), speedup_1e5=1e5 * perV / (1e5 * perK + fixed),
             speedup_1e6=1e6 * perV / (1e6 * perK + fixed))
    out.append(r); print({k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open('study/insitu/out/speed2.json', 'w'))
