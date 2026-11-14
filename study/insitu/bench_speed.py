"""Cost of one Zernike gradient over N SMLM patches: vectorial model per emitter (route V) against knot gradient +
one pass through the vectorial model (route K). Then both routes inside the same fit (same optimizer, same data)."""
import time, json, torch, sys
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf import localize
D = torch.load('study/insitu/out_dev_A.pt', weights_only=False)
rois = D['rois'].to(C.DEV); zc0 = D['zc_s'].to(C.DEV)
spl = F.new_spline()
big, small = C.model_vec(spl.Pl, zc0), C.model_vec(C.P, zc0)
spl.knots = spl.knots_from_vectorial(big, C.insitu_pos)
th, _ = localize(spl, rois)
def tm(fn, rep=3):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep): r = fn()
    torch.cuda.synchronize(); return (time.time() - t) / rep, r
res = dict(N=[], tK_knot=[], tK_vjp=[], tK_build=[], tV=[], cos=[], t_loc=[])
t_build, _ = tm(lambda: spl.knots_from_vectorial(big, C.insitu_pos))
for N in (250, 1000, 4000, 16000, 64000):
    idx = torch.arange(N, device=C.DEV) % len(rois)
    d, t = rois[idx], th[idx]
    tk, (gK, _) = tm(lambda: F.knot_grad(spl, d, t))
    tv_, gz = tm(lambda: spl.vjp_vectorial(big, C.insitu_pos, gK)['zcoef'])
    tl, _ = tm(lambda: localize(spl, d, t, iters=3), rep=2)
    if N <= 16000:
        tV, (gV, _) = tm(lambda: F.vec_grad_direct(small, d, t), rep=1 if N > 1000 else 3)
        cos = float(torch.nn.functional.cosine_similarity(gz, gV, 0))
    else:
        tV, cos = res['tV'][-1] * N / res['N'][-1], float('nan')       # extrapolated (linear in N)
    for k, v in zip(res, (N, tk, tv_, t_build, tV, cos, tl)): res[k].append(v)
    print(f'N {N:6d}  K: knot grad {tk*1e3:7.1f} ms + vjp {tv_*1e3:6.1f} ms + build {t_build*1e3:5.1f} ms   V: {tV*1e3:9.1f} ms   speedup {tV/(tk+tv_+t_build):6.1f}x  cos {cos:.5f}  (localize 3 its {tl*1e3:.0f} ms)', flush=True)
# same fit, both routes
tr = {}
for route, iters in (('K', 60), ('V', 60)):
    L = []
    F.fit_vec_insitu(spl, rois, zc0, iters=iters, route=route, log=lambda it, tot, t, zc: L.append((it, t, tot, float((zc.cpu() - C.zc_insitu()).pow(2).mean().sqrt()))))
    tr[route] = L
    print(route, 'N', len(rois), 'total', L[-1][1], 's  final rms zc err (waves)', L[-1][3], flush=True)
res['trace'] = tr; res['N_fit'] = len(rois)
json.dump(res, open('study/insitu/out/speed.json', 'w'))
