"""Speed of the two gradient routes, after CRITIC_INSITU_1 (C-03):
 * route V with batches of 1000 (100 for the 256 / 6 model),
 * the localization step that both routes share is reported, and the speed-up per whole iteration next to the
   speed-up of the gradient alone,
 * the fixed cost of route K measured two ways: build + re-build with backward (low memory), or one build that keeps
   the graph + one backward,
 * each vectorial setting is paired with its own accuracy (relative L2 against a 256 / 7 model, 100 emitters).
Self-contained (simulates its own patches). Writes study/insitu/out/speed3.json."""
import time, json, sys, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf import localize
from torch_psf.fit import poisson_nll
from torch_psf import VectorialPSF
NREF = 16000
rois = C.sim_test('A', NREF, seed=5)['d']
zc0 = C.zc_insitu().to(C.DEV) + 0.01 * torch.randn(len(C.NOLL), device=C.DEV, generator=torch.Generator(device=C.DEV).manual_seed(1))


def tm(fn, rep=3):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep):
        r = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / rep, r


def one_pass(spl, big, gK):
    """Knot build that keeps the autograd graph, then one backward."""
    prm = {k: v.detach() for k, v in list(big.named_parameters()) + list(big.named_buffers())}
    prm['zcoef'] = prm['zcoef'].clone().requires_grad_(True)
    K = spl.build_planes(big, prm, 0, spl.nz, C.insitu_pos)
    return torch.autograd.grad(K, prm['zcoef'], gK.float())[0]


g = torch.Generator(device=C.DEV).manual_seed(9)
pos = torch.stack([torch.rand(100, generator=g, device=C.DEV) - 0.5, torch.rand(100, generator=g, device=C.DEV) - 0.5,
                   0.05 + 0.7 * torch.rand(100, generator=g, device=C.DEV)], 1)
f0 = torch.full((100,), C.F0, device=C.DEV)


def vec(roisize, npupil, osamp):
    psf = VectorialPSF(roisize, zernike_indices=C.NOLL, device=C.DEV, **dict(C.OPT, npupil=npupil, oversample=osamp))
    with torch.no_grad():
        psf.zcoef.copy_(zc0)
    return psf


ref = C.render(vec(C.P, 256, 7), pos, f0, nb=10)
out = []
for npupil, osamp in ((64, 3), (128, 4), (256, 6)):
    spl = F.new_spline()
    big, small = vec(spl.Pl, npupil, osamp), vec(C.P, npupil, osamp)
    im = C.render(small, pos, f0, nb=10)
    acc = float(((im - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).median())
    ch = 4 if npupil < 256 else 1
    t_build, K = tm(lambda: spl.knots_from_vectorial(big, C.insitu_pos, chunk=ch))
    spl.knots = K
    acc_spl = float(((spl.render(pos) - im).flatten(1).norm(dim=1) / im.flatten(1).norm(dim=1)).median())
    th, _ = localize(spl, rois)
    N = NREF
    tk, (gK, _) = tm(lambda: F.knot_grad(spl, rois, th))
    tl, _ = tm(lambda: localize(spl, rois, th, iters=3), rep=2)
    tvjp, gz = tm(lambda: spl.vjp_vectorial(big, C.insitu_pos, gK, chunk=ch)['zcoef'])
    try:
        t1, gz1 = tm(lambda: one_pass(spl, big, gK))
        fixed1 = t1
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); fixed1 = float('nan')
    nV = 4000 if npupil < 256 else 1000
    tV, (gV, _) = tm(lambda: F.vec_grad_direct(small, rois[:nV], th[:nV], nb=1000 if npupil < 256 else 100), rep=2)
    gzs = spl.vjp_vectorial(big, C.insitu_pos, F.knot_grad(spl, rois[:nV], th[:nV])[0], chunk=ch)['zcoef']
    cos = float(torch.nn.functional.cosine_similarity(gzs, gV, 0)); rel = float((gzs - gV).norm() / gV.norm())
    perK, perV, perL = tk / N, tV / nV, tl / N
    fixed2 = t_build + tvjp
    fixed = min(fixed2, fixed1) if fixed1 == fixed1 else fixed2
    r = dict(npupil=npupil, oversample=osamp, vec_err=acc, spline_err_vs_vec=acc_spl, us_K=perK * 1e6, us_V=perV * 1e6, us_loc=perL * 1e6,
             fixed_two_builds=fixed2, fixed_one_pass=fixed1, breakeven=fixed / (perV - perK), cos=cos, rel_diff=rel)
    for n in (1e4, 1e5, 1e6):
        r[f'grad_{n:g}'] = n * perV / (n * perK + fixed)
        r[f'iter_{n:g}'] = (n * (perV + perL) + t_build) / (n * (perK + perL) + fixed)   # route V also builds the spline to localize
    out.append(r)
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
# whole fit with both routes (same data, start, optimizer), 64 / 3 model
C.OPT.update(npupil=64, oversample=3)
spl = F.new_spline()
tr = {}
for route in 'KV':
    L = []
    t = time.time()
    F.fit_vec_insitu(spl, rois, zc0, iters=40, route=route, log=lambda it, tot, tt, zc: L.append((it, tt, tot, float((zc.cpu() - C.zc_insitu()).pow(2).mean().sqrt()))))
    tr[route] = L
    print(route, 'whole fit', NREF, 'emitters, 40 its:', round(time.time() - t, 1), 's, final Zernike rms error', L[-1][3], flush=True)
json.dump(dict(table=out, trace=tr, N_fit=NREF, gpu=torch.cuda.get_device_name(0)), open('study/insitu/out/speed3.json', 'w'), indent=1)
