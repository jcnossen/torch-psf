"""Step 3: PSF fits on the training patches (150,000 spots, flags == 0, even 1000-frame blocks), all on the movie spline
grid (17 px patch, +-3 px offset, S 2, hz 50 nm, z +-0.8 um):

  a   bead spline as is (locs/calib/psf_spline.pt)
  b   bead vectorial model as is (knots built from insitu_vecbeads.py's result)
  c   vectorial model, Zernikes refined in situ through the knot route: per iteration (1) knots from the vectorial model,
      (2) emitter parameters on the spline (Levenberg-Marquardt, warm start, 3 iterations; 3 starting heights in the first
      iteration), (3) gradient of the total NLL on the knots, (4) one vector-Jacobian product through the knot build to
      the Zernike coefficients, (5) Adam step.
      Knot z axis: z_k -> (emitter height D, focus f0 + s z_k): the PSF of an emitter at fixed height D above the
      coverslip with the focus moved by z_k (stage-equivalent um, the same axis as the bead spline). --D 0 = the bead
      geometry (supercritical light as for a bead on the glass); --D 0.5 = 500 nm into water (no supercritical light,
      index-mismatch spherical aberration of that depth). The z-dependence of the mismatch aberration over the emitter
      range is neglected; Zernike spherical terms (Noll 11, 22) are free and absorb the depth-dependent difference.
  d   as c plus a free pupil phase map (waves) and log-amplitude map (piston, tilt and defocus projected out of the
      phase, the mean out of the log amplitude), starting from c's result.
  e   free knots refined in situ, starting from the bead spline: alternate emitter parameters and L-BFGS on the knots,
      roughness penalty as in the bead calibration, pedestal and per-plane scale fixed (gauges).
usage: python insitu_fit.py --method c --tag c_D0 [--D 0] [--iters 150] [--init knots/state]"""
import sys, time, argparse, math, json
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np, torch
from insitu_common import *
from torch_psf.zernike import zernike

ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--vec', default=f'{OUT}/vecbeads_best.json')
ap.add_argument('--D', type=float, default=-1.0)          # emitter height (um); < 0: the bead geometry (bead centre height)
ap.add_argument('--blur', type=int, default=0)             # 1: keep the bead-fit blur in the emitter PSF; 2: also refine it
ap.add_argument('--iters', type=int, default=150)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--lr_p', type=float, default=2e-3)
ap.add_argument('--lr_a', type=float, default=1e-2)
ap.add_argument('--lam', type=float, default=0.0)          # pupil-map roughness (d) / knot roughness (e)
ap.add_argument('--init', default='')                      # d: state json of a c run
ap.add_argument('--ntrain', type=int, default=150000)
ap.add_argument('--outer', type=int, default=10)
ap.add_argument('--val_every', type=int, default=25)
A = ap.parse_args()
torch.manual_seed(0)
t_all = time.time()
spl, Kb = movie_spline()
dtr = torch.from_numpy(np.load(f'{OUT}/patches_train.npy', mmap_mode='r')[:A.ntrain].copy()).to(DEV)
dval = torch.from_numpy(np.load(f'{OUT}/patches_val.npy').copy()).to(DEV)
print(f'method {A.method}, tag {A.tag}: {len(dtr)} training spots, {len(dval)} validation spots', flush=True)
log = dict(args=vars(A), it=[])


def val_nll(knots):
    spl.knots = knots
    th, nll = localize_full(spl, dval)
    c2, dev, _ = fit_stats(spl, dval, th)
    return float(dev.mean()), float(c2.mean()), limit_fractions(th)


def save(knots, extra=None):
    out = dict(knots=knots.cpu(), method=A.method, tag=A.tag, S=spl.S, hz=spl.hz, z0=spl.z0, nz=spl.nz, c=spl.c, P=P,
               xmax=XMAX)
    if extra:
        out.update(extra)
    torch.save(out, f'{OUT}/psf_{A.tag}.pt')
    jdump(log, f'{OUT}/fit_{A.tag}.json')


sync = torch.cuda.synchronize

def bead_D(st):
    return float(st.get('bead_height', 0.0)) if st is not None else 0.0


if A.method in ('a', 'b', 'p'):
    if A.method == 'a':
        K = Kb
        st = None
    elif A.method == 'p':          # a previously fitted vectorial state (e.g. c or d) as is
        st = json.load(open(A.vec))
        big = make_vec(spl.Pl, st, use_blur=st.get('use_blur', False))
        K = knots_from_vec(spl, big, pos_fn_maker(st.get('f0_D', st['f0']), st['s'], st['D'], st.get('zscale', 1.0)))
    else:
        st = json.load(open(A.vec))
        big = make_vec(spl.Pl, st, use_blur=A.blur > 0)
        A.D = bead_D(st) if A.D < 0 else A.D
        K = knots_from_vec(spl, big, pos_fn_maker(st['f0'], st['s'], A.D, st.get('zscale', 1.0)))
    spl.knots = K
    th, nll = localize_full(spl, dtr)
    c2, dev, _ = fit_stats(spl, dtr, th)
    log.update(train_dev=float(dev.mean()), train_chi2=float(c2.mean()), train_limits=limit_fractions(th))
    v = val_nll(K)
    log.update(val_dev=v[0], val_chi2=v[1], val_limits=v[2])
    print(f'{A.tag}: train dev/spot {log["train_dev"]:.3f} chi2 {log["train_chi2"]:.4f} limits {log["train_limits"]}; '
          f'val dev/spot {v[0]:.3f} chi2 {v[1]:.4f} limits {v[2]}', flush=True)
    save(K, dict(vec_state=st, D=A.D))
    sys.exit(0)

if A.method in ('c', 'd'):
    st = json.load(open(A.vec))
    big = make_vec(spl.Pl, st, use_blur=A.blur > 0)
    Db = bead_D(st)
    A.D = Db if A.D < 0 else A.D
    if A.D != Db and 'f0_D' not in st:
        # emitter at height D: the focus offset that keeps the knot z origin where the bead PSF has it (least-squares
        # match of the knot stacks); paraxially f0 + D n_imm / n_med
        K0 = knots_from_vec(spl, big, pos_fn_maker(st['f0'], st['s'], Db, st.get('zscale', 1.0)))
        best = None
        for fct in np.linspace(0.6, 1.6, 41):
            f0D = st['f0'] + (A.D - Db) * fct * 1.518 / 1.33
            KD = knots_from_vec(spl, big, pos_fn_maker(f0D, st['s'], A.D, st.get('zscale', 1.0)))
            e = float((KD - K0).norm() / K0.norm())
            if best is None or e < best[0]:
                best = (e, f0D, fct)
        st['f0_D'] = best[1]
        log['f0_D'] = dict(f0=best[1], factor_vs_paraxial=best[2], rel_diff_to_bead_knots=best[0])
        print(f'D {A.D}: focus offset {best[1]:.4f} um ({best[2]:.3f} x paraxial), knot difference to D = 0: {best[0]:.4f}', flush=True)
    pos_fn = pos_fn_maker(st.get('f0_D', st['f0']) if A.D != Db else st['f0'], st['s'], A.D, st.get('zscale', 1.0))
    mask = big.mask.bool()
    prm = dict(zcoef=big.zcoef.detach().clone())
    if A.init:
        st2 = json.load(open(A.init))
        prm['zcoef'] = torch.tensor(st2['zcoef'], device=DEV)
        if A.blur > 0 and st2.get('blur_sigma', 0) > 0:
            with torch.no_grad():
                big.blur.fill_(float(st2['blur_sigma']))
    names = ('zcoef',)
    lrs = dict(zcoef=A.lr)
    if A.blur == 2:
        if float(big.blur) <= 0:
            with torch.no_grad():
                big.blur.fill_(0.03)        # start at 30 nm (|blur| has no gradient at 0)
            big.use_blur = True
        prm['blur'] = big.blur.detach().clone()
        names = names + ('blur',)
        lrs['blur'] = 1e-3
    if A.method == 'd':
        low = torch.stack([zernike(j, big.rho.clamp(max=1), big.phi)[mask] for j in (1, 2, 3, 4)], 1)
        Q, _ = torch.linalg.qr(low)
        prm.update(phase=torch.zeros_like(big.pupil_phase), la=torch.zeros_like(big.pupil_phase))
        names = names + ('pupil_phase', 'pupil_amp')
        lrs.update(phase=A.lr_p, la=A.lr_a)

    def lap(x):
        return (4 * x[1:-1, 1:-1] - x[2:, 1:-1] - x[:-2, 1:-1] - x[1:-1, 2:] - x[1:-1, :-2]) * mask[1:-1, 1:-1]

    def set_model():
        with torch.no_grad():
            big.zcoef.copy_(prm['zcoef'])
            if 'blur' in prm:
                big.blur.copy_(prm['blur'].abs())
            if A.method == 'd':
                big.pupil_phase.copy_(prm['phase'] * mask)
                big.pupil_amp.copy_(torch.exp(prm['la']) * mask + (~mask).float())

    opt = Adam(prm, lrs)
    th = None
    tm = dict(build=0.0, loc=0.0, kgrad=0.0, vjp=0.0)
    for it in range(A.iters + 1):
        sync(); t0 = time.time()
        set_model()
        spl.knots = knots_from_vec(spl, big, pos_fn)
        sync(); t1 = time.time()
        if th is None:
            th, nll = localize_full(spl, dtr)
        else:
            th, nll = lm(spl, dtr, th, 3)
        sync(); t2 = time.time()
        if it % A.val_every == 0 or it == A.iters:
            v = val_nll(spl.knots)
            c2, dev, _ = fit_stats(spl, dtr, th)
            log['it'].append(dict(it=it, train_dev=float(dev.mean()), train_chi2=float(c2.mean()), val_dev=v[0], val_chi2=v[1],
                                  val_limits=v[2], train_limits=limit_fractions(th), zcoef=prm['zcoef'].cpu().numpy()))
            print(f'  it {it:4d}: train dev/spot {float(dev.mean()):.4f} chi2 {float(c2.mean()):.4f}  val dev/spot {v[0]:.4f} '
                  f'chi2 {v[1]:.4f}  limits {v[2]}  blur {float(big.blur) * 1000:.1f} nm  {time.time() - t_all:.0f}s', flush=True)
            if it == A.iters:
                break
            sync(); t2 = time.time()
        gK, tot = knot_grad(spl, dtr, th)
        sync(); t3 = time.time()
        g = spl.vjp_vectorial(big, pos_fn, gK, names=names, chunk=spl.nz)
        sync(); t4 = time.time()
        gr = dict(zcoef=g['zcoef'])
        if 'blur' in prm:
            gr['blur'] = g['blur'] * torch.sign(prm['blur'])
        if A.method == 'd':
            gr.update(phase=g['pupil_phase'] * mask, la=g['pupil_amp'] * big.pupil_amp * mask)
            if A.lam > 0:
                for k in ('phase', 'la'):
                    x = prm[k].clone().requires_grad_(True)
                    gr[k] = gr[k] + torch.autograd.grad(A.lam * (lap(x) ** 2).sum(), x)[0]
        # cosine decay of the step size to 10 % over the run
        f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / A.iters))
        opt.lr = {k: v * f for k, v in lrs.items()}
        opt.step(gr)
        if A.method == 'd':
            ph = opt.p['phase'][mask]
            opt.p['phase'][mask] = ph - Q @ (Q.T @ ph)
            opt.p['la'][mask] = opt.p['la'][mask] - opt.p['la'][mask].mean()
            opt.p['phase'] = opt.p['phase'] * mask; opt.p['la'] = opt.p['la'] * mask
        prm = opt.p
        sync(); t5 = time.time()
        if it > 0:      # iteration 0 has the full 3-start localization
            tm['build'] += t1 - t0; tm['loc'] += t2 - t1; tm['kgrad'] += t3 - t2; tm['vjp'] += t4 - t3
        if it % 10 == 0:
            print(f'  it {it:4d}: nll {tot:.6e}  build {t1 - t0:.2f} loc {t2 - t1:.2f} kgrad {t3 - t2:.2f} vjp {t4 - t3:.2f} s', flush=True)
    n = max(A.iters - 1, 1)
    log['timing'] = dict(per_iter={k: v / n for k, v in tm.items()}, per_iter_total=sum(tm.values()) / n,
                         total_s=time.time() - t_all, n_emitters=len(dtr), iters=A.iters,
                         note='per iteration: knot build, warm-start localization (3 LM iterations), knot gradient, '
                              'VJP to the pupil parameters; iteration 0 (3-start localization) and the validation '
                              'evaluations excluded')
    stv = dict(st)
    stv['zcoef'] = prm['zcoef'].cpu().numpy().tolist()
    stv['D'] = A.D
    stv['blur_sigma'] = float(big.blur) if A.blur > 0 else 0.0
    stv['use_blur'] = A.blur > 0
    if A.method == 'd':
        stv['pupil_phase'] = (prm['phase'] * mask).cpu().numpy().tolist()
        stv['pupil_amp'] = (torch.exp(prm['la']) * mask + (~mask).float()).cpu().numpy().tolist()
    jdump(stv, f'{OUT}/state_{A.tag}.json')
    print('timing', log['timing'], flush=True)
    print('zcoef', {j: round(c, 4) for j, c in zip(st['noll'], stv['zcoef'])}, flush=True)
    save(spl.knots, dict(vec_state=stv, D=A.D))
    sys.exit(0)

if A.method == 'e':
    S = spl.S
    u = spl.u0 + torch.arange(spl.nu, device=DEV, dtype=torch.float32) / S
    r = torch.maximum(u.abs()[None, :], u.abs()[:, None])
    ring = (r >= 10.5) & (r <= 11.5)
    inner = r <= 11.5
    s0 = Kb[:, inner].sum(1)

    def normalize(K):
        K = K - K[:, ring].mean(1)[:, None, None]
        return K * (s0 / K[:, inner].sum(1))[:, None, None]

    K = Kb.clone()
    if A.init:                     # continue from the knots of an earlier e run (roughness weights still from the beads)
        K = torch.load(A.init, weights_only=False)['knots'].to(DEV)
    wsm = 1.0 / (Kb.abs() + 1e-4)
    th = None
    best_e = None
    t0 = time.time()
    for o in range(A.outer + 1):
        spl.knots = K
        v = val_nll(K)
        if best_e is None or v[0] < best_e[0]:
            best_e = (v[0], K.clone(), o)        # early stopping on the validation set (not the test set)
        th, nll = localize_full(spl, dtr) if th is None else lm(spl, dtr, th, 5)
        c2, dev, _ = fit_stats(spl, dtr, th)
        log['it'].append(dict(it=o, train_dev=float(dev.mean()), train_chi2=float(c2.mean()), val_dev=v[0], val_chi2=v[1],
                              val_limits=v[2], train_limits=limit_fractions(th)))
        print(f'  outer {o}: train dev/spot {float(dev.mean()):.4f} chi2 {float(c2.mean()):.4f}  val dev/spot {v[0]:.4f} '
              f'chi2 {v[1]:.4f} limits {v[2]}  {time.time() - t0:.0f}s', flush=True)
        if o == A.outer:
            break
        Kp = K.clone().requires_grad_(True)
        lb = torch.optim.LBFGS([Kp], max_iter=8, history_size=10, line_search_fn='strong_wolfe')
        L0 = [None]

        def closure():
            lb.zero_grad()
            tot = 0.0
            for s_ in range(0, len(dtr), 20000):
                t = th[s_:s_ + 20000]
                mu = (t[:, 3, None, None] * spl.render(t[:, :3], knots=Kp) + t[:, 4, None, None]).clamp_min(1e-4)
                l = (mu - dtr[s_:s_ + 20000] * torch.log(mu)).double().sum()
                l.backward()
                tot += float(l)
            if A.lam > 0:
                dzz = Kp[2:] - 2 * Kp[1:-1] + Kp[:-2]
                dy = Kp[:, 2:] - 2 * Kp[:, 1:-1] + Kp[:, :-2]
                dx = Kp[:, :, 2:] - 2 * Kp[:, :, 1:-1] + Kp[:, :, :-2]
                reg = A.lam * ((wsm[1:-1] * dzz ** 2).sum() + (wsm[:, 1:-1] * dy ** 2).sum() + (wsm[:, :, 1:-1] * dx ** 2).sum())
                reg.backward()
                tot += float(reg)
            if L0[0] is None:
                L0[0] = tot
            return torch.tensor(tot - L0[0], device=DEV, dtype=torch.float64)
        lb.step(closure)
        K = normalize(Kp.detach())
    log['timing'] = dict(total_s=time.time() - t_all)
    log['best_outer'] = best_e[2]
    print(f'best validation deviance {best_e[0]:.4f} at outer {best_e[2]}', flush=True)
    save(best_e[1])
