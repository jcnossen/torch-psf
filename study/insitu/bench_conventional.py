"""Convolution spline (knots, this repo) against the conventional cubic-spline PSF (64 polynomial coefficients per
voxel, evaluated pixel by pixel in a CUDA kernel): the fastpsf library (CSplineMethods), same GPU, same spots.

Spots: simulated with the vectorial model (scenario A, sample condition), Poisson noise, 15 x 15 px.
Conventional: spline through the PSF sampled at the camera pixel spacing (as calibrated from a bead stack), 40 nm in z.
Convolution spline: knots at half-pixel spacing (S = 2), 40 nm in z.
Both fit (x, y, z, I, bg) by Levenberg-Marquardt for a fixed number of iterations from the same start.
Writes study/insitu/out/conventional.json."""
import sys, time, json, numpy as np, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
try:
    from fastpsf import Context, CSplineMethods
    from fastpsf.estim_queue import EstimQueue
except ImportError:
    Context = None                      # native reference library not installed: PyTorch comparison only
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N = 200000
test = C.sim_test('A', 20000, seed=77)
d0, th0 = test['d'], test['th']
rep = N // len(d0)
spl = F.new_spline()
vec = C.make_vec(spl.Pl, 'A', 'insitu')
spl.knots = spl.sample(vec, C.insitu_pos)
# pixel-sampled z-stack for the conventional spline: emitter at the patch centre, the knot z planes of spl
zs = spl.zs.float()
zstack = C.render(vec, torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], 1), torch.full_like(zs, C.F0)).cpu().numpy()
res = dict(iters=ITERS, N=N, gpu=torch.cuda.get_device_name(0), knots_MB=spl.knots.numel() * 4 / 1e6)
start = th0.clone()
g = torch.Generator(device=C.DEV).manual_seed(1)
start[:, :2] += 0.3 * torch.randn(len(start), 2, generator=g, device=C.DEV)
start[:, 2] += 0.05 * torch.randn(len(start), generator=g, device=C.DEV)
start[:, 3] *= 0.8


def stats(est, name):
    e = est - th0.cpu().numpy()
    e[:, :2] -= np.median(e[:, :2], 0)                       # coordinate-origin convention of the library
    ok = np.abs(e[:, 2]) < 0.15
    r = dict(x_rms=float(e[ok, 0].std() * 108), y_rms=float(e[ok, 1].std() * 108), z_rms=float(np.sqrt((e[ok, 2] ** 2).mean()) * 1000),
             z_bias=float(e[ok, 2].mean() * 1000), wrong_z=float(1 - ok.mean()), I_ratio=float(np.median(est[ok, 3] / th0.cpu().numpy()[ok, 3])))
    print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    return r


def best(fn, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return min(ts), r


# ---------------------------------------------------------------- convolution spline
D = d0.repeat(rep, 1, 1); S0 = start.repeat(rep, 1)
localize(spl, D[:3000], S0[:3000].clone(), iters=ITERS)
t_fit, (th_c, _) = best(lambda: localize(spl, D, S0.clone(), iters=ITERS, batch=20000))
Dn = D.cpu().numpy(); Sn = S0.cpu().numpy()
t_fit_io, _ = best(lambda: localize(spl, torch.from_numpy(Dn).to(C.DEV), torch.from_numpy(Sn).to(C.DEV), iters=ITERS, batch=20000)[0].cpu().numpy())
P3 = S0[:, :3]
t_ev, _ = best(lambda: [spl.render(P3[s:s + 20000]) for s in range(0, N, 20000)][-1])
t_evd, _ = best(lambda: [spl.render(P3[s:s + 20000], grad=True) for s in range(0, N, 20000)][-1])
res['conv'] = dict(fit_us=t_fit / N * 1e6, fit_us_with_transfer=t_fit_io / N * 1e6, eval_us=t_ev / N * 1e6, eval_deriv_us=t_evd / N * 1e6,
                   **stats(th_c[:len(d0)].cpu().numpy(), 'conv spline  '))
print('conv spline: fit', res['conv']['fit_us'], 'us/spot (', res['conv']['fit_us_with_transfer'], 'with host transfer ), eval', res['conv']['eval_us'],
      'eval+deriv', res['conv']['eval_deriv_us'], flush=True)

# ---------------------------------------------------------------- conventional form, also in PyTorch
class CoefPSF:
    """The SAME spline as spl, stored the conventional way: 64 polynomial coefficients per voxel (one voxel = one
    knot interval), evaluated per pixel as coefficients . monomials. Built from the knots, so it stays
    differentiable with respect to them through the conversion."""
    def __init__(self, spl):
        self.spl = spl
        self.nv, self.nu = spl.nv - 3, spl.nu - 3
        r = torch.arange(spl.roisize, device=spl.dev) * spl.S
        self.pat = (r[:, None] * self.nu + r[None, :]).reshape(-1)
        self.e = torch.arange(4, device=spl.dev)

    def coefs(self, K):
        win = K.unfold(0, 4, 1).unfold(1, 4, 1).unfold(2, 4, 1)                 # [nz-3, nv-3, nu-3, 4, 4, 4]
        G = self.spl.G
        return torch.einsum('ia,jb,ke,zyxabe->zyxijk', G, G, G, win).reshape(-1, 64)

    def evaluate(self, off, coef, grad=False):
        spl, S = self.spl, self.spl.S
        tz = ((off[:, 2] - spl.z0) / spl.hz).clamp(1, spl.nz - 2 - 1e-4)
        t = torch.stack([tz, spl.t0 - off[:, 1].clamp(-spl.xmax, spl.xmax) * S, spl.t0 - off[:, 0].clamp(-spl.xmax, spl.xmax) * S], 1)
        ix = torch.floor(t); fr = t - ix; i0 = ix.long() - 1
        base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
        c = coef[base[:, None] + self.pat[None]]                                # [N, P*P, 64]
        p = fr[..., None] ** self.e                                             # [N, 3, 4] powers of (tz, ty, tx)
        mon = (p[:, 0, :, None, None] * p[:, 1, None, :, None] * p[:, 2, None, None, :]).reshape(-1, 64)
        if not grad:
            return torch.einsum('npc,nc->np', c, mon).view(-1, spl.roisize, spl.roisize)
        dp = self.e * fr[..., None] ** (self.e - 1).clamp_min(0)
        mz = (dp[:, 0, :, None, None] * p[:, 1, None, :, None] * p[:, 2, None, None, :]).reshape(-1, 64)
        my = (p[:, 0, :, None, None] * dp[:, 1, None, :, None] * p[:, 2, None, None, :]).reshape(-1, 64)
        mx = (p[:, 0, :, None, None] * p[:, 1, None, :, None] * dp[:, 2, None, None, :]).reshape(-1, 64)
        r = torch.einsum('npc,ndc->ndp', c, torch.stack([mon, mx, my, mz], 1)).view(-1, 4, spl.roisize, spl.roisize)
        return r[:, 0], -S * r[:, 1], -S * r[:, 2], r[:, 3] / spl.hz


cp = CoefPSF(spl)
coef = cp.coefs(spl.knots[0])
res['coef_MB'] = coef.numel() * 4 / 1e6
a_, b_ = spl.render(P3[:2000], grad=True), cp.evaluate(P3[:2000], coef, grad=True)
res['coef_vs_conv_maxdiff'] = [float((x - y).abs().max() / x.abs().max()) for x, y in zip(a_, b_)]
print('coefficient form vs convolution form, max relative difference (value, dx, dy, dz):', res['coef_vs_conv_maxdiff'], flush=True)
NB = 5000
resid = torch.randn(N, C.P, C.P, device=C.DEV)


def back_conv():
    K = spl.knots.clone().requires_grad_(True)
    for s in range(0, N, NB):
        (spl.render_tables(P3[s:s + NB], knots=K) * resid[s:s + NB]).sum().backward()
    return K.grad


def back_coef():
    K = spl.knots.clone().requires_grad_(True)
    for s in range(0, N, NB):
        (cp.evaluate(P3[s:s + NB], cp.coefs(K.reshape(spl.nz, spl.nv, spl.nu))) * resid[s:s + NB]).sum().backward()
    return K.grad


def back_coef_direct():                      # gradient on the 64-coefficient table itself (not a valid spline)
    cf = coef.clone().requires_grad_(True)
    for s in range(0, N, NB):
        (cp.evaluate(P3[s:s + NB], cf) * resid[s:s + NB]).sum().backward()
    return cf.grad


tt = {}
with torch.no_grad():
    tt['conv_eval'], _ = best(lambda: [spl.render(P3[s:s + NB]) for s in range(0, N, NB)][-1])
    tt['coef_eval'], _ = best(lambda: [cp.evaluate(P3[s:s + NB], coef) for s in range(0, N, NB)][-1])
    tt['conv_eval_deriv'], _ = best(lambda: [spl.render(P3[s:s + NB], grad=True) for s in range(0, N, NB)][-1])
    tt['coef_eval_deriv'], _ = best(lambda: [cp.evaluate(P3[s:s + NB], coef, grad=True) for s in range(0, N, NB)][-1])
tt['conv_knot_grad'], g1 = best(back_conv)
tt['coef_knot_grad'], g2 = best(back_coef)
tt['coef_table_grad'], _ = best(back_coef_direct)
res['knot_grad_rel_diff'] = float((g1 - g2).norm() / g1.norm())
res['torch_us'] = {k: v / N * 1e6 for k, v in tt.items()}
for k in ('eval', 'eval_deriv', 'knot_grad'):
    print(f"PyTorch, {k:11s}: convolution {res['torch_us']['conv_' + k]:.3f} us/spot, coefficient table {res['torch_us']['coef_' + k]:.3f} us/spot, "
          f"ratio {tt['coef_' + k] / tt['conv_' + k]:.1f}", flush=True)
print('gradient on the coefficient table itself:', res['torch_us']['coef_table_grad'], 'us/spot; knot gradients agree to', res['knot_grad_rel_diff'],
      '; memory: knots', res['knots_MB'], 'MB, coefficients', res['coef_MB'], 'MB', flush=True)
# three cases: (1) convolution on the knots, (2) coefficient table rebuilt from the knots at every use,
# (3) coefficient table computed once. Rebuild time, and peak GPU memory of one batch of NB spots.
t_rebuild, _ = best(lambda: cp.coefs(spl.knots[0]), n=10)
res['rebuild_ms'] = t_rebuild * 1e3


def peak(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


o, rs = P3[:NB], resid[:NB]


def g_conv():
    K = spl.knots.clone().requires_grad_(True); (spl.render_tables(o, knots=K) * rs).sum().backward()


def g_coef():
    K = spl.knots.clone().requires_grad_(True); (cp.evaluate(o, cp.coefs(K.reshape(spl.nz, spl.nv, spl.nu))) * rs).sum().backward()


with torch.no_grad():
    res['peak_MB'] = dict(conv_eval=peak(lambda: spl.render(o)), rebuilt_eval=peak(lambda: cp.evaluate(o, cp.coefs(spl.knots[0]))),
                          table_eval=peak(lambda: cp.evaluate(o, coef)))
    tt['rebuilt_eval'], _ = best(lambda: [cp.evaluate(P3[s:s + NB], cp.coefs(spl.knots[0])) for s in range(0, N, NB)][-1])
    tt['rebuilt_eval_deriv'], _ = best(lambda: [cp.evaluate(P3[s:s + NB], cp.coefs(spl.knots[0]), grad=True) for s in range(0, N, NB)][-1])
res['peak_MB'].update(conv_grad=peak(g_conv), rebuilt_grad=peak(g_coef))
res['torch_us'] = {k: v / N * 1e6 for k, v in tt.items()}
res['Mspots_per_s'] = {k: N / v / 1e6 for k, v in tt.items()}
print('rebuild of the coefficient table from the knots:', round(res['rebuild_ms'], 2), 'ms;  peak memory per batch of', NB, 'spots (MB):',
      {k: round(v, 1) for k, v in res['peak_MB'].items()}, flush=True)
print('million spots per second:', {k: round(v, 2) for k, v in res['Mspots_per_s'].items()}, flush=True)
json.dump(res, open('study/insitu/out/conventional.json', 'w'), indent=1)

# ---------------------------------------------------------------- conventional spline, native CUDA (fastpsf)
if '--torch-only' in sys.argv:
    sys.exit()
if Context is None:
    print(json.dumps(res, indent=1)); sys.exit()
with Context(debugMode=False) as ctx:
    api = CSplineMethods(ctx)
    psf = api.CreatePSFFromZStack(C.P, zstack, [float(zs[0]), float(zs[-1])], cuda=True)
    # coordinate convention: find the offset between patch-centre offsets and the library's x, y
    par = th0.cpu().numpy().astype(np.float32).copy()
    c0 = None
    for cen in (C.P / 2, (C.P - 1) / 2):
        p = par.copy(); p[:, 0] += cen; p[:, 1] += cen
        ev = psf.ExpectedValue(p[:200])
        err = np.abs(ev - test['mu'][:200].cpu().numpy()).sum() / test['mu'][:200].cpu().numpy().sum()
        print('centre', cen, 'rel L1 difference of expected value from the vectorial truth', round(float(err), 4), flush=True)
        if c0 is None or err < c0[1]:
            c0 = (cen, err)
    cen = c0[0]
    res['conventional_model_err_L1'] = float(c0[1])
    evc = spl.render(th0[:200, :3]) * th0[:200, 3, None, None] + th0[:200, 4, None, None]
    res['conv_model_err_L1'] = float((evc - test['mu'][:200]).abs().sum() / test['mu'][:200].sum())
    lmp, stepcoeff, stoplimit = psf.GetLevMarParams()
    psf.SetLevMarParams(stepcoeff, ITERS, normalizeWeights=True, stoplimits=np.zeros_like(stoplimit))
    ini = Sn.astype(np.float32).copy(); ini[:, 0] += cen; ini[:, 1] += cen
    smp = np.ascontiguousarray(Dn, dtype=np.float32)
    est, diag, traces = psf.Estimate(smp[:len(d0)], initial=ini[:len(d0)])
    its = np.array([len(t) for t in traces])
    res['conventional_iterations'] = [int(its.min()), float(its.mean()), int(its.max())]
    print('conventional LM iterations used: min/mean/max', res['conventional_iterations'], flush=True)

    def run_queue():
        q = EstimQueue(psf, batchSize=4096, numStreams=4)
        q.Schedule(smp, initial=ini)
        q.Flush()
        while not q.IsIdle():
            time.sleep(0.002)
        r = q.GetResults()
        q.Destroy()
        return r
    run_queue()
    t_q, _ = best(run_queue)
    pe = ini[:, :5].copy(); pe[:, 3] = 1; pe[:, 4] = 0
    t_ev, _ = best(lambda: psf.ExpectedValue(pe))
    t_evd, _ = best(lambda: psf.Derivatives(pe))
    est[:, 0] -= cen; est[:, 1] -= cen
    res['conventional'] = dict(fit_us=t_q / N * 1e6, eval_us=t_ev / N * 1e6, eval_deriv_us=t_evd / N * 1e6, **stats(est, 'conventional '))
    print('conventional: fit (queue)', res['conventional']['fit_us'], 'us/spot, eval', res['conventional']['eval_us'], 'eval+deriv',
          res['conventional']['eval_deriv_us'], '(all include host <-> GPU transfer)', flush=True)
json.dump(res, open('study/insitu/out/conventional.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
