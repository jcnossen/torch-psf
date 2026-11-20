"""Critic test 1: (a) is route K's Zernike / pupil-map gradient equal to route V's away from and at the truth;
(b) how large is the interpolation bias of route K at the truth on noise-free data, compared with the statistical
error at 1e4..1e6 emitters; (c) how much faster can route V be made with standard tricks (shared pupil precomputed,
x/y shift folded into the MFT matrices, larger batches, TF32 matmuls)?
Run from the repo root:  python study/insitu/critic/grad_speed.py"""
import math, sys, time, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
from torch_psf.fit import poisson_nll

torch.manual_seed(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
spl = F.new_spline()
zt = C.zc_insitu().to(C.DEV)
zs = torch.load('study/insitu/out_dev_A.pt', weights_only=False)['zc_s'].to(C.DEV)


def sync_time(fn, rep=3):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep):
        r = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / rep, r


def routes(zc, d, th, names=('zcoef',)):
    big, small = C.model_vec(spl.Pl, zc), C.model_vec(C.P, zc)
    spl.knots = spl.sample(big, C.insitu_pos)
    gK, _ = F.knot_grad(spl, d, th)
    gk = spl.vjp_vectorial(big, C.insitu_pos, gK, names=names)
    # route V for all named parameters (buffers included)
    base = {k: v.detach() for k, v in list(small.named_parameters()) + list(small.named_buffers())}
    gv = {k: torch.zeros_like(base[k]) for k in names}
    for s in range(0, len(d), 250):
        prm = dict(base)
        for k in names:
            prm[k] = base[k].clone().requires_grad_(True)
        t = th[s:s + 250]
        g_ = torch.func.functional_call(small, prm, (t[:, :3], torch.full((len(t),), C.F0, device=C.DEV)))
        l = poisson_nll(d[s:s + 250], t[:, 3, None, None] * g_ + t[:, 4, None, None]).sum()
        gr = torch.autograd.grad(l, [prm[k] for k in names])
        for k, g in zip(names, gr):
            gv[k] += g
    return gk, gv


def cmp(a, b):
    a, b = a.flatten(), b.flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, 0)), float((a - b).norm() / b.norm())


# ---------------------------------------------------------------- (a) gradient identity at several points, noisy data
test = C.sim_test('A', N, seed=31)
d, tht = test['d'], test['th']
print(f'== (a) gradient identity, {N} noisy clean patches (scenario A truth), emitter parameters fixed at the truth')
pts = dict(bead_fit=zs, truth=zt, truth_plus_0p01=zt + 0.01 * torch.randn(len(zt), device=C.DEV))
for name, zc in pts.items():
    gk, gv = routes(zc, d, tht)
    c, r = cmp(gk['zcoef'], gv['zcoef'])
    print(f'  zcoef at {name:16s}: cos {c:.6f}  rel diff {r:.2e}  |gV| {gv["zcoef"].norm():.3e}')
gk, gv = routes(zt, d, tht, names=('pupil_phase', 'pupil_amp'))
for k in ('pupil_phase', 'pupil_amp'):
    m = C.model_vec(C.P).mask.bool()
    c, r = cmp(gk[k][m], gv[k][m])
    print(f'  {k:11s} map at truth: cos {c:.6f}  rel diff {r:.2e}  (pixels inside the aperture)')

# ---------------------------------------------------------------- (b) interpolation bias at the truth, noise-free data
print('== (b) interpolation bias of route K: noise-free data (d = mu), true emitter parameters, true Zernikes')
dm = test['mu']
gk, gv = routes(zt, dm, tht)
bK, bV = gk['zcoef'], gv['zcoef']
print(f'  |grad V| (should be ~0) {bV.norm():.3e}   |grad K| {bK.norm():.3e}')
# Fisher information of zcoef (emitter parameters known) from the vectorial model, per emitter
small = C.model_vec(C.P, zt)
base = {k: v.detach() for k, v in small.named_parameters()}
nF = min(N, 600)


def mu_of(zc, t):
    prm = dict(base); prm['zcoef'] = zc
    return t[:, 3, None, None] * torch.func.functional_call(small, prm, (t[:, :3], torch.full((len(t),), C.F0, device=C.DEV))) + t[:, 4, None, None]


Fi = torch.zeros(len(zt), len(zt), device=C.DEV, dtype=torch.float64)
for s in range(0, nF, 50):
    t = tht[s:s + 50]
    J = torch.func.jacfwd(lambda z: mu_of(z, t))(zt)            # [n, P, P, 17]
    mu = mu_of(zt, t).detach()
    Jf = J.flatten(1, 2).double()
    Fi += torch.einsum('npa,npb->ab', Jf / mu.flatten(1)[..., None].double(), Jf)
Fi /= nF                                                        # per emitter
bias_per = (bK - bV).double() / N                               # gradient bias per emitter
dphi = -torch.linalg.solve(Fi, bias_per)                        # fixed-point shift of route K (independent of N)
print(f'  fixed-point shift of the Zernikes from interpolation error: rms {dphi.pow(2).mean().sqrt():.2e} waves, max {dphi.abs().max():.2e}')
crlb1 = torch.sqrt(torch.diagonal(torch.linalg.inv(Fi)))
for n in (1e4, 1e5, 1e6):
    print(f'  N {n:.0e}: CRLB per Zernike (rms over modes) {(crlb1 / math.sqrt(n)).pow(2).mean().sqrt():.2e} waves;  bias/CRLB (rms) {(dphi / (crlb1 / math.sqrt(n))).pow(2).mean().sqrt():.2f}')
print('  (emitter parameters known; with them free the information is lower, so the CRLB is larger and these bias/CRLB ratios are upper bounds)')
del test

# ---------------------------------------------------------------- (c) speed of route V, as written and optimised
print('== (c) route V cost per emitter (64 pupil, oversampling 3, 15 px), interleaved with the knot gradient')
rois = torch.load('study/insitu/out_dev_A.pt', weights_only=False)['rois'].to(C.DEV)[:N]
big = C.model_vec(spl.Pl, zs); small = C.model_vec(C.P, zs)
spl.knots = spl.sample(big, C.insitu_pos)
th, _ = localize(spl, rois)


def vec_grad_fast(psf, d, th, nb=1000):
    """Same model and same gradient as F.vec_grad_direct; shared pupil (fields x aberration phase) computed once
    per call, x/y shift folded into the MFT matrices, only the z phase per emitter."""
    zc = psf.zcoef.detach().clone().requires_grad_(True)
    tot, g = 0.0, torch.zeros_like(zc)
    with torch.no_grad():
        P, kz_med, kz_imm, kx, ky = psf.pupil_fields()
    kx1, ky1 = kx[0], ky[:, 0]
    dk = (kx1[1] - kx1[0]) * (ky1[1] - ky1[0])
    energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
    R, o = psf.roisize, psf.os
    for s in range(0, len(d), nb):
        t = th[s:s + nb]
        Wp = torch.einsum('j,jnm->nm', zc, psf.Z) * (2 * math.pi)
        Pa = P * torch.exp(1j * Wp)                                          # [6, N, N], shared
        dph = torch.exp(1j * (kz_med[None] * t[:, 2, None, None] - kz_imm[None] * C.F0))   # [B, N, N]
        x = t[:, 0, None, None] * psf.pixel_um; y = t[:, 1, None, None] * psf.pixel_um
        Fx = torch.exp(1j * (psf.u[None, :, None] - x) * kx1[None, None, :])                # [B, ns, N]
        Fy = torch.exp(1j * (psf.u[None, :, None] - y) * ky1[None, None, :])
        E = Fy[:, None] @ (Pa[None] * dph[:, None]) @ Fx[:, None].transpose(-1, -2)          # [B, 6, ns, ns]
        I = (E.real ** 2 + E.imag ** 2).sum(1) / 3 * (dk / (2 * math.pi) ** 2) ** 2
        I = I.reshape(len(t), R, o, R, o).mean((2, 4)) * psf.pixel_um ** 2 / energy
        l = poisson_nll(d[s:s + nb], t[:, 3, None, None] * I + t[:, 4, None, None]).sum()
        g += torch.autograd.grad(l, zc)[0]
        tot += float(l)
    return g, tot


gref, _ = F.vec_grad_direct(small, rois, th)
gf, _ = vec_grad_fast(small, rois, th)
print(f'  fast route V vs as written: cos {cmp(gf, gref)[0]:.6f} rel diff {cmp(gf, gref)[1]:.2e}')
res = {}
for rnd in range(2):
    for name, fn in (('V as written (nb 250)', lambda: F.vec_grad_direct(small, rois, th)),
                     ('V as written, nb 1000', lambda: F.vec_grad_direct(small, rois, th, nb=1000)),
                     ('V fast, fp32', lambda: vec_grad_fast(small, rois, th)),
                     ('K knot gradient', lambda: F.knot_grad(spl, rois, th)),
                     ('K vjp (fixed)', lambda: spl.vjp_vectorial(big, C.insitu_pos, F.knot_grad(spl, rois, th)[0])),
                     ('K build (fixed)', lambda: spl.sample(big, C.insitu_pos)),
                     ('localize 3 its (both)', lambda: localize(spl, rois, th, iters=3))):
        tt, _ = sync_time(fn, rep=2)
        res.setdefault(name, []).append(tt)
    torch.backends.cuda.matmul.allow_tf32 = True
    tt, (gt, _) = sync_time(lambda: vec_grad_fast(small, rois, th), rep=2)
    res.setdefault('V fast, TF32 matmul', []).append(tt)
    torch.backends.cuda.matmul.allow_tf32 = False
print(f'  TF32 gradient vs fp32: cos {cmp(gt, gref)[0]:.6f} rel diff {cmp(gt, gref)[1]:.2e}')
for k, v in res.items():
    tm = min(v)
    per = tm / N * 1e6
    print(f'  {k:24s} {tm * 1e3:8.1f} ms   {per:7.1f} us/emitter' + ('' if 'fixed' not in k else '  (per iteration)'))
