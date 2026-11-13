"""Does a structured-illumination pattern fix the tilt of positions with z? (suggested by the simcode2 session)

Each emitter is seen in 6 frames: 2 pattern orientations (x, y) x 3 phases, pitch 220 nm, modulation depth M.
Frame intensity I_t = N r_t(X, Y), r_t = (1 + M cos(k_t . R - phi_t)) / 6, R = absolute emitter position.
The emitter position (x, y, z) is shared by the PSF term and the pattern term, so the pattern is a ruler for x, y
that does not depend on the PSF shape. The pattern (pitch, phases, depth) is taken as known here.
Control: M = 0 (same frames and photons, uniform illumination, so no pattern information).
"""
import sys, time, json, math, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_convspline.knotpsf import localize, poisson_nll, _clamp
M = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
scns = sys.argv[2] if len(sys.argv) > 2 else 'AB'
NE, NTOT, BG, PITCH = 3000, 12000.0, 30.0, 220 / 108
K0 = 2 * math.pi / PITCH
KV = torch.tensor([[K0, 0.]] * 3 + [[0., K0]] * 3, device=C.DEV)                  # [6, 2]
PH = torch.tensor([0, 1, 2, 0, 1, 2], device=C.DEV) * 2 * math.pi / 3


def relint(xy, corner, deriv=False):
    """r_t [Ne, 6] at patch offsets xy [Ne, 2] (absolute position = corner + xy), and d r / dx, d r / dy."""
    arg = (corner + xy) @ KV.T - PH[None]
    r = (1 + M * torch.cos(arg)) / 6
    if not deriv:
        return r
    dr = -M * torch.sin(arg) / 6
    return r, dr * KV[None, :, 0], dr * KV[None, :, 1]


def simulate(scn, seed=0):
    g = torch.Generator(device=C.DEV).manual_seed(500 + seed)
    r = lambda *s: torch.rand(*s, generator=g, device=C.DEV)
    psf = C.make_vec(C.P, scn, 'insitu')
    th = torch.stack([r(NE) - 0.5, r(NE) - 0.5, 0.8 * r(NE), NTOT * torch.exp(0.4 * torch.randn(NE, generator=g, device=C.DEV)),
                      torch.full((NE,), BG, device=C.DEV)], 1)
    corner = torch.randint(0, 100, (NE, 2), generator=g, device=C.DEV).float()
    img = C.render(psf, th[:, :3], torch.full((NE,), C.F0, device=C.DEV))
    mu = th[:, 3, None, None, None] * relint(th[:, :2], corner)[:, :, None, None] * img[:, None] + BG
    return torch.poisson(mu, generator=g), corner, th


class PatternLoc:
    """Emitter parameters (x, y, z, N, bg) from the 6 frames of each emitter, damped Newton on the spline; returns
    the per-frame parameters (x, y, z, N r_t, bg) [Ne * 6, 5] that the knot / pupil gradients need."""
    def __init__(self, d6, corner):
        self.d6, self.corner, self.th = d6, corner, None

    def model(self, spl, t, corner):
        g, gx, gy, gz = spl.evaluate(t[:, :3], grad=True)
        r, rx, ry = relint(t[:, :2], corner, True)
        N = t[:, 3, None, None, None]
        G = lambda a: a[:, None]
        Rr = lambda a: a[:, :, None, None]
        mu = N * Rr(r) * G(g) + t[:, 4, None, None, None]
        J = torch.stack([N * (Rr(rx) * G(g) + Rr(r) * G(gx)), N * (Rr(ry) * G(g) + Rr(r) * G(gy)), N * Rr(r) * G(gz),
                         Rr(r) * G(g), torch.ones_like(mu)], 1)                 # [Ne, 5, 6, P, P]
        return mu, J

    @torch.no_grad()
    def __call__(self, spl, d_unused, th_unused, nb=1000):
        if self.th is None:
            t0, _ = localize(spl, self.d6.sum(1))
            t0[:, 4] /= 6
            self.th, iters = t0, 30
        else:
            iters = 3
        out, onll = [], []
        for s in range(0, len(self.d6), nb):
            d, corner = self.d6[s:s + nb], self.corner[s:s + nb]
            th = _clamp(self.th[s:s + nb].clone(), spl)
            lam = torch.ones(len(th), device=C.DEV)
            nl = lambda mu: poisson_nll(d.flatten(1, 2), mu.flatten(1, 2))
            mu, J = self.model(spl, th, corner)
            nll = nl(mu)
            for _ in range(iters):
                mu = mu.clamp_min(1e-6)
                Jf = J.flatten(2)
                gr = (Jf * (d / mu - 1).flatten(1)[:, None]).sum(2)
                H = torch.einsum('nap,nbp->nab', Jf / mu.flatten(1)[:, None], Jf)
                H = H + torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-12) * lam[:, None])
                tn = _clamp(th + torch.linalg.solve(H.double(), gr.double()[..., None])[..., 0].float(), spl)
                mun, Jn = self.model(spl, tn, corner)
                nn_ = nl(mun)
                ok = nn_ < nll
                th[ok], mu[ok], J[ok], nll[ok] = tn[ok], mun[ok], Jn[ok], nn_[ok]
                lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-4, 1e6)
            out.append(th); onll.append(nll)
        self.th = th = torch.cat(out)
        r = relint(th[:, :2], self.corner)
        pf = th[:, None, :].expand(-1, 6, -1).clone()
        pf[:, :, 3] = th[:, 3, None] * r
        return pf.reshape(-1, 5), torch.cat(onll)


res = {}
for scn in scns:
    spl = F.new_spline()
    Ktrue = spl.stack_from_vectorial(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
    test = C.sim_test(scn, 6000)
    spl.knots = Ktrue
    _, nll_true = localize(spl, test['d'])
    d6, corner, th_true = simulate(scn)
    dflat = d6.reshape(-1, C.P, C.P)
    zc_s = F.fit_vec_stack(C.sim_stack(scn))

    def rec(name, K, sec):
        r = F.evaluate(spl, K, test, nll_true)
        th = r['th']
        e = th - test['th']
        z = test['th'][:, 2]
        good = e[:, 2].abs() < 0.15
        b = torch.linspace(0.05, 0.75, 8, device=C.DEV)
        ib = torch.bucketize(z, b).clamp(1, 7) - 1
        cur = {n: [float(e[good & (ib == i), k].mean()) * sc for i in range(7)]
               for k, (n, sc) in enumerate((('x', 108), ('y', 108), ('z', 1000)))}
        rms = [float(e[good, k].pow(2).mean().sqrt()) * sc for k, sc in enumerate((108, 108, 1000))]
        res[f'{scn}/{name}'] = dict(dnll=r['dnll'], curve=cur, sec=sec, wrong_z=float(1 - good.float().mean()), rms=rms)
        f = lambda v: f'{min(v):6.1f}..{max(v):5.1f}'
        print(f"M {M} {scn} {name:12s} dnll {r['dnll']:6.2f}  rms x {rms[0]:.1f} y {rms[1]:.1f} z {rms[2]:.1f}  "
              f"mean err over z: x {f(cur['x'])}  y {f(cur['y'])}  z {f(cur['z'])} nm  {sec:.0f}s", flush=True)

    t = time.time()
    # stage 0: Zernikes from the summed frames without the pattern. Starting the pattern fit from the bead PSF locks
    # emitters onto the wrong fringe (the PSF-only position is then off by more than half a pitch at some heights).
    zc0, _ = F.fit_vec_insitu(spl, d6.sum(1), zc_s, iters=60)
    spl.knots = F.knots_from(spl, zc0)
    loc = PatternLoc(d6, corner)
    zc, thf = F.fit_vec_insitu(spl, dflat, zc0, iters=40, loc=loc)
    Kz = F.knots_from(spl, zc)
    rec('Zernike', Kz, time.time() - t)
    th_e = loc.th.clone()
    t = time.time()
    Kp, prm, _ = F.fit_pupil_insitu(spl, dflat, zc, iters=200, th=thf, loc=loc)
    rec('free pupil', Kp, time.time() - t)
    t = time.time()
    loc2 = PatternLoc(d6, corner)
    loc2.th = th_e
    Kk, _ = F.fit_knots(spl, Kz, insitu=dflat, ref=Kz, lam_ref=3000., lam_s=1000., th_ins=thf, loc=loc2)
    rec('free knots', Kk, time.time() - t)
json.dump(res, open(f'study/insitu/out/pattern_M{M:g}.json', 'w'), indent=1)
