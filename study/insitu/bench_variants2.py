import sys, time, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
spl = F.new_spline(); vec = C.make_vec(spl.Pl, 'A', 'insitu'); spl.knots = spl.sample(vec, C.insitu_pos)
N, NB = 200000, 5000
g = torch.Generator(device=C.DEV).manual_seed(1)
P3 = torch.stack([torch.rand(N, generator=g, device=C.DEV) - .5, torch.rand(N, generator=g, device=C.DEV) - .5, 0.05 + 0.7 * torch.rand(N, generator=g, device=C.DEV)], 1)
resid = torch.randn(N, C.P, C.P, device=C.DEV)
def ev(self, off, K, mode):
    N, P, S, Lw = off.shape[0], self.roisize, self.S, self.Lw
    tz = ((off[:, 2] - self.z0) / self.hz).clamp(1, self.nz - 2 - 1e-4)
    t = torch.stack([tz, self.t0 - off[:, 1] * S, self.t0 - off[:, 0] * S], 1)
    ix = torch.floor(t); fr = t - ix; i0 = ix.long() - 1
    base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
    idx = base[:, None] + self.pat[None]
    if mode == 'index':
        sub = K.reshape(-1)[idx]
    elif mode == 'index_select':
        sub = K.reshape(-1).index_select(0, idx.reshape(-1))
    elif mode == 'take':
        sub = torch.take(K, idx)
    elif mode == 'rows':            # gather whole rows of Lw knots: unfold along x gives a view [nz, nv, nu-Lw+1, Lw]
        rows = K.reshape(self.nz, self.nv, self.nu).unfold(2, Lw, 1).reshape(-1, Lw)
        r = torch.arange(4, device=K.device)[:, None] * self.nv + torch.arange(Lw, device=K.device)[None]
        ridx = ((i0[:, 0] * self.nv + i0[:, 1])[:, None, None] + r[None]) * (self.nu - Lw + 1) + i0[:, 2, None, None]
        sub = rows.index_select(0, ridx.reshape(-1))
    sub = sub.reshape(1, N * 4, Lw, Lw)
    wa = self._axis(fr, False)
    w = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
    return torch.nn.functional.conv2d(sub, w, groups=N, stride=S).view(N, P, P)
def tm(fn, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time(); r = fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return min(ts) / N * 1e6, r
ref = spl.render(P3[:1000]); gref = None
for mode in ('index', 'index_select', 'take', 'rows'):
    d = float((ev(spl, P3[:1000], spl.knots, mode) - ref).abs().max())
    def f():
        K = spl.knots.clone().requires_grad_(True)
        for s in range(0, N, NB):
            (ev(spl, P3[s:s + NB], K, mode) * resid[s:s + NB]).sum().backward()
        return K.grad
    with torch.no_grad():
        tf, _ = tm(lambda: [ev(spl, P3[s:s + NB], spl.knots, mode) for s in range(0, N, NB)][-1])
    tb, gr = tm(f)
    gref = gr if gref is None else gref
    print(f'{mode:13s} maxdiff {d:.1e} forward {tf:.3f}  forward+knot gradient {tb:.3f} us/spot  grad rel diff {float((gr-gref).norm()/gref.norm()):.1e}', flush=True)
