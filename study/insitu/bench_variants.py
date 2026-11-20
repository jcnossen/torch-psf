import sys, time, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
spl = F.new_spline(); vec = C.make_vec(spl.Pl, 'A', 'insitu'); spl.knots = spl.sample(vec, C.insitu_pos)
N, NB = 200000, 5000
g = torch.Generator(device=C.DEV).manual_seed(1)
P3 = torch.stack([torch.rand(N, generator=g, device=C.DEV) - .5, torch.rand(N, generator=g, device=C.DEV) - .5, 0.05 + 0.7 * torch.rand(N, generator=g, device=C.DEV)], 1)
resid = torch.randn(N, C.P, C.P, device=C.DEV)
def ev_unfold(self, off, K):
    N, P, S, Lw = off.shape[0], self.roisize, self.S, self.Lw
    tz = ((off[:, 2] - self.z0) / self.hz).clamp(1, self.nz - 2 - 1e-4)
    t = torch.stack([tz, self.t0 - off[:, 1] * S, self.t0 - off[:, 0] * S], 1)
    ix = torch.floor(t); fr = t - ix; i0 = ix.long() - 1
    base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
    sub = K.reshape(-1)[base[:, None] + self.pat[None]].reshape(N, 4, Lw, Lw)
    wa = self._axis(fr, False)
    win = sub.unfold(2, 4, S).unfold(3, 4, S)                       # [N, 4, P, P, 4, 4]
    return torch.einsum('nzpqab,nz,na,nb->npq', win, wa[:, 0], wa[:, 1], wa[:, 2])
def ev_sep(self, off, K):
    # separable: z first (4 planes -> 1), then y, then x with small matmuls
    N, P, S, Lw = off.shape[0], self.roisize, self.S, self.Lw
    tz = ((off[:, 2] - self.z0) / self.hz).clamp(1, self.nz - 2 - 1e-4)
    t = torch.stack([tz, self.t0 - off[:, 1] * S, self.t0 - off[:, 0] * S], 1)
    ix = torch.floor(t); fr = t - ix; i0 = ix.long() - 1
    base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
    sub = K.reshape(-1)[base[:, None] + self.pat[None]].reshape(N, 4, Lw, Lw)
    wa = self._axis(fr, False)
    pl = torch.einsum('nzyx,nz->nyx', sub, wa[:, 0])                # [N, Lw, Lw]
    r = torch.einsum('npxa,na->npx', pl.unfold(1, 4, S), wa[:, 1])  # [N, P, Lw]
    return torch.einsum('npqb,nb->npq', r.unfold(2, 4, S), wa[:, 2])
def tm(fn, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time(); r = fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return min(ts) / N * 1e6, r
def back(ev):
    def f():
        K = spl.knots.clone().requires_grad_(True)
        for s in range(0, N, NB):
            (ev(P3[s:s + NB], K) * resid[s:s + NB]).sum().backward()
        return K.grad
    return f
ref = spl.render(P3[:1000])
for name, ev in (('conv2d', lambda o, K: spl.render(o, knots=K)), ('unfold+einsum', lambda o, K: ev_unfold(spl, o, K)), ('separable', lambda o, K: ev_sep(spl, o, K))):
    d = float((ev(P3[:1000], spl.knots) - ref).abs().max() / ref.abs().max())
    with torch.no_grad():
        tf, _ = tm(lambda: [ev(P3[s:s + NB], spl.knots) for s in range(0, N, NB)][-1])
    tb, gr = tm(back(ev))
    print(f'{name:14s} maxdiff {d:.1e}  forward {tf:.3f} us/spot  forward+knot gradient {tb:.3f} us/spot', flush=True)
