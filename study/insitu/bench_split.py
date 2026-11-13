"""Where does the time of the convolution spline go: reading the knot windows (index_select) or the convolution?
Forward and backward separately. Decides whether a custom block-copy kernel (fast_slice) with a matching
block-add backward would pay off. Writes study/insitu/out/split.json."""
import sys, time, json, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
spl = F.new_spline()
spl.knots = spl.stack_from_vectorial(C.make_vec(spl.Pl, 'A', 'insitu'), C.insitu_pos)
N, NB = 200000, 5000
g = torch.Generator(device=C.DEV).manual_seed(1)
r = lambda: torch.rand(N, generator=g, device=C.DEV)
P3 = torch.stack([r() - .5, r() - .5, 0.05 + 0.7 * r()], 1)
resid = torch.randn(N, C.P, C.P, device=C.DEV)
S, P, Lw = spl.S, spl.P, spl.Lw


def prep(off):
    tz = ((off[:, 2] - spl.z0) / spl.hz).clamp(1, spl.nz - 2 - 1e-4)
    t = torch.stack([tz, spl.oS0 - off[:, 1] * S, spl.oS0 - off[:, 0] * S], 1)
    ix = torch.floor(t); fr = t - ix; i0 = ix.long() - 1
    idx = ((i0[:, 0] * spl.nv * spl.nu + i0[:, 1] * spl.nu + i0[:, 2])[:, None] + spl.pat[None]).reshape(-1)
    wa = spl._axis(fr, False)
    w = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
    return idx, w


def best(fn, n=5):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return min(ts) / N * 1e6


B = [(s, min(s + NB, N)) for s in range(0, N, NB)]
pre = [prep(P3[a:b]) for a, b in B]
Kf = spl.knots.reshape(-1)
subs = [Kf.index_select(0, idx).reshape(1, -1, Lw, Lw) for idx, w in pre]
res = {}
with torch.no_grad():
    res['prep_index_weights'] = best(lambda: [prep(P3[a:b]) for a, b in B])
    res['fwd_window_read'] = best(lambda: [Kf.index_select(0, idx) for idx, w in pre])
    res['fwd_convolution'] = best(lambda: [torch.nn.functional.conv2d(sub, w, groups=w.shape[0], stride=S) for sub, (idx, w) in zip(subs, pre)])
    res['fwd_total'] = best(lambda: [spl.evaluate(P3[a:b]) for a, b in B])


def bwd_conv():                      # forward + backward of the convolution only (gradient on the windows)
    for sub, (idx, w), (a, b) in zip(subs, pre, B):
        s_ = sub.clone().requires_grad_(True)
        (torch.nn.functional.conv2d(s_, w, groups=w.shape[0], stride=S).view(-1, P, P) * resid[a:b]).sum().backward()


def bwd_read():                      # forward + backward of the window read only (scatter-add into the knots)
    for (idx, w), sub in zip(pre, subs):
        K = Kf.clone().requires_grad_(True)
        K.index_select(0, idx).backward(sub.reshape(-1))


def bwd_total():
    K = spl.knots.clone().requires_grad_(True)
    for a, b in B:
        (spl.evaluate(P3[a:b], knots=K) * resid[a:b]).sum().backward()


res['fwdbwd_convolution'] = best(bwd_conv)
res['fwdbwd_window_read'] = best(bwd_read)
res['fwdbwd_total'] = best(bwd_total)
res['gpu'] = torch.cuda.get_device_name(0)
for k, v in res.items():
    print(f'{k:22s} {v:.3f} us/spot' if isinstance(v, float) else f'{k} {v}', flush=True)
json.dump(res, open('study/insitu/out/split.json', 'w'), indent=1)
