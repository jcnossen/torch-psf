"""Step 2 check: streaming cumulants of a simulated dense movie vs the analytic expectation G = sum_p rho_p (*) K_{d,p}
(pixel-integrated PSFs, emitters on a sub-pixel grid so the grid model is exact). Also checks the explicit patch
matrices of the Fisher code against the convolution model.

python validate_accumulator.py [--T 20000]
"""
import argparse
from common import *
from torch_psf.sofi.model import kernels, model_G, rho_from_emitters
from torch_psf.sofi.fisher import patch_matrix, pairs

ap = argparse.ArgumentParser()
ap.add_argument('--T', type=int, default=20000)
ap.add_argument('--H', type=int, default=40)
a = ap.parse_args()

Q, os_, R = 15, 2, 5
zern = random_aberrations(0.05, seed=1)
psf = make_psf(Q, zern, astig=0.08)
blink = BlinkModel(duty=0.3, t_on=2.0)
spec = DenseMovieSpec(H=a.H, W=a.H, T=a.T, density_um2=60.0, photons=400.0, bg=40.0, blink=blink,
                      camera=CameraModel(gain_e_per_adu=1.0, offset_adu=100.0, readnoise_e=1.5), Q=Q, grid_os=os_,
                      chunk=1000, seed=3)
mv = DenseMovie(psf, spec)
print(f'{mv.n} emitters ({spec.density_um2}/um^2), {a.H}x{a.H} px, T={a.T}')
st = sofi.trace_stats(blink, n=2000, T=4000)
acc = CumulantAccumulator(a.H, a.H, R, block=500)
t0 = time.time()
for t, adu in mv.frames():
    acc.update(mv.to_photons(adu))
acc.finish()
print(f'simulate + accumulate {time.time() - t0:.1f}s, {len(acc.blocks)} blocks')
res = acc.result()

gp = GridPSF(psf, Q, os_, (0.0,))
h = gp.h().double()
K = kernels(h, acc.D)
rho = rho_from_emitters(mv.x.double(), mv.y.double(), mv.z.double(), (mv.eps.double() ** 2) * st['var'], a.H, a.H, Q,
                        os_, (0.0,))
Gm = model_G(rho, K)
# block-mean bias of the sample covariance: E[G_block] = G (1 - (beta - 1) / (n - 1)) approximately
bias = 1 - (st["beta"] - 1) / (acc.block - 1)
Gh = res['G']
mk = res['mask']
c = (Q - 1) // 2
inner = torch.zeros_like(mk)
inner[:, c:-c, c:-c] = True
sel = mk & inner
rel = ((Gh - Gm * bias)[sel]).norm() / (Gm[sel]).norm()
slope = float((Gh[sel] * Gm[sel]).sum() / (Gm[sel] ** 2).sum())
print(f'G: relative residual {rel:.4f}, fitted scale {slope:.4f} (block-mean bias factor {bias:.4f})')
per_d = [float((Gh[k][sel[k]] * Gm[k][sel[k]]).sum() / (Gm[k][sel[k]] ** 2).sum()) for k in range(len(acc.D))]
print('per-separation scale: min %.4f max %.4f' % (min(per_d), max(per_d)))

# explicit patch matrices vs convolution model on a P x P sub-patch
P = 12
y0 = x0 = 10
sub = rho[:, y0:y0 + P + Q - 1, x0:x0 + P + Q - 1]
Hm = patch_matrix(h, P)
S = (Hm * sub.reshape(-1)[None]) @ Hm.T
I, J, Kd, D = pairs(P, R)
g_mat = S[I, J]
g_conv = Gm[:, y0:y0 + P, x0:x0 + P][Kd, I // P, I % P]
print(f'patch matrix vs conv model: max rel diff {float((g_mat - g_conv).abs().max() / g_conv.abs().max()):.2e}')
save_json(dict(T=a.T, n=mv.n, rel_residual=float(rel), scale=slope, bias_factor=bias, per_d_scale=per_d,
               patch_vs_conv=float((g_mat - g_conv).abs().max() / g_conv.abs().max())), 'validate_accumulator.json')
