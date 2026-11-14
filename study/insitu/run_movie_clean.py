"""What in the movie patches causes the y-vs-z tilt of the free pupil fit (section 3.3)? Scenario A.
The detected patches are split with the ground truth into 'single' (exactly one emitter within 14 px of the patch
centre in that frame) and 'all'. The free pupil is fitted on each set."""
import sys, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf import localize
scn = sys.argv[1] if len(sys.argv) > 1 else 'A'
spl = F.new_spline()
Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
test = C.sim_test(scn, 6000)
spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
movie, truth = C.sim_movie(scn, T=6000)
rois, fr, cor = C.detect_crop(movie)
cx, cy = cor[:, 0].float() + 7, cor[:, 1].float() + 7
# count true emitters near each patch, frame by frame (sorted truth, searchsorted ranges; padded neighbour lists)
tf = truth[:, 0].long()
order = torch.argsort(tf); tf, tx, ty = tf[order], truth[order, 1], truth[order, 2]
lo = torch.searchsorted(tf, fr); hi = torch.searchsorted(tf, fr, right=True)
kmax = int((hi - lo).max())
idx = (lo[:, None] + torch.arange(kmax, device=C.DEV)[None]).clamp(max=len(tf) - 1)
valid = torch.arange(kmax, device=C.DEV)[None] < (hi - lo)[:, None]
dist = torch.maximum((tx[idx] - cx[:, None]).abs(), (ty[idx] - cy[:, None]).abs())
n14 = ((dist < 14) & valid).sum(1)
dmin = torch.where(valid, dist, torch.full_like(dist, 1e9)).min(1).values
single = (n14 == 1) & (dmin < 2)
print(f'{scn}: patches {len(rois)}, single emitter {int(single.sum())}, second emitter within 14 px {int((n14 > 1).sum())}, no emitter within 2 px {int((dmin >= 2).sum())}', flush=True)
zc_s = F.fit_vec_stack(C.sim_stack(scn))


def rec(name, K):
    r = F.evaluate(spl, K, test, nll_true)
    e = r['th'] - test['th']; z = test['th'][:, 2]; good = e[:, 2].abs() < 0.15
    b = torch.linspace(0.05, 0.75, 8, device=C.DEV); ib = torch.bucketize(z, b).clamp(1, 7) - 1
    cy_ = [float(e[good & (ib == i), 1].mean()) * 108 for i in range(7)]; cz = [float(e[good & (ib == i), 2].mean()) * 1000 for i in range(7)]
    print(f"movie {scn} {name:22s} dnll {r['dnll']:6.2f}  mean err over z: y {min(cy_):6.1f}..{max(cy_):5.1f}  z {min(cz):6.1f}..{max(cz):5.1f} nm", flush=True)


for name, sel in (('all patches', torch.ones_like(single)), ('single-emitter patches', single)):
    d = rois[sel]
    zc, th = F.fit_vec_insitu(spl, d, zc_s, iters=60); rec(name + ' Zernike'[:0] + ' / Zernike', F.knots_from(spl, zc))
    Kp, _, _ = F.fit_pupil_insitu(spl, d, zc, iters=200, th=th); rec(name + ' / pupil', Kp)
