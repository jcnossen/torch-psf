"""Which part of the detection step tilts the free pupil fit (section 3.4)? Scenario A, free pupil after Zernike.

 a  movie, single-emitter patches, cropped at round(TRUE position)      selection yes, centring no
 b  movie, single-emitter patches, cropped at the DoG peak              selection yes, centring yes   (= 3.4)
 c  isolated emitters in 64 px frames, cropped at round(true position)  selection no,  centring no
 d  the same frames, cropped at the DoG maximum of the frame            selection no,  centring yes
c and d are rendered like the movie (23 px PSF support, z in 0..0.8).
"""
import sys, math, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
scn = sys.argv[1] if len(sys.argv) > 1 else 'A'
which = sys.argv[2] if len(sys.argv) > 2 else 'abcd'
P, h = C.P, (C.P - 1) // 2
spl = F.new_spline()
Ktrue = spl.knots_from_vectorial(C.make_vec(spl.Pl, scn, 'insitu'), C.insitu_pos)
test = C.sim_test(scn, 6000)
spl.knots = Ktrue; _, nll_true = localize(spl, test['d'])
zc_s = F.fit_vec_stack(C.sim_stack(scn))
ii = torch.arange(P, device=C.DEV)


def crop(frames, t, x0, y0):
    yy = (y0[:, None, None] + ii[None, :, None]).expand(-1, P, P)
    xx = (x0[:, None, None] + ii[None, None, :]).expand(-1, P, P)
    return frames[t[:, None, None], yy, xx].float()


def rec(name, d):
    zc, th = F.fit_vec_insitu(spl, d, zc_s, iters=60)
    out = []
    for nm, K in (('Zernike', F.knots_from(spl, zc)), ('pupil', F.fit_pupil_insitu(spl, d, zc, iters=200, th=th)[0])):
        r = F.evaluate(spl, K, test, nll_true)
        e = r['th'] - test['th']; z = test['th'][:, 2]; good = e[:, 2].abs() < 0.15
        b = torch.linspace(0.05, 0.75, 8, device=C.DEV); ib = torch.bucketize(z, b).clamp(1, 7) - 1
        cy = [float(e[good & (ib == i), 1].mean()) * 108 for i in range(7)]
        cx = [float(e[good & (ib == i), 0].mean()) * 108 for i in range(7)]
        cz = [float(e[good & (ib == i), 2].mean()) * 1000 for i in range(7)]
        print(f"{scn} {name:44s} n {len(d):6d} {nm:8s} dnll {r['dnll']:6.2f}  x {min(cx):6.1f}..{max(cx):5.1f}  "
              f"y {min(cy):6.1f}..{max(cy):5.1f}  z {min(cz):6.1f}..{max(cz):5.1f} nm", flush=True)
    return K


if 'a' in which or 'b' in which:
    movie, truth = C.sim_movie(scn, T=6000)
    rois, fr, cor = C.detect_crop(movie)
    cx, cy = cor[:, 0].float() + h, cor[:, 1].float() + h
    tf = truth[:, 0].long()
    order = torch.argsort(tf); tf, tx, ty = tf[order], truth[order, 1], truth[order, 2]
    lo = torch.searchsorted(tf, fr); hi = torch.searchsorted(tf, fr, right=True)
    kmax = int((hi - lo).max())
    idx = (lo[:, None] + torch.arange(kmax, device=C.DEV)[None]).clamp(max=len(tf) - 1)
    valid = torch.arange(kmax, device=C.DEV)[None] < (hi - lo)[:, None]
    dist = torch.maximum((tx[idx] - cx[:, None]).abs(), (ty[idx] - cy[:, None]).abs())
    dist = torch.where(valid, dist, torch.full_like(dist, 1e9))
    n14 = (dist < 14).sum(1)
    dmin, j = dist.min(1)
    single = (n14 == 1) & (dmin < 2)
    k = idx.gather(1, j[:, None])[:, 0]
    X, Y = tx[k][single], ty[k][single]
    x0, y0 = torch.round(X).long() - h, torch.round(Y).long() - h
    inb = (x0 >= 0) & (y0 >= 0) & (x0 + P <= movie.shape[2]) & (y0 + P <= movie.shape[1])
    off = torch.stack([cx[single] - X, cy[single] - Y], 1) * 108
    print(f'{scn}: detected {len(rois)}, single {int(single.sum())}; DoG peak minus true position: mean x {off[:,0].mean():.1f} '
          f'y {off[:,1].mean():.1f} nm, rms x {off[:,0].std():.1f} y {off[:,1].std():.1f} nm', flush=True)
    if 'a' in which:
        mv = movie.to(C.DEV).float()
        rec('a movie single, crop at true position', crop(mv, fr[single][inb], x0[inb], y0[inb]))
        del mv
    if 'b' in which:
        rec('b movie single, crop at DoG peak', rois[single])
    if 'e' in which:
        # fix: re-centre every detected patch (all of them, no ground truth) on the position fitted with the current
        # PSF model, so that the offset inside the patch no longer depends on the PSF shape; two rounds.
        mv = movie.to(C.DEV).float()
        d, x0, y0, t = rois, cor[:, 0].clone(), cor[:, 1].clone(), fr
        zc, th = F.fit_vec_insitu(spl, d, zc_s, iters=60)
        K = F.knots_from(spl, zc)
        for rnd in range(2):
            spl.knots = K
            th, _ = localize(spl, d)
            x0, y0 = x0 + torch.round(th[:, 0]).long(), y0 + torch.round(th[:, 1]).long()
            inb = (x0 >= 0) & (y0 >= 0) & (x0 + P <= mv.shape[2]) & (y0 + P <= mv.shape[1])
            x0, y0, t = x0[inb], y0[inb], t[inb]
            d = crop(mv, t, x0, y0)
            print(f're-centre round {rnd}: moved {float((torch.round(th[:, :2]).abs().sum(1) > 0).float().mean()):.3f} of patches', flush=True)
            K = rec(f'e movie all, re-centred on fit, round {rnd}', d)
        del mv
    del movie

if 'c' in which or 'd' in which:
    N, S, PR = 11000, 64, C.PR
    g = torch.Generator(device=C.DEV).manual_seed(900)
    r = lambda *s: torch.rand(*s, generator=g, device=C.DEV)
    psf = C.make_vec(PR, scn, 'insitu')
    X, Y, Z = 24 + 16 * r(N), 24 + 16 * r(N), 0.8 * r(N)
    phot = torch.exp(math.log(4000.0) + 0.4 * torch.randn(N, generator=g, device=C.DEV))
    hr = (PR - 1) // 2
    ox, oy = torch.round(X).long() - hr, torch.round(Y).long() - hr
    pat = phot[:, None, None] * C.render(psf, torch.stack([X - torch.round(X), Y - torch.round(Y), Z], 1), torch.full((N,), C.F0, device=C.DEV))
    fr_ = torch.full((N, S, S), 30.0, device=C.DEV)
    jj = torch.arange(PR, device=C.DEV)
    n_ = torch.arange(N, device=C.DEV)
    fr_[n_[:, None, None], (oy[:, None, None] + jj[None, :, None]).expand(-1, PR, PR), (ox[:, None, None] + jj[None, None, :]).expand(-1, PR, PR)] += pat
    fr_ = torch.poisson(fr_, generator=g)
    if 'c' in which:
        rec('c isolated, crop at true position', crop(fr_, n_, torch.round(X).long() - h, torch.round(Y).long() - h))
    if 'd' in which:
        def gk(s):
            x = torch.arange(-int(4 * s), int(4 * s) + 1, device=C.DEV, dtype=torch.float32)
            k = torch.exp(-x ** 2 / (2 * s * s)); return k / k.sum()

        def blur(x, s):
            k = gk(s)
            x = torch.nn.functional.conv2d(x[:, None], k[None, None, :, None], padding=(len(k) // 2, 0))
            return torch.nn.functional.conv2d(x, k[None, None, None, :], padding=(0, len(k) // 2))[:, 0]
        dog = torch.cat([blur(fr_[s:s + 1000], 2.0) - blur(fr_[s:s + 1000], 6.0) for s in range(0, N, 1000)])
        am = dog[:, 12:52, 12:52].flatten(1).argmax(1)
        py, px = am // 40 + 12, am % 40 + 12
        off = torch.stack([px - X, py - Y], 1) * 108
        zb = (Z / 0.2).long().clamp(max=3)
        print('d: DoG peak minus true position, mean y per z quarter (nm):', [round(float(off[zb == q, 1].mean()), 1) for q in range(4)],
              ' x:', [round(float(off[zb == q, 0].mean()), 1) for q in range(4)], flush=True)
        rec('d isolated, crop at DoG peak', crop(fr_, n_, px - h, py - h))
