"""Critic test 2: does the vectorial model need the expensive (pupil 128/256, oversampling 4/6) settings of
bench_speed2 at all? Relative L2 difference of 15 px patches against a 256 / 7 reference, scenario A and B pupils.
Run from the repo root:  python study/insitu/critic/vec_sampling.py"""
import sys, torch
sys.path.insert(0, 'study/insitu')
import common as C

torch.manual_seed(0)
n = 200
pos = torch.cat([torch.rand(n, 2) - 0.5, torch.rand(n, 1) * 0.8], 1).to(C.DEV)
f = torch.full((n,), C.F0, device=C.DEV)


def model(scn, npupil, os):
    C.OPT.update(npupil=npupil, oversample=os)
    return C.make_vec(C.P, scn, 'insitu')       # scenario A only: B's maps are defined on the 64 grid


for scn in 'A':
    out = {}
    for npupil, os in ((256, 7), (64, 3), (64, 5), (128, 3), (128, 4), (256, 6), (32, 3)):
        psf = model(scn, npupil, os)
        out[(npupil, os)] = C.render(psf, pos, f, nb=50 if npupil >= 256 else 200)
    ref = out[(256, 7)]
    for k, v in out.items():
        r = ((v - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1))
        mu = 4000 * ref + 30
        chi = ((4000 * (v - ref)) ** 2 / mu).flatten(1).sum(1) / 2
        print(f'scenario {scn} pupil {k[0]:3d} oversampling {k[1]}: rel L2 vs 256/7 median {r.median():.2e} max {r.max():.2e}; '
              f'nll cost @4000 ph median {chi.median():.4f}')
C.OPT.update(npupil=64, oversample=3)
