"""Critic test 3: what is different about the detected movie patches (scenario A) compared with the clean patches?
Matches every detected patch to the true emitters of its frame and reports: distinct sites, z distribution,
offset of the true emitter from the patch centre against z, photons against z, clamping at the knot range.
Saves the matched parameters for movie_fits.py.
Run from the repo root:  python study/insitu/critic/movie_diag.py OUTFILE.pt"""
import sys, torch
sys.path.insert(0, 'study/insitu')
import common as C

out = sys.argv[1]
scn = 'A'
movie, truth = C.sim_movie(scn, T=6000)
rois, fr, cor = C.detect_crop(movie)
h = (C.P - 1) // 2
cx, cy = cor[:, 0].float() + h, cor[:, 1].float() + h
tf = truth[:, 0].long()
order = torch.argsort(tf); tf, tr = tf[order], truth[order]
lo = torch.searchsorted(tf, fr); hi = torch.searchsorted(tf, fr, right=True)
kmax = int((hi - lo).max())
idx = (lo[:, None] + torch.arange(kmax, device=C.DEV)[None]).clamp(max=len(tf) - 1)
valid = torch.arange(kmax, device=C.DEV)[None] < (hi - lo)[:, None]
dist = torch.maximum((tr[idx, 1] - cx[:, None]).abs(), (tr[idx, 2] - cy[:, None]).abs())
dist = torch.where(valid, dist, torch.full_like(dist, 1e9))
j = dist.argmin(1)
m = tr[idx[torch.arange(len(j)), j]]                      # nearest true emitter: frame, X, Y, z, photons, site
n14 = (dist < 14).sum(1)
dx, dy = m[:, 1] - cx, m[:, 2] - cy
single = (n14 == 1) & (dist.min(1).values < 2)
print(f'patches {len(rois)}, on-events {len(truth)}, distinct sites among patches {len(torch.unique(m[:, 5]))} of 700, '
      f'single {int(single.sum())}')
print(f'true offset from patch centre: |dx| max {dx.abs().max():.2f} px, |dy| max {dy.abs().max():.2f} px, '
      f'fraction beyond the knot clamp 1.45 px: {((dx.abs() > 1.45) | (dy.abs() > 1.45)).float().mean():.4f}')
allz = truth[:, 3]
print('z bin     events  detected  frac   mean dx   mean dy   sd dy   median photons (detected / all)')
for a in torch.arange(0, 0.8, 0.1):
    e = (allz >= a) & (allz < a + 0.1); s = (m[:, 3] >= a) & (m[:, 3] < a + 0.1)
    print(f'{a:.1f}-{a + 0.1:.1f}  {int(e.sum()):7d}  {int(s.sum()):7d}  {s.sum() / e.sum():.2f}  {dx[s].mean():+.3f}  {dy[s].mean():+.3f}  '
          f'{dy[s].std():.3f}  {m[s, 4].median():.0f} / {truth[e, 4].median():.0f}')
# repetitions per site
_, cnt = torch.unique(m[:, 5], return_counts=True)
print(f'detections per site: median {cnt.float().median():.0f}, max {int(cnt.max())}')
torch.save(dict(rois=rois.cpu(), off=torch.stack([dx, dy], 1).cpu(), z=m[:, 3].cpu(), phot=m[:, 4].cpu(),
                site=m[:, 5].cpu(), single=single.cpu()), out)
