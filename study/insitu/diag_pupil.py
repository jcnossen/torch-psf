"""Diagnostics of the free pupil map fit on scenario B: recovered pupil against truth, failures against z."""
import sys, torch
sys.path.insert(0, 'study/insitu')
import common as C, fit as F
from torch_psf.fit import localize
from torch_psf.zernike import zernike
scn = 'B'
spl = F.new_spline()
tv = C.make_vec(spl.Pl, scn, 'insitu')
Ktrue = spl.sample(tv, C.insitu_pos)
test = C.sim_test(scn, 6000)
prm = torch.load('study/insitu/out/pupil_B.pt', weights_only=False)
big = C.model_vec(spl.Pl, prm['zcoef'])
mask = big.mask.bool()
with torch.no_grad():
    big.pupil_phase.copy_(prm['phase'].to(C.DEV)); big.pupil_amp.copy_(torch.exp(prm['la'].to(C.DEV)))
K = spl.sample(big, C.insitu_pos)
ph_t = (torch.einsum('j,jnm->nm', tv.zcoef, tv.Z) + tv.pupil_phase)[mask]; ph_e = (torch.einsum('j,jnm->nm', big.zcoef, big.Z) + big.pupil_phase)[mask]
low = torch.stack([zernike(j, big.rho.clamp(max=1), big.phi)[mask] for j in (1, 2, 3, 4)], 1); Q, _ = torch.linalg.qr(low)
d = ph_e - ph_t; co = torch.linalg.lstsq(low, d[:, None]).solution[:, 0]; d2 = d - Q @ (Q.T @ d)
print('true phase rms', (ph_t - Q @ (Q.T @ ph_t)).std().item(), 'phase diff rms (waves)', d.std().item(), 'after removing piston/tilt/defocus', d2.std().item(), 'tilt x,y defocus coef', co[1:].tolist())
la_t = torch.log(tv.pupil_amp[mask]); la_e = torch.log(big.pupil_amp[mask])
print('log-amp corr', torch.corrcoef(torch.stack([la_t - la_t.mean(), la_e - la_e.mean()]))[0, 1].item(), 'true std', la_t.std().item(), 'est std', la_e.std().item())
for name, KK in (('true', Ktrue), ('pupil', K)):
    spl.knots = KK; th, nll = localize(spl, test['d'])
    e = th - test['th']; z = test['th'][:, 2]
    bad_z = e[:, 2].abs() > 0.15; bad_xy = e[:, :2].abs().max(1).values > 0.5
    print(name, 'bad z', bad_z.float().mean().item(), 'bad xy', bad_xy.float().mean().item())
    for lo in (0.05, 0.2, 0.35, 0.5, 0.65):
        m = (z >= lo) & (z < lo + 0.15)
        print(f'   z {lo:.2f}-{lo+0.15:.2f}: bad z {bad_z[m].float().mean():.3f}  median z err {e[m, 2].median()*1000:.1f} nm  y err median {e[m,1].median()*108:.1f} nm  z err of bad: {e[m & bad_z, 2].median()*1000:.0f}')
