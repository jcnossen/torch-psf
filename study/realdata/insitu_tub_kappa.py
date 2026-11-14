"""Predicted axial scale factor kappa = d(depth) / d(z_fit) from the vectorial models (GPU, small).

Every fitted PSF here is a 'bead geometry' PSF: knot z = stage (focus) position for an emitter at a fixed height D
above the coverslip. A sample emitter at depth d in water with the focus fixed gives a different PSF. For a vectorial
physics model P (state of fit b, c or d), noise-free 17 x 17 px images of emitters at depths d = 0 ... 1.5 um
(focus fixed at several positions) are localized with the spline of every method A (same localizer as the test set);
z_A(d) gives kappa_A,P(z) = dd / dz_A, the factor that converts method A's z into height under physics P (it contains
the depth-dependent index-mismatch aberration, which the bead-geometry spline does not). Compare with the
microtubule ruler (insitu_tub_mt.py), which measures the same factor from the round cross-section.
usage: INSITU_DS=tubulin python insitu_tub_kappa.py out_tag phys1=tag1,... axes=label1=tag1,label2=tag2,...
-> insitu/kappa_<out_tag>.json, fig_kappa_<out_tag>.png"""
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'study/realdata')
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from insitu_common import *

otag = sys.argv[1]
phys = dict(kv.split('=') for kv in sys.argv[2].split(','))
axes = dict(kv.split('=') for kv in sys.argv[3].split(','))
spl, _ = movie_spline()
res = dict(phys=phys, axes=axes, curves={})
dgrid = np.arange(-0.2, 1.6001, 0.025)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for pn, ptag in phys.items():
    st = torch.load(f'{OUT}/psf_{ptag}.pt', weights_only=False)['vec_state']
    v = make_vec(P, st, npupil=64, oversample=3, use_blur=st.get('use_blur', False))
    D0 = float(st.get('D', st.get('bead_height', 0.05)))
    f0 = float(st.get('f0_D', st['f0'])); s = float(st['s']); kap = float(st.get('zscale', 1.0))
    for zc in (-0.4, 0.0, 0.4):
        F = f0 + s * kap * zc                         # focus at which an emitter at height D0 has knot z = zc
        dd = np.clip(D0 + dgrid, 0.0, None)
        dd = np.unique(dd)
        pos = torch.zeros(len(dd), 3, device=DEV); pos[:, 2] = torch.tensor(dd, device=DEV, dtype=torch.float32)
        with torch.no_grad():
            img = v(pos, torch.full((len(dd),), F, device=DEV))
        data = 20000.0 * img / img.flatten(1).sum(1)[:, None, None] + 20.0
        for an, atag in axes.items():
            spl.knots = torch.load(f'{OUT}/psf_{atag}.pt', weights_only=False)['knots'].to(DEV)
            th, _ = localize_full(spl, data, starts=(-0.6, -0.3, 0.0, 0.3, 0.6))
            z = th[:, 2].cpu().numpy()
            ok = np.abs(z) < ZLIM - 0.02
            key = f'{pn}|{an}|focus{zc:+.1f}'
            with np.errstate(divide='ignore'):
                dz = np.gradient(z, dd)
            res['curves'][key] = dict(depth=dd.tolist(), z=z.tolist(), ok=ok.tolist(), kappa=(1 / dz).tolist())
            if zc == 0.0:
                ax[0].plot(dd[ok], z[ok] * 1000, '-', label=f'{pn} imaged, {an} z')
            if zc == 0.0:
                ax[1].plot(dd[ok], z[ok] * 1000, '-', color=f'C{list(axes).index(an)}', lw=0.8,
                           label=f'{an}' if pn == list(phys)[0] else None)
# summary: kappa of each axis under each physics model in 200 nm bins of z_A: per focus curve a least-squares slope
# of depth against z_A over the points in the bin (>= 4, no wrong-branch jumps: |dz| < 150 nm between neighbours),
# median over the focus positions (pointwise gradients are noisy)
res['kappa_by_z'] = {}
for pn in phys:
    for an in axes:
        rows = []
        for lo in range(-600, 400, 200):
            ks = []
            for zc in (-0.4, 0.0, 0.4):
                c = res['curves'][f'{pn}|{an}|focus{zc:+.1f}']
                z, dep, ok = np.array(c['z']) * 1000, np.array(c['depth']) * 1000, np.array(c['ok'])
                jump = np.r_[False, np.abs(np.diff(z)) > 150] | np.r_[np.abs(np.diff(z)) > 150, False]
                q = ok & ~jump & (z >= lo) & (z < lo + 200)
                if q.sum() >= 4:
                    ks.append(abs(np.polyfit(z[q], dep[q], 1)[0]))
            if ks:
                rows.append(dict(z=[lo, lo + 200], kappa=float(np.median(ks)), kappa_range=[float(min(ks)), float(max(ks))], n_curves=len(ks)))
        res['kappa_by_z'][f'{pn}|{an}'] = rows
        print(f'physics {pn}, axis {an}: kappa by z ' + ' '.join(f'[{r["z"][0]},{r["z"][1]}) {r["kappa"]:.3f}' for r in rows), flush=True)
    # direction: does z decrease with depth?
    c = res['curves'][f'{pn}|{list(axes)[0]}|focus+0.0']
    z = np.array(c['z'])
    res[f'direction_{pn}'] = 'z decreases with depth' if np.nanmedian(np.gradient(z)) < 0 else 'z increases with depth'
    print(pn, res[f'direction_{pn}'])
ax[0].set_xlabel('emitter height above the coverslip (um)'); ax[0].set_ylabel('fitted z (nominal nm)'); ax[0].legend(fontsize=6)
ax[1].set_xlabel('emitter height (um)'); ax[1].set_ylabel('fitted z (nominal nm), all physics models'); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_kappa_{otag}.png', dpi=120); plt.close(fig)
jdump(res, f'{OUT}/kappa_{otag}.json')
