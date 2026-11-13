import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = 'study/insitu/out/'; FIG = 'docs/fig/'
import os; os.makedirs(FIG, exist_ok=True)
J = json.load(open(OUT + 'results.json')); G = np.load(OUT + 'gallery.npz'); S = json.load(open(OUT + 'speed.json')); S2 = json.load(open(OUT + 'speed2.json'))
# data
f, ax = plt.subplots(1, 6, figsize=(13, 2.4), gridspec_kw=dict(width_ratios=[2.2, 1, 1, 1, 1, 1]))
ax[0].imshow(G['frame'], cmap='gray'); ax[0].set_title('one SMLM frame (128 x 128 px)', fontsize=9)
for i in range(5):
    ax[i + 1].imshow(G['stack'][i], cmap='gray'); ax[i + 1].set_title(f'bead, nominal z {(-0.1 + 0.2 * i) * 1000:.0f} nm', fontsize=9)
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(FIG + 'data.png', dpi=130); plt.close()
# speed
f, ax = plt.subplots(1, 3, figsize=(13, 3.6))
N = np.array(S['N']); tK = np.array(S['tK_knot']) + np.array(S['tK_vjp']) + np.array(S['tK_build']); tV = np.array(S['tV'])
ax[0].loglog(N, tV, 'o-', label='vectorial model per emitter'); ax[0].loglog(N, tK, 's-', label='via spline knots (total)')
ax[0].loglog(N, np.array(S['tK_knot']), ':', color='C1', label='  of which knot gradient')
ax[0].set_xlabel('emitters'); ax[0].set_ylabel('seconds per Zernike gradient'); ax[0].legend(fontsize=8); ax[0].set_title('cost of one gradient (pupil 64, 15 px patch)', fontsize=9)
Ns = np.logspace(2.5, 6, 50)
for r in S2:
    ax[1].semilogx(Ns, Ns * r['us_per_emitter_V'] * 1e-6 / (Ns * r['us_per_emitter_K'] * 1e-6 + r['fixed_K_s']), label=f"pupil {r['npupil']}, oversampling {r['oversample']}")
ax[1].axhline(1, color='k', lw=0.5); ax[1].set_xlabel('emitters'); ax[1].set_ylabel('speed-up of the gradient'); ax[1].legend(fontsize=8); ax[1].set_title('speed-up vs cost of the vectorial model', fontsize=9)
for k, lab in (('V', 'vectorial model per emitter'), ('K', 'via spline knots')):
    L = np.array(S['trace'][k]); ax[2].semilogy(L[:, 1], L[:, 3], label=lab)
ax[2].set_xlabel('wall time (s)'); ax[2].set_ylabel('rms Zernike error (waves)'); ax[2].legend(fontsize=8); ax[2].set_title(f"same fit, {S['N_fit']} emitters, laptop RTX 4060", fontsize=9)
plt.tight_layout(); plt.savefig(FIG + 'speed.png', dpi=130); plt.close()
# bias curves
f, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
for r, scn in enumerate('AB'):
    for name in ('true PSF', 'spline, beads + SMLM', 'Zernike, beads -> SMLM', 'Zernike + free knots'):
        c = J['curves'][f'{scn}/{name}']; z = np.array(c['z']) * 1000
        for k, (key, sc, lab) in enumerate((('x', 108, 'x'), ('y', 108, 'y'), ('zz', 1000, 'z'))):
            ax[r, k].plot(z, np.array(c[key]) * sc, label=name); ax[r, k].set_title(f'scenario {scn}: mean {lab} error (nm)', fontsize=9)
    ax[r, 0].legend(fontsize=7)
for a in ax[1]: a.set_xlabel('true emitter z (nm)')
plt.tight_layout(); plt.savefig(FIG + 'bias.png', dpi=130); plt.close()
for scn in 'AB':
    for n in ('Zernike, beads -> SMLM', 'Zernike + free knots', 'true PSF'):
        c = J['curves'][f'{scn}/{n}']; print(scn, n, 'y bias range nm', round(min(c['y']) * 108, 1), round(max(c['y']) * 108, 1), 'x', round(min(c['x']) * 108, 1), round(max(c['x']) * 108, 1), 'z', round(min(c['zz']) * 1000, 1), round(max(c['zz']) * 1000, 1))
