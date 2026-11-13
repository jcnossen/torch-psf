"""Compare our localizations (x, y in camera px, z from the bead calibration) with a participant's result file of the
challenge (frame, x nm, y nm, z nm, photons; drift corrected by the organizers) to find the pixel size and the z scale
that the participant used. Step 1: coarse scale, axis orientation and offset from the cross-correlation of renders.
Step 2: per-frame nearest-neighbour matching and least squares x_ref = s x + b(t) (offsets per time window, they absorb
the drift correction), same for y, and z_ref = a z + c(t).
usage: python compare_ref.py ours.csv ref.csv out.json [--frame-offset 0] [--zunit-ours nm]"""
import sys, json, argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve

ap = argparse.ArgumentParser()
ap.add_argument('ours'); ap.add_argument('ref'); ap.add_argument('out')
ap.add_argument('--frame-offset', type=int, default=0)     # ref frame = our frame + offset
ap.add_argument('--win', type=int, default=5000)
ap.add_argument('--smin', type=float, default=80); ap.add_argument('--smax', type=float, default=200)
ap.add_argument('--bin', type=float, default=0.25)          # px, coarse render bin
ap.add_argument('--bin-nm', type=float, default=0)
ap.add_argument('--coarse-all', action='store_true')       # coarse step on all frames (frame numbering not trusted)
A = ap.parse_args()
O = np.loadtxt(A.ours, delimiter=',', skiprows=1)
cols = open(A.ours).readline().strip().split(',')
O = {c: O[:, i] for i, c in enumerate(cols)}
Rr = np.loadtxt(A.ref, delimiter=',', usecols=(0, 1, 2, 3, 4))
fr_r, xr, yr, zr, pr = Rr.T
fmax = min(O['frame'].max(), fr_r.max() - A.frame_offset)
ok_o = (O['flags'] == 0) & (O['frame'] <= fmax)
fo, xo, yo, zo = O['frame'][ok_o], O['x'][ok_o], O['y'][ok_o], O['z'][ok_o]
ok_r = (fr_r - A.frame_offset >= 0) & (fr_r - A.frame_offset <= fmax)
fr, xr, yr, zr = fr_r[ok_r] - A.frame_offset, xr[ok_r], yr[ok_r], zr[ok_r]
print(f'ours {len(fo)} (flags 0), ref {len(fr)} in the common frames 0..{int(fmax)}')
res = dict(n_ours=int(len(fo)), n_ref=int(len(fr)))
# ---- step 1: coarse search over scale and the 8 axis orientations (render cross-correlation, first 20000 frames)
sel_o = fo < 20000; sel_r = fr < 20000
if A.coarse_all:
    sel_o = np.ones(len(fo), bool); sel_r = np.ones(len(fr), bool)
best = None
for s in np.arange(A.smin, A.smax, 2.0):
    bins = A.bin_nm if A.bin_nm > 0 else A.bin * s     # render bin in nm (a fixed nm bin avoids a bias towards large s)
    for swap in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                X, Y = (yo, xo) if swap else (xo, yo)
                X, Y = sx * X * s, sy * Y * s
                ex = np.arange(min(X.min(), xr.min()) - bins, max(X.max(), xr.max()) + 2 * bins, bins)
                ey = np.arange(min(Y.min(), yr.min()) - bins, max(Y.max(), yr.max()) + 2 * bins, bins)
                if len(ex) * len(ey) > 6e6:
                    continue
                Io, _, _ = np.histogram2d(Y[sel_o], X[sel_o], [ey, ex])
                Ir, _, _ = np.histogram2d(yr[sel_r], xr[sel_r], [ey, ex])
                cc = fftconvolve(Ir, Io[::-1, ::-1], mode='same')
                k = np.unravel_index(np.argmax(cc), cc.shape)
                sc = cc[k] / np.sqrt((Io ** 2).sum() * (Ir ** 2).sum())
                if best is None or sc > best[0]:
                    best = (sc, s, swap, sx, sy, k, cc.shape, bins)
sc, s, swap, sx, sy, k, shp, bins = best
print(f'coarse: scale {s} nm/px, swap {swap}, flips ({sx}, {sy}), normalised cc peak {sc:.3f}')
X, Y = (yo, xo) if swap else (xo, yo)
X, Y = sx * X * s, sy * Y * s
# offset from the cc peak position (mode 'same': centre index = shape // 2)
dy_, dx_ = (k[0] - shp[0] // 2) * bins, (k[1] - shp[1] // 2) * bins
# ---- step 2: matching per frame, iterate fit
Xo_px, Yo_px = ((yo, xo) if swap else (xo, yo))
Xo_px, Yo_px = sx * Xo_px, sy * Yo_px
bx, by = dx_, dy_
for it in range(4):
    tx, ty = s * Xo_px + bx, s * Yo_px + by
    # match: same frame, within r nm
    r = 300 if it == 0 else 100
    tree = cKDTree(np.stack([tx, ty, fo * 1e6], 1))
    dist, idx = tree.query(np.stack([xr, yr, fr * 1e6], 1), distance_upper_bound=r)
    m = np.isfinite(dist)
    ir, io = np.nonzero(m)[0], idx[m]
    # least squares: x_ref = s X + b_w (per window); y likewise with the same s
    w = (fr[ir] // A.win).astype(int)
    uw, wi = np.unique(w, return_inverse=True)
    nW = len(uw)
    Mx = np.zeros((2 * len(ir), 1 + 2 * nW))
    Mx[:len(ir), 0] = Xo_px[io]; Mx[len(ir):, 0] = Yo_px[io]
    Mx[np.arange(len(ir)), 1 + wi] = 1
    Mx[len(ir) + np.arange(len(ir)), 1 + nW + wi] = 1
    rhs = np.concatenate([xr[ir], yr[ir]])
    sol, *_ = np.linalg.lstsq(Mx, rhs, rcond=None)
    resid = rhs - Mx @ sol
    s = sol[0]
    bxw, byw = sol[1:1 + nW], sol[1 + nW:]
    bx, by = np.zeros(len(fo)), np.zeros(len(fo))
    wo = np.clip(np.searchsorted(uw, (fo // A.win).astype(int)), 0, nW - 1)
    bx, by = bxw[wo], byw[wo]
    print(f'iter {it}: matched {len(ir)} of {len(fr)} ref locs; scale {s:.3f} nm/px; residual rms {np.sqrt(np.mean(resid ** 2)):.1f} nm, '
          f'robust {1.4826 * np.median(np.abs(resid)):.1f} nm')
# full affine model (rotation, shear, anisotropic scale) with per-window offsets, and a scan of a frame offset
Ma = np.zeros((2 * len(ir), 4 + 2 * nW))
Ma[:len(ir), 0] = Xo_px[io]; Ma[:len(ir), 1] = Yo_px[io]
Ma[len(ir):, 2] = Xo_px[io]; Ma[len(ir):, 3] = Yo_px[io]
Ma[np.arange(len(ir)), 4 + wi] = 1
Ma[len(ir) + np.arange(len(ir)), 4 + nW + wi] = 1
sa, *_ = np.linalg.lstsq(Ma, rhs, rcond=None)
ra = rhs - Ma @ sa
Aff = sa[:4].reshape(2, 2)
sv = np.linalg.svd(Aff, compute_uv=False)
res['affine'] = dict(matrix_nm_per_px=Aff.tolist(), singular_values=sv.tolist(),
                     rotation_deg=float(np.degrees(np.arctan2(Aff[1, 0] - Aff[0, 1], Aff[0, 0] + Aff[1, 1]))),
                     resid_robust_nm=float(1.4826 * np.median(np.abs(ra))))
print('affine:', res['affine'], flush=True)
tx, ty = s * Xo_px + bx, s * Yo_px + by
fo_cnt = {}
for fofs in (-2, -1, 0, 1, 2):
    tree = cKDTree(np.stack([tx, ty, (fo + fofs) * 1e6], 1))
    dist, idx = tree.query(np.stack([xr, yr, fr * 1e6], 1), distance_upper_bound=100)
    fo_cnt[fofs] = int(np.isfinite(dist).sum())
res['matches_vs_frame_offset'] = fo_cnt
print('matches (<100 nm) vs frame offset (ref frame = our frame + k):', fo_cnt, flush=True)
# separate scales for x and y as a check
for ax, (P_, R_) in (('x', (Xo_px, xr)), ('y', (Yo_px, yr))):
    M = np.zeros((len(ir), 1 + nW)); M[:, 0] = P_[io]; M[np.arange(len(ir)), 1 + wi] = 1
    so, *_ = np.linalg.lstsq(M, R_[ir], rcond=None)
    res[f'scale_{ax}'] = float(so[0])
# z: z_ref = a z_ours + c_w
M = np.zeros((len(ir), 1 + nW)); M[:, 0] = zo[io]; M[np.arange(len(ir)), 1 + wi] = 1
good = np.abs(zo[io]) < 600
so, *_ = np.linalg.lstsq(M[good], zr[ir][good], rcond=None)
rz = zr[ir][good] - M[good] @ so
# robust: repeat after dropping > 3 robust sigma
keep = np.abs(rz) < 3 * 1.4826 * np.median(np.abs(rz))
so2, *_ = np.linalg.lstsq(M[good][keep], zr[ir][good][keep], rcond=None)
res.update(scale_nm_per_px=float(s), swap_xy=bool(swap), flip=[int(sx), int(sy)], n_matched=int(len(ir)),
           xy_resid_robust_nm=float(1.4826 * np.median(np.abs(resid))), z_slope=float(so2[0]),
           z_slope_first_pass=float(so[0]), z_offset_median=float(np.median(so2[1:])),
           z_resid_robust=float(1.4826 * np.median(np.abs(rz[keep]))))
# z slope in bins of our z (linearity)
zb = np.arange(-600, 601, 200)
res['z_ref_median_in_our_z_bins'] = [[float(a), float(np.median((zr[ir] - so2[1:][wi])[(zo[io] >= a) & (zo[io] < a + 200)]))
                                      if ((zo[io] >= a) & (zo[io] < a + 200)).sum() > 20 else None] for a in zb[:-1]]
print(json.dumps(res, indent=1))
json.dump(res, open(A.out, 'w'), indent=1)
