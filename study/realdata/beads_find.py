"""Step 1 of the bead calibration: find bead candidates in the z-stack, Gaussian widths per plane, select good beads.
usage: python beads_find.py stack.tif zpos.json outdir [--roi x0,y0,w,h] [--minamp A] [--zwin 0.6]
zpos.json: list of the stage positions (um) of the planes (from the MicroManager labels), or 'meta' to read them."""
import sys, json, functools
print = functools.partial(print, flush=True)
import numpy as np, tifffile
from scipy.ndimage import gaussian_filter, maximum_filter
sys.path.insert(0, '.')
from rdcommon import fit_gauss

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('stack'); ap.add_argument('zsrc'); ap.add_argument('outdir')
ap.add_argument('--roi', default='')                 # x0,y0,w,h of the movie on the camera chip
ap.add_argument('--minamp', type=float, default=0)   # drop faint candidates (DoG amplitude)
ap.add_argument('--zwin', type=float, default=0.6)   # half range of the width-curve comparison (um)
ap.add_argument('--PB', type=int, default=23)
a_ = ap.parse_args()
f, zsrc, outdir = a_.stack, a_.zsrc, a_.outdir
roi = [int(v) for v in a_.roi.split(',')] if a_.roi else None
PB = a_.PB
hb = PB // 2
with tifffile.TiffFile(f) as t:
    st = t.asarray().astype(np.float64)
    if zsrc == 'meta':
        L = t.imagej_metadata['Labels']
        zs = np.array([json.loads(l[l.index('{'):])['SlicePosition'] for l in L], float)
    else:
        zs = np.array(json.load(open(zsrc)), float)
nz, H, W = st.shape
print('stack', st.shape, 'z', zs[0], zs[-1], 'step', np.diff(zs).mean())
bgp = np.median(st.reshape(nz, -1), 1)
dog = np.stack([gaussian_filter(p, 1.5) - gaussian_filter(p, 6) for p in st])
proj = dog.max(0)
noise = 1.4826 * np.median(np.abs(proj - np.median(proj)))
mx = maximum_filter(proj, 9)
pk = (proj == mx) & (proj > max(15 * noise, a_.minamp))
ys, xs = np.nonzero(pk)
amp = proj[ys, xs]
print(f'background per plane median {np.median(bgp):.1f} ADU (range {bgp.min():.1f}-{bgp.max():.1f}); DoG noise {noise:.2f}; candidates {len(xs)}')
# isolation: no other candidate brighter than 5 % of this one within 20 px; border margin hb + 1
cands = []
for i in range(len(xs)):
    dd = np.hypot(xs - xs[i], ys - ys[i])
    near = (dd > 0) & (dd < 20) & (amp > 0.05 * amp[i])
    border = xs[i] < hb + 1 or ys[i] < hb + 1 or xs[i] >= W - hb - 1 or ys[i] >= H - hb - 1
    cands.append(dict(x=int(xs[i]), y=int(ys[i]), amp=float(amp[i]), isolated=bool(not near.any()), border=bool(border),
                      nnear=int(near.sum())))
keep = [c for c in cands if c['isolated'] and not c['border']]
print(f'isolated and not at the border: {len(keep)} of {len(cands)}')
# Gaussian widths per plane
rows = []
for c in keep:
    p = st[:, c['y'] - hb:c['y'] + hb + 1, c['x'] - hb:c['x'] + hb + 1]
    par = []
    for k in range(nz):
        q, rms = fit_gauss(p[k])
        par.append(list(q) + [rms])
    par = np.array(par)                     # x y sx sy I bg rms
    c['par'] = par.tolist()
    sx, sy = par[:, 2], par[:, 3]
    dif = sx - sy
    # crossing sx = sy nearest the plane of minimal sx + sy
    k0 = int(np.argmin(sx + sy))
    zc = np.nan
    nw = int(round(0.4 / abs(zs[1] - zs[0])))
    for k in range(max(k0 - nw, 0), min(k0 + nw, nz - 1)):
        if dif[k] * dif[k + 1] <= 0 and dif[k] != dif[k + 1]:
            zc = zs[k] + (zs[k + 1] - zs[k]) * dif[k] / (dif[k] - dif[k + 1])
            break
    c['zc'] = float(zc)
    c['kmin'] = k0
    c['Ifocus'] = float(par[k0, 4]); c['peak'] = float(p[k0].max() - par[k0, 5])
    rows.append(c)
zc = np.array([c['zc'] for c in rows])
ok = np.isfinite(zc)
# shape consistency: width curves aligned on zc, compared with the median curve over |z - zc| < 0.6 um
zg = np.arange(-a_.zwin, a_.zwin + 1e-4, 0.05)
curves = np.full((len(rows), len(zg), 2), np.nan)
for i, c in enumerate(rows):
    if not ok[i]:
        continue
    par = np.array(c['par'])
    curves[i, :, 0] = np.interp(zg, zs - c['zc'], par[:, 2], left=np.nan, right=np.nan)
    curves[i, :, 1] = np.interp(zg, zs - c['zc'], par[:, 3], left=np.nan, right=np.nan)
med = np.nanmedian(curves[ok], 0)
Imed = np.median([c['Ifocus'] for c in rows])
for i, c in enumerate(rows):
    c['shape_rms'] = float(np.sqrt(np.nanmean((curves[i] - med) ** 2))) if ok[i] else np.inf
    c['Irel'] = c['Ifocus'] / Imed
    c['in_roi'] = bool(roi is not None and roi[0] <= c['x'] < roi[0] + roi[2] and roi[1] <= c['y'] < roi[1] + roi[3])
    c['dist_roi'] = float(0 if roi is None else np.hypot(max(roi[0] - c['x'], 0, c['x'] - roi[0] - roi[2]),
                                                     max(roi[1] - c['y'], 0, c['y'] - roi[1] - roi[3])))
    c['good'] = bool(ok[i] and c['shape_rms'] < 0.35 and 0.3 < c['Irel'] < 2.5 and abs(c['zc']) < 0.5)
print(' x    y   amp   zc(um) kmin sx,sy@5   sx,sy@20  sx,sy@35  Irel  shape_rms  dist_roi good')
for c in rows:
    par = np.array(c['par'])
    print(f"{c['x']:3d} {c['y']:3d} {c['amp']:6.0f} {c['zc']:6.3f} {c['kmin']:3d}  {par[5,2]:.2f},{par[5,3]:.2f}  {par[20,2]:.2f},{par[20,3]:.2f}  "
          f"{par[35,2]:.2f},{par[35,3]:.2f}  {c['Irel']:.2f}  {c['shape_rms']:.3f}  {c['dist_roi']:5.0f} {int(c['good'])}")
good = [c for c in rows if c['good']]
print(f'good beads: {len(good)}; within ROI: {sum(c["in_roi"] for c in good)}; within 100 px of ROI: {sum(c["dist_roi"] < 100 for c in good)}')
print('median width curve (z rel. crossing, sx, sy):', [(round(z, 2), round(a, 2), round(b, 2)) for z, (a, b) in zip(zg[::2], med[::2])])
json.dump(dict(zs=zs.tolist(), PB=PB, bg_plane=bgp.tolist(), candidates=cands, beads=rows, roi=roi,
               median_curve=dict(z=zg.tolist(), sx=med[:, 0].tolist(), sy=med[:, 1].tolist())),
          open(f'{outdir}/beads_found.json', 'w'))
