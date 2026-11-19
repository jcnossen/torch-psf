"""lambda-DNA, one hand-picked straight strand, the paper's way (Fig. 4b-d: drift-corrected localisations of one strand,
the DNA axis from a spline fit, Delta phi = dipole azimuth - local axis angle, its median and MAD; the g2 histogram; the
FWHM of the localisations across the strand), with their model reproduced exactly (vortex_exact.Exact: their NAT map,
plate, immersion mismatch, their detection, their initial values + a z grid of starts), next to the same analysis of
their own hdf5 localisations (2500_2_X1_10_drift_corrected.hdf5, the first movie) on the same strand.

Strand: picked by eye in the crop x 400-656, y 16-272 of stk_0001 (their localisations of frames 0-1979): the long
straight one through (450, 124) and (650, 226) px; localisations within 4 px of that line are taken, then the axis is
a smoothing spline y'(x') in the strand frame, iterated twice (8 px residual cut, then 4 px).
Drift: torch_psf Dataset.estimate_drift (DME v2) on all our localisations of the strip, then drift_corrected.

usage: CUDA_VISIBLE_DEVICES=2 DME_PATH=~/simcode2/external/drift-estimation python -I study/orientation/vortex_strand.py
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np                                                                     # noqa: E402
import torch                                                                           # noqa: E402
import vortex_real as VR                                                              # noqa: E402
import vortex_baseline as VB                                                          # noqa: E402
import vortex_exact as VE                                                             # noqa: E402
from torch_psf.dataset import Dataset                                                  # noqa: E402
from torch_psf.orientation import fit_orientation, unit                                # noqa: E402

OUT = VR.OUT
DEV = VR.DEV
P0, P1 = np.array([450.0, 124.0]), np.array([650.0, 226.0])      # the strand (x = column, y = row), px
STRIP = (430, 100, 240, 150)                                      # x0, y0, w, h of the processed strip


def log(*a):
    print(*a, flush=True)


def detect_and_fit(frames=(0, 1980), chunk=200):
    from vecfit.funfit.initialvalues import initialvalues
    t0 = time.time()
    x0, y0, w, h = STRIP
    E = VE.Exact(refmed=1.333)
    out = {k: [] for k in ('frame', 'col', 'row', 'theta', 'nll', 'crlb_x', 'crlb_y')}
    for f0 in range(frames[0], frames[1], chunk):
        mov = VR._movie_photons(VR.DNA_MOVIE, frames=(f0, min(f0 + chunk, frames[1]))).numpy()
        mov = np.where(mov <= 0, 1e-3, mov)
        rois, rc, fr = [], [], []
        for t in range(mov.shape[0]):
            r, c = VB.their_detect(mov[t])
            sel = (c >= x0) & (c < x0 + w) & (r >= y0) & (r < y0 + h)
            for rr, cc in zip(r[sel], c[sel]):
                rois.append(mov[t, rr - 7:rr + 8, cc - 7:cc + 8]); rc.append((rr, cc)); fr.append(f0 + t)
        if not rois:
            continue
        rc = np.array(rc)
        draw = torch.as_tensor(np.stack(rois), dtype=torch.float32, device=DEV)
        th_t = initialvalues(draw.double(), E.p)                       # their initial values (their frame)
        t0i = E.their_to_ours(th_t.cpu().numpy())
        nat = E.nat(np.stack([rc[:, 1] + 1, rc[:, 0] + 1], 1), 1024)
        zo = E.offsets(nat)
        d = torch.flip(draw, (-1, -2))
        best = None
        for z in (None, -0.6, -0.3, 0.3, 0.6):
            st = t0i.clone()
            if z is not None:
                st[:, 2] = z
            r = fit_orientation(E.psf, d, theta0=st, iters=50, z_limits=(-1.1, 1.1), g_limits=(0.0, 1.0), batch=800,
                                zcoef_offset=zo)
            if best is None:
                best = r
            else:
                b = r['nll'] < best['nll']
                best['theta'][b] = r['theta'][b]; best['nll'][b] = r['nll'][b]
                for k in ('x', 'y'):
                    best['crlb'][k][b] = r['crlb'][k][b]
        out['frame'].append(np.array(fr)); out['col'].append(rc[:, 1]); out['row'].append(rc[:, 0])
        out['theta'].append(best['theta'].cpu().numpy()); out['nll'].append(best['nll'].cpu().numpy())
        out['crlb_x'].append(best['crlb']['x'].cpu().numpy()); out['crlb_y'].append(best['crlb']['y'].cpu().numpy())
        log(f'frames {f0}-{f0 + chunk}: {len(fr)} spots ({time.time() - t0:.0f} s)')
    R = {k: np.concatenate(v) for k, v in out.items()}
    th = torch.as_tensor(R['theta'], device=DEV)
    g2, Nf = VB.to_their_g2(E.v, th[:, 5:8], th[:, 8])
    mu_t = E.ours_to_their_axis(unit(th[:, 5:8])).cpu().numpy()      # their axis: (row, column, z)
    # positions in the frame (0-based column, row): our rot180 frame -> x_col = col - x_fit, y_row = row - y_fit
    R.update(x=R['col'] - R['theta'][:, 0], y=R['row'] - R['theta'][:, 1], z_nm=R['theta'][:, 2] * 1e3,
             g2=g2.cpu().numpy(), N=(th[:, 3] * Nf).cpu().numpy(), mu_row=mu_t[:, 0], mu_col=mu_t[:, 1], mu_z=mu_t[:, 2])
    np.savez(os.path.join(OUT, 'strand_fits.npz'), **R)
    return R


def strand_analysis(x, y, ang_dip, g2, label):
    """x, y (px, column / row), ang_dip: in-plane dipole axis angle in the image (deg, from +column towards +row)"""
    from scipy.interpolate import UnivariateSpline
    d = P1 - P0; L = np.linalg.norm(d); u = d / L; nvec = np.array([-u[1], u[0]])
    s = (np.stack([x, y], 1) - P0) @ u; t = (np.stack([x, y], 1) - P0) @ nvec
    sel = (s > 0) & (s < L) & (np.abs(t) < 4)
    for cut in (8.0, 4.0, 2.0):
        o = np.argsort(s[sel]); ss, tt = s[sel][o], t[sel][o]
        ss, idx = np.unique(ss, return_index=True); tt = tt[idx]
        sp = UnivariateSpline(ss, tt, s=len(ss) * 1.0 ** 2, k=3)
        r = t - sp(s)
        sel = (s > 0) & (s < L) & (np.abs(r) < cut)
    r = (t - sp(s)) * 65.0
    sel = (s > 0) & (s < L) & (np.abs(r) < 130)
    slope = sp.derivative()(s[sel])
    axis_ang = np.degrees(np.arctan2(u[1] + slope * nvec[1], u[0] + slope * nvec[0]))
    dphi = (ang_dip[sel] - axis_ang) % 180.0
    h, e = np.histogram(r[sel], bins=np.arange(-60, 61, 2))
    above = np.nonzero(h >= h.max() / 2)[0]
    fwhm = float((above[-1] - above[0] + 1) * 2) if len(above) else float('nan')
    hg, _ = np.histogram(g2[sel], bins=20, range=(0, 1))
    mad = lambda a: float(np.median(np.abs(a - np.median(a))))
    res = dict(label=label, n=int(sel.sum()), length_um=float(L * 0.065), dphi_median=float(np.median(dphi)),
               dphi_mad=mad(dphi), dphi_folded_median=float(np.median(np.minimum(dphi, 180 - dphi))),
               g2_peak=float((np.argmax(hg) + 0.5) / 20), g2_median=float(np.median(g2[sel])),
               fwhm_nm=fwhm, sd_across_nm=float(np.std(r[sel][np.abs(r[sel]) < 60])))
    log(json.dumps(res))
    return res, dict(dphi=dphi.tolist(), across=r[sel].tolist(), g2=g2[sel].tolist())


def main():
    t0 = time.time()
    p = os.path.join(OUT, 'strand_fits.npz')
    R = dict(np.load(p)) if os.path.exists(p) else detect_and_fit()
    n = len(R['x'])
    ds = Dataset.from_arrays(np.stack([R['x'], R['y']], 1), R['frame'].astype(np.int32),
                             crlb=np.stack([R['crlb_x'], R['crlb_y']], 1).astype(np.float32),
                             photons=R['N'].astype(np.float32), meta=dict(pixel_nm=65.0, n_dims=2))
    res = {}
    try:
        dr = ds.estimate_drift(n_frames=int(R['frame'].max()) + 1)
        dc = ds.drift_corrected(dr)
        xy_dc = dc['pos'].numpy()
        res['drift_rms_px'] = float(np.sqrt(((dr.curve - dr.curve.mean(0)) ** 2).sum(1).mean()))
    except Exception as e:                        # noqa: BLE001
        log(f'drift estimation failed: {e}')
        xy_dc = ds['pos'].numpy()
        res['drift_error'] = str(e)
    ang = np.degrees(np.arctan2(R['mu_row'], R['mu_col']))           # image angle of the in-plane axis
    lists = {}
    res['ours_raw'], lists['ours_raw'] = strand_analysis(R['x'], R['y'], ang, R['g2'], 'ours, no drift correction')
    res['ours_dc'], lists['ours_dc'] = strand_analysis(xy_dc[:, 0], xy_dc[:, 1], ang, R['g2'], 'ours, drift corrected')
    import h5py
    for name, fn in (('theirs_dc', '2500_2_X1_10_drift_corrected.hdf5'), ('theirs_raw', '2500_2_X1_10.hdf5')):
        with h5py.File(os.path.join(VR.DATA, fn), 'r') as f:
            L = f['locs'][:]
        L = L[L['frame'] < 1980]
        az = np.radians(L['sx'])                      # their azim: from their x (row) towards y (column)
        ang_t = np.degrees(np.arctan2(np.cos(az), np.sin(az)))      # image angle (from +column towards +row)
        # their hdf5 x / y are Picasso (column, row) with pixel centres at integer + 0.5? (roi2fov: 1-based) -> -1
        res[name], lists[name] = strand_analysis(L['x'] - 1, L['y'] - 1, ang_t, L['ellipticity'], name)
    res['seconds'] = time.time() - t0
    json.dump(dict(summary=res, lists=lists), open(os.path.join(OUT, 'strand.json'), 'w'))


if __name__ == '__main__':
    main()
