"""Matched-filter detection of tetrapod spots and ROI extraction (study-side; torch_psf.detect finds lobes, not emitters).

A difference-of-Gaussians peak of a tetrapod spot sits on one lobe, 5 px (in focus) to 20 px (at +-3 um) from the
emitter (detect_check.py). Here every frame (photons) is correlated with zero-mean, unit-norm templates of a bead-fitted
VectorialPSF at nz heights (FFT); the response of each template is divided by its robust noise (1.4826 MAD over the
frame), the maximum over the templates is taken, and local maxima (window nms px) above `threshold` are the
detections: emitter centres to about a pixel, plus the template z.

Output npz: frame, x, y (px, integer peak), score, zt (template height, um; focus at --focus), isolated (no other
detection within --isolation px Chebyshev in the same frame), and for the isolated ones inside the border the ROIs
(uint16 ADU, roisize^2) with their corners.

    python -I detect_tetrapod.py MOVIE BEADS_JSON OUT_NPZ [--threshold 7] [--roisize 41] [--frames 0 800]
    python -I detect_tetrapod.py --selftest BEADS_JSON OUT_JSON
"""
import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np
import torch
import torch.nn.functional as Fn
from common import OPTICS, setup_path

setup_path()
from torch_psf.vectorial import VectorialPSF   # noqa: E402
from torch_psf.movie import open_movie         # noqa: E402


def templates(beads_json, focus, size=61, dz=0.2, z_range=None, device='cuda'):
    """zero-mean unit-norm PSF templates [nz, size, size] and their heights (um)"""
    b = json.load(open(beads_json))
    vec = VectorialPSF.from_optics(OPTICS.__class__.from_dict(b['optics']), size, zernike_indices=tuple(b['noll']),
                                   npupil=64, oversample=3, device=device, focus_um=focus)
    zc = b['joint']['zcoef'] if 'zcoef_start' not in b else b['zcoef_start']
    vec.set_zernike(zc)
    lo, hi = (0.0, 2 * focus) if z_range is None else z_range
    zs = torch.arange(lo, hi + 1e-6, dz)
    with torch.no_grad():
        T = vec(torch.stack([torch.zeros_like(zs), torch.zeros_like(zs), zs], 1).to(device).float(), focus)
    T = T - T.mean((1, 2), keepdim=True)
    T = T / T.norm(dim=(1, 2), keepdim=True)
    return T, zs.numpy()


class MatchedFilter:
    def __init__(self, T, H, W, nms=21):
        self.nz, s = T.shape[0], T.shape[-1]
        self.H, self.W = H, W
        self.Hp, self.Wp = H + s, W + s                  # zero padding against wrap-around
        K = torch.zeros(self.nz, self.Hp, self.Wp, device=T.device)
        h = s // 2
        K[:, :s, :s] = T
        K = torch.roll(K, (-h, -h), (1, 2))              # template centre at the origin
        self.Kf = torch.conj(torch.fft.rfft2(K))
        self.nms = nms

    @torch.no_grad()
    def __call__(self, frame, threshold):
        """frame [H, W] photons -> (y, x, score, iz) of the detections"""
        f = frame - frame.median()
        F = torch.fft.rfft2(f, s=(self.Hp, self.Wp))
        best = torch.full((self.H, self.W), -1e9, device=frame.device)
        arg = torch.zeros((self.H, self.W), dtype=torch.long, device=frame.device)
        for z0 in range(0, self.nz, 8):
            r = torch.fft.irfft2(F[None] * self.Kf[z0:z0 + 8], s=(self.Hp, self.Wp))[:, :self.H, :self.W]
            med = r.flatten(1).median(1).values
            sig = 1.4826 * (r.flatten(1) - med[:, None]).abs().median(1).values
            r = (r - med[:, None, None]) / sig[:, None, None]
            m, a = r.max(0)
            upd = m > best
            best = torch.where(upd, m, best); arg = torch.where(upd, a + z0, arg)
        mx = Fn.max_pool2d(best[None, None], self.nms, 1, self.nms // 2)[0, 0]
        pk = (best == mx) & (best > threshold)
        y, x = torch.nonzero(pk, as_tuple=True)
        return y, x, best[y, x], arg[y, x]


def isolated(frame_idx, x, y, dist):
    """True where no other detection of the same frame is within dist px (Chebyshev)"""
    ok = np.ones(len(x), bool)
    for fr in np.unique(frame_idx):
        i = np.flatnonzero(frame_idx == fr)
        if len(i) < 2:
            continue
        dx = np.abs(x[i, None] - x[None, i]); dy = np.abs(y[i, None] - y[None, i])
        dd = np.maximum(dx, dy) + np.eye(len(i), dtype=np.int64) * 10 ** 6
        ok[i] = dd.min(1) > dist
    return ok


def selftest(a):
    """simulated single spots in 256 x 256 frames: detection rate, multiplicity, centre offset and z per height"""
    dev = torch.device('cuda')
    T, zs = templates(a.beads, a.focus, device=dev)
    b = json.load(open(a.beads))
    vec = VectorialPSF.from_optics(OPTICS.__class__.from_dict(b['optics']), 81, zernike_indices=tuple(b['noll']),
                                   device=dev, focus_um=a.focus)
    vec.set_zernike(b['joint']['zcoef'])
    F, c, n = 256, 40, 60
    mf = MatchedFilter(T, F, F, nms=a.nms)
    g = torch.Generator().manual_seed(1)
    out = []
    for z in np.arange(0.0, 2 * a.focus + 1e-6, 0.25):
        sh = torch.rand(n, 2, generator=g) - 0.5
        with torch.no_grad():
            img = vec(torch.cat([sh, torch.full((n, 1), float(z))], 1).to(dev), a.focus) * a.photons
        cnt, off, dzs = [], [], []
        for i in range(n):
            mu = torch.full((F, F), a.bg, device=dev)
            mu[128 - c:128 + c + 1, 128 - c:128 + c + 1] += img[i]
            y, x, s, iz = mf(torch.poisson(mu), a.threshold)
            dx, dy = x.float() - 128 - sh[i, 0].to(dev), y.float() - 128 - sh[i, 1].to(dev)
            near = torch.maximum(dx.abs(), dy.abs()) <= 30
            cnt.append(int(near.sum()))
            if near.any():
                j = torch.argmax(torch.where(near, s, torch.full_like(s, -1e9)))
                off.append(float(torch.hypot(dx[j], dy[j]))); dzs.append(float(zs[int(iz[j])] - z))
        cnt = np.array(cnt)
        rec = dict(z_rel=float(z - a.focus), detected=float((cnt > 0).mean()), multi=float((cnt > 1).mean()),
                   offset_med=float(np.median(off)) if off else None, offset_p95=float(np.percentile(off, 95)) if off else None,
                   dz_med=float(np.median(dzs)) if dzs else None)
        out.append(rec)
        print('z %+5.2f det %.2f multi %.2f offset med %s p95 %s template dz %s' % (
            rec['z_rel'], rec['detected'], rec['multi'], rec['offset_med'] and round(rec['offset_med'], 2),
            rec['offset_p95'] and round(rec['offset_p95'], 2), rec['dz_med']))
    json.dump(dict(photons=a.photons, bg=a.bg, threshold=a.threshold, nms=a.nms, per_z=out), open(a.out, 'w'), indent=1)


def run(a):
    dev = torch.device('cuda')
    mv = open_movie(a.movie)
    H, W = mv.shape
    T, zs = templates(a.beads, a.focus, device=dev)
    mf = MatchedFilter(T, H, W, nms=a.nms)
    f0, f1 = a.frames if a.frames else (0, mv.n)
    t0 = time.time()
    rec = {k: [] for k in ('frame', 'x', 'y', 'score', 'zt')}
    for t in range(f0, f1, a.chunk):
        fr = torch.as_tensor(mv.read(t, min(t + a.chunk, f1)), device=dev)
        ph = (fr - a.offset) * a.gain
        for k in range(len(ph)):
            y, x, s, iz = mf(ph[k], a.threshold)
            rec['frame'].append(np.full(len(x), t + k)); rec['x'].append(x.cpu().numpy()); rec['y'].append(y.cpu().numpy())
            rec['score'].append(s.cpu().numpy()); rec['zt'].append(zs[iz.cpu().numpy()])
        if (t - f0) % (10 * a.chunk) == 0:
            print(f'frame {t}: {sum(len(v) for v in rec["x"])} detections ({time.time() - t0:.0f} s)', flush=True)
    det = {k: np.concatenate(v) for k, v in rec.items()}
    det['isolated'] = isolated(det['frame'], det['x'].astype(np.int64), det['y'].astype(np.int64), a.isolation)
    h = a.roisize // 2
    inb = (det['x'] >= h) & (det['y'] >= h) & (det['x'] < W - h) & (det['y'] < H - h)
    sel = np.flatnonzero(det['isolated'] & inb)
    rois = np.zeros((len(sel), a.roisize, a.roisize), np.uint16)
    order = np.argsort(det['frame'][sel], kind='stable')
    sel = sel[order]
    for t in range(f0, f1, a.chunk):
        fr = mv.read(t, min(t + a.chunk, f1))
        for i in np.flatnonzero((det['frame'][sel] >= t) & (det['frame'][sel] < t + a.chunk)):
            j = sel[i]
            rois[i] = fr[det['frame'][j] - t, det['y'][j] - h:det['y'][j] + h + 1, det['x'][j] - h:det['x'][j] + h + 1]
    np.savez(a.out, **det, roi_index=sel, rois=rois, corners=np.stack([det['x'][sel] - h, det['y'][sel] - h], 1),
             roisize=a.roisize, gain=a.gain, offset=a.offset, threshold=a.threshold, isolation=a.isolation,
             focus=a.focus)
    print(f'{len(det["x"])} detections ({len(det["x"]) / (f1 - f0):.0f} per frame), {int(det["isolated"].sum())} '
          f'isolated, {len(sel)} ROIs; {time.time() - t0:.0f} s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('movie'); ap.add_argument('beads'); ap.add_argument('out', nargs='?')
    ap.add_argument('--threshold', type=float, default=7.0); ap.add_argument('--nms', type=int, default=21)
    ap.add_argument('--roisize', type=int, default=41); ap.add_argument('--isolation', type=int, default=41)
    ap.add_argument('--focus', type=float, default=3.0)
    ap.add_argument('--gain', type=float, default=0.818); ap.add_argument('--offset', type=float, default=100.0)
    ap.add_argument('--photons', type=float, default=4000.0); ap.add_argument('--bg', type=float, default=56.0)
    ap.add_argument('--frames', type=int, nargs=2, default=None); ap.add_argument('--chunk', type=int, default=20)
    a = ap.parse_args()
    if a.selftest:
        a.beads, a.out = a.movie, a.beads
        selftest(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
