"""How the tetrapod PSF behaves for detection and ROI choice: simulated SMLM-like spots (a bead-fitted VectorialPSF,
photons / background like the FD-DeepLoc movie) over z = -3 .. 3 um around the focal plane.

Per z: the fraction of the PSF energy inside centred R x R boxes, the number of detected peaks per emitter and the
offset of the detected peak from the true position for a few detection settings, and the z found by the localizer
(spline, multi-start) vs the truth.

    python -I detect_check.py BEADS_JSON OUT_JSON [--photons 2500 --bg 55]
"""
import argparse
import dataclasses
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np
import torch
from common import OPTICS, setup_path

setup_path()
from torch_psf.vectorial import VectorialPSF   # noqa: E402
from torch_psf.detect import detect_spots      # noqa: E402
from torch_psf.fit import localize             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('beads'); ap.add_argument('out')
    ap.add_argument('--photons', type=float, default=4000.0); ap.add_argument('--bg', type=float, default=55.0)
    ap.add_argument('--focus', type=float, default=3.0)
    ap.add_argument('--roisizes', type=int, nargs='*', default=[31, 35, 41, 45, 51])
    a = ap.parse_args()
    dev = torch.device('cuda')
    b = json.load(open(a.beads))
    optics = OPTICS.__class__.from_dict(b['optics'])
    big = 81
    vec = VectorialPSF.from_optics(optics, big, zernike_indices=tuple(b['noll']), npupil=64, oversample=3, device=dev,
                                   focus_um=a.focus, z_range=(0.0, 2 * a.focus))
    vec.set_zernike(b['joint']['zcoef'])
    zs = np.round(np.arange(0.0, 2 * a.focus + 1e-6, 0.25), 3)
    res = dict(z_rel_um=(zs - a.focus).tolist(), energy={}, detect={})
    with torch.no_grad():
        I = vec(torch.tensor([[0.0, 0.0, z] for z in zs], device=dev, dtype=torch.float32), a.focus)
    c = big // 2
    tot = I.sum((1, 2))
    for R in a.roisizes:
        h = R // 2
        res['energy'][R] = (I[:, c - h:c + h + 1, c - h:c + h + 1].sum((1, 2)) / tot).cpu().numpy().round(3).tolist()
        print(f'R {R}: energy fraction min {min(res["energy"][R]):.3f}', res['energy'][R])
    res['total_energy_in_81px'] = tot.cpu().numpy().round(3).tolist()
    # detection: one emitter in the centre of a 256 x 256 frame (the noise estimate is then background-dominated, as in
    # the movie), 100 noise realisations per z, random subpixel shift; peaks within 30 px of the emitter are counted
    g = torch.Generator(device='cpu').manual_seed(0)
    n, F = 100, 256
    cf = F // 2
    for sig, nms in (((2.0, 6.0), 9), ((2.0, 6.0), 15), ((3.0, 8.0), 15), ((3.0, 8.0), 21), ((4.0, 10.0), 21)):
        for thr in (3.0, 4.0, 5.0):
            key = f'sigma{sig[0]:g}_{sig[1]:g}_nms{nms}_thr{thr:g}'
            rec = []
            for k, z in enumerate(zs):
                sh = torch.rand(n, 2, generator=g) - 0.5
                pos = torch.cat([sh, torch.full((n, 1), float(z))], 1).to(dev)
                with torch.no_grad():
                    mu = torch.full((n, F, F), a.bg, device=dev)
                    mu[:, cf - c:cf + c + 1, cf - c:cf + c + 1] += vec(pos, a.focus) * a.photons
                p = detect_spots(torch.poisson(mu), threshold=thr, sigma=sig, nms=nms, border=0)
                dx = p['x'].float() - cf - sh[p['frame'].cpu(), 0].to(dev)
                dy = p['y'].float() - cf - sh[p['frame'].cpu(), 1].to(dev)
                near = torch.maximum(dx.abs(), dy.abs()) <= 30
                fr_, dd = p['frame'][near], torch.hypot(dx, dy)[near]
                cnt = torch.bincount(fr_, minlength=n).cpu().numpy()
                off = [float(dd[fr_ == i][torch.argmax(p['score'][near][fr_ == i])]) for i in range(n) if cnt[i]]
                rec.append(dict(z_rel=float(z - a.focus), detected=float((cnt > 0).mean()), peaks_mean=float(cnt.mean()),
                                multi=float((cnt > 1).mean()), offset_med=float(np.median(off)) if off else None,
                                offset_p95=float(np.percentile(off, 95)) if off else None,
                                false_per_frame=float((~near).sum()) / n))
            res['detect'][key] = rec
            det = np.array([r['detected'] for r in rec]); mul = np.array([r['multi'] for r in rec])
            o95 = [r['offset_p95'] for r in rec if r['offset_p95'] is not None]
            print(f'{key}: detected min {det.min():.2f} mean {det.mean():.2f} | multi max {mul.max():.2f} mean {mul.mean():.2f}'
                  f' | offset p95 max {max(o95) if o95 else float("nan"):.1f} px | false/frame {np.mean([r["false_per_frame"] for r in rec]):.2f}')
    # localization z from the multi-start (zstarts=5) on the spline of this PSF, ROI 41, emitter at the centre +- 2 px
    R = 41
    vs = vec.resized(R)
    spl = vs.to_spline(R, (0.0, 2 * a.focus), oversample=2, dz_um=0.04, zpad=5, xy_limit=4.0)
    zt = torch.rand(2000, generator=g) * 2 * a.focus
    pos = torch.cat([(torch.rand(2000, 2, generator=g) - 0.5) * 4, zt[:, None]], 1).to(dev)
    with torch.no_grad():
        mu = vs.forward_batched(pos, a.focus) * a.photons + a.bg
    d = torch.poisson(mu)
    for zst in (3, 5, 9):
        th, _ = localize(spl, d, zstarts=zst)
        err = (th[:, 2] - pos[:, 2]).cpu().numpy()
        res[f'loc_zstarts{zst}'] = dict(frac_gt_200nm=float((np.abs(err) > 0.2).mean()), rms_ok_nm=float(1000 * np.sqrt(np.mean(err[np.abs(err) < 0.2] ** 2))),
                                        xy_rms_px=float(torch.sqrt(((th[:, :2] - pos[:, :2]) ** 2).mean()).cpu()))
        print(f'zstarts {zst}:', res[f'loc_zstarts{zst}'])
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
