"""Per-bead vectorial Zernike fits of FD-DeepLoc tetrapod bead stacks (torch_psf.estimation.fit_beads).

For one bead stack (61 planes, 100 nm steps): the beads of their calibration (their pos list) are cut out as roisize
ROIs; first one joint fit of all beads (shared Zernikes, started from zero aberrations, i.e. independent of their
numbers), then every bead on its own, started from the joint fit, and again from the fits of its 2 nearest beads and
from the median of all, each with that start's bead height (pass 2 against local minima; the lowest NLL is kept): Zernikes, xy, per-plane photons / background, a
cubic lateral drift over the planes and the bead height (= axial offset from the stack centre; n_med = n_imm, so
height and focus offset are the same thing). The movie is converted to photons with the photon-transfer gain.

    python -I beads_fit.py STACK_TIF THEIR_PREFIX OUT_JSON [--modes theirs|noll5_21|noll5_37] [--sign 1]

THEIR_PREFIX: the stack's calibration files without the suffix (..._MMStack_Default.ome), for their bead list.
"""
import argparse
import copy
import json
import time
import numpy as np
import tifffile
import torch
import dataclasses
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
from common import OPTICS, THEIR_NOLL, load_their_beads, setup_path

setup_path()
from torch_psf.vectorial import VectorialPSF            # noqa: E402
from torch_psf.estimation import fit_beads, stage_focus  # noqa: E402

MODES = {'theirs': THEIR_NOLL, 'noll5_21': list(range(5, 22)), 'noll5_37': list(range(5, 38))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stack'); ap.add_argument('their_prefix'); ap.add_argument('out')
    ap.add_argument('--modes', default='theirs', choices=list(MODES))
    ap.add_argument('--sign', type=float, default=1.0, help='focus = sign dz (k - 30): +1 = focus moves up with k')
    ap.add_argument('--roisize', type=int, default=61)
    ap.add_argument('--gain', type=float, default=0.818); ap.add_argument('--offset', type=float, default=100.0)
    ap.add_argument('--joint_iters', type=int, default=600); ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--npupil', type=int, default=64); ap.add_argument('--oversample', type=int, default=3)
    ap.add_argument('--beads', type=int, nargs='*', default=None, help='subset of bead indices')
    ap.add_argument('--chunk', type=int, default=3)
    ap.add_argument('--pixel_um', type=float, default=0.108)
    ap.add_argument('--restarts', type=int, default=2, help='pass 2: restarts from the nearest beads')
    a = ap.parse_args()
    dev = torch.device('cuda')
    t0 = time.time()
    th = load_their_beads(a.their_prefix)
    st = tifffile.imread(a.stack).astype(np.float32)
    nk = st.shape[0]
    st = (st - a.offset) * a.gain
    P, h = a.roisize, a.roisize // 2
    H, W = st.shape[1:]
    idx = list(range(len(th['pos']))) if a.beads is None else a.beads
    rois, corners, keep = [], [], []
    for i in idx:
        x, y = th['pos'][i]
        x0, y0 = int(round(x)) - h, int(round(y)) - h
        if x0 < 0 or y0 < 0 or x0 + P > W or y0 + P > H:
            print('bead', i, 'too close to the border, skipped'); continue
        rois.append(st[:, y0:y0 + P, x0:x0 + P]); corners.append((x0, y0)); keep.append(i)
    d = torch.as_tensor(np.stack(rois), device=dev)
    focus = stage_focus(nk, 0.1, a.sign).to(dev)
    zidx = MODES[a.modes]
    optics = dataclasses.replace(OPTICS, pixel_um=a.pixel_um)
    vec = VectorialPSF.from_optics(optics, P, zernike_indices=tuple(zidx), npupil=a.npupil, oversample=a.oversample,
                                   device=dev)
    sid = torch.zeros(len(keep), dtype=torch.long, device=dev)
    # joint fit (shared Zernikes, one drift for the stack): a coarse then a fine pass
    r = fit_beads(d, focus, vec, iters=a.joint_iters, lr=1e-2, stack_id=sid, drift=True, fit_height=True,
                  emitter_z_um=0.0, chunk=a.chunk)
    r = fit_beads(d, focus, r['psf'], iters=a.joint_iters // 2, lr=2e-3, stack_id=sid, drift=True, fit_height=True,
                  emitter_z_um=r['height_um'], chunk=a.chunk)
    joint = dict(zcoef=r['psf'].zcoef.detach().cpu().tolist(), nll_per_px=r['report']['nll_per_px'],
                 chi2_median=r['report']['chi2_median'], height_um=r['height_um'])
    print(f'joint fit: nll/px {joint["nll_per_px"]:.4f} chi2 {joint["chi2_median"]:.3f} height {joint["height_um"]:.3f} '
          f'um ({time.time() - t0:.0f} s)', flush=True)
    vj = r['psf']

    def fit_one(b, v_start, h_start):
        db = d[b:b + 1]
        r1 = fit_beads(db, focus, v_start, iters=a.iters, lr=3e-3, drift=True, fit_height=True, emitter_z_um=h_start)
        return fit_beads(db, focus, r1['psf'], iters=a.iters // 2, lr=1e-3, drift=True, fit_height=True,
                         emitter_z_um=r1['height_um'])

    def record(b, i, r1, start):
        xy = r1['xy'][0].cpu().numpy()
        cx, cy = corners[b][0] + h + xy[0], corners[b][1] + h + xy[1]
        return dict(index=i, x=float(cx), y=float(cy), zcoef=r1['psf'].zcoef.detach().cpu().tolist(),
                    height_um=r1['height_um'], nll_per_px=r1['report']['nll_per_px'],
                    chi2_median=r1['report']['chi2_median'], photons_median=float(r1['photons'].median()),
                    bg_median=float(r1['bg'].median()), their_coef=th['coef'][i].tolist(),
                    their_z_pos_nm=float(th['z_pos'][i]), start=start)

    # pass 1: every bead from the joint fit
    beads = []
    for b, i in enumerate(keep):
        beads.append(record(b, i, fit_one(b, vj, joint['height_um']), 'joint'))
        print(f'pass 1 bead {i:2d} nll/px {beads[-1]["nll_per_px"]:.4f} ({time.time() - t0:.0f} s)', flush=True)
    # pass 2 (local minima): restart every bead from the pass-1 fits of its nearest beads and from the median of all
    # beads; keep the lowest NLL. Independent of their numbers.
    Z1 = np.array([q['zcoef'] for q in beads]); XY = np.array([[q['x'], q['y']] for q in beads])
    H1 = np.array([q['height_um'] for q in beads])
    for b, i in enumerate(keep):
        dd = np.hypot(*(XY - XY[b]).T); dd[b] = np.inf
        starts = [('median', np.median(Z1, 0), float(np.median(H1)))] + [
            (f'bead {keep[j]}', Z1[j], float(H1[j])) for j in np.argsort(dd)[:a.restarts]]
        for name, zc, h0 in starts:
            v = copy.deepcopy(vj).set_zernike(list(map(float, zc)))
            r1 = fit_one(b, v, h0)
            if r1['report']['nll_per_px'] < beads[b]['nll_per_px'] - 1e-6:
                beads[b] = record(b, i, r1, name)
        rec = beads[b]
        print(f'bead {i:2d} at ({rec["x"]:7.1f}, {rec["y"]:7.1f}) nll/px {rec["nll_per_px"]:.4f} chi2 '
              f'{rec["chi2_median"]:.3f} height {rec["height_um"]:+.3f} um photons {rec["photons_median"]:.0f} bg '
              f'{rec["bg_median"]:.1f} start {rec["start"]} ({time.time() - t0:.0f} s)', flush=True)
    out = dict(stack=a.stack, modes=a.modes, noll=zidx, sign=a.sign, roisize=P, gain=a.gain, offset=a.offset,
               optics=optics.to_dict(), npupil=a.npupil, oversample=a.oversample, joint=joint, beads=beads,
               their_mean=th['mean'].tolist(), seconds=time.time() - t0)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
