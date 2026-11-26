"""Paired 1809 localization check with equal lateral bounds and a physical refinement."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_psf import PSF
from torch_psf.estimation import localize_field, nat_field, split_checkerboard
from torch_psf.fit import localize


class MatchedBounds:
    def __init__(self, model):
        self.render = model.render
        self.z_range = model.z_range
        self.xy_limit = 3.0
        self.dtype = torch.float32


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample, cam = np.load(args.sample), np.load(args.camera)
    field, frame = sample["xy"], sample["frame"]
    held = split_checkerboard(field[:, 0], field[:, 1], 16) == 1
    rows = []
    for r in json.loads(args.fitted.read_text()):
        angle = r["angle"]
        selected = np.random.default_rng(1809 + angle).choice(np.flatnonzero(held & (frame % 6 // 3 == angle)), 48, replace=False)
        xy = field[selected]
        y = xy[:, 1, None, None] + np.arange(-7, 8)[None, :, None]
        x = xy[:, 0, None, None] + np.arange(-7, 8)[None, None, :]
        obs = torch.as_tensor((sample["adu"][selected] - cam["offset_adu"][y, x]) * cam["gain"], device=device, dtype=torch.float32)
        var = torch.as_tensor(cam["readnoise_e"][y, x] ** 2, device=device, dtype=torch.float32)
        base = PSF.load(args.calibration, roisize=15, angle=angle, device=device)
        base.z_range = (0., .4)
        base.zcoef.zero_()
        nat = nat_field(base, r["fit"]["gamma"], r["fit"]["norm"])
        spline = nat.to_tiled(tile=30., kernel="bspline", integrated=True, oversample=3, dz_um=.025, xy_limit=3.)
        xy = torch.as_tensor(xy, device=device, dtype=torch.float32)
        opt = dict(var=var, iters=25, zstarts=3, batch=48)
        seed, _ = localize_field(spline, obs, xy, **opt)
        direct = MatchedBounds(nat.bind(xy))
        cold, loss = localize(direct, obs, **opt)
        warm, warm_loss = localize(direct, obs, theta0=seed, **opt)
        rms = lambda t: float(1000 * (t[:, 2]-cold[:, 2]).square().mean().sqrt())
        rows.append(dict(angle=angle, n=len(selected), spline_z_rms_nm=rms(seed), polished_z_rms_nm=rms(warm),
                         mean_polished_minus_cold_nll=float((warm_loss-loss).mean()),
                         mean_best_of_both_minus_cold_nll=float(torch.minimum(warm_loss-loss, torch.zeros_like(loss)).mean()),
                         polished_improves_by_point_one=int((warm_loss < loss-.1).sum()),
                         cold_improves_by_point_one=int((loss < warm_loss-.1).sum())))
    args.output.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("sample", "camera", "calibration", "fitted", "output"):
        p.add_argument(key, type=Path)
    run(p.parse_args())
