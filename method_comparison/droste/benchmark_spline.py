"""Time table construction, warm localization, and batched one-table PSF evaluation."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch_psf import PSF
from torch_psf.estimation import localize_field, nat_field, split_checkerboard


@torch.no_grad()
def run(args):
    def elapsed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    d, camera = np.load(args.sample), np.load(args.camera)
    xy = d["xy"]
    held = (split_checkerboard(*xy.T, 16) == 1) & (d["frame"] % 6 // 3 == 0)
    xy = xy[held]
    dy = xy[:, 1, None, None] + np.arange(-7, 8)[None, :, None]
    dx = xy[:, 0, None, None] + np.arange(-7, 8)[None, None, :]
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda")
    obs = tensor((d["adu"][held] - camera["offset_adu"][dy, dx]) * camera["gain"])
    var = tensor(camera["readnoise_e"][dy, dx] ** 2)
    xy = tensor(xy)
    base = PSF.load(args.calibration, roisize=15, angle=0, device="cuda")
    base.z_range = (0., .4)
    base.zcoef.zero_()
    fit = json.loads(args.fitted.read_text())[0]["fit"]
    nat = nat_field(base, fit["gamma"], fit["norm"])
    tiled = nat.to_tiled(tile=30., kernel="bspline", integrated=True, oversample=3, dz_um=.025, xy_limit=3.)
    ids = torch.unique(tiled.tile_index(xy)).tolist()
    build = elapsed(lambda: [tiled.spline(i) for i in ids])
    warm = elapsed(lambda: localize_field(tiled, obs, xy, var, iters=25, zstarts=3, batch=48))
    direct, spline = nat.at(*tiled.centers[0]), tiled.spline(0)
    pos = tensor(np.random.default_rng(42).uniform([-.5, -.5, 0], [.5, .5, .4], (48, 3)))
    result = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, held=len(xy),
                  axis=0, occupied_tiles=len(ids), build_s=build, warm_fit_s=warm, batch=len(pos))
    for key, model in (("vectorial", direct), ("spline", spline)):
        for grad in (False, True):
            def evaluate():
                for _ in range(10):
                    model.render(pos, grad=grad)
            evaluate()
            result[key + ("_jacobian_ms" if grad else "_forward_ms")] = statistics.median(
                elapsed(evaluate) * 100 for _ in range(5))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("sample", "camera", "calibration", "fitted", "output"):
        p.add_argument(key, type=Path)
    run(p.parse_args())
