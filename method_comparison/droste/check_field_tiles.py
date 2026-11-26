"""Separate spline interpolation error at a tile centre from the field approximation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_psf import PSF
from torch_psf.estimation import nat_field


@torch.no_grad()
def measure(calibration, fitted):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for result in fitted:
        reference = PSF.load(calibration, roisize=15, angle=result["angle"], device=device)
        reference.z_range = (0.0, 0.4)
        reference.zcoef.zero_()
        fit = result["fit"]
        nat = nat_field(reference, fit["gamma"], fit["norm"])
        tiled = nat.to_tiled(
            tile=30, kernel="bspline", integrated=True, oversample=3, dz_um=0.025, xy_limit=3.0
        )
        pos = torch.tensor(
            [[0.15, -0.25, z] for z in np.linspace(0, 0.4, 5)], device=device, dtype=torch.float32
        )
        errors = []
        for tile, center in enumerate(tiled.centers):
            table = tiled.spline(tile)
            approx = table.render(pos)
            approx = approx / approx.sum((-1, -2), keepdim=True)
            for offset in ((0, 0), (8, 8), (-8, 8), (8, -8), (-8, -8)):
                xy = np.tile(center + offset, (len(pos), 1))
                direct = nat.render(pos, xy=xy)
                direct = direct / direct.sum((-1, -2), keepdim=True)
                relative = (approx - direct).flatten(1).norm(dim=1) / direct.flatten(1).norm(dim=1)
                errors.append((offset == (0, 0), relative.cpu().numpy()))
        row = {"angle": result["angle"]}
        for flag, label in ((True, "center"), (False, "offcenter")):
            values = np.concatenate([v for c, v in errors if c == flag])
            row[label] = {"relative_l2_mean": float(values.mean()), "relative_l2_max": float(values.max())}
        rows.append(row)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("calibration", "fitted", "output"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    rows = measure(args.calibration, json.loads(args.fitted.read_text()))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
