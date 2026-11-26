"""Compare first-order NAT against the DNA-PAINT v4 PSF on separate field tiles."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch_psf import PSF
from torch_psf.fit import localize
from torch_psf.estimation import fit_nat, split_checkerboard, localize_field


def run(args):
    sample = np.load(args.sample)
    cam = np.load(args.camera)
    xy, frames = sample["xy"], sample["frame"]
    yy = xy[:, 1, None, None] + np.arange(-7, 8)[None, :, None]
    xx = xy[:, 0, None, None] + np.arange(-7, 8)[None, None, :]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    obs = T((sample["adu"] - cam["offset_adu"][yy, xx]) * cam["gain"])
    var = T(cam["readnoise_e"][yy, xx] ** 2)
    parity = split_checkerboard(xy[:, 0], xy[:, 1], 16)
    results = []
    for angle in range(2):
        angle_mask = frames % 6 // 3 == angle
        tr = np.flatnonzero(angle_mask & (parity == 0))
        te = np.flatnonzero(angle_mask & (parity == 1))
        original = PSF.load(args.calibration, roisize=15, angle=angle, device=device)
        original.z_range = (0.0, 0.4)
        start = original.resized(15)
        with torch.no_grad():
            start.zcoef.zero_()
        options = dict(
            norm=(59.5, 59.5, 59.5, 59.5),
            tile=args.tile,
            iters=args.iters,
            global_iters=3,
            dz_um=0.025,
            oversample=3,
            batch=48,
            zstarts=3,
        )
        fit = fit_nat(
            start,
            obs[tr],
            T(xy[tr]),
            backend=args.backend,
            var=var[tr],
            log=lambda it, nll, sec, g: print(angle, it, round(nll), round(sec, 1), flush=True),
            **options,
        )
        nat = fit["psf"]
        if device == "cuda":
            torch.cuda.synchronize()

        def score(model):
            tic = time.monotonic()
            with torch.no_grad():
                if hasattr(model, "bind"):
                    th, nll = localize_field(
                        model, obs[te], T(xy[te]), var[te], iters=25, zstarts=3, batch=48
                    )
                else:
                    th, nll = localize(model, obs[te], var=var[te], iters=25, zstarts=3, batch=48)
            return th, nll.cpu().numpy(), time.monotonic() - tic

        tref, nref, sec_ref = score(original)
        tn, nn, sec_nat = score(nat)
        spline = nat.to_tiled(
            tile=30.0, kernel="bspline", integrated=True, oversample=3, dz_um=0.025, xy_limit=3.0
        )
        ts, ns, sec_spl = score(spline)
        delta = nn - nref
        # Equal xyz and equal flux; remove ROI throughput differences for the shape comparison.
        field = T(np.array([(x, y) for x in [15, 45, 75, 105] for y in [15, 45, 75, 105]]))
        shapes = []
        with torch.no_grad():
            for z in (0.0, 0.1, 0.2, 0.3, 0.4):
                pos = T([[0.0, 0.0, z]]).expand(len(field), -1)
                ref = original.render(pos)
                new = nat.render(pos, xy=field)
                ref = ref / ref.sum((-1, -2), keepdim=True)
                new = new / new.sum((-1, -2), keepdim=True)
                shapes.append(
                    float(((new - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).mean())
                )
        out = dict(
            angle=angle,
            train=len(tr),
            held=len(te),
            fit=fit["report"],
            steps=fit["history"],
            mean_delta_nll=float(delta.mean()),
            se_delta_nll=float(delta.std(ddof=1) / np.sqrt(len(delta))),
            fraction_nat_better=float((delta < 0).mean()),
            mean_z_shift_nm=float(1000 * (tn[:, 2] - tref[:, 2]).mean()),
            normalized_shape_relative_l2=shapes,
            spline_minus_direct_nll=float(np.mean(ns - nn)),
            spline_direct_z_rms_nm=float(1000 * (ts[:, 2] - tn[:, 2]).square().mean().sqrt()),
            seconds=dict(v4=sec_ref, nat_direct=sec_nat, nat_tiled_cold=sec_spl),
        )
        results.append(out)
        print(angle, json.dumps(out), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("sample", "camera", "calibration", "output"):
        p.add_argument(arg, type=Path)
    p.add_argument("--backend", choices=["direct", "bspline"], default="bspline")
    p.add_argument("--tile", type=float, default=60.0)
    p.add_argument("--iters", type=int, default=10)
    run(p.parse_args())
