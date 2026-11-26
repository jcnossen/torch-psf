"""Compare nested six- and thirteen-parameter pupils on held-out experimental field tiles."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_psf import PSF, io
from torch_psf.estimation import fit_nat, localize_field, nat_field, split_checkerboard


def paired_summary(delta, xy):
    cells, idx = np.unique(xy // 16, axis=0, return_inverse=True)
    sums, counts = np.bincount(idx, weights=delta), np.bincount(idx)
    # Paired spatial cluster bootstrap: resample whole occupied tiles.
    draws = np.random.default_rng(42).integers(len(cells), size=(10000, len(cells)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(
        mean_delta_nll=float(delta.mean()),
        tile_bootstrap_95=np.quantile(means, [0.025, 0.975]).tolist(),
        fraction_better=float(np.mean(delta < 0)),
        n_tiles=len(cells),
        n_tiles_better=int(sum(sums < 0)),
        tiles=[
            dict(x=int(x), y=int(y), n=int(n), delta_sum=float(s))
            for (x, y), n, s in zip(cells, counts, sums)
        ],
    )


@torch.no_grad()
def export_axis(psf, path):
    # This file represents one illumination axis. Keep its selected dipole and
    # blur and omit the original multi-axis lookup that could override them.
    model = psf.resized(psf.roisize)
    model.base.dipM_angle = None
    model.base.blur_angle = None
    io.save_zip(path, model)
    restored = PSF.load(path, device=model.dev)
    pos = torch.tensor([[0.12, -0.23, 0.05], [-0.25, 0.07, 0.35]], device=model.dev)
    xy = torch.tensor([[15, 15], [105, 105]], device=model.dev)
    torch.testing.assert_close(restored.render(pos, xy=xy), psf.render(pos, xy=xy), rtol=2e-5, atol=1e-7)


def observations(path, camera, device):
    sample, cam = np.load(path), np.load(camera)
    xy, frame = sample["xy"], sample["frame"]
    y = xy[:, 1, None, None] + np.arange(-7, 8)[None, :, None]
    x = xy[:, 0, None, None] + np.arange(-7, 8)[None, None, :]
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    obs = tensor((sample["adu"] - cam["offset_adu"][y, x]) * cam["gain"])
    var = tensor(cam["readnoise_e"][y, x] ** 2)
    return xy, frame, obs, var


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    xy, frame, obs, var = observations(args.sample, args.camera, device)
    parity = split_checkerboard(*xy.T, 16)
    records = []
    for r in json.loads(args.fitted.read_text()):
        angle = r["angle"]
        tr = np.flatnonzero((frame % 6 // 3 == angle) & (parity == 0))
        te = np.flatnonzero((frame % 6 // 3 == angle) & (parity == 1))
        assert (len(tr), len(te)) == (r["train"], r["held"])
        base = PSF.load(args.calibration, roisize=15, angle=angle, device=device)
        base.z_range = (0.0, 0.4)
        with torch.no_grad():
            base.zcoef.zero_()
        norm = r["fit"]["norm"]
        varying = nat_field(base, r["fit"]["gamma"], norm)
        constant = fit_nat(
            base,
            obs[tr],
            tensor(xy[tr]),
            norm=norm,
            backend="bspline",
            tile=60,
            iters=10,
            global_iters=3,
            dz_um=0.025,
            oversample=3,
            batch=48,
            zstarts=3,
            var=var[tr],
            field_varying=False,
            log=lambda it, nll, sec, g: print(angle, it, round(nll), flush=True),
        )
        psfs = [constant["psf"], varying]
        if args.export:
            args.export.mkdir(parents=True, exist_ok=True)
            for label, psf in zip(("constant", "nat"), psfs):
                export_axis(psf, args.export / f"dna_paint_axis{angle}_{label}.psf.zip")
        with torch.no_grad():
            scores = []
            for psf in psfs:
                th, loss = localize_field(
                    psf, obs[te], tensor(xy[te]), var[te], iters=25, zstarts=3, batch=48
                )
                scores.append((th.cpu().numpy(), loss.double().cpu().numpy()))
        (tc, lc), (tf, lf) = scores
        # The earlier fit used the same exact field model and held-out observations.
        points = tensor([[x, y] for x in (15, 45, 75, 105) for y in (15, 45, 75, 105)])
        modes = [4, 5, 6, 7, 8, 11]
        coeff = varying.zernikes_at(points)[:, [varying.zidx.index(m) for m in modes]]
        row = dict(
            axis=angle,
            train=len(tr),
            held=len(te),
            constant_fit=constant["report"],
            field_fit=r["fit"],
            constant_steps=constant["history"],
            **paired_summary(lf - lc, xy[te]),
            mean_z_difference_nm=float(1000 * (tf[:, 2] - tc[:, 2]).mean()),
            z_difference_rms_nm=float(1000 * np.sqrt(np.mean((tf[:, 2] - tc[:, 2]) ** 2))),
            field_peak_to_peak_waves=dict(zip(map(str, modes), np.ptp(coeff, axis=0).tolist())),
        )
        records.append(row)
        print(angle, row["mean_delta_nll"], row["tile_bootstrap_95"], flush=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


@torch.no_grad()
def validate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xy, frame, obs, var = observations(args.sample, args.camera, device)
    train_windows = [0, 6000, 12000, 18000, 24000, 29400]
    windows = frame // 600 * 600
    assert not set(train_windows).intersection(windows.tolist())
    records = []
    for fit in json.loads(args.fitted.read_text()):
        a = fit["axis"]
        groups = []
        rng = np.random.default_rng(410 + a)
        for w in np.unique(windows):
            eligible = np.flatnonzero((windows == w) & (frame % 6 // 3 == a))
            groups.append(rng.choice(eligible, 64, replace=False))
        ix = np.concatenate(groups)
        base = PSF.load(args.calibration, roisize=15, angle=a, device=device)
        base.z_range = (0.0, 0.4)
        base.zcoef.zero_()
        predictions = []
        for key in ("constant_fit", "field_fit"):
            gamma = fit[key]["gamma"]
            model = nat_field(base, gamma, fit[key]["norm"])
            th, loss = localize_field(
                model,
                obs[ix],
                torch.as_tensor(xy[ix], device=device),
                var[ix],
                iters=25,
                zstarts=3,
                batch=48,
            )
            predictions.append((th.cpu().numpy(), loss.double().cpu().numpy()))
        (tc, lc), (tf, lf) = predictions
        delta = lf - lc
        means = [float(delta[windows[ix] == w].mean()) for w in np.unique(windows)]
        row = dict(
            axis=a,
            n=len(ix),
            eligible=int(sum(frame % 6 // 3 == a)),
            windows=np.unique(windows).tolist(),
            window_means=means,
            **paired_summary(delta, xy[ix]),
            mean_z_difference_nm=float(1000 * (tf[:, 2] - tc[:, 2]).mean()),
        )
        records.append(row)
        print(a, row["mean_delta_nll"], means, flush=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("sample", "camera", "calibration", "fitted", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--export", type=Path)
    p.add_argument(
        "--validate", action="store_true", help="Score two frozen pupils on separate exposure windows."
    )
    args = p.parse_args()
    (validate if args.validate else run)(args)
