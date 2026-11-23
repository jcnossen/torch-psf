"""Render an overview and a scan preview from a completed bead-gel study."""
import os
import argparse
import json
import subprocess
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scan import PIXEL, BURST, SIZE


def summarize(folder):
    raw = tifffile.memmap(folder / "movie.tif").reshape(-1, BURST, SIZE, SIZE)
    true = np.load(folder / "truth.npz")
    fit = np.load(folder / "fit_3pass.npz")
    metrics = json.loads((folder / "metrics_3pass.json").read_text())
    schedule = np.loadtxt(folder / "scan.csv", skiprows=1, delimiter=",")[::BURST]
    nominal = schedule[:, 1]
    main = schedule[:, 3] >= 0 if schedule.shape[1] > 3 else np.ones(len(schedule), bool)
    xyz = true["xyz"] * [PIXEL, PIXEL, 1]
    est = fit["xyz"] * [PIXEL, PIXEL, 1]
    distance = np.linalg.norm(xyz[:, None] - est[None], axis=2)
    i, j = linear_sum_assignment(np.where(distance <= 0.25, distance, 1e4))
    select = distance[i, j] <= 0.25
    i, j = i[select], j[select]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    a, b, c, d = axes.flat
    a.scatter(xyz[:, 0], xyz[:, 2], s=23, facecolors="none", edgecolors="#537877", label="True")
    a.scatter(est[:, 0], est[:, 2], s=9, c="#b15329", marker="+", label="Fit")
    a.set(
        xlabel="X (µm)", ylabel="Depth in gel (µm)", title=f"Bead positions · {len(i)} / {len(xyz)} matched"
    )
    a.legend()
    colors = ["#267a79", "#ba5a27", "#5b598b"]
    n = len(fit["drift_um"])
    x = np.arange(n)
    for k, color in enumerate(colors):
        b.plot(x, 1000 * true["drift_um"][:n, k], ls="--", color=color, lw=2)
        b.plot(x, 1000 * fit["drift_um"][:, k], color=color, lw=1.1, label="XYZ"[k])
    b.set(xlabel="Six-exposure burst", ylabel="Drift (nm)", title="Drift · dashed truth, solid fit")
    b.legend(ncol=3)
    c.plot(x, nominal, color="#538185", lw=1)
    if (~main).any():
        c.scatter(x[~main], nominal[~main], s=5, color="#b15329", label="Reference planes")
        c.legend()
    c.set(xlabel="Six-exposure burst", ylabel="Commanded focus (µm)", title="10 µm scan and return visits")
    # Show a typical matched trace using the median normalized intensity RMSE.
    flux = true["flux"]
    target = (flux[:n].reshape(-1, flux.shape[-1])[:,i] if flux.ndim == 3 else np.tile(flux[:,i], (n, 1)))
    observed = fit["flux"][:, j]
    valid = fit["usable"][:, j]
    rmse = np.sqrt(np.sum(((target - observed) / target.mean(0)) ** 2 * valid, axis=0) / valid.sum(0))
    mid = np.argsort(rmse)[len(rmse) // 2]
    active = np.flatnonzero(valid[:, mid] & np.repeat(main, BURST))
    # Draw two entire illumination cycles at the best focused burst and next scan plane.
    best = (
        active[
            np.argmin(np.abs(np.repeat(nominal + fit["drift_um"][:, 2], BURST)[active] - est[j[mid], 2]))
        ]
        // BURST
    )
    neighbours = np.flatnonzero(main & (x >= best))[:2]
    frames = (neighbours[:, None] * BURST + np.arange(BURST)).ravel()
    d.plot(np.arange(len(frames)), target[frames, mid], ".--", label="True", color="#537877")
    d.plot(np.arange(len(frames)), observed[frames, mid], ".-", label="Fitted", color="#b15329")
    d.set(
        xlabel="Exposure in two consecutive scan bursts",
        ylabel="Bead photons",
        title="Example near-focus intensity trace",
    )
    d.legend()
    fig.suptitle("TorchPSF · 3D bead gel with moving focus", fontsize=16)
    fig.savefig(folder / "overview.png", dpi=150)
    plt.close(fig)
    # The preview shows one raw exposure per scan plane; the TIFF contains every exposure.
    low, high = np.quantile(raw[::7, 0], [0.01, 0.995])
    cmd = [
        os.environ.get("FFMPEG", "ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "384x414",
        "-r",
        "18",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(folder / "scan-preview.mp4"),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for index in np.flatnonzero(main):
            gray = (255 * np.clip((raw[index, 0] - low) / (high - low), 0, 1) ** 0.7).astype("uint8")
            im = Image.new("RGB", (384, 414), "#14282a")
            im.paste(Image.fromarray(gray).resize((384, 384)), (0, 30))
            ImageDraw.Draw(im).text(
                (8, 8),
                f"Scan focus {nominal[index]:.1f} um  |  burst {index+1}  |  exposure 1/6",
                fill="white",
            )
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("ffmpeg failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
