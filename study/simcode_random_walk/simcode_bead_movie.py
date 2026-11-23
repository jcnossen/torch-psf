"""Reproduce the SIMCODE browser example and score blind fits against withheld truth."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
import tifffile
import torch
from torch_psf import VectorialPSF

OPTICS = dict(pixel_um=0.10833, NA=1.49, n_imm=1.518, n_med=1.33, wavelength_um=0.7)
MODES = list(range(5, 12))


def model(device):
    return VectorialPSF(
        33,
        **OPTICS,
        npupil=32,
        oversample=2,
        pixel_integration="midpoint",
        zernike_indices=MODES,
        device=device,
    )


def walk(rng, frames, span):
    a = np.r_[0.0, np.cumsum(rng.normal(size=frames - 1))]
    return (a - (a.max() + a.min()) / 2) * span / np.ptp(a)


@torch.no_grad()
def simulate(directory, seed, frames, device, bead_count=12000):
    rng = np.random.default_rng(seed)
    coef = rng.uniform(
        [0.015, 0.10, 0.05, 0.05, 0.015, 0.015, 0.015],
        [0.055, 0.14, 0.09, 0.09, 0.055, 0.055, 0.055],
    ) * rng.choice([-1, 1], 7)
    psf = model(device)
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    psf.zcoef.copy_(tensor(coef))
    size = 400
    x0, y0 = rng.uniform(0, size, (2, bead_count))
    intensity = rng.uniform(2500, 6500, len(x0))
    dx, dy, z = (
        walk(rng, frames, 0.56),
        walk(rng, frames, 0.42),
        0.08 + walk(rng, frames, 0.4),
    )
    radius = 0.020
    nodes = np.vstack([np.eye(3), -np.eye(3)]) * radius * np.sqrt(0.6)
    nodes[:, :2] /= OPTICS["pixel_um"]
    pos = tensor(
        np.concatenate(
            [nodes + [sx / 3, sy / 3, radius] for sy in range(3) for sx in range(3)]
        )
    )
    q = np.pi * 40 / 200
    transfer = 3 * (np.sin(q) - q * np.cos(q)) / q**3
    movie = tifffile.memmap(
        directory / f"{seed}.tif",
        shape=(frames, size, size),
        dtype="float32",
        photometric="minisblack",
    )
    for t in range(frames):
        axis = (t // 3) % 2
        angle = np.deg2rad([13, 104][axis])
        x, y = x0 + dx[t], y0 + dy[t]
        u, v = 2 * x / size - 1, 2 * y / size - 1
        depth = 0.88 + 0.05 * v if axis else 0.84 + 0.06 * u
        phase = (
            2
            * np.pi
            * (108.33 / 200 * (x * np.cos(angle) + y * np.sin(angle)) + t % 3 / 3)
            + 0.25 * u * v
        )
        flux = (
            intensity
            * (1 if axis else 1.12 - 0.16 * u)
            * (1 + transfer * depth * np.cos(phase))
        )
        stamps = (
            psf(pos, tensor(np.full(len(pos), radius + z[t])))
            .reshape(9, 6, 33, 33)
            .mean(1)
            .cpu()
            .numpy()
        )
        ix, iy = np.floor(x * 3 + 0.5).astype(int), np.floor(y * 3 + 0.5).astype(int)
        # Padded impulses retain partial footprints at the edge of the camera.
        impulses = np.zeros((9, size + 2, size + 2), np.float32)
        np.add.at(impulses, ((iy % 3) * 3 + ix % 3, iy // 3 + 1, ix // 3 + 1), flux)
        mu = (
            25
            + fftconvolve(impulses, stamps, mode="same", axes=(1, 2)).sum(0)[1:-1, 1:-1]
        )
        movie[t] = np.maximum(0, np.round(mu + rng.normal(size=mu.shape) * np.sqrt(mu)))
    movie.flush()
    (directory / f"{seed}-truth.json").write_text(
        json.dumps(
            dict(coef=coef.tolist(), z=z.tolist(), dx=dx.tolist(), dy=dy.tolist())
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 83, 244, 519])
    p.add_argument("--beads", type=int, default=12000)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        simulate(args.directory, seed, args.frames, args.device, args.beads)
        print("Rendered", seed, flush=True)
