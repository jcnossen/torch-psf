"""Feasibility study: recover a stationary 3D bead gel from repeated known focus scans.

Run: python study/bead_gel/scan.py --out path/to/output
Distances in micrometres except emitter x/y in camera pixels. CUDA recommended.
"""
import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from torch_psf import VectorialPSF

PIXEL = 0.10833
SIZE = 64
SIDE = 256  # half-pixel model grid; covers all camera/emitter separations
BEAD_DIAMETER = 0.1
BURST = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def optical_bank(step):
    """Index-matched optics make the PSF shift-invariant in z. Axis 0 is focus minus bead depth."""
    nz = round(25.6 / step)
    stage = (torch.arange(nz, device=DEVICE) - nz // 2) * step
    model = VectorialPSF(
        129,
        pixel_um=PIXEL,
        NA=1.2,
        n_imm=1.33,
        n_med=1.33,
        wavelength_um=0.70,
        zernike_indices=(6,),
        npupil=96,
        oversample=2,
        device=DEVICE,
    ).set_zernike({6: 0.075})
    bank = torch.empty((nz, SIDE, SIDE), device=DEVICE)
    for start in range(0, nz, 8):
        z = stage[start : start + 8]
        for iy in range(2):
            for ix in range(2):
                pos = torch.zeros((len(z), 3), device=DEVICE)
                pos[:, 0], pos[:, 1], pos[:, 2] = -ix / 2, -iy / 2, 5
                bank[start : start + len(z), iy::2, ix::2] = model(pos, 5 + z)[:, :128, :128]
    # Convolution with a uniform fluorescent sphere, using its analytic 3D transfer function.
    kz = torch.fft.fftfreq(nz, step, device=DEVICE)
    ky = torch.fft.fftfreq(SIDE, PIXEL / 2, device=DEVICE)
    kx = torch.fft.rfftfreq(SIDE, PIXEL / 2, device=DEVICE)
    q = np.pi * BEAD_DIAMETER * torch.sqrt(kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx**2)
    transfer = torch.where(q < 0.05, 1 - q * q / 10, 3 * (q.sin() - q * q.cos()) / q.clamp_min(1e-8) ** 3)
    return torch.fft.irfftn(torch.fft.rfftn(bank) * transfer, s=bank.shape).clamp_min_(0)


def camera_grid():
    y, x = torch.meshgrid(
        torch.arange(SIZE, device=DEVICE), torch.arange(SIZE, device=DEVICE), indexing="ij"
    )
    return x.float(), y.float()


def kernels(bank, step, xyz, focus, shift):
    """Return complete camera-plane images for all beads, including out-of-focus neighbours."""
    x, y = camera_grid()
    xx = 2 * (x[None] - xyz[:, 0, None, None] - shift[0]) + SIDE // 2
    yy = 2 * (y[None] - xyz[:, 1, None, None] - shift[1]) + SIDE // 2
    zz = ((focus - xyz[:, 2]) / step + len(bank) // 2)[:, None, None].expand_as(xx)
    grid = torch.stack(
        (2 * xx / (SIDE - 1) - 1, 2 * yy / (SIDE - 1) - 1, 2 * zz / (len(bank) - 1) - 1), dim=-1
    )
    return F.grid_sample(bank[None, None], grid[None], align_corners=True)[0, 0].flatten(1)


def simulate(out, seed, count, references):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xyz = rng.uniform([0, 0, 0.05], [SIZE - 1, SIZE - 1, 9.95], (count, 3))
    up = np.linspace(0, 10, 101)
    stage = np.concatenate([up, up[::-1], up])
    scan_index = np.arange(len(stage))
    if references:
        # Reference triplets revisit the same gel slab throughout the acquisition.
        # Camera exposures start only after the stage has settled at each commanded z.
        indexed = []
        for i, z in enumerate(stage):
            if i % 12 == 0:
                indexed.extend([(-1, 4.7), (-1, 5.0), (-1, 5.3)])
            indexed.append((i, z))
        scan_index, stage = np.array(indexed).T
    t = np.linspace(0, 1, len(stage))
    motion = np.stack(
        [
            0.19 * np.sin(2 * np.pi * t) + 0.12 * t,
            0.17 * (np.cos(2 * np.pi * t) - 1) + 0.05 * t,
            0.36 * np.sin(2 * np.pi * t) + 0.13 * np.sin(4 * np.pi * t) + 0.16 * t,
        ],
        axis=1,
    )
    # Six repeated illumination states. Neither the simulator phases nor bead brightnesses enter the fit.
    phase = rng.uniform(0, 2 * np.pi, (2, count))
    bright = rng.uniform(4500, 7500, count)
    flux = np.stack(
        [bright * (1 + 0.72 * np.cos(phase[j // 3] + j % 3 * 2 * np.pi / 3)) for j in range(BURST)]
    )
    x, y = camera_grid()
    bg = 24 + 5 * x / (SIZE - 1) + 3 * y / (SIZE - 1)
    bank = optical_bank(0.05)
    movie = np.empty((len(stage), BURST, SIZE, SIZE), np.uint16)
    coords = tensor(xyz)
    flux_gpu = tensor(flux)
    for i, z in enumerate(stage):
        k = kernels(bank, 0.05, coords, z + motion[i, 2], tensor(motion[i, :2] / PIXEL))
        mu = (flux_gpu @ k).reshape(BURST, SIZE, SIZE) + bg
        movie[i] = torch.poisson(mu).cpu().numpy().astype(np.uint16)
    # An exact decimal ratio keeps TIFF resolution fields within signed 32-bit range for Fiji.
    scale = 1 / Fraction(str(PIXEL))
    resolution = (scale.numerator, scale.denominator)
    tifffile.imwrite(
        out / "movie.tif",
        movie.reshape(-1, SIZE, SIZE),
        photometric="minisblack",
        imagej=True,
        metadata={"axes": "TYX", "unit": "um"},
        resolution=(resolution, resolution),
    )
    np.savez_compressed(out / "truth.npz", xyz=xyz, drift_um=motion, flux=flux)
    np.savetxt(
        out / "scan.csv",
        np.column_stack(
            (
                np.arange(movie.shape[0] * BURST),
                np.repeat(stage, BURST),
                np.tile(np.arange(BURST), len(stage)),
                np.repeat(scan_index, BURST),
            )
        ),
        delimiter=",",
        header="frame,nominal_focus_um,condition,scan_index",
        comments="",
        fmt=["%d", "%.3f", "%d", "%d"],
    )


def detect(obs):
    """Seed from one 3D pass, with no separation constraint across the projected XY field."""
    v = ndimage.gaussian_filter(obs[:101], (0.85, 0.65, 0.65))
    signal = v - ndimage.gaussian_filter(v, (3, 3, 3))
    peaks = (signal == ndimage.maximum_filter(signal, (5, 3, 3))) & (signal > 35)
    z, y, x = np.nonzero(peaks)
    return np.stack((x, y, z * 0.1), axis=1).astype(np.float32)


def drift_basis(n):
    # Seven linear intervals on a 303-burst scan. Anchor the first displacement at zero.
    num = max(4, round((n - 1) / 38) + 1)
    knots = np.linspace(0, n - 1, num)
    return tensor(np.stack([np.interp(np.arange(n), knots, b) for b in np.eye(num)], axis=1)[:, 1:])


class Scene:
    """Splat the bead volume, convolve with the 3D vectorial kernel, then sample the moving camera planes."""

    def __init__(self, bank, stage):
        self.bank = torch.fft.rfftn(torch.fft.ifftshift(bank))
        self.shape = bank.shape
        self.stage = tensor(stage)
        self.basis = drift_basis(len(stage))
        self.x, self.y = camera_grid()
        self.background_basis = torch.stack(
            [torch.ones_like(self.x), self.x / (SIZE - 1), self.y / (SIZE - 1)]
        )
        self.offset = (SIDE - 2 * SIZE) // 2
        self.neighbours = torch.cartesian_prod(*[torch.arange(2, device=DEVICE)] * 3)

    def render(self, xyz, photons, knots, bg):
        nz, ny, nx = self.shape
        pos = torch.stack([xyz[:, 2] / 0.1, xyz[:, 1] * 2 + self.offset, xyz[:, 0] * 2 + self.offset], 1)
        cell = pos.floor().long()
        frac = pos - cell
        idx = (cell[:, None] + self.neighbours) % tensor(self.shape).long()
        weights = torch.where(self.neighbours.bool(), frac[:, None], 1 - frac[:, None]).prod(2)
        flat = (idx[:, :, 0] * ny + idx[:, :, 1]) * nx + idx[:, :, 2]
        emit = xyz.new_zeros(nz * ny * nx).scatter_add_(
            0, flat.flatten(), (weights * photons[:, None]).flatten()
        )
        vol = torch.fft.irfftn(torch.fft.rfftn(emit.view(self.shape)) * self.bank, s=self.shape)
        drift = self.basis @ knots
        xx = 2 * (self.x[None] - drift[:, 0, None, None] / PIXEL) + self.offset
        yy = 2 * (self.y[None] - drift[:, 1, None, None] / PIXEL) + self.offset
        # Shift the z origin into the middle of the grid to allow negative stage positions.
        vol = vol.roll(nz // 4, 0)
        zz = ((self.stage + drift[:, 2]) / 0.1 + nz // 4)[:, None, None].expand_as(xx)
        grid = torch.stack((2 * xx / (nx - 1) - 1, 2 * yy / (ny - 1) - 1, 2 * zz / (nz - 1) - 1), -1)
        img = F.grid_sample(vol[None, None], grid[None], align_corners=True)[0, 0]
        return img + torch.einsum("i,ihw->hw", bg, self.background_basis), drift


def residual_seeds(scene, residual, drift, previous):
    # Align residuals in object coordinates before looking for overlooked 3D peaks.
    volume = np.zeros((101, SIZE, SIZE), np.float32)
    weights = np.zeros(101)
    focus = scene.stage.cpu().numpy() + drift[:, 2]
    for image, z, d in zip(residual, focus, drift):
        index = int(np.rint(z / 0.1))
        if 0 <= index < 101:
            volume[index] += ndimage.shift(image, (-d[1] / PIXEL, -d[0] / PIXEL), order=1, mode="nearest")
            weights[index] += 1
    volume /= np.maximum(weights, 1)[:, None, None]
    candidates = detect(volume)
    if not len(candidates):
        return candidates
    distance = np.linalg.norm((candidates[:, None] - previous[None]) * [PIXEL, PIXEL, 0.65], axis=2)
    return candidates[distance.min(1) > 0.18]


def fit_geometry(scene, avg, steps, first_up, initial=None):
    seed = detect(avg[first_up]) if initial is None else initial[0]
    print("3D candidates:", len(seed), flush=True)
    xyz = tensor(seed).requires_grad_()
    logs = tensor(np.zeros(len(seed))).requires_grad_()
    knots = tensor(np.zeros((scene.basis.shape[1], 3))).requires_grad_()
    bg = tensor([25, 0, 0]).requires_grad_()
    if initial is not None:
        with torch.no_grad():
            logs.copy_(tensor(np.log(initial[1] / 5500)))
            knots.copy_(tensor(initial[2]))
            bg.copy_(tensor(initial[3]))
    obs = tensor(avg)
    opt = torch.optim.Adam(
        [
            {"params": [xyz], "lr": 0.045},
            {"params": [logs], "lr": 0.04},
            {"params": [knots], "lr": 0.012},
            {"params": [bg], "lr": 0.45},
        ]
    )
    if initial is not None:
        for g in opt.param_groups:
            g["lr"] *= 0.35
    for it in range(steps):
        if it in (steps * 2 // 3, steps * 5 // 6):
            for g in opt.param_groups:
                g["lr"] *= 0.4
        mu, drift = scene.render(xyz, 5500 * logs.exp(), knots, bg)
        mu = mu.clamp_min(1)
        # Poisson deviance of burst means; second differences suppress unobservable fast drift.
        loss = (mu - obs + obs * torch.log(obs.clamp_min(1) / mu)).mean() + 0.5 * knots.diff(
            n=2, dim=0
        ).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 25 == 0:
            print("geometry", it, round(loss.item(), 4), flush=True)
    with torch.no_grad():
        photons = (5500 * logs.exp()).cpu().numpy()
        keep = photons > 800
        points = xyz.cpu().numpy().copy()
        motions = (scene.basis @ knots).cpu().numpy()
        bg_np = bg.cpu().numpy().copy()
        new = np.empty((0, 3))
        if initial is None:
            mu, _ = scene.render(xyz, 5500 * logs.exp(), knots, bg)
            new = residual_seeds(scene, (obs - mu).cpu().numpy(), motions, points[keep])
        state = (
            np.vstack([points[keep], new]),
            np.r_[photons[keep], np.full(len(new), 1800)],
            knots.cpu().numpy().copy(),
            bg_np,
        )
    if len(new):
        print("Added residual candidates:", len(new), flush=True)
        return fit_geometry(scene, avg, max(70, steps // 4), first_up, state)
    return points[keep], photons[keep], motions, bg_np


@torch.no_grad()
def photometry(bank, movie, stage, xyz, photons, drift, bg):
    """Profile each exposure on all beads together; report in-focus usable rows separately."""
    xy = tensor(xyz)
    initial = tensor(photons)
    x, y = camera_grid()
    background = (bg[0] + bg[1] * x / (SIZE - 1) + bg[2] * y / (SIZE - 1)).flatten()
    result = np.empty((len(stage), BURST, len(xyz)), np.float32)
    for i, z in enumerate(stage):
        k = kernels(bank, 0.1, xy, z + drift[i, 2], tensor(drift[i, :2] / PIXEL))
        obs = tensor(movie[i]).flatten(1)
        # Regularize poorly observed tails toward their mean brightness and solve
        # a Poisson-weighted joint linear system, preserving every near-focus neighbour.
        weight = (initial @ k + background).clamp_min(2).reciprocal()
        gram = (k * weight) @ k.T
        prior = 1 / (0.8 * initial).clamp_min(100).square()
        gram.diagonal().add_(prior)
        rhs = ((obs - background) * weight) @ k.T + prior * initial
        result[i] = torch.linalg.solve(gram, rhs.T).T.clamp_min(0).cpu().numpy()
    visible = np.abs(stage[:, None] + drift[:, 2, None] - xyz[None, :, 2]) <= 0.55
    return result, visible


def recover(out, passes, steps):
    schedule = np.loadtxt(out / "scan.csv", skiprows=1, delimiter=",")[::BURST]
    scan_index = schedule[:, 3] if schedule.shape[1] > 3 else np.arange(len(schedule))
    stop = np.flatnonzero(scan_index == 101 * passes - 1)[0] + 1
    raw = tifffile.memmap(out / "movie.tif").reshape(-1, BURST, SIZE, SIZE)[:stop]
    stage = schedule[:stop, 1]
    first_up = (scan_index[:stop] >= 0) & (scan_index[:stop] < 101)
    avg = raw.mean(1, dtype=np.float32)
    bank = optical_bank(0.1)
    scene = Scene(bank, stage)
    xyz, photons, drift, bg = fit_geometry(scene, avg, steps, first_up)
    del scene
    flux, visible = photometry(bank, raw, stage, xyz, photons, drift, bg)
    np.savez_compressed(
        out / f"fit_{passes}pass.npz",
        xyz=xyz,
        drift_um=drift,
        nominal_focus_um=stage,
        flux=flux.reshape(-1, len(xyz)),
        usable=np.repeat(visible, BURST, axis=0),
        background_plane=bg,
    )


def evaluate(out, passes):
    truth = np.load(out / "truth.npz")
    fit = np.load(out / f"fit_{passes}pass.npz")
    xyz, est = truth["xyz"], fit["xyz"]
    delta = (xyz[:, None] - est[None]) * [PIXEL, PIXEL, 1]
    dist = np.linalg.norm(delta, axis=2)
    # Maximum-cardinality assignment inside a predeclared 250 nm 3D gate.
    i, j = linear_sum_assignment(np.where(dist <= 0.25, dist, 1e4))
    ok = dist[i, j] <= 0.25
    i, j = i[ok], j[ok]
    err = delta[i, j] * 1000
    d = fit["drift_um"]
    ref = truth["drift_um"][: len(d)]
    signal = np.tile(truth["flux"][:, i], (len(d), 1))
    estimate = fit["flux"][:, j]
    valid = fit["usable"][:, j]
    e = (estimate - signal) / signal.mean(0)
    projected = np.linalg.norm((xyz[:, None, :2] - xyz[None, :, :2]) * PIXEL, axis=2)
    np.fill_diagonal(projected, np.inf)
    close = projected.min(1) < 0.25
    stats = dict(
        passes=passes,
        true_beads=len(xyz),
        candidates=len(est),
        matched=len(i),
        recall=len(i) / len(xyz),
        precision=len(i) / len(est),
        rms_xyz_nm=np.sqrt((err**2).mean(0)).tolist(),
        drift_rms_xyz_nm=(np.sqrt(((d - ref) ** 2).mean(0)) * 1000).tolist(),
        usable_trace_relative_rms=float(np.sqrt(np.mean(e[valid] ** 2))),
        usable_fraction=float(valid.mean()),
        projected_close=int(close.sum()),
        projected_close_matched=int(close[i].sum()),
    )
    (out / f"metrics_{passes}pass.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument(
        "--references",
        action="store_true",
        help="Add repeat reference triplets at 4.7, 5.0, 5.3 um every 12 scan planes",
    )
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--passes", nargs="+", type=int, choices=[1, 2, 3], default=[1, 3])
    ap.add_argument("--reuse-movie", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("device:", DEVICE, flush=True)
    if not args.reuse_movie:
        simulate(args.out, args.seed, args.count, args.references)
    for passes in args.passes:
        tic = time.monotonic()
        recover(args.out, passes, args.steps)
        if (args.out / "truth.npz").exists():
            evaluate(args.out, passes)
        print("fit seconds:", round(time.monotonic() - tic, 1), flush=True)
