"""Estimate frame focus from phase-matched power spectra, anchored by cycle diversity."""
import numpy as np
import torch


def frequencies():
    return [
        (x, y)
        for y in range(0, 11)
        for x in range(-10, 11)
        if (y > 0 or x > 0) and 3 <= np.hypot(x, y) <= 10
    ]


def measure(frames):
    freq = np.array(frequencies())
    win = np.outer(np.hanning(64), np.hanning(64))
    power = []
    for frame in frames:
        h, w = frame.shape
        tiles = np.stack(
            [
                frame[y : y + 64, x : x + 64]
                for y in np.linspace(0, h - 64, 4, dtype=int)
                for x in np.linspace(0, w - 64, 4, dtype=int)
            ]
        )
        mean = tiles.mean(axis=(1, 2), keepdims=True)
        ft = np.fft.fft2((tiles - mean) * win)
        vals = abs(ft[:, 2 * freq[:, 1], 2 * freq[:, 0]]) ** 2
        power.append(
            np.log(np.maximum(vals.mean(0) - mean.mean() * (win**2).sum(), 1))
        )
    return np.array(power)


def track(obs, templates, grid, centers, focus, period):
    seed = np.interp(np.arange(len(obs)), centers, focus)
    idx = np.argmin(abs(seed[:, None] - grid), axis=1)
    for _ in range(6):
        for phase in range(period):
            group = np.arange(phase, len(obs), period)
            if not len(group):
                continue
            delta = obs[group] - templates[idx[group]]
            bias = np.median(delta, axis=0)
            diff = obs[group, None, :] - bias - templates[None, :, :]
            # Log power ratios measure changes across repetitions. Remove an exposure gain.
            diff -= diff.mean(axis=2, keepdims=True)
            cost = np.mean(np.minimum(diff**2, 1), axis=2)
            idx[group] = np.argmin(cost, axis=1)
        # Fix the constant focus gauge with the measured cycle centers.
        residual = np.median(
            [
                focus[k]
                - grid[
                    idx[int(c - (period - 1) / 2) : int(c - (period - 1) / 2) + period]
                ].mean()
                for k, c in enumerate(centers)
            ]
        )
        idx = np.clip(idx + round(residual / (grid[1] - grid[0])), 0, len(grid) - 1)
    return grid[idx]


@torch.no_grad()
def focus_trace(psf, obs, centers, focus, period, height=0.02):
    grid = np.arange(np.min(focus) - 0.24, np.max(focus) + 0.241, 0.005)
    pos = psf.zcoef.new_tensor([[0.0, 0.0, height]]).repeat(len(grid), 1)
    im = psf(pos, psf.zcoef.new_tensor(height + grid)).cpu().numpy()
    freq = np.array(frequencies()) / 32
    yy, xx = np.mgrid[-16:17, -16:17]
    dft = np.exp(
        -2j
        * np.pi
        * (xx.ravel()[:, None] * freq[:, 0] + yy.ravel()[:, None] * freq[:, 1])
    )
    templates = np.log(np.maximum(abs(im.reshape(len(grid), -1) @ dft) ** 2, 1e-5))
    return track(obs, templates, grid, centers, focus, period)
