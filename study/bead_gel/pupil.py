"""Recover a shared vectorial pupil from registered local stacks of detected beads."""

import numpy as np
import tifffile
import torch
from scipy import ndimage
from torch_psf import VectorialPSF

MODES = list(range(5, 12))


def patches(movie, scan, gel):
    schedule = np.loadtxt(scan, delimiter=",", skiprows=1)[::6]
    ordinary = (schedule[:, 3] >= 0) & (schedule[:, 3] < 101)
    raw = tifffile.memmap(movie).reshape(-1, 6, gel.SIZE, gel.SIZE)
    obs = raw[ordinary].mean(1, dtype=np.float32)
    seed = gel.detect(obs)
    r = 6
    dz, dy, dx = np.mgrid[-5:6, -r : r + 1, -r : r + 1]
    samples = []
    # Align 3D peaks before computing a robust ensemble PSF.
    smooth = ndimage.gaussian_filter(obs, (0.75, 0.65, 0.65))
    for x, y, z in seed:
        index = np.array([round(z * 10), round(y), round(x)], dtype=int)
        if np.any(index < [6, r + 1, r + 1]) or np.any(
            index >= np.array(obs.shape) - [6, r + 1, r + 1]
        ):
            continue
        offset = []
        for j in range(3):
            delta = np.eye(3, dtype=int)[j]
            a, b, c = [smooth[tuple(index + v * delta)] for v in [-1, 0, 1]]
            offset.append(np.clip(0.5 * (a - c) / min(-1e-6, a + c - 2 * b), -0.5, 0.5))
        center = index + offset
        patch = ndimage.map_coordinates(
            obs, [dz + center[0], dy + center[1], dx + center[2]], order=1
        )
        bg = np.median(patch[:, np.maximum(abs(dx[0]), abs(dy[0])) == r], axis=1)
        patch -= bg[:, None, None]
        peak = patch[5, r, r]
        if peak > 40:
            samples.append(patch / peak)
    if len(samples) < 5:
        raise ValueError("Fewer than five usable through-focus bead patches")
    return np.median(samples, axis=0), len(samples)


def fit_pupil(directory, gel):
    target, n = patches(directory / "movie.tif", directory / "scan.csv", gel)
    device = gel.DEVICE
    model = VectorialPSF(
        13,
        pixel_um=gel.PIXEL,
        NA=1.2,
        n_imm=1.33,
        n_med=1.33,
        wavelength_um=0.70,
        zernike_indices=MODES,
        npupil=64,
        oversample=2,
        device=device,
    )
    obs = torch.as_tensor(target, device=device, dtype=torch.float32)
    grid = torch.arange(-5, 6, device=device) * 0.1
    offset = torch.nn.Parameter(torch.zeros(3, device=device))
    opt = torch.optim.LBFGS(
        [model.zcoef, offset], max_iter=90, line_search_fn="strong_wolfe"
    )

    # Average resampling blur from subpixel peak alignment and the finite sphere.
    def closure():
        opt.zero_grad()
        pos = torch.stack(
            [
                offset[0].expand_as(grid),
                offset[1].expand_as(grid),
                torch.zeros_like(grid) + 5,
            ],
            1,
        )
        pred = model(pos, grid + 5 + offset[2])
        kernel = pred.new_tensor([0.10, 0.80, 0.10])
        pred = torch.nn.functional.conv2d(
            pred[:, None], (kernel[:, None] * kernel[None, :])[None, None], padding=1
        )[:, 0]
        pred = pred[:, 1:-1, 1:-1]
        data = obs[:, 1:-1, 1:-1]
        pred = pred - pred.mean((1, 2), keepdim=True)
        data = data - data.mean((1, 2), keepdim=True)
        scale = (pred * data).sum() / pred.square().sum().clamp_min(1e-9)
        loss = torch.nn.functional.smooth_l1_loss(pred * scale, data, beta=0.04)
        loss.backward()
        return loss

    opt.step(closure)
    return dict(zip(MODES, model.zcoef.detach().cpu().tolist())), n


@torch.no_grad()
def psf_error(actual, estimate, gel):
    model = VectorialPSF(
        33,
        pixel_um=gel.PIXEL,
        NA=1.2,
        n_imm=1.33,
        n_med=1.33,
        wavelength_um=0.7,
        zernike_indices=MODES,
        npupil=96,
        oversample=2,
        device=gel.DEVICE,
    )
    focus = torch.linspace(4.45, 5.55, 23, device=gel.DEVICE)
    pos = torch.zeros((23, 3), device=gel.DEVICE)
    pos[:, 2] = 5

    def render(c):
        model.set_zernike(c)
        return model(pos, focus)

    truth = render(actual)
    return float(torch.linalg.norm(render(estimate) - truth) / torch.linalg.norm(truth))
