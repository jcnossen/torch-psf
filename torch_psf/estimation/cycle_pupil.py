"""Fit Noll 5–11 from repeated illumination cycles of a planar bead field.

Use measured coarse optics when available, or initialize from image gradients.
The output retains both signed focus hypotheses in order of residual.
"""

from pathlib import Path
import numpy as np
import torch, tifffile

from torch_psf import VectorialPSF
from torch_psf.estimation.lateral_registration import (
    lateral_registration,
    registered_frame,
)


def data(file, period):
    if isinstance(file, (str, Path)):
        try:
            raw = tifffile.memmap(file)
        except ValueError:
            raw = tifffile.imread(file)
    else:
        raw = np.asarray(file)
    if raw.ndim != 3:
        raise ValueError(
            "Expected a single-channel TIFF with shape (frames,height,width)"
        )
    cycles = len(raw) // period
    if cycles < 5 or min(raw.shape[1:]) < 96:
        raise ValueError(
            "At least five complete cycles and a 96 by 96 pixel field are required"
        )
    count = min(16, cycles)
    starts = np.rint(np.linspace(0, cycles - 1, count)).astype(int) * period
    side = min(512, *raw.shape[1:])
    y, x = ((v - side) // 2 for v in raw.shape[1:])
    drift = lateral_registration(raw)
    imgs = np.stack(
        [
            np.mean(
                [
                    registered_frame(np.asarray(raw[i], dtype=np.float32), drift[i])[
                        y : y + side, x : x + side
                    ]
                    for i in range(t, t + period)
                ],
                axis=0,
            )
            for t in starts
        ]
    )
    h, w = imgs.shape[1:]
    # Same fluorophore field after averaging a complete illumination cycle.
    F = np.fft.fft2(
        (imgs - imgs.mean((1, 2), keepdims=True))
        * np.outer(np.hanning(h), np.hanning(w)),
        axes=(-2, -1),
    )
    indices = []
    obs = []
    power = []
    for y in range(0, 13):
        for x in range(-12, 13):
            if y == 0 and x < 0 or not 2 <= np.hypot(x, y) <= 12:
                continue
            iy, ix = round(y * h / 32), round(x * w / 32)
            block = F[:, (iy + np.arange(-3, 4)) % h][
                :, :, (ix + np.arange(-3, 4)) % w
            ].reshape(len(imgs), -1)
            cov = block @ block.conj().T / block.shape[1]
            lam, v = np.linalg.eigh(cov)
            indices.append((x, y))
            obs.append(v[:, -1])
            power.append(lam[-1])
    return (
        np.array(indices),
        np.array(obs).T,
        np.array(power),
        (starts + (period - 1) / 2) / (len(raw) - 1),
    )


def fit(file, base, period, optics, height_um, device, offsets=None):
    def tensor(x):
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    freq, raw, power, frac = data(file, period)
    n = raw.shape[0]
    f = tensor(freq / 32)
    obs = torch.as_tensor(raw, device=device, dtype=torch.complex64)
    # Weight frequencies by measured signal; per-frequency object phases and amplitudes are nuisance variables.
    weights = tensor((power / np.percentile(power, 90)).clip(0.03, 1) ** 0.5)
    q = VectorialPSF(
        33,
        **optics,
        npupil=32,
        oversample=2,
        pixel_integration="midpoint",
        zernike_indices=range(5, 12),
        device=device,
    )
    pos = tensor(np.tile([0, 0, height_um], (n, 1)))
    grid = torch.arange(-16, 17, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    dft = torch.exp(
        -2j
        * torch.pi
        * (xx.flatten()[:, None] * f[:, 0] + yy.flatten()[:, None] * f[:, 1])
    )
    if base is None:
        planes = (
            np.asarray(file)[: len(file) // period * period]
            .reshape(-1, period, *file.shape[1:])
            .mean(1)
        )
        features = []
        for image in planes:
            dy, dx = np.gradient(image)
            features.append([np.mean(dx * dx), np.mean(dy * dy), np.mean(dx * dy)])
        features = np.asarray(features)
        if np.allclose(features, features[0]):
            return []
        principal = np.linalg.svd(features - features.mean(0), full_matrices=False)[0][
            :, 0
        ]
        base = dict(
            z=principal / max(np.ptp(principal), 1e-6) * 0.20,
            coef=[0, 0.06, 0, 0, 0, 0, 0],
        )
    basez = np.interp(frac, np.linspace(0, 1, len(base["z"])), base["z"])

    offsets_tensor = tensor(offsets) if offsets is not None else None

    def run(sign):
        z0 = basez * sign
        c0 = np.array(base["coef"]) * np.array([sign, sign, 1, 1, 1, 1, sign])
        with torch.no_grad():
            q.zcoef.copy_(tensor(c0))
        z = torch.nn.Parameter(tensor(z0))
        shift = torch.nn.Parameter(torch.zeros((n, 2), device=device))
        opt = torch.optim.LBFGS(
            [q.zcoef, z, shift],
            max_iter=140,
            tolerance_grad=5e-8,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            if offsets_tensor is None:
                p = q(pos, height_um + z)
            else:
                planes = z[:, None] + offsets_tensor * sign
                p = (
                    q(
                        pos.repeat_interleave(planes.shape[1], dim=0),
                        height_um + planes.flatten(),
                    )
                    .reshape(n, -1, 33, 33)
                    .mean(1)
                )
            H = (p.flatten(1).to(torch.complex64) @ dft) * torch.exp(
                -2j * torch.pi * ((shift - shift[:1]) @ f.T)
            )
            a = (H.conj() * obs).sum(0) / H.abs().square().sum(0).clamp_min(1e-8)
            resid = (H * a - obs).abs().square().sum(0)
            loss = (resid * weights).mean() + 0.002 * (z[1:] - z[:-1]).square().mean()
            loss.backward()
            return loss

        opt.step(closure)
        return dict(
            cost=closure().item(),
            coef=q.zcoef.detach().tolist(),
            z=z.detach().tolist(),
            shift=(shift - shift[:1]).detach().tolist(),
        )

    return sorted([run(1), run(-1)], key=lambda c: c["cost"])


def movie_optics(movie, groups, optics, radius):
    """Blind pupil and initial focus for a repeated planar illumination schedule."""
    if (
        optics is None
        or groups is None
        or "diameter_um" not in optics
        or optics.get("calibrated_psf") is not None
    ):
        return None
    labels = np.asarray(groups)
    if (
        movie.ndim != 3
        or labels.shape != (len(movie),)
        or not len(movie)
        or min(movie.shape[1:]) < 96
    ):
        return None
    repeats = np.flatnonzero(labels[1:] == labels[0]) + 1
    if not len(repeats):
        return None
    period = int(repeats[0])
    if len(movie) // period < 5 or not np.array_equal(
        labels, np.resize(labels[:period], len(labels))
    ):
        return None
    options = {
        key: optics[key]
        for key in ("pixel_um", "NA", "n_med", "n_imm", "wavelength_um")
        if key in optics
    }
    device = optics.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    radius_um = optics["diameter_um"] / 2
    with torch.enable_grad():
        trials = fit(movie, None, period, options, radius_um, device)
    if not trials or trials[0]["cost"] >= 0.08 or np.ptp(trials[0]["z"]) < 0.04:
        return None
    best = trials[0]
    count = len(movie) // period
    centers = (
        np.rint(np.linspace(0, count - 1, min(16, count))) * period + (period - 1) / 2
    )
    focus = np.interp(np.arange(len(movie)), centers, best["z"])
    pupil = VectorialPSF(
        2 * radius + 1,
        **options,
        npupil=32,
        oversample=2,
        zernike_indices=range(5, 12),
        pixel_integration="midpoint",
        device=device,
    )
    with torch.no_grad():
        pupil.zcoef.copy_(torch.as_tensor(best["coef"], device=pupil.dev))
    return dict(
        optics,
        calibrated_psf=pupil,
        focus_um=radius_um + focus.mean(),
        frame_focus_offsets_um=focus - focus.mean(),
        focus_span_um=max(0.325, np.abs(focus - focus.mean()).max() + 0.10),
    )
