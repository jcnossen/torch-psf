"""Fit Noll 5–11 from repeated illumination cycles of a planar bead field.

Use a coarse dense-movie bead-fit.json for optical settings and initialization.
The output retains both signed focus hypotheses in order of residual.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch, tifffile

from torch_psf import VectorialPSF


def data(file, period):
    try:
        raw = tifffile.memmap(file)
    except ValueError:
        raw = tifffile.imread(file)
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
    imgs = np.stack(
        [raw[t : t + period, y : y + side, x : x + side].mean(0) for t in starts]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("coarse_fit", type=Path, help="Dense-movie bead-fit.json")
    parser.add_argument(
        "--period",
        type=int,
        required=True,
        help="Frames per complete illumination cycle",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.period < 1:
        parser.error("--period must be positive")
    initial = json.loads(args.coarse_fit.read_text())
    model = initial["model"]
    modes = dict(zip(model["noll"], model["zcoef"]))
    base = dict(
        coef=[modes.get(j, 0) for j in range(5, 12)],
        z=np.asarray(initial["drift"])[:, 2] / 1000,
    )
    optics = {
        k: model["optics"][k]
        for k in ["pixel_um", "NA", "n_med", "n_imm", "wavelength_um"]
    }
    fits = fit(
        args.movie,
        base,
        args.period,
        optics,
        initial["settings"]["diameterNm"] / 2000,
        args.device,
    )
    args.output.write_text(
        json.dumps(dict(noll=list(range(5, 12)), branches=fits), indent=2)
    )
