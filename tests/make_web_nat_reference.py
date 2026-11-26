"""Generate the small PyTorch oracle used by the WebGPU NAT parity test."""
import json
from pathlib import Path

import torch
from torch_psf import VectorialPSF
from torch_psf.estimation import fit_nat, fit_field_knots, localize_field, nat_field, nat_polynomial
from torch_psf.estimation.nat import NAT_MODES
from torch_psf.field import FieldZernikes
from torch_psf.spline import SplinePSF


def reference(data=None):
    noll = [5, 6, 7, 8, 11, 4]
    optics = dict(pixel_um=0.108, NA=1.45, wavelength_um=0.68, n_imm=1.518, n_med=1.33)
    base = VectorialPSF(
        7,
        npupil=16,
        oversample=2,
        zernike_indices=noll,
        pixel_integration="midpoint",
        focus_um=0.1,
        z_range=(0.0, 0.2),
        device="cpu",
        **optics,
    ).set_zernike({6: 0.08})
    gamma = [0.007, 0.01, -0.008, 0.004, 0.07, -0.02, 0.015, -0.01, 0.008, 0.004, -0.003, 0.007, 0.01]
    norm = (48, 48, 48, 48)
    true = nat_field(base, gamma, norm)
    rng = torch.Generator().manual_seed(321)
    xy = torch.tensor([[x, y] for x in (16, 48, 80) for y in (16, 48, 80) for _ in range(4)])
    pos = torch.rand((len(xy), 3), generator=rng) * torch.tensor([0.8, 0.8, 0.18]) + torch.tensor(
        [-0.4, -0.4, 0.01]
    )
    with torch.no_grad():
        image = true.render(pos, xy=xy)
        if data is None:
            data = torch.poisson(2500 * image + 9, generator=rng)
        else:
            data = torch.tensor(data, dtype=torch.float32).reshape_as(image)
    result = fit_nat(
        base, data, xy, norm=norm, tile=32, backend="web", oversample=2, dz_um=0.04, iters=3, var=2.0
    )
    grid = dict(kernel="catmull", integrated=False, oversample=2, dz_um=0.04, zpad=5, xy_limit=3.0)
    field = FieldZernikes(NAT_MODES, [2, 2, 2, 1, 1, 0], norm=norm, prior_sd=float("inf"))
    independent = fit_field_knots(
        SplinePSF(7, (0.0, 0.2), device="cpu", **grid),
        data,
        xy,
        base,
        field,
        tile=32,
        iters=3,
        loc_iters=5,
        min_spots=1,
        lr=0.002,
        var=torch.tensor(2.0),
    )
    slots = [[0, 1, 2, 3, 4, 5]] * 3 + [[0, 1, 3]] * 2 + [[0]]
    coefficients = independent["psf"].field_block()["coef"]
    independent_params = [row[i] for row, ids in zip(coefficients, slots) for i in ids]
    _, loss = localize_field(true.to_tiled(tile=32, **grid), data, xy, torch.tensor(2.0))
    out = dict(
        optics=optics,
        noll=noll,
        base=base.zcoef.tolist(),
        gamma=gamma,
        xy=xy.tolist(),
        pos=pos.tolist(),
        pixels=image.flatten(1).tolist(),
        data=data.flatten().tolist(),
        expected=result["report"]["gamma"],
        independent=independent_params,
        nll=loss.tolist(),
        columns=[nat_polynomial(col).tolist() for col in torch.eye(13, dtype=torch.float64)],
    )
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    path = Path(__file__).parents[1] / "web/test/nat_reference.json"
    path.write_text(json.dumps(reference(), separators=(",", ":")) + "\n")
