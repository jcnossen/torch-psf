"""Fit both signed planar pupil hypotheses from a TIFF and a coarse fit."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch_psf.estimation.cycle_pupil import data, fit

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
