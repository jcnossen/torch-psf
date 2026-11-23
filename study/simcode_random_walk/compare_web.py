"""Compare two WebGPU fit exports with independently saved simulator truth."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from simcode_bead_movie import model


@torch.no_grad()
def metrics(psf, coef, z, truth):
    def render(c, f):
        psf.zcoef.copy_(psf.zcoef.new_tensor(c))
        pos = psf.zcoef.new_tensor([[0, 0, 0.02]]).repeat(len(f), 1)
        return psf(pos, 0.02 + psf.zcoef.new_tensor(f)).cpu().numpy()

    a = render(truth["coef"], truth["z"])
    b = render(coef, z)
    return [
        np.sqrt(np.mean((np.array(coef) - truth["coef"]) ** 2)) * 1000,
        np.sqrt(np.mean((np.array(z) - truth["z"]) ** 2)) * 1000,
        100 * np.linalg.norm(b - a) / np.linalg.norm(a),
    ]


if __name__ == "__main__":
    psf = model("cuda")
    for arg in sys.argv[1:]:
        d = Path(arg)
        for file in sorted(d.glob("*-truth.json")):
            seed = file.name.split("-")[0]
            t = json.loads(file.read_text())
            for version in ["baseline", "updated"]:
                p = d / f"{seed}-{version}.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    print(
                        d.name,
                        seed,
                        version,
                        *[
                            round(float(x), 3)
                            for x in metrics(
                                psf,
                                r["model"]["zcoef"],
                                np.array(r["drift"])[:, 2] / 1000,
                                t,
                            )
                        ],
                        flush=True,
                    )
