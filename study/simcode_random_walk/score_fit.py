import argparse, json, time
from pathlib import Path
import numpy as np, tifffile, torch
from simcode_bead_movie import model, OPTICS
from frame_focus import measure, focus_trace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_cycle_pupil import fit


def run(d, seed):
    file = d / f"{seed}.tif"
    base = json.loads((d / f"{seed}-baseline.json").read_text())
    c = np.array(base["model"]["zcoef"])
    zb = np.array(base["drift"])[:, 2] / 1000
    raw = tifffile.memmap(file)
    period = 6
    n = len(raw) // period
    starts = np.rint(np.linspace(0, n - 1, min(16, n))).astype(int) * period
    centers = starts + (period - 1) / 2
    psf = model("cuda")
    obs = measure(raw)

    def estimate(c, z):
        with torch.no_grad():
            psf.zcoef.copy_(psf.zcoef.new_tensor(c))
        return focus_trace(psf, obs, centers, z, period)

    optics = fit(file, dict(coef=c, z=zb), period, OPTICS, 0.02, "cuda")[0]
    z = estimate(optics["coef"], optics["z"])
    offsets = z[starts[:, None] + np.arange(period)]
    offsets -= offsets.mean(1, keepdims=True)
    refined = fit(
        file,
        dict(coef=optics["coef"], z=z),
        period,
        OPTICS,
        0.02,
        "cuda",
        offsets=offsets,
    )[0]
    final = estimate(refined["coef"], refined["z"])

    def scores(coef, focus):
        truth = json.loads((d / f"{seed}-truth.json").read_text())

        def stack(c, z):
            with torch.no_grad():
                psf.zcoef.copy_(psf.zcoef.new_tensor(c))
                return (
                    psf(
                        psf.zcoef.new_tensor([[0.0, 0.0, 0.02]]).repeat(len(z), 1),
                        psf.zcoef.new_tensor(z) + 0.02,
                    )
                    .cpu()
                    .numpy()
                )

        t = stack(truth["coef"], truth["z"])
        e = stack(coef, focus)
        return [
            float(np.sqrt(np.mean((np.array(coef) - truth["coef"]) ** 2)) * 1000),
            float(np.sqrt(np.mean((focus - truth["z"]) ** 2)) * 1000),
            float(np.linalg.norm(t - e) / np.linalg.norm(t)),
        ]

    result = dict(
        base=scores(c, zb),
        frame=scores(optics["coef"], z),
        refined=scores(refined["coef"], final),
        coef=refined["coef"],
        z=final.tolist(),
    )
    (d / f"{seed}-python.json").write_text(json.dumps(result))
    print(seed, result["base"], result["frame"], result["refined"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score the browser baseline and frame-focus refinement against withheld truth."
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 83, 244, 519])
    args = parser.parse_args()
    for seed in args.seeds:
        run(args.directory, seed)
