"""Extract isolated single-frame DNA-PAINT spot ROIs on CPU using existing SIMCODE detections."""
from pathlib import Path
import argparse
import numpy as np
import tifffile
from scipy.spatial import cKDTree


def sample(movie, cache, output):
    rng = np.random.default_rng(1809)
    images, positions, frames = [], [], []
    with tifffile.TiffFile(movie) as tif:
        for start in (0, 6000, 12000, 18000, 24000, 29400):
            locs = np.load(cache / f"spots_{start}.npz")["locs"]
            absolute = locs["frame"].astype(int)
            if absolute.max() < 600:
                absolute += start
            assert absolute.min() >= start and absolute.max() < start + 600
            # One exposure per 12 frames limits reuse of the same blinking event.
            for frame in range(start, start + 600, 12):
                t = frame + (frame // 12) % 6
                group = locs[absolute == t]
                if len(group) < 2:
                    continue
                xy = np.c_[group["x"], group["y"]]
                distance = cKDTree(xy).query(xy, k=2)[0][:, 1]
                centers = np.rint(xy).astype(int)
                good = (
                    (distance >= 10)
                    & (group["snr"] >= 6)
                    & (centers.min(1) >= 7)
                    & (centers.max(1) < 113)
                )
                page = tif.pages[t].asarray()
                for x, y in centers[good]:
                    images.append(page[y - 7 : y + 8, x - 7 : x + 8])
                    positions.append((x, y))
                    frames.append(t)
    ix = rng.permutation(len(images))[:6000]
    np.savez_compressed(
        output, adu=np.asarray(images)[ix], xy=np.asarray(positions)[ix], frame=np.asarray(frames)[ix]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("movie", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    sample(**vars(p.parse_args()))
