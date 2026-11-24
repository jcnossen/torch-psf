"""Translation of each exposure in the mean bead-catalog coordinate system (pixels)."""
import numpy as np
from scipy.ndimage import map_coordinates


def lateral_registration(movie):
    """Iteratively align the central camera patch to a common image template."""
    frames, h, w = movie.shape
    side = min(160, h, w)
    drift = np.zeros((frames, 2))
    if frames < 2 or side < 24:
        return drift
    y0, x0 = (h - side) // 2, (w - side) // 2
    patch = np.asarray(movie[:, y0 : y0 + side, x0 : x0 + side], dtype=np.float32)
    yy, xx = np.mgrid[4 : side - 4 : 2, 4 : side - 4 : 2]
    for _ in range(10):
        values, gx, gy = [], [], []
        for frame, (dx, dy) in zip(patch, drift):

            def sample(x, y):
                return map_coordinates(
                    frame, [yy + y, xx + x], order=1, mode="nearest"
                ).ravel()

            v = sample(dx, dy)
            scale = max(np.std(v), 1e-6)
            values.append((v - v.mean()) / scale)
            gx.append((sample(dx + 1, dy) - sample(dx - 1, dy)) / (2 * scale))
            gy.append((sample(dx, dy + 1) - sample(dx, dy - 1)) / (2 * scale))
        target = np.mean(values, axis=0)
        for t, (x, y) in enumerate(zip(gx, gy)):
            g = np.stack([x, y])
            # A shared template suppresses the changing illumination phase.
            delta = np.linalg.solve(g @ g.T + np.eye(2), g @ (target - values[t]))
            drift[t] += np.clip(delta, -0.5, 0.5)
        drift -= drift.mean(0)
    return drift


def registered_frame(image, displacement):
    """Sample the raw exposure at catalog pixel + estimated camera displacement."""
    from scipy.ndimage import shift

    return shift(
        image, -np.asarray(displacement)[::-1], order=1, mode="nearest", prefilter=False
    )


def catalog_seeds(image, frames):
    """Subpixel maxima above a local annular background and shot noise."""
    from scipy.ndimage import maximum_filter

    # Inner 3x3 binomial smoothing, outer 12 samples on a 3-pixel square.
    pad = np.pad(image, 3, mode="edge")
    h, w = image.shape

    def sample(x, y):
        return pad[3 + y : 3 + y + h, 3 + x : 3 + x + w]

    smooth = (
        sum(
            wx * wy * sample(x, y)
            for x, wx in ((-1, 1), (0, 2), (1, 1))
            for y, wy in ((-1, 1), (0, 2), (1, 1))
        )
        / 16
    )
    ring = (
        sum(
            sample(x, y)
            for x in (-3, -1, 1, 3)
            for y in (-3, -1, 1, 3)
            if abs(x) == 3 or abs(y) == 3
        )
        / 12
    )
    score = smooth - ring
    # Local contrast rejects shoulders in the broad pupil halo.
    keep = (score == maximum_filter(score, 3)) & (
        score > np.maximum(6 * np.sqrt(np.maximum(ring, 1) / frames), 0.12 * smooth)
    )
    keep[:2] = keep[-2:] = False
    keep[:, :2] = keep[:, -2:] = False
    y, x = np.nonzero(keep)

    def offset(lo, hi):
        return (0.5 * (lo - hi) / np.minimum(-1e-6, lo + hi - 2 * score[y, x])).clip(
            -0.5, 0.5
        )

    return np.column_stack(
        (
            x + offset(score[y, x - 1], score[y, x + 1]),
            y + offset(score[y - 1, x], score[y + 1, x]),
            score[y, x],
        )
    )


def movie_candidates(movie, drift):
    """Combine subpixel detections across exposures in the fixed catalog frame."""
    h, w = movie.shape[1:]
    votes = np.zeros((2 * h, 2 * w), np.float32)
    amplitude = np.zeros_like(votes)
    for image, delta in zip(movie, drift):
        detections = residual_detections(image)
        x, y = (2 * (detections[:, :2] - delta)).T
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        for offx, wx in ((0, 1 - (x - ix)), (1, x - ix)):
            for offy, wy in ((0, 1 - (y - iy)), (1, y - iy)):
                px, py = ix + offx, iy + offy
                ok = (px >= 0) & (px < 2 * w) & (py >= 0) & (py < 2 * h)
                weight = (wx * wy)[ok]
                np.add.at(votes, (py[ok], px[ok]), weight)
                np.add.at(amplitude, (py[ok], px[ok]), weight * detections[ok, 2])
    from scipy.ndimage import convolve1d, maximum_filter

    for axis in (0, 1):
        votes = convolve1d(votes, [0.25, 0.5, 0.25], axis=axis, mode="constant")
        amplitude = convolve1d(amplitude, [0.25, 0.5, 0.25], axis=axis, mode="constant")
    y, x = np.nonzero(
        (votes == maximum_filter(votes, 3)) & (votes > max(0.5, 0.014 * len(movie)))
    )
    return np.column_stack((x * 0.5, y * 0.5, amplitude[y, x] / votes[y, x]))


def residual_detections(image):
    """Use the median observed bead footprint in this exposure for residual deblending."""
    h, w = image.shape
    initial = catalog_seeds(image, 1)
    if not len(initial):
        return initial
    good = initial[
        (initial[:, 0] >= 6)
        & (initial[:, 0] < w - 6)
        & (initial[:, 1] >= 6)
        & (initial[:, 1] < h - 6)
    ]
    if len(good) < 4:
        return initial
    good = good[np.argsort(-good[:, 2], kind="stable")[:96]]
    center = np.floor(good[:, :2] + 0.5).astype(int)
    dy, dx = np.mgrid[-5:6, -5:6]
    patches = image[center[:, 1, None, None] + dy, center[:, 0, None, None] + dx]
    edge = np.minimum(patches[:, 0, :].mean(1), patches[:, -1, :].mean(1))
    template = np.maximum(
        0,
        np.median(
            (patches - edge[:, None, None]) / np.maximum(1, good[:, 2, None, None]),
            axis=0,
        ),
    )
    residual = image.copy()
    all_peaks = [initial]
    peaks = initial
    for iteration in range(2):
        center = np.floor(peaks[:, :2] + 0.5).astype(int)
        for iy in range(11):
            for ix in range(11):
                x = center[:, 0] + ix - 5
                y = center[:, 1] + iy - 5
                ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                np.add.at(
                    residual, (y[ok], x[ok]), -0.75 * template[iy, ix] * peaks[ok, 2]
                )
        peaks = catalog_seeds(residual, 1)
        # Subtraction can reduce the local noise estimate. Require contrast in original count units.
        if len(peaks):
            x, y = np.floor(peaks[:, :2] + 0.5).astype(int).T
            peaks = peaks[peaks[:, 2] > 4 * np.sqrt(np.maximum(image[y, x], 1))]
        if not len(peaks):
            break
        all_peaks.append(peaks)
    return np.concatenate(all_peaks)
