import numpy as np
from scipy.ndimage import gaussian_filter, shift
from scipy.spatial import cKDTree
from torch_psf.estimation.lateral_registration import lateral_registration, movie_candidates


def test_registered_catalog_is_in_mean_camera_coordinates():
    rng = np.random.default_rng(19)
    base = np.zeros((80, 80), dtype=np.float32)
    x, y = np.meshgrid(np.arange(14, 71, 13), np.arange(15, 70, 13))
    pos = np.column_stack((x.ravel(), y.ravel()))
    drift = np.column_stack((np.linspace(-2, 2, 16), 1.5*np.sin(np.arange(16)*np.pi/8)))
    frames = []
    for i, (dx, dy) in enumerate(drift):
        base[y,x] = 3000*(1+0.45*np.sin(i*np.pi/3+x/15))
        image = shift(gaussian_filter(base, 1.2), [dy,dx], order=1, mode="nearest")
        frames.append(rng.poisson(image+5).astype(np.float32))
    movie = np.stack(frames)
    measured = lateral_registration(movie)
    assert np.max(np.abs(measured.mean(0))) < 1e-6
    assert np.max(np.std(measured-drift, axis=0)) < .15
    fitted = movie_candidates(movie, measured)[:,:2]
    distances, matches = cKDTree(fitted).query(pos)
    assert np.max(distances) < .8
    assert len(set(matches)) == len(pos)
    assert len(fitted) < 1.2*len(pos)
