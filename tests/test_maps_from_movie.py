"""maps_from_movie: per-pixel sCMOS offset and read-noise maps from a sparse movie (no dark frames)"""
import numpy as np
from torch_psf.camera import maps_from_movie


def _movie(seed=0, T=400, H=40, W=48, gain=0.5, off=100.0):
    rng = np.random.default_rng(seed)
    off_map = off + rng.normal(0, 1.5, (H, W))
    rn_map = np.where(rng.random((H, W)) < 0.05, 4.0, 1.5)              # e-; a few noisy pixels
    yy, xx = np.mgrid[:H, :W]
    bg = 8 + 4 * np.sin(xx / 9.0)                                       # smooth background (photons)
    e = rng.poisson(np.broadcast_to(bg, (T, H, W))).astype(np.float64)
    # sparse emitters, 3 frames each
    sp = dict(x=rng.uniform(3, W - 4, 150), y=rng.uniform(3, H - 4, 150), frame=rng.integers(0, T - 3, 150),
              win=np.full(150, 2))
    for x, y, f in zip(sp['x'], sp['y'], sp['frame']):
        g = 400 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.3 ** 2)) / (2 * np.pi * 1.3 ** 2)
        e[f:f + 3] += rng.poisson(np.broadcast_to(g, (3, H, W)))
    e += rng.normal(0, 1, e.shape) * rn_map
    return (e / gain + off_map).astype(np.float32), sp, off_map, rn_map


def test_camera_maps_recover_offset_and_readnoise():
    mov, sp, off_map, rn_map = _movie()
    cam, rep = maps_from_movie(mov, sp, 0.5, 100.0, lag=6)
    od = cam.offset_adu - 100.0
    assert cam.offset_adu.shape == off_map.shape and cam.readnoise_e.shape == rn_map.shape
    assert np.corrcoef(od.ravel(), (off_map - 100).ravel())[0, 1] > 0.9
    hi, lo = rn_map > 3, rn_map < 3
    assert abs(np.median(cam.readnoise_e[lo]) - 1.5) < 0.4 and abs(np.median(cam.readnoise_e[hi]) - 4.0) < 0.6
    assert 0 < rep['masked_frac'] < 0.2 and abs(rep['corr_offset_density']) < 0.1
    assert set(cam.map_extra) == {'low5', 'var_adu', 'readnoise_shift', 'readnoise_covered'} and cam.scalars['offset_adu'] == 100.0


def test_camera_maps_reference_readnoise_shift():
    """the read-noise map of a reference recording (same size) at another sensor-ROI position: shift found
    automatically, covered pixels from the reference, the rest from the own high-passed estimate"""
    mov, sp, off_map, rn_map = _movie(1, T=600, H=44, W=52)
    h, w, dy, dx = 36, 42, 3, 5
    crop = lambda y0, x0: (mov[:, y0:y0 + h, x0:x0 + w], dict(sp, x=sp['x'] - x0, y=sp['y'] - y0))
    ref, _ = maps_from_movie(*crop(0, 0), 0.5, 100.0)
    R = dict(readnoise_e=ref.readnoise_e, var_adu=ref.map_extra['var_adu'], offset_adu=ref.offset_adu)
    cam, rep = maps_from_movie(*crop(dy, dx), 0.5, 100.0, readnoise_ref=R)    # this[y, x] = ref[y + dy, x + dx]
    assert rep['readnoise_shift'] == [dy, dx]
    cov = cam.map_extra['readnoise_covered']
    assert cov[:h - dy, :w - dx].all() and not cov[h - dy:].any() and not cov[:, w - dx:].any()
    assert np.allclose(cam.readnoise_e[:h - dy, :w - dx], ref.readnoise_e[dy:, dx:])
    cam2, _ = maps_from_movie(*crop(dy, dx), 0.5, 100.0, readnoise_ref=R, readnoise_shift=(dy, dx))
    assert np.array_equal(cam2.readnoise_e, cam.readnoise_e)


def test_white_photon_transfer_gain_with_fixed_pattern_and_sample_fluctuations():
    """gain from the white noise: a background that fluctuates smoothly in time and space (sample) and an offset fixed
    pattern do not bias it; photon_transfer-style binning by absolute level would"""
    from torch_psf.camera import white_photon_transfer
    rng = np.random.default_rng(3)
    T, H, W, gain, off = 600, 64, 64, 0.45, 100.0
    yy, xx = np.mgrid[:H, :W]
    fpn = rng.normal(0, 2.0, (H, W))
    rn = np.where(rng.random((H, W)) < 0.05, 3.5, 1.2)
    base = 6 + 6 * (xx / W)                                             # level span across the field
    from scipy import ndimage as ndi
    e = np.empty((T, H, W))
    for t in range(T):
        fl = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 3) * 8         # band-limited sample fluctuation (photons)
        e[t] = rng.poisson(np.clip(base * (1 + 0.1 * np.sin(t / 7)) + fl, 0, None))
    e += rng.normal(0, 1, e.shape) * rn
    mov = (e / gain + off + fpn).astype(np.float32)
    cam, rep = white_photon_transfer(mov, nblocks=2, block=300, lag=1, offset_adu=off, device='cpu')
    assert abs(cam.gain_e_per_adu / gain - 1) < 0.08, rep['gain_e_per_adu']             # statistics of 2 x 300 frames: ~4 %
    assert abs(cam.readnoise_e - np.sqrt(np.mean(rn ** 2))) < 0.6, cam.readnoise_e
    cam2, rep2 = white_photon_transfer(mov, nblocks=2, block=300, lag=1, readnoise_e=float(np.sqrt(np.mean(rn ** 2))), device='cpu')
    assert abs(cam2.offset_adu - off) < 3.0, cam2.offset_adu                         # = slope error x (level - offset)
    assert rep['fpn_sd_adu'] > 1.5 and len(rep['bins']) > 10
