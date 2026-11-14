"""End-to-end entry points (also available as the command line, python -m torch_psf):

  localize_movie(movie, psf, camera)               movie -> detect -> spots -> MLE fit -> localization table
  estimate_psf_beads(stacks, dz_um, ...)           bead z-stacks -> PSFEstimate (vectorial PSF)
  estimate_psf_insitu(movie, ...)                  SMLM movie -> spots -> PSFEstimate (in-situ vectorial PSF)

Both estimators take the starting point the same way: psf (a VectorialPSF or a PSF file path) or optics (Optics, a
dict with its fields, or None for the defaults), plus an optional camera (Camera or calibration json path)."""
import time
import numpy as np
import torch
from torch_psf.config import Optics, Camera, resolve_device
from torch_psf.fit import localize, crlb, model_and_jacobian, at_limit
from torch_psf.movie import open_movie
from torch_psf.spots import extract_spots
from torch_psf.table import make_table, LOC_DTYPE
from torch_psf.estimate import PSFEstimate


def _camera(camera):
    if camera is None or isinstance(camera, Camera):
        return camera
    if isinstance(camera, dict):
        return Camera.from_dict(camera)
    from torch_psf.io import load_camera
    return load_camera(camera)


def _psf(psf, roisize=None, device=None):
    if psf is None or not isinstance(psf, (str, bytes)) and not hasattr(psf, '__fspath__'):
        return psf
    from torch_psf.io import load_psf
    return load_psf(psf, roisize=roisize, device=device)


def _start_vectorial(psf, optics, roisize, device, zernike_indices=tuple(range(5, 22)), npupil=64, oversample=3):
    from torch_psf.vectorial import VectorialPSF
    psf = _psf(psf, roisize, device)
    if psf is not None:
        return psf.resized(roisize) if psf.roisize != roisize else psf
    o = optics if isinstance(optics, Optics) else Optics.from_dict(optics or {})
    return VectorialPSF.from_optics(o, roisize, zernike_indices=zernike_indices, npupil=npupil, oversample=oversample,
                                    device=device)


@torch.no_grad()
def localize_spots(spots, psf, batch=4000):
    """MLE fit of Spots (torch_psf.spots) with any PSF model -> localization table (numpy structured array,
    torch_psf.table.LOC_DTYPE; image coordinates in px, z in um)."""
    rows = []
    for s in range(0, len(spots), batch):
        sp = spots.subset(slice(s, s + batch))
        v = sp.var
        th, nll = localize(psf, sp.rois, var=v)
        cr = crlb(psf, th, var=v)
        mu, _ = model_and_jacobian(psf, th)
        vv = 0.0 if v is None else v
        chi2 = ((sp.rois - mu) ** 2 / (mu + vv).clamp_min(1e-6)).flatten(1).mean(1)
        rows.append(make_table(sp.frame, sp.image_xy(th), th, cr, nll, chi2, at_limit(th, psf)))
    return np.concatenate(rows) if rows else np.zeros(0, LOC_DTYPE)


def localize_movie(movie, psf, camera=None, frames=None, threshold=6.0, chunk=500, block=5000, device=None,
                   progress=None, stop=None, **detect_kw):
    """Localize a movie with a PSF model.

    movie: path (.tif / .npy), array [T, H, W] or reader (torch_psf.movie). psf: PSF model or PSF file path (ROI size
    = psf.roisize; use a ConvSplinePSF for speed). camera: Camera, calibration json path or None (movie already in
    photons). frames: (f0, f1). threshold: detection threshold (x noise of the band-pass filtered frame).
    Frames are processed in blocks of `block` frames, detection in GPU chunks of `chunk` frames.
    progress(frames_done, frames_total, n_locs), stop() -> bool: callbacks. detect_kw: torch_psf.detect.detect_spots.
    Returns the localization table (numpy structured array, LOC_DTYPE: frame, x, y (px), z (um), photons, bg,
    crlb_*, loglik, chi2, at_limit)."""
    mv = open_movie(movie)
    psf = _psf(psf, device=device)
    camera = _camera(camera)
    dev = resolve_device(device if device is not None else getattr(psf, 'dev', None))
    f0, f1 = (0, mv.n) if frames is None else frames
    out = []
    for b0 in range(f0, f1, block):
        b1 = min(b0 + block, f1)
        sp = extract_spots(mv, psf.roisize, camera, (b0, b1), chunk, dev, threshold=threshold, **detect_kw)
        if len(sp):
            out.append(localize_spots(sp, psf))
        if progress is not None:
            progress(b1 - f0, f1 - f0, sum(len(o) for o in out))
        if stop is not None and stop():
            break
    return np.concatenate(out) if out else np.zeros(0, LOC_DTYPE)


def find_beads(stack, roisize, nmax=32, min_sep=None, sigma=2.0, border=None):
    """Bright isolated beads in a z-stack [nk, H, W] (any units): maxima of the blurred maximum projection, at least
    min_sep (default roisize) px from each other and roisize / 2 + 1 px from the border, brightest first.
    Returns corners [n, 2] = (x0, y0) of roisize x roisize ROIs centred on them (numpy int)."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    mp = gaussian_filter(np.asarray(stack, np.float32).max(0), sigma)
    mp = mp - np.median(mp)
    h = roisize // 2
    sep = roisize if min_sep is None else min_sep
    pk = (mp == maximum_filter(mp, size=2 * h + 1)) & (mp > 0.1 * mp.max())
    y, x = np.nonzero(pk)
    b = h + 1 if border is None else border
    ok = (x >= b) & (y >= b) & (x < mp.shape[1] - b) & (y < mp.shape[0] - b)
    x, y = x[ok], y[ok]
    o = np.argsort(-mp[y, x])
    x, y = x[o], y[o]
    keep = []
    for i in range(len(x)):
        dd = np.max(np.abs(np.stack([x - x[i], y - y[i]], 1)), 1)
        dd[i] = 10 ** 9
        if (dd[keep] < sep).any() if keep else False:
            continue
        keep.append(i)
        if len(keep) >= nmax:
            break
    return np.stack([x[keep] - h, y[keep] - h], 1).astype(int)


def estimate_psf_beads(stacks, dz_um, psf=None, optics=None, camera=None, roisize=21, sign=1.0, nbeads=8, iters=300,
                       holdout=None, drift=True, zernike_indices=tuple(range(5, 22)), npupil=64, oversample=3,
                       device=None, log=None, stop=None):
    """Bead z-stacks -> vectorial PSF.

    stacks: list of [nk, H, W] arrays or TIFF paths (ADU with a camera, else photons), all with the same nk; plane k
    is at the focus position sign dz_um (k - (nk - 1) / 2) um (focal plane relative to the coverslip; the stack centre
    is focus 0, so stacks should be centred on the beads). The beads sit on the coverslip.
    psf (VectorialPSF or PSF file) or optics (Optics / dict / None = defaults): the starting model. camera: Camera,
    calibration json or None. roisize: ROI size (px); nbeads: beads per stack (brightest isolated ones);
    holdout: index of a stack kept out of the fit (its NLL per pixel is reported); drift: linear xy drift per stack.
    Returns PSFEstimate (psf: VectorialPSF with focus_um = 0; report: n_beads, nll_per_px, heldout_nll_per_px, ...)."""
    import tifffile
    from torch_psf.insitu.beads import fit_beads, stage_focus
    from torch_psf.fit import poisson_nll
    dev = resolve_device(device)
    camera = _camera(camera)
    vec = _start_vectorial(psf, optics, roisize, dev, zernike_indices, npupil, oversample)
    rois, sid = [], []
    ii = torch.arange(roisize, device=dev)
    for i, s in enumerate(stacks):
        a = tifffile.imread(s) if isinstance(s, str) else np.asarray(s)
        a = torch.as_tensor(np.asarray(a, np.float32), device=dev)
        if camera is not None:
            a = camera.to_photons(a)
        for x0, y0 in find_beads(a.cpu().numpy(), roisize, nmax=nbeads):
            rois.append(a[:, y0 + ii[:, None], x0 + ii[None, :]])
            sid.append(i)
    d = torch.stack(rois)
    sid = torch.tensor(sid, device=dev)
    nk = d.shape[1]
    focus = stage_focus(nk, dz_um, sign).to(dev)
    w = None if holdout is None else (sid != holdout).float()
    est = fit_beads(d, focus, vec, iters=iters, stack_id=sid, drift=drift, weights=w, log=log, stop=stop)
    est.report.update(n_stacks=len(stacks), dz_um=dz_um, sign=sign)
    est.psf.focus_um = 0.0
    if holdout is not None:
        m = sid == holdout
        with torch.no_grad():
            n = int(m.sum())
            pos = torch.cat([est.data['xy'][m][:, None].expand(-1, nk, -1), torch.zeros(n, nk, 1, device=dev)],
                            -1).reshape(-1, 3)
            mu = est.data['photons'][m].reshape(-1, 1, 1) * est.psf(pos, focus[None].expand(n, -1).reshape(-1)) \
                + est.data['bg'][m].reshape(-1, 1, 1)
            est.report['heldout_nll_per_px'] = float(poisson_nll(d[m].reshape(-1, roisize, roisize), mu).sum()) / d[m].numel()
    est.data.update(rois=d, stack_id=sid, focus_um=focus)
    return est


def estimate_psf_insitu(movie, psf=None, optics=None, camera=None, focus_um=None, roisize=15, z_range=(0.0, 0.8),
                        frames=None, nspots=8000, nheldout=4000, method='zernike', iters=80, oversample=2, dz_um=0.04,
                        xy_limit=3.0, isolation=None, threshold=6.0, tile=16, device=None, log=None, stop=None,
                        **detect_kw):
    """SMLM movie -> in-situ vectorial PSF by the knot route (torch_psf.insitu.fit_zernike_knots / fit_pupil_knots).

    movie: path / array / reader (ADU with a camera, else photons). psf (VectorialPSF or PSF file, e.g. a bead fit)
    or optics: the starting model; focus_um: the focal plane during the acquisition (default: the starting PSF's
    focus_um). camera: Camera / calibration json / None. roisize (px), z_range (um): emitter heights. nspots /
    nheldout: spots used for the fit / the held-out test (split by a checkerboard of tile px, so the same structures
    never appear in both). method: 'zernike' or 'pupil' (+ free pupil maps). oversample / dz_um / xy_limit: spline
    grid. isolation (px), threshold, detect_kw: spot detection (torch_psf.detect.detect_spots).
    Returns PSFEstimate (psf: VectorialPSF; report: at_limit, z_scale (relative to the starting PSF), heldout
    (NLL per pixel of the held-out spots for the starting and the fitted PSF), frames_read, ...)."""
    from torch_psf.spline import ConvSplinePSF
    from torch_psf.insitu import fit_zernike_knots, fit_pupil_knots, split_checkerboard, compare_heldout
    dev = resolve_device(device)
    camera = _camera(camera)
    vec = _start_vectorial(psf, optics, roisize, dev)
    if focus_um is not None:
        vec.focus_um = float(focus_um)
    vec.z_range = tuple(z_range)
    sp = extract_spots(movie, roisize, camera, frames, device=dev, max_spots=2 * (nspots + nheldout),
                       threshold=threshold, isolation=isolation, **detect_kw)
    h = (roisize - 1) / 2
    par = split_checkerboard(sp.corner[:, 0] + h, sp.corner[:, 1] + h, tile)
    tr = np.flatnonzero(par == 0)[:nspots]
    te = np.flatnonzero(par == 1)[:nheldout]
    grid = dict(oversample=oversample, dz_um=dz_um, zpad=5, xy_limit=xy_limit)
    spline = ConvSplinePSF(roisize, z_range, device=dev, **grid)
    if method == 'zernike':
        est = fit_zernike_knots(spline, sp.rois[tr], vec, iters=iters, log=log, stop=stop)
    elif method == 'pupil':
        est = fit_pupil_knots(spline, sp.rois[tr], vec, iters=iters, log=log, stop=stop)
    else:
        raise ValueError(method)
    s0 = ConvSplinePSF.from_vectorial(vec, roisize, z_range, **grid)
    s1 = ConvSplinePSF.from_vectorial(est.psf, roisize, z_range, **grid)
    est.report.update(n_spots=len(sp), n_train=len(tr), n_test=len(te), frames_read=sp.frames_read,
                      heldout=compare_heldout({'start': s0, 'fitted': s1}, sp.rois[te]) if len(te) else None)
    est.data['spline'] = s1
    return est
