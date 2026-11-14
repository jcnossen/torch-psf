"""Command line of torch-psf (the same PSF json schema as the web app's export).

  python -m torch_psf beads stack.tif [stack2.tif ...] --dz-um 0.01 --pixel-um 0.108 --na 1.49 -o psf.json
  python -m torch_psf insitu movie.tif --psf beads.json --focus-um 0.4 -o insitu.json
  python -m torch_psf localize movie.tif --psf psf.json -o locs.csv
  python -m torch_psf view [psf.json] [--port 8000]

Camera: --camera calib.json (its camera block), or --gain (e-/ADU) --offset (ADU) --readnoise (e-); without either the
data are taken as photons. Run `python -m torch_psf <command> -h` for all options.
"""
import argparse
import json
import sys
import numpy as np
import torch


def _optics_args(p):
    g = p.add_argument_group('optics (used when no --psf is given)')
    g.add_argument('--pixel-um', type=float, default=0.10833, help='back-projected pixel size (um)')
    g.add_argument('--na', type=float, default=1.49, help='numerical aperture')
    g.add_argument('--wavelength-um', type=float, default=0.68, help='emission wavelength (um)')
    g.add_argument('--n-imm', type=float, default=1.518, help='immersion / coverslip refractive index')
    g.add_argument('--n-med', type=float, default=1.33, help='sample medium refractive index')


def _camera_args(p):
    g = p.add_argument_group('camera')
    g.add_argument('--camera', default=None, help='json with a camera block (gain_e_per_adu, offset_adu, readnoise_e, maps)')
    g.add_argument('--gain', type=float, default=None, help='e- per ADU')
    g.add_argument('--offset', type=float, default=None, help='ADU')
    g.add_argument('--readnoise', type=float, default=0.0, help='e- rms')


def _common(p):
    p.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')


def _camera(a):
    from torch_psf.config import Camera
    from torch_psf.io import load_camera
    if a.camera:
        return load_camera(a.camera)
    if a.gain is not None or a.offset is not None:
        return Camera(gain_e_per_adu=a.gain or 1.0, offset_adu=a.offset or 0.0, readnoise_e=a.readnoise)
    return None


def _optics(a):
    from torch_psf.config import Optics
    return Optics(pixel_um=a.pixel_um, NA=a.na, wavelength_um=a.wavelength_um, n_imm=a.n_imm, n_med=a.n_med)


def _frames(s):
    if not s:
        return None
    a, b = s.split(':')
    return int(a), int(b)


def _progress(stage, it, value, sec):
    if it % 10 == 0:
        print(f'  {stage}: iteration {it:4d}  {value:.6g}  {sec:6.1f} s', flush=True)


def _jsonable(v):
    if torch.is_tensor(v):
        return v.detach().cpu().tolist() if v.numel() <= 64 else f'<tensor {tuple(v.shape)}>'
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _print_report(est):
    print(json.dumps(_jsonable(est.report), indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m torch_psf', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('beads', help='bead z-stacks -> PSF')
    p.add_argument('stacks', nargs='+', help='TIFF z-stacks (one bead field each)')
    p.add_argument('--dz-um', type=float, required=True, help='z step of the stage scan (um)')
    p.add_argument('--sign', type=float, default=1.0, help='direction of the scan (+1 / -1)')
    p.add_argument('--roisize', type=int, default=21)
    p.add_argument('--nbeads', type=int, default=8, help='beads per stack')
    p.add_argument('--iters', type=int, default=300)
    p.add_argument('--holdout', type=int, default=None, help='index of a stack kept out of the fit')
    p.add_argument('--model', default='vectorial', choices=['vectorial', 'spline', 'astig_gauss'])
    p.add_argument('--psf', default=None, help='starting PSF file (else the optics options)')
    p.add_argument('-o', '--out', required=True, help='PSF json to write')
    _optics_args(p); _camera_args(p); _common(p)

    p = sub.add_parser('insitu', help='SMLM movie -> in-situ vectorial PSF (knot route)')
    p.add_argument('movie')
    p.add_argument('--psf', default=None, help='starting PSF file (e.g. a bead fit; else the optics options)')
    p.add_argument('--focus-um', type=float, default=None, help='focal plane during the acquisition (um)')
    p.add_argument('--z-range', type=float, nargs=2, default=(0.0, 0.8), help='emitter heights (um)')
    p.add_argument('--roisize', type=int, default=15)
    p.add_argument('--frames', default='', help='a:b')
    p.add_argument('--nspots', type=int, default=8000)
    p.add_argument('--method', default='zernike', choices=['zernike', 'pupil'])
    p.add_argument('--iters', type=int, default=80)
    p.add_argument('--threshold', type=float, default=6.0)
    p.add_argument('--isolation', type=float, default=None, help='px: drop spots with a neighbour this close')
    p.add_argument('--refine', action='store_true', help='cubic-spline refinement (kept if the held-out NLL improves)')
    p.add_argument('-o', '--out', required=True)
    _optics_args(p); _camera_args(p); _common(p)

    p = sub.add_parser('localize', help='movie -> localizations (csv or npy)')
    p.add_argument('movie')
    p.add_argument('--psf', required=True, help='PSF file (vectorial models are converted to a spline for speed)')
    p.add_argument('--roisize', type=int, default=None)
    p.add_argument('--z-range', type=float, nargs=2, default=None, help='z range (um) of the spline')
    p.add_argument('--frames', default='', help='a:b')
    p.add_argument('--threshold', type=float, default=6.0)
    p.add_argument('--isolation', type=float, default=None)
    p.add_argument('-o', '--out', required=True, help='.csv or .npy')
    _camera_args(p); _common(p)

    p = sub.add_parser('view', help='interactive PSF viewer in the browser')
    p.add_argument('psf', nargs='?', default=None)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--no-browser', action='store_true')
    _common(p)

    a = ap.parse_args(argv)
    if a.cmd == 'beads':
        from torch_psf.estimators import ZStackEstimator
        est = ZStackEstimator(a.model, start=a.psf, optics=_optics(a), camera=_camera(a), roisize=a.roisize,
                              sign=a.sign, nbeads=a.nbeads, iters=a.iters, holdout=a.holdout, device=a.device,
                              progress=_progress)
        psf = est.estimate(a.stacks, a.dz_um)
        psf.save(a.out, camera=_camera(a), provenance=dict(report=_jsonable(est.report)))
        _print_report(est)
    elif a.cmd == 'insitu':
        from torch_psf.estimators import InSituEstimator
        est = InSituEstimator(start=a.psf, optics=_optics(a), camera=_camera(a), focus_um=a.focus_um,
                              roisize=a.roisize, z_range=tuple(a.z_range), nspots=a.nspots, method=a.method,
                              iters=a.iters, threshold=a.threshold, isolation=a.isolation, refine=a.refine,
                              device=a.device, progress=_progress)
        psf = est.estimate(a.movie, frames=_frames(a.frames))
        psf.save(a.out, camera=_camera(a), provenance=dict(report=_jsonable(est.report)))
        _print_report(est)
    elif a.cmd == 'localize':
        from torch_psf.psf import PSF
        from torch_psf.vectorial import VectorialPSF
        from torch_psf.table import to_csv
        psf = PSF.load(a.psf, roisize=a.roisize, device=a.device)
        if isinstance(psf, VectorialPSF):
            psf = psf.to_spline(psf.roisize, tuple(a.z_range) if a.z_range else None)
        L = psf.localize(a.movie, _camera(a), frames=_frames(a.frames), threshold=a.threshold,
                         isolation=a.isolation,
                         progress=lambda d, t, n: print(f'  {d}/{t} frames, {n} localizations', flush=True))
        if a.out.endswith('.npy'):
            np.save(a.out, L)
        else:
            to_csv(L, a.out)
        print(f'{len(L)} localizations -> {a.out}')
    elif a.cmd == 'view':
        from torch_psf.viewer import serve
        serve(a.psf, port=a.port, device=a.device, open_browser=not a.no_browser)
    return 0


if __name__ == '__main__':
    sys.exit(main())
