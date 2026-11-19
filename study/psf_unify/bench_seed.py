"""Spline PSF benchmark: accuracy vs the exact (spectral) vectorial model, noise-free localization bias, speed.
Candidates per lateral knot density S (knots / px):
  cr_pix  : CubicSplinePSF, Catmull-Rom interpolation of EXACT pixel values at sub-pixel shifts (vec spectral .to_spline)
  cr_int  : PixelSplinePSF kernel 'catmull', integrate=True (point-density knots, exact pixel integral of the Catmull-Rom)
  bs_int  : PixelSplinePSF kernel 'bspline', integrate=True (= IntegratedSplinePSF, prefiltered cubic B-spline)
  box_os2 : legacy: CubicSplinePSF of the box (midpoint, oversample 2) vectorial model (reference for the old default)
usage: python bench.py <psf.zip> <out.json>"""
import json, sys, time, math
import torch
from torch_psf import PSF
from torch_psf.spline import CubicSplinePSF
from torch_psf.pixint import PixelSplinePSF, point_knots
from torch_psf import fit as F

torch.manual_seed(0)
dev = 'cuda'
path, out = sys.argv[1], sys.argv[2]
P, ZR, DZ, XL = 13, (0.0, 0.4), 0.01, 2.0
vec = PSF.load(path, roisize=P, device=dev)
vec.pixel_integration = 'spectral'
focus = vec.focus_um
print('model', type(vec).__name__, 'NA', float(vec.NA), 'px', vec.pixel_um, 'focus', focus, flush=True)

# test positions
N = 3000
pos = torch.stack([torch.empty(N).uniform_(-1.5, 1.5), torch.empty(N).uniform_(-1.5, 1.5),
                   torch.empty(N).uniform_(ZR[0] + 0.02, ZR[1] - 0.02)], 1).to(dev)
with torch.no_grad():
    ref = torch.cat([vec.render(p) for p in pos.split(250)])
peak = ref.amax((-2, -1))


def build(name, S):
    if name == 'cr_pix':
        return vec.to_spline(P, ZR, oversample=S, dz_um=DZ, zpad=3, xy_limit=XL)
    if name == 'box_os2':
        vb = PSF.load(path, roisize=P, device=dev)
        vb.pixel_integration = 'box'
        return vb.to_spline(P, ZR, oversample=S, dz_um=DZ, zpad=3, xy_limit=XL)
    geo = CubicSplinePSF(P, ZR, oversample=S, dz_um=DZ, zpad=6, xy_limit=XL + 1.0, device=dev, pixel_um=vec.pixel_um)
    big = vec.resized(geo.Pl)
    with torch.no_grad():
        smp = point_knots(big, geo, focus, chunk=4)
    return PixelSplinePSF(geo, smp, kernel='catmull' if name == 'cr_int' else 'bspline', integrate=True, backend='sep')


def err(psf):
    with torch.no_grad():
        r = torch.cat([psf.render(p) for p in pos.split(500)])
    e = (r - ref).abs().amax((-2, -1)) / peak
    return float(e.max()), float(e.median())


def bias(psf, Nph=5000.0, bg=10.0, n=400):
    """noise-free expected images of the exact model, fitted with psf: position / z error (nm)"""
    p = pos[:n]
    with torch.no_grad():
        rois = ref[:n] * Nph + bg
    th0 = torch.cat([p + torch.tensor([0.05, -0.05, 0.02], device=dev), torch.full((n, 1), Nph, device=dev),
                     torch.full((n, 1), bg, device=dev)], 1)
    th = F.localize(psf, rois, theta0=th0, iters=60, zstarts=1)
    th = th[0] if isinstance(th, tuple) else th
    th = torch.as_tensor(th, device=dev)
    dx = (th[:, 0] - p[:, 0]).abs() * vec.pixel_um * 1e3
    dy = (th[:, 1] - p[:, 1]).abs() * vec.pixel_um * 1e3
    dz = (th[:, 2] - p[:, 2]).abs() * 1e3
    return dict(x_max=float(dx.max()), y_max=float(dy.max()), z_max=float(dz.max()),
                xy_rms=float(torch.sqrt((dx ** 2 + dy ** 2).mean() / 2)), z_rms=float(torch.sqrt((dz ** 2).mean())))


def speed(psf, n=20000, reps=5, grad=True):
    p = pos[torch.randint(0, N, (n,), device=dev)]
    with torch.no_grad():
        psf.render(p[:1000], grad)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            psf.render(p, grad)
        torch.cuda.synchronize()
    return n * reps / (time.perf_counter() - t)


res = []
for S in (1, 2, 3, 4):
    for name in ('cr_pix', 'cr_int', 'bs_int') + (('box_os2',) if S == 2 else ()):
        try:
            psf = build(name, S)
            emax, emed = err(psf)
            r = dict(kind=name, S=S, err_max=emax, err_med=emed)
            try:
                r.update(bias(psf))
            except Exception as ex:          # report, keep going
                r['bias_error'] = repr(ex)[:200]
            if hasattr(psf, 'backend'):
                for be, tf32 in (('sep', False), ('conv', False), ('conv', True)):
                    psf.backend = be
                    torch.backends.cudnn.allow_tf32 = tf32
                    r[f'spots_s_{be}{"_tf32" if tf32 else ""}'] = speed(psf)
                    if be == 'conv':
                        r[f'err_max_{be}{"_tf32" if tf32 else ""}'] = err(psf)[0]
                psf.backend = 'sep'; torch.backends.cudnn.allow_tf32 = False
            else:
                for tf32 in (False, True):
                    torch.backends.cudnn.allow_tf32 = tf32
                    r['spots_s_conv' + ('_tf32' if tf32 else '')] = speed(psf)
                torch.backends.cudnn.allow_tf32 = False
            res.append(r)
            print(json.dumps(r), flush=True)
        except Exception as ex:
            print(name, S, 'FAILED', repr(ex)[:300], flush=True)
        torch.cuda.empty_cache()

# CRLB at the same photons for scale
with torch.no_grad():
    th = torch.cat([pos[:200], torch.full((200, 1), 5000.0, device=dev), torch.full((200, 1), 10.0, device=dev)], 1)
try:
    c = F.crlb(vec, th)
    c = torch.as_tensor(c)
    crlb = dict(x_nm=float(c[:, 0].median()) * vec.pixel_um * 1e3, z_nm=float(c[:, 2].median()) * 1e3)
except Exception as ex:
    crlb = dict(error=repr(ex)[:200])
print('CRLB 5000 ph, bg 10:', crlb, flush=True)
json.dump(dict(results=res, crlb=crlb, gpu=torch.cuda.get_device_name()), open(out, 'w'), indent=1)
