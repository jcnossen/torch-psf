"""Regenerate only after reviewing an intentional output change.

    python tests/make_unify_pins.py

CPU float32, one thread. Archive pins include the real TIFF, PNG and PDF bytes,
so reproducing those SHA-256 digests also requires the baseline writer stack.
"""
from copy import copy
import hashlib
import json
import platform
import zipfile
import zlib
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torch_psf import (AstigmaticGaussianPSF, ChannelsPSF, FourPiPSF, GaussianPSF,
                       to_spline, VectorialPSF, io)
from torch_psf.field import FieldPSF, FieldZernikes
from torch_psf.sofi.model import GridPSF
from torch_psf.composition import ZMode, Polychromatic
from torch_psf.orientation import dipole_M, fit_orientation, from_angles, model_and_jacobian

FILE = ROOT / 'tests/data/unify_pins.json'
POS = torch.tensor([[.23, -.31, .012], [-.37, .18, .033]])
GRID = dict(z_range=(0., .05), oversample=2, dz_um=.015, xy_limit=.75)
M = dipole_M(torch.tensor([[1., -.4, .7], [.4, .7, .3]]), torch.tensor([.65, .84]))
ZIP_CASES = ('box', 'gauss2d', 'gauss3d', 'cubic', 'integrated', 'integrated_angle_dipole',
             'fourpi_box', 'phase_cubic', 'phase_integrated', 'phase_dipole', 'channels', 'field', 'tiles')


def vec(mode='midpoint'):
    return VectorialPSF(7, npupil=16, oversample=3, pixel_um=.108, focus_um=.08,
                        z_range=GRID['z_range'], device='cpu', pixel_integration=mode).set_zernike({6: .04, 7: -.025})


def models():
    blur = torch.tensor([.14, .08, -.035])
    # pin names keep the earlier mode names ('box' = midpoint, 'spectral' = exact)
    for name, mode in (('box', 'midpoint'), ('spectral', 'exact'), ('gauss_legendre', 'gauss_legendre')):
        v = vec(mode)
        yield name, v, {}
        yield name + '_dipole', v, dict(dipM=M)
        if mode != 'gauss_legendre':
            vb = vec(mode)
            vb.blur = blur
            yield name + '_blur', vb, {}
            yield name + '_blur_dipole', vb, dict(dipM=M)
        # Exercise the file loader's angle selection, including anisotropic blur where supported.
        spec, arrays = v._file_dict()
        spec['psf_dipole_M_angle'] = M.tolist()
        if mode != 'gauss_legendre':
            spec['psf_blur_angle_px2'] = torch.stack([blur, blur * .4]).tolist()
        a = VectorialPSF._from_file(spec, arrays, roisize=7, angle=1, device='cpu')
        yield name + '_angle', a, {}
        if mode == 'exact':
            yield 'spectral_angle_mean', VectorialPSF._from_file(spec, arrays, roisize=7, device='cpu'), {}

    v = vec()
    yield 'exact_prototype', vec('exact'), {}
    yield 'focus_z', ZMode(v, height=.017), {}
    yield 'polychromatic', Polychromatic(v, [.64, .72], [.3, .7]), {}
    yield 'gauss2d', GaussianPSF(7, (1.15, 1.4), pixel_um=.108, device='cpu'), {}
    yield 'gauss3d', AstigmaticGaussianPSF(7, (1.15, 1.4), pixel_um=.108, device='cpu'), {}
    yield 'cubic', v.to_spline(kernel='catmull', value='pixel', **GRID), {}
    yield 'cubic_dipole', v.to_spline(kernel='catmull', value='pixel', **GRID, dipole_components=True), dict(dipM=M)
    yield 'integrated', to_spline(v, kernel='bspline', value='density', **GRID), {}
    other = vec().set_zernike({6: -.03})
    integrated = to_spline([v, other], kernel='bspline', value='density', **GRID)
    yield 'integrated_angle', integrated.for_angle(1), {}
    yield 'integrated_angle_mean', integrated, {}
    yield 'integrated_dipole', to_spline(v, kernel='bspline', value='density', **GRID, dipole_components=True), dict(dipM=M)
    comps = to_spline([v, other], kernel='bspline', value='density', **GRID, dipole_components=True)
    yield 'integrated_angle_dipole', comps.for_angle(1), dict(dipM=M)
    for kernel in ('catmull', 'bspline'):
        s = to_spline(v, GRID['z_range'], roisize=7, oversample=2, value='density', dz_um=.015, xy_limit=.75, kernel=kernel)
        yield 'pixel_' + kernel, s, {}
        conv = copy(s)
        conv.backend = 'conv'
        yield 'pixel_' + kernel + '_conv', conv, {}
    phase = dict(phi=torch.tensor([.45, -1.2]))
    for name, mode in (('box', 'midpoint'), ('spectral', 'exact')):
        f = FourPiPSF(7, npupil=16, oversample=2, pol=(0, 1), phi=(.2, 1.8), tau=(.4, .6), a=.87,
                      device='cpu', z_range=GRID['z_range'], pixel_integration=mode)
        f.bot.set_zernike({6: .03})
        yield 'fourpi_' + name, f, phase
        kind = 'cubic' if mode == 'midpoint' else 'integrated'
        yield 'phase_' + kind, f.to_spline(kind=kind, **GRID), phase
    yield 'phase_dipole', f.to_spline(kind='integrated', dipole_components=True, **GRID), dict(**phase, dipM=M)
    yield 'channels', ChannelsPSF([v, other], [[.15, -.1], [-.08, .2]], [.7, .3]), {}
    field = FieldZernikes([5, 6], order=1, norm=(16., 8., 16., 8.), device='cpu')
    with torch.no_grad():
        field.coef.copy_(torch.tensor([[.025, -.05], [.08, .03]]))
    f = FieldPSF(v, field)
    where = dict(xy=torch.tensor([[3., 4.], [28., 11.]]))
    yield 'field', f, where
    yield 'field_bound', f.bind(**where), {}
    yield 'tiles', f.to_tiled(tile=16, **GRID), where


def moments(images):
    """Unnormalised signed moments about the ROI centre, including the cross moment."""
    n = images.shape[-1]
    y, x = np.mgrid[:n, :n].astype(float) - (n - 1) / 2
    weights = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y])
    return np.einsum('...ij,kij->...k', images.astype(float), weights)


def numbers(a):
    a = np.asarray(a)
    if a.ndim == 0:
        return float(format(float(a), '.7g'))
    return [numbers(row) for row in a]


@torch.no_grad()
def outputs(m, kwargs):
    value = m.render(POS, **kwargs)
    grad = m.render(POS, grad=True, **kwargs)
    fields = ('value', 'dx', 'dy', 'dz', 'dphi')
    assert len(grad) in (4, 5)
    return {k: t.detach().numpy() for k, t in zip(fields, grad)}, value.detach().numpy()


def image_record(im):
    return dict(pixels=numbers(im), moments=numbers(moments(im)))


@torch.no_grad()
def orientation():
    v = vec()
    axes = from_angles(torch.tensor([35., 65.]), torch.tensor([28., 155.]))
    t = torch.cat([POS, torch.tensor([[4000., 8.], [6100., 10.]]), axes, torch.tensor([[.72], [.83]])], dim=1)
    mu, jac = model_and_jacobian(v, t)
    # Fixed integer counts and an offset start exercise actual nonlinear updates.
    start = t.clone()
    start[:, :3] += torch.tensor([.04, -.03, .004])
    result = fit_orientation(v, mu[:, 0].round(), theta0=start, iters=12, mirror=False)
    return mu[:, 0].numpy(), jac[:, :, 0].numpy(), result['theta'].numpy()


def sofi():
    v = VectorialPSF(7, npupil=16, zernike_indices=(5, 6), device='cpu', pixel_integration='midpoint').set_zernike({6: .04})
    grid = GridPSF(v, 7, os=1, z_layers=(.012, .033), focus=.08)
    value, jac = grid.jac()
    return value.numpy(), jac.numpy()


def writer_stack():
    return dict(platform=sys.platform, python=platform.python_version(), zlib=zlib.ZLIB_RUNTIME_VERSION,
                packages={name: version(name) for name in ('torch', 'numpy', 'matplotlib', 'tifffile', 'pillow')})


def archive_pins(path, m):
    if isinstance(m, ChannelsPSF):
        io.save_zip(path, m.models, channels=[dict(intensity_ratio=float(r)) for r in m.intensity_ratio],
                    zstack_planes=3)
    elif type(m).__repr__ is object.__repr__:
        # The report embeds repr(model). Freeze only the allocation address in the
        # inherited Python repr; a future model-defined repr goes through unchanged.
        def stable_repr(obj):
            return object.__repr__(obj).split(' at 0x')[0] + ' at 0x00000000>'
        with patch.object(type(m), '__repr__', stable_repr):
            io.save_zip(path, m, zstack_planes=3)
    else:
        io.save_zip(path, m, zstack_planes=3)
    # Compressed graphics, font and library version strings and ZIP creator OS vary
    # across writer stacks. Keep the format manifest independently checkable on CI.
    with zipfile.ZipFile(path) as z:
        layout = [(e.filename, e.compress_type, e.date_time) for e in z.infolist()]
        manifest = z.read('psf.json') + json.dumps(layout).encode()
    return dict(file=hashlib.sha256(path.read_bytes()).hexdigest(),
                manifest=hashlib.sha256(manifest).hexdigest())


def main():
    torch.set_num_threads(1)
    data = dict(positions=POS.tolist(), moments=['sum', 'x', 'y', 'xx', 'xy', 'yy'],
                cases={}, writer_stack=writer_stack(), zip_sha256={})
    with TemporaryDirectory() as tmp:
        for name, model, kwargs in models():
            outputs_grad, value = outputs(model, kwargs)
            record = {k: image_record(im) for k, im in outputs_grad.items()}
            # Both render entry points share a pin when their values agree exactly.
            if not np.array_equal(value, outputs_grad['value']):
                record['forward'] = image_record(value)
            data['cases'][name] = record
            if name in ZIP_CASES:
                data['zip_sha256'][name] = archive_pins(Path(tmp)/'case.psf.zip', model)
    mu, jac, fitted = orientation()
    data['orientation'] = dict(mu=image_record(mu), jac=image_record(jac), theta=numbers(fitted))
    value, jac = sofi()
    data['sofi'] = dict(value=image_record(value), jac=image_record(jac))
    # Break between pixel rows and records for readable diffs without per-pixel indentation.
    compact = json.dumps(data, separators=(',', ':'), allow_nan=False)
    with FILE.open('w', encoding='utf-8', newline='\n') as out:
        out.write(compact.replace('],[', '],\n[').replace('},"', '},\n"') + '\n')
    print(len(data['cases']), 'render cases,', FILE.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
