"""Interactive PSF viewer on localhost: python -m torch_psf view [psf.json] [--port 8000].

A standard-library HTTP server (no extra dependencies) with a JSON / binary API that computes PSF volumes,
isosurface meshes, pixel images, gradient images and CRLB curves with torch (docs/VIEWER_API.md), and serves the
bundled front end (torch_psf/viewer_static/, built from the web/ project's viewer entry at packaging time, see
scripts/build_viewer.sh) from the same origin. Without the bundle, / shows how to build it; the API works anyway.

    python -m torch_psf.viewer --dump-fixtures DIR     one response of every endpoint (tetrapod preset) for offline
                                                        front-end tests
"""
import argparse
import json
import os
import struct
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
from torch_psf.optics import Optics, resolve_device
from torch_psf.fit import crlb
from torch_psf.psf import PSF
from torch_psf.vectorial import VectorialPSF
from torch_psf.spline import CubicSplinePSF
from torch_psf.gaussian import GaussianPSF, AstigmaticGaussianPSF

API_VERSION = 1
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer_static')
NOLL = list(range(5, 23))

DEFAULT = dict(model='vectorial', optics=dict(pixel_um=0.108, NA=1.49, wavelength_um=0.68, n_imm=1.518, n_med=1.33),
               focus_um=0.4, z_um=0.4, z_range=[0.0, 0.8], zernike={'6': 0.08}, roisize=17, photons=3000.0, bg=20.0,
               sigma_px=1.3, levels=[0.5, 0.1, 0.01], meshes=True, file=None)
PRESETS = dict(
    astigmatic=dict(DEFAULT),
    tetrapod=dict(DEFAULT, focus_um=1.0, z_um=1.0, z_range=[0.0, 2.0], zernike={'6': 0.2, '12': -0.2}, roisize=25),
)

NOT_BUNDLED = b"""<!doctype html><html><head><meta charset="utf-8"><title>torch-psf viewer</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 46em; margin: 3em auto">
<h1>torch-psf viewer</h1>
<p>The viewer front end is not bundled in this installation. The API is running at this address
(<a href="/api/default">/api/default</a>; see docs/VIEWER_API.md).</p>
<p>To bundle the front end from a source checkout (needs Node.js): <code>scripts/build_viewer.sh</code>
(runs <code>npm --prefix web run build:viewer</code> and copies <code>web/dist-viewer</code> into
<code>torch_psf/viewer_static</code>), then restart <code>python -m torch_psf view</code>.</p></body></html>"""


class ApiError(Exception):
    def __init__(self, status, error, detail=''):
        super().__init__(error)
        self.status, self.error, self.detail = status, error, detail


def pack(header, arrays):
    """Binary response: uint32 LE header length N, N bytes JSON header (space-padded so that 4 + N is a multiple of 4),
    then the arrays (little endian, C order) in the order of header['arrays'] (name, dtype, shape, axes, offset,
    nbytes; offset in bytes from the start of the array section, every offset a multiple of 4)."""
    desc, blobs, off = [], [], 0
    for name, a, axes in arrays:
        a = np.ascontiguousarray(a)
        a = a.astype('<f4') if a.dtype.kind == 'f' else a.astype('<u4')
        b = a.tobytes()
        desc.append(dict(name=name, dtype='float32' if a.dtype.kind == 'f' else 'uint32', shape=list(a.shape),
                         axes=axes, offset=off, nbytes=len(b)))
        blobs.append(b)
        off += len(b)
    h = json.dumps(dict(header, api_version=API_VERSION, arrays=desc)).encode()
    h += b' ' * ((-(4 + len(h))) % 4)
    return struct.pack('<I', len(h)) + h + b''.join(blobs)


def unpack(buf):
    """Inverse of pack (tests, fixtures): (header, {name: array})."""
    n = struct.unpack('<I', buf[:4])[0]
    h = json.loads(buf[4:4 + n])
    base = 4 + n
    out = {}
    for d in h['arrays']:
        dt = '<f4' if d['dtype'] == 'float32' else '<u4'
        out[d['name']] = np.frombuffer(buf, dt, count=int(np.prod(d['shape'])),
                                       offset=base + d['offset']).reshape(d['shape'])
    return h, out


def _check_params(p):
    if p.get('model') not in ('vectorial', 'spline', 'gaussian'):
        raise ApiError(400, 'bad_request', f"model must be vectorial, spline or gaussian, not {p.get('model')!r}")
    zr = p.get('z_range')
    if not (isinstance(zr, (list, tuple)) and len(zr) == 2 and zr[1] > zr[0]):
        raise ApiError(400, 'bad_request', 'z_range must be [lo, hi] with hi > lo (um)')
    R = int(p.get('roisize', 0))
    if R < 5 or R > 81 or R % 2 == 0:
        raise ApiError(400, 'bad_request', 'roisize must be odd, 5 .. 81')


class Engine:
    """Builds the PSF models of a parameter set (cached) and computes the API results."""

    def __init__(self, device=None, psf_path=None):
        self.dev = resolve_device(device)
        self.lock = threading.Lock()
        self.cache = {}
        self.loaded = None                 # a spline loaded from a file (its knots are shown as they are)
        self.default = dict(DEFAULT)
        if psf_path:
            self.default.update(self.load(psf_path))

    def load(self, path):
        if not path or not os.path.isfile(str(path)):
            raise ApiError(404, 'not_found', f'no such file: {path}')
        try:
            psf = PSF.load(path, roisize=DEFAULT['roisize'], device=self.dev)
        except Exception as e:
            raise ApiError(422, 'unreadable_psf', f'{type(e).__name__}: {e}')
        p = dict(file=os.path.basename(str(path)))
        if isinstance(psf, VectorialPSF):
            p.update(model='vectorial', optics=psf.optics.to_dict(), focus_um=psf.focus_um,
                     zernike={str(j): float(c) for j, c in zip(psf.zidx, psf.zcoef.detach().cpu()) if c != 0})
            if psf.z_range is not None:
                p['z_range'] = list(psf.z_range)
            self.loaded = None
        elif isinstance(psf, (CubicSplinePSF, AstigmaticGaussianPSF)):
            p.update(model='spline', z_range=list(psf.z_range), z_um=float(np.mean(psf.z_range)), roisize=psf.roisize)
            self.loaded = psf if isinstance(psf, CubicSplinePSF) else psf.to_spline(psf.roisize, dz_um=0.02)
        elif isinstance(psf, GaussianPSF):
            p.update(model='gaussian', sigma_px=psf.sigma[0], roisize=psf.roisize)
            self.loaded = None
        return p

    def _vectorial(self, p, roisize):
        v = VectorialPSF.from_optics(Optics.from_dict(p['optics']), roisize, zernike_indices=NOLL, npupil=64,
                                     oversample=3, device=self.dev, focus_um=float(p['focus_um']),
                                     z_range=tuple(p['z_range']))
        return v.set_zernike({int(k): float(x) for k, x in p.get('zernike', {}).items() if int(k) in NOLL})

    def models(self, p):
        """(model for the pixel panels, CubicSplinePSF whose knots give the volume (or None))."""
        key = json.dumps({k: p.get(k) for k in ('model', 'optics', 'focus_um', 'z_range', 'zernike', 'roisize',
                                                 'sigma_px')}, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        R = int(p['roisize'])
        if p['model'] == 'gaussian':
            out = (GaussianPSF(R, sigma_px=float(p['sigma_px']), device=self.dev), None)
        elif p['model'] == 'spline' and self.loaded is not None:
            out = (self.loaded, self.loaded)
        else:
            spl = self._vectorial(p, R).to_spline(R, tuple(p['z_range']), oversample=2, dz_um=0.02, zpad=2,
                                                  xy_limit=1.0, chunk=8)
            out = ((spl if p['model'] == 'spline' else self._vectorial(p, R)), spl)
        if len(self.cache) > 8:
            self.cache.clear()
        self.cache[key] = out
        return out

    @torch.no_grad()
    def volume(self, p):
        """(header, arrays): the PSF volume [nz, ny, nx] (x fastest) on a 2x oversampled xy grid, plus isosurface
        meshes at relative levels of the peak when requested and scikit-image is available."""
        _check_params(p)
        render, spl = self.models(p)
        R = int(p['roisize'])
        px = float(p['optics']['pixel_um'])
        if spl is not None:          # the knots are the PSF on a 1/S px grid (S = 2) and dz_um in z
            zs = spl.zs.cpu().numpy()
            keep = (zs >= spl.zlo - 1e-9) & (zs <= spl.zhi + 1e-9)
            n = (R + 2) * spl.S                                # the ROI plus 1 px on every side
            c0 = (spl.knots.shape[-1] - n) // 2
            vol = spl.knots[torch.tensor(keep, device=spl.dev)][:, c0:c0 + n, c0:c0 + n].cpu().numpy()
            dx, dz, z0 = px / spl.S, spl.hz, float(zs[keep][0])
        else:                        # other models: render at the 2 x 2 sub-pixel shifts and interleave
            zr = render.z_range or tuple(p['z_range'])
            z = torch.arange(zr[0], zr[1] + 1e-9, 0.02, device=self.dev)
            a = torch.tensor([0.0, 0.5], device=self.dev)
            B_, A_ = torch.meshgrid(a, a, indexing='ij')
            pos = torch.stack([A_.reshape(-1)[None].expand(len(z), -1), B_.reshape(-1)[None].expand(len(z), -1),
                               z[:, None].expand(-1, 4)], -1).reshape(-1, 3)
            imgs = render.render(pos).reshape(len(z), 2, 2, R, R)
            vol = imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(len(z), 2 * R, 2 * R).cpu().numpy()
            dx, dz, z0 = px / 2, 0.02, float(z[0])
        nz, ny, nx = vol.shape
        vmax = float(vol.max())
        header = dict(kind='volume', model=p['model'], dx_um=dx, dz_um=dz, z0_um=z0, shape=[nz, ny, nx],
                      axis_order='z,y,x', origin='x = y = 0 at the volume centre; z of plane k = z0_um + k dz_um',
                      vmax=vmax, levels=[], meshes=[])
        arrays = [('volume', vol, 'z,y,x')]
        if p.get('meshes', True):
            try:
                from skimage.measure import marching_cubes
            except ImportError:
                marching_cubes = None
                header['mesh_note'] = 'scikit-image not installed: no meshes (compute them in the page)'
            if marching_cubes is not None and vmax > 0:
                for i, lv in enumerate(sorted(p.get('levels', DEFAULT['levels']), reverse=True)):
                    if not (vol.min() < lv * vmax < vmax):
                        continue
                    verts, faces, normals, _ = marching_cubes(vol, lv * vmax, spacing=(dz, dx, dx))
                    xyz = np.stack([verts[:, 2] - (nx - 1) / 2 * dx, verts[:, 1] - (ny - 1) / 2 * dx,
                                    verts[:, 0] + z0], 1)
                    nrm = np.stack([normals[:, 2], normals[:, 1], normals[:, 0]], 1)
                    header['levels'].append(lv)
                    header['meshes'].append(dict(level=lv, n_vertices=len(xyz), n_faces=len(faces),
                                                 vertices=f'mesh{i}_vertices', normals=f'mesh{i}_normals',
                                                 faces=f'mesh{i}_faces'))
                    arrays += [(f'mesh{i}_vertices', xyz, 'vertex,xyz_um'), (f'mesh{i}_normals', nrm, 'vertex,xyz'),
                               (f'mesh{i}_faces', faces, 'face,corner')]
        return header, arrays

    @torch.no_grad()
    def render(self, p):
        """(header, arrays): the normalized PSF at (0, 0, z_um) on the pixel grid, d/dx, d/dy (per px), d/dz (per
        um), and the CRLB of x, y, z (nm) over 41 heights in z_range at the given photons / bg."""
        _check_params(p)
        render, spl = self.models(p)
        pos = torch.tensor([[0.0, 0.0, float(p['z_um'])]], device=self.dev)
        g, gx, gy, gz = [t.detach()[0].cpu().numpy() for t in render.render(pos, grad=True)]
        zr = p['z_range'] if render.z_range is not None else [0.0, 0.0]
        zz = torch.linspace(zr[0], zr[1], 41, device=self.dev)
        th = torch.stack([torch.zeros_like(zz), torch.zeros_like(zz), zz, torch.full_like(zz, float(p['photons'])),
                          torch.full_like(zz, float(p['bg']))], 1)
        cr = crlb(render, th).cpu().numpy()
        px_nm = 1000 * float(p['optics']['pixel_um'])
        header = dict(kind='render', model=p['model'], z_um=float(p['z_um']), roisize=int(p['roisize']),
                      pixel_um=float(p['optics']['pixel_um']), sum=float(g.sum()), peak=float(g.max()),
                      photons=float(p['photons']), bg=float(p['bg']))
        arrays = [('psf', g, 'y,x'), ('d_dx', gx, 'y,x'), ('d_dy', gy, 'y,x'), ('d_dz', gz, 'y,x'),
                  ('crlb_z_um', zz.cpu().numpy(), 'z'), ('crlb_x_nm', cr[:, 0] * px_nm, 'z'),
                  ('crlb_y_nm', cr[:, 1] * px_nm, 'z'), ('crlb_z_nm', np.nan_to_num(cr[:, 2] * 1000, nan=0.0), 'z')]
        return header, arrays


def _params(engine, req):
    p = dict(engine.default)
    p.update(req)
    if 'optics' in req:
        p['optics'] = dict(engine.default['optics'], **req['optics'])
    return p


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype='application/json'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, error, detail=''):
            self._send(status, dict(error=error, detail=detail, api_version=API_VERSION))

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/api/default':
                return self._send(200, dict(engine.default, api_version=API_VERSION, presets=list(PRESETS)))
            if path.startswith('/api/'):
                return self._error(404, 'unknown_endpoint', path)
            if not os.path.isdir(STATIC):
                return self._send(200, NOT_BUNDLED, 'text/html; charset=utf-8')
            rel = 'viewer.html' if path in ('/', '/index.html') else path.lstrip('/')
            f = os.path.realpath(os.path.join(STATIC, rel))
            if not f.startswith(os.path.realpath(STATIC)) or not os.path.isfile(f):
                return self._error(404, 'not_found', path)
            ctype = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
                     '.png': 'image/png', '.svg': 'image/svg+xml', '.json': 'application/json',
                     '.wasm': 'application/wasm'}.get(os.path.splitext(f)[1], 'application/octet-stream')
            with open(f, 'rb') as fh:
                return self._send(200, fh.read(), ctype)

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            try:
                try:
                    req = json.loads(self.rfile.read(n) or b'{}')
                except json.JSONDecodeError as e:
                    raise ApiError(400, 'bad_request', f'invalid JSON: {e}')
                with engine.lock:
                    if self.path == '/api/load':
                        return self._send(200, dict(engine.load(req.get('path')), api_version=API_VERSION))
                    if self.path == '/api/preset':
                        name = req.get('name')
                        if name not in PRESETS:
                            raise ApiError(404, 'not_found', f'presets: {list(PRESETS)}')
                        engine.loaded = None
                        return self._send(200, dict(PRESETS[name], api_version=API_VERSION))
                    if self.path == '/api/volume':
                        h, arr = engine.volume(_params(engine, req))
                    elif self.path == '/api/render':
                        h, arr = engine.render(_params(engine, req))
                    else:
                        raise ApiError(404, 'unknown_endpoint', self.path)
                self._send(200, pack(h, arr), 'application/octet-stream')
            except ApiError as e:
                self._error(e.status, e.error, e.detail)
            except Exception as e:          # report errors to the page instead of dropping the connection
                self._error(500, 'internal', f'{type(e).__name__}: {e}')
    return Handler


def serve(psf=None, port=8000, device=None, open_browser=True, host='127.0.0.1'):
    """Start the viewer server (blocking). psf: optional PSF file shown first."""
    engine = Engine(device, psf)
    srv = ThreadingHTTPServer((host, port), make_handler(engine))
    url = f'http://{host}:{srv.server_address[1]}/'
    print(f'torch-psf viewer on {url} (device {engine.dev}; API: docs/VIEWER_API.md)'
          + ('' if os.path.isdir(STATIC) else '\n  front end not bundled: see scripts/build_viewer.sh')
          + '\nCtrl-C to stop', flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def dump_fixtures(out, device=None):
    """One response of every endpoint for the tetrapod preset (plus the astigmatic volume) into directory out."""
    os.makedirs(out, exist_ok=True)
    eng = Engine(device)
    p = _params(eng, PRESETS['tetrapod'])
    wj = lambda name, obj: json.dump(obj, open(os.path.join(out, name), 'w'), indent=1)
    wj('default.json', dict(eng.default, api_version=API_VERSION, presets=list(PRESETS)))
    eng._vectorial(p, p['roisize']).save(os.path.join(out, 'tetrapod_psf.json'))
    wj('load_tetrapod.json', dict(eng.load(os.path.join(out, 'tetrapod_psf.json')), api_version=API_VERSION))
    wj('request_tetrapod.json', dict(p, api_version=API_VERSION))
    for name, fn in (('volume', eng.volume), ('render', eng.render)):
        open(os.path.join(out, f'{name}_tetrapod.bin'), 'wb').write(pack(*fn(p)))
    open(os.path.join(out, 'volume_astigmatic.bin'), 'wb').write(pack(*eng.volume(_params(eng, PRESETS['astigmatic']))))
    wj('error_example.json', dict(error='bad_request', detail='roisize must be odd, 5 .. 81', api_version=API_VERSION))
    print('fixtures written to', out, sorted(os.listdir(out)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='torch-psf viewer server')
    ap.add_argument('psf', nargs='?')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--device', default=None)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--dump-fixtures', default=None, metavar='DIR')
    a = ap.parse_args()
    if a.dump_fixtures:
        dump_fixtures(a.dump_fixtures, a.device)
    else:
        serve(a.psf, a.port, a.device, not a.no_browser)
