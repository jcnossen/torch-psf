"""PSF viewer API (docs/VIEWER_API.md) on a free port, without a browser."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
import numpy as np
from conftest import DEV
from torch_psf.viewer import Engine, make_handler, unpack, API_VERSION, dump_fixtures


def test_viewer_api(tmp_path):
    from torch_psf import VectorialPSF, save_psf
    v = VectorialPSF(15, npupil=32, oversample=2, device=DEV, focus_um=0.3).set_zernike({6: 0.1})
    save_psf(tmp_path / 'p.json', v)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Engine(DEV)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{srv.server_address[1]}'
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    post = lambda u, b: op.open(urllib.request.Request(url + u, json.dumps(b).encode())).read()
    try:
        d = json.loads(op.open(url + '/api/default').read())
        assert d['api_version'] == API_VERSION
        p = json.loads(post('/api/load', dict(path=str(tmp_path / 'p.json'))))
        assert abs(p['zernike']['6'] - 0.1) < 1e-6 and p['focus_um'] == 0.3
        for model in ('vectorial', 'spline', 'gaussian'):
            q = dict(p, model=model, roisize=11)
            h, arr = unpack(post('/api/volume', q))
            assert h['api_version'] == API_VERSION and h['kind'] == 'volume'
            vol = arr['volume']
            assert vol.shape == tuple(h['shape']) and np.isfinite(vol).all() and vol.max() > 0
            for m in h['meshes']:
                assert arr[m['vertices']].shape == (m['n_vertices'], 3) and arr[m['faces']].max() < m['n_vertices']
            h, arr = unpack(post('/api/render', q))
            assert arr['psf'].shape == (11, 11) and len(arr['crlb_x_nm']) == 41 and h['sum'] > 0.3
        try:
            post('/api/render', dict(p, roisize=10))
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 400 and json.loads(e.read())['error'] == 'bad_request'
        assert b'<html' in op.open(url + '/').read().lower()
    finally:
        srv.shutdown()


def test_dump_fixtures(tmp_path):
    dump_fixtures(str(tmp_path), DEV)
    h, arr = unpack(open(tmp_path / 'volume_tetrapod.bin', 'rb').read())
    assert h['kind'] == 'volume' and arr['volume'].ndim == 3 and len(h['meshes']) > 0
