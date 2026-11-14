"""Headless browser test runner for the web app (Python Playwright + Chromium with WebGPU).

    python test/run_tests.py                 # starts `vite` (dev server), runs test.html, prints the results
    python test/run_tests.py --only forward  # a subset (test names containing the string)
    python test/run_tests.py --screenshot    # also drives the app page with the test stack and saves a screenshot

The reference data are generated into test/.cache/ by ../tests/make_web_reference.py (needs torch and SIMCODE's
vectorial PSF code, ~2 min on a GPU) when missing or when the generator has changed.
Needs `pip install playwright && python -m playwright install chromium`. The GPU flags below give the NVIDIA (Vulkan)
adapter on a headless Linux box; on a machine without a GPU Chromium falls back to SwiftShader (correct but slow).
Exit code 0 when all tests pass.
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
# (--disable-gpu-compositing: with GPU compositing, headless screenshots of the page with the WebGL view come out
# garbled; WebGPU compute is not affected)
FLAGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist',
         '--disable-gpu-compositing']


def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p


def start_vite(port):
    p = subprocess.Popen(['npx', 'vite', '--port', str(port), '--strictPort', '--host', '127.0.0.1'], cwd=WEB,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(100):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1); return p
        except Exception:
            time.sleep(0.2)
    p.kill(); raise RuntimeError('vite did not start')


GEN = os.path.join(WEB, '..', 'tests', 'make_web_reference.py')
CACHE = os.path.join(HERE, '.cache')


GEN_SMLM = os.path.join(WEB, '..', 'tests', 'make_web_smlm_reference.py')


def ensure_references():
    """(Re)generate the reference data in test/.cache/ when missing or when a generator has changed."""
    import hashlib
    for gen, out, stamp_name in ((GEN, CACHE, 'stamp.txt'), (GEN_SMLM, os.path.join(CACHE, 'smlm'), 'stamp_smlm.txt')):
        h = hashlib.sha256(open(gen, 'rb').read()).hexdigest()
        stamp = os.path.join(CACHE, stamp_name)
        if os.path.exists(stamp) and open(stamp).read().strip() == h:
            continue
        print(f'generating the reference data ({os.path.basename(gen)}, a few minutes on a GPU) ...', flush=True)
        subprocess.run([sys.executable, gen, '--out', out], check=True)
        os.makedirs(CACHE, exist_ok=True)
        open(stamp, 'w').write(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--timeout', type=float, default=900)
    ap.add_argument('--screenshot', action='store_true')
    ap.add_argument('--no-tests', action='store_true')
    ap.add_argument('--url', default=None, help='use a running server instead of starting vite')
    a = ap.parse_args()
    ensure_references()
    from playwright.sync_api import sync_playwright
    port = free_port()
    srv = None if a.url else start_vite(port)
    base = a.url or f'http://127.0.0.1:{port}'
    ok = True
    try:
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=True, args=FLAGS)
            pg = br.new_page(viewport={'width': 1500, 'height': 1000})
            pg.on('console', lambda m: print('  [browser]', m.text) if m.type in ('error', 'warning') or
                  m.text.startswith('#') else None)
            pg.on('pageerror', lambda e: print('  [pageerror]', e))
            if not a.no_tests:
                pg.goto(f'{base}/test.html?only={a.only}')
                pg.wait_for_function('window.__testsDone === true', timeout=a.timeout * 1000)
                res = pg.evaluate('window.__testResults')
                os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
                json.dump(res, open(os.path.join(HERE, 'out', 'results.json'), 'w'), indent=1)
                print(f"GPU: {res.get('gpu')}")
                for t in res['tests']:
                    print(f"{'PASS' if t['pass'] else 'FAIL'}  {t['name']:40s} {t['ms']:8.0f} ms  {t.get('msg', '')}")
                ok = all(t['pass'] for t in res['tests']) and len(res['tests']) > 0
                print('ALL PASS' if ok else 'SOME FAILED')
            if a.screenshot:
                from screenshot import drive_app
                drive_app(pg, base, os.path.join(WEB, 'docs'))
            br.close()
    finally:
        if srv:
            import signal
            os.killpg(srv.pid, signal.SIGTERM)   # npx and its node child
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
