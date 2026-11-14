"""Drive the app page headlessly: landing view (tetrapod preset), then a bead fit of the generated tetrapod-like test
stack (test/.cache/stack_tetrapod.tif), stop controls, exports. Used by `python test/run_tests.py --screenshot`; writes
web/docs/landing.png / result.png (palette-compressed, small) and checks the exported files in Python."""
import json, os, time

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')


def compress(path, width=1100, colors=128):
    """downscale and palette-quantize a screenshot (keeps the committed PNGs small)"""
    from PIL import Image
    im = Image.open(path).convert('RGB')
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f'{os.path.basename(path)}: {os.path.getsize(path) / 1024:.0f} kB')


def shot(pg, path, **kw):
    # pause the auto-rotation while capturing (headless captures of a continuously redrawn WebGL canvas can come out
    # garbled), then resume it
    pg.evaluate('window.__view.controls.autoRotate = false')
    time.sleep(0.5)
    pg.screenshot(path=path, **kw)
    pg.evaluate('window.__view.controls.autoRotate = window.__view.autoRotate')


def drive_app(pg, base, outdir):
    os.makedirs(outdir, exist_ok=True)
    pg.goto(f'{base}/index.html')
    pg.wait_for_function("window.__presetShown === 'tetrapod'", timeout=60000)
    time.sleep(1.5)
    shot(pg, os.path.join(outdir, 'landing.png'))
    compress(os.path.join(outdir, 'landing.png'))
    print('landing screenshot:', pg.inner_text('#gpu-status'))
    for name in ['astig', 'focus']:
        pg.select_option('#preset', name)
        pg.wait_for_function(f"window.__presetShown === '{name}'", timeout=60000)
    print('presets ok')
    # the generated tetrapod-like stack with its known acquisition parameters
    t = json.load(open(os.path.join(CACHE, 'tetrapod.json')))
    vals = {'f-na': t['NA'], 'f-wl': t['wavelength_um'], 'f-nimm': t['n_imm'], 'f-nmed': t['n_med'],
            'f-px': t['pixel_um'], 'f-dz': t['dz_um'], 'f-gain': t['gain'], 'f-offset': t['offset'],
            'f-ze': t['ze_um'], 'f-roi': t['R']}
    for k, v in vals.items():
        pg.fill('#' + k, str(v))
    pg.select_option('#f-sign', str(int(t['sign'])))
    pg.uncheck('#f-fitdz')
    pg.set_input_files('#file', os.path.join(CACHE, 'stack_tetrapod.tif'))
    # stop controls: Abort discards, "Stop and keep current estimate" gives a partial result
    pg.click('#run')
    pg.wait_for_function("document.getElementById('ptext').textContent.startsWith('fit:')", timeout=60000)
    pg.click('#abort')
    pg.wait_for_function("document.getElementById('ptext').textContent === 'aborted'", timeout=60000)
    print('abort ok')
    pg.click('#run')
    pg.wait_for_function("document.getElementById('ptext').textContent.startsWith('fit:')", timeout=60000)
    pg.click('#enough')
    pg.wait_for_function('window.__fitDone === true', timeout=60000)
    print('enough ok:', pg.inner_text('#ptext'))
    pg.evaluate('window.__fitDone = false')
    t0 = time.time()
    pg.click('#run')
    pg.wait_for_function('window.__fitDone === true', timeout=300000)
    print(f'fit done in {time.time() - t0:.1f} s (page):', pg.inner_text('#ptext'))
    print(pg.inner_text('#res-table'))
    time.sleep(1.0)
    shot(pg, os.path.join(outdir, 'result.png'), full_page=True)
    compress(os.path.join(outdir, 'result.png'), width=900, colors=96)
    files = {}
    for btn in ['x-json', 'x-knots', 'x-tif']:
        with pg.expect_download() as dl:
            pg.click('#' + btn)
        d = dl.value
        p = os.path.join(outdir, '..', 'test', 'out', d.suggested_filename)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        d.save_as(p)
        files[btn] = p
    import numpy as np, tifffile
    j = json.load(open(files['x-json']))
    z = np.load(files['x-knots'])
    t = tifffile.imread(files['x-tif'])
    print('exports: json', j['format'], len(j['zernike_waves']), 'Zernikes; knots', z['knots'].shape, float(z['hz']),
          '; model tif', t.shape, t.dtype, f'sum of the middle plane {t[len(t) // 2].sum():.3f}')
