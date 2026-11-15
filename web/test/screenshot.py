"""Drive the app page headlessly: landing view (the astigmatic preset and the simulated demo movie), then a bead fit of the generated tetrapod-like test
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
    pg.wait_for_function("window.__presetShown === 'astig' && window.__demoReady === true", timeout=120000)
    time.sleep(1.5)
    shot(pg, os.path.join(outdir, 'landing.png'))
    compress(os.path.join(outdir, 'landing.png'))
    print('landing screenshot:', pg.inner_text('#gpu-status'))
    # 3D view interaction: hover highlights a slice plane, dragging moves it (orbit disabled meanwhile); z exaggeration
    # without isosurface recomputation
    pg.evaluate('window.__view.controls.autoRotate = false')
    time.sleep(1.0)                             # let the orbit damping settle
    box = pg.locator('#view3d canvas').bounding_box()
    cx, cy = box['x'] + box['width'] * 0.56, box['y'] + box['height'] * 0.6   # off the planes' common centre
    pg.mouse.move(cx, cy)
    hov = pg.evaluate('window.__view.outlines.findIndex((o) => o.visible)')
    before = pg.evaluate('[...window.__view.slicePos]')
    cam0 = pg.evaluate('window.__view.camera.position.toArray()')
    pg.mouse.down(); pg.mouse.move(cx, cy - 60, steps=8); pg.mouse.move(cx + 40, cy - 90, steps=8)
    during = pg.evaluate('window.__view.controls.enabled')
    pg.mouse.up()
    after = pg.evaluate('[...window.__view.slicePos]')
    cam1 = pg.evaluate('window.__view.camera.position.toArray()')
    sliders = pg.evaluate("['sx', 'sy', 'sz'].map((i) => +document.getElementById(i).value / 100)")
    ax = [2, 1, 0][hov] if hov >= 0 else 0      # slice meshes are [xy, xz, yz]
    ok = hov >= 0 and after[ax] != before[ax] and not during and pg.evaluate('window.__view.controls.enabled') \
        and max(abs(a - b) for a, b in zip(cam0, cam1)) < 0.05 and abs(sliders[ax] - after[ax]) < 2e-3
    t = pg.evaluate('''() => { const v = window.__view; let t0 = performance.now(); for (let i = 0; i < 10; i++) v.setZExag(1 + i / 10);
        const tz = (performance.now() - t0) / 10; v.setZExag(1); t0 = performance.now(); v.isoBuilt = false; v.rebuild(); return [tz, performance.now() - t0]; }''')
    print(f'drag: hovered plane {hov} (axis {ax}), slice {before[ax]:.3f} -> {after[ax]:.3f}, orbit disabled while dragging, '
          f'slider in sync: {"ok" if ok else "FAILED"}; z exaggeration update {t[0]:.1f} ms vs full rebuild {t[1]:.0f} ms')
    if not ok:
        raise RuntimeError(f'slice-plane drag check failed: cam {cam0} -> {cam1}, sliders {sliders}, after {after}')
    for name in ['tetrapod', 'focus']:
        pg.select_option('#preset', name)
        pg.wait_for_function(f"window.__presetShown === '{name}'", timeout=60000)
    print('presets ok')
    # the generated tetrapod-like stack with its known acquisition parameters
    t = json.load(open(os.path.join(CACHE, 'tetrapod.json')))
    pg.click('.tabs button[data-tab=stack]')
    pg.evaluate("document.querySelectorAll('details.adv').forEach((d) => { d.open = true; })")
    # pixel size and wavelength are entered in nm; the stack-fit fields live in the (hidden) Fit PSF step
    vals = {'o-na': t['NA'], 'o-wl': 1000 * t['wavelength_um'], 'o-nimm': t['n_imm'], 'o-nmed': t['n_med'],
            'o-px': 1000 * t['pixel_um'], 'f-dz': t['dz_um'], 'o-gain': t['gain'], 'o-offset': t['offset'],
            'f-ze': t['ze_um'], 'f-roi': t['R'], 'f-sign': int(t['sign'])}
    pg.evaluate("v => { for (const [k, x] of Object.entries(v)) document.getElementById(k).value = String(+x.toPrecision(10)); "
                "document.getElementById('f-fitdz').checked = false; }", vals)
    pg.evaluate("window.__setStep('fit')")
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
    pg.evaluate("window.__setStep('export')")
    for btn in ['x-json', 'x-knots', 'x-tif', 'x-pdf']:
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
    pdf = open(files['x-pdf'], 'rb').read()
    if not (pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF')):
        raise RuntimeError('fit_report.pdf is not a PDF')
    print(f'fit report: {len(pdf) // 1024} kB, {pdf.count(b"/Type /Page ")} pages')
