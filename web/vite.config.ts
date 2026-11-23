import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { createReadStream, existsSync, statSync } from 'node:fs';

// Test data for the browser test page (test.html), served at /testdata/ in dev mode only (not part of the build):
// the generated references in test/.cache/ (tests/make_web_reference.py, run by test/run_tests.py; not tracked) and
// the tracked real-data crop in ../tests/data/web/.
const testDirs = [resolve(__dirname, 'test/.cache'), resolve(__dirname, '../tests/data/web')];

export default defineConfig({
  // Relative asset paths by default (works under any path); the Pages workflow sets BASE=/torch-psf/.
  base: process.env.BASE ?? './',
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    rollupOptions: { input: { main: resolve(__dirname, 'index.html'), dense: resolve(__dirname, 'dense.html') } },
  },
  plugins: [{
    name: 'testdata',
    configureServer(server) {
      server.middlewares.use('/testdata/', (req, res, next) => {
        const rel = decodeURIComponent((req.url || '').split('?')[0].replace(/^\//, ''));
        const f = testDirs.map((d) => resolve(d, rel)).find((p, i) => p.startsWith(testDirs[i]) && existsSync(p) && statSync(p).isFile());
        if (!f) return next();
        res.setHeader('Content-Type', 'application/octet-stream');
        createReadStream(f).pipe(res);
      });
    },
  }],
});
