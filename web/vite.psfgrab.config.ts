import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// PSF grab (psfgrab/index.html): a separate static page, built on its own into dist-psfgrab/ (or --outDir) with
// relative asset paths, so the folder can be copied to the root (or any path) of a web server.
//   npm run build:psfgrab                      -> web/dist-psfgrab/
//   npm run build:psfgrab -- --outDir /some/dir
//   npm run dev:psfgrab                        -> http://localhost:5174/psfgrab/
// The dev server serves from web/ (not web/psfgrab/): index.html loads ../src/psfgrab/main.ts, which the browser
// resolves against the page URL, so with root = web/psfgrab it would point above the server root and 404.
export default defineConfig(({ command }) => ({
  root: command === 'serve' ? __dirname : resolve(__dirname, 'psfgrab'),
  base: './',
  publicDir: false,
  worker: { format: 'es' },
  server: { port: 5174, fs: { allow: [__dirname] } },
  build: {
    target: 'es2022',
    outDir: resolve(__dirname, 'dist-psfgrab'),
    emptyOutDir: true,
    rollupOptions: { input: { index: resolve(__dirname, 'psfgrab/index.html') } },
  },
}));
