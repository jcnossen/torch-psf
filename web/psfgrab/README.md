# PSF grab

A small static web page that collects movies for PSF estimation: recordings made with a phone camera (for example
through a microscope eyepiece adapter), focus sweeps, and dropped TIFF stacks, image sequences or videos. It shows a
live power spectrum and a sharpness value, and exports each recording as a zip for analysis in Python. The PSF
estimation itself (NA from the power spectrum, mode classification) is not part of this page yet.

Everything runs in the browser. Nothing is uploaded: the frames stay in the page's memory until you download the
zip.

## Build

```
cd web
npm ci
npm run build:psfgrab                       # -> web/dist-psfgrab/ (ignored by git)
npm run build:psfgrab -- --outDir /some/dir # or any other folder
npm run dev:psfgrab                         # dev server, http://localhost:5174/
npm run test:psfgrab                        # node unit checks (FFT, spectrum, TIFF and zip writers), no browser
```

The build is separate from the main app (`npm run build`, `vite.config.ts`, GitHub Pages): it uses
`vite.psfgrab.config.ts`, builds only `psfgrab/index.html` and writes relative asset paths (`./assets/...`), so the
folder works at the root of a server or under any sub-path. The output is about 80 kB (20 kB gzipped): `index.html` and
`assets/` (one script, one worker, one stylesheet).

## Deploy (static site, e.g. https://psfgrab.jcnossen.nl)

Copy the contents of the build folder to the web root of the server:

```
npm run build:psfgrab
rsync -av --delete dist-psfgrab/ user@vps:/var/www/psfgrab/
```

The camera works only over **HTTPS** (or on `localhost`): browsers do not offer `getUserMedia` on plain HTTP.
Opening files works either way.

A minimal [Caddy](https://caddyserver.com/) site (Caddy gets and renews the TLS certificate automatically; the DNS
record of the name must point at the server and ports 80 and 443 must be open):

```
# /etc/caddy/Caddyfile
psfgrab.jcnossen.nl {
	root * /var/www/psfgrab
	file_server
	encode zstd gzip
	header {
		# the camera is used by this page itself only
		Permissions-Policy "camera=(self)"
		X-Content-Type-Options nosniff
		Referrer-Policy no-referrer
	}
	@assets path /assets/*
	header @assets Cache-Control "public, max-age=31536000, immutable"
	@html path / /index.html
	header @html Cache-Control "no-cache"
}
```

then `sudo systemctl reload caddy`. Any other static server (nginx, Apache) works the same way; no server-side code
and no special headers are needed.

## Using it

- **Camera**: asks for the camera (the back camera by default; other cameras, the requested resolution and frame
  rate are selectable). The card lists what the camera reports through `MediaStreamTrack.getCapabilities()` /
  `getSettings()` (resolution, frame rate, focus mode and distance, exposure mode and time, ISO, white balance,
  zoom, torch, ...) and shows sliders / switches for the ones it can set.
- **Tap the image** to move the crop. Blue: the recorded crop; dashed: the analysed square (a power of two, at most
  the analysis crop size, centred in the recorded crop).
- **Spectrum and sharpness** (live, a few times per second, in a web worker): the radially averaged log10 power
  spectrum of the analysed square per colour channel (mean subtracted, 2D Hann window, power normalised so that
  white noise of variance s^2 reads s^2), against spatial frequency in cycles per pixel (0 .. 0.5 = Nyquist);
  sharpness = var(Laplacian) / mean^2 of the channel mean; and the share of the power above half the Nyquist
  frequency per channel. Exact definitions: `src/psfgrab/spectrum.ts`.
- **Record**: N frames of the crop (optionally downscaled by 2 or 4) as uncompressed 8-bit RGB. The estimate below
  the settings shows the memory it takes; keep it well under a few hundred MB on a phone. Optionally a full-
  resolution still through `ImageCapture.takePhoto()` (Chrome / Android; not in Safari or Firefox).
- **Focus sweep** (only shown when the camera supports `focusMode: 'manual'` with a `focusDistance` range, in
  practice Chrome on Android): steps the focus distance from / to in the given number of steps, waits for the
  reported focus distance to reach the target and the settle time to pass, skips two frames, and records the given
  number of frames per step. Frames are not averaged in the browser (`step` in the JSON groups them). Focus is set
  back to the previous mode afterwards. The focus distance is the value the browser reports, in its units (meters
  per the spec; some devices report the requested value rather than a measured one).
- **Files**: a TIFF stack (the main app's reader: uint8 / uint16 / float32, uncompressed, Deflate, LZW, PackBits;
  an ImageJ hyperstack with `channels=3`, as this page writes, is read as R, G, B), a set of PNG / JPEG images
  (sorted by name, numbers compared numerically), or a video (MP4 / WebM, whatever the browser decodes; frames are
  read by seeking to `(i + 0.5) / fps`, the frame rate measured during a short muted playback and editable). Same
  spectrum and sharpness display per frame, "Analyse all frames" for a sharpness trace, and "Export as TIFF +
  JSON" for a frame range with the same crop / downscale.
- **Download .zip** (or **Share** where the browser can share files, e.g. to Drive on a phone).

## The zip

`psfgrab_<label>_<yyyymmdd-hhmmss>.zip`:

| file | content |
|---|---|
| `frames.tif` | ImageJ hyperstack, little endian, uncompressed, one page per frame and channel in the order frame x C + channel. RGB sources (camera, video, images) are uint8 with C = 3 (R, G, B as separate channels, not interleaved RGB samples); grayscale TIFF input stays grayscale (C = 1), uint8 if it was 8-bit, else float32. Recordings and file exports are written as `frames=N`, focus sweeps as `slices=N`. |
| `meta.json` | everything known (see below) |
| `still.jpg` | the ImageCapture still, when taken |

Reading it in Python:

```python
import json, zipfile, io, tifffile
z = zipfile.ZipFile('psfgrab_....zip')
meta = json.loads(z.read('meta.json'))
stack = tifffile.imread(io.BytesIO(z.read('frames.tif')))   # (N, 3, H, W) uint8, or (N, H, W) for grayscale
focus = [f['settings'].get('focusDistance') for f in meta['frames']]
```

`meta.json` (format `psfgrab/1`): `app` (version, page URL), `created`, `label`, `notes`, `device` (user agent,
`navigator.userAgentData` incl. the phone model where Chrome reports it, screen, CPU count, device memory),
`source` (camera: track label, settings, capabilities and constraints at the start; files: names, sizes, dtype,
video duration and frame rate), `recording` (mode `frames` / `focus_sweep` / `file`, frame count, channels, dtype,
output size, the crop `region` in source pixels, `downscale`, the TIFF layout, and for sweeps the sweep parameters
and the settings before), `still` (size, requested size, photo capabilities and settings), and `frames`: one entry
per frame with `t_ms` (since the start of the recording), the `requestVideoFrameCallback` metadata where the browser
gives it (`mediaTime`, `presentedFrames`, `captureTime`, `expectedDisplayTime`, ...), the track `settings` at that
frame (`focusDistance`, `exposureTime`, `iso`, `zoom`, `colorTemperature`, ... as reported), for sweeps `step`,
`step_frame`, `target_focusDistance`, `settled`, `settle_ms`, for file exports the source frame (and the video
seek time), and the per-frame `sharpness`, `hfRatio` and channel `mean` computed after the recording.

## Code

`psfgrab/index.html`, `src/psfgrab/`: `main.ts` (UI), `camera.ts` (getUserMedia, capabilities, ImageCapture),
`sources.ts` (TIFF / image / video files), `grab.ts` (canvas read-back, crop, binning), `fft.ts` + `spectrum.ts`
(FFT, power spectrum, radial average, sharpness; pure TS), `analysis.worker.ts` + `analyzer.ts`, `plot.ts`,
`tiffstack.ts` (hyperstack writer as a Blob of the frame planes), `zipblob.ts` (zip of Blobs, streamed CRC and
deflate; uses `src/io/zip.ts`). Unit checks: `test/psfgrab_node.ts`.
