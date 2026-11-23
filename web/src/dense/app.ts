import { getDevice } from '../gpu/device';
import { fitMovie, exportPsf, type MovieFit, type MovieObserver, type MovieFrameObserver } from './movie';
import './dense.css';
export function download(el: HTMLElement, name: string, data: BlobPart, type: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.textContent = name;
  el.append(a);
}
export function mount(
  title: string,
  finish?: (fit: MovieFit, el: HTMLElement) => Promise<void>,
  live?: { start(): void; update: MovieObserver; frame?: MovieFrameObserver; end(): void },
) {
  const root = document.querySelector('main')!;
  root.innerHTML =
    '<h1></h1><p id="purpose"></p><form><section id="inputs"></section><details><summary>Acquisition settings</summary><section id="settings"></section></details><button>Fit movie</button></form><p role="status"></p><article></article>';
  root.querySelector('h1')!.textContent = title;
  root.querySelector('#purpose')!.textContent = finish
    ? 'Fit a bead movie to recover the PSF, illumination patterns and field maps. Use the repeating phase order A0°, A120°, A240°, B0°, B120°, B240°.'
    : 'Recover a vectorial PSF, per-bead intensities and frame drift from a dense bead movie.';
  const files: (File | null)[] = [null, null];
  ['Bead movie · TYX', 'Calibration Z-stack · optional'].forEach((name, i) => {
    const label = document.createElement('label');
    label.className = 'drop';
    label.innerHTML =
      '<strong></strong><span>Drop a TIFF or browse</span><input type="file" accept=".tif,.tiff">';
    label.querySelector('strong')!.textContent = name;
    const set = (file?: File) => {
      files[i] = file ?? null;
      label.querySelector('span')!.textContent = file?.name ?? 'Drop a TIFF or browse';
    };
    label.ondragover = (e) => e.preventDefault();
    label.ondrop = (e) => {
      e.preventDefault();
      set(e.dataTransfer?.files[0]);
    };
    const input = label.querySelector('input')!;
    input.onchange = () => set(input.files?.[0]);
    root.querySelector('#inputs')!.append(label);
  });
  const fields = [
    ['Pixel size (nm)', 108.33],
    ['Numerical aperture', 1.49],
    ['Emission wavelength (nm)', 700],
    ['Immersion index', 1.518],
    ['Sample index', 1.33],
    ['Bead diameter (nm)', 40],
    ['Signed stack step (nm)', ''],
    ['Camera offset (ADU)', 0],
    ['Photons per ADU', 1],
  ];
  const inputs = fields.map(([name, value], i) => {
    const label = document.createElement('label');
    label.textContent = String(name);
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.required = i !== 6;
    input.value = String(value);
    if (i === 6) input.placeholder = 'From TIFF metadata';
    label.append(input);
    root.querySelector('#settings')!.append(label);
    return input;
  });
  const form = root.querySelector('form')!,
    status = root.querySelector('[role=status]')!,
    out = root.querySelector('article')!;
  form.onsubmit = async (e) => {
    e.preventDefault();
    if (!files[0]) {
      status.textContent = 'Select a movie TIFF.';
      return;
    }
    const button = form.querySelector('button')!;
    button.disabled = true;
    for (const a of out.querySelectorAll('a')) URL.revokeObjectURL(a.href);
    out.replaceChildren();
    const log = (s: string) => {
      status.textContent = s;
    };
    try {
      live?.start();
      const [px, NA, wl, ni, ns, diameterNm, stepNm, offset, gain] = inputs.map((i) => Number(i.value));
      if (Math.min(px, NA, wl, ni, ns, gain) <= 0 || NA > ni || diameterNm < 0)
        throw Error('Check the optical and camera settings.');
      log('Opening WebGPU device');
      const { device } = await getDevice();
      const r = await fitMovie(
        device,
        files[0],
        files[1],
        {
          optics: { pixel_um: px / 1000, NA, wavelength_um: wl / 1000, n_imm: ni, n_med: ns },
          diameterNm,
          stepNm,
          offset,
          gain,
          period: finish ? 6 : 1,
        },
        log,
        live?.update,
        live?.frame,
      );
      log('Writing PSF archive');
      download(out, 'psf.zip', await exportPsf(device, r), 'application/zip');
      download(
        out,
        'bead-fit.json',
        JSON.stringify({
          coordinates:
            'xy in pixels; drift rows [dx_px,dy_px,stage_focus_nm relative to bead plane]; intensity[frame][bead] in photons',
          ...r,
        }),
        'application/json',
      );
      const summary = document.createElement('p');
      summary.textContent =
        r.xy.length +
        ' fitted candidates · ' +
        r.drift.length +
        ' frames · ' +
        (r.stackCalibrated
          ? 'Stack-calibrated pupil.'
          : 'Movie-only pupil. The sign of Z can be ambiguous near focus.') +
        ' ' +
        r.boundaryFrames +
        ' frames at the axial fit limit.';
      out.append(summary);
      if (finish) await finish(r, out);
      log('Fit complete');
    } catch (err) {
      log(err instanceof Error ? err.message : String(err));
    } finally {
      live?.end();
      button.disabled = false;
    }
  };
}
