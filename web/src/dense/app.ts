import { getDevice } from '../gpu/device';
import {
  fitMovie,
  exportMovieReport,
  type MovieFit,
  type MovieObserver,
  type PhotometrySnapshot,
  type MovieFrameObserver,
} from './movie';
import { moviePupilReport } from './report';
import './dense.css';
import {exampleMovie,fieldReport} from './field-example';
import {movieInspector} from './movie-inspector';
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
  live?: {
    start(): void;
    status?(message: string): void;
    period?: () => number;
    update: MovieObserver;
    frame?: MovieFrameObserver;
    photometry?: (snapshot: PhotometrySnapshot) => void;
    end(): void;
  },
  selectedInput?: ()=>{file?:File; framesPerStep?:number},
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
  let exampleTruth:ReturnType<typeof exampleMovie>|undefined;
  let fieldControl:HTMLInputElement|undefined,scanControl:HTMLInputElement|undefined,repeats:HTMLInputElement|undefined;
  if(!finish){
    const options=document.createElement('section');
    options.innerHTML='<label><input class="field" type="checkbox" checked> Fit field-dependent aberrations</label><label><input class="scan" type="checkbox"> Input movie is a planar Z stack</label><label>Frames per Z step <input class="repeats" type="number" min="1" step="1" value="1" disabled></label>';
    const inputsBox=root.querySelector<HTMLElement>('#inputs')!;inputsBox.after(options);
    fieldControl=options.querySelector<HTMLInputElement>('.field')!;
    scanControl=options.querySelector<HTMLInputElement>('.scan')!;
    repeats=options.querySelector<HTMLInputElement>('.repeats')!;
    scanControl.onchange=()=>{repeats!.disabled=!scanControl!.checked;};
    exampleTruth=exampleMovie(options,()=>{
      const [px,NA,wl,n_imm,n_med,diameterNm,stepNm,offset,gain]=inputs.map(i=>Number(i.value));
      return {optics:{pixel_um:px/1000,NA,wavelength_um:wl/1000,n_imm,n_med},diameterNm,stepNm,offset,gain};
    },file=>{
      files[0]=file;inputsBox.querySelector('span')!.textContent=file.name;
      files[1]=null;inputsBox.querySelectorAll('span')[1].textContent='Drop a TIFF or browse';
      scanControl!.checked=false;repeats!.disabled=true;
    });
  }
  form.onsubmit = async (e) => {
    e.preventDefault();
    const sourceInput=selectedInput?.();
    const movieFile=sourceInput ? sourceInput.file : files[0];
    if (!movieFile) {
      status.textContent = 'Select a movie TIFF.';
      return;
    }
    const button = form.querySelector<HTMLButtonElement>('button:not([type=button])')!;
    button.disabled = true;
    for (const a of out.querySelectorAll('a')) URL.revokeObjectURL(a.href);
    out.replaceChildren();
    const log = (s: string) => {
      status.textContent = s;
      live?.status?.(s);
    };
    try {
      live?.start();
      const [px, NA, wl, ni, ns, diameterNm, stepNm, offset, gain] = inputs.map((i) => Number(i.value));
      if (Math.min(px, NA, wl, ni, ns, gain) <= 0 || NA > ni || diameterNm < 0)
        throw Error('Check the optical and camera settings.');
      log('Opening WebGPU device');
      const { device } = await getDevice();
      const renderPupil = moviePupilReport(out);
      const inspector=finish?undefined:movieInspector(out);
      const stackFile = selectedInput ? null : files[1];
      const r = await fitMovie(
        device,
        movieFile,
        stackFile,
        {
          optics: { pixel_um: px / 1000, NA, wavelength_um: wl / 1000, n_imm: ni, n_med: ns },
          diameterNm,
          field:fieldControl?.checked,
          framesPerStep:sourceInput?.framesPerStep ?? (scanControl?.checked?repeats!.valueAsNumber:undefined),
          stepNm,
          offset,
          gain,
          period: live?.period?.() ?? (finish ? 6 : 1),
        },
        log,
        async (dev, iteration) => {
          renderPupil(iteration.model);
          await live?.update(dev, iteration);
        },
        live?.frame ?? inspector?.frame,
        live?.photometry,
      );
      if(inspector){ await inspector.finish(device,movieFile,r); renderPupil(r.model); fieldReport(out,r,exampleTruth?.(movieFile)); }
      log('Writing PSF archive');
      const source = stackFile ? movieFile.name + ' / calibration: ' + stackFile.name : movieFile.name;
      const { bytes, pdf } = await exportMovieReport(device, r, source);
      const report = out.querySelector<HTMLElement>('#psf-report')!;
      const open = document.createElement('a');
      open.textContent = 'Open PSF report (PDF)';
      open.target = '_blank';
      open.rel = 'noopener';
      open.href = URL.createObjectURL(new Blob([pdf], { type: 'application/pdf' }));
      report.append(open);
      const save = document.createElement('a');
      save.textContent = 'Download PDF';
      save.href = open.href;
      save.download = 'fit_report.pdf';
      report.append(save);
      download(out, 'psf.zip', bytes, 'application/zip');
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
          : r.diversityScores ? 'Movie-only pupil refined across complete illumination cycles.' : 'Coarse movie-only pupil. The sign of Z can be ambiguous near focus.') +
        ' ' +
        r.focusWindowShifts +
        ' axial lookup window shifts.';
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
