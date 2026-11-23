import { pupilMaps } from '../report/fromapp';
import { zernikeName } from '../report/fitreport';
import { cmapRgb, dataRange, imageRgb, type Cmap, type Image2D } from '../report/plot';
import type { VectorialModel } from '../psf/zstack';

function mapFigure(title: string, unit: string, map: Image2D, cmap: Cmap, range: [number, number]) {
  const figure = document.createElement('figure');
  const caption = document.createElement('figcaption');
  caption.textContent = title;
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-label', title + ' in ' + unit);
  const { rgb, width, height } = imageRgb(map, cmap, ...range, { nanColor: [233, 239, 240] });
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!,
    image = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    image.data[4 * i] = rgb[3 * i];
    image.data[4 * i + 1] = rgb[3 * i + 1];
    image.data[4 * i + 2] = rgb[3 * i + 2];
    image.data[4 * i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  const bar = document.createElement('div');
  bar.className = 'pupil-scale';
  const colors = Array.from(
    { length: 17 },
    (_, i) =>
      'rgb(' +
      cmapRgb(cmap, i / 16)
        .map(Math.round)
        .join(',') +
      ')',
  );
  bar.style.background = 'linear-gradient(to right,' + colors.join(',') + ')';
  const ticks = document.createElement('div');
  ticks.className = 'pupil-ticks';
  for (const text of [range[0].toFixed(2), unit, range[1].toFixed(2)]) {
    const tick = document.createElement('span');
    tick.textContent = text;
    ticks.append(tick);
  }
  figure.append(caption, canvas, bar, ticks);
  return figure;
}

/** Refresh only when the fitted pupil changes. Frame focus changes the PSF image but leaves the pupil unchanged. */
export function moviePupilReport(out: HTMLElement) {
  const section = document.createElement('section');
  section.id = 'psf-report';
  section.className = 'dense-pupil';
  section.hidden = true;
  out.append(section);
  let previous = '';
  return (m: VectorialModel) => {
    const key = JSON.stringify([m.optics, m.noll, Array.from(m.zcoef), m.npupil]);
    if (key === previous) return;
    const maps = pupilMaps(m.optics, m.noll, Array.from(m.zcoef), m.npupil);
    section.innerHTML =
      '<h2>PSF report</h2><div class="pupil-grid"></div><p class="pupil-axes"></p><h3>Zernike aberrations</h3><table><thead><tr><th>Noll index</th><th>Mode</th><th>Coefficient (waves RMS)</th><th aria-label="Signed coefficient relative to largest absolute mode"></th></tr></thead><tbody></tbody></table>';
    section
      .querySelector('.pupil-grid')!
      .append(
        mapFigure('Pupil amplitude', 'relative', maps.amplitude, 'viridis', [0, 1]),
        mapFigure('Pupil phase', 'waves', maps.phase, 'diverging', dataRange(maps.phase.data, true)),
      );
    section.querySelector('.pupil-axes')!.textContent =
      'Pupil coordinates: x increases right, y increases up. Aperture radius = NA ' + m.optics.NA + '.';
    const body = section.querySelector('tbody')!;
    const largest = Math.max(...Array.from(m.zcoef, Math.abs), 1e-12);
    m.noll.forEach((n, i) => {
      const row = body.insertRow(),
        v = m.zcoef[i];
      for (const value of [String(n), zernikeName(n), (v > 0 ? '+' : '') + v.toFixed(4)])
        row.insertCell().textContent = value;
      const signedBar = document.createElement('div');
      signedBar.className = 'coefficient-axis';
      signedBar.setAttribute('aria-hidden', 'true');
      const bar = document.createElement('i');
      const percent = (Math.abs(v) / largest) * 50;
      bar.style.left = (v < 0 ? 50 - percent : 50) + '%';
      bar.style.width = percent + '%';
      bar.style.background = v < 0 ? '#2166ac' : '#b2182b';
      signedBar.append(bar);
      row.insertCell().append(signedBar);
    });
    section.hidden = false;
    previous = key;
  };
}
