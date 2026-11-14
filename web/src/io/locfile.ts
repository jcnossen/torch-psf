// Localization files for the viewer, read into the rows of smlm/movie.ts (frame, x px, y px, z um, photons, bg,
// crlb_x px, crlb_y px, crlb_z um, crlb_photons, crlb_bg, loglik, chi2, at_limit; pixel centres at integers):
//  - torch_psf Dataset HDF5 (file attribute format = 'torch-psf dataset'; /localizations/<field>, /metadata
//    attributes: pixel_nm, image_size [h, w]; pos z in px of the lateral pixel size, see docs/DATASET_FORMAT.md);
//  - Picasso HDF5 ('locs' table: frame, x, y [px], photons, bg, lpx, lpy [px precision], optional z [nm]; pixel
//    size and image size from the .yaml next to it if it was dropped too: 'Pixelsize', 'Width', 'Height');
//  - ThunderSTORM / torch_psf csv (x [nm], y [nm], z [nm], frame from 1, uncertainty(_x/_y/_z) [nm], intensity
//    [photon], offset [photon]; the first pixel centre at 0.5 px), and plain csv with x, y (px) columns.
import { NCOL } from '../smlm/movie';

export interface LocFile {
  rows: Float32Array; n: number; W: number; H: number; pixel_nm: number; nFrames: number; has3d: boolean;
  format: string; notes: string[];
}

const isH5 = (f: File) => /\.(h5|hdf5|hdf)$/i.test(f.name);
export const isLocFile = (f: File) => isH5(f) || /\.(csv|txt)$/i.test(f.name);

/** files: the localization file, optionally with a Picasso .yaml. pixel_nm: used when the file does not say. */
export async function readLocFile(files: File[], pixel_nm: number): Promise<LocFile> {
  const f = files.find(isLocFile);
  if (!f) throw new Error('no .h5 / .hdf5 / .csv file');
  const yaml = files.find((x) => /\.ya?ml$/i.test(x.name));
  if (isH5(f)) return readH5(f, pixel_nm, yaml);
  return readCsv(f, pixel_nm);
}

function finish(cols: { frame: ArrayLike<number>; x: ArrayLike<number>; y: ArrayLike<number>; z?: ArrayLike<number> | null; photons?: ArrayLike<number> | null;
  bg?: ArrayLike<number> | null; cx?: ArrayLike<number> | null; cy?: ArrayLike<number> | null; cz?: ArrayLike<number> | null; chi2?: ArrayLike<number> | null; loglik?: ArrayLike<number> | null },
  pixel_nm: number, size: [number, number] | null, format: string, notes: string[]): LocFile {
  const n = cols.x.length, rows = new Float32Array(n * NCOL);
  let mx = 0, my = 0, mf = 0;
  for (let i = 0; i < n; i++) {
    const o = i * NCOL;
    rows[o] = cols.frame[i]; rows[o + 1] = cols.x[i]; rows[o + 2] = cols.y[i]; rows[o + 3] = cols.z ? cols.z[i] : 0;
    rows[o + 4] = cols.photons ? cols.photons[i] : 0; rows[o + 5] = cols.bg ? cols.bg[i] : 0;
    rows[o + 6] = cols.cx ? cols.cx[i] : NaN; rows[o + 7] = cols.cy ? cols.cy[i] : (cols.cx ? cols.cx[i] : NaN); rows[o + 8] = cols.cz ? cols.cz[i] : NaN;
    rows[o + 11] = cols.loglik ? cols.loglik[i] : NaN; rows[o + 12] = cols.chi2 ? cols.chi2[i] : NaN;
    mx = Math.max(mx, cols.x[i]); my = Math.max(my, cols.y[i]); mf = Math.max(mf, cols.frame[i]);
  }
  const [H, W] = size ?? [Math.ceil(my + 1), Math.ceil(mx + 1)];
  if (!size) notes.push(`image size not in the file: ${W} x ${H} px from the localizations`);
  return { rows, n, W, H, pixel_nm, nFrames: mf + 1, has3d: !!cols.z, format, notes };
}

let h5mod: Promise<any> | null = null;
async function h5() {
  if (!h5mod) h5mod = import('h5wasm').then(async (m) => { const h = (m as any).default ?? m; await h.ready; return h; });
  return h5mod;
}

async function readH5(file: File, pixelDefault: number, yaml?: File): Promise<LocFile> {
  const h = await h5();
  const { FS } = await h.ready;
  const name = '/tmp_' + Date.now() + '_' + file.name.replace(/[^\w.]/g, '_');
  FS.writeFile(name, new Uint8Array(await file.arrayBuffer()));
  const f = new h.File(name, 'r');
  const notes: string[] = [];
  try {
    const fmt = f.attrs?.format?.value;
    if (fmt === 'torch-psf dataset') {
      const g = f.get('localizations');
      const get = (k: string) => (g.keys().includes(k) ? g.get(k).value : null);
      const meta: Record<string, any> = {};
      if (f.keys().includes('metadata')) for (const [k, a] of Object.entries(f.get('metadata').attrs as Record<string, any>)) {
        let v = a.value; if (typeof v === 'string' && v.startsWith('json:')) v = JSON.parse(v.slice(5)); meta[k] = v;
      }
      const px = Number(meta.pixel_nm) || pixelDefault;
      if (!meta.pixel_nm) notes.push(`pixel size not in the file: ${px} nm (the pixel size field)`);
      const pos = get('pos') as Float64Array, D = g.get('pos').shape[1], n = g.get('pos').shape[0];
      const crlb = get('crlb') as Float32Array | null;
      const col = (a: ArrayLike<number> | null, k: number, D_: number, sc = 1) => a ? Float32Array.from({ length: n }, (_, i) => Number(a[i * D_ + k]) * sc) : null;
      const zUm = meta.z_unit === 'um' ? 1 : px / 1000;            // pos z in px of the lateral pixel unless z_unit says um
      const size = Array.isArray(meta.image_size) ? [Number(meta.image_size[0]), Number(meta.image_size[1])] as [number, number] : null;
      return finish({ frame: get('frame'), x: col(pos, 0, D)!, y: col(pos, 1, D)!, z: D > 2 ? col(pos, 2, D, zUm) : null, photons: get('photons'), bg: get('bg'),
        cx: crlb ? col(crlb, 0, D) : null, cy: crlb ? col(crlb, 1, D) : null, cz: crlb && D > 2 ? col(crlb, 2, D, zUm) : null, chi2: get('chi2'), loglik: get('loglik') },
        px, size, 'torch-psf dataset (HDF5)', notes);
    }
    if (f.keys().includes('locs')) {
      const d = f.get('locs');
      const names: string[] = d.metadata.compound_type.members.map((m: any) => m.name);
      const v = d.value as any[];
      const c = (k: string) => { const j = names.indexOf(k); return j < 0 ? null : Float64Array.from(v, (r: any) => Number(r[j])); };
      let px = pixelDefault, size: [number, number] | null = null;
      if (yaml) {
        const t = await yaml.text();
        const m = (k: string) => { const r = new RegExp('^\\s*' + k + '\\s*:\\s*([0-9.eE+-]+)', 'm').exec(t); return r ? Number(r[1]) : null; };
        if (m('Pixelsize')) px = m('Pixelsize')!; else notes.push(`Pixelsize not in ${yaml.name}: ${px} nm`);
        if (m('Width') && m('Height')) size = [m('Height')!, m('Width')!];
      } else notes.push(`no Picasso .yaml dropped with it: pixel size ${px} nm (the pixel size field), image size from the localizations`);
      const z = c('z');
      return finish({ frame: c('frame')!, x: c('x')!, y: c('y')!, z: z ? z.map((v) => v / 1000) : null, photons: c('photons'), bg: c('bg'),
        cx: c('lpx'), cy: c('lpy'), cz: c('lpz') ? c('lpz')!.map((v) => v / 1000) : null }, px, size, 'Picasso (HDF5)', notes);
    }
    throw new Error(`${file.name}: neither a torch-psf dataset nor a Picasso 'locs' table (keys: ${f.keys().join(', ')})`);
  } finally { f.close(); FS.unlink(name); }
}

async function readCsv(file: File, pixelDefault: number): Promise<LocFile> {
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const head = lines[0].split(/[,\t;]/).map((s) => s.replace(/^"|"$/g, '').trim().toLowerCase());
  const idx = (...names: string[]) => { for (const nm of names) { const j = head.indexOf(nm); if (j >= 0) return j; } return -1; };
  const n = lines.length - 1, data = new Float64Array(n * head.length);
  for (let i = 0; i < n; i++) { const p = lines[i + 1].split(/[,\t;]/); for (let k = 0; k < head.length; k++) data[i * head.length + k] = parseFloat(p[k]); }
  const col = (j: number, f: (v: number) => number = (v) => v) => j < 0 ? null : Float64Array.from({ length: n }, (_, i) => f(data[i * head.length + j]));
  const notes: string[] = [], px = pixelDefault;
  const xnm = idx('x [nm]', 'x_nm', 'xnm');
  if (xnm >= 0) {                                      // ThunderSTORM conventions (nm, frames from 1, first pixel centre at 0.5 px)
    notes.push(`ThunderSTORM csv: positions converted with the pixel size field (${px} nm)`);
    const fr = idx('frame');
    const unc = idx('uncertainty [nm]', 'uncertainty_xy [nm]');
    return finish({ frame: col(fr, (v) => v - 1) ?? new Float64Array(n), x: col(xnm, (v) => v / px - 0.5)!, y: col(idx('y [nm]', 'y_nm', 'ynm'), (v) => v / px - 0.5)!,
      z: col(idx('z [nm]', 'z_nm', 'znm'), (v) => v / 1000), photons: col(idx('intensity [photon]', 'intensity [photons]', 'photons')),
      bg: col(idx('offset [photon]', 'bkgstd [photon]', 'bg')), cx: col(idx('uncertainty_x [nm]') >= 0 ? idx('uncertainty_x [nm]') : unc, (v) => v / px),
      cy: col(idx('uncertainty_y [nm]') >= 0 ? idx('uncertainty_y [nm]') : unc, (v) => v / px), cz: col(idx('uncertainty_z [nm]'), (v) => v / 1000),
      chi2: col(idx('chi2')), loglik: col(idx('loglik')) }, px, null, 'ThunderSTORM csv', notes);
  }
  const xi = idx('x', 'x [px]'), yi = idx('y', 'y [px]');
  if (xi < 0 || yi < 0) throw new Error(`${file.name}: no x [nm] / x columns in the header (${head.join(', ')})`);
  notes.push('csv with x, y in camera px (pixel centres at integers), z in um, frame from 0');
  return finish({ frame: col(idx('frame')) ?? new Float64Array(n), x: col(xi)!, y: col(yi)!, z: col(idx('z', 'z [um]')), photons: col(idx('photons', 'intensity')),
    bg: col(idx('bg', 'background')), cx: col(idx('crlb_x')), cy: col(idx('crlb_y')), cz: col(idx('crlb_z')), chi2: col(idx('chi2')), loglik: col(idx('loglik')) },
    px, null, 'csv (px)', notes);
}
