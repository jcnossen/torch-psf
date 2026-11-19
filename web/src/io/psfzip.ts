// psf.zip (docs/PSF_FORMAT.md): read 0.2.x and 0.3.x, write 0.2.0 box vectorial / Catmull-Rom spline models.
// Read channel 0 and referenced arrays. Integrated splines render from their own coefficients even without a source.
import { integratedRenderer, vectorialRenderer, type ImportedRenderer } from '../psf/imported';
import { splineRender } from '../psf/zstack';
import type { Optics } from '../psf/vectorial';
import type { SplineModel, VectorialModel, ZStack } from '../psf/zstack';
import { NdArray, parseNpy, writeNpy } from './npy';
import { pyFloat, writeImageJHyperstack } from './imagejtif';
import { ZipReader, isZip, writeZip } from './zip';

export const FORMAT = 'torch-psf', ZIP_VERSION = '0.2.0';
export const JSON_NAME = 'psf.json', ZSTACK_NAME = 'zstack.tif', PREVIEW_NAME = 'preview.png', REPORT_NAME = 'fit_report.pdf';
export const UNITS = { x: 'px', y: 'px', z: 'um', I: 'photons', bg: 'photons / px', phi: 'rad' };
export const ZSTACK_VALUE = 'pixel-integrated normalized PSF (render): fraction of the photons in a camera pixel centred at the sample';

// ------------------------------------------------------------------------------------------------ writing
/** The 0.1.0 keys of a vectorial model (torch_psf VectorialPSF._file_dict), as a 0.2.0 channel `model` block. */
export function vectorialBlock(m: VectorialModel): Record<string, unknown> {
  const o = m.optics;
  return {
    optics: { pixel_um: o.pixel_um, NA: o.NA, wavelength_um: o.wavelength_um, n_imm: o.n_imm, n_med: o.n_med },
    zernike_noll: [...m.noll], zernike_waves: Array.from(m.zcoef, Number),
    apod: [...(m.apod ?? [0, 0, 0])], saf: [...(m.saf ?? [0, 0, 0, 0])], focus_um: m.focus_um,
    psf_blur_px2: null, psf_dipole_M: null, diattenuation: null,
    npupil: m.npupil, oversample: m.oversample, roisize: m.roisize, z_range_um: m.z_range_um ? [...m.z_range_um] : null,
    pixelsize_um: o.pixel_um, wavelength_um: o.wavelength_um, NA: o.NA, n_imm: o.n_imm, n_med: o.n_med,
  };
}

/** The 'spline' dict of a CubicSplinePSF (to_dict). */
export function splineGridDict(s: Omit<SplineModel, 'knots'>): Record<string, unknown> {
  return { roisize: s.roisize, z_range: [...s.z_range], oversample: s.oversample, dz_um: s.dz_um, zpad: s.zpad, xy_limit: s.xy_limit,
    c: s.c, z0_um: s.z0_um, nz: s.nz, u0_px: s.u0_px, pixel_um: s.pixel_um };
}

/** Camera block (torch_psf Camera.to_dict; scalars only). */
export interface CameraInfo { gain_e_per_adu: number; offset_adu: number; readnoise_e?: number; excess_noise?: number; type?: 'scmos' | 'emccd' }
export function cameraBlock(c: CameraInfo): Record<string, unknown> {
  const F = c.excess_noise ?? 1;
  return { gain_e_per_adu: c.gain_e_per_adu, offset_adu: c.offset_adu, readnoise_e: c.readnoise_e ?? 0, type: c.type ?? (F > 1.5 ? 'emccd' : 'scmos'), maps: null, excess_noise: F };
}

export interface PsfZipParts {
  vectorial: VectorialModel;            // the model; the source of the spline when `spline` is given
  spline?: SplineModel | null;          // write channel 0 as this spline (type 'spline', source = the vectorial model)
  camera?: CameraInfo | null;
  provenance?: Record<string, unknown>;
  /** estimator report for psf.json `report` (JSON values) and the sections of the PDF */
  report?: { estimator?: string; results?: Record<string, unknown>; settings?: Record<string, unknown>; sections?: string[] };
  zstack: ZStack;                       // rendered from the channel model (psf/zstack.ts)
  png: Uint8Array;                      // preview.png
  pdf: Uint8Array;                      // fit_report.pdf
  channelName?: string;                 // default 'ch0'
}

/** psf.json (0.2.0) and the arrays {member: npy bytes} of a single-channel file. */
export function makePsfZipJson(p: PsfZipParts): { meta: Record<string, unknown>; arrays: Record<string, Uint8Array> } {
  const name = p.channelName ?? 'ch0', arrays: Record<string, Uint8Array> = {};
  const vb = vectorialBlock(p.vectorial);
  let ch: Record<string, unknown>;
  if (p.spline) {
    const s = p.spline, n = (2 * s.c + 1) * s.oversample;
    if (s.knots.length !== s.nz * n * n) throw new Error(`spline knots: ${s.knots.length} values for [${s.nz}, ${n}, ${n}]`);
    arrays[`${name}/knots.npy`] = writeNpy(s.knots, [s.nz, n, n]);
    ch = { name, type: 'spline', model: { spline: splineGridDict(s) }, arrays: { knots: `${name}/knots.npy` },
      source: { type: 'vectorial', model: vb, arrays: {} } };
  } else ch = { name, type: 'vectorial', model: vb, arrays: {} };
  ch.mapping = { type: 'identity' };
  ch.intensity_ratio = 1.0;
  const psf_type = p.spline ? 'spline' : 'vectorial';
  const meta: Record<string, unknown> = { format: FORMAT, version: ZIP_VERSION, psf_type, n_dims: 3, emitter_params: ['x', 'y', 'z', 'I', 'bg'], units: { ...UNITS } };
  meta.optics = { ...(vb.optics as Optics) };
  if (p.camera) meta.camera = cameraBlock(p.camera);
  meta.provenance = p.provenance ?? {};
  meta.channels = [ch];
  const z = p.zstack, px = p.vectorial.optics.pixel_um;
  meta.zstack = { file: ZSTACK_NAME, axes: 'ZCYX', shape: [z.Z, z.C, z.H, z.W], dtype: 'float32', z_um: [...z.z_um], sample_px: z.sample_px,
    sample_um: px ? px * z.sample_px : null, channels: [name], position: { x: 0.0, y: 0.0 }, params: {}, value: ZSTACK_VALUE };
  meta.preview = { file: PREVIEW_NAME };
  const rep: Record<string, unknown> = { file: REPORT_NAME, sections: p.report?.sections ?? ['summary', 'aberrations', 'pupil', 'psf'] };
  if (p.report?.estimator) {
    rep.estimator = p.report.estimator;
    rep.results = p.report.results ?? {};
    if (p.report.settings) rep.settings = p.report.settings;
  }
  meta.report = rep;
  return { meta, arrays };
}

/** json.dumps of Python (', ' / ': ' separators, numbers as Python floats, ASCII escapes): the Info string as
 *  torch_psf writes it (all numbers of the Info block are floats). */
export function pyJson(v: unknown): string {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'number') return pyFloat(v);
  if (typeof v === 'boolean') return String(v);
  if (typeof v === 'string') return JSON.stringify(v).replace(/[\u0080-\uffff]/g, (c) => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
  if (Array.isArray(v)) return '[' + v.map(pyJson).join(', ') + ']';
  return '{' + Object.entries(v as Record<string, unknown>).map(([k, x]) => `${pyJson(k)}: ${pyJson(x)}`).join(', ') + '}';
}

/** zstack.tif of psf.json's zstack block. */
export function zstackTif(z: ZStack, meta: Record<string, any>): Uint8Array {
  const zb = meta.zstack;
  const info = pyJson({ format: FORMAT, version: ZIP_VERSION, psf_type: meta.psf_type, z_um: zb.z_um, channels: zb.channels, sample_px: zb.sample_px });
  return writeImageJHyperstack(z.stack, z.Z, z.C, z.H, z.W, {
    unit: 'um', spacing: z.Z > 1 ? z.z_um[1] - z.z_um[0] : undefined, resolution: zb.sample_um ? 1 / zb.sample_um : undefined, info });
}

/** The psf.zip bytes: psf.json first, then the arrays, zstack.tif, preview.png (stored), fit_report.pdf; deflated
 *  where CompressionStream is available; fixed member times. Returns the bytes and the psf.json dict. */
export async function writePsfZip(p: PsfZipParts): Promise<{ bytes: Uint8Array; meta: Record<string, unknown> }> {
  const { meta, arrays } = makePsfZipJson(p);
  const enc = new TextEncoder();
  const members = [
    { name: JSON_NAME, data: enc.encode(JSON.stringify(meta, null, 1)) },
    ...Object.entries(arrays).map(([name, data]) => ({ name, data })),
    { name: ZSTACK_NAME, data: zstackTif(p.zstack, meta) },
    { name: PREVIEW_NAME, data: p.png, store: true },
    { name: REPORT_NAME, data: p.pdf },
  ];
  return { bytes: await writeZip(members), meta };
}

// ------------------------------------------------------------------------------------------------ reading
export interface PsfFileInfo {
  kind: 'psf.zip';
  renderer?: ImportedRenderer;
  optics?: Optics; noll: number[]; zcoef: number[]; focus_um: number;
  roisize?: number; npupil?: number; oversample?: number; z_range_um?: [number, number] | null;
  channelType: string;                  // type of channel 0 ('vectorial', 'spline', ...)
  spline?: SplineModel | null;          // the spline of a spline channel (with its knots, from a zip)
  camera?: Record<string, unknown> | null;
  meta?: Record<string, any>;           // psf.json
  arrays: Record<string, NdArray>;      // the parsed members (by member name)
  notes: string[];                      // what the web app does not use or could not read
}

function checkVersion(meta: Record<string, any>) {
  if (meta.format !== FORMAT) throw new Error(`not a torch-psf psf.json (format ${JSON.stringify(meta.format)})`);
  const v = String(meta.version ?? '');
  if (!/^0\.[23]\.\d+$/.test(v)) throw new Error(`psf.zip version ${v}: this reader supports 0.2.x and 0.3.x`);
}

/** Member names that psf.json references: {member: owner} (channel arrays, sources, mapping grids, field
 *  coefficients, camera maps, camera extras). */
export function referencedMembers(meta: Record<string, any>): Record<string, string> {
  const out: Record<string, string> = {};
  const add = (blk: Record<string, unknown> | undefined | null, owner: string) => {
    for (const [k, v] of Object.entries(blk ?? {})) if (typeof v === 'string') out[v] = `${owner}.${k}`;
  };
  (meta.channels ?? []).forEach((c: any, i: number) => {
    const nm = c.name ?? `ch${i}`;
    add(c.arrays, nm);
    if (c.source) add(c.source.arrays, `${nm}.source`);
    if (c.mapping?.type === 'grid') add({ fwd: c.mapping.fwd, inv: c.mapping.inv }, `${nm}.mapping`);
    if (c.field && typeof c.field.coef === 'string') out[c.field.coef] = `${nm}.field.coef`;
  });
  if (meta.source) add(meta.source.arrays, 'source');
  if (meta.camera) { add(meta.camera.arrays, 'camera'); add(meta.camera.extra_arrays, 'camera.extra'); }
  return out;
}

const VEC_UNUSED = ['psf_blur_px2', 'psf_dipole_M', 'psf_dipole_w', 'diattenuation', 'psf_dipole_M_angle', 'psf_blur_angle_px2'];

function integrationMode(value: unknown): 'box' | 'spectral' | 'gauss_legendre' {
  if (value == null || value === 'box' || value === 'midpoint') return 'box';
  if (value === 'spectral' || value === 'exact') return 'spectral';
  if (value === 'gauss_legendre' || value === 'gauss') return 'gauss_legendre';
  throw new Error(`unsupported pixel_integration ${JSON.stringify(value)}`);
}

/** Channel 0 of a psf.json (channel 0, or the source of a spline channel). arrays: parsed members. */
export function psfFromMeta(meta: Record<string, any>, arrays: Record<string, NdArray> = {}, notes: string[] = []): Omit<PsfFileInfo, 'kind'> {
  checkVersion(meta);
  const chans: any[] = meta.channels ?? [];
  if (!chans.length) throw new Error('psf.json has no channels');
  const types = new Set(chans.map((c) => c.type));
  if ([...types].every((t) => t === 'source_output' || t === 'phase_spline')) throw new Error('a coupled (4Pi) model: not supported by the web app');
  if (chans.length > 1) notes.push(`${chans.length} channels: channel 0 (${chans[0].name}) is used`);
  const ch = chans[0];
  const common = { channelType: ch.type, camera: meta.camera ?? null, meta, arrays, notes };
  for (const block of [meta.source, ...chans.flatMap(c => [c, c.source])]) integrationMode(block?.model?.pixel_integration);
  let spline: SplineModel | null = null;
  if (ch.type === 'spline' && ch.model?.spline) {
    const g = ch.model.spline, K = arrays[ch.arrays?.knots], n = (2 * g.c + 1) * g.oversample;
    if (K?.data instanceof Float32Array && K.shape.join() === [g.nz, n, n].join() && K.data.length === g.nz * n * n)
      spline = { ...g, pixel_um: g.pixel_um ?? null, knots: K.data };
  }
  if (ch.type === 'integrated_spline' || (ch.type === 'spline' && meta.version.startsWith('0.3.'))) {
    let renderer: ImportedRenderer;
    if (ch.type === 'integrated_spline') renderer = integratedRenderer(ch.model?.integrated_spline, ch.arrays ?? {}, arrays);
    else {
      if (!spline) throw new Error('spline: missing or invalid knots');
      const s = spline;
      renderer = { feature: 'spline', roisize: s.roisize, z_range: s.z_range, render: (x, y, z) => splineRender(s, x, y, z) };
    }
    const source = ch.source?.type === 'vectorial' ? ch.source.model : undefined;
    return { ...common, optics: source?.optics, noll: source?.zernike_noll ?? [], zcoef: source?.zernike_waves ?? [],
      focus_um: source?.focus_um ?? 0, roisize: renderer.roisize, z_range_um: renderer.z_range, renderer, spline };
  }
  const blk = ch.type === 'vectorial' ? ch : ch.source;
  if (blk?.type !== 'vectorial') throw new Error(`channel 0 is a ${ch.type} model without a vectorial source: the web app needs a vectorial PSF`);
  if (blk !== ch) notes.push(`channel 0 is a ${ch.type} model: its vectorial source is used`);
  const m = blk.model ?? {};
  if (!Array.isArray(m.zernike_noll) || !Array.isArray(m.zernike_waves)) throw new Error('the vectorial model has no zernike_noll / zernike_waves');
  const o = m.optics ?? meta.optics ?? {};
  const optics: Optics = { pixel_um: o.pixel_um ?? m.pixelsize_um, NA: o.NA ?? m.NA, wavelength_um: o.wavelength_um ?? m.wavelength_um, n_imm: o.n_imm ?? m.n_imm, n_med: o.n_med ?? m.n_med };
  const mode = integrationMode(m.pixel_integration);
  let renderer: ImportedRenderer | undefined;
  if (mode !== 'box') {
    if (ch.type !== 'vectorial') throw new Error(`cannot render ${ch.type} from its ${mode} source`);
    const maps: Record<string, Float32Array> = {};
    for (const [key, member] of Object.entries(blk.arrays ?? {})) {
      if (key !== 'pupil_amp' && key !== 'pupil_phase') continue;
      const arr = arrays[String(member)];
      if (!(arr?.data instanceof Float32Array) || arr.shape.join() !== [m.npupil, m.npupil].join() || arr.data.length !== m.npupil ** 2)
        throw new Error(`${mode}: expected ${key} float32 [${m.npupil}, ${m.npupil}]`);
      maps[key] = arr.data;
    }
    renderer = vectorialRenderer({ ...m, optics, noll: m.zernike_noll, zcoef: m.zernike_waves, focus_um: m.focus_um ?? 0 }, mode, maps);
  }
  if (!renderer) {
    for (const k of VEC_UNUSED) if (m[k] !== null && m[k] !== undefined) notes.push(`${k} is not used by the web app`);
    for (const k of Object.keys(blk.arrays ?? {})) notes.push(`pupil map ${k} is not used by the web app`);
  }
  if ((m.apod ?? []).some((v: number) => v !== 0) || (m.saf ?? []).some((v: number) => v !== 0)) notes.push('apod / saf terms are kept for rendering only');
  if (ch.field) notes.push(`field dependence (${ch.field.type}) is not used: the Zernikes at the reference point`);
  return {
    optics, noll: m.zernike_noll.map(Number), zcoef: m.zernike_waves.map(Number), focus_um: Number(m.focus_um ?? 0),
    roisize: m.roisize, npupil: m.npupil, oversample: m.oversample, z_range_um: m.z_range_um ?? null,
    ...common, spline, renderer,
  };
}

/** Parse the referenced .npy members of a zip; members missing, unreadable or of a dtype parseNpy does not support
 *  are skipped with a note. skipOwners: owner prefixes not to parse at all (e.g. 'camera.extra'). */
export async function readReferencedArrays(z: ZipReader, meta: Record<string, any>, notes: string[], skipOwners: string[] = []): Promise<Record<string, NdArray>> {
  const out: Record<string, NdArray> = {};
  for (const [member, owner] of Object.entries(referencedMembers(meta))) {
    if (skipOwners.some((s) => owner.startsWith(s))) continue;
    if (!z.has(member)) { notes.push(`${owner}: member ${member} is missing`); continue; }
    try { out[member] = parseNpy((await z.read(member)).buffer as ArrayBuffer); }
    catch (e) { notes.push(`${owner}: ${member} skipped (${(e as Error).message})`); }
  }
  return out;
}

export async function readPsfZip(buf: ArrayBuffer | Uint8Array, opts: { cameraExtras?: boolean } = {}): Promise<PsfFileInfo> {
  const z = new ZipReader(buf);
  if (!z.has(JSON_NAME)) throw new Error('not a psf.zip (no psf.json member)');
  const meta = JSON.parse(await z.text(JSON_NAME));
  checkVersion(meta);
  const notes: string[] = [];
  const arrays = await readReferencedArrays(z, meta, notes, opts.cameraExtras ? [] : ['camera.extra']);
  return { kind: 'psf.zip', ...psfFromMeta(meta, arrays, notes) };
}

/** A PSF file the app can start from: a psf.zip. */
export async function readPsfFile(f: Blob | ArrayBuffer | Uint8Array): Promise<PsfFileInfo> {
  const b = f instanceof Uint8Array ? f : new Uint8Array(f instanceof ArrayBuffer ? f : await f.arrayBuffer());
  if (!isZip(b)) throw new Error('not a zip file');
  return readPsfZip(b);
}
