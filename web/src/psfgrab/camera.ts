// Camera access for PSF grab: getUserMedia, the MediaStreamTrack capabilities / settings (the image-capture
// extensions focusMode / focusDistance / exposureTime / zoom / torch ... are not in lib.dom, so they are handled as
// plain records), constraint helpers, and ImageCapture.takePhoto() where the browser has it.

export type Rec = Record<string, unknown>;

export interface CameraOpts { deviceId?: string; width: number; height: number; frameRate: number }

export async function openCamera(o: CameraOpts): Promise<MediaStream> {
  if (!window.isSecureContext) throw new Error('the camera needs a secure context (HTTPS or localhost)');
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('this browser has no camera API (navigator.mediaDevices.getUserMedia)');
  const video: MediaTrackConstraints = {
    width: { ideal: o.width }, height: { ideal: o.height }, frameRate: { ideal: o.frameRate },
  };
  if (o.deviceId) video.deviceId = { exact: o.deviceId };
  else video.facingMode = { ideal: 'environment' };
  return navigator.mediaDevices.getUserMedia({ video, audio: false });
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  try { return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput'); } catch { return []; }
}

export const capsOf = (t: MediaStreamTrack): Rec => {
  try { return (typeof t.getCapabilities === 'function' ? t.getCapabilities() : {}) as Rec; } catch { return {}; }
};
export const settingsOf = (t: MediaStreamTrack): Rec => { try { return t.getSettings() as Rec; } catch { return {}; } };

/** The settings recorded per frame (when the track reports them). */
export const FRAME_KEYS = ['width', 'height', 'frameRate', 'focusMode', 'focusDistance', 'exposureMode', 'exposureTime',
  'exposureCompensation', 'iso', 'whiteBalanceMode', 'colorTemperature', 'zoom', 'torch', 'pointsOfInterest'];

export function frameSettings(t: MediaStreamTrack): Rec {
  const s = settingsOf(t), out: Rec = {};
  for (const k of FRAME_KEYS) if (s[k] !== undefined) out[k] = s[k];
  return out;
}

export interface Range { min: number; max: number; step?: number }
export const isRange = (v: unknown): v is Range => !!v && typeof v === 'object' && typeof (v as Range).min === 'number' && typeof (v as Range).max === 'number';

/** Capabilities shown in the table, in this order (others are listed after them). */
export const SHOWN_KEYS = ['width', 'height', 'aspectRatio', 'frameRate', 'resizeMode', 'facingMode', 'focusMode', 'focusDistance',
  'exposureMode', 'exposureTime', 'exposureCompensation', 'iso', 'whiteBalanceMode', 'colorTemperature', 'zoom', 'pan', 'tilt',
  'torch', 'brightness', 'contrast', 'saturation', 'sharpness'];

export function fmtCap(v: unknown): string {
  if (isRange(v)) return `${fmtNum(v.min)} .. ${fmtNum(v.max)}${v.step ? ` (step ${fmtNum(v.step)})` : ''}`;
  if (Array.isArray(v)) return v.length ? v.join(', ') : '(none)';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (v === undefined) return '';
  if (typeof v === 'number') return fmtNum(v);
  return typeof v === 'object' ? JSON.stringify(v) : String(v);
}
export function fmtNum(x: number): string {
  if (!isFinite(x)) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(2);
  return String(Math.round(x * 1000) / 1000);
}

/** Manual focus as the focus sweep needs it: focusMode 'manual' and a focusDistance range. */
export function manualFocus(caps: Rec): Range | null {
  const fm = caps.focusMode, fd = caps.focusDistance;
  return Array.isArray(fm) && fm.includes('manual') && isRange(fd) && fd.max > fd.min ? fd : null;
}

/** Apply one image-capture setting (with the matching manual mode for the value settings). */
export async function applySetting(t: MediaStreamTrack, key: string, value: unknown): Promise<void> {
  const c: Rec = { [key]: value };
  if (key === 'focusDistance') c.focusMode = 'manual';
  if (key === 'exposureTime') c.exposureMode = 'manual';
  if (key === 'colorTemperature') c.whiteBalanceMode = 'manual';
  await t.applyConstraints({ advanced: [c as MediaTrackConstraintSet] });
}

// ---- ImageCapture (Chrome / Android; not in Safari or Firefox)
interface ImageCaptureLike {
  takePhoto(s?: Rec): Promise<Blob>;
  getPhotoCapabilities?(): Promise<Rec>;
  getPhotoSettings?(): Promise<Rec>;
}
type ImageCaptureCtor = new (t: MediaStreamTrack) => ImageCaptureLike;
const IC = (globalThis as unknown as { ImageCapture?: ImageCaptureCtor }).ImageCapture;
export const hasImageCapture = (): boolean => typeof IC === 'function';

export interface Still { blob: Blob; photoCapabilities: Rec | null; photoSettings: Rec | null; requested: Rec; ms: number }

/** A full-resolution still (largest imageWidth / imageHeight the photo capabilities list). */
export async function takeStill(t: MediaStreamTrack): Promise<Still> {
  if (!IC) throw new Error('ImageCapture is not available in this browser');
  const ic = new IC(t);
  let pc: Rec | null = null, ps: Rec | null = null;
  try { pc = (await ic.getPhotoCapabilities?.()) ?? null; } catch { /* optional */ }
  try { ps = (await ic.getPhotoSettings?.()) ?? null; } catch { /* optional */ }
  const req: Rec = {};
  if (pc && isRange(pc.imageWidth) && isRange(pc.imageHeight)) { req.imageWidth = pc.imageWidth.max; req.imageHeight = pc.imageHeight.max; }
  const t0 = performance.now();
  const blob = await ic.takePhoto(req);
  return { blob, photoCapabilities: plain(pc), photoSettings: plain(ps), requested: req, ms: performance.now() - t0 };
}

/** JSON-safe copy of a settings / capabilities object (MediaSettingsRange etc. are not plain objects everywhere). */
export function plain<T>(x: T): T {
  if (x === null || x === undefined) return x;
  const seen = (v: unknown): unknown => {
    if (v === null || typeof v !== 'object') return v;
    if (Array.isArray(v)) return v.map(seen);
    const o: Rec = {};
    for (const k in v as Rec) { const e = (v as Rec)[k]; if (typeof e !== 'function') o[k] = seen(e); }
    return o;
  };
  return seen(x) as T;
}
