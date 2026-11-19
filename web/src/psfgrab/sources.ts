// Dropped / opened files as frame sources: a TIFF stack (the main app's streaming reader, io/tiff.ts; an ImageJ
// hyperstack with channels=3, as PSF grab writes, is read as R, G, B), an image sequence (PNG / JPEG / WebP ...,
// sorted by name with numbers compared numerically) or a video (MP4 / WebM, decoded by a <video> element: frame i is
// shown by seeking to (i + 0.5) / fps and waiting for the 'seeked' event and requestVideoFrameCallback).
import { TiffMovie } from '../io/tiff';
import { Region, Planes, grabDrawable, cropBin, toU8 } from './grab';

export interface FileSource {
  kind: 'tiff' | 'images' | 'video';
  name: string;
  W: number; H: number;
  n: number;                    // frames
  channels: 1 | 3;
  u8: boolean;                  // 8-bit data (exported as uint8; else float32)
  info: Record<string, unknown>;
  /** Make frame i current; returns per-frame metadata for the export JSON. */
  load(i: number): Promise<Record<string, unknown>>;
  /** The current frame's region, downscaled by f (integer). */
  grab(r: Region, f: number): Planes;
  /** Draw the current frame into a W x H canvas context (for the view). */
  draw(c: CanvasRenderingContext2D): void;
  close(): void;
}

const IMG_RE = /\.(png|jpe?g|webp|gif|bmp|avif)$/i, VID_RE = /\.(mp4|m4v|mov|webm|mkv|ogv)$/i, TIF_RE = /\.tiff?$/i;

export function classify(files: File[]): 'tiff' | 'images' | 'video' | null {
  if (!files.length) return null;
  const f = files[0];
  if (TIF_RE.test(f.name) || f.type === 'image/tiff') return 'tiff';
  if (VID_RE.test(f.name) || f.type.startsWith('video/')) return 'video';
  if (IMG_RE.test(f.name) || f.type.startsWith('image/')) return 'images';
  return null;
}

export async function openFiles(files: File[], onMsg: (m: string) => void): Promise<FileSource> {
  const k = classify(files);
  if (k === 'tiff') return TiffSource.open(files.filter((f) => TIF_RE.test(f.name) || f.type === 'image/tiff'), onMsg);
  if (k === 'video') return VideoSource.open(files[0], onMsg);
  if (k === 'images') return ImageSeqSource.open(files.filter((f) => IMG_RE.test(f.name) || f.type.startsWith('image/')));
  throw new Error(`unsupported file type: ${files[0]?.name ?? '(none)'} (TIFF, PNG / JPEG sequence, MP4 / WebM)`);
}

const natural = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });

// ---------- TIFF
class TiffSource implements FileSource {
  kind = 'tiff' as const;
  name = ''; W = 0; H = 0; n = 0; channels: 1 | 3 = 1; u8 = false; info: Record<string, unknown> = {};
  private mv!: TiffMovie;
  private cur: Float32Array[] = [];
  private view: { lo: number[]; hi: number[] } | null = null;
  private img: ImageData | null = null;

  static async open(files: File[], onMsg: (m: string) => void): Promise<TiffSource> {
    const s = new TiffSource();
    s.mv = await TiffMovie.open(files, onMsg);
    const ch = /channels=(\d+)/.exec(s.mv.description);
    s.channels = ch && +ch[1] === 3 && s.mv.n % 3 === 0 ? 3 : 1;
    s.n = s.mv.n / s.channels; s.W = s.mv.W; s.H = s.mv.H; s.u8 = s.mv.dtype === 'uint8';
    s.name = files.map((f) => f.name).join(', ');
    s.info = { files: files.map((f) => ({ name: f.name, size: f.size, lastModified: f.lastModified })), dtype: s.mv.dtype,
      pages: s.mv.n, width: s.W, height: s.H, channels: s.channels, compression: s.mv.compression, description: s.mv.description.slice(0, 2000) };
    return s;
  }
  async load(i: number) {
    const C = this.channels, a = await this.mv.read(i * C, (i + 1) * C), n = this.W * this.H;
    this.cur = Array.from({ length: C }, (_, c) => a.subarray(c * n, (c + 1) * n));
    this.img = null;
    return { index: i, pages: Array.from({ length: C }, (_, c) => i * C + c) };
  }
  grab(r: Region, f: number): Planes {
    const planes = this.cur.map((p) => cropBin(p, this.W, r, f));
    return { w: Math.floor(r.w / f), h: Math.floor(r.h / f), planes: this.u8 ? planes.map(toU8) : planes };
  }
  draw(c: CanvasRenderingContext2D) {
    if (!this.cur.length) return;
    if (!this.view) {                        // display range from the first frame shown: 8-bit as is, else 0.1 / 99.9 %
      const lo: number[] = [], hi: number[] = [];
      for (const p of this.cur) {
        if (this.u8) { lo.push(0); hi.push(255); continue; }
        const s = Float32Array.from(p.length > 200000 ? p.filter((_, i) => i % Math.ceil(p.length / 200000) === 0) : p).sort();
        lo.push(s[Math.floor(0.001 * (s.length - 1))]); hi.push(Math.max(s[Math.floor(0.999 * (s.length - 1))], lo[lo.length - 1] + 1e-6));
      }
      this.view = { lo, hi };
    }
    if (!this.img) {
      const n = this.W * this.H, im = new ImageData(this.W, this.H), d = im.data, { lo, hi } = this.view;
      for (let c = 0; c < 3; c++) {
        const k = this.channels === 3 ? c : 0, p = this.cur[k], a = lo[k], s = 255 / (hi[k] - a);
        for (let i = 0; i < n; i++) d[4 * i + c] = (p[i] - a) * s;
      }
      for (let i = 0; i < n; i++) d[4 * i + 3] = 255;
      this.img = im;
    }
    c.putImageData(this.img, 0, 0);
  }
  close() { this.cur = []; this.img = null; }
}

// ---------- image sequence
class ImageSeqSource implements FileSource {
  kind = 'images' as const;
  name = ''; W = 0; H = 0; n = 0; channels = 3 as const; u8 = true; info: Record<string, unknown> = {};
  private files: File[] = [];
  private bmp: ImageBitmap | null = null;

  static async open(files: File[]): Promise<ImageSeqSource> {
    const s = new ImageSeqSource();
    s.files = [...files].sort((a, b) => natural(a.name, b.name));
    const b0 = await createImageBitmap(s.files[0]);
    s.W = b0.width; s.H = b0.height; s.n = s.files.length; s.bmp = b0;
    s.name = s.n === 1 ? s.files[0].name : `${s.files[0].name} .. ${s.files[s.n - 1].name} (${s.n} images)`;
    s.info = { files: s.files.map((f) => ({ name: f.name, size: f.size, type: f.type, lastModified: f.lastModified })), width: s.W, height: s.H };
    return s;
  }
  async load(i: number) {
    const f = this.files[i];
    const b = await createImageBitmap(f);
    if (b.width !== this.W || b.height !== this.H) { b.close(); throw new Error(`${f.name}: ${b.width} x ${b.height}, the first image is ${this.W} x ${this.H}`); }
    this.bmp?.close(); this.bmp = b;
    return { index: i, file: f.name };
  }
  grab(r: Region, f: number): Planes { return grabDrawable(this.bmp!, r, Math.floor(r.w / f), Math.floor(r.h / f)); }
  draw(c: CanvasRenderingContext2D) { if (this.bmp) c.drawImage(this.bmp, 0, 0); }
  close() { this.bmp?.close(); this.bmp = null; }
}

// ---------- video
type VFCMeta = { mediaTime: number; presentedFrames: number; width: number; height: number };
type VFCVideo = HTMLVideoElement & { requestVideoFrameCallback?: (cb: (now: number, m: VFCMeta) => void) => number };

const once = (el: EventTarget, ev: string, ms: number) => new Promise<boolean>((res) => {
  const t = setTimeout(() => { el.removeEventListener(ev, h); res(false); }, ms);
  const h = () => { clearTimeout(t); el.removeEventListener(ev, h); res(true); };
  el.addEventListener(ev, h);
});

/** The next presented frame's metadata (requestVideoFrameCallback), or null after `ms` or without the API. */
export function nextFrameMeta(v: HTMLVideoElement, ms: number): Promise<VFCMeta | null> {
  const vv = v as VFCVideo;
  if (!vv.requestVideoFrameCallback) return Promise.resolve(null);
  return new Promise((res) => {
    const t = setTimeout(() => res(null), ms);
    vv.requestVideoFrameCallback!((_n, m) => { clearTimeout(t); res(m); });
  });
}

class VideoSource implements FileSource {
  kind = 'video' as const;
  name = ''; W = 0; H = 0; n = 0; channels = 3 as const; u8 = true; info: Record<string, unknown> = {};
  fps = 30; fpsMeasured = false;
  private v!: HTMLVideoElement;
  private url = '';

  static async open(file: File, onMsg: (m: string) => void): Promise<VideoSource> {
    const s = new VideoSource();
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.preload = 'auto';
    // in the document (invisible) so that every browser decodes, seeks and fires requestVideoFrameCallback for it
    v.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;z-index:-1';
    document.body.appendChild(v);
    s.url = URL.createObjectURL(file); v.src = s.url; s.v = v;
    const ok = await Promise.race([once(v, 'loadeddata', 15000), once(v, 'error', 15000).then(() => false)]);
    if (!ok || !v.videoWidth) { s.close(); throw new Error(`${file.name}: this browser cannot decode the video${v.error ? ` (${v.error.message || 'code ' + v.error.code})` : ''}`); }
    s.W = v.videoWidth; s.H = v.videoHeight; s.name = file.name;
    onMsg('measuring the frame rate');
    await s.measureFps();
    s.setFps(s.fps);
    s.info = { file: { name: file.name, size: file.size, type: file.type, lastModified: file.lastModified }, width: s.W, height: s.H,
      duration_s: v.duration, fps: s.fps, fps_measured: s.fpsMeasured };
    return s;
  }
  /** Median frame interval over ~1 s of muted playback (requestVideoFrameCallback mediaTime). */
  private async measureFps() {
    const v = this.v as VFCVideo;
    if (!v.requestVideoFrameCallback) return;
    const times: number[] = [];
    let stop = false;
    const cb = (_n: number, m: VFCMeta) => { times.push(m.mediaTime); if (!stop && times.length < 40) v.requestVideoFrameCallback!(cb); };
    v.requestVideoFrameCallback(cb);
    try { await v.play(); } catch { return; }
    await new Promise((r) => setTimeout(r, 1000));
    stop = true; v.pause(); v.currentTime = 0;
    await once(v, 'seeked', 2000);
    const d = times.slice(1).map((t, i) => t - times[i]).filter((x) => x > 1e-4).sort((a, b) => a - b);
    if (d.length >= 4) { const med = d[d.length >> 1]; if (med > 0) { this.fps = Math.round(1000 / med) / 1000; this.fpsMeasured = true; } }
  }
  setFps(fps: number) {
    this.fps = fps;
    this.n = Math.max(1, Math.floor(this.v.duration * fps + 1e-6));
    this.info.fps = fps;
  }
  async load(i: number) {
    const v = this.v, t = Math.min((i + 0.5) / this.fps, Math.max(0, v.duration - 1e-3));
    let m: VFCMeta | null = null;
    if (Math.abs(v.currentTime - t) > 1e-6 || v.readyState < 2) {
      const meta = nextFrameMeta(v, 400);
      const seeked = once(v, 'seeked', 5000);
      v.currentTime = t;
      await seeked;
      m = await meta;
    }
    return { index: i, seek_s: t, mediaTime: m?.mediaTime ?? null, currentTime: v.currentTime };
  }
  grab(r: Region, f: number): Planes { return grabDrawable(this.v, r, Math.floor(r.w / f), Math.floor(r.h / f)); }
  draw(c: CanvasRenderingContext2D) { c.drawImage(this.v, 0, 0, this.W, this.H); }
  close() { this.v.pause(); this.v.removeAttribute('src'); this.v.load(); this.v.remove(); URL.revokeObjectURL(this.url); }
}
