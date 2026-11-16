// Point-cloud view of the localizations (Reconstruction panel), on the principles of webSMLM's point-cloud renderer
// and its tilt / rotate view, rendered on the GPU (gpu/cloud.ts):
//  - screen resolution for the current view (pan / zoom); each localization a Gaussian sized in nm (its CRLB,
//    projected through the rotation and capped, or one fixed blur) integrated over the screen pixel, additive;
//  - turntable 3D: rotate about z, tilt about the horizontal screen axis (0 = top view, 90 = side view), z
//    exaggeration; colour = world z;
//  - live: localizations are appended to GPU buffers with spare capacity as they stream in (no reallocation until
//    full), a grid index keeps zooming in cheap, a per-point flag applies the CRLB filter;
//  - display maximum global: 99.5 % of the density per nm^2 of a 256 x 256 top-view grid of the whole field, kept
//    while panning / zooming / orbiting (updated incrementally as points arrive);
//  - point cap (default 2,000,000 visible points): above it a fixed hash-ranked subset, weighted by visible / drawn.
import { NCOL } from '../smlm/movie';
import { GpuCloud } from '../gpu/cloud';
import { trackPinch } from './gestures';

export interface CloudOptions {
  mode: 'precision' | 'fixed';
  blurNm: number;          // fixed mode: sigma (nm); precision mode: minimum sigma
  capNm: number;           // precision mode: maximum sigma (nm)
  maxCrlbNm: number;       // drop localizations with crlb_x > this (0 = keep all)
  colorZ: boolean;
  zlo: number; zhi: number; // colour range (um)
  zExag: number;           // z exaggeration in tilted views
}

// fixed rank in [0, 1) from x, y, frame (the same localization always gets the same rank)
function rank(x: number, y: number, f: number): number {
  let h = Math.imul(Math.round(x * 1024) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(Math.round(y * 1024) + 0x632be5ab, 0xc2b2ae35) ^ Math.imul(f | 0, 0x27d4eb2f);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12; h = Math.imul(h, 0x297a2d39); h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

const HG = 256;   // top-view grid for the global display maximum

export class PointCloudView {
  tilt = 0; rot = 0;               // degrees
  scale = 0;                       // nm per screen pixel (0 = fit on the next draw)
  cx = 0; cy = 0;                  // view centre in the rotated, tilted plane (nm, relative to the field centre)
  maxPoints = 2_000_000;           // point cap (visible points)
  private gpu: GpuCloud;
  private overlay: HTMLCanvasElement;
  private geo = { W: 1, H: 1, pixel_nm: 100 };
  private hist = new Float32Array(HG * HG);
  private vmax = 0;
  private filter = 0;
  private pending = false;
  private foldTimer = 0;
  onChange: () => void = () => {};
  last = { n: 0, visible: 0, drawn: 0, nmPerPx: 0, capacity: 0, reallocations: 0, rebuilds: 0, ms: 0 };

  constructor(dev: GPUDevice, private cv: HTMLCanvasElement, private opts: () => CloudOptions) {
    this.gpu = new GpuCloud(dev, cv);
    this.overlay = document.createElement('canvas');
    this.overlay.className = 'cloud-overlay';
    cv.parentElement!.style.position = 'relative';
    cv.parentElement!.appendChild(this.overlay);
    let last: { x: number; y: number; orbit: boolean; id: number } | null = null;
    // two fingers: pinch zooms about their midpoint, moving them pans
    const pinch = trackPinch(cv, (f, mx, my, dx, dy) => {
      this.cx += mx * this.scale * (1 - f) - dx * this.scale; this.cy += my * this.scale * (1 - f) - dy * this.scale;
      this.scale *= f; this.request();
    });
    cv.addEventListener('pointerdown', (e) => {
      if (!last) last = { x: e.clientX, y: e.clientY, orbit: e.shiftKey || e.button === 2, id: e.pointerId };
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', (e) => {
      if (!last || e.pointerId !== last.id) return;
      if (pinch.pinching()) { last.x = e.clientX; last.y = e.clientY; return; }
      const dx = e.clientX - last.x, dy = e.clientY - last.y; last.x = e.clientX; last.y = e.clientY;
      const dpr = window.devicePixelRatio || 1;
      if (last.orbit) { this.rot = (((this.rot - dx * 0.4) % 360) + 360) % 360; this.tilt = Math.max(0, Math.min(180, this.tilt + dy * 0.4)); }
      else { this.cx -= dx * dpr * this.scale; this.cy -= dy * dpr * this.scale; }
      this.request();
    });
    const end = (e: PointerEvent) => {
      if (cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId);
      if (!last || e.pointerId !== last.id) return;
      last = null; this.request();
    };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      const mx = (e.clientX - r.left) * dpr - cv.width / 2, my = (e.clientY - r.top) * dpr - cv.height / 2;
      const f = Math.exp(e.deltaY * 0.0015);
      this.cx += mx * this.scale * (1 - f); this.cy += my * this.scale * (1 - f);
      this.scale *= f; this.request();
    }, { passive: false });
    new ResizeObserver(() => this.request()).observe(cv);
  }

  /** A new data set: the field (camera px), its pixel size; points, index and view are reset (buffers are kept). */
  start(W: number, H: number, pixel_nm: number) {
    this.geo = { W, H, pixel_nm };
    this.gpu.clear(W * pixel_nm, H * pixel_nm);
    this.hist.fill(0); this.vmax = 0;
    this.tilt = 0; this.rot = 0; this.scale = 0; this.cx = 0; this.cy = 0;
    this.request();
  }

  /** Append localization rows (smlm/movie.ts layout) while they stream in. */
  append(rows: Float32Array) {
    const px = this.geo.pixel_nm, n = rows.length / NCOL, p = new Float32Array(n * 8);
    const sx = HG / (this.geo.W * px), sy = HG / (this.geo.H * px);
    for (let i = 0; i < n; i++) {
      const o = i * NCOL, x = (rows[o + 1] + 0.5) * px, y = (rows[o + 2] + 0.5) * px;
      const fin = (v: number, s: number) => (Number.isFinite(v) ? v * s : 0);   // no CRLB in the file: the min. sigma / fixed blur applies
      p.set([x, y, rows[o + 3] * 1000, fin(rows[o + 6], px), fin(rows[o + 7], px), fin(rows[o + 8], 1000), rows[o + 3], rank(rows[o + 1], rows[o + 2], rows[o])], i * 8);
      if (!(this.filter > 0 && rows[o + 6] * px > this.filter)) {
        const hx = Math.floor(x * sx), hy = Math.floor(y * sy);
        if (hx >= 0 && hy >= 0 && hx < HG && hy < HG) this.hist[hy * HG + hx]++;
      }
    }
    this.gpu.append(p, this.filter);
    this.vmax = 0;
    // when the stream pauses, fold the newest points into the spatial index
    clearTimeout(this.foldTimer);
    this.foldTimer = window.setTimeout(() => { this.gpu.fold(this.filter); this.request(); }, 600);
    this.request();
  }

  /** The CRLB filter changed: per-point enabled flags (no point upload); the display maximum follows. */
  setFilter(maxCrlbNm: number) {
    this.filter = maxCrlbNm;
    this.gpu.setFilter(maxCrlbNm);
    this.vmax = 0; this.request();
  }

  reset() { this.tilt = 0; this.rot = 0; this.scale = 0; this.cx = 0; this.cy = 0; this.request(); }
  invalidate() { this.vmax = 0; this.request(); }

  request() {
    if (this.pending) return;
    this.pending = true;
    // one frame in flight; requestAnimationFrame is paused in background tabs, so a timer backs it up
    const go = () => { if (!this.pending) return; this.pending = false; this.draw(); };
    requestAnimationFrame(go);
    setTimeout(go, 250);
  }

  private displayMax(): number {
    if (!this.vmax) {
      const nz: number[] = []; for (const v of this.hist) if (v > 0) nz.push(v);
      nz.sort((a, b) => a - b);
      const cell = (this.geo.W * this.geo.pixel_nm / HG) * (this.geo.H * this.geo.pixel_nm / HG);
      this.vmax = (nz.length ? nz[Math.floor(0.995 * (nz.length - 1))] : 1) / cell;
    }
    return this.vmax;
  }

  draw() {
    const t0 = performance.now();
    const g = this.geo, o = this.opts();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(16, Math.round(this.cv.clientWidth * dpr)), H = Math.max(16, Math.round(this.cv.clientHeight * dpr));
    if (this.cv.width !== W || this.cv.height !== H) { this.cv.width = W; this.cv.height = H; }
    if (!this.scale) this.scale = Math.max(g.W * g.pixel_nm / W, g.H * g.pixel_nm / H) * 1.02;
    const vmaxPx = this.displayMax() * this.scale * this.scale;
    this.gpu.draw({ W, H, scale: this.scale, cx: this.cx, cy: this.cy, X0: g.W * g.pixel_nm / 2, Y0: g.H * g.pixel_nm / 2, Z0: 500 * (o.zlo + o.zhi),
      tilt: this.tilt, rot: this.rot, zExag: o.zExag, mode: o.mode, blur: o.blurNm, cap: o.capNm, keep: this.maxPoints,
      vmaxPx, colorZ: o.colorZ, zlo: o.zlo, zhi: o.zhi });
    // overlay: scale bar, orientation
    const ov = this.overlay;
    ov.width = W; ov.height = H; ov.style.width = this.cv.clientWidth + 'px'; ov.style.height = this.cv.clientHeight + 'px';
    const ctx = ov.getContext('2d')!;
    ctx.clearRect(0, 0, W, H);
    const barNm = niceBar(this.scale * W / 5), barPx = barNm / this.scale;
    ctx.fillStyle = '#ddd'; ctx.fillRect(W - barPx - 14 * dpr, H - 14 * dpr, barPx, 3 * dpr);
    ctx.font = `${11 * dpr}px sans-serif`;
    ctx.fillText(barNm >= 1000 ? `${barNm / 1000} um` : `${barNm} nm`, W - barPx - 14 * dpr, H - 20 * dpr);
    if (this.tilt || this.rot) ctx.fillText(`tilt ${this.tilt.toFixed(0)}°, rotate ${this.rot.toFixed(0)}°`, 10 * dpr, 18 * dpr);
    const d = this.gpu.lastDraw;
    this.last = { n: this.gpu.n, visible: d.visible, drawn: d.drawn, nmPerPx: this.scale, capacity: this.gpu.capacity, reallocations: this.gpu.reallocations,
      rebuilds: this.gpu.rebuilds, ms: performance.now() - t0 };
    this.onChange();
  }
}

function niceBar(nm: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(nm))), m = nm / p;
  return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * p;
}
