// Point-cloud view of the localizations (Reconstruction panel), on the principles of webSMLM's point-cloud renderer
// and its tilt / rotate view:
//  - rendered at screen resolution for the current view (pan / zoom), each localization splatted as a Gaussian whose
//    size is in world units (nm): its own CRLB ("precision" mode, capped) or one fixed blur, integrated over the
//    screen pixel (erf differences); additive, i.e. a density image; no point sprites, no size attenuation;
//  - turntable 3D: rotate about the z axis, then tilt about the horizontal screen axis (0 = top view, 90 = side view);
//    the projected sigma follows the rotation (sx'^2 = sx^2 c^2 + sy^2 s^2, sy'^2 = (sx^2 s^2 + sy^2 c^2) ct^2
//    + sz^2 st^2); colour stays the world z;
//  - display maximum global: a percentile of the density (per nm^2) of the whole field in the top view, kept while
//    panning / zooming / orbiting, so the brightness does not jump;
//  - above a point cap (while dragging) a fixed subset is drawn: a hash rank per localization (no flicker while
//    moving), weighted by shown / drawn so the brightness is preserved; the full set is drawn on release.
import { NCOL } from '../smlm/movie';

export interface CloudOptions {
  mode: 'precision' | 'fixed';
  blurNm: number;          // fixed mode: sigma (nm); precision mode: minimum sigma
  capNm: number;           // precision mode: maximum sigma (nm)
  maxCrlbNm: number;       // drop localizations with crlb_x > this (0 = keep all)
  colorZ: boolean;
  zlo: number; zhi: number; // colour range (um)
  zExag: number;           // z exaggeration in tilted views
}

// erf (Abramowitz and Stegun 7.1.26, |error| < 1.5e-7)
function erf(x: number): number {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
// fixed rank in [0, 1) from x, y, frame (the same localization always gets the same rank)
function rank(x: number, y: number, f: number): number {
  let h = Math.imul(Math.round(x * 1024) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(Math.round(y * 1024) + 0x632be5ab, 0xc2b2ae35) ^ Math.imul(f | 0, 0x27d4eb2f);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12; h = Math.imul(h, 0x297a2d39); h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

export class PointCloudView {
  tilt = 0; rot = 0;               // degrees
  scale = 0;                       // nm per screen pixel (0 = fit on the next draw)
  cx = 0; cy = 0;                  // view centre in the rotated, tilted plane (nm, relative to the field centre)
  maxDrawn = 300000;               // point cap while dragging
  private vmax = 0;                // global display maximum (density per nm^2)
  private dragging = false;
  private pending = false;
  onChange: () => void = () => {};

  constructor(private cv: HTMLCanvasElement, private getRows: () => Float32Array[], private geom: () => { W: number; H: number; pixel_nm: number; n: number },
    private opts: () => CloudOptions) {
    let last: { x: number; y: number; orbit: boolean } | null = null;
    cv.addEventListener('pointerdown', (e) => { last = { x: e.clientX, y: e.clientY, orbit: e.shiftKey || e.button === 2 }; this.dragging = true; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', (e) => {
      if (!last) return;
      const dx = e.clientX - last.x, dy = e.clientY - last.y; last.x = e.clientX; last.y = e.clientY;
      const dpr = window.devicePixelRatio || 1;
      if (last.orbit) { this.rot = (((this.rot - dx * 0.4) % 360) + 360) % 360; this.tilt = Math.max(0, Math.min(180, this.tilt + dy * 0.4)); }
      else { this.cx -= dx * dpr * this.scale; this.cy -= dy * dpr * this.scale; }
      this.request();
    });
    const end = (e: PointerEvent) => { if (!last) return; last = null; this.dragging = false; if (cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId); this.request(); };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      const mx = (e.clientX - r.left) * dpr - cv.width / 2, my = (e.clientY - r.top) * dpr - cv.height / 2;
      const f = Math.exp(e.deltaY * 0.0015);
      // keep the point under the mouse fixed
      this.cx += mx * this.scale * (1 - f); this.cy += my * this.scale * (1 - f);
      this.scale *= f; this.request();
    }, { passive: false });
    new ResizeObserver(() => this.request()).observe(cv);
  }

  /** Back to the top view of the whole field. */
  reset() { this.tilt = 0; this.rot = 0; this.scale = 0; this.cx = 0; this.cy = 0; this.vmax = 0; this.request(); }
  /** New data or new render settings: the global display maximum is recomputed. */
  invalidate() { this.vmax = 0; this.request(); }

  request() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  private splat(W: number, H: number, scale: number, cx: number, cy: number, tilt: number, rot: number, cap: number) {
    const g = this.geom(), o = this.opts(), px = g.pixel_nm;
    const X0 = g.W * px / 2, Y0 = g.H * px / 2, Z0 = 500 * (o.zlo + o.zhi);
    const ct = Math.cos(tilt * Math.PI / 180), st = Math.sin(tilt * Math.PI / 180), cr = Math.cos(rot * Math.PI / 180), sr = Math.sin(rot * Math.PI / 180);
    const ze = o.zExag;
    const maxC = o.maxCrlbNm > 0 ? o.maxCrlbNm : Infinity;
    const D = new Float32Array(W * H), Z = new Float32Array(W * H);
    // how many are in the field (for the cap)
    let inView = 0;
    const chunks = this.getRows();
    for (const c of chunks) for (let k = 0; k < c.length; k += NCOL) if (c[k + 6] * px <= maxC) inView++;
    const keep = cap > 0 && inView > cap ? cap / inView : 1, wgt = 1 / keep;
    let drawn = 0;
    const s2 = Math.SQRT2;
    for (const c of chunks) for (let k = 0; k < c.length; k += NCOL) {
      const sxn = c[k + 6] * px;
      if (!(sxn <= maxC)) continue;
      if (keep < 1 && rank(c[k + 1], c[k + 2], c[k]) >= keep) continue;
      const Xp = (c[k + 1] + 0.5) * px - X0, Yp = (c[k + 2] + 0.5) * px - Y0, Zp = (c[k + 3] * 1000 - Z0) * ze;
      const u = Xp * cr + Yp * sr, v = -Xp * sr + Yp * cr;
      const sxp = (u - cx) / scale + W / 2, syp = (v * ct - Zp * st - cy) / scale + H / 2;
      let sgx: number, sgy: number;
      if (o.mode === 'precision') {
        const sx2 = sxn * sxn, syn = c[k + 7] * px, sy2 = syn * syn, szn = (c[k + 8] * 1000) * ze;
        sgx = Math.sqrt(sx2 * cr * cr + sy2 * sr * sr);
        sgy = Math.sqrt((sx2 * sr * sr + sy2 * cr * cr) * ct * ct + szn * szn * st * st);
        sgx = Math.min(o.capNm, Math.max(o.blurNm, sgx)); sgy = Math.min(o.capNm, Math.max(o.blurNm, sgy));
      } else { sgx = sgy = o.blurNm; }
      sgx /= scale; sgy /= scale;                               // screen px
      const zc = c[k + 3];
      drawn++;
      if (sgx < 0.3 && sgy < 0.3) {                             // smaller than a pixel: point sample
        const ix = Math.floor(sxp), iy = Math.floor(syp);
        if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
        D[iy * W + ix] += wgt; Z[iy * W + ix] += wgt * zc; continue;
      }
      const rx = Math.ceil(3 * sgx), ry = Math.ceil(3 * sgy);
      const ix0 = Math.max(0, Math.floor(sxp) - rx), ix1 = Math.min(W - 1, Math.floor(sxp) + rx);
      const iy0 = Math.max(0, Math.floor(syp) - ry), iy1 = Math.min(H - 1, Math.floor(syp) + ry);
      if (ix0 > ix1 || iy0 > iy1) continue;
      const wx: number[] = [];
      for (let ix = ix0; ix <= ix1; ix++) wx.push(0.5 * (erf((ix + 1 - sxp) / (s2 * sgx)) - erf((ix - sxp) / (s2 * sgx))));
      for (let iy = iy0; iy <= iy1; iy++) {
        const wy = 0.5 * (erf((iy + 1 - syp) / (s2 * sgy)) - erf((iy - syp) / (s2 * sgy))) * wgt;
        if (wy < 1e-7) continue;
        const row = iy * W;
        for (let ix = ix0; ix <= ix1; ix++) { const w = wy * wx[ix - ix0]; D[row + ix] += w; Z[row + ix] += w * zc; }
      }
    }
    return { D, Z, inView, drawn };
  }

  draw() {
    const g = this.geom(), o = this.opts();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(16, Math.round(this.cv.clientWidth * dpr)), H = Math.max(16, Math.round(this.cv.clientHeight * dpr));
    if (this.cv.width !== W || this.cv.height !== H) { this.cv.width = W; this.cv.height = H; }
    if (!this.scale) this.scale = Math.max(g.W * g.pixel_nm / W, g.H * g.pixel_nm / H) * 1.02;
    // global display maximum: 99.5 % of the non-empty density (per nm^2) of the whole field, top view
    if (!this.vmax && g.n > 0) {
      const sc = Math.max(g.W * g.pixel_nm / 256, g.H * g.pixel_nm / 256);
      const r = this.splat(256, 256, sc, 0, 0, 0, 0, 0);
      const nz: number[] = []; for (const v of r.D) if (v > 0) nz.push(v);
      nz.sort((a, b) => a - b);
      this.vmax = (nz.length ? nz[Math.floor(0.995 * (nz.length - 1))] : 1) / (sc * sc);
    }
    const r = this.splat(W, H, this.scale, this.cx, this.cy, this.tilt, this.rot, this.dragging ? this.maxDrawn : 0);
    const ctx = this.cv.getContext('2d')!;
    const img = ctx.createImageData(W, H);
    const vmax = this.vmax * this.scale * this.scale || 1;
    for (let i = 0; i < W * H; i++) {
      const d = r.D[i];
      if (d <= 0) { img.data[4 * i + 3] = 255; continue; }
      const v = Math.min(1, Math.pow(d / vmax, 0.6));
      let cr: number, cg: number, cb: number;
      if (o.colorZ) {
        const t = Math.max(0, Math.min(1, (r.Z[i] / d - o.zlo) / Math.max(1e-9, o.zhi - o.zlo)));
        [cr, cg, cb] = hue(t); cr *= v; cg *= v; cb *= v;
      } else [cr, cg, cb] = hot(v);
      img.data[4 * i] = 255 * cr; img.data[4 * i + 1] = 255 * cg; img.data[4 * i + 2] = 255 * cb; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    // scale bar and orientation
    const barNm = niceBar(this.scale * W / 5), barPx = barNm / this.scale;
    ctx.fillStyle = '#ddd'; ctx.fillRect(W - barPx - 14 * dpr, H - 14 * dpr, barPx, 3 * dpr);
    ctx.font = `${11 * dpr}px sans-serif`; ctx.fillText(barNm >= 1000 ? `${barNm / 1000} um` : `${barNm} nm`, W - barPx - 14 * dpr, H - 20 * dpr);
    if (this.tilt || this.rot) ctx.fillText(`tilt ${this.tilt.toFixed(0)}°, rotate ${this.rot.toFixed(0)}°`, 10 * dpr, 18 * dpr);
    this.last = { inView: r.inView, drawn: r.drawn, nmPerPx: this.scale };
    this.onChange();
  }
  last = { inView: 0, drawn: 0, nmPerPx: 0 };
}

function niceBar(nm: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(nm))), m = nm / p;
  return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * p;
}
/** blue (0) -> cyan -> green -> yellow -> red (1) */
function hue(t: number): [number, number, number] {
  const hh = (1 - t) * 240 / 60, x = 1 - Math.abs((hh % 2) - 1);
  if (hh < 1) return [1, x, 0]; if (hh < 2) return [x, 1, 0]; if (hh < 3) return [0, 1, x]; return [0, x, 1];
}
function hot(v: number): [number, number, number] {
  return [Math.min(1, 3 * v), Math.max(0, Math.min(1, 3 * v - 1)), Math.max(0, Math.min(1, 3 * v - 2))];
}
