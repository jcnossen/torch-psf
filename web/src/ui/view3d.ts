// The 3D PSF view (three.js): nested translucent isosurfaces at relative levels of the peak and three orthogonal
// slice planes on a log colour scale (bilinear, interpolated between planes), moved through the volume with sliders or
// by dragging a plane along its normal (hover highlights it). Orbit controls; optional slow auto-rotation that stops on
// the first mouse / touch interaction. Shared by the web app and the viewer page.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { isosurface } from './isosurface';
import { toRgba } from './colormap';

export interface Volume {
  data: Float32Array;   // [nz, ny, nx]
  nx: number; ny: number; nz: number;
  dx_um: number;        // xy voxel size
  dz_um: number;        // z voxel size
  z0_um: number;        // z of plane 0
}

/** look 'slices': slice planes only (default); 'volume': faint slice planes plus many nested white isosurfaces. */
export type ViewLook = 'slices' | 'volume';
export interface ViewOptions { levels: number[]; zExag: number; decades: number; showSlices: boolean; showIso: boolean; look: ViewLook; }

// volume look: 10 log-spaced levels from 60 % to 2 % of the peak, white and faint (additive: the inner shells add up
// to a glow); the outer shells are fainter than the inner ones
const VOLUME_LEVELS = Array.from({ length: 10 }, (_, i) => 0.6 * Math.pow(0.02 / 0.6, i / 9));
const SLICE_OPACITY: Record<ViewLook, number> = { slices: 0.85, volume: 0.2 };

const ISO_COLORS = [0xffd84d, 0xff8a3d, 0x3d8bff, 0x9b6bff];
const ISO_OPACITY = [0.9, 0.45, 0.18, 0.12];
const SLICE_AXIS: (0 | 1 | 2)[] = [2, 1, 0];   // normal axis of the slice meshes [xy, xz, yz]

export class View3D {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private group = new THREE.Group();
  private isoGroup = new THREE.Group();
  private slices: THREE.Mesh[] = [];
  private outlines: THREE.LineSegments[] = [];
  private box: THREE.LineSegments | null = null;
  private vol: Volume | null = null;
  private vmax = 1;
  private isoKey = '';
  opts: ViewOptions = { levels: [0.5, 0.1, 0.01], zExag: 1, decades: 3, showSlices: true, showIso: true, look: 'slices' };
  slicePos = [0.5, 0.5, 0.5];   // fractions along x, y, z
  autoRotate = true;
  /** called when a slice plane is moved by dragging (to keep sliders in sync) */
  onSliceChange: ((axis: 0 | 1 | 2, frac: number) => void) | null = null;
  private raf = 0;
  private lastExtent = 0;
  private hovered = -1;
  private drag: { axis: 0 | 1 | 2; p: THREE.Vector3; n: THREE.Vector3; c0: number } | null = null;
  private ray = new THREE.Raycaster();

  constructor(private el: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x14161c);
    el.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.camera.position.set(3, -4, 2.6);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.8;
    const stop = () => { this.autoRotate = false; this.controls.autoRotate = false; };
    for (const ev of ['pointerdown', 'wheel', 'touchstart']) this.renderer.domElement.addEventListener(ev, stop, { passive: true });
    this.initDrag();
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dl = new THREE.DirectionalLight(0xffffff, 1.1); dl.position.set(2, -3, 4); this.scene.add(dl);
    const dl2 = new THREE.DirectionalLight(0xffffff, 0.4); dl2.position.set(-3, 2, -2); this.scene.add(dl2);
    this.group.add(this.isoGroup);
    this.scene.add(this.group);
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => { this.raf = requestAnimationFrame(loop); this.controls.update(); this.renderer.render(this.scene, this.camera); };
    loop();
  }

  resize() {
    const w = this.el.clientWidth || 600, h = this.el.clientHeight || 500;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + 'px';
    this.renderer.domElement.style.height = h + 'px';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ---------------------------------------------------------------- dragging the slice planes
  private pick(ev: PointerEvent): THREE.Intersection | null {
    if (!this.slices.length) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    const m = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(m, this.camera);
    return this.ray.intersectObjects(this.slices, false)[0] ?? null;
  }

  private setHover(i: number) {
    if (i === this.hovered) return;
    this.hovered = i;
    this.outlines.forEach((o, k) => { o.visible = k === i; });
    this.renderer.domElement.style.cursor = i >= 0 ? 'grab' : '';
  }

  /** world coordinate along `axis` of the slice at fraction f, and back */
  private axisWorld(axis: number, f: number) {
    const v = this.vol!, s = this.scale(), n = [v.nx, v.ny, v.nz][axis], sc = [s.x, s.y, s.z][axis];
    return (f * (n - 1) - (n - 1) / 2) * sc;
  }

  private axisFrac(axis: number, w: number) {
    const v = this.vol!, s = this.scale(), n = [v.nx, v.ny, v.nz][axis], sc = [s.x, s.y, s.z][axis];
    return Math.max(0, Math.min(1, (w / sc + (n - 1) / 2) / Math.max(1, n - 1)));
  }

  private initDrag() {
    const cv = this.renderer.domElement;
    // capture phase on the container: runs before OrbitControls' own pointerdown handler on the canvas
    this.el.addEventListener('pointerdown', (ev) => {
      const hit = this.pick(ev);
      if (!hit) return;
      const axis = SLICE_AXIS[this.slices.indexOf(hit.object as THREE.Mesh)];   // slices are [xy, xz, yz]
      const n = new THREE.Vector3(axis === 0 ? 1 : 0, axis === 1 ? 1 : 0, axis === 2 ? 1 : 0);
      this.drag = { axis, p: hit.point.clone(), n, c0: this.axisWorld(axis, this.slicePos[axis]) };
      this.controls.enabled = false;
      cv.setPointerCapture(ev.pointerId);
      cv.style.cursor = 'grabbing';
    }, { capture: true });
    cv.addEventListener('pointermove', (ev) => {
      if (!this.drag) { this.setHover(this.slices.indexOf((this.pick(ev)?.object ?? null) as THREE.Mesh)); return; }
      const r = cv.getBoundingClientRect();
      this.ray.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), this.camera);
      // closest point of the mouse ray to the line through the grab point along the plane normal
      const d = this.ray.ray.direction, o = this.ray.ray.origin, { p, n } = this.drag;
      const w = o.clone().sub(p);
      const b = d.dot(n), den = 1 - b * b;   // |d| = |n| = 1
      if (den < 1e-4) return;                // looking along the normal
      const t = (n.dot(w) - b * d.dot(w)) / den;
      const f = this.axisFrac(this.drag.axis, this.drag.c0 + t);
      this.setSlice(this.drag.axis, f);
      this.onSliceChange?.(this.drag.axis, f);
    });
    const end = (ev: PointerEvent) => {
      if (!this.drag) return;
      this.drag = null;
      this.controls.enabled = true;
      if (cv.hasPointerCapture(ev.pointerId)) cv.releasePointerCapture(ev.pointerId);
      cv.style.cursor = this.hovered >= 0 ? 'grab' : '';
    };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', () => { if (!this.drag) this.setHover(-1); });
  }

  // ---------------------------------------------------------------- volume, isosurfaces, transforms
  setVolume(v: Volume) {
    this.vol = v;
    let m = 0; for (const x of v.data) m = Math.max(m, x);
    this.vmax = m || 1;
    this.isoKey = '';
    this.rebuild();
  }

  /** world coordinates (um, z exaggerated) of grid point (i, j, k), centred on the volume */
  private scale(): THREE.Vector3 {
    const v = this.vol!;
    return new THREE.Vector3(v.dx_um, v.dx_um, v.dz_um * this.opts.zExag);
  }

  /** Change the z exaggeration: only transforms are updated (no isosurface extraction). */
  setZExag(z: number) { this.opts.zExag = z; this.applyTransform(); }

  /** Switch between the slice look and the volume look (isosurfaces are rebuilt, slice opacity changes). */
  setLook(look: ViewLook) {
    this.opts.look = look;
    for (const m of this.slices) (m.material as THREE.MeshBasicMaterial).opacity = SLICE_OPACITY[look];
    this.rebuild();
  }

  /** Full update after a change of options. Isosurfaces are recomputed only when the volume, the levels or the
   *  surface toggle changed; a z-exaggeration change only updates the transforms. */
  rebuild() {
    const v = this.vol;
    if (!v) return;
    const key = JSON.stringify([this.opts.levels, this.opts.showIso, this.opts.look]);
    if (key !== this.isoKey) { this.buildIso(); this.isoKey = key; }
    if (!this.box) {
      const bx = new THREE.BoxGeometry(1, 1, 1);
      this.box = new THREE.LineSegments(new THREE.EdgesGeometry(bx), new THREE.LineBasicMaterial({ color: 0x556070 }));
      bx.dispose();
      this.group.add(this.box);
    }
    if ((this.slices.length > 0) !== this.opts.showSlices) this.buildSlices();
    this.applyTransform();
    this.updateSlices();
    const s = this.scale();
    const r = Math.max((v.nx - 1) * s.x, (v.ny - 1) * s.y, (v.nz - 1) * s.z);
    this.controls.target.set(0, 0, 0);
    if (Math.abs(r - this.lastExtent) > 0.05 * r) {   // a new kind of volume: frame it
      this.camera.position.copy(this.camera.position.clone().normalize().multiplyScalar(1.7 * r));
      this.lastExtent = r;
    }
  }

  private buildIso() {
    const v = this.vol!;
    for (const c of [...this.isoGroup.children]) { this.isoGroup.remove(c); (c as THREE.Mesh).geometry.dispose(); }
    const vol = this.opts.look === 'volume';
    if (!vol && (this.opts.look === 'slices' || !this.opts.showIso)) return;
    const norm = new Float32Array(v.data.length);
    for (let i = 0; i < norm.length; i++) norm[i] = v.data[i] / this.vmax;
    // innermost (highest level) first; the outer, more transparent shells are drawn after it
    const levels = [...(vol ? VOLUME_LEVELS : this.opts.levels)].sort((a, b) => b - a);
    levels.forEach((lv, i) => {
      const m = isosurface(norm, v.nx, v.ny, v.nz, lv);
      if (!m.positions.length) return;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
      // normals are in grid units; with an anisotropic scale three.js uses the normal matrix, which is correct
      g.setAttribute('normal', new THREE.BufferAttribute(m.normals, 3));
      const mat = vol
        ? new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.09 - 0.06 * i / (levels.length - 1),
          side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending })
        : new THREE.MeshPhongMaterial({
          color: ISO_COLORS[i % ISO_COLORS.length], transparent: true, opacity: ISO_OPACITY[i % ISO_OPACITY.length],
          side: THREE.DoubleSide, depthWrite: i === 0, shininess: 40,
        });
      const mesh = new THREE.Mesh(g, mat);
      mesh.renderOrder = i;
      this.isoGroup.add(mesh);
    });
  }

  /** Scale-dependent transforms: isosurface group, box, slice-plane sizes and positions. */
  private applyTransform() {
    const v = this.vol;
    if (!v) return;
    const s = this.scale();
    const W = (v.nx - 1) * s.x, H = (v.ny - 1) * s.y, Z = (v.nz - 1) * s.z;
    this.isoGroup.position.set(-(v.nx - 1) / 2 * s.x, -(v.ny - 1) / 2 * s.y, -(v.nz - 1) / 2 * s.z);
    this.isoGroup.scale.copy(s);
    this.box?.scale.set(W, H, Z);
    if (this.slices.length === 3) {
      const [xy, xz, yz] = this.slices;
      xy.scale.set(W, H, 1); xz.scale.set(W, Z, 1); yz.scale.set(H, Z, 1);
      this.positionSlices();
    }
  }

  private positionSlices() {
    if (this.slices.length < 3 || !this.vol) return;
    const [xy, xz, yz] = this.slices;
    xy.position.set(0, 0, this.axisWorld(2, this.slicePos[2]));
    xz.position.set(0, this.axisWorld(1, this.slicePos[1]), 0);
    yz.position.set(this.axisWorld(0, this.slicePos[0]), 0, 0);
  }

  private buildSlices() {
    for (const m of this.slices) { this.group.remove(m); m.geometry.dispose(); ((m.material as THREE.MeshBasicMaterial).map)?.dispose(); }
    for (const o of this.outlines) o.geometry.dispose();
    this.slices = []; this.outlines = []; this.hovered = -1;
    if (!this.vol || !this.opts.showSlices) return;
    const pg = new THREE.PlaneGeometry(1, 1);
    const eg = new THREE.EdgesGeometry(pg);
    const mk = () => {
      const m = new THREE.Mesh(pg.clone(), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: SLICE_OPACITY[this.opts.look], depthWrite: false }));
      m.renderOrder = 10;
      const o = new THREE.LineSegments(eg.clone(), new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
      o.renderOrder = 11;
      o.visible = false;
      m.add(o);
      this.group.add(m);
      this.slices.push(m);
      this.outlines.push(o);
      return m;
    };
    mk();                                        // xy (normal z)
    mk().rotation.x = Math.PI / 2;               // xz (normal y)
    mk().rotation.set(Math.PI / 2, Math.PI / 2, 0);   // yz (normal x)
    pg.dispose(); eg.dispose();
  }

  /** The three section images at the current slice positions, linearly interpolated between the two neighbouring
   *  planes: xy [ny, nx], xz [nz, nx], yz [nz, ny]; ix, iy, iz are the (fractional) plane indices. */
  sections(): { xy: Float32Array; xz: Float32Array; yz: Float32Array; ix: number; iy: number; iz: number } | null {
    const v = this.vol;
    if (!v) return null;
    const { nx, ny, nz, data } = v;
    const ix = this.slicePos[0] * (nx - 1), iy = this.slicePos[1] * (ny - 1), iz = this.slicePos[2] * (nz - 1);
    const split = (t: number, n: number): [number, number, number] => { const a = Math.min(Math.floor(t), Math.max(0, n - 2)); return [a, Math.min(a + 1, n - 1), t - a]; };
    const [z0, z1, fz] = split(iz, nz), [y0, y1, fy] = split(iy, ny), [x0, x1, fx] = split(ix, nx);
    const xy = new Float32Array(nx * ny);
    for (let i = 0; i < nx * ny; i++) xy[i] = (1 - fz) * data[z0 * nx * ny + i] + fz * data[z1 * nx * ny + i];
    const xz = new Float32Array(nz * nx), yz = new Float32Array(nz * ny);
    for (let k = 0; k < nz; k++) {
      for (let i = 0; i < nx; i++) xz[k * nx + i] = (1 - fy) * data[(k * ny + y0) * nx + i] + fy * data[(k * ny + y1) * nx + i];
      for (let j = 0; j < ny; j++) yz[k * ny + j] = (1 - fx) * data[(k * ny + j) * nx + x0] + fx * data[(k * ny + j) * nx + x1];
    }
    return { xy, xz, yz, ix, iy, iz };
  }

  updateSlices() {
    const v = this.vol, sec = this.sections();
    if (!v || !sec || this.slices.length < 3) return;
    const tex = (a: Float32Array, w: number, h: number) => {
      const t = new THREE.DataTexture(toRgba(a, w, h, { vmax: this.vmax, decades: this.opts.decades, alpha: true }), w, h, THREE.RGBAFormat);
      t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearFilter; t.generateMipmaps = false;
      t.needsUpdate = true; t.colorSpace = THREE.SRGBColorSpace;
      return t;
    };
    const set = (m: THREE.Mesh, t: THREE.DataTexture) => {
      const mat = m.material as THREE.MeshBasicMaterial;
      mat.map?.dispose(); mat.map = t; mat.needsUpdate = true;
    };
    const [xy, xz, yz] = this.slices;
    set(xy, tex(sec.xy, v.nx, v.ny));
    set(xz, tex(sec.xz, v.nx, v.nz));
    set(yz, tex(sec.yz, v.ny, v.nz));
    this.positionSlices();
  }

  setSlice(axis: 0 | 1 | 2, frac: number) { this.slicePos[axis] = frac; this.updateSlices(); }

  dispose() { cancelAnimationFrame(this.raf); this.renderer.dispose(); }
}
