// The 3D PSF view (three.js): nested translucent isosurfaces at relative levels of the peak and three orthogonal
// slice planes on a log colour scale, moved through the volume. Orbit controls; optional slow auto-rotation that
// stops on the first mouse / touch interaction. Shared by the web app and the viewer page.
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

export interface ViewOptions { levels: number[]; zExag: number; decades: number; showSlices: boolean; showIso: boolean; }

const ISO_COLORS = [0xffd84d, 0xff8a3d, 0x3d8bff, 0x9b6bff];
const ISO_OPACITY = [0.9, 0.45, 0.18, 0.12];

export class View3D {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private group = new THREE.Group();
  private isoGroup = new THREE.Group();
  private slices: THREE.Mesh[] = [];
  private box: THREE.LineSegments | null = null;
  private vol: Volume | null = null;
  private vmax = 1;
  opts: ViewOptions = { levels: [0.5, 0.1, 0.01], zExag: 1, decades: 3, showSlices: true, showIso: true };
  slicePos = [0.5, 0.5, 0.5];   // fractions along x, y, z
  autoRotate = true;
  private raf = 0;
  private lastExtent = 0;

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

  setVolume(v: Volume) {
    this.vol = v;
    let m = 0; for (const x of v.data) m = Math.max(m, x);
    this.vmax = m || 1;
    this.rebuild();
  }

  /** world coordinates (um, z exaggerated) of grid point (i, j, k), centred on the volume */
  private scale(): THREE.Vector3 {
    const v = this.vol!;
    return new THREE.Vector3(v.dx_um, v.dx_um, v.dz_um * this.opts.zExag);
  }

  rebuild() {
    const v = this.vol;
    if (!v) return;
    for (const c of [...this.isoGroup.children]) { this.isoGroup.remove(c); (c as THREE.Mesh).geometry.dispose(); }
    const s = this.scale();
    const cx = (v.nx - 1) / 2, cy = (v.ny - 1) / 2, cz = (v.nz - 1) / 2;
    this.isoGroup.position.set(-cx * s.x, -cy * s.y, -cz * s.z);
    this.isoGroup.scale.copy(s);
    const norm = new Float32Array(v.data.length);
    for (let i = 0; i < norm.length; i++) norm[i] = v.data[i] / this.vmax;
    if (this.opts.showIso) {
      // innermost (highest level) first; the outer, more transparent shells are drawn after it
      const levels = [...this.opts.levels].sort((a, b) => b - a);
      levels.forEach((lv, i) => {
        const m = isosurface(norm, v.nx, v.ny, v.nz, lv);
        if (!m.positions.length) return;
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
        // normals are in grid units; with an anisotropic scale three.js uses the normal matrix, which is correct
        g.setAttribute('normal', new THREE.BufferAttribute(m.normals, 3));
        const mat = new THREE.MeshPhongMaterial({
          color: ISO_COLORS[i % ISO_COLORS.length], transparent: true, opacity: ISO_OPACITY[i % ISO_OPACITY.length],
          side: THREE.DoubleSide, depthWrite: i === 0, shininess: 40,
        });
        const mesh = new THREE.Mesh(g, mat);
        mesh.renderOrder = i;
        this.isoGroup.add(mesh);
      });
    }
    // bounding box
    if (this.box) { this.group.remove(this.box); this.box.geometry.dispose(); }
    const bx = new THREE.BoxGeometry((v.nx - 1) * s.x, (v.ny - 1) * s.y, (v.nz - 1) * s.z);
    this.box = new THREE.LineSegments(new THREE.EdgesGeometry(bx), new THREE.LineBasicMaterial({ color: 0x556070 }));
    bx.dispose();
    this.group.add(this.box);
    this.buildSlices();
    const r = Math.max((v.nx - 1) * s.x, (v.ny - 1) * s.y, (v.nz - 1) * s.z);
    this.controls.target.set(0, 0, 0);
    if (Math.abs(r - this.lastExtent) > 0.05 * r) {   // a new kind of volume: frame it
      this.camera.position.copy(this.camera.position.clone().normalize().multiplyScalar(1.7 * r));
      this.lastExtent = r;
    }
  }

  private buildSlices() {
    for (const m of this.slices) { this.group.remove(m); m.geometry.dispose(); ((m.material as THREE.MeshBasicMaterial).map)?.dispose(); }
    this.slices = [];
    const v = this.vol;
    if (!v || !this.opts.showSlices) return;
    const s = this.scale();
    const W = (v.nx - 1) * s.x, Hh = (v.ny - 1) * s.y, Z = (v.nz - 1) * s.z;
    const mk = (w: number, h: number) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.85, depthWrite: false }));
      m.renderOrder = 10;
      this.group.add(m);
      this.slices.push(m);
      return m;
    };
    mk(W, Hh);                                   // xy (normal z)
    const xz = mk(W, Z); xz.rotation.x = Math.PI / 2;
    const yz = mk(Hh, Z); yz.rotation.set(Math.PI / 2, Math.PI / 2, 0);
    this.updateSlices();
  }

  /** The three section images (log scale) at the current slice positions: xy [ny, nx], xz [nz, nx], yz [nz, ny]. */
  sections(): { xy: Float32Array; xz: Float32Array; yz: Float32Array; ix: number; iy: number; iz: number } | null {
    const v = this.vol;
    if (!v) return null;
    const ix = Math.round(this.slicePos[0] * (v.nx - 1)), iy = Math.round(this.slicePos[1] * (v.ny - 1)), iz = Math.round(this.slicePos[2] * (v.nz - 1));
    const xy = v.data.slice(iz * v.nx * v.ny, (iz + 1) * v.nx * v.ny);
    const xz = new Float32Array(v.nz * v.nx), yz = new Float32Array(v.nz * v.ny);
    for (let k = 0; k < v.nz; k++) {
      for (let i = 0; i < v.nx; i++) xz[k * v.nx + i] = v.data[(k * v.ny + iy) * v.nx + i];
      for (let j = 0; j < v.ny; j++) yz[k * v.ny + j] = v.data[(k * v.ny + j) * v.nx + ix];
    }
    return { xy, xz, yz, ix, iy, iz };
  }

  updateSlices() {
    const v = this.vol, sec = this.sections();
    if (!v || !sec || this.slices.length < 3) return;
    const s = this.scale();
    const tex = (a: Float32Array, w: number, h: number) => {
      const t = new THREE.DataTexture(toRgba(a, w, h, { vmax: this.vmax, decades: this.opts.decades, alpha: true }), w, h, THREE.RGBAFormat);
      t.magFilter = THREE.NearestFilter; t.needsUpdate = true; t.colorSpace = THREE.SRGBColorSpace;
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
    xy.position.set(0, 0, (sec.iz - (v.nz - 1) / 2) * s.z);
    xz.position.set(0, (sec.iy - (v.ny - 1) / 2) * s.y, 0);
    yz.position.set((sec.ix - (v.nx - 1) / 2) * s.x, 0, 0);
  }

  setSlice(axis: 0 | 1 | 2, frac: number) { this.slicePos[axis] = frac; this.updateSlices(); }

  dispose() { cancelAnimationFrame(this.raf); this.renderer.dispose(); }
}
