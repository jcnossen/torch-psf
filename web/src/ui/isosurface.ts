// Isosurface extraction by marching tetrahedra (6 tetrahedra per cube; no lookup tables), with vertex normals from
// the interpolated volume gradient. vol is [nz, ny, nx]; positions come out in grid units (x, y, z) = (i, j, k).

const CUBE: [number, number, number][] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const TETS = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];

export interface Mesh { positions: Float32Array; normals: Float32Array; }

export function isosurface(vol: Float32Array, nx: number, ny: number, nz: number, level: number): Mesh {
  const pos: number[] = [], nrm: number[] = [];
  const at = (x: number, y: number, z: number) => vol[(z * ny + y) * nx + x];
  const grad = (x: number, y: number, z: number): [number, number, number] => [
    at(Math.min(nx - 1, x + 1), y, z) - at(Math.max(0, x - 1), y, z),
    at(x, Math.min(ny - 1, y + 1), z) - at(x, Math.max(0, y - 1), z),
    at(x, y, Math.min(nz - 1, z + 1)) - at(x, y, Math.max(0, z - 1)),
  ];
  const cv = new Float64Array(8);
  const cp: [number, number, number][] = new Array(8);
  const edge = (a: number, b: number) => {
    const t = (level - cv[a]) / (cv[b] - cv[a]);
    const pa = cp[a], pb = cp[b];
    pos.push(pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2]));
    const ga = grad(pa[0], pa[1], pa[2]), gb = grad(pb[0], pb[1], pb[2]);
    let g0 = ga[0] + t * (gb[0] - ga[0]), g1 = ga[1] + t * (gb[1] - ga[1]), g2 = ga[2] + t * (gb[2] - ga[2]);
    const l = Math.hypot(g0, g1, g2) || 1;
    nrm.push(-g0 / l, -g1 / l, -g2 / l);
  };
  for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
    let above = 0;
    for (let c = 0; c < 8; c++) {
      const [dx, dy, dz] = CUBE[c];
      cv[c] = at(x + dx, y + dy, z + dz);
      if (cv[c] > level) above++;
      cp[c] = [x + dx, y + dy, z + dz];
    }
    if (above === 0 || above === 8) continue;
    for (const t of TETS) {
      const ins = t.filter((v) => cv[v] > level), out = t.filter((v) => cv[v] <= level);
      if (ins.length === 0 || ins.length === 4) continue;
      if (ins.length === 1 || ins.length === 3) {
        const [s, o] = ins.length === 1 ? [ins[0], out] : [out[0], ins];
        edge(s, o[0]); edge(s, o[1]); edge(s, o[2]);
      } else {
        const [a, b] = ins, [c, d] = out;
        edge(a, c); edge(a, d); edge(b, d);
        edge(a, c); edge(b, d); edge(b, c);
      }
    }
  }
  return { positions: new Float32Array(pos), normals: new Float32Array(nrm) };
}
