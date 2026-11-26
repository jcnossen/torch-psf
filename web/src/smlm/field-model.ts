import type { MovieField } from '../dense/movie-field';

export const NAT_MODES = [4, 5, 6, 7, 8, 11];
export const CONSTANTS = [0, 5, 4, 10, 9, 12];
export type FieldMethod = "nat" | "independent";
export interface PupilField { method: FieldMethod; parameters: number[]; norm: MovieField['feats_norm']; tile: number; }

/** torch_psf.estimation.nat_polynomial: RMS waves, columns 1,v,v²,u,uv,u². */
export function natPolynomial(g: number[]): number[][] {
  const [d, dx, dy, k, a, b, ax, ay, q, cx, cy, r, sph] = g;
  const s = Math.sqrt(3), t = Math.sqrt(5);
  return [[d - t * k, s * dy, 1.5 * t * k, s * dx, 0, 1.5 * t * k], [b, s * ax, 0, s * ay, 3 * q, 0],
  [a, -s * ay, -1.5 * q, s * ax, 0, 1.5 * q], [cy, s * r, 0, 0, 0, 0], [cx, 0, 0, s * r, 0, 0], [sph, 0, 0, 0, 0, 0]];
}

/** Read the explicit polynomial, including PyTorch files without browser provenance. */
export function readPupilField(field: MovieField | undefined): PupilField | undefined {
  if (!field || field.type !== 'zernike_poly' || field.order !== 2 ||
    !Array.isArray(field.zernike_noll) || field.zernike_noll.length !== 6 ||
    !Array.isArray(field.coef)) return;
  const norm = field.feats_norm;
  if (!norm || ![norm.cx, norm.cy, norm.sx, norm.sy].every(Number.isFinite) || Math.min(norm.sx, norm.sy) <= 0) return;
  const parameters: number[] = [];
  for (const [i, mode] of NAT_MODES.entries()) {
    const row = field.coef[field.zernike_noll.indexOf(mode)], slots = FIELD_TERMS[i];
    if (!Array.isArray(row) || row.length !== 6 || !row.every(Number.isFinite) ||
      row.some((v, j) => v !== 0 && !slots.includes(j))) return;
    parameters.push(...slots.map(j => row[j]));
  }
  return { method: 'independent', parameters, norm, tile: Math.min(64, 2 * Math.min(norm.sx, norm.sy) / 3) };
}

export const FIELD_TERMS = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 3], [0, 1, 3], [0]];
function polynomial(model: PupilField) {
  if (model.method === 'nat') return natPolynomial(model.parameters);
  let k = 0;
  return FIELD_TERMS.map(slots => {
    const row = [0, 0, 0, 0, 0, 0];
    for (const j of slots) row[j] = model.parameters[k++];
    return row;
  });
}
export function fieldBlock(model: PupilField): MovieField {
  return { type: 'zernike_poly', order: 2, zernike_noll: NAT_MODES, coef: polynomial(model), feats_norm: model.norm };
}

export function fieldPupil(model: PupilField, noll: number[], base: number[], x: number, y: number) {
  const { cx, cy, sx, sy } = model.norm, clip = (z: number) => Math.max(-1, Math.min(1, z));
  const u = clip((x - cx) / sx), v = clip((y - cy) / sy), f = [1, v, v * v, u, u * v, u * u];
  const rows = polynomial(model);
  return noll.map((j, i) => NAT_MODES.includes(j) ? rows[NAT_MODES.indexOf(j)].reduce((s, c, k) => s + c * f[k], 0) : base[i]);
}
