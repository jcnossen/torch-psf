// torch-psf PSF file (JSON, same schema as torch_psf.io.save_psf; the PSF keys as in SIMCODE's calib json).
import { Optics } from '../psf/vectorial';

export interface PsfJson {
  format: 'torch-psf';
  version: string;
  type: 'vectorial';
  optics: Optics;
  zernike_noll: number[];
  zernike_waves: number[];
  apod: number[];
  saf: number[];
  focus_um: number;
  psf_blur_px2: null;
  psf_dipole_M: null;
  diattenuation: null;
  npupil: number;
  oversample: number;
  arrays?: string;
  provenance: Record<string, unknown>;
  // SIMCODE calib-json names (so SIMCODE's loader reads the file as is)
  pixelsize_um: number; wavelength_um: number; NA: number; n_imm: number; n_med: number;
}

export function makePsfJson(o: { optics: Optics; noll: number[]; zcoef: ArrayLike<number>; focus_um: number; npupil: number; oversample: number; provenance: Record<string, unknown>; arrays?: string }): PsfJson {
  return {
    format: 'torch-psf', version: '0.1.0', type: 'vectorial', optics: { ...o.optics },
    zernike_noll: [...o.noll], zernike_waves: Array.from(o.zcoef, (v) => +v.toFixed(7)),
    apod: [0, 0, 0], saf: [0, 0, 0, 0], focus_um: o.focus_um, psf_blur_px2: null, psf_dipole_M: null, diattenuation: null,
    npupil: o.npupil, oversample: o.oversample, ...(o.arrays ? { arrays: o.arrays } : {}), provenance: o.provenance,
    pixelsize_um: o.optics.pixel_um, wavelength_um: o.optics.wavelength_um, NA: o.optics.NA, n_imm: o.optics.n_imm, n_med: o.optics.n_med,
  };
}

export function readPsfJson(text: string): { optics: Optics; noll: number[]; zcoef: number[]; focus_um: number } {
  const j = JSON.parse(text);
  const optics: Optics = j.optics ?? { pixel_um: j.pixelsize_um, NA: j.NA, wavelength_um: j.wavelength_um, n_imm: j.n_imm, n_med: j.n_med };
  return { optics, noll: j.zernike_noll, zcoef: j.zernike_waves, focus_um: j.focus_um ?? 0 };
}
