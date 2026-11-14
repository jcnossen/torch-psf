// Messages between the UI thread and the compute worker.
import type { Optics } from '../psf/vectorial';

export interface VolumeSpec {
  optics: Optics; npupil: number; noll: number[]; zcoef: number[];
  nxy: number; dxy_um: number; zmin: number; zmax: number; nz: number; ze_um: number;
  /** SMLM geometry: z = emitter height above the coverslip (zmin .. zmax) at this fixed focal plane; without it, z is
   *  the defocus of an emitter at height ze_um (bead-stack geometry). */
  focus_um?: number;
}

export interface FitForm {
  optics: Optics;
  sign: number; dz_um: number; gain: number; offset: number; ze_um: number;
  roisize: number; npupil: number; oversample: number;
  fitDz: boolean; fitDrift: boolean; holdout: 'last-bead' | 'last-stack' | 'none';
  maxIter: number; minRel: number;
}

export interface FitSummary {
  zcoef: number[]; noll: number[]; dz_scale: number; drift: number[];
  beads: { stack: number; y0: number; x0: number; x: number; y: number; zoff: number; heldout: boolean; nz: number }[];
  stacks: { name: string; nz: number; H: number; W: number }[];
  nll_train_px: number; nll_heldout_px: number | null; iterations: number; heldout_iterations: number;
  converged: boolean; partial: boolean; partial_read: boolean; fit_seconds: number; read_seconds: number;
  frames_used: number; frames_total: number; R: number;
  data: Float32Array[]; model: Float32Array[];
  volume: Float32Array; volumeSpec: VolumeSpec;
}

export interface MovieForm {
  optics: Optics; gain: number; offset: number; readnoise_e: number;
  focus_um: number; zlo: number; zhi: number;            // focal plane during the acquisition; emitter heights (um)
  roisize: number; threshold: number; isolation: number; chunk: number; maxFrames: number;
  npupil: number; oversample: number; noll: number[]; zcoef: number[];   // the PSF used for localization
  zpad?: number;                                          // spline knot planes beyond the z range (default 2)
}

export interface InsituForm extends MovieForm { nspots: number; nheldout: number; iters: number; tile: number; }
export type MovieSourceSpec = 'demo' | File[];

export type Req =
  | { id: number; type: 'init' }
  | { id: number; type: 'volume'; spec: VolumeSpec }
  | { id: number; type: 'fit'; files: File[]; form: FitForm }
  | { id: number; type: 'movie'; source: MovieSourceSpec; form: MovieForm }
  | { id: number; type: 'estimate'; source: MovieSourceSpec; form: InsituForm; localize: boolean }
  | { id: number; type: 'demo' }
  | { id: number; type: 'movieHeader'; files: File[] }
  | { id: number; type: 'export'; kind: 'knots' | 'tif'; params: { focus_um: number; splineRoi: number; zmax: number; zmin: number } }
  | { type: 'stop'; signal: 'enough' | 'abort' };
