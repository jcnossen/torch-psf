// Messages between the UI thread and the compute worker.
import type { Optics } from '../psf/vectorial';

export interface VolumeSpec {
  optics: Optics; npupil: number; noll: number[]; zcoef: number[];
  nxy: number; dxy_um: number; zmin: number; zmax: number; nz: number; ze_um: number;
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

export type Req =
  | { id: number; type: 'init' }
  | { id: number; type: 'volume'; spec: VolumeSpec }
  | { id: number; type: 'fit'; files: File[]; form: FitForm }
  | { id: number; type: 'export'; kind: 'knots' | 'tif'; params: { focus_um: number; splineRoi: number; zmax: number; zmin: number } }
  | { type: 'stop'; signal: 'enough' | 'abort' };
