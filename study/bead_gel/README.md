# Through-focus bead gel study

A Python/CUDA feasibility test of a gel scan with overlapping beads, a calibrated vectorial PSF and unknown slowly varying XYZ drift. This study uses the fixed acquisition below.

## Acquisition

160 nonblinking, 100 nm fluorescent spheres are uniformly distributed throughout a 64 x 64 pixel field (108.33 nm pixels) and 0.05–9.95 µm in depth. A known water-immersion PSF uses NA 1.2, index 1.33 in both medium and immersion, 700 nm emission and 0.075 waves RMS of Noll 6. Index matching makes the shape a function of relative defocus. Real gels with index mismatch or unknown pupils require additional validation.

The stage scans 0–10 µm in 100 nm steps, upward, downward and upward again. Each stop records six Poisson exposures. Each bead has its own phase and brightness. Repeated six-state intensities average to the bead's stable mean. The reference variant adds three settled exposure groups, at nominal 4.7, 5.0 and 5.3 µm, every 12 scan planes. There are 381 bursts or 2,286 exposures with references. The simulation omits stage settling, within-burst motion and bleaching. Drift varies across bursts, with its origin fixed at the first burst.

## Run

From the TorchPSF checkout with NumPy, SciPy, PyTorch and tifffile installed:

    python study/bead_gel/scan.py --out gel/ordinary --passes 1 3
    python study/bead_gel/scan.py --references --out gel/reference --passes 1 3
    python study/bead_gel/scan.py --references --seed 42 --out gel/replicate --passes 3
    python study/bead_gel/overview.py gel/reference

The overview requires Matplotlib, Pillow and ffmpeg with H.264 support. Set FFMPEG to an alternative executable if necessary. The preview plays one native exposure per ordinary scan plane. The movie TIFF contains all six exposures per burst, including references.

## Fit and outputs

The simulator projects every bead onto every detector pixel using a finite-sphere, pixel-integrated TorchPSF vectorial response sampled every 50 nm axially. The fit uses a separate 100 nm axial grid. Three-dimensional detection on the first ordinary pass initializes a joint pixel fit of positions, mean photons, planar background and a piecewise-linear drift trajectory. Aligned residual volumes supply additional candidates. Per-exposure weighted, regularized joint photometry includes out-of-focus and overlapping neighbours. The nominal scan, images and optical assumptions enter the fit. Simulation truth is read only for evaluation and plots.

movie.tif is a TYX stack. scan.csv contains zero-based frame, nominal stage focus (µm), condition and ordinary scan index (−1 for reference). With --reuse-movie, recovery reads these files instead of simulating. fit_3pass.npz (or fit_1pass.npz) contains:

- xyz[bead]: reference camera x and y in pixels, and bead depth in µm.
- flux[frame, bead]: estimated photons in original TIFF frame order. The column is the persistent bead ID.
- usable[frame, bead]: geometric near-focus mask, within 0.55 µm of fitted focus. Farther frames are often poorly constrained and regularized toward a bead's mean brightness.
- drift_um[burst, xyz]: lateral sample translation and axial focus offset. nominal_focus_um + drift_um[:,2] is the actual fitted focus. Each row covers six exposures.
- background_plane: intercept and two slopes across normalized field coordinates.

For later pattern estimation, use flux with usable, condition IDs, and the bead's drift-corrected camera position. XY position in a burst equals xyz[:,:2] + drift_um[burst,:2]/0.10833.

## Synthetic validation

Positions are matched one-to-one within 250 nm in 3D. Position RMSE is over matched beads. Drift RMSE uses the fixed first-burst origin. Near-focus photon RMSE is normalized by each bead's true mean intensity and pooled across matched beads' usable exposures.

| Seed and acquisition | Matched / true | Fitted candidates | Position RMSE x/y/z (nm) | Drift RMSE x/y/z (nm) | Near-focus photon RMSE |
| --- | ---: | ---: | --- | --- | ---: |
| 19, one up pass | 98 / 160 | 170 | 85 / 62 / 99 | 122 / 90 / 142 | 19.6% |
| 19, up/down/up | 160 / 160 | 168 | 110 / 90 / 38 | 112 / 88 / 24 | 14.0% |
| 19, one up pass with references | 160 / 160 | 167 | 16 / 19 / 34 | 5 / 5 / 22 | 15.4% |
| 19, up/down/up with references | 160 / 160 | 167 | 16 / 21 / 32 | 7 / 11 / 11 | 14.4% |
| 42, up/down/up with references | 159 / 160 | 164 | 14 / 30 / 25 | 10 / 23 / 9 | 16.1% |

The single-pass rows use the first pass of their corresponding three-pass movie, so they cover a shorter time interval. The added reference bursts are extra acquisition time. About 11% of frame/bead pairs were within the usable band. In the two three-pass reference scans, 65/65 and 79/80 beads with another bead within 250 nm in XY projection were matched.
