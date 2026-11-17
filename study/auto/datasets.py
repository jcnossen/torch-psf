"""Data sets for the auto-PSF study (spectrum NA estimate, mode classifier): paths, known optics with their sources,
the expected mode, and readers. Read-only use of the data; nothing is written next to it.

Each entry: path, kind ('tif' | 'raw' | 'strided': a large multi-page TIFF read without parsing every page),
interleave (k, of): channel k of frames alternating between `of` channels, crop (y0, x0, H, W) or None, pixel_um, NA (nominal objective NA), wavelength_um
(emission), n_med (sample medium, for the expected effective NA = min(NA, n_med) away from the coverslip),
mode (expected: smlm | sofi | zstack | static | no_signal), src (where the numbers come from), note.
"""
import os
import numpy as np
from torch_psf.movie import TiffMovie

D = '/mnt/data'
SC = f'{D}/SMLM_challenge'
FD = f'{D}/dmdsf/public_fd_deeploc/extracted'
MC = f'{D}/multicolor/june/astig/2col'
BN = f'{D}/dmdsf/bn_dnapaint'

SETS = {
    # ---------------------------------------------------------------- known optics
    'epfl_sofi_2d': dict(
        path=f'{SC}/SOFI/Tubulin-COS7-Flip565-2D/Tubulin_SOFI_2D_flip.tif', pixel_um=0.108, NA=1.27,
        wavelength_um=0.60, n_med=1.40, mode='sofi',
        src='Tubulin-COS7-Flip565-2D.json (60x/1.27 Nikon, 108 nm px, Flip565; 50 % glycerol); emission ~600 nm '
            '(Abberior FLIP 565, the json "565 nm" is the dye name / line)',
        note='dense SOFI labelling, 8000 frames 500 x 500, ORCA Flash 4.0'),
    'fd_deeploc_tetrapod': dict(
        path=f'{FD}/example_data/Li_Lab_DMO_Tetrapod6um_mito_TOM20/example_frames.tif', pixel_um=0.110, NA=1.35,
        wavelength_um=0.67, n_med=1.33, mode='smlm', crop=(400, 400, 768, 768),
        src='docs/notes/field_psf_study.md (Olympus 100XS NA 1.35 silicone, 110 nm localisation px, 670-680 nm)',
        note='tetrapod 6 um, mitochondria TOM20, 800 frames 1608 x 1608 (central 768 x 768 used)'),
    'fd_deeploc_beads20': dict(
        path=f'{FD}/bead stacks/Tetrapod_Crismo_beads_6um_100nm_1608_200mw_20/'
             'Tetrapod_Crismo_beads_6um_100nm_1608_200mw_20_MMStack_Default.ome.tif',
        pixel_um=0.108, NA=1.35, wavelength_um=0.68, n_med=1.406, mode='zstack', crop=(400, 400, 768, 768),
        src='field_psf_study.md (bead calibration: 108 nm, 680 nm, 61 planes x 100 nm; fitted with n_med = n_imm)',
        note='tetrapod bead stack'),
    'mc_beads100_left': dict(
        path=f'{MC}/beads100nm_10nm_steps/beads.tif', pixel_um=0.097, NA=1.49, wavelength_um=0.67, n_med=1.518,
        mode='zstack', crop=(0, 0, 400, 475),
        src='docs/notes/dense_beads.md (97 nm assumed from the DECODE configs of the setup, NA 1.49 nominal, '
            'fitted 1.39-1.40; 670 nm assumed)',
        note='user 2-colour astigmatic setup, left channel, 150 planes x 10 nm'),
    'mc_smlm3_roi0': dict(
        path=f'{MC}/3/data.tif', kind='strided', interleave=(0, 2), pixel_um=0.097, NA=1.49, wavelength_um=0.67,
        n_med=1.33, mode='smlm', src='as mc_beads100_left; info.yaml (two 288 x 288 ROIs, 30 ms)',
        note='user 2-colour astigmatic SMLM movie; data.tif holds the two 288 x 288 channel ROIs interleaved frame '
             'by frame (roi0 = even frames used)'),
    'dnapaint_simflux_1809': dict(
        path=f'{BN}/1809/cover1/1/simflux_5.5ms_500mW.tif', pixel_um=0.10833, NA=1.49, wavelength_um=0.70,
        n_med=1.33, mode='smlm',
        src='simcode2_out/auto/1809_2/psf/calib.json (0.10833 um, NA 1.49, 0.70 um, n_med 1.33, TIRF depth 82 nm, '
            'median emitter z 0.10 um)',
        note='DNA-PAINT with SIMFLUX patterned illumination (pattern peaks inside the band), 120 x 120'),
    'dnapaint_20nM': dict(
        path=f'{BN}/dnapaint/20nM/1/cover1_500mW_5.5ms_cf660.tif', pixel_um=0.10833, NA=1.49, wavelength_um=0.70,
        n_med=1.33, mode='smlm|sofi', src='same setup as dnapaint_simflux_1809',
        note='DNA-PAINT, 20 nM imager: dense, both routes are defensible'),
    'dnapaint_psfcalib': dict(
        path=f'{BN}/PSF_calib/raw/zstack2-1.tif', pixel_um=0.10833, NA=1.49, wavelength_um=0.70, n_med=1.33,
        mode='zstack', src='same setup; PSF_calib/text.txt: -750 .. 750 nm, 10 nm steps',
        note='bead z-stack'),
    'ao_astig_flip565': dict(
        path=f'{SC}/Tubulin-COS7-Flip565-AO-astigmatism/DATA_RAW_first1000.tif', pixel_um=0.105, NA=1.49,
        wavelength_um=0.60, n_med=1.33, mode='smlm',
        src='Tubulin-COS7-Flip565-AO-astigmatism.json (100x/1.49 Olympus, 105 nm, Flip565, degassed PBS)',
        note='medium density, AO remote focusing over 21 planes 1 um apart (up to 10 um deep)'),
    'ao_astig_psfcalib': dict(
        path=f'{SC}/Tubulin-COS7-Flip565-AO-astigmatism/PSF_calibration_20nm_step.dat', kind='raw', W=450,
        pixel_um=0.105, NA=1.49, wavelength_um=0.60, n_med=1.33, mode='zstack',
        src='same json; loaddat.py (uint16, 450 x 450 frames)', note='astigmatic bead stack, 20 nm steps'),
    # ------------------------------------------------------------- partly known optics
    'epfl_tubulin_a647_3d': dict(
        path=f'{SC}/Tubulin-A647-3D/Tubulin-A647-3D-stacks_1.tif', pixel_um=0.117, NA=None, wavelength_um=0.67,
        n_med=1.33, mode='smlm',
        src='Tubulin-A647-3D/locs/README.md: pixel not documented, ~117 x 118 nm implied by the Cspline submission; '
            'NA not documented; AF647 (~670 nm)',
        note='EPFL 2016 challenge real 3D, astigmatism, Evolve 512'),
    'epfl_tubulin_a647_beads': dict(
        path=f'{SC}/Tubulin-A647-3D/Tubulin-A647-3D-BEADS-as-stacks.tif', pixel_um=0.117, NA=None,
        wavelength_um=0.67, n_med=None, mode='zstack', crop=(0, 0, 512, 512),
        src='as above; 401 planes, 10 nm per plane assumed', note=''),
    'epfl_npc_a647_3d': dict(
        path=f'{SC}/NPC-A647-3D/NPC-crop-10000-frames.tif', pixel_um=0.138, NA=None, wavelength_um=0.67,
        n_med=1.33, mode='smlm',
        src='NPC-A647-3D/locs/README.md: pixel unknown, 135 x 142 nm implied by the Cspline submission; NA not '
            'documented; AF647', note='100 x 100 crop, 10000 frames, iXon 897'),
    'epfl_npc_beads': dict(
        path=f'{SC}/NPC-A647-3D/NPC-A647-3D-BEADS-as-stacks.tif', pixel_um=0.138, NA=None, wavelength_um=0.67,
        n_med=None, mode='zstack', src='as above; 41 planes x 50 nm (MicroManager metadata)', note=''),
    'decode_tubules': dict(
        path=f'{D}/decode-smlm/tubules/frames.tif', pixel_um=0.122, NA=None, wavelength_um=0.67, n_med=1.33,
        mode='smlm',
        src='tubules/param_run.yaml: px_size 127 x 117 nm (x, y; 122 = geometric mean); NA not in the files; '
            'U2OS alpha+beta tubulin AF647 (frames_meta.yaml), EMCCD',
        note='DECODE experimental tubulin data, 20000 x 180 x 179'),
    'decode_beads_pos0': dict(
        path=f'{D}/decode-smlm/tubules/beads/Pos0_beads_Z_1/Pos0_beads_Z_1_MMStack_Default.ome.tif',
        pixel_um=0.122, NA=None, wavelength_um=0.67, n_med=None, mode='zstack',
        src='as decode_tubules', note='bead stack, 200 planes'),
    # -------------------------------------------------------------- no signal / not fluorescence / slowly varying
    'dark_offset_frames': dict(
        path=f'{D}/dmdsf/cos7-apr11_6dmd_4exp/offset/data.tif', pixel_um=0.10833, NA=None, wavelength_um=None,
        n_med=None, mode='no_signal', src='folder name "offset": camera offset (dark) frames',
        note='600 x 190 x 190'),
    'flat_gain_frames': dict(
        path=f'{D}/SIMFLUX Z/24_12_gain_1/24_12_gain_1_MMStack_Default.ome.tif', pixel_um=None, NA=None,
        wavelength_um=None, n_med=None, mode='no_signal',
        src='folder name "gain": photon-transfer frames (uniform light)', note='2000 x 220 x 220'),
    'live_er_mEmerald': dict(
        path=f'{D}/agraas/datasets/zenodo.8325164/Set03_COS7_PrSS-mEmerald-KDEL.tif', pixel_um=None, NA=None,
        wavelength_um=0.51, n_med=1.33, mode='static',
        src='public eSRRF data (Zenodo 8325164), live COS7 ER (mEmerald); optics not in the file',
        note='live-cell ER: slowly moving structure, 5621 x 128 x 128'),
}


class RawMovie:
    """Headerless uint16 little-endian stack of W x W (or H x W) frames (memory-mapped), with the movie reader
    interface of torch_psf.movie."""
    def __init__(self, path, W, H=None):
        H = H or W
        n = os.path.getsize(path) // (2 * H * W)
        self.a = np.memmap(path, '<u2', 'r', shape=(n, H, W))
        self.n, self.shape, self.parts = n, (H, W), [path]

    def read(self, f0, f1, y0=0, x0=0, S=None):
        a = self.a[f0:f1]
        a = a[..., y0:y0 + S, x0:x0 + S] if S is not None else a[..., y0:, x0:]
        return np.asarray(a, np.float32)


class StridedTiff:
    """Fast reader for large multi-page TIFFs with one uncompressed frame per page at a constant stride (Andor SOLIS
    / MicroManager movies, movie.tif + movie_X1.tif + ...): only the first two pages of each part are parsed (parsing
    every IFD of a 4 GB file costs minutes on a busy disk); the frame count follows from the file size, and the stride
    is checked on page 2. Falls back to TiffMovie when the layout does not fit."""
    def __init__(self, path):
        import tifffile
        from torch_psf.movie import _part_paths
        self.parts, self.mms = _part_paths(path), []
        for p in self.parts:
            with tifffile.TiffFile(p) as t:
                pg = t.pages
                p0, p1, p2 = pg[0], pg[1], pg[2]
                if p0.compression != 1 or len(p0.dataoffsets) != 1:
                    raise ValueError('not a simple strided TIFF')
                dt = np.dtype(t.byteorder + p0.dtype.str[1:])
                H, W = p0.shape[-2:]
                o0, o1, o2 = p0.dataoffsets[0], p1.dataoffsets[0], p2.dataoffsets[0]
                stride = o1 - o0
                if o2 - o1 != stride or stride < H * W * dt.itemsize:
                    raise ValueError('pages not evenly spaced')
            nbytes = H * W * dt.itemsize
            n = (os.path.getsize(p) - o0 - nbytes) // stride + 1
            raw = np.memmap(p, np.uint8, 'r', offset=int(o0), shape=((n - 1) * stride + nbytes,))
            fr = np.lib.stride_tricks.as_strided(raw, shape=(n, nbytes), strides=(stride, 1), writeable=False)
            self.mms.append(fr.view(dt).reshape(n, H, W))
        self.shape = self.mms[0].shape[1:]
        self.bounds = np.cumsum([0] + [len(m) for m in self.mms])
        self.n = int(self.bounds[-1])

    def read(self, f0, f1, y0=0, x0=0, S=None):
        out = []
        for j, m in enumerate(self.mms):
            a, b = max(f0, int(self.bounds[j])), min(f1, int(self.bounds[j + 1]))
            if b > a:
                out.append(np.asarray(m[a - self.bounds[j]:b - self.bounds[j]], np.float32))
        a = np.concatenate(out)
        return a[:, y0:y0 + S, x0:x0 + S] if S is not None else a[:, y0:, x0:]


class Interleaved:
    """Channel k of a movie whose frames alternate between `of` channels (frame k, k + of, k + 2 of, ...)."""
    def __init__(self, m, k, of):
        self.m, self.k, self.of = m, k, of
        self.n = (m.n - k + of - 1) // of
        self.shape, self.parts = m.shape, m.parts

    def read(self, f0, f1, y0=0, x0=0, S=None):
        a = np.stack([self.m.read(self.k + self.of * i, self.k + self.of * i + 1)[0] for i in range(f0, f1)])
        return a[:, y0:y0 + S, x0:x0 + S] if S is not None else a[:, y0:, x0:]


class CropMovie:
    """A fixed crop (y0, x0, H, W) of another reader."""
    def __init__(self, m, crop):
        self.m, self.crop = m, crop
        y0, x0, H, W = crop
        self.n = m.n
        self.shape = (min(H, m.shape[0] - y0), min(W, m.shape[1] - x0))
        self.parts = m.parts

    def read(self, f0, f1, y0=0, x0=0, S=None):
        cy, cx, H, W = self.crop
        a = self.m.read(f0, f1)[:, cy:cy + H, cx:cx + W]
        return a[:, y0:y0 + S, x0:x0 + S] if S is not None else a[:, y0:, x0:]


def open_set(name):
    e = SETS[name]
    if e.get('kind') == 'raw':
        m = RawMovie(e['path'], e['W'])
    elif e.get('kind') == 'strided':
        m = StridedTiff(e['path'])
    else:
        m = TiffMovie(e['path'])
    if e.get('interleave'):
        m = Interleaved(m, *e['interleave'])
    if e.get('crop'):
        m = CropMovie(m, e['crop'])
    return m, e
