"""Synthetic checks of the auto analysis (band limit and mode) with known optics.

A. Band limit with realistic PSFs (VectorialPSF via torch_psf.simulate.simulate_movie): in-focus 2D, astigmatic 3D and
   tetrapod-like 3D PSFs; index-matched sample (n_med = n_imm) and a water sample (n_med 1.33) with an NA 1.49
   objective; NA 1.0 .. 1.49; photons per emitter. Reports f_c(fit) / f_c(true) and the status.
B. Undersampling and low signal (scalar synth movies).
C. Mode classifier on synthetic movies of every class (scalar synth and vectorial SMLM).
D. Astigmatism in the time-averaged spectrum of SMLM movies: contrast_m2 with and without astigmatism.
Output: ~/autopsf_out/synth/*.json, synth_table.md, figs/synth_*.png.

python -I study/auto/run_synth.py [--device cuda:N] [A B C D]
"""
import os
import sys
import json
import time
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plots import plot_result                                               # noqa: E402
from torch_psf.vectorial import VectorialPSF                               # noqa: E402
from torch_psf.simulate import simulate_movie                              # noqa: E402
from torch_psf.auto import synthetic_movie, analyze_movie, band_limit      # noqa: E402
from torch_psf.auto.analyze import analyze_frames                          # noqa: E402

OUT = os.path.expanduser('~/autopsf_out')
PX, LAM = 0.1, 0.6

PSFS = dict(inf2d=dict(zernike={}, focus=0.0, z=(0.0, 0.0)),
            astig=dict(zernike={6: 0.10}, focus=0.3, z=(0.0, 0.6)),
            tetrapod=dict(zernike={6: 0.20, 12: -0.20}, focus=1.0, z=(0.0, 2.0)))


def vec_movie(NA, n_med, psf, photons, device, T=400, H=128, W=128, nsites=600, seed=0, z_offset=0.0):
    pr = PSFS[psf]
    R = 31 if psf == 'tetrapod' else 21
    vec = VectorialPSF(R, pixel_um=PX, NA=NA, n_imm=1.518, n_med=n_med, wavelength_um=LAM,
                       zernike_indices=tuple(range(5, 23)), device=device)
    if pr['zernike']:
        vec.set_zernike(pr['zernike'])
    fz = torch.full((1,), pr['focus'] + z_offset, device=device)
    z = (pr['z'][0] + z_offset, pr['z'][1] + z_offset)
    mov, _ = simulate_movie(lambda p: vec.forward_batched(p, fz.expand(len(p))), R, T=T, H=H, W=W, nsites=nsites,
                            p_on=0.004, mean_on=3, bg=20, photons=photons, z_range=z, seed=seed, device=device)
    return mov.numpy().astype(np.float32)


def part_a(device):
    rows = []
    for psf in ('inf2d', 'astig', 'tetrapod'):
        for NA, n_med, zoff in ((1.0, 1.518, 0), (1.2, 1.518, 0), (1.4, 1.518, 0), (1.49, 1.33, 0.0),
                                (1.49, 1.33, 1.0)):
            for photons in (1000, 5000):
                t = time.time()
                mov = vec_movie(NA, n_med, psf, photons, device, z_offset=zoff)
                bl, spec = band_limit(mov, pixel_um=PX, wavelength_um=LAM, spectrum_kw=dict(device=device), n_boot=50)
                fc_true = 2 * NA / LAM * PX
                fc_med = 2 * min(NA, n_med) / LAM * PX
                rows.append(dict(psf=psf, NA=NA, n_med=n_med, z_offset_um=zoff, photons=photons, fc_true=fc_true,
                                 fc_nmed=fc_med, fc_fit=bl.fc_px, fc_err=bl.fc_px_err, f_detect=bl.f_detect_px,
                                 status=bl.status, hf_contrast=bl.hf_contrast, NA_est=bl.NA, NA_err=bl.NA_err, band_snr=bl.band_snr,
                                 ratio=bl.fc_px / fc_med))
                print(f"A {psf:8s} NA {NA} n_med {n_med} zoff {zoff} ph {photons}: fc {bl.fc_px:.3f}+-{bl.fc_px_err:.3f}"
                      f" (true {fc_true:.3f}, n_med-limited {fc_med:.3f}, ratio {bl.fc_px / fc_med:.3f}) "
                      f"fdet {bl.f_detect_px:.3f} hf {bl.hf_contrast:.3g} {bl.status} {time.time() - t:.1f}s", flush=True)
    return rows


def part_b():
    rows = []
    for name, kw in (('px 0.10 um, NA 1.2 (fc 0.40)', dict(pixel_um=0.10, NA=1.2)),
                     ('px 0.12 um, NA 1.2 (fc 0.48)', dict(pixel_um=0.12, NA=1.2)),
                     ('px 0.14 um, NA 1.2 (fc 0.56)', dict(pixel_um=0.14, NA=1.2)),
                     ('px 0.16 um, NA 1.4 (fc 0.75)', dict(pixel_um=0.16, NA=1.4)),
                     ('low signal: 150 photons, T 50', dict(pixel_um=0.10, NA=1.2, photons=150, T=50)),
                     ('very low: 50 photons, T 20', dict(pixel_um=0.10, NA=1.2, photons=50, T=20)),
                     ('noise only', dict(mode='noise'))):
        kw = dict(kw)
        mode = kw.pop('mode', 'smlm')
        kw.setdefault('T', 200)
        mov, info = synthetic_movie(mode, N=128, density=0.02, duty=0.02, seed=4, **kw)
        bl, _ = band_limit(mov, pixel_um=info['pixel_um'], wavelength_um=0.6, n_boot=50, spectrum_kw=dict(device='cpu'))
        rows.append(dict(case=name, fc_true=info['fc_px'], fc_fit=bl.fc_px, fc_err=bl.fc_px_err,
                         f_detect=bl.f_detect_px, status=bl.status, message=bl.message))
        print(f'B {name}: true {info["fc_px"]:.3f} fit {bl.fc_px:.3f}+-{bl.fc_px_err:.3f} fdet {bl.f_detect_px:.3f} '
              f'{bl.status}: {bl.message}', flush=True)
    return rows


def part_c(device):
    rows = []
    cases = [('smlm', 'scalar smlm', dict(mode='smlm', density=0.02, duty=0.01, T=400)),
             ('smlm', 'scalar smlm, dense 0.1 / PSF', dict(mode='smlm', density=0.05, duty=0.05, T=400)),
             ('sofi', 'scalar sofi', dict(mode='sofi', density=0.3, duty=0.3, T=400)),
             ('sofi', 'scalar sofi, low duty', dict(mode='sofi', density=0.5, duty=0.1, mean_on=5, T=400)),
             ('zstack', 'scalar beads through focus', dict(mode='zstack', density=0.002, photons=20000, T=101)),
             ('zstack', 'astigmatic beads through focus', dict(mode='zstack', density=0.002, photons=20000, T=101,
                                                              astig_waves=0.15)),
             ('static', 'static filaments', dict(mode='static', T=200)),
             ('static', 'static filaments, dim', dict(mode='static', T=200, photons=300)),
             ('no_signal', 'noise only', dict(mode='noise', T=200)),
             ('smlm', 'vectorial astig smlm', 'vec_astig'),
             ('smlm', 'vectorial tetrapod smlm', 'vec_tetra')]
    for truth, name, kw in cases:
        if kw == 'vec_astig':
            mov = vec_movie(1.49, 1.33, 'astig', 3000, device, T=600)
        elif kw == 'vec_tetra':
            mov = vec_movie(1.35, 1.33, 'tetrapod', 5000, device, T=600)
        else:
            kw = dict(kw); mode = kw.pop('mode')
            mov, _ = synthetic_movie(mode, N=128, seed=5, **kw)
        r = analyze_movie(mov, pixel_um=PX, wavelength_um=LAM, n_boot=30, device=device)
        rows.append(dict(case=name, truth=truth, mode=r.mode.mode, confidence=r.mode.confidence, probs=r.mode.probs,
                         reason=r.mode.reason, key=r.mode.key, fc_fit=r.band.fc_px, status=r.band.status))
        plot_result(r, f'{OUT}/figs/synth_{name.replace(" ", "_").replace(",", "").replace("/", "")}.png',
                    title=f'synthetic: {name} (truth {truth})')
        print(f'C {name}: truth {truth} -> {r.mode.mode} ({r.mode.confidence:.2f}) {r.mode.reason}', flush=True)
    return rows


def part_d(device):
    rows = []
    for psf in ('inf2d', 'astig'):
        for zr in ('symmetric', 'below focus only'):
            pr = PSFS[psf]
            vec = VectorialPSF(21, pixel_um=PX, NA=1.3, n_imm=1.518, n_med=1.518, wavelength_um=LAM,
                               zernike_indices=tuple(range(5, 23)), device=device)
            if pr['zernike']:
                vec.set_zernike(pr['zernike'])
            fz = torch.full((1,), 0.3, device=device)
            z = (0.0, 0.6) if zr == 'symmetric' else (0.0, 0.25)
            mov, _ = simulate_movie(lambda p: vec.forward_batched(p, fz.expand(len(p))), 21, T=400, H=128, W=128,
                                    nsites=600, p_on=0.004, mean_on=3, bg=20, photons=3000, z_range=z, device=device)
            r = analyze_movie(mov.numpy().astype(np.float32), pixel_um=PX, wavelength_um=LAM, n_boot=30,
                              device=device)
            an = r.band.anisotropy or {}
            rows.append(dict(psf=psf, z=zr, contrast_m2=an.get('contrast_m2'), contrast_m2_se=an.get('contrast_m2_se'),
                             contrast_angle=an.get('contrast_angle_deg'), ellipse_ratio=an.get('ellipse_ratio'),
                             m2_block_sd=float(np.std(an.get('contrast_m2_blocks', [np.nan]))),
                             aniso_frame_rms=r.mode.stats.get('aniso_frame_rms')))
            print(f"D {psf} z {zr}: m2 {an.get('contrast_m2'):.3f} +- {an.get('contrast_m2_se'):.3f} at "
                  f"{an.get('contrast_angle_deg'):.0f} deg, support ellipse {an.get('ellipse_ratio')}, block sd "
                  f"{rows[-1]['m2_block_sd']:.3f}, frame rms {rows[-1]['aniso_frame_rms']:.3f}", flush=True)
    return rows


def main():
    device = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'cuda'
    parts = [a for a in sys.argv[1:] if a in 'ABCD' and len(a) == 1] or list('ABCD')
    os.makedirs(f'{OUT}/synth', exist_ok=True); os.makedirs(f'{OUT}/figs', exist_ok=True)
    fn = dict(A=lambda: part_a(device), B=part_b, C=lambda: part_c(device), D=lambda: part_d(device))
    for p in parts:
        rows = fn[p]()
        with open(f'{OUT}/synth/part_{p}.json', 'w') as f:
            json.dump(rows, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


if __name__ == '__main__':
    main()
