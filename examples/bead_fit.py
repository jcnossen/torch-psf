"""Bead z-stacks -> vectorial PSF (torch_psf.estimate_psf_beads), on simulated stacks with a known aberration set.

    python examples/bead_fit.py [--out bead_psf.json]
    python examples/bead_fit.py --epfl       # the one-bead crop of the public EPFL SMLM challenge 2016 bead stack

The same as the command line:  python -m torch_psf beads stack.tif --dz-um 0.02 --pixel-um 0.108 --na 1.49 -o psf.json
"""
import argparse, json, os, tempfile
import numpy as np, tifffile, torch
import torch_psf as tp

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='bead_psf.json')
ap.add_argument('--epfl', action='store_true')
a = ap.parse_args()

if a.epfl:   # real data: one bead, astigmatic (cylindrical lens); acquisition settings in the json next to the crop
    crop = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'web', 'epfl_npc_bead.tif')
    meta = json.load(open(crop.replace('.tif', '.json')))
    optics = tp.Optics(pixel_um=meta['pixel_um'], NA=meta['NA'], wavelength_um=meta['wavelength_um'])
    camera = tp.Camera(gain_e_per_adu=meta['gain'], offset_adu=meta['offset'])
    est = tp.estimate_psf_beads([crop], dz_um=meta['dz_um'], optics=optics, camera=camera, roisize=25, nbeads=1,
                                drift=False)
else:        # simulated: 3 stacks with 4 beads each, astigmatism + coma + spherical, Poisson noise
    truth = {5: 0.02, 6: 0.10, 7: -0.03, 8: 0.015, 11: 0.04}
    vec = tp.VectorialPSF(21, pixel_um=0.108, NA=1.49, wavelength_um=0.68).set_zernike(truth)
    nk, H, W, dz = 41, 96, 96, 0.025
    focus = dz * (torch.arange(nk, device=vec.dev) - (nk - 1) / 2)
    g = torch.Generator(device=vec.dev).manual_seed(0)
    tmp = tempfile.mkdtemp()
    files = []
    for s in range(3):
        mov = torch.full((nk, H, W), 30.0, device=vec.dev)
        for cx, cy in ((20, 20), (70, 24), (25, 72), (72, 70)):
            off = torch.rand(2, generator=g, device=vec.dev) - 0.5
            img = vec.forward_batched(torch.cat([off, torch.zeros(1, device=vec.dev)])[None].expand(nk, -1), focus)
            mov[:, cy - 10:cy + 11, cx - 10:cx + 11] += 20000 * img
        adu = torch.poisson(mov, generator=g) / 0.5 + 100          # camera: 0.5 e-/ADU, offset 100 ADU
        files.append(os.path.join(tmp, f'stack{s}.tif'))
        tifffile.imwrite(files[-1], adu.cpu().numpy().astype(np.uint16))
    est = tp.estimate_psf_beads(files, dz_um=dz, optics=tp.Optics(pixel_um=0.108, NA=1.49, wavelength_um=0.68),
                                camera=tp.Camera(gain_e_per_adu=0.5, offset_adu=100.0), roisize=21, holdout=2,
                                log=lambda it, nll, sec, zc: it % 50 == 0 and print(f'  it {it}  nll {nll:.1f}'))
    err = (est.psf.zcoef.detach() - vec.zcoef.detach()).pow(2).mean().sqrt()
    print(f'Zernike error (rms over Noll 5-21): {float(err):.4f} waves')

print(est)
print('fitted Zernikes (waves):', {j: round(float(c), 3) for j, c in zip(est.psf.zidx, est.psf.zcoef.detach()) if abs(c) > 0.005})
est.save(a.out)
print('saved', a.out, '(view it with: python -m torch_psf view', a.out + ')')
