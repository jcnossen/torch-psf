"""Step 3: the orientation MLE (torch_psf.orientation.fit_orientation) on simulated SIMCODE spots vs the CRLB.

PSF: the 1809 in-situ PSF (focus 308 nm) and the 2.5 nM in-situ PSF (focus 101 nm, blur); emitters at z = 50 nm,
bg 10 photons / px, Poisson noise; x, y uniform in +-0.5 px. Cases: single spots at 1000 and 5000 photons, and
6-frame tracks (3 frames per pattern angle, 1000 photons / frame) with the in-situ excitation tensors K.
Errors of the axis are measured in the tangent coordinates of the true axis (a: polar, b: in-plane, deg), after
choosing the sign of the estimate closest to the truth; 'outliers' = axis error > 30 deg (mostly the mirror solution
theta -> 180 - theta, i.e. the opposite tilt sign), excluded from bias / sd.

usage: CUDA_VISIBLE_DEVICES=2 python study/orientation/fit_validation.py   (writes results/fit_validation.json)
"""
import json
import math
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, file_model, ensemble_to_K, PIX                  # noqa: E402
from torch_psf.orientation import (from_angles, crlb_orientation, model_and_jacobian, fit_orientation, tangent_basis,
                                   canonical)                             # noqa: E402

OUT = os.path.join(HERE, 'results')
ORIENT = [(60.0, 30.0, 1.0), (80.0, 120.0, 0.7), (40.0, 200.0, 1.0), (20.0, 45.0, 1.0), (60.0, 30.0, 0.4)]


def run_case(psf, t, p, g, phot, n, T=1, K=None, z=0.05, bg=10.0):
    dev = psf.dev
    mu = from_angles(torch.full((n,), t), torch.full((n,), p)).to(dev)
    th = torch.zeros(n, 9, device=dev)
    th[:, 0] = torch.rand(n, device=dev) - 0.5; th[:, 1] = torch.rand(n, device=dev) - 0.5
    th[:, 2] = z; th[:, 3] = phot; th[:, 4] = bg; th[:, 5:8] = mu; th[:, 8] = g
    with torch.no_grad():
        m, _ = model_and_jacobian(psf, th, T=T, K=K)
    d = torch.poisson(m)
    d = d[:, 0] if T == 1 else d
    t0 = time.time()
    r = fit_orientation(psf, d, K=K, iters=50, z_limits=(0.0, 0.4))
    el = time.time() - t0
    est = r['theta']
    mt = canonical(mu)
    me = est[:, 5:8] * torch.sign((est[:, 5:8] * mt).sum(1, keepdim=True))
    e_th, e_ph = tangent_basis(mt)
    a = torch.rad2deg((me * e_th).sum(1)); b = torch.rad2deg((me * e_ph).sum(1))
    cr = crlb_orientation(psf, th[:1], T=T, K=K)
    ok = (me * mt).sum(1) > math.cos(math.radians(30))
    f = lambda v: float(v)
    return dict(theta=t, phi=p, g=g, photons=phot, T=T, K=K is not None, n=n, fit_s=el,
                outliers=f(1 - ok.float().mean()),
                bias_a=f(a[ok].mean()), sd_a=f(a[ok].std()), crlb_a=f(cr['theta'][0]),
                bias_b=f(b[ok].mean()), sd_b=f(b[ok].std()), crlb_b=f(cr['inplane'][0]),
                bias_g=f((est[ok, 8] - g).mean()), sd_g=f(est[ok, 8].std()), crlb_g=f(cr['g'][0]),
                bias_x_nm=f((est[ok, 0] - th[ok, 0]).mean() * PIX * 1e3), sd_x_nm=f((est[ok, 0] - th[ok, 0]).std() * PIX * 1e3),
                crlb_x_nm=f(cr['x'][0] * PIX * 1e3),
                bias_z_nm=f((est[ok, 2] - z).mean() * 1e3), sd_z_nm=f(est[ok, 2].std() * 1e3), crlb_z_nm=f(cr['z'][0] * 1e3))


def main():
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = int(os.environ.get('NFIT', 400))
    rows = []
    for mname in ('m1809', 'm25'):
        psf = file_model(mname, dev)
        Kt = ensemble_to_K('m1809')['K'].to(dev)
        Kseq = torch.stack([Kt[0]] * 3 + [Kt[1]] * 3)
        for (t, p, g) in ORIENT:
            for phot in (1000.0, 5000.0):
                r = run_case(psf, t, p, g, phot, n)
                r['model'] = mname; rows.append(r); print(r, flush=True)
            r = run_case(psf, t, p, g, 1000.0, n, T=6, K=Kseq)
            r['model'] = mname; rows.append(r); print(r, flush=True)
    json.dump(rows, open(os.path.join(OUT, 'fit_validation.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
