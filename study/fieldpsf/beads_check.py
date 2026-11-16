"""For selected beads: our NLL at our fitted Zernikes vs at theirs (converted, held fixed; only position / photons /
background / drift / height refitted), and a refit started from theirs. Tells a local minimum of our fit from a real
disagreement.

    python -I beads_check.py STACK_TIF THEIR_PREFIX BEADS_JSON OUT_JSON --beads 0 3 4
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I: the script folder is not on sys.path
import numpy as np, tifffile, torch
from common import OPTICS, THEIR_NOLL, theirs_to_ours, setup_path
setup_path()
from torch_psf.vectorial import VectorialPSF            # noqa: E402
from torch_psf.estimation import fit_beads, stage_focus  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('stack'); ap.add_argument('their_prefix'); ap.add_argument('beads_json'); ap.add_argument('out')
ap.add_argument('--beads', type=int, nargs='+', required=True)
ap.add_argument('--iters', type=int, default=400)
a = ap.parse_args()
dev = torch.device('cuda')
B = json.load(open(a.beads_json))
P, h = B['roisize'], B['roisize'] // 2
st = (tifffile.imread(a.stack).astype(np.float32) - B['offset']) * B['gain']
focus = stage_focus(st.shape[0], 0.1, B['sign']).to(dev)
o = OPTICS.__class__.from_dict(B['optics'])
out = []
for b in B['beads']:
    if b['index'] not in a.beads:
        continue
    x0, y0 = int(round(b['x'])) - h, int(round(b['y'])) - h
    d = torch.as_tensor(st[None, :, y0:y0 + P, x0:x0 + P].copy(), device=dev)
    rec = dict(index=b['index'], ours_nll_per_px=b['nll_per_px'])
    for name, zc, lr in (('theirs_fixed', theirs_to_ours(b['their_coef']), 0.0),
                         ('from_theirs', theirs_to_ours(b['their_coef']), 3e-3),
                         ('ours_fixed', b['zcoef'], 0.0)):
        v = VectorialPSF.from_optics(o, P, zernike_indices=tuple(THEIR_NOLL), device=dev).set_zernike(list(map(float, zc)))
        r = fit_beads(d, focus, v, iters=a.iters, lr=lr, drift=True, fit_height=True, emitter_z_um=b['height_um'])
        rec[name] = dict(nll_per_px=r['report']['nll_per_px'], chi2=r['report']['chi2_median'], height_um=r['height_um'],
                         zcoef=r['psf'].zcoef.detach().cpu().tolist())
    rec['rms_from_theirs_vs_ours'] = float(np.sqrt(np.mean((np.array(rec['from_theirs']['zcoef']) - np.array(b['zcoef'])) ** 2)))
    print(b['index'], 'nll/px ours %.4f | ours fixed %.4f | theirs fixed %.4f (chi2 %.2f) | refit from theirs %.4f (rms to ours %.3f)' % (
        b['nll_per_px'], rec['ours_fixed']['nll_per_px'], rec['theirs_fixed']['nll_per_px'], rec['theirs_fixed']['chi2'],
        rec['from_theirs']['nll_per_px'], rec['rms_from_theirs_vs_ours']), flush=True)
    out.append(rec)
json.dump(out, open(a.out, 'w'), indent=1)
