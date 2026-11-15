"""Survey bead z-stacks on the data drive for density / overlap: per TIFF the stack shape, the number of beads in the
max projection (local maxima of the 2-px-blurred projection above median + 8 MAD) and nearest-neighbour distances.
python survey_beads.py list.txt  (one path per line) -> results/bead_survey.json"""
import sys, json, os
import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

rows = []
for p in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    try:
        with tifffile.TiffFile(p) as t:
            n = len(t.pages)
            shp = t.pages[0].shape
            if len(shp) != 2:
                rows.append(dict(path=p, pages=n, shape=list(shp), note='not 2D pages')); continue
            idx = np.linspace(0, n - 1, min(n, 200)).astype(int)
            a = np.stack([t.pages[int(i)].asarray() for i in idx]).astype(np.float32)
        mp = ndimage.gaussian_filter(a.max(0), 2.0)
        med = np.median(mp); mad = 1.4826 * np.median(np.abs(mp - med)) + 1e-6
        mx = (mp == ndimage.maximum_filter(mp, 7)) & (mp > med + 8 * mad)
        yx = np.argwhere(mx)
        nn = cKDTree(yx).query(yx, 2)[0][:, 1] if len(yx) > 1 else np.array([np.inf])
        rows.append(dict(path=p, pages=n, shape=list(shp), nbeads=int(len(yx)),
                         beads_per_100px2=float(len(yx) / (shp[0] * shp[1]) * 1e4),
                         nn_median_px=float(np.median(nn)), frac_nn_below_10px=float((nn < 10).mean()),
                         frac_nn_below_15px=float((nn < 15).mean())))
    except Exception as e:
        rows.append(dict(path=p, error=f'{type(e).__name__}: {e}'))
    print(json.dumps(rows[-1]), flush=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'bead_survey.json'), 'w'), indent=1)
