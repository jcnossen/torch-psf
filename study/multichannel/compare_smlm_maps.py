"""Compare the bead mappings (reference = left half -> right half, crop px) with the affine maps of the SMLM analyses
of the same setup (3col/*/affine_mapping.txt: a 3 x 3 matrix between the 200 px ROIs of info.yaml, roi0 = right half,
roi1 = left half; plain text / yaml, read as data). Our map is expressed in ROI coordinates
(u0 = A u1 + t, u = camera px - ROI origin) and compared with their matrix as given and transposed.
  python -I compare_smlm_maps.py [report.json of a 3-colour run] [report.json of a 2-colour run]"""
import os, sys, json, glob
import numpy as np
import yaml

def ours(rep):
    d = json.load(open(rep))['registration'][1]['describe']
    return np.array(d['A']), np.array(d['b'])

def in_roi(A, b, roi_r, roi_l, x_split=475):
    """u0 (right ROI) = A u1 (left ROI) + t"""
    o1, o0 = np.array(roi_l[:2], float), np.array(roi_r[:2], float)
    t = A @ o1 + b + np.array([x_split, 0.0]) - o0
    return A, t

out = {}
for rep in sys.argv[1:]:
    A, b = ours(rep)
    key = os.path.basename(os.path.dirname(os.path.dirname(rep)))
    col = '3col' if '3col' in rep else '2col'
    for f in sorted(glob.glob(f'/mnt/data/multicolor/june/astig/{col}/*/*/affine_mapping.txt')):
        info = yaml.safe_load(open(os.path.join(os.path.dirname(f), 'info.yaml')))
        roi_r, roi_l = info['rois'][0], info['rois'][1]
        M = np.loadtxt(f)
        Ar, t = in_roi(A, b, roi_r, roi_l)
        e = dict(ours_A=Ar.round(5).tolist(), ours_t=t.round(3).tolist(), theirs_A=M[:2, :2].round(5).tolist(),
                 theirs_t=M[:2, 2].round(3).tolist(),
                 diff_A_as_given=float(np.abs(Ar - M[:2, :2]).max()), diff_A_transposed=float(np.abs(Ar - M[:2, :2].T).max()),
                 diff_t=(t - M[:2, 2]).round(3).tolist(), rois=info['rois'])
        # displacement difference over the ROI (transposed linear part, their translation)
        g = np.stack(np.meshgrid(np.linspace(0, 199, 5), np.linspace(0, 199, 5)), -1).reshape(-1, 2)
        Pm = np.array([[0.0, 1.0], [1.0, 0.0]])
        for name, L, tt in (('as_given', M[:2, :2], M[:2, 2]), ('transposed', M[:2, :2].T, M[:2, 2]),
                            ('yx_order', Pm @ M[:2, :2] @ Pm, Pm @ M[:2, 2])):
            e[f'max_px_diff_{name}'] = float(np.abs(g @ Ar.T + t - (g @ L.T + tt)).max())
        e['scale_theirs'] = np.sqrt(np.abs(np.linalg.det(M[:2, :2]))).round(4).tolist()
        out[f'{col}:{f.split(col + "/")[1]}'] = e
        print(col, f.split(col + '/')[1], 'ours t', t.round(2), 'theirs t', M[:2, 2].round(2),
              'max px diff as given / transposed / yx', round(e['max_px_diff_as_given'], 2),
              round(e['max_px_diff_transposed'], 2), round(e['max_px_diff_yx_order'], 2), 'scale', e['scale_theirs'])
json.dump(out, open(os.path.expanduser('~/mcfield_out/compare_smlm_maps.json'), 'w'), indent=1)
