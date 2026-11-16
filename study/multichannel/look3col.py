"""3-colour bead stacks: detections per half, registration (flip / shift / affine), per-bead photometry (background-
subtracted box sums over all planes) in both halves and the ratio right / left per bead, crops of matched beads.
  python -I look3col.py   -> OUT/3col/look3col.json, look3col.png"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *                                     # noqa: F401,F403
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from torch_psf.estimation.multichannel_beads import detect_channels, register_channels

out = os.path.join(OUT, '3col'); os.makedirs(out, exist_ok=True)
res = {}
fig, ax = plt.subplots(4, 8, figsize=(20, 10))
for s, (f, raw) in enumerate(zip(FILES3, load(FILES3))):
    ph = (raw - CAM.offset_adu) * CAM.gain_e_per_adu
    nk = ph.shape[0]
    cs = [[torch.as_tensor(ph[::2, y0:y1, x0:x1].copy())] for x0, y0, x1, y1 in REGIONS]
    dets = detect_channels(cs, threshold=8.0)
    maps, info = register_channels(dets, [(400, 475)] * 2)
    i = info[1]
    pairs = [(a, b) for (_, a, b) in i['pairs']]
    rows = []
    for k, (a, b) in enumerate(pairs):
        v = []
        for c, idx in ((0, a), (1, b)):
            x, y = np.round(dets[c][0]['xy'][idx]).astype(int)
            x0 = REGIONS[c][0]
            box = ph[:, max(y - 12, 0):y + 13, x0 + max(x - 12, 0):x0 + x + 13]
            ring = np.concatenate([box[:, 0].ravel(), box[:, -1].ravel(), box[:, :, 0].ravel(), box[:, :, -1].ravel()])
            v.append(float(box.sum() - np.median(ring) * box.size))
            if k < 4:
                for j, z in enumerate((nk // 2 - 50, nk // 2, nk // 2 + 50, int(np.argmax(box.reshape(nk, -1).max(1))))):
                    ax[s * 2 + c, k * 2 + (j // 2)].imshow(box[z], cmap='magma') if j in (1, 3) else None
        rows.append(dict(xy_left=dets[0][0]['xy'][a].tolist(), xy_right=dets[1][0]['xy'][b].tolist(),
                         sum_left=v[0], sum_right=v[1], ratio_right_left=v[1] / v[0] if v[0] > 0 else None,
                         amp_left=float(dets[0][0]['amp'][a]), amp_right=float(dets[1][0]['amp'][b])))
    res[os.path.basename(f)] = dict(n_det=[int(len(d[0]['xy'])) for d in dets],
                                    n_bright=[int(d[0]['bright'].sum()) for d in dets],
                                    registration={k: v for k, v in i.items() if k in ('flip', 'n_pairs', 'rms_px', 'coarse_shift', 'describe')},
                                    beads=rows)
    print(f, res[os.path.basename(f)]['n_det'], res[os.path.basename(f)]['n_bright'], json.dumps(res[os.path.basename(f)]['registration'], default=str))
    for r in sorted(rows, key=lambda r: -r['sum_left']):
        print('  L %.0f %.0f  R %.0f %.0f  sumL %.3g sumR %.3g ratio %.3f' % (*r['xy_left'], *r['xy_right'], r['sum_left'], r['sum_right'], r['ratio_right_left'] or np.nan))
plt.savefig(os.path.join(out, 'look3col.png'), dpi=60)
json.dump(res, open(os.path.join(out, 'look3col.json'), 'w'), indent=1, default=float)
