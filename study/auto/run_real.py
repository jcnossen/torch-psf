"""Band limit and mode of the real data sets (study/auto/datasets.py). Output in ~/autopsf_out (never in the repo):
cache/<name>_{spec,class}.npy (the frames read, so reruns do not touch the slow multi-GB TIFFs again),
json/<name>.json, figs/<name>.png, real_table.md.

Run with the data treated as untrusted: python -I study/auto/run_real.py [names...] [--device cuda:N]
"""
import os
import sys
import json
import time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import SETS, open_set                                    # noqa: E402
from plots import plot_result                                          # noqa: E402
from torch_psf.auto.analyze import choose_frames, analyze_frames      # noqa: E402
from torch_psf.auto.spectrum import read_frames                       # noqa: E402

OUT = os.path.expanduser('~/autopsf_out')


def frames(name, which, m, idx):
    p = f'{OUT}/cache/{name}_{which}.npy'
    pi = f'{OUT}/cache/{name}_{which}_idx.npy'
    if os.path.exists(p) and os.path.exists(pi) and np.array_equal(np.load(pi), idx):
        return np.load(p, allow_pickle=False)
    a = read_frames(m, idx)
    np.save(p, a.astype(np.float32)); np.save(pi, np.asarray(idx))
    return a


def expected(e):
    out = []
    if e.get('NA') and e.get('pixel_um') and e.get('wavelength_um'):
        out.append((2 * e['NA'] / e['wavelength_um'] * e['pixel_um'], f"2 NA/lambda (NA {e['NA']})"))
        if e.get('n_med') and e['n_med'] < e['NA']:
            out.append((2 * e['n_med'] / e['wavelength_um'] * e['pixel_um'], f"2 n_med/lambda ({e['n_med']})"))
    return out


def run(name, device):
    t0 = time.time()
    e = SETS[name]
    m, e = open_set(name)
    si, ci = choose_frames(m.n, frame_px=m.shape[0] * m.shape[1])
    fs = frames(name, 'spec', m, si)
    fc_ = fs if np.array_equal(si, ci) else frames(name, 'class', m, ci)
    r = analyze_frames(fs, fc_, pixel_um=e['pixel_um'], wavelength_um=e['wavelength_um'], device=device, si=si,
                       ci=ci)
    exp = expected(e)
    d = dict(name=name, meta={k: v for k, v in e.items()}, n_frames=m.n, frame_shape=list(m.shape),
             expected_fc_px=exp, result=r.to_dict(), seconds=time.time() - t0)
    with open(f'{OUT}/json/{name}.json', 'w') as f:
        json.dump(d, f, indent=1, default=str)
    plot_result(r, f'{OUT}/figs/{name}.png', title=f"{name}: {e['note']}  (expected mode {e['mode']})",
                expected_fc=exp)
    return d, r


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    device = 'cuda'
    if '--device' in sys.argv:
        device = sys.argv[sys.argv.index('--device') + 1]
        args = [a for a in args if a != device]
    for sub in ('cache', 'json', 'figs'):
        os.makedirs(f'{OUT}/{sub}', exist_ok=True)
    for name in args or list(SETS):
        try:
            d, r = run(name, device)
            print(f'{name}: {r.band.summary()} | mode {r.mode.mode} ({r.mode.confidence:.2f}) | {d["seconds"]:.0f}s',
                  flush=True)
        except Exception as ex:                                            # keep going over the data sets
            import traceback
            traceback.print_exc()
            print(f'{name}: FAILED {ex!r}', flush=True)


if __name__ == '__main__':
    main()
