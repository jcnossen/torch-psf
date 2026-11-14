"""3D localization of an astigmatic EMCCD SMLM movie with the bead-calibrated spline PSF.

Per chunk of frames: ADU -> photoelectrons ((ADU - offset) / g), difference-of-Gaussians detection with a threshold
relative to the per-frame robust noise of the DoG image, P x P patches around the local maxima (pixels outside the frame
are masked out of the likelihood), Poisson maximum-likelihood fit of (x, y, z, I, bg) on the spline (Levenberg-Marquardt,
GPU) with several starting heights. EMCCD excess noise: the Poisson ML estimate is unchanged by the excess factor 2
(the score equations are the same for data and model scaled by 1/2), but the Fisher information is halved, so the
reported CRLBs are the Poisson CRLBs times sqrt(2), and the goodness of fit is chi2 = mean over pixels of
(d - mu)^2 / (2 mu) (expected 1 for a correct model).

usage: python loc.py --spline psf_spline.pt --gain 10.9 --offset 295 --out locs.csv movie1.tif [movie2.tif ...]
       [--frames a:b] [--crop-ref full.tif]   (for a crop file: its offset in the full frame is found from frame 0)
"""
import sys, time, argparse, json, math
import numpy as np, torch, tifffile
sys.path.insert(0, '.')
from torch_psf import ConvCubicSplinePSF
from torch_psf.fit import crlb as crlb_psf
from torch_psf.fit import lm_poisson, model_and_jacobian
from rdcommon import transfer_knots, DEV

ap = argparse.ArgumentParser()
ap.add_argument('movies', nargs='+')
ap.add_argument('--spline', required=True)
ap.add_argument('--gain', type=float, required=True)      # ADU per photoelectron (EM gain included)
ap.add_argument('--offset', type=float, required=True)    # ADU
ap.add_argument('--out', required=True)
ap.add_argument('--frames', default='')
ap.add_argument('--crop-ref', default='')
ap.add_argument('--P', type=int, default=17)
ap.add_argument('--xmax', type=float, default=3.0)
ap.add_argument('--zlim', type=float, default=0.8)         # um, fit range (calibrated range of the spline)
ap.add_argument('--thresh', type=float, default=6.0)
ap.add_argument('--chunk', type=int, default=500)
ap.add_argument('--spots', type=int, default=20000)    # spots per GPU batch
ap.add_argument('--starts', default='-0.45,0,0.45')
ap.add_argument('--iters', type=int, default=12)
ap.add_argument('--iters2', type=int, default=20)
A = ap.parse_args()
P, h = A.P, A.P // 2

# ------------------------------------------------------------------------------------------------ spline
cal = torch.load(A.spline)
spl = ConvCubicSplinePSF(P, (cal['zlo'], cal['zhi']), oversample=cal['S'], dz_um=cal['hz'], zpad=cal['zpad'], xy_limit=A.xmax, device=DEV)
assert spl.nz == cal['nz'] and abs(spl.z0 - cal['z0']) < 1e-9
spl.knots = transfer_knots(cal['knots'].to(DEV), cal['c'], spl.c, cal['S'])
ZLO, ZHI = -A.zlim, A.zlim
assert spl.z0 + spl.hz <= ZLO and ZHI <= spl.z0 + (spl.nz - 2) * spl.hz
off0 = torch.zeros(1, 3, device=DEV)
frac = float(spl.render(off0).sum())
print(f'spline: P {P}, xmax {A.xmax}, c {spl.c}, S {spl.S}, hz {spl.hz}, z0 {spl.z0:.3f}, nz {spl.nz}; '
      f'fraction of the PSF inside the {P}x{P} patch at z = 0: {frac:.4f}', flush=True)


def clamp(th):
    th[:, 0].clamp_(-A.xmax + 0.05, A.xmax - 0.05)
    th[:, 1].clamp_(-A.xmax + 0.05, A.xmax - 0.05)
    th[:, 2].clamp_(ZLO, ZHI)
    th[:, 3].clamp_min_(1.0)
    th[:, 4].clamp_min_(1e-3)
    return th


def model(t):
    return model_and_jacobian(spl, t)


@torch.no_grad()
def lm(d, w, th, iters, nb=6000):
    """torch_psf.lm_poisson with the pixel mask w (frame border), mu >= 1e-4 and this script's clamp."""
    out, onll = [], []
    for s in range(0, len(d), nb):
        t, nll = lm_poisson(model, d[s:s + nb], th[s:s + nb], weight=w[s:s + nb], clamp=clamp, iters=iters,
                            mu_min=1e-4)
        out.append(t); onll.append(nll)
    return torch.cat(out), torch.cat(onll)


@torch.no_grad()
def crlb(th, w):
    return crlb_psf(spl, th, weight=w, ridge=1e-9)


def gk(s):
    x = torch.arange(-int(4 * s), int(4 * s) + 1, device=DEV, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * s * s))
    return k / k.sum()


def blur(x, s):
    # reflect padding: zero padding makes the frame border look like a bright edge in the DoG (false detections)
    k = gk(s)
    r = len(k) // 2
    x = torch.nn.functional.pad(x[:, None], (r, r, r, r), mode='reflect')
    x = torch.nn.functional.conv2d(x, k[None, None, :, None])
    return torch.nn.functional.conv2d(x, k[None, None, None, :])[:, 0]


# ------------------------------------------------------------------------------------------------ movie
from rdcommon import open_stack
mm = [open_stack(f) for f in A.movies]
T = [m.shape[0] for m in mm]
cum = np.cumsum([0] + T)
TT, H, W = int(cum[-1]), mm[0].shape[1], mm[0].shape[2]
f0, f1 = (0, TT)
if A.frames:
    a, b = A.frames.split(':'); f0, f1 = int(a), min(int(b), TT)
cx0 = cy0 = 0
if A.crop_ref:
    ref = tifffile.memmap(A.crop_ref)[0].astype(np.int32)
    c0 = np.asarray(mm[0][0]).astype(np.int32)
    hits = [(x, y) for y in range(ref.shape[0] - H + 1) for x in range(ref.shape[1] - W + 1)
            if np.array_equal(ref[y:y + H, x:x + W], c0)]
    assert len(hits) == 1, hits
    cx0, cy0 = hits[0]
    print(f'crop offset in the full frame: x0 {cx0}, y0 {cy0}', flush=True)


def frames(a, b):
    out = []
    for k in range(len(mm)):
        lo, hi = max(a, cum[k]), min(b, cum[k + 1])
        if lo < hi:
            out.append(np.asarray(mm[k][lo - cum[k]:hi - cum[k]]))
    return np.concatenate(out).astype(np.float32)


starts = [float(s) for s in A.starts.split(',')]
cols = ['frame', 'x', 'y', 'z', 'photons', 'bg', 'crlb_x', 'crlb_y', 'crlb_z', 'crlb_photons', 'chi2', 'nll_ratio',
        'flags', 'n_neighbors', 'peak_x', 'peak_y']
res = []
ii = torch.arange(P, device=DEV)

def fit_spots(t, y, x, nnb, pad, mpad, a):
    yy = (y[:, None, None] + ii[None, :, None]).expand(-1, P, P)
    xx = (x[:, None, None] + ii[None, None, :]).expand(-1, P, P)
    d = pad[t[:, None, None], yy, xx]
    w = mpad[yy, xx]
    edge = (w.flatten(1).min(1).values < 0.5)
    d = d * w
    # background start: 20 % quantile of the valid pixels (masked pixels set to +inf)
    dq = torch.where(w > 0, d, torch.full_like(d, float('inf'))).flatten(1)
    nv = w.flatten(1).sum(1)
    srt = dq.sort(1).values
    bg0 = srt.gather(1, (0.2 * nv).long().clamp(0, P * P - 1)[:, None])[:, 0].clamp_min(1.0)
    I0 = ((d - bg0[:, None, None]) * w).flatten(1).sum(1).clamp_min(50.0) / frac
    best, bnll = None, None
    for zs in starts:
        th0 = torch.stack([torch.zeros_like(I0), torch.zeros_like(I0), torch.full_like(I0, zs), I0, bg0], 1)
        th, nll = lm(d, w, th0, A.iters)
        if best is None:
            best, bnll = th, nll
        else:
            m = nll < bnll
            best[m], bnll[m] = th[m], nll[m]
    th, nll = lm(d, w, best, A.iters2)
    cr = crlb(th, w) * math.sqrt(2.0)
    mu, _ = model(th)
    mu = mu.clamp_min(1e-4)
    c2 = (((d - mu) ** 2 / (2 * mu)) * w).flatten(1).sum(1) / nv
    # log-likelihood ratio per pixel against the saturated model (Poisson deviance / 2 / npix, with data halved for
    # the EMCCD excess noise): ~0.5 for a correct model
    dh, mh = d.clamp_min(0) / 2, mu / 2
    dev = torch.where(dh > 0, dh * torch.log(dh.clamp_min(1e-9) / mh), torch.zeros_like(dh)) - (dh - mh)
    llr = (dev * w).flatten(1).sum(1) / nv
    flags = ((th[:, :2].abs().max(1).values > A.xmax - 0.06).int() * 1
             + ((th[:, 2] <= ZLO + 0.005) | (th[:, 2] >= ZHI - 0.005)).int() * 2
             + edge.int() * 4 + (nnb > 0).int() * 8
             + ((th[:, 3] <= 1.001) | (th[:, 4] <= 1.001e-3)).int() * 16)
    X = x.float() + th[:, 0] + cx0
    Y = y.float() + th[:, 1] + cy0
    return torch.stack([(t + a).float(), X, Y, th[:, 2] * 1000, th[:, 3], th[:, 4], cr[:, 0], cr[:, 1], cr[:, 2] * 1000,
                        cr[:, 3], c2, llr, flags.float(), nnb.float(), (x + cx0).float(), (y + cy0).float()], 1)


t_start = time.time()
stats = dict(detected=0)
fmt = ['%d', '%.4f', '%.4f', '%.2f', '%.1f', '%.3f', '%.5f', '%.5f', '%.2f', '%.1f', '%.4f', '%.4f', '%d', '%d', '%d', '%d']
fout = open(A.out, 'w')
fout.write(','.join(cols) + chr(10))
nrow = 0
for a in range(f0, f1, A.chunk):
    b = min(a + A.chunk, f1)
    fr = torch.from_numpy(frames(a, b)).to(DEV)
    ph = (fr - A.offset) / A.gain                           # photoelectrons
    dog = blur(ph, 1.5) - blur(ph, 5.0)
    med = dog.flatten(1).median(1).values[:, None, None]
    sig = 1.4826 * (dog - med).abs().flatten(1).median(1).values[:, None, None]
    mx = torch.nn.functional.max_pool2d(dog[:, None], 7, 1, 3)[:, 0]
    pk = (dog == mx) & (dog - med > A.thresh * sig)
    pk[:, :2] = False; pk[:, -2:] = False; pk[:, :, :2] = False; pk[:, :, -2:] = False
    t, y, x = torch.nonzero(pk, as_tuple=True)
    n = len(t)
    if n == 0:
        continue
    # neighbours: other peaks inside the patch of this one
    cnt = torch.nn.functional.conv2d(pk[:, None].float(), torch.ones(1, 1, P, P, device=DEV), padding=h)[:, 0]
    nnb = (cnt[t, y, x] - 1).round().int()
    pad = torch.nn.functional.pad(ph, (h, h, h, h))
    mpad = torch.nn.functional.pad(torch.ones_like(ph[:1]), (h, h, h, h))[0]
    blks = []
    for s0 in range(0, n, A.spots):                          # spots in batches (GPU memory)
        sl = slice(s0, min(s0 + A.spots, n))
        blks.append(fit_spots(t[sl], y[sl], x[sl], nnb[sl], pad, mpad, a))
    blk = torch.cat(blks)
    del pad, mpad, dog, mx, pk, ph, fr, cnt
    Rb = blk.cpu().numpy().astype(np.float64)
    np.savetxt(fout, Rb, delimiter=',', fmt=fmt)
    fout.flush()
    nrow += len(Rb)
    stats['detected'] += n
    el = time.time() - t_start
    print(f'frames {a}-{b}: {n} spots ({n / (b - a):.1f}/frame), total {stats["detected"]}, {el:.0f}s, '
          f'{(b - f0) / el:.0f} frames/s', flush=True)
fout.close()
json.dump(dict(args=vars(A), frac_in_patch_z0=frac, crop_offset=[int(cx0), int(cy0)], n=int(nrow),
               frames=[f0, f1], seconds=time.time() - t_start, H=H, W=W),
          open(A.out.replace('.csv', '_run.json'), 'w'), indent=1)
print(f'done: {nrow} localizations in {time.time() - t_start:.0f}s -> {A.out}', flush=True)
