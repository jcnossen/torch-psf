"""Matched-filter spot detection for engineered PSFs (tetrapod, large-range double helix, astigmatism over several um).

A peak of the band-pass filtered image (torch_psf.detect.detect_spots) sits on a lobe of such a PSF, several px to
tens of px from the emitter, and one spot can give two peaks. Here every frame (photons, minus its median) is
correlated (FFT, zero padded) with zero-mean, unit-norm templates of the PSF model at nz heights; each template
response is divided by its robust noise (1.4826 MAD over the frame), the maximum over the templates is taken, and
local maxima (window nms px) above `threshold` are the detections: the emitter centre to about a pixel, plus the
height of the best template. (Ported from the field-PSF study, study/fieldpsf/detect_tetrapod.py: on the FD-DeepLoc
6 um tetrapod the centre error is < 1 px and the template z within 0.1 um on simulated spots.)

    det = MatchedFilterDetector(psf, dz_um=0.2, size=61, nms=21)
    spots = extract_spots(movie, 41, camera, detector=det, threshold=7.0, isolation=41)
    InSituEstimator(..., detector=det, threshold=7.0, isolation=41, zstarts=9)

Coordinates: x = column, y = row, pixel centres at integers.
"""
import torch
import torch.nn.functional as Fn


@torch.no_grad()
def psf_templates(psf, z_um=None, dz_um=0.2, size=None):
    """Zero-mean, unit-norm templates [nz, size, size] of a PSF model (render at x = y = 0, heights z_um, default
    every dz_um over psf.z_range) and their heights [nz] (um). size: template size (odd px; default the model's
    roisize; a larger one holds more of a defocused engineered PSF)."""
    m = psf if size is None or size == psf.roisize else psf.resized(int(size))
    if z_um is None:
        lo, hi = psf.z_range
        z_um = torch.arange(float(lo), float(hi) + 1e-6, float(dz_um))
    z = torch.as_tensor(z_um, dtype=torch.float32, device=m.dev).reshape(-1)
    T = m.render(torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], 1))
    T = T - T.mean((1, 2), keepdim=True)
    return T / T.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12), z


class MatchedFilter:
    """Correlation of [H, W] frames with templates T [nz, s, s] (zero padding against wrap-around)."""

    def __init__(self, T, H, W, chunk=8):
        self.nz, s = T.shape[0], T.shape[-1]
        self.H, self.W, self.chunk = H, W, chunk
        self.Hp, self.Wp = H + s, W + s
        K = torch.zeros(self.nz, self.Hp, self.Wp, device=T.device)
        h = s // 2
        K[:, :s, :s] = T
        K = torch.roll(K, (-h, -h), (1, 2))              # template centre at the origin
        self.Kf = torch.conj(torch.fft.rfft2(K))

    @torch.no_grad()
    def response(self, frame):
        """(normalised best response [H, W], index of the best template [H, W]) of one frame (photons)"""
        f = frame - frame.median()
        F = torch.fft.rfft2(f, s=(self.Hp, self.Wp))
        best = torch.full((self.H, self.W), -1e9, device=frame.device)
        arg = torch.zeros((self.H, self.W), dtype=torch.long, device=frame.device)
        for z0 in range(0, self.nz, self.chunk):
            r = torch.fft.irfft2(F[None] * self.Kf[z0:z0 + self.chunk], s=(self.Hp, self.Wp))[:, :self.H, :self.W]
            med = r.flatten(1).median(1).values
            sig = 1.4826 * (r.flatten(1) - med[:, None]).abs().median(1).values
            r = (r - med[:, None, None]) / sig.clamp_min(1e-12)[:, None, None]
            m, a = r.max(0)
            upd = m > best
            best = torch.where(upd, m, best)
            arg = torch.where(upd, a + z0, arg)
        return best, arg


class MatchedFilterDetector:
    """A detector for extract_spots / InSituEstimator (detector=): __call__(frames [T, H, W] photons, threshold=7,
    border=0, isolation=None, nms=None) -> dict(frame, y, x, score, z) like detect_spots, plus z = the height (um)
    of the best template. psf: the PSF model of the templates (e.g. a bead fit or the start of an in-situ fit);
    z_um / dz_um / size: see psf_templates; nms: max-pool window (px; default 21). isolation (px): detections with
    another one within this Chebyshev distance in the same frame are dropped (both)."""

    def __init__(self, psf, z_um=None, dz_um=0.2, size=None, nms=21, chunk=8):
        self.T, self.z = psf_templates(psf, z_um, dz_um, size)
        self.nms, self.chunk = int(nms), chunk
        self._mf = {}

    def filter_for(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._mf:
            self._mf[key] = MatchedFilter(self.T.to(device), H, W, self.chunk)
        return self._mf[key]

    @torch.no_grad()
    def __call__(self, frames, threshold=7.0, border=0, isolation=None, nms=None, **ignored):
        T, H, W = frames.shape
        mf = self.filter_for(H, W, frames.device)
        nms = self.nms if nms is None else int(nms)
        out = {k: [] for k in ('frame', 'y', 'x', 'score', 'z')}
        z = self.z.to(frames.device)
        for t in range(T):
            best, arg = mf.response(frames[t])
            mx = Fn.max_pool2d(best[None, None], nms, 1, nms // 2)[0, 0]
            pk = (best == mx) & (best > threshold)
            y, x = torch.nonzero(pk, as_tuple=True)
            keep = torch.ones_like(y, dtype=torch.bool)
            if isolation is not None and len(y) > 1:     # pairwise Chebyshev distances (few detections per frame)
                dd = torch.maximum((x[:, None] - x[None]).abs(), (y[:, None] - y[None]).abs())
                dd.fill_diagonal_(10 ** 9)
                keep &= dd.min(1).values > int(isolation)
            if border:
                b = int(border)
                keep &= (x >= b) & (x < W - b) & (y >= b) & (y < H - b)
            y, x = y[keep], x[keep]
            out['frame'].append(torch.full_like(y, t)); out['y'].append(y); out['x'].append(x)
            out['score'].append(best[y, x]); out['z'].append(z[arg[y, x]])
        return {k: torch.cat(v) for k, v in out.items()}
