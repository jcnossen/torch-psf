"""Dense blinking movies with a known PSF (SOFI data and dense SMLM movies), streamed in chunks of frames.

The sample is a set of static emitters (2D at one height, or uniform in a z range, optionally snapped to a sub-pixel
grid so that a grid model of the density rho is exact). Each emitter has a fixed brightness eps (photons per frame
when fully on) and an independent blinking trace (torch_psf.sofi.blink). The PSF is a VectorialPSF; every emitter is
rendered once on a Q x Q patch (and again whenever the drift moves it). Camera: Poisson shot noise on the expected
photons (+ background), optional EMCCD gain (Gamma, excess noise factor 2), Gaussian read noise (sCMOS, per-pixel sd
map optional), optional row- and column-correlated read noise, gain and offset (ADU).
"""
from dataclasses import dataclass, field, asdict
import torch
from torch_psf.sofi.blink import BlinkModel, simulate_traces


@dataclass
class CameraModel:
    gain_e_per_adu: float = 1.0
    offset_adu: float = 100.0
    readnoise_e: float = 1.5
    row_noise_e: float = 0.0       # sd of a per-frame, per-row offset (e-)
    col_noise_e: float = 0.0       # sd of a per-frame, per-column offset (e-)
    emccd_gain: float = 0.0        # 0 = sCMOS; > 0: electron multiplication (Gamma), excess noise factor 2

    def to_dict(self):
        return asdict(self)


def place_emitters(n, H, W, margin, z_range=None, z0=0.0, grid_os=None, nz_layers=None, generator=None, device='cuda'):
    """Uniform emitter positions x [n] (column), y [n] (row) in px within [margin, W - 1 - margin], z [n] (um).
    grid_os: snap x, y to multiples of 1 / grid_os px; nz_layers: snap z to nz_layers evenly spaced heights."""
    r = lambda: torch.rand(n, generator=generator, device=device)
    x = margin + (W - 1 - 2 * margin) * r()
    y = margin + (H - 1 - 2 * margin) * r()
    if z_range is None:
        z = torch.full((n,), float(z0), device=device)
    else:
        z = z_range[0] + (z_range[1] - z_range[0]) * r()
        if nz_layers:
            zl = torch.linspace(z_range[0], z_range[1], nz_layers, device=device)
            z = zl[torch.argmin((z[:, None] - zl[None]).abs(), 1)]
    if grid_os:
        x = torch.round(x * grid_os) / grid_os
        y = torch.round(y * grid_os) / grid_os
    return x, y, z


def render_patches(psf, x, y, z, Q, focus=None, nb=512):
    """PSF patches [n, Q, Q] of emitters at (x, y) px (frame coordinates) and z um, and the patch corner (row, col)
    [n, 2] (long): pixel (corner + u) of the frame holds patch[u]. Pixel-integrated, normalized (VectorialPSF)."""
    if psf.roisize != Q:
        psf = psf.resized(Q)
    f = psf.focus_um if focus is None else focus
    cx, cy = torch.floor(x + 0.5), torch.floor(y + 0.5)
    pos = torch.stack([x - cx, y - cy, z], 1)
    with torch.no_grad():
        h = torch.cat([psf(pos[s:s + nb], f) for s in range(0, len(x), nb)]) if len(x) else \
            torch.zeros(0, Q, Q, device=x.device)
    hr = (Q - 1) // 2
    corner = torch.stack([cy.long() - hr, cx.long() - hr], 1)
    return h, corner


def _scatter_matrix(h, corner, H, W):
    """Dense [n, H * W] footprint matrix (patches cropped at the frame edge)."""
    n, Q, _ = h.shape
    dev = h.device
    u = torch.arange(Q, device=dev)
    yy = corner[:, 0, None, None] + u[None, :, None]
    xx = corner[:, 1, None, None] + u[None, None, :]
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    M = torch.zeros(n, H * W, device=dev)
    rows = torch.arange(n, device=dev)[:, None, None].expand(n, Q, Q)
    M.index_put_((rows[ok], (yy * W + xx)[ok]), h[ok], accumulate=True)
    return M


@dataclass
class DenseMovieSpec:
    H: int = 64
    W: int = 64
    T: int = 10000
    density_um2: float = 50.0       # emitters per um^2
    z_range: tuple = None           # None: thin sample at z0
    z0: float = 0.0
    photons: float = 500.0          # eps: photons per frame when fully on (median)
    photons_logsd: float = 0.0      # log-normal spread of eps between emitters
    bg: float = 50.0                # photons / px / frame
    blink: BlinkModel = field(default_factory=BlinkModel)
    camera: CameraModel = field(default_factory=CameraModel)
    drift_px: tuple = None          # (dx, dy) total linear drift over the movie (px), applied per drift_block frames
    drift_block: int = 100
    Q: int = 15                     # render patch size (px)
    grid_os: int = None             # snap positions to a 1 / grid_os px grid (exact grid models)
    nz_layers: int = None
    margin: float = 0.0             # keep emitters this far inside the frame (px); 0 = emitters up to the edge
    chunk: int = 500
    seed: int = 0


class DenseMovie:
    """Simulated dense blinking movie. sample(): truth (x, y, z, eps). frames(): iterator over (t0, adu chunk [t, H,
    W] float32, photon-expectation chunk or None). Everything on `device`."""

    def __init__(self, psf, spec, device='cuda'):
        self.psf, self.spec, self.dev = psf, spec, torch.device(device)
        s = spec
        g = torch.Generator(device=self.dev).manual_seed(s.seed)
        self.g = g
        area_um2 = s.H * s.W * psf.pixel_um ** 2
        n = int(torch.poisson(torch.tensor(s.density_um2 * area_um2, device=self.dev), generator=g))
        # emitters may lie up to Q/2 outside the frame so that the density is uniform up to the edges
        ext = (s.Q - 1) / 2 if s.margin == 0 else -s.margin
        Hx, Wx = s.H + 2 * ext, s.W + 2 * ext
        n = int(round(n * Hx * Wx / (s.H * s.W)))
        x, y, z = place_emitters(n, int(round(Hx)), int(round(Wx)), 0.0, s.z_range, s.z0, s.grid_os, s.nz_layers,
                                 generator=g, device=self.dev)
        self.x, self.y, self.z = x - ext, y - ext, z
        self.eps = s.photons * torch.exp(s.photons_logsd * torch.randn(n, generator=g, device=self.dev))
        self.n = n

    def footprints(self, dx=0.0, dy=0.0):
        s = self.spec
        h, c = render_patches(self.psf, self.x + dx, self.y + dy, self.z, s.Q)
        return _scatter_matrix(h, c, s.H, s.W)

    def expected_mean(self):
        """Expected photon image (no drift): bg + sum_k eps_k E[s] h_k."""
        s = self.spec
        Es = s.blink.duty
        M = self.footprints()
        return (s.bg + (self.eps * Es) @ M).view(s.H, s.W)

    def frames(self, return_photons=False):
        s, g, dev = self.spec, self.g, self.dev
        cam = s.camera
        state = None
        M = None
        last_shift = None
        for t0 in range(0, s.T, s.chunk):
            tn = min(s.chunk, s.T - t0)
            tr, state = simulate_traces(s.blink, self.n, tn, generator=g, device=dev, state=state, return_state=True)
            out = torch.empty(tn, s.H, s.W, device=dev)
            for b0 in range(0, tn, s.drift_block if s.drift_px else tn):
                b1 = min(tn, b0 + (s.drift_block if s.drift_px else tn))
                if s.drift_px:
                    f = (t0 + b0 + 0.5 * (b1 - b0)) / s.T
                    shift = (s.drift_px[0] * f, s.drift_px[1] * f)
                else:
                    shift = (0.0, 0.0)
                if M is None or shift != last_shift:
                    M = self.footprints(*shift)
                    last_shift = shift
                mu = s.bg + (tr[b0:b1] * self.eps[None]) @ M              # [t, H*W] expected photons
                out[b0:b1] = mu.view(-1, s.H, s.W)
            ph = torch.poisson(out, generator=g)
            if cam.emccd_gain > 0:   # Gamma(k = n, scale = G): mean G n, variance G^2 n (excess noise factor 2)
                e = torch.where(ph > 0, torch._standard_gamma(ph.clamp(min=1e-6)) * cam.emccd_gain, torch.zeros_like(ph))
            else:
                e = ph
            e = e + cam.readnoise_e * torch.randn(e.shape, generator=g, device=dev)
            if cam.row_noise_e > 0:
                e = e + cam.row_noise_e * torch.randn(tn, s.H, 1, generator=g, device=dev)
            if cam.col_noise_e > 0:
                e = e + cam.col_noise_e * torch.randn(tn, 1, s.W, generator=g, device=dev)
            adu = e / cam.gain_e_per_adu + cam.offset_adu
            yield (t0, adu, out) if return_photons else (t0, adu)

    def to_photons(self, adu):
        """ADU -> photons (offset and gain; EMCCD gain divided out)."""
        cam = self.spec.camera
        e = (adu - cam.offset_adu) * cam.gain_e_per_adu
        return e / cam.emccd_gain if cam.emccd_gain > 0 else e
