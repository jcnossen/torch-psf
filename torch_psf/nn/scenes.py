"""Training scenes of SMLMNet: simulated plain-SMLM windows of T frames with their labels (the SIMCODE plain-SMLM
simulator, simcode.plain.simulate.PlainSim, and its labels, on torch_psf PSFs and a torch.Generator).

One scene batch = nb fields of view of H x W px (size + 2 halo), T frames each (T = window; frame T // 2 is labelled):

  emitters   blinking (dSTORM): Poisson on-events per field (density `dens` per px^2, log-uniform per field) with
             continuous on-intervals [t0, t1) (exponential duration, mean on-time `tau` frames, log-uniform per field)
             and the exact per-frame on-fraction; a fraction p_reblink of the events blinks on again at the same
             position after an exponential gap (mean gap_mean frames); close pairs (pair_frac, separation pair_sep px,
             their own timing); positions uniform, or for a share struct_frac of the fields mostly (85 %) on smooth
             random filaments (fil_density filaments of 256 px per px of the frame perimeter scale, width fil_w_nm);
             z uniform over z_range_um (the PSF's z). Emitters are simulated in a margin of 6 px around the field.
  photons    per-field median log-uniform `phot` (photoelectrons per full frame), per emitter log-normal (log sd
             phot_sd, clipped to phot_clip), per frame x (1 + frame_sd N(0, 1)) clamped at 0.3, x the on-fraction
  PSF        a bank of PSF files (torch_psf PSF.load, resized to roisize; each emitter draws one with psf_weights) at
             the sub-pixel offset and z, and a per-emitter Gaussian blur of one of 4 levels evenly over blur_px (px)
  background per field log-uniform `bg` (photoelectrons / px / frame) x (1 + bg_sd S_8 + bg_hp S_2.5) (S_s: smooth
             unit-variance random fields of Gaussian scale s px), clamped at 0.3 x the level, x a per-frame global
             AR(1) fluctuation (coefficient 0.5, relative sd bg_fluct)
  camera     EMCCD: n ~ Poisson(mu) photoelectrons -> EM register Gamma(n) x gain -> + read noise N(0, rn_adu) ADU ->
             + offset (+ per-field offset error off_err ADU), rounded, clipped to 0 .. 65535; per-field gain error
             (relative gain_err); the frames are converted back to photoelectrons with the nominal gain and offset

Labels (targets of the centre frame): emitters whose rounded position lies in the supervised square (size x size px
inside the halo), with on-fraction >= min_on and photons >= label_min in the centre frame; target columns active, x,
y (px, image coordinates of the supervised square), z (um), bg (the centre-frame background at the emitter's pixel),
photons (centre frame).

All random numbers come from the given torch.Generator (on the scene device): the same seed gives the same scenes.
"""
import math
import torch
import torch.nn.functional as F
from torch_psf.optics import resolve_device

# The simulator ranges (SIMCODE's plain-SMLM defaults, simcode.plain.simulate.RANGES).
RANGES = dict(dens=(3e-4, 1.5e-2), tau=(0.7, 5.0), p_reblink=0.15, gap_mean=1.5, pair_frac=0.1, pair_sep=(0.3, 3.0),
              struct_frac=0.5, fil_density=0.22, fil_w_nm=12.0, z_range_um=(-0.75, 0.75), phot=(400.0, 6000.0),
              phot_sd=0.5, phot_clip=(60.0, 60000.0), frame_sd=0.1, bg=(6.0, 80.0), bg_sd=0.25, bg_hp=0.04,
              bg_fluct=0.03, blur_px=(0.0, 0.3), gain_err=0.04, off_err=1.5)
# A scenes config: the PSF bank, the frame window, the camera, the pixel size, the label rule, the ranges.
SCENE_DEFAULTS = dict(psf=[], psf_weights=None, roisize=23, window=3, pixel_nm=117.6, pixel_nm_y=None,
                      camera=dict(adu_per_e=17.0, offset_adu=400.0, readnoise_adu=4.5), label_min=100.0, min_on=0.0,
                      ranges={})
BLUR_LEVELS = 4
MARGIN = 6


def cumsum(x, dim):
    """torch.cumsum; on CUDA in strict deterministic mode on the CPU (CUDA has no deterministic float prefix sum)."""
    if x.is_cuda and torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled():
        return torch.cumsum(x.cpu(), dim).to(x.device)
    return torch.cumsum(x, dim)


def gamma_unit(alpha, generator):
    """Gamma(alpha, 1) samples for alpha >= 1 (Marsaglia & Tsang 2000), from `generator`."""
    d = alpha - 1. / 3.
    c = 1. / torch.sqrt(9. * d)
    out = torch.empty_like(alpha)
    todo = torch.ones_like(alpha, dtype=torch.bool)
    while bool(todo.any()):
        idx = torch.nonzero(todo, as_tuple=True)
        dd, cc = d[idx], c[idx]
        x = torch.randn(dd.shape, generator=generator, device=alpha.device)
        u = torch.rand(dd.shape, generator=generator, device=alpha.device)
        v = (1. + cc * x) ** 3
        ok = (v > 0) & (torch.log(u.clamp(min=1e-38)) < 0.5 * x * x + dd - dd * v + dd * torch.log(v.clamp(min=1e-38)))
        sel = tuple(i[ok] for i in idx)
        out[sel] = (dd * v)[ok]
        todo[sel] = False
    return out


def gauss_blur(img, s):
    """[N, R, R] blurred by a Gaussian of s px (replicate edges), the sum kept."""
    k = max(1, int(math.ceil(3 * s)))
    o = torch.arange(-k, k + 1, device=img.device, dtype=torch.float32)
    g = torch.exp(-0.5 * (o / s) ** 2); g = g / g.sum()
    x = img[:, None]
    x = F.conv2d(F.pad(x, (k, k, 0, 0), mode='replicate'), g.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, k, k), mode='replicate'), g.view(1, 1, -1, 1))
    x = x[:, 0]
    return x * (img.sum((1, 2), keepdim=True) / x.sum((1, 2), keepdim=True).clamp(min=1e-12))


def scene_config(cfg=None, **over):
    """SCENE_DEFAULTS <- cfg <- over (the ranges block merged with RANGES); checks the keys."""
    c = dict(SCENE_DEFAULTS, **(cfg or {}), **over)
    unknown = set(c) - set(SCENE_DEFAULTS)
    if unknown:
        raise ValueError(f'unknown scene settings {sorted(unknown)}')
    bad = set(c['ranges']) - set(RANGES)
    if bad:
        raise ValueError(f'unknown scene ranges {sorted(bad)}')
    c['ranges'] = dict(RANGES, **c['ranges'])
    c['camera'] = dict(SCENE_DEFAULTS['camera'], **c['camera'])
    if not c['psf']:
        raise ValueError('scenes: psf (a list of PSF files) is required')
    if int(c['window']) < 1 or int(c['window']) % 2 == 0:
        raise ValueError('scenes: window must be odd (the labelled frame in the centre)')
    return c


class Scenes:
    """Scene generator (module docstring). cfg: a scenes config (scene_config); size: the supervised square (px),
    halo: context simulated around it (px); batch: fields per batch; device; generator / seed: the random stream."""
    def __init__(self, cfg, batch=16, size=40, halo=0, device=None, generator=None, seed=0):
        from torch_psf.psf import PSF
        self.cfg = c = scene_config(cfg)
        self.p = c['ranges']
        self.dev = resolve_device(device)
        self.batch, self.size, self.halo = int(batch), int(size), int(halo)
        self.H = self.W = self.size + 2 * self.halo
        self.T, self.R = int(c['window']), int(c['roisize'])
        zr = tuple(self.p['z_range_um'])
        self.bank = []
        for path in c['psf']:
            psf = PSF.load(str(path), device=self.dev)
            if psf.z_range is not None and (zr[0] < psf.z_range[0] - 1e-9 or zr[1] > psf.z_range[1] + 1e-9):
                raise ValueError(f'{path}: z range {zr} beyond the PSF z range {psf.z_range}')
            self.bank.append(psf if psf.roisize == self.R else psf.resized(self.R))
        w = torch.tensor(c['psf_weights'] if c['psf_weights'] is not None else [1.0] * len(self.bank),
                         dtype=torch.float32)
        self.w = (w / w.sum()).to(self.dev)
        self.g = generator if generator is not None else torch.Generator(device=self.dev).manual_seed(int(seed))

    def seed(self, seed):
        """Restart the random stream at `seed`."""
        self.g.manual_seed(int(seed))
        return self

    # ------------------------------------------------------------------ random numbers
    def _rand(self, *n):
        return torch.rand(*n, generator=self.g, device=self.dev)

    def _randn(self, *n):
        return torch.randn(*n, generator=self.g, device=self.dev)

    def _lu(self, r, *n):
        lo, hi = r
        return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * self._rand(*n))

    def _uni(self, r, *n):
        return r[0] + (r[1] - r[0]) * self._rand(*n)

    def _smooth_field(self, n, sig):
        H, W = self.H, self.W
        k = int(3 * sig)
        o = torch.arange(-k, k + 1, device=self.dev, dtype=torch.float32)
        g = torch.exp(-0.5 * (o / sig) ** 2); g = g / g.sum()
        x = self._randn(n, 1, H + 2 * k, W + 2 * k)
        x = F.conv2d(F.conv2d(x, g.view(1, 1, 1, -1)), g.view(1, 1, -1, 1))[:, 0]
        return (x - x.mean((1, 2), keepdim=True)) / x.std((1, 2), keepdim=True).clamp(min=1e-6)

    # ------------------------------------------------------------------ emitters
    def _filament_xy(self, nb, fov, margin):
        p, H, W, dev = self.p, self.H, self.W, self.dev
        L, ns = 256.0, 128
        nfil = max(1, int(round(p['fil_density'] * (W + 2 * margin) * (H + 2 * margin) / L)))
        p0 = torch.stack([-margin + (W + 2 * margin) * self._rand(nb, nfil),
                          -margin + (H + 2 * margin) * self._rand(nb, nfil)], -1)
        th0 = self._rand(nb, nfil) * 2 * math.pi
        dth = cumsum(self._randn(nb, nfil, ns) * 0.12, -1) + th0[..., None]
        d = torch.stack([torch.cos(dth), torch.sin(dth)], -1) * (L / ns)
        pts = p0[..., None, :] + cumsum(d, -2) - L / 2 * torch.stack([torch.cos(th0), torch.sin(th0)], -1)[..., None, :]
        N = len(fov)
        fi = torch.randint(0, nfil, (N,), generator=self.g, device=dev)
        s = self._rand(N) * (ns - 1)
        si = s.floor().long(); fr = (s - si)[:, None]
        P0 = pts[fov, fi, si]; P1 = pts[fov, fi, (si + 1).clamp(max=ns - 1)]
        return P0 + (P1 - P0) * fr + self._randn(N, 2) * (p['fil_w_nm'] / self.cfg['pixel_nm'])

    def emitters(self, nb, margin=MARGIN):
        """Emitters of nb fields: dict fov, x, y (px), z (um), bright (photons per full frame), t0, t1 (frames)."""
        p, H, W, T, dev = self.p, self.H, self.W, self.T, self.dev
        d, ta, Bm = self._lu(p['dens'], nb), self._lu(p['tau'], nb), self._lu(p['phot'], nb)
        area = (H + 2 * margin) * (W + 2 * margin)
        t_lo = -6 * ta
        n_ev = torch.poisson(d * area * (T - t_lo) / ta, generator=self.g).long()
        fov = torch.repeat_interleave(torch.arange(nb, device=dev), n_ev)
        N = len(fov)
        t0 = t_lo[fov] + (T - t_lo[fov]) * self._rand(N)
        t1 = t0 - torch.log(self._rand(N).clamp(min=1e-9)) * ta[fov]
        x = -margin + (W + 2 * margin) * self._rand(N)
        y = -margin + (H + 2 * margin) * self._rand(N)
        st = self._rand(nb) < p['struct_frac']
        if st.any() and N:
            q = self._filament_xy(nb, fov, margin)
            use = (st[fov] & (q[:, 0] > -margin) & (q[:, 0] < W + margin) & (q[:, 1] > -margin)
                   & (q[:, 1] < H + margin) & (self._rand(N) < 0.85))
            x = torch.where(use, q[:, 0], x); y = torch.where(use, q[:, 1], y)
        z = self._uni(p['z_range_um'], N)
        br = (Bm[fov] * torch.exp(p['phot_sd'] * self._randn(N))).clamp(*p['phot_clip'])
        rb = torch.nonzero(self._rand(N) < p['p_reblink'])[:, 0]            # re-blinks after a gap
        if len(rb):
            g0 = t1[rb] - torch.log(self._rand(len(rb)).clamp(min=1e-9)) * p['gap_mean']
            g1 = g0 - torch.log(self._rand(len(rb)).clamp(min=1e-9)) * ta[fov[rb]]
            fov = torch.cat([fov, fov[rb]]); x = torch.cat([x, x[rb]]); y = torch.cat([y, y[rb]])
            z = torch.cat([z, z[rb]]); br = torch.cat([br, br[rb]]); t0 = torch.cat([t0, g0]); t1 = torch.cat([t1, g1])
        N = len(fov)
        pi = torch.nonzero(self._rand(N) < p['pair_frac'])[:, 0]           # close pairs with their own timing
        M = len(pi)
        if M:
            sep = self._uni(p['pair_sep'], M); th = self._rand(M) * 2 * math.pi
            tp = ta[fov[pi]]
            pt0 = -tp + (T + tp) * self._rand(M)
            pt1 = pt0 + 0.3 - torch.log(self._rand(M).clamp(min=1e-9)) * tp
            fov = torch.cat([fov, fov[pi]]); t0 = torch.cat([t0, pt0]); t1 = torch.cat([t1, pt1])
            x = torch.cat([x, x[pi] + sep * torch.cos(th)]); y = torch.cat([y, y[pi] + sep * torch.sin(th)])
            z = torch.cat([z, self._uni(p['z_range_um'], M)])
            br = torch.cat([br, (Bm[fov[-M:]] * torch.exp(p['phot_sd'] * self._randn(M))).clamp(*p['phot_clip'])])
        keep = (t1 > 0) & (t0 < T) & (x > -margin) & (x < W + margin) & (y > -margin) & (y < H + margin)
        return dict(fov=fov[keep], x=x[keep], y=y[keep], z=z[keep], bright=br[keep], t0=t0[keep], t1=t1[keep])

    # ------------------------------------------------------------------ images
    @torch.no_grad()
    def render(self, nb, em, margin=MARGIN):
        """Noise-free emitter images [nb, T, H, W], on-fractions [N, T], photons [N, T]."""
        p, H, W, T, R, dev = self.p, self.H, self.W, self.T, self.R, self.dev
        fov, x, y, z = em['fov'], em['x'], em['y'], em['z']
        N = len(fov)
        fr = torch.arange(T, device=dev, dtype=torch.float32)
        onf = (torch.minimum(em['t1'][:, None], fr[None] + 1) - torch.maximum(em['t0'][:, None], fr[None])).clamp(0, 1)
        phot = em['bright'][:, None] * onf * (1 + p['frame_sd'] * self._randn(N, T)).clamp(min=0.3)
        hr = R // 2
        img = torch.zeros(nb, T, H + 2 * (margin + hr) + 2, W + 2 * (margin + hr) + 2, device=dev)
        if N:
            pos = torch.stack([x - torch.round(x), y - torch.round(y), z], 1)
            grp = (torch.multinomial(self.w, N, replacement=True, generator=self.g) if len(self.bank) > 1
                   else torch.zeros(N, dtype=torch.long, device=dev))
            psf = torch.zeros(N, R, R, device=dev)
            for gi, pm in enumerate(self.bank):
                m = grp == gi
                if m.any():
                    psf[m] = pm.render(pos[m]).float()
            b0, b1 = p['blur_px']
            if b1 > 0:
                lvl = torch.randint(0, BLUR_LEVELS, (N,), generator=self.g, device=dev)
                for li in range(BLUR_LEVELS):
                    s = b0 + (b1 - b0) * li / (BLUR_LEVELS - 1)
                    m = lvl == li
                    if s > 0 and m.any():
                        psf[m] = gauss_blur(psf[m], s)
            cx = torch.round(x).long() + margin + 1; cy = torch.round(y).long() + margin + 1
            oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
            iy = (cy[:, None, None] + oy).reshape(N, -1); ix = (cx[:, None, None] + ox).reshape(N, -1)
            Hp, Wp = img.shape[-2:]
            lin = (fov[:, None, None] * T * Hp * Wp + torch.arange(T, device=dev)[None, :, None] * Hp * Wp
                   + (iy * Wp + ix)[:, None, :])
            img.view(-1).index_add_(0, lin.reshape(-1), (phot[:, :, None] * psf.reshape(N, 1, -1)).reshape(-1))
        off = margin + hr + 1
        return img[:, :, off:off + H, off:off + W], onf, phot

    def background(self, nb):
        """Background [nb, T, H, W] (photoelectrons / px / frame)."""
        p, T = self.p, self.T
        b0 = self._lu(p['bg'], nb).reshape(nb, 1, 1)
        st = (b0 * (1 + p['bg_sd'] * self._smooth_field(nb, 8.0) + p['bg_hp'] * self._smooth_field(nb, 2.5))
              ).clamp(min=0.3 * b0)
        a = 0.5
        gl = torch.zeros(nb, T, device=self.dev)
        gl[:, 0] = self._randn(nb)
        for t in range(1, T):
            gl[:, t] = a * gl[:, t - 1] + math.sqrt(1 - a * a) * self._randn(nb)
        return st[:, None] * (1 + p['bg_fluct'] * gl)[..., None, None]

    def camera(self, mu):
        """EMCCD chain (module docstring): expected photoelectrons mu [nb, T, H, W] -> frames in photoelectrons."""
        p, cam, nb = self.p, self.cfg['camera'], mu.shape[0]
        G = cam['adu_per_e'] * (1 + p['gain_err'] * self._randn(nb, 1, 1, 1))
        n = torch.poisson(mu.clamp(min=0), generator=self.g)   # (spline tables can ring slightly below 0)
        pos = n > 0
        amp = torch.zeros_like(n)
        amp[pos] = gamma_unit(n[pos], self.g)
        amp = amp * G
        adu = torch.round(amp + cam['readnoise_adu'] * self._randn(*mu.shape) + cam['offset_adu']
                          + p['off_err'] * self._randn(nb, 1, 1, 1)).clamp(0, 65535)
        return (adu - cam['offset_adu']) / cam['adu_per_e']

    @torch.no_grad()
    def scene(self, nb=None):
        """One scene batch: dict frames [nb, T, H, W] (photoelectrons), bg [nb, T, H, W], gt (fov, x, y, z, bright,
        t0, t1, onf [N, T], phot [N, T])."""
        nb = self.batch if nb is None else int(nb)
        em = self.emitters(nb)
        img, onf, phot = self.render(nb, em)
        b = self.background(nb)
        frames = self.camera(img + b)
        return dict(frames=frames, bg=b, gt=dict(em, onf=onf, phot=phot))

    # ------------------------------------------------------------------ labels
    def labels(self, scene):
        """Labels of the centre frame (module docstring): dict fov, x, y (px, supervised-square coordinates), z (um),
        bg, photons, onf, index (row of scene['gt'])."""
        gt, h, t = scene['gt'], self.halo, self.T // 2
        x, y = gt['x'], gt['y']
        xr, yr = x.round(), y.round()
        phot, onf = gt['phot'][:, t], gt['onf'][:, t]
        keep = ((xr >= h) & (xr < self.W - h) & (yr >= h) & (yr < self.H - h) & (onf >= self.cfg['min_on'])
                & (phot >= self.cfg['label_min']))
        i = torch.nonzero(keep)[:, 0]
        fov = gt['fov'][i]
        xs, ys = x[i] - h, y[i] - h
        bgc = scene['bg'][:, t, h:self.H - h, h:self.W - h]
        xi = xs.round().long().clamp(0, self.size - 1); yi = ys.round().long().clamp(0, self.size - 1)
        return dict(fov=fov, x=xs, y=ys, z=gt['z'][i], bg=bgc[fov, yi, xi], photons=phot[i], onf=onf[i], index=i)

    def targets(self, labels, nb, z=True):
        """Labels -> targets [nb, S, 1 + n] (torch_psf.nn.loss), S = max labels per field (>= 1), in row order."""
        fov = labels['fov']
        counts = torch.bincount(fov, minlength=nb)
        S = max(1, int(counts.max()) if len(fov) else 1)
        start = torch.cumsum(counts, 0) - counts
        order = torch.argsort(fov, stable=True)
        slot = torch.empty_like(fov)
        slot[order] = torch.arange(len(fov), device=fov.device) - start[fov[order]]
        cols = ['x', 'y', 'z', 'bg', 'photons'] if z else ['x', 'y', 'bg', 'photons']
        out = torch.zeros(nb, S, 1 + len(cols), device=fov.device)
        out[fov, slot, 0] = 1.
        for j, k in enumerate(cols, 1):
            out[fov, slot, j] = labels[k].float()
        return out

    def batches(self, n, seed=None, z=True):
        """n training batches: (frames [B, T, H, W], targets [B, S, 1 + n_params], labels). seed: restart the stream."""
        if seed is not None:
            self.seed(seed)
        for _ in range(int(n)):
            sc = self.scene()
            lab = self.labels(sc)
            yield sc['frames'], self.targets(lab, sc['frames'].shape[0], z), lab
