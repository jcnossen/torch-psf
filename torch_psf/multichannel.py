"""MultiChannelPSF: the base class of PSF models whose emitter is seen in C channels at once ([N, C, P, P] images), and
ChannelsPSF: C independent single-channel models (biplane, multicolour, dual view).

A multi-channel model renders an emitter at a reference position (x, y, z) into C channel images. Each channel c sees
it at its own position

    pos_c = (x + dx_c + ox_c, y + dy_c + oy_c, z)

with channel_offsets [C, 2] = (dx, dy) px, a fixed image offset of the channel's model (default zeros; for 4Pi the
per-channel image offsets of a setup), and the optional per-spot offsets [N, C, 2] (o; e.g. the ROI-corner differences
from a channel mapping when the ROIs are cut per channel). A subclass implements only

    render_channels(pos_c [N, C, 3], extra [N, k] or None, grad=False) -> [N, C, P, P]
        (grad=True: (value, d/dx, d/dy, d/dz, d/d extra_0, ...), each [N, C, P, P]; channel c differentiated with
         respect to its own position pos_c[:, c])

and the base adds the channel geometry, the expected image, Fisher information / CRLB and the Poisson
maximum-likelihood fit of all channels jointly.

Emitter parameters: theta = [x, y, z, I, bg_0 .. bg_{C-1}, *extra] (4 + C + k columns), x, y relative to the centre of
the reference ROI (px), z (um), I (photons), bg_c (photons / px in channel c), extra = the model's extra emitter
parameters by name (extra_params; ('phi',) for 4Pi, () for independent channels). 2D models (z_range None) keep the
z column, held fixed (as PSF does). Expected image

    mu[:, c] = I intensity_ratio[c] g_c(pos_c; extra) + bg_c      (g = render_channels)

intensity_ratio [C] (default ones) is the relative brightness of the channels (psf.zip `intensity_ratio`); render()
includes it.

Units as in PSF: px laterally, um in z, photons.
"""
import math
import torch
from torch_psf.psf import PSF
from torch_psf.fit import lm_poisson, poisson_nll
from torch_psf.dataset import Dataset


def _as_t(v, dev, dtype=torch.float32):
    return torch.as_tensor(v, dtype=dtype, device=dev)


class MultiChannelPSF(PSF):
    """Abstract multi-channel PSF (module docstring). Subclasses call _init_channels(C, channel_offsets,
    intensity_ratio) in their constructor, set extra_params, and implement render_channels; they inherit render,
    expected, mu_and_jacobian, fisher, crlb, fit_theta and fit."""
    extra_params = ()            # names of the extra emitter parameters (theta columns after the backgrounds)

    # ------------------------------------------------------------------ channel geometry
    def _init_channels(self, C, channel_offsets=None, intensity_ratio=None):
        """Set channel_offsets [C, 2] (px) and intensity_ratio [C] (buffers of a torch.nn.Module subclass; the ratio
        is not persistent, i.e. not in its state_dict)."""
        dev = self.dev
        off = torch.zeros(C, 2, device=dev) if channel_offsets is None else _as_t(channel_offsets, dev).reshape(C, 2).clone()
        rat = torch.ones(C, device=dev) if intensity_ratio is None else _as_t(intensity_ratio, dev).reshape(C).clone()
        if isinstance(self, torch.nn.Module):
            self.register_buffer('channel_offsets', off)
            self.register_buffer('intensity_ratio', rat, persistent=False)
        else:
            self.channel_offsets, self.intensity_ratio = off, rat

    @property
    def n_channels(self):
        raise NotImplementedError

    @property
    def n_extra(self):
        return len(self.extra_params)

    @property
    def n_params(self):
        """Length of theta: 4 + C + len(extra_params)."""
        return 4 + self.n_channels + self.n_extra

    @property
    def param_names(self):
        """Names of the theta columns: x, y, z, I, bg0 .. bg{C-1}, *extra_params."""
        return ['x', 'y', 'z', 'I'] + [f'bg{c}' for c in range(self.n_channels)] + list(self.extra_params)

    def set_channel_offsets(self, offsets, trainable=False):
        """Set the channel offsets [C, 2] (px); trainable=True makes them a torch.nn.Parameter (a nn.Module model) or
        a tensor that requires grad (gradients flow through render / expected; channel 0 is not held fixed here)."""
        v = _as_t(offsets, self.dev).reshape(self.n_channels, 2).clone()
        if isinstance(self, torch.nn.Module):
            if 'channel_offsets' in self._buffers:
                del self._buffers['channel_offsets']
            elif 'channel_offsets' in self._parameters:
                del self._parameters['channel_offsets']
            if trainable:
                self.channel_offsets = torch.nn.Parameter(v)
            else:
                self.register_buffer('channel_offsets', v)
        else:
            self.channel_offsets = v.requires_grad_(bool(trainable))
        return self

    def has_offsets(self):
        """True if any channel offset is nonzero."""
        return bool((self.channel_offsets != 0).any())

    def _shared_position(self, offsets=None):
        """True if all channels see the reference position: no per-spot offsets, channel offsets zero and not trainable
        (a trainable offset, even zero, takes the per-channel path so that it gets a gradient)."""
        return offsets is None and not self.channel_offsets.requires_grad and not self.has_offsets()

    def _offsets(self, N, offsets=None):
        """channel_offsets (+ per-spot offsets) -> [N, C, 2] (or [1, C, 2] without per-spot offsets)."""
        d = self.channel_offsets[None]
        if offsets is not None:
            d = d + _as_t(offsets, self.dev).reshape(N, self.n_channels, 2)
        return d

    def channel_positions(self, pos, offsets=None):
        """Per-channel positions [N, C, 3] = pos [N, 3] + (channel_offsets + offsets [N, C, 2]) in x, y. Without
        offsets (all zero and not trainable, none passed) the positions are pos itself (an expanded view)."""
        N, C = pos.shape[0], self.n_channels
        if self._shared_position(offsets):
            return pos[:, None, :].expand(N, C, 3)
        d = self._offsets(N, offsets).to(pos.dtype)
        return torch.cat([pos[:, None, :2] + d, pos[:, None, 2:3].expand(N, C, 1)], 2)

    # ------------------------------------------------------------------ the model
    def render_channels(self, pos_c, extra=None, grad=False):
        """pos_c [N, C, 3] (per-channel x, y px, z um), extra [N, k] or None -> [N, C, P, P]; grad=True: (value,
        d/dx, d/dy, d/dz, d/d extra...) each [N, C, P, P] (channel c with respect to its own position). Without the
        intensity ratio (render applies it)."""
        raise NotImplementedError

    def _extra(self, N, extra=None, dev=None, **named):
        """[N, k] extra parameters from a tensor or by name (missing names: 0), or None if the model has none."""
        if not self.extra_params:
            return None
        if extra is not None:
            return _as_t(extra, dev or self.dev).reshape(N, self.n_extra)
        cols = []
        for k in self.extra_params:
            v = named.get(k)
            cols.append(torch.zeros(N, device=dev or self.dev) if v is None
                        else _as_t(v, dev or self.dev).reshape(-1).expand(N))
        return torch.stack(cols, 1)

    def _ratio(self, out):
        if bool((self.intensity_ratio == 1).all()):
            return out
        r = self.intensity_ratio[None, :, None, None]
        return out * r if torch.is_tensor(out) else tuple(o * r for o in out)

    def render(self, pos, grad=False, offsets=None, **extra):
        """Normalized channel images [N, C, P, P] at the reference position pos [N, 3] (x, y px, z um), per-spot
        offsets [N, C, 2] (px) or None, and the extra parameters by name (e.g. phi=[N]; default 0), intensity ratio
        included. grad=True: (value, d/dx, d/dy, d/dz, d/d extra...)."""
        pos = _as_t(pos, self.dev)
        ex = self._extra(len(pos), None, pos.device, **extra)
        return self._ratio(self.render_channels(self.channel_positions(pos, offsets), ex, grad))

    def _render_theta(self, theta, offsets=None, grad=False):
        C = self.n_channels
        ex = theta[:, 4 + C:] if self.extra_params else None
        return self._ratio(self.render_channels(self.channel_positions(theta[:, :3], offsets), ex, grad))

    def __call__(self, pos, intensity=None, bg=None, offsets=None, **extra):
        """intensity [N] * render(pos) + bg ([N] or [N, C])."""
        out = self.render(pos, offsets=offsets, **extra)
        if intensity is not None:
            out = out * intensity[:, None, None, None]
        if bg is not None:
            out = out + (bg[:, :, None, None] if bg.dim() == 2 else bg[:, None, None, None])
        return out

    def theta(self, pos, intensity, bg, **extra):
        """theta [N, 4 + C + k] from pos [N, 3], intensity [N], bg [N] or [N, C] and the extra parameters by name."""
        pos = _as_t(pos, self.dev)
        N, C = len(pos), self.n_channels
        bg = _as_t(bg, self.dev)
        bg = bg.reshape(N, 1).expand(N, C) if bg.dim() < 2 else bg
        cols = [pos, _as_t(intensity, self.dev).reshape(N, 1), bg]
        ex = self._extra(N, None, None, **extra)
        if ex is not None:
            cols.append(ex)
        return torch.cat(cols, 1)

    # ------------------------------------------------------------------ statistics
    def _check_theta(self, theta):
        theta = _as_t(theta, self.dev)
        if theta.dim() != 2 or theta.shape[1] != self.n_params:
            raise ValueError(f'theta must be [N, {self.n_params}] = {self.param_names}, not {tuple(theta.shape)}')
        return theta

    def expected(self, theta, offsets=None):
        """Expected image mu [N, C, P, P] = I render(x, y, z; extra) + bg_c, theta [N, 4 + C + k]."""
        theta = self._check_theta(theta)
        C = self.n_channels
        return theta[:, 3, None, None, None] * self._render_theta(theta, offsets) + theta[:, 4:4 + C, None, None]

    def mu_and_jacobian(self, theta, offsets=None):
        """(mu [N, C, P, P], J [N, 4 + C + k, C, P, P]): the expected image and its derivatives with respect to theta
        (x, y, z, I, bg_0 .. bg_{C-1}, extra)."""
        theta = self._check_theta(theta)
        N, C = len(theta), self.n_channels
        g, gx, gy, gz, *gex = self._render_theta(theta, offsets, grad=True)
        I = theta[:, 3, None, None, None]
        mu = I * g + theta[:, 4:4 + C, None, None]
        eye = torch.eye(C, device=g.device, dtype=g.dtype)[None, :, :, None, None].expand(N, C, C, *g.shape[-2:])
        J = torch.cat([torch.stack([I * gx, I * gy, I * gz, g], 1), eye] +
                      ([torch.stack([I * e for e in gex], 1)] if gex else []), 1)
        return mu, J

    def _free(self, fixed=()):
        """Indices of the estimated theta columns: all but `fixed` (and z for 2D models)."""
        fx = set(self._indices(fixed))
        if self.n_dims == 2:
            fx.add(2)
        return [i for i in range(self.n_params) if i not in fx]

    def _indices(self, names):
        """theta column indices of names / indices ('x', 'y', 'z', 'I', 'bg' (all backgrounds), 'bg<c>', extra names,
        or ints)."""
        out = []
        nm = self.param_names
        for v in ([names] if isinstance(names, (str, int)) else list(names)):
            if isinstance(v, str):
                if v == 'bg':
                    out += list(range(4, 4 + self.n_channels))
                elif v in nm:
                    out.append(nm.index(v))
                else:
                    raise ValueError(f'unknown parameter {v!r} (known: {nm} or bg)')
            else:
                out.append(int(v))
        return sorted(set(out))

    @torch.no_grad()
    def fisher(self, theta, var=None, offsets=None, fixed=(), weight=None):
        """Poisson Fisher information [N, p, p] of the estimated parameters (all theta columns but `fixed` and, for
        2D models, z; in theta order) at theta [N, 4 + C + k], all channels jointly. var: sCMOS read-noise variance
        (e^2; scalar or [N, C, P, P]), weight: pixel weights [N, C, P, P]."""
        mu, J = self.mu_and_jacobian(theta, offsets)
        J = J[:, self._free(fixed)].flatten(2)
        w = 1.0 / (mu.clamp_min(1e-6) + (0.0 if var is None else var)).flatten(1)
        if weight is not None:
            w = w * weight.flatten(1)
        return torch.einsum('nap,nbp->nab', J * w[:, None], J)

    @torch.no_grad()
    def covariance(self, theta, var=None, offsets=None, fixed=(), weight=None, ridge=0.0):
        """Inverse Fisher matrix [N, 4 + C + k, 4 + C + k] (float64); rows / columns of parameters not estimated
        (fixed, z of a 2D model) are NaN."""
        F = self.fisher(theta, var, offsets, fixed, weight).double()
        free = self._free(fixed)
        if ridge:
            F = F + ridge * torch.eye(len(free), device=F.device, dtype=F.dtype)
        N, n = F.shape[0], self.n_params
        cov = torch.full((N, n, n), float('nan'), device=F.device, dtype=torch.float64)
        idx = torch.tensor(free, device=F.device)
        cov[:, idx[:, None], idx[None, :]] = torch.linalg.inv(F)
        return cov

    def crlb(self, theta, var=None, offsets=None, fixed=(), weight=None, ridge=0.0):
        """Cramer-Rao lower bound (standard deviation) [N, 4 + C + k] at theta [N, 4 + C + k], all channels jointly;
        NaN for parameters that are not estimated (fixed; z of a 2D model)."""
        cov = self.covariance(theta, var, offsets, fixed, weight, ridge)
        return torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0)).float()

    # ------------------------------------------------------------------ fitting
    def default_extra_starts(self):
        """Starting values of the extra parameters for the multi-start fit: a list of [k] vectors (default zeros)."""
        return [[0.0] * self.n_extra] if self.extra_params else [[]]

    def wrap_extra(self, theta):
        """Map the extra parameters of theta to their canonical range after a fit (in place; e.g. a phase to
        (-pi, pi]); default no change."""
        return theta

    def clamp_theta(self, th, offsets=None, margin=0.05):
        """Clamp theta [N, 4 + C + k] in place: every channel position |x + d_c|, |y + d_c| <= xy_limit - margin, z
        within z_limits, I >= 1, bg >= 1e-3."""
        N, C = th.shape[0], self.n_channels
        xm = self.xy_limit - margin
        d = self._offsets(N, offsets).to(th.dtype).expand(N, C, 2)
        lo, hi = -xm - d.min(1).values, xm - d.max(1).values           # [N, 2]
        hi = torch.maximum(hi, lo)
        th[:, :2] = torch.minimum(torch.maximum(th[:, :2], lo), hi)
        zl = self.z_limits
        if zl is not None and self.n_dims == 3:
            th[:, 2].clamp_(zl[0], zl[1])
        th[:, 3].clamp_min_(1.0)
        th[:, 4:4 + C].clamp_min_(1e-3)
        return th

    @torch.no_grad()
    def initial_theta(self, rois, z=0.0, extra=None, offsets=None, bg_quantile=0.2, min_photons=50.0):
        """Starting values: x, y = minus the mean channel offset (the reference position that puts the channels on
        average at their ROI centres), z, bg_c = a low quantile of ROI c, I = the summed signal above the background
        divided by the summed model image at that start, extra (default zeros)."""
        N, C, P = rois.shape[0], rois.shape[1], rois.shape[-1]
        bg = rois.flatten(2).quantile(bg_quantile, 2)                                  # [N, C]
        xy = -self._offsets(N, offsets).to(rois.dtype).expand(N, C, 2).mean(1)        # [N, 2]
        cols = [xy, torch.full((N, 1), float(z), device=rois.device), torch.ones(N, 1, device=rois.device), bg]
        if self.extra_params:
            ex = torch.zeros(N, self.n_extra, device=rois.device) if extra is None else \
                _as_t(extra, rois.device).reshape(-1, self.n_extra).expand(N, self.n_extra)
            cols.append(ex)
        th = self.clamp_theta(torch.cat(cols, 1), offsets)
        s = self._render_theta(th, offsets).sum((1, 2, 3)).clamp_min(1e-3)
        sig = (rois.sum((2, 3)) - bg * P * P).sum(1)
        th[:, 3] = (sig / s).clamp_min(min_photons)
        return th

    @torch.no_grad()
    def fit_theta(self, rois, theta0=None, var=None, offsets=None, fixed=(), weight=None, iters=30, zstarts=5,
                  extra_starts=None, batch=2000):
        """Maximum-likelihood theta [N, 4 + C + k] of the channel ROIs rois [N, C, P, P] (photons) by Levenberg-
        Marquardt on all channels jointly (torch_psf.fit.lm_poisson). theta0: starting values (then one start);
        without it a 3D model is started at zstarts heights evenly inside z_range (as torch_psf.fit.localize) times
        the extra starts (extra_starts: list of [k] vectors, default default_extra_starts(); for 4Pi four phases), and
        the lowest NLL wins. fixed: parameters held at their start (names or indices: e.g. ('phi',), see _indices);
        z of a 2D model is always fixed. var: read-noise variance (scalar or [N, C, P, P]), offsets [N, C, 2]: per-spot
        channel offsets, weight [N, C, P, P]: pixel weights. Returns (theta, nll [N])."""
        rois = _as_t(rois, self.dev)
        N = len(rois)
        fx = sorted(set(self._indices(fixed)) | ({2} if self.n_dims == 2 else set()))
        if N == 0:
            return torch.zeros(0, self.n_params, device=self.dev), torch.zeros(0, device=self.dev)
        sl = lambda a, s: a if (a is None or not torch.is_tensor(a) or a.dim() == 0 or len(a) != N) else a[s:s + batch]
        if theta0 is None:
            if self.n_dims == 2:
                zl = [0.0]
            else:
                zl = [float(z) for z in torch.linspace(self.z_range[0], self.z_range[1], zstarts + 2)[1:-1]]
            starts = [(z, e) for z in zl for e in (extra_starts if extra_starts is not None
                                                     else self.default_extra_starts())]
        else:
            starts = [None]
        out, onll = [], []
        for s in range(0, N, batch):
            d, v, o, w = rois[s:s + batch], sl(var, s), sl(offsets, s), sl(weight, s)
            best = bnll = None
            for st in starts:
                if st is None:
                    t0 = _as_t(theta0[s:s + batch], self.dev)
                else:
                    t0 = self.initial_theta(d, st[0], st[1] if self.extra_params else None, o)
                th, nll = lm_poisson(lambda t, o=o: self.mu_and_jacobian(t, o), d, t0, var=v, weight=w,
                                     clamp=lambda t, o=o: self.clamp_theta(t, o), fixed=fx, iters=iters)
                if best is None:
                    best, bnll = th, nll
                else:
                    m = nll < bnll
                    best[m], bnll[m] = th[m], nll[m]
            out.append(self.wrap_extra(best)); onll.append(bnll)
        return torch.cat(out), torch.cat(onll)

    def fit(self, rois, camera=None, theta0=None, corners=None, frames=None, var=None, offsets=None, fixed=(),
            iters=30, zstarts=5, extra_starts=None, batch=2000, pixel_um=None):
        """Maximum-likelihood fit of multi-channel ROIs rois [N, C, P, P] (photons) -> Dataset (as PSF.fit).

        All channels are fitted jointly with the theta layout of the module docstring (fit_theta). camera: only a
        scalar read noise is used (per-pixel maps need per-channel corners: pass var [N, C, P, P]). corners [N, 2]:
        (x0, y0) of the reference ROI in the image (positions are then image coordinates of the reference frame, else
        relative to the reference ROI centre). offsets [N, C, 2]: per-spot channel offsets (px). fixed: parameters held
        at their start (e.g. ('phi',)). frames [N].
        Dataset fields: frame, pos [N, D] px (z in px of pixel_um), crlb [N, D], photons, bg (mean over the channels),
        crlb_photons, crlb_bg (of that mean), bg_channels [N, C], crlb_bg_channels [N, C], one field per extra
        parameter and its crlb_<name> (NaN when fixed), theta [N, 4 + C + k], loglik (= -NLL), chi2 (mean of
        (d - mu)^2 / (mu + var) over the pixels of all channels), at_limit."""
        rois = _as_t(rois, self.dev)
        if rois.dim() != 4 or rois.shape[1] != self.n_channels:
            raise ValueError(f'rois must be [N, {self.n_channels}, P, P], not {tuple(rois.shape)}')
        N, C, P = rois.shape[0], self.n_channels, rois.shape[-1]
        if var is None and camera is not None:
            var = camera.variance(rois[:, 0], None)
            if torch.is_tensor(var) and var.dim() > 0:
                raise ValueError('a camera with per-pixel read noise needs var [N, C, P, P] (cut per channel)')
        if torch.is_tensor(var) and var.dim() == 0:
            var = None if float(var) == 0 else var
        th, nll = self.fit_theta(rois, theta0, var, offsets, fixed, None, iters, zstarts, extra_starts, batch)
        if N:
            mu = self.expected(th, offsets=offsets)
            chi2 = ((rois - mu) ** 2 / (mu + (0.0 if var is None else var)).clamp_min(1e-6)).flatten(1).mean(1)
            cov = self.covariance(th, var, offsets, fixed)
        else:
            chi2 = torch.zeros(0, device=self.dev)
            cov = torch.zeros(0, self.n_params, self.n_params, dtype=torch.float64, device=self.dev)
        sd = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0)).float()
        vb = cov[:, 4:4 + C, 4:4 + C].sum((1, 2)) / C ** 2
        xy = th[:, :2].double()
        if corners is not None:
            xy = xy + _as_t(corners, self.dev, torch.float64) + (P - 1) / 2
        px = pixel_um if pixel_um is not None else getattr(self, 'pixel_um', None)
        if self.n_dims == 3 and not px:
            raise ValueError(f'{type(self).__name__} has no pixel size: z is stored in px of the lateral pixel size, so '
                             f'pass pixel_um=... or set psf.pixel_um')
        pos, crp = [xy], [sd[:, :2]]
        if self.n_dims == 3:
            pos.append(th[:, 2:3].double() / px)
            crp.append(sd[:, 2:3] / px)
        xm = self.xy_limit - 0.06
        d = self._offsets(N, offsets).expand(N, C, 2) if N else torch.zeros(0, C, 2, device=self.dev)
        lim = ((th[:, None, :2] + d).abs().amax((1, 2)) > xm) if N else torch.zeros(0, dtype=torch.bool)
        cpu = lambda t: t.detach().cpu()
        f = dict(frame=cpu(torch.as_tensor(frames)) if frames is not None else torch.zeros(N, dtype=torch.int32),
                 pos=cpu(torch.cat(pos, 1)), crlb=cpu(torch.cat(crp, 1)), photons=cpu(th[:, 3]),
                 bg=cpu(th[:, 4:4 + C].mean(1)), crlb_photons=cpu(sd[:, 3]), crlb_bg=cpu(torch.sqrt(vb.clamp_min(0))).float(),
                 bg_channels=cpu(th[:, 4:4 + C]), crlb_bg_channels=cpu(sd[:, 4:4 + C]), theta=cpu(th),
                 loglik=cpu(-nll), chi2=cpu(chi2), at_limit=cpu(lim).bool())
        for i, k in enumerate(self.extra_params):
            f[k], f[f'crlb_{k}'] = cpu(th[:, 4 + C + i]), cpu(sd[:, 4 + C + i])
        meta = dict(method=f'torch_psf {type(self).__name__}.fit', n_dims=self.n_dims, n_channels=C,
                    theta=self.param_names)
        if N:
            meta['chi2_median'] = float(chi2.median())
        if px:
            meta['pixel_nm'] = 1000.0 * float(px)
        return Dataset(f, meta)

    def localize(self, movie, *args, **kwargs):
        # TODO: multi-channel movie localization: detection in the reference channel, the channel mapping
        # (psf.zip `mapping`) to cut the ROI of every channel and its per-spot offsets, then fit().
        raise NotImplementedError(f'{type(self).__name__}.localize: cutting per-channel ROIs from a movie needs the '
                                  f'channel mapping; cut them yourself and call fit(rois [N, C, P, P], offsets=...)')


class ChannelsPSF(MultiChannelPSF):
    """C independent single-channel models seen at once (biplane, multicolour, dual view): channel c is models[c]
    rendered at pos_c[:, c]. All models have one ROI size and the same dimensionality; z_range / z_limits are their
    intersection, xy_limit the smallest, pixel_um that of model 0.

    ChannelsPSF(models, channel_offsets=None [C, 2] px, intensity_ratio=None [C], names=None)."""
    type_name = None             # not a psf.json type (independent channels are a psf.zip: torch_psf.psfzip)

    def __init__(self, models, channel_offsets=None, intensity_ratio=None, names=None):
        self.models = list(models)
        if not self.models:
            raise ValueError('ChannelsPSF needs at least one model')
        if any(isinstance(m, MultiChannelPSF) for m in self.models):
            raise ValueError('ChannelsPSF channels are single-channel models')
        if len({int(m.roisize) for m in self.models}) != 1:
            raise ValueError(f'all channel models need one roisize, not {[m.roisize for m in self.models]}')
        if len({m.n_dims for m in self.models}) != 1:
            raise ValueError('all channel models need the same dimensionality (2D / 3D)')
        self.roisize = int(self.models[0].roisize)
        self.dev = self.models[0].dev
        self.pixel_um = getattr(self.models[0], 'pixel_um', None)
        self.names = [f'ch{c}' for c in range(len(self.models))] if names is None else list(names)
        self._init_channels(len(self.models), channel_offsets, intensity_ratio)

    @property
    def n_channels(self):
        return len(self.models)

    @property
    def z_range(self):
        zr = [m.z_range for m in self.models]
        if zr[0] is None:
            return None
        return (max(z[0] for z in zr), min(z[1] for z in zr))

    @property
    def z_limits(self):
        zl = [m.z_limits for m in self.models]
        if zl[0] is None:
            return None
        return (max(z[0] for z in zl), min(z[1] for z in zl))

    @property
    def xy_limit(self):
        return min(m.xy_limit for m in self.models)

    def render_channels(self, pos_c, extra=None, grad=False):
        outs = [m.render(pos_c[:, c], grad) for c, m in enumerate(self.models)]
        if not grad:
            return torch.stack(outs, 1)
        return tuple(torch.stack([o[k] for o in outs], 1) for k in range(4))

    def to(self, device):
        for m in self.models:
            m.to(device)
        self.dev = self.models[0].dev
        self.channel_offsets = self.channel_offsets.to(self.dev)
        self.intensity_ratio = self.intensity_ratio.to(self.dev)
        return self

    def resized(self, roisize):
        return ChannelsPSF([m.resized(roisize) for m in self.models], self.channel_offsets, self.intensity_ratio,
                           self.names)

    def to_spline(self, roisize=None, z_range=None, **kw):
        """ChannelsPSF of the channel models' to_spline (the same offsets and ratios; the offsets stay a channel
        property, they are not baked into the tables)."""
        return ChannelsPSF([m.to_spline(roisize, z_range, **kw) for m in self.models], self.channel_offsets,
                           self.intensity_ratio, self.names)

    def __repr__(self):
        return (f'ChannelsPSF({self.n_channels} channels: {[type(m).__name__ for m in self.models]}, '
                f'roisize={self.roisize}, offsets={self.channel_offsets.tolist()})')
