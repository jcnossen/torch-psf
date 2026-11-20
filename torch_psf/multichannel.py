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

Mapped channels (torch_psf.mapping.ChannelMap per channel, set_mappings / ChannelsPSF(mappings=)): instead of the
constant offsets, channel c sees the emitter at

    pos_c = from_ref_c(corner_0 + h + (x, y)) - corner_c - h (+ channel_offsets[c] + offsets[:, c]),  h = (P - 1) / 2

with the per-spot ROI corners [N, C, 2] of every channel in that channel's crop coordinates (`corners=`, required in
this mode; torch_psf.mapping.channel_rois cuts such ROIs); x, y of theta are relative to the reference ROI's centre and
the derivatives with respect to them go through the mapping's Jacobian (rotation, scale, flip). A field-dependent
channel model (torch_psf.field.FieldPSF / TiledSplinePSF) is evaluated at its ROI centre corner_c + h (channel
coordinates). Without mappings (default) nothing changes: the formulas above are not used.

Units as in PSF: px laterally, um in z, photons.
"""
import math
import torch
from torch_psf.spline import IndexedSplines, SplinePSF
from torch_psf.psf import PSF
from torch_psf.fit import lm_poisson, poisson_nll
from torch_psf.dataset import Dataset
from torch_psf.vectorial import VectorialPSF


def _as_t(v, dev, dtype=torch.float32):
    return torch.as_tensor(v, dtype=dtype, device=dev)


class MultiChannelPSF(PSF):
    """Abstract multi-channel PSF (module docstring). Subclasses call _init_channels(C, channel_offsets,
    intensity_ratio) in their constructor, set extra_params, and implement render_channels; they inherit render,
    expected, mu_and_jacobian, fisher, crlb, fit_theta and fit."""
    extra_params = ()            # names of the extra emitter parameters (theta columns after the backgrounds)
    accepts_xy = False     # render_channels takes xy [N, C, 2] (field-dependent channel models)

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
        self.__dict__['mappings'] = None             # plain attribute (not an nn.Module child); set_mappings

    # ------------------------------------------------------------------ mapped channels
    def set_mappings(self, mappings):
        """Per-channel torch_psf.mapping.ChannelMap (reference -> channel c; None entries = identity), or None to go back
        to the offset-only geometry. With mappings every position-dependent call needs corners [N, C, 2] (module
        docstring)."""
        if mappings is None:
            self.__dict__['mappings'] = None
            return self
        from torch_psf.mapping import ChannelMap
        ms = list(mappings)
        if len(ms) != self.n_channels:
            raise ValueError(f'{len(ms)} mappings for {self.n_channels} channels')
        ms = [ChannelMap.identity_map(c, device=self.dev) if m is None else m.to(self.dev) for c, m in enumerate(ms)]
        self.__dict__['mappings'] = ms
        return self

    @property
    def mapped(self):
        return getattr(self, 'mappings', None) is not None

    def _corners(self, N, corners):
        if corners is None:
            raise ValueError(f'{type(self).__name__} with channel mappings needs corners [N, C, 2] (the ROI corner of '
                             f'every channel, channel crop px; torch_psf.mapping.channel_rois)')
        cor = _as_t(corners, self.dev)
        if not self.mapped and cor.shape == (N, 2):
            return cor[:, None].expand(N, self.n_channels, 2)
        return cor.reshape(N, self.n_channels, 2)

    def mapped_positions(self, pos, offsets=None, corners=None):
        """(pos_c [N, C, 3], J [N, C, 2, 2] = d (x_c, y_c) / d (x, y), xy [N, C, 2] = the ROI centres in channel
        coordinates) of reference positions pos [N, 3] (relative to the reference ROI centre) with the mappings"""
        N, C = pos.shape[0], self.n_channels
        cor = self._corners(N, corners)
        h = (self.roisize - 1) / 2
        ref = (pos[:, :2].double() + cor[:, 0].double() + h)                       # [N, 2] reference channel px
        xs, Js = [], []
        for c, m in enumerate(self.mappings):
            xs.append(m.from_ref(ref))
            Js.append(m.jacobian(ref.detach()))
        xc = torch.stack(xs, 1) - cor.double() - h                                  # [N, C, 2]
        xc = xc + self.channel_offsets[None].double()
        if offsets is not None:
            xc = xc + _as_t(offsets, self.dev).reshape(N, C, 2).double()
        pos_c = torch.cat([xc.to(pos.dtype), pos[:, None, 2:3].expand(N, C, 1)], 2)
        return pos_c, torch.stack(Js, 1).to(pos.dtype), (cor + h)

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

    def channel_positions(self, pos, offsets=None, corners=None):
        """Per-channel positions [N, C, 3] = pos [N, 3] + (channel_offsets + offsets [N, C, 2]) in x, y. Without
        offsets (all zero and not trainable, none passed) the positions are pos itself (an expanded view). With
        mappings: mapped_positions (needs corners)."""
        N, C = pos.shape[0], self.n_channels
        if self.mapped:
            return self.mapped_positions(pos, offsets, corners)[0]
        if self._shared_position(offsets):
            return pos[:, None, :].expand(N, C, 3)
        d = self._offsets(N, offsets).to(pos.dtype)
        return torch.cat([pos[:, None, :2] + d, pos[:, None, 2:3].expand(N, C, 1)], 2)

    # ------------------------------------------------------------------ the model
    def render_channels(self, pos_c, extra=None, grad=False, dipM=None):
        """pos_c [N, C, 3] (per-channel x, y px, z um), extra [N, k] or None -> [N, C, P, P]; grad=True: (value,
        d/dx, d/dy, d/dz, d/d extra...) each [N, C, P, P] (channel c with respect to its own position). Without the
        intensity ratio (render applies it). dipM is the shared [3,3] or per-emitter [N,3,3] emission tensor."""
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

    def _render(self, pos, grad=False, *, dipM=None, xy=None, phi=None, focus=None, angle=None):
        kw = {k: v for k, v in dict(focus=focus, angle=angle).items() if v is not None}
        if self.mapped:
            corners = None if xy is None else torch.as_tensor(xy, device=self.dev) - (self.roisize - 1) / 2
        else:
            corners = None
            if xy is not None:
                kw['xy'] = xy
        return self.render_geometry(pos, grad, corners=corners, dipM=dipM, phi=phi, **kw)

    def render_geometry(self, pos, grad=False, *, offsets=None, corners=None, dipM=None, phi=None, **kw):
        """Normalized channel images [N, C, P, P] at the reference position pos [N, 3] (x, y px, z um), per-spot
        offsets [N, C, 2] (px) or None, and the extra parameters by name (e.g. phi=[N]; default 0), intensity ratio
        included. grad=True: (value, d/dx, d/dy, d/dz, d/d extra...). corners [N, C, 2]: per-channel ROI corners
        (needed with mappings)."""
        if phi is not None and 'phi' not in self.extra_params:
            raise ValueError(f'{type(self).__name__} does not support phi')
        pos = torch.as_tensor(pos, device=self.dev)
        ex = self._extra(len(pos), None, pos.device, phi=phi)
        if self.mapped:
            return self._ratio(self._render_mapped(pos, ex, offsets, corners, grad, dipM, **kw))
        if self.accepts_xy and corners is not None and any(self.field_dependent):
            kw.setdefault('xy', self._corners(len(pos), corners) + (self.roisize - 1) / 2)
        if dipM is not None:
            kw['dipM'] = dipM
        return self._ratio(self.render_channels(self.channel_positions(pos, offsets), ex, grad, **kw))

    def _render_mapped(self, pos, ex, offsets, corners, grad, dipM=None, **kw):
        """render_channels at the mapped positions; x / y derivatives chained through the mapping Jacobians"""
        pos_c, J, fxy = self.mapped_positions(pos, offsets, corners)
        if self.accepts_xy:
            kw['xy'] = fxy
        if dipM is not None:
            kw['dipM'] = dipM
        out = self.render_channels(pos_c, ex, grad, **kw)
        return self._map_derivatives(out, J) if grad else out

    @staticmethod
    def _map_derivatives(out, J):
        g, gx, gy, gz, *ge = out
        J = J.reshape(*J.shape, *([1] * (gx.ndim - 2)))
        dx = gx * J[:, :, 0, 0] + gy * J[:, :, 1, 0]
        dy = gx * J[:, :, 0, 1] + gy * J[:, :, 1, 1]
        return (g, dx, dy, gz, *ge)

    def _render_theta(self, theta, offsets=None, grad=False, corners=None):
        C = self.n_channels
        ex = theta[:, 4 + C:] if self.extra_params else None
        if self.mapped:
            return self._ratio(self._render_mapped(theta[:, :3], ex, offsets, corners, grad))
        kw = {}
        if self.accepts_xy and corners is not None and any(self.field_dependent):
            kw['xy'] = self._corners(len(theta), corners) + (self.roisize - 1) / 2
        return self._ratio(self.render_channels(self.channel_positions(theta[:, :3], offsets), ex, grad, **kw))

    def __call__(self, pos, intensity=None, bg=None, offsets=None, corners=None, **extra):
        return self.forward(pos, intensity, bg, offsets, corners, **extra)

    def forward(self, pos, intensity=None, bg=None, offsets=None, corners=None, dipM=None, **extra):
        """intensity [N] * render(pos) + bg ([N] or [N, C])."""
        if dipM is not None:
            extra['dipM'] = dipM
        out = self.render_geometry(pos, offsets=offsets, corners=corners, **extra)
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

    def expected(self, theta, offsets=None, corners=None):
        """Expected image mu [N, C, P, P] = I render(x, y, z; extra) + bg_c, theta [N, 4 + C + k]."""
        theta = self._check_theta(theta)
        C = self.n_channels
        return (theta[:, 3, None, None, None] * self._render_theta(theta, offsets, corners=corners)
                + theta[:, 4:4 + C, None, None])

    def mu_and_jacobian(self, theta, offsets=None, corners=None):
        """(mu [N, C, P, P], J [N, 4 + C + k, C, P, P]): the expected image and its derivatives with respect to theta
        (x, y, z, I, bg_0 .. bg_{C-1}, extra)."""
        theta = self._check_theta(theta)
        N, C = len(theta), self.n_channels
        g, gx, gy, gz, *gex = self._render_theta(theta, offsets, grad=True, corners=corners)
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
    def fisher(self, theta, var=None, offsets=None, fixed=(), weight=None, corners=None):
        """Poisson Fisher information [N, p, p] of the estimated parameters (all theta columns but `fixed` and, for
        2D models, z; in theta order) at theta [N, 4 + C + k], all channels jointly. var: sCMOS read-noise variance
        (e^2; scalar or [N, C, P, P]), weight: pixel weights [N, C, P, P]."""
        mu, J = self.mu_and_jacobian(theta, offsets, corners)
        J = J[:, self._free(fixed)].flatten(2)
        w = 1.0 / (mu.clamp_min(1e-6) + (0.0 if var is None else var)).flatten(1)
        if weight is not None:
            w = w * weight.flatten(1)
        return torch.einsum('nap,nbp->nab', J * w[:, None], J)

    @torch.no_grad()
    def covariance(self, theta, var=None, offsets=None, fixed=(), weight=None, ridge=0.0, corners=None):
        """Inverse Fisher matrix [N, 4 + C + k, 4 + C + k] (float64); rows / columns of parameters not estimated
        (fixed, z of a 2D model) are NaN."""
        F = self.fisher(theta, var, offsets, fixed, weight, corners).double()
        free = self._free(fixed)
        if ridge:
            F = F + ridge * torch.eye(len(free), device=F.device, dtype=F.dtype)
        N, n = F.shape[0], self.n_params
        cov = torch.full((N, n, n), float('nan'), device=F.device, dtype=torch.float64)
        idx = torch.tensor(free, device=F.device)
        cov[:, idx[:, None], idx[None, :]] = torch.linalg.inv(F)
        return cov

    def crlb(self, theta, var=None, offsets=None, fixed=(), weight=None, ridge=0.0, corners=None):
        """Cramer-Rao lower bound (standard deviation) [N, 4 + C + k] at theta [N, 4 + C + k], all channels jointly;
        NaN for parameters that are not estimated (fixed; z of a 2D model)."""
        cov = self.covariance(theta, var, offsets, fixed, weight, ridge, corners)
        return torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0)).float()

    # ------------------------------------------------------------------ fitting
    def default_extra_starts(self):
        """Starting values of the extra parameters for the multi-start fit: a list of [k] vectors (default zeros)."""
        return [[0.0] * self.n_extra] if self.extra_params else [[]]

    def wrap_extra(self, theta):
        """Map the extra parameters of theta to their canonical range after a fit (in place; e.g. a phase to
        (-pi, pi]); default no change."""
        return theta

    def clamp_theta(self, th, offsets=None, margin=0.05, corners=None):
        """Clamp theta [N, 4 + C + k] in place: every channel position |x + d_c|, |y + d_c| <= xy_limit - margin, z
        within z_limits, I >= 1, bg >= 1e-3. With mappings: x, y moved (through the inverse mapping Jacobian) until
        every channel's mapped position is inside, one channel after the other."""
        N, C = th.shape[0], self.n_channels
        xm = self.xy_limit - margin
        if self.mapped:
            for c in range(C):
                pc, J, _ = self.mapped_positions(th[:, :3], offsets, corners)
                e = pc[:, c, :2].clamp(-xm, xm) - pc[:, c, :2]
                if bool((e != 0).any()):
                    th[:, :2] += torch.linalg.solve(J[:, c].double(), e.double()[..., None])[..., 0].to(th.dtype)
        else:
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
    def initial_theta(self, rois, z=0.0, extra=None, offsets=None, bg_quantile=0.2, min_photons=50.0, corners=None):
        """Starting values: x, y = minus the mean channel offset (the reference position that puts the channels on
        average at their ROI centres; with mappings the least-squares position of that kind through the mapping
        Jacobians), z, bg_c = a low quantile of ROI c, I = the summed signal above the background divided by the
        summed model image at that start, extra (default zeros)."""
        N, C, P = rois.shape[0], rois.shape[1], rois.shape[-1]
        bg = rois.flatten(2).quantile(bg_quantile, 2)                                  # [N, C]
        if self.mapped:
            pc, J, _ = self.mapped_positions(torch.zeros(N, 3, device=rois.device), offsets, corners)
            Jd = J.double()
            A = torch.einsum('ncki,nckj->nij', Jd, Jd)
            b = torch.einsum('ncki,nck->ni', Jd, pc[..., :2].double())
            xy = -torch.linalg.solve(A, b[..., None])[..., 0].to(rois.dtype)
        else:
            xy = -self._offsets(N, offsets).to(rois.dtype).expand(N, C, 2).mean(1)        # [N, 2]
        cols = [xy, torch.full((N, 1), float(z), device=rois.device), torch.ones(N, 1, device=rois.device), bg]
        if self.extra_params:
            ex = torch.zeros(N, self.n_extra, device=rois.device) if extra is None else\
                _as_t(extra, rois.device).reshape(-1, self.n_extra).expand(N, self.n_extra)
            cols.append(ex)
        th = self.clamp_theta(torch.cat(cols, 1), offsets, corners=corners)
        s = self._render_theta(th, offsets, corners=corners).sum((1, 2, 3)).clamp_min(1e-3)
        sig = (rois.sum((2, 3)) - bg * P * P).sum(1)
        th[:, 3] = (sig / s).clamp_min(min_photons)
        return th

    @torch.no_grad()
    def fit_theta(self, rois, theta0=None, var=None, offsets=None, fixed=(), weight=None, iters=30, zstarts=5,
                  extra_starts=None, batch=2000, corners=None):
        """Maximum-likelihood theta [N, 4 + C + k] of the channel ROIs rois [N, C, P, P] (photons) by Levenberg-
        Marquardt on all channels jointly (torch_psf.fit.lm_poisson). theta0: starting values (then one start);
        without it a 3D model is started at zstarts heights evenly inside z_range (as torch_psf.fit.localize) times
        the extra starts (extra_starts: list of [k] vectors, default default_extra_starts(); for 4Pi four phases), and
        the lowest NLL wins. fixed: parameters held at their start (names or indices: e.g. ('phi',), see _indices);
        z of a 2D model is always fixed. var: read-noise variance (scalar or [N, C, P, P]), offsets [N, C, 2]: per-spot
        channel offsets, weight [N, C, P, P]: pixel weights, corners [N, C, 2]: per-channel ROI corners (mapped
        models). Returns (theta, nll [N])."""
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
            cr = None if corners is None else _as_t(corners, self.dev)[s:s + batch]
            best = bnll = None
            for st in starts:
                if st is None:
                    t0 = _as_t(theta0[s:s + batch], self.dev)
                else:
                    t0 = self.initial_theta(d, st[0], st[1] if self.extra_params else None, o, corners=cr)
                th, nll = lm_poisson(lambda t, o=o, cr=cr: self.mu_and_jacobian(t, o, cr), d, t0, var=v, weight=w,
                                     clamp=lambda t, o=o, cr=cr: self.clamp_theta(t, o, corners=cr), fixed=fx,
                                     iters=iters)
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
        relative to the reference ROI centre); with mappings corners [N, C, 2] (every channel's ROI corner, channel
        crop px; positions are then reference-channel coordinates). offsets [N, C, 2]: per-spot channel offsets (px).
        fixed: parameters held at their start (e.g. ('phi',)). frames [N].
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
        mc = None
        if self.mapped or (corners is not None and self.accepts_xy and any(self.field_dependent)):
            mc = self._corners(N, corners)
        th, nll = self.fit_theta(rois, theta0, var, offsets, fixed, None, iters, zstarts, extra_starts, batch, mc)
        if N:
            mu = self.expected(th, offsets=offsets, corners=mc)
            chi2 = ((rois - mu) ** 2 / (mu + (0.0 if var is None else var)).clamp_min(1e-6)).flatten(1).mean(1)
            cov = self.covariance(th, var, offsets, fixed, corners=mc)
        else:
            chi2 = torch.zeros(0, device=self.dev)
            cov = torch.zeros(0, self.n_params, self.n_params, dtype=torch.float64, device=self.dev)
        sd = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(0)).float()
        vb = cov[:, 4:4 + C, 4:4 + C].sum((1, 2)) / C ** 2
        xy = th[:, :2].double()
        if mc is not None:
            xy = xy + mc[:, 0].double() + (P - 1) / 2
        elif corners is not None:
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
        if N and self.mapped:
            lim = self.mapped_positions(th[:, :3], offsets, mc)[0][..., :2].abs().amax((1, 2)) > xm
        else:
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

    ChannelsPSF(models, channel_offsets=None [C, 2] px, intensity_ratio=None [C], names=None, mappings=None).
    mappings: per channel a torch_psf.mapping.ChannelMap (reference -> channel; None = identity): the mapped geometry
    of the module docstring (corners [N, C, 2] needed). A channel model may be field dependent
    (torch_psf.field.FieldPSF / TiledSplinePSF): it is evaluated at its ROI centre in channel coordinates (needs the
    corners; without them, at its reference point)."""
    type_name = None             # not a psf.json type (independent channels are a psf.zip: torch_psf.psfzip)
    accepts_xy = True

    def __init__(self, models, channel_offsets=None, intensity_ratio=None, names=None, mappings=None):
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
        from torch_psf.field import TiledSplinePSF
        self._tiled_channels = (any(isinstance(m, TiledSplinePSF) for m in self.models) and
                                all(isinstance(m, (SplinePSF, TiledSplinePSF)) for m in self.models))
        self._tile_models = {}
        self._tile_bank = None
        self.indexed = IndexedSplines(self.models, 'channel') if IndexedSplines.compatible(self.models) else None
        if self.indexed is not None:
            self.models = self.indexed.members
        self._init_channels(len(self.models), channel_offsets, intensity_ratio)
        if mappings is not None:
            self.set_mappings(mappings)

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

    @property
    def field_dependent(self):
        """per channel: True for a field-dependent model (FieldPSF / TiledSplinePSF)"""
        return [getattr(m, 'uses_field_xy', False) for m in self.models]

    def _dipole_basis(self, pos, grad=False, *, xy=None, focus=None, angle=None):
        raw = pos.ndim == 3
        corners = xy - (self.roisize - 1) / 2 if self.mapped and xy is not None else None
        jacobian = None
        if self.mapped and not raw:
            pc, jacobian, xy = self.mapped_positions(pos, corners=corners)
        else:
            pc = pos if raw else self.channel_positions(pos)
        images = self.render_channels(pc, grad=grad, xy=xy, focus=focus, angle=angle, basis=True)
        if grad and jacobian is not None:
            images = self._map_derivatives(images, jacobian)
        if raw:
            return images
        ratio = self.intensity_ratio[None, :, None, None, None]
        return tuple(v * ratio for v in images) if grad else images * ratio

    def _render_tile_channels(self, pos_c, xy, grad, dipM, angle, basis):
        """Gather all channel, tile and response indices on a common grid in one evaluation."""
        n, c = pos_c.shape[:2]
        ids, weights = [], []
        for k, model in enumerate(self.models):
            if xy is not None and hasattr(model, 'tile_weights'):
                indices, w = model.tile_weights(xy[:, k] if xy.ndim == 3 else xy)
            else:
                indices = torch.full((n, 1), -1, device=self.dev, dtype=torch.long)
                w = pos_c.new_ones(n, 1)
            # Include the reference spline for an empty batch so output shape and dtype are preserved.
            rows = torch.empty_like(indices)
            for tile in indices.unique().tolist() or [-1]:
                key = (k, tile)
                if key not in self._tile_models:
                    if isinstance(model, SplinePSF):
                        table = model
                    else:
                        table = model.reference_spline() if tile == -1 else model.spline(tile)
                    self._tile_models[key] = (len(self._tile_models), table)
                rows[indices == tile] = self._tile_models[key][0]
            ids.append(rows)
            weights.append(w)
        if self._tile_bank is None or len(self._tile_bank.members) != len(self._tile_models):
            models = [m for _, m in self._tile_models.values()]
            if not IndexedSplines.compatible(models):
                self._tiled_channels = False
                self._tile_models.clear()
                self._tile_bank = None
                return None
            self._tile_bank = IndexedSplines(models, 'tile')
        # Channels may mix nearest and bilinear. Pad with valid zero-weight entries.
        width = max(w.shape[1] for w in weights)
        member = torch.stack([torch.cat((r, r[:, :1].expand(-1, width-r.shape[1])), 1) for r in ids], 1)
        weight = torch.stack([torch.nn.functional.pad(w, (0, width-w.shape[1])) for w in weights], 1)
        pos = pos_c[:, :, None].expand(-1, -1, width, -1).reshape(n, c*width, 3)
        images = self._tile_bank.evaluate(pos, member.reshape(n, c*width), grad, dipM=dipM, angle=angle, basis=basis)
        def blend(v):
            v = v.unflatten(1, (c, width))
            return (v * weight.reshape(*weight.shape, *([1] * (v.ndim-3)))).sum(2)
        return tuple(blend(v) for v in images) if grad else blend(images)

    def _shared_optical(self, pos_c, grad, kwargs, method, focus, xy):
        """Resolve stage/field coordinates before batching channels backed by the same optics."""
        from torch_psf.composition import ZMode
        from torch_psf.field import FieldPSF, TiledSplinePSF
        from torch_psf.psf import render_with_grad

        stages = []
        for model in self.models:
            axial = model if isinstance(model, ZMode) else None
            model = axial.base if axial is not None else model
            field = model if isinstance(model, FieldPSF) and not isinstance(model, TiledSplinePSF) else None
            base = field.base if field is not None else model
            if not isinstance(base, VectorialPSF) or (stages and base is not stages[0][0]):
                return None
            stages.append((base, axial, field))

        n, c = pos_c.shape[:2]
        base = stages[0][0]
        # A stored per-emitter response already has an N-sized batch axis. Its native
        # renderer handles that axis independently in each channel.
        if any(value is not None and torch.as_tensor(value).ndim == dim for value, dim in
               ((base.blur, 2), (base.dipM, 3), (base.dipw, 2))):
            return None
        if 'angle' in kwargs and torch.as_tensor(kwargs['angle']).ndim:
            kwargs = dict(kwargs, angle=torch.as_tensor(kwargs['angle']).repeat_interleave(c))
        if 'dipM' in kwargs and torch.as_tensor(kwargs['dipM']).ndim == 3:
            kwargs = dict(kwargs, dipM=torch.as_tensor(kwargs['dipM']).repeat_interleave(c, 0))
        if xy is not None and any(field is not None for _, _, field in stages):
            offsets = [field.offsets(xy[:, k] if xy.ndim == 3 else xy) if field is not None
                       else base.zcoef.new_zeros(n, len(base.zidx)) for k, (_, _, field) in enumerate(stages)]
            base = base.with_field_offset(torch.stack(offsets, 1).flatten(0, 1))
        render = getattr(base, method)
        if not any(axial is not None for _, axial, _ in stages):
            f = None if focus is None else focus.reshape(-1)
            out = render(pos_c.reshape(-1, 3), grad, focus=f, **kwargs)
            return tuple(v.unflatten(0, (n, c)) for v in out) if grad else out.unflatten(0, (n, c))

        def sample(p):
            heights, foci = [], []
            for k, (_, axial, _) in enumerate(stages):
                z = p[:, k, 2]
                f = None if focus is None else focus[:, k]
                if axial is not None:
                    z, f = axial.coordinates(z, f)
                elif f is None:
                    f = torch.full_like(z, base.focus_um)
                heights.append(z)
                foci.append(f)
            pc = torch.cat((p[..., :2], torch.stack(heights, 1)[..., None]), -1)
            return render(pc.reshape(-1, 3), focus=torch.stack(foci, 1).reshape(-1), **kwargs).unflatten(0, (n, c))
        return render_with_grad(sample, pos_c, grad)

    def render_channels(self, pos_c, extra=None, grad=False, xy=None, dipM=None, focus=None, angle=None, basis=False):
        n, c = pos_c.shape[:2]
        if xy is not None:
            xy = torch.as_tensor(xy, device=pos_c.device)
        if self.indexed is not None:
            if focus is not None or xy is not None and not self.mapped:
                raise ValueError('spline channels accept neither focus nor field xy')
            ids = torch.arange(c, device=pos_c.device)
            return self.indexed.evaluate(pos_c, ids, grad, dipM=dipM, angle=angle, basis=basis)
        if self._tiled_channels:
            if focus is not None:
                raise ValueError('focus cannot change after tabulation')
            images = self._render_tile_channels(pos_c, xy, grad, dipM, angle, basis)
            if images is not None:
                return images
        fd = self.field_dependent
        if xy is not None and not any(fd) and not self.mapped:
            raise ValueError('xy requires field-dependent channels or channel mappings')
        kwargs = {k: v for k, v in dict(dipM=dipM, angle=angle).items() if v is not None}
        if focus is not None:
            f = torch.as_tensor(focus, device=pos_c.device)
            focus = f[:, None] if f.ndim == 1 else f
            focus = focus.expand(n, c)
        method = 'dipole_basis' if basis else 'render'
        out = self._shared_optical(pos_c, grad, kwargs, method, focus, xy)
        if out is not None:
            return out
        outs = []
        for k, m in enumerate(self.models):
            kw = dict(kwargs)
            if focus is not None:
                kw['focus'] = focus[:, k]
            if xy is not None and fd[k]:
                kw['xy'] = xy[:, k] if xy.ndim == 3 else xy
            outs.append(getattr(m, method)(pos_c[:, k], grad, **kw))
        return tuple(torch.stack(v, 1) for v in zip(*outs)) if grad else torch.stack(outs, 1)

    def to(self, device=None, dtype=None):
        self._tile_models.clear()
        self._tile_bank = None
        for m in self.models:
            m.to(device=device, dtype=dtype)
        if self.indexed is not None:
            self.indexed = IndexedSplines(self.models, 'channel')
            self.models = self.indexed.members
        self.dev = self.models[0].dev
        self.channel_offsets = self.channel_offsets.to(device=self.dev, dtype=dtype)
        self.intensity_ratio = self.intensity_ratio.to(device=self.dev, dtype=dtype)
        if self.mapped:
            self.set_mappings(self.mappings)
        return self

    def map_models(self, fn):
        """Apply an optical composition once per shared model, preserving acquisition geometry."""
        mapped = {}
        for m in self.models:
            if id(m) not in mapped:
                mapped[id(m)] = fn(m)
        view = ChannelsPSF([mapped[id(m)] for m in self.models], names=self.names, mappings=self.mappings)
        view.channel_offsets, view.intensity_ratio = self.channel_offsets, self.intensity_ratio
        return view

    def at_wavelength(self, wavelength):
        return self.map_models(lambda m: m.at_wavelength(wavelength))

    def format_version(self, source=None):
        return max(m.format_version() for m in self.models)

    def _maps_copy(self):
        return None if not self.mapped else [m.copy() for m in self.mappings]

    def resized(self, roisize):
        return self.map_models(lambda m: m.resized(roisize))

    def to_spline(self, roisize=None, z_range=None, **grid):
        """ChannelsPSF of the channel models' to_spline (the same offsets, ratios and mappings; the offsets stay a
        channel property, they are not baked into the splines). A FieldPSF channel becomes the spline of its reference
        point (use to_tiled per channel for the field dependence)."""
        return ChannelsPSF([m.to_spline(roisize, z_range, **grid) for m in self.models], self.channel_offsets,
                           self.intensity_ratio, self.names, self._maps_copy())

    def to_tiled(self, tile=128, roisize=None, z_range=None, **grid):
        """ChannelsPSF with every FieldPSF channel replaced by its TiledSplinePSF (FieldPSF.to_tiled; other channels
        unchanged), the same offsets, ratios and mappings: localization at spline speed with a tile per channel."""
        ms = [m.to_tiled(tile=tile, roisize=roisize, z_range=z_range, **grid) if hasattr(m, 'to_tiled') else m
              for m in self.models]
        return ChannelsPSF(ms, self.channel_offsets, self.intensity_ratio, self.names, self._maps_copy())

    def __repr__(self):
        return (f'ChannelsPSF({self.n_channels} channels: {[type(m).__name__ for m in self.models]}, '
                f'roisize={self.roisize}, offsets={self.channel_offsets.tolist()}'
                + (f', mappings={[m.extra_repr() for m in self.mappings]}' if self.mapped else '') + ')')
