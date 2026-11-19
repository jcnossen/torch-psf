"""
CubicSplinePSF: a 3D PSF stored as Catmull-Rom spline knots on an oversampled grid, evaluated with one strided, grouped
conv2d (the convolutional 3D spline of torch-spline-3d, J. Cnossen).

The pixel-integrated PSF is shift invariant: patch[i, j] = g(o_j - x, o_i - y, z), with o the pixel offset from the
patch centre and (x, y) the emitter offset in px. g is stored on a uniform grid with spacing 1/S px in u and v and hz um
in z: knots [nz, nv, nu]. Output pixels are S knots apart and share the fractional knot offset, so a P x P patch is the
4x4x4 Catmull-Rom kernel (value, d/dz, d/dy, d/dx) applied with stride S to the block of knots around the emitter: one
grouped conv2d for a whole batch of emitters.

The knot block of each emitter is read with one linear index and index_select (its backward is a plain scatter-add,
which makes the gradient with respect to the knots about 5x faster than advanced indexing).

The knots are either
  - free parameters: the patch is linear in the knots, d patch / d knots by autograd (render(..., knots=K)), or
  - computed from a vectorial model (knots_from_vectorial / from_vectorial); a gradient with respect to the knots is
    passed on to the vectorial parameters with one vector-Jacobian product (vjp_vectorial), or
  - a measured z-stack used as knots directly (from_zstack, S = 1).

Offset limit: the stored support allows emitter offsets |x|, |y| <= xmax px from the patch centre; positions beyond are
clamped (the fitter clamps at xmax - 0.05). Use xmax >= 3 for patches centred on detections of defocused spots: with a
smaller limit a flexible PSF model absorbs the clamped spots as a z-dependent position error.

Units: x, y in px relative to the patch centre (pixel centres at integers), z in um.
"""
import math
import torch
from torch_psf.optics import resolve_device
from torch_psf.psf import PSF
from torch_psf._dipole import DipoleSpline


def spline_kind(kind, pixel_integration):
    if kind is None:
        return 'cubic' if pixel_integration == 'box' else 'integrated'
    if kind not in ('cubic', 'integrated'):
        raise ValueError("kind must be 'cubic' or 'integrated'")
    return kind


def geometry_matrix(device, tau: float = 0.5, dtype=torch.float32):
    """Catmull-Rom basis: per-axis weights w = [1, t, t^2, t^3] @ G."""
    return torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [-tau, 0.0, tau, 0.0],
        [2 * tau, tau - 3, 3 - 2 * tau, -tau],
        [-tau, 2 - tau, tau - 2, tau]
    ], device=device, dtype=dtype)


def emitter_height_pos_fn(focus_um):
    """pos_fn for knots_from_vectorial: knot z = emitter height above the coverslip, focus fixed (SMLM geometry)."""
    def pos_fn(z):
        return z, torch.full_like(z, float(focus_um))
    return pos_fn


def stage_scan_pos_fn(focus0_um, scale, emitter_z=0.0):
    """pos_fn for a stage / focus scan (bead stacks): an emitter at a fixed height emitter_z, focus = focus0 - z / scale.
    scale = n_med / n_imm (paraxial focal shift) gives a knot z that is a nominal emitter-height equivalent."""
    def pos_fn(z):
        return torch.full_like(z, float(emitter_z)), focus0_um - z / scale
    return pos_fn


class CubicSplinePSF(DipoleSpline, PSF):
    type_name = 'spline'
    """Spline PSF for roisize x roisize patches (odd or even roisize) and emitter heights z_range = (lo, hi) um.

    oversample: knots per px laterally; dz_um: knot spacing in z (um); zpad: extra knot planes below and above
    z_range; xy_limit: offset limit (px; >= 3 recommended for patches centred on detections); device: default cuda if
    available. knots: [nz, nv, nu] (zeros until set).
    Knot-grid attributes (short names, used by the evaluation code): S = oversample, hz = dz_um, xmax = xy_limit,
    zlo / zhi = z_range, c = stored half width (px), Pl = 2c + 1 (size of the vectorial images that build the knots),
    z0 = z of knot plane 0, zs = z of all planes, nz / nv / nu = knot array shape."""

    def __init__(self, roisize, z_range, oversample=2, dz_um=0.04, zpad=2, xy_limit=3.0, device=None,
                 dtype=torch.float32, knots=None, knot_u0_px=None, knot_z0_um=None, pixel_um=None):
        """roisize may be odd or even (pixel offsets from the ROI centre are j - (roisize - 1) / 2). To wrap a stored
        knot array of another geometry, pass knots [nz, n, n] (n a multiple of oversample) and optionally its origin:
        knot_u0_px = lateral offset (px) of knot index 0 from the PSF centre (default -(c S + S - 1) / S with
        c = (n / S - 1) / 2, the layout this class builds), knot_z0_um = z of knot plane 0 (default
        z_range[0] - zpad dz_um). The offset limit is reduced to what the stored knots support. pixel_um: lateral pixel
        size (um), needed by fit / localize (z is stored in px)."""
        self.pixel_um = None if pixel_um is None else float(pixel_um)
        zlo, zhi = float(z_range[0]), float(z_range[1])
        S, hz, xmax = int(oversample), float(dz_um), float(xy_limit)
        self.roisize, self.S, self.hz = roisize, S, hz
        self.h = (roisize - 1) / 2
        self.dev, self.dtype = resolve_device(device), dtype
        self.zlo, self.zhi, self.zpad = zlo, zhi, zpad
        if knots is None:
            self.c = int(math.ceil(self.h + xmax)) + 2            # half width (px) of the stored PSF
            self.nz = int(round((zhi - zlo) / hz)) + 1 + 2 * zpad
        else:
            nz_, nv_, nu_ = knots.shape
            assert nv_ == nu_ and nu_ % S == 0 and (nu_ // S) % 2 == 1, 'knots [nz, n, n], n / oversample odd'
            self.c = (nu_ // S - 1) // 2
            self.nz = nz_
        self.Pl = 2 * self.c + 1
        self.z0 = zlo - zpad * hz if knot_z0_um is None else float(knot_z0_um)
        self.nv = self.nu = self.Pl * S
        self.u0 = -(self.c * S + S - 1) / S if knot_u0_px is None else float(knot_u0_px)   # u (px) of knot 0
        self.oS0 = float((-self.h - self.u0) * S)                # knot coordinate of patch pixel 0 for x = 0
        # largest offset for which the 4-knot window of every ROI pixel stays inside the array
        lim = min((self.oS0 - 1) / S, (self.nu - 2 - self.oS0 - (roisize - 1) * S) / S - 1e-4)
        self.xmax = min(xmax, max(lim, 0.0))
        self.Lw = (roisize - 1) * S + 4
        r = lambda n: torch.arange(n, device=self.dev)
        self.pat = (r(4)[:, None, None] * self.nv * self.nu + r(self.Lw)[None, :, None] * self.nu
                    + r(self.Lw)[None, None, :]).reshape(-1)
        self.G = geometry_matrix(self.dev, dtype=dtype)
        self.e = torch.arange(4, device=self.dev)
        if knots is None:
            self.knots = torch.zeros(self.nz, self.nv, self.nu, device=self.dev, dtype=dtype)
        else:
            self.knots = torch.as_tensor(knots).to(self.dev, dtype)

    # ------------------------------------------------------------------ geometry
    @property
    def zs(self):
        """z (um) of the knot planes [nz] (float64)."""
        return self.z0 + self.hz * torch.arange(self.nz, device=self.dev, dtype=torch.float64)

    @property
    def z_range(self):
        return (self.zlo, self.zhi)

    @property
    def z_limits(self):
        """z interval (um) on which the spline is defined (the fitter clamps z to it)."""
        return (self.z0 + self.hz, self.z0 + (self.nz - 2) * self.hz)

    @property
    def xy_limit(self):
        return self.xmax

    def __repr__(self):
        return (f'CubicSplinePSF(roisize={self.roisize}, z_range=({self.zlo}, {self.zhi}), oversample={self.S}, '
                f'dz_um={self.hz}, zpad={self.zpad}, xy_limit={self.xmax}, knots {(self.nz, self.nv, self.nu)})')

    # ------------------------------------------------------------------ evaluation
    def _axis(self, t, deriv):
        p = t[..., None] ** self.e
        if deriv:
            p = self.e * t[..., None] ** (self.e - 1).clamp_min(0)
        return p @ self.G

    def render(self, pos, grad=False, knots=None, dipM=None):
        """pos [N, 3] = (x px, y px, z um) -> normalized patches [N, P, P]; with grad also d/dx, d/dy (per px) and
        d/dz (per um). Outside the supported range x, y and z are clamped (to xy_limit and the knot z range) and the
        derivatives are those of the spline at the clamped position (not zero): a fit that starts outside still sees a
        slope towards the inside. Differentiable with respect to the knots (pass knots=K to use other
        knots than self.knots, e.g. a tensor that requires grad)."""
        if dipM is not None:
            return self._dipole_render(pos, grad, dipM, knots)
        K = self.knots if knots is None else knots
        N, P, S, Lw = pos.shape[0], self.roisize, self.S, self.Lw
        pos = pos.to(self.dtype)
        tz = ((pos[:, 2] - self.z0) / self.hz)
        tz = tz.clamp(1, self.nz - 2 - 1e-4)
        t = torch.stack([tz, self.oS0 - pos[:, 1].clamp(-self.xmax, self.xmax) * S,
                         self.oS0 - pos[:, 0].clamp(-self.xmax, self.xmax) * S], 1)
        ix = torch.floor(t)
        fr = t - ix
        i0 = ix.long() - 1
        base = i0[:, 0] * self.nv * self.nu + i0[:, 1] * self.nu + i0[:, 2]
        # index_select, not K.reshape(-1)[idx]: its backward is a plain scatter-add (5x faster knot gradient)
        sub = K.reshape(-1).index_select(0, (base[:, None] + self.pat[None]).reshape(-1)).reshape(1, N * 4, Lw, Lw)
        wa = self._axis(fr, False)                             # [N, 3, 4]
        w = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
        if not grad:
            return torch.nn.functional.conv2d(sub, w, groups=N, stride=S).view(N, P, P)
        da = self._axis(fr, True)
        wz = da[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
        wy = wa[:, 0, :, None, None] * da[:, 1, None, :, None] * wa[:, 2, None, None, :]
        wx = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * da[:, 2, None, None, :]
        W = torch.stack([w, wx, wy, wz], 1).view(-1, 4, 4, 4)
        r = torch.nn.functional.conv2d(sub, W, groups=N, stride=S).view(N, 4, P, P)
        return r[:, 0], -S * r[:, 1], -S * r[:, 2], r[:, 3] / self.hz

    def render_batched(self, pos, nb=4000):
        """render() (value only) in batches of nb emitters."""
        return torch.cat([self.render(pos[s:s + nb]) for s in range(0, len(pos), nb)])

    def to(self, device=None, dtype=None):
        """Move the knots and the index tensors to another device (and the knots to another float dtype), in place;
        returns self (torch.nn.Module semantics)."""
        dev = self.dev if device is None else torch.device(device)
        if dtype is not None:
            self.dtype = dtype
            self.G = self.G.to(dtype)
        if self._knots is not None:
            self.knots = self._knots.to(dev, self.dtype)
        self.pat, self.G, self.e = self.pat.to(dev), self.G.to(dev), self.e.to(dev)
        self.dev = self.pat.device
        self._set_dipole_knots(self.dipole_knots, self.component_weights)
        return self

    def resized(self, roisize, device=None):
        """The same spline (the same knots, shared, not copied) for roisize x roisize ROIs. The knots tabulate the PSF
        independently of the ROI size; a larger ROI than the knots cover reduces the offset limit (xy_limit), a
        smaller one keeps it."""
        s = CubicSplinePSF(roisize, self.z_range, oversample=self.S, dz_um=self.hz, zpad=self.zpad, xy_limit=self.xmax,
                           device=self.dev if device is None else device, dtype=self.dtype,
                           knots=self._knots if self._knots is not None else self.dipole_knots[0],
                           knot_u0_px=self.u0, knot_z0_um=self.z0, pixel_um=self.pixel_um)
        s.provenance = self.provenance
        s._set_dipole_knots(self.dipole_knots, self.component_weights)
        return s

    def to_spline(self, roisize=None, z_range=None, **kw):
        """A cubic spline is already a spline: returns self, or resized(roisize) for another ROI size (other
        arguments are not supported)."""
        if (z_range is not None and tuple(z_range) != self.z_range) or kw:
            raise ValueError('re-tabulating a spline on another grid is not supported; build it from the source model')
        return self if roisize in (None, self.roisize) else self.resized(roisize)

    # ------------------------------------------------------------------ knots from a vectorial model
    def _vec_positions(self):
        a = torch.arange(self.S, device=self.dev, dtype=torch.float64) / self.S
        B, A = torch.meshgrid(a, a, indexing='ij')
        return A.reshape(-1).float(), B.reshape(-1).float()

    def build_planes(self, vectorial, params, zi0, zi1, pos_fn):
        """Knot planes zi0..zi1-1 from the vectorial model (roisize Pl) with parameters `params` (a dict for
        torch.func.functional_call; autograd-able). pos_fn(z_knot [n]) -> (z_emitter [n], focus [n]) maps the knot z
        coordinate to the arguments of the vectorial model."""
        S, Pl = self.S, self.Pl
        A, B = self._vec_positions()
        z = self.zs[zi0:zi1].float()
        ze, f = pos_fn(z)
        n = zi1 - zi0
        pos = torch.stack([A[None].expand(n, -1), B[None].expand(n, -1), ze[:, None].expand(-1, S * S)], -1)
        imgs = torch.func.functional_call(vectorial, params, (pos.reshape(-1, 3), f[:, None].expand(-1, S * S).reshape(-1)))
        imgs = imgs.reshape(n, S, S, Pl, Pl)                   # [z, b, a, i, j]
        # g at u = (j - c) - a / S -> knot index j S + (S - 1 - a) (contiguous); the same for v with (i, b)
        return imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(n, Pl * S, Pl * S)

    def knots_from_model(self, model, chunk=8):
        """Knot array [nz, nv, nu] from any PSF model with roisize Pl: model.render at the S x S sub-pixel shifts of
        every knot plane (z = the knot z), interleaved as in build_planes."""
        assert model.roisize == self.Pl, f'the model needs roisize {self.Pl}'
        S, Pl = self.S, self.Pl
        A, B = self._vec_positions()
        out = []
        with torch.no_grad():
            for z0 in range(0, self.nz, chunk):
                z = self.zs[z0:min(z0 + chunk, self.nz)].float()
                n = len(z)
                pos = torch.stack([A[None].expand(n, -1), B[None].expand(n, -1), z[:, None].expand(-1, S * S)], -1)
                imgs = model.render(pos.reshape(-1, 3)).reshape(n, S, S, Pl, Pl)
                out.append(imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(n, Pl * S, Pl * S))
        return torch.cat(out).to(self.dtype)

    def knots_from_vectorial(self, vectorial, pos_fn=None, params=None, chunk=4):
        """Knot array [nz, nv, nu] from a VectorialPSF with roisize Pl (see VectorialPSF.resized). pos_fn: knot z ->
        (emitter z, focus) (um); default: knot z = emitter height at vectorial.focus_um."""
        vec = vectorial
        assert vec.roisize == self.Pl, f'the vectorial model needs roisize {self.Pl} (vectorial.resized({self.Pl}))'
        pos_fn = emitter_height_pos_fn(vec.focus_um) if pos_fn is None else pos_fn
        params = dict(vec.named_parameters()) if params is None else params
        with torch.no_grad():
            return torch.cat([self.build_planes(vec, params, z, min(z + chunk, self.nz), pos_fn)
                              for z in range(0, self.nz, chunk)]).to(self.dtype)

    def vjp_vectorial(self, vectorial, pos_fn, grad_knots, names=('zcoef',), chunk=4):
        """(d knots / d parameters)^T grad_knots for the named parameters / buffers of the vectorial model (roisize
        Pl): one build plus one backward, chunked over z. Returns {name: gradient}."""
        vec = vectorial
        pos_fn = emitter_height_pos_fn(vec.focus_um) if pos_fn is None else pos_fn
        base = {k: v.detach() for k, v in list(vec.named_parameters()) + list(vec.named_buffers())}   # buffers too:
        out = {k: torch.zeros_like(base[k]) for k in names}                # free pupil maps are buffers
        for z in range(0, self.nz, chunk):
            z1 = min(z + chunk, self.nz)
            prm = dict(base)
            for k in names:
                prm[k] = base[k].clone().requires_grad_(True)
            st = self.build_planes(vec, prm, z, z1, pos_fn)
            gr = torch.autograd.grad(st, [prm[k] for k in names], grad_knots[z:z1].float())
            for k, g in zip(names, gr):
                out[k] += g
        return out


    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_zstack(cls, zstack, z0_um, dz_um, roisize, pixel_um=None, device=None, normalize=False):
        """Spline with oversample 1 whose knots are a measured (or simulated) z-stack [nz, H, W] of an emitter centred
        on the middle pixel (H = W odd), plane k at z = z0_um + k dz_um. The offset limit follows from the stack size:
        xy_limit = (W - 1) / 2 - 2 - (roisize - 1) / 2. normalize: scale each plane to sum 1. pixel_um: lateral pixel
        size (um) of the stack (required for fit / localize)."""
        z0, hz = z0_um, dz_um
        zs = torch.as_tensor(zstack, dtype=torch.float32)
        if device is not None:
            zs = zs.to(device)
        nz, H, W = zs.shape
        assert H == W and W % 2 == 1, 'the z-stack must be square with an odd size'
        h = (roisize - 1) // 2
        xmax = (W - 1) / 2 - 2 - h
        assert xmax > 0, f'the z-stack ({W} px) is too small for roisize {roisize}'
        if normalize:
            zs = zs / zs.sum((1, 2), keepdim=True)
        return cls(roisize, (z0, z0 + (nz - 1) * hz), oversample=1, dz_um=hz, zpad=0, xy_limit=xmax, device=zs.device,
                   knots=zs.contiguous(), pixel_um=pixel_um)

    def _file_dict(self, source=None):
        d, arrays = {}, {}
        if source is not None:
            d, arrays = source._file_dict()
        d['spline'] = self.to_dict()
        if self.component_weights is None:
            arrays['knots'] = self.knots.detach().cpu().numpy()
        else:
            d['spline']['default_weights'] = self.component_weights.tolist()
        if self.dipole_knots is not None:
            arrays['dipole_knots'] = self.dipole_knots.detach().cpu().numpy()
        return d, arrays

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        g = d['spline']
        px = g.get('pixel_um') or (d['optics'].get('pixel_um') if 'optics' in d else None)
        model = cls(g['roisize'], tuple(g['z_range']), oversample=g['oversample'], dz_um=g['dz_um'], zpad=g['zpad'],
                   xy_limit=g['xy_limit'], device=device,
                   knots=arrays['dipole_knots'][0] if 'default_weights' in g else arrays['knots'],
                   knot_u0_px=g.get('u0_px'), knot_z0_um=g.get('z0_um'), pixel_um=px)
        if g.get('dipole_components'):
            model._set_dipole_knots(arrays['dipole_knots'], g.get('default_weights'))
        return model

    @staticmethod
    def load_source(path, device=None):
        """The vectorial model a saved spline was built from (save(..., source=...)), or None."""
        d, arrays, _ = PSF.read_file(path)
        if 'zernike_waves' not in d:
            return None
        return PSF._registry['vectorial']._from_file(d, arrays, roisize=d.get('roisize', 15), device=device)

    def to_dict(self):
        """Grid description (without the knots) for the PSF file."""
        return dict(roisize=self.roisize, z_range=[self.zlo, self.zhi], oversample=self.S, dz_um=self.hz,
                    zpad=self.zpad, xy_limit=self.xmax, c=self.c, z0_um=self.z0, nz=self.nz, u0_px=self.u0,
                    pixel_um=self.pixel_um, **({'dipole_components': True} if self.dipole_knots is not None else {}))
