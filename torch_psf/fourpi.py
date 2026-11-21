"""Coherent two-objective (4Pi) PSF with C polarization / phase-split interference channels, and its spline.

FourPiPSF: the vectorial model, built on VectorialPSF.pupil_fields() of two arms (top, bottom objective).
FourPiSplinePSF: its multi-channel spline, with the global 4Pi phase as an exact analytic dependence.

Geometry (both arms focused at z = 0):
  top arm    : the VectorialPSF field of a dipole mu at (x, y, z), pupil aberration W_top (top.zcoef).
  bottom arm : the same kind of objective viewing from -z. In the bottom frame (rotation by pi about the x axis) the
               emitter is at (x, -y, -z) with dipole (mu_x, -mu_y, -mu_z); its image is mirrored back (y -> -y,
               E_y -> -E_y) by the detection optics so that it overlays the top image (mirror='x': rotation about the
               y axis, mirrored x -> -x, E_x -> -E_x). Own aberration W_bot (bot.zcoef), offset d_bot = (dx px, dy px,
               dz um) of the bottom-arm position, amplitude ratio a.
  channel c  : polarization pol_c (0 = E_x, 1 = E_y), phase phi_c, transmission tau_c (softmax: sums to 1), coherence
               gamma (global, <= 1), Gaussian image blur (isotropic, blur_px2 px^2):
                 I_c = 2 tau_c [ |E_t|^2 + a^2 |E_b|^2 + 2 a gamma Re(E_t . conj(E_b) e^{i (phi_c + Phi)}) ]_pol_c / (1 + a^2)
               summed over the dipole components with the emission-dipole tensor M (isotropic = I / 3); Phi = the
               global 4Pi phase of the emitter (forward(..., phi=Phi), render(..., phi=Phi)).
  The lateral position difference of the arms (d_bot) makes the interference pattern, and with it the centroid of
  each channel, depend on Phi.
  channel_offsets [C, 2] (px; MultiChannelPSF, default zeros): a fixed image offset per channel (channels imaged on
               different camera regions): channel c shows the emitter at (x + dx_c, y + dy_c, z), i.e. the same image
               shift of both arms (a phase ramp on both pupil fields).
Multi-channel interface (torch_psf.multichannel.MultiChannelPSF): theta = [x, y, z, I, bg_0 .. bg_{C-1}, phi], phi =
the global 4Pi phase Phi; fit / fisher / crlb on all channels jointly.
Normalization: with tau_c = 1 / C, balanced polarizations and the interference averaged out, the sum over the channels
integrates to ~1 for an in-focus, aberration-free emitter (the top arm's pupil energy is the reference).

Axial coordinate z (um):
  depth_um None (default; index-matched arms, the sample medium n_med): the emitter displacement from the common
               focus; arm phase k_med cos(theta_med) z (top) and -(z + dz) (bottom).
  depth_um (d_top, d_bot) (asymmetric arms, each its own NA / immersion index, the sample medium n_med): d_arm = the
               emitter distance from that arm's coverslip at z = 0 (sets the supercritical / evanescent part); the
               signed z of an arm (+z top, -z bottom) is interpreted by zmode:
                 'emitter': the emitter moves in the medium: phase k_med cos(theta_med) max(d_arm + z, 0)   (movies)
                 'stage'  : the sample moves through the focus: phase k_med cos(theta_med) max(d_arm, 0)
                            - k_imm cos(theta_imm) z   (bead z-stacks, z = stage displacement)

Parameters (trainable, as in the fitted states): top.zcoef, bot.zcoef (waves rms, Noll zernike_indices), phi [C],
tau_logit [C], a_log, gamma_logit, d_bot [3], logblur, depth [2] (asymmetric only). Natural values: tau, a, gamma,
blur_px2. Buffer: channel_offsets [C, 2] (set_channel_offsets(..., trainable=True) makes it a parameter).
state() / from_state(): the parameter dict {meta, zt, zb, phi, tau, a, gamma, d_bot, blur_px2, (depth),
(channel_offsets: only when nonzero; absent = zeros)}.

Units: px for lateral positions, um for z and lengths, radians for phases.
"""
import copy
import math
import torch
from torch_psf.composition import _AxialCoordinate
from torch_psf.optics import resolve_device, Optics, pixel_area, pixel_json, pixel_size
from torch_psf.psf import PSF, pixel_version, render_with_grad
from torch_psf import extensions as _ext
from torch_psf.multichannel import MultiChannelPSF
from torch_psf._dipole import M_to_m6, contract_fields
from torch_psf.integrate import PixelIntegrator, pixel_integration_mode, spectral_cutoff, spectral_min_oversample
from torch_psf.vectorial import VectorialPSF, dipole_tensor, _list
from torch_psf.spline import SplinePSF

DEFAULT_ZIDX = tuple(range(5, 22))


def _pair(v):
    return (float(v[0]), float(v[1])) if isinstance(v, (tuple, list)) else (float(v), float(v))


def _phase(pos, phi):
    if phi is None:
        return pos.new_zeros(len(pos))
    return torch.as_tensor(phi, device=pos.device, dtype=torch.float32).reshape(-1).expand(len(pos))


def _arm_state(v):
    """the per-objective terms of a 4Pi arm (VectorialPSF) beyond its Zernikes: NA, apodization, supercritical-angle
    terms, diattenuation, free pupil maps (only when not neutral)"""
    d = dict(NA=float(v.NA.detach()), apod=_list(v.apod), saf=_list(v.saf), diattenuation=None if v.diat is None else float(v.diat))
    if bool((v.pupil_phase != 0).any()):
        d['pupil_phase'] = _list(v.pupil_phase)
    if bool((v.pupil_amp != 1).any()):
        d['pupil_amp'] = _list(v.pupil_amp)
    return d


def _arm_load(v, d):
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=v.dev)
    with torch.no_grad():
        if d.get('NA') is not None:
            v.NA.fill_(float(d['NA']))
        if d.get('apod') is not None:
            v.apod.copy_(t(d['apod']))
        if d.get('saf') is not None:
            v.saf.copy_(t(d['saf']))
        if d.get('pupil_phase') is not None:
            v.pupil_phase.copy_(t(d['pupil_phase']).reshape(v.pupil_phase.shape))
        if d.get('pupil_amp') is not None:
            v.pupil_amp.copy_(t(d['pupil_amp']).reshape(v.pupil_amp.shape))
    v.diat = None if d.get('diattenuation') is None else float(d['diattenuation'])


class _CoupledChannels(MultiChannelPSF):
    """Channel models with a common emitter phase and independent per-channel backgrounds."""
    extra_params = ('phi',)

    def weights(self, pos, *, phi=None, **options):
        phase = _phase(pos, phi)
        return torch.stack((torch.ones_like(phase), phase.cos(), -phase.sin()), -1)[:, None].expand(-1, self.n_channels, -1)

    def basis(self, pos, grad=False, *, dipM=None, xy=None, phi=None):
        """Incoherent, real and imaginary interference images for each channel."""
        def sample(p):
            a, b, c = (self.render(p, dipM=dipM, xy=xy, phi=phase) for phase in (0., math.pi, math.pi/2))
            inc = (a + b) * .5
            return torch.stack((inc, (a-b)*.5, inc-c), 2)
        return render_with_grad(sample, pos, grad)


class FourPiPSF(torch.nn.Module, _CoupledChannels):
    """Coherent 4Pi PSF on a roisize x roisize patch with C channels -> images [N, C, P, P] (module docstring).

    Constructor: roisize (px), pixel_um, wavelength_um, NA, n_imm (a number, or (top, bottom) per arm), n_med (sample
    medium; default n_imm of the top arm), zernike_indices (Noll, both arms), npupil, oversample, mirror ('y' | 'x'),
    pol (polarization per channel, 0 = E_x, 1 = E_y; its length is the channel count C), depth_um (None or
    (top, bottom) um: asymmetric-arm geometry), z_range (um; default range of to_spline),
    device, pixel_integration ('exact' by default, or 'midpoint' / 'point'; a file without it is 'midpoint'). Initial values: phi [C] (default
    -2 pi c / C mod 2 pi), tau [C] (default 1 / C), a (1), gamma
    (sigmoid(2)), d_bot (0, 0, 0), blur_px2 (0.1), channel_offsets [C, 2] px (zeros).
    Optional plain attribute dipM: emission-dipole tensor (3 x 3 or [N, 3, 3]) used when forward gets dipM=None.
    A MultiChannelPSF with extra_params ('phi',); tau plays the role of the channel intensity ratio (intensity_ratio
    stays ones and is not stored)."""
    type_name = 'fourpi'
    dtype = property(lambda self: self.phi.dtype)
    extra_params = ('phi',)

    def __init__(self, roisize, pixel_um=0.10833, wavelength_um=0.68, NA=1.35, n_imm=1.406, n_med=None,
                 zernike_indices=DEFAULT_ZIDX, npupil=48, oversample=2, mirror='y', pol=(0, 1, 0, 1), depth_um=None,
                 z_range=(-0.6, 0.6), device=None, phi=None, tau=None, a=1.0, gamma=None,
                 d_bot=(0.0, 0.0, 0.0), blur_px2=0.1, channel_offsets=None, pixel_integration='exact'):
        super().__init__()
        if mirror not in ('x', 'y'):
            raise ValueError("mirror must be 'x' or 'y'")
        self.roisize, self.pixel_um, self.os, self.mirror = int(roisize), pixel_size(pixel_um), int(oversample), mirror
        self.axial = _AxialCoordinate('emitter')
        self.dev = resolve_device(device)
        self.z_range = None if z_range is None else (float(z_range[0]), float(z_range[1]))
        NA_t, NA_b = _pair(NA)
        ni_t, ni_b = _pair(n_imm)
        n_med = ni_t if n_med is None else float(n_med)
        self.zidx = list(zernike_indices)
        mk = lambda na, ni: VectorialPSF(self.roisize, pixel_um=pixel_um, NA=na, n_imm=ni, n_med=n_med,
                                         wavelength_um=wavelength_um, zernike_indices=tuple(self.zidx), npupil=npupil,
                                         oversample=oversample, device=self.dev)
        self.top, self.bot = mk(NA_t, ni_t), mk(NA_b, ni_b)
        self.pixel_integration = pixel_integration
        self.register_buffer('pol', torch.tensor(list(pol), device=self.dev))
        self.detection = None         # optional detection model (channel_field_weights); None = pure Ex / Ey channels
        C = len(pol)
        if phi is None:
            phi = [(-2 * math.pi * c / C) % (2 * math.pi) for c in range(C)]
        self.phi = torch.nn.Parameter(torch.tensor([float(v) for v in phi], device=self.dev))
        tl = torch.zeros(C, device=self.dev) if tau is None else torch.log(torch.as_tensor(tau, dtype=torch.float32)).to(self.dev)
        self.tau_logit = torch.nn.Parameter(tl)
        self.a_log = torch.nn.Parameter(torch.tensor(math.log(a), device=self.dev))
        gl = 2.0 if gamma is None else math.log(gamma / max(1 - gamma, 1e-6))
        self.gamma_logit = torch.nn.Parameter(torch.tensor(float(gl), device=self.dev))
        self.d_bot = torch.nn.Parameter(torch.tensor([float(v) for v in d_bot], device=self.dev))
        self.logblur = torch.nn.Parameter(torch.tensor(math.log(blur_px2), device=self.dev))
        self.depth = None if depth_um is None else torch.nn.Parameter(
            torch.tensor([float(depth_um[0]), float(depth_um[1])], device=self.dev))
        self.dipM = None
        self._init_channels(C, channel_offsets)

    # ------------------------------------------------------------------ natural parameters
    @property
    def n_channels(self):
        return len(self.pol)

    @property
    def asym(self):
        """True for the asymmetric-arm geometry (depth_um set)."""
        return self.depth is not None

    @property
    def tau(self):
        return torch.softmax(self.tau_logit, 0)

    @property
    def a(self):
        return torch.exp(self.a_log)

    @property
    def gamma(self):
        return torch.sigmoid(self.gamma_logit)

    @property
    def blur_px2(self):
        return torch.exp(self.logblur)

    @property
    def n_med(self):
        return float(self.top.n_med)

    # ------------------------------------------------------------------ physics
    @property
    def pixel_integration(self):
        return self._pixint

    @pixel_integration.setter
    def pixel_integration(self, mode):
        mode = pixel_integration_mode(mode)
        if mode == 'gauss_legendre':
            raise ValueError("FourPiPSF has image-plane blur, incompatible with 'gauss_legendre'")
        self._pixint = mode   # 'exact' raises its sampling grid as needed (PixelIntegrator)

    def band_limit(self):
        return max(self.top.band_limit(), self.bot.band_limit())

    def _sampling_model(self, size, density):
        if self.roisize == size and not density:
            return self
        model = self.resized(size)
        if density:
            model.os = max(model.os, spectral_min_oversample(model.band_limit()))
            model.pixel_integration = 'point'
        return model

    def _integrator(self):
        bv = self.blur_px2
        return PixelIntegrator(self, torch.stack([bv, bv, torch.zeros_like(bv)]), coherent=True)

    def _with_zmode(self, mode, height, scale, focus, z_range):
        if height != 0 or scale != 1 or focus is not None:
            raise ValueError('4Pi stage scans use signed arm displacement and calibrated depth_um')
        view = copy.copy(self)
        view.axial = _AxialCoordinate(mode)
        if z_range is not None:
            view.z_range = tuple(z_range)
        return view

    def at_wavelength(self, wavelength):
        view = copy.copy(self)
        view._modules = self._modules.copy()
        view.top, view.bot = self.top.at_wavelength(wavelength), self.bot.at_wavelength(wavelength)
        return view

    def _pupil_at_depth(self, arm, pos, pixels):
        height, stage = None, None
        if self.depth is not None:
            depth = self.depth[0] if arm is self.top else self.depth[1]
            # The bead-stage coordinate of a coherent arm has the opposite sign to emitter defocus.
            if self.axial.mode == 'stage':
                height, stage = self.axial.map(-pos[:, 2], 0., depth)
            else:
                height, _ = self.axial.map(pos[:, 2], 0., depth)
        return arm._pupil(pos, 0., (pixels.u, pixels.uy), height=height, stage_um=stage)

    def arm_fields(self, arm, pos, sgn_mu, pixels=None):
        """Fields [N, 3, 2, n, n] of one arm in its own frame, and its pupil energy."""
        pupil = self._pupil_at_depth(arm, pos, self._integrator() if pixels is None else pixels)
        E = pupil.image_fields().unflatten(1, (3, 2))
        E = E * sgn_mu.to(E.real.dtype)[None, :, None, None, None]
        return E * pupil.amplitude_scale, pupil.energy

    def fields(self, pos, pixels=None):
        """(E_top, E_bottom, top pupil energy): fields [B, 3, 2, ns, ns] of both arms in the top (camera) frame."""
        dev = pos.device
        pixels = self._integrator() if pixels is None else pixels
        Et, energy = self.arm_fields(self.top, pos, torch.ones(3, device=dev), pixels)
        pb = pos + self.d_bot[None]
        if self.mirror == 'y':
            pb = torch.stack([pb[:, 0], -pb[:, 1], -pb[:, 2]], 1)
            Eb, _ = self.arm_fields(self.bot, pb, torch.tensor([1.0, -1.0, -1.0], device=dev), pixels)
            Eb = torch.flip(Eb, dims=[-2])
            Eb = torch.stack([Eb[:, :, 0], -Eb[:, :, 1]], 2)
        else:
            pb = torch.stack([-pb[:, 0], pb[:, 1], -pb[:, 2]], 1)
            Eb, _ = self.arm_fields(self.bot, pb, torch.tensor([-1.0, 1.0, -1.0], device=dev), pixels)
            Eb = torch.flip(Eb, dims=[-1])
            Eb = torch.stack([-Eb[:, :, 0], Eb[:, :, 1]], 2)
        return Et, Eb, energy

    def forward(self, pos, dipM=None, phi=None, intensity=None, bg=None, offsets=None):
        """pos [B, 3] (x, y px relative to the ROI centre, z um), dipM: emission-dipole tensor [3, 3] or
        [B, 3, 3] (default self.dipM, else isotropic), phi: global 4Pi phase Phi [B] (or a number), intensity [B], bg [B] or [B, C],
        offsets [B, C, 2]: per-spot channel offsets (px; added to channel_offsets)
        -> [B, C, P, P] expected photons per emitted photon (times intensity, plus bg, when given). Channel c is
        rendered at (x, y) + channel_offsets[c] (+ offsets[:, c]); without offsets all channels share one evaluation."""
        I = self._channel_images(pos, dipM, phi, offsets)
        if intensity is not None:
            I = I * intensity[:, None, None, None]
        if bg is not None:
            I = I + (bg[:, :, None, None] if bg.dim() == 2 else bg[:, None, None, None])
        return I

    def _dipole_basis(self, pos, grad=False, *, phi=None, xy=None):
        if xy is not None and not self.mapped:
            raise ValueError('xy requires channel mappings')
        corners = None if xy is None else xy - (self.roisize - 1) / 2
        if grad:
            return render_with_grad(lambda p, phi: self._dipole_basis(p, phi=phi, xy=xy), pos, True, phi=_phase(pos, phi))

        phi = _phase(pos, phi)
        if pos.ndim == 3:
            return self._forward_channels(pos, phi=phi, basis=True)
        if self.mapped:
            pc = self.channel_positions(pos, corners=corners)
            images = self._forward_channels(pc, phi=phi, basis=True)
        else:
            images = self._channel_images(pos, phi=phi, basis=True)
        return images * self.intensity_ratio[None, :, None, None, None]

    def _channel_images(self, pos, dipM=None, phi=None, offsets=None, basis=False):
        if self._shared_position(offsets):
            return self._forward_ref(pos, dipM, phi, basis)
        return self._forward_channels(self.channel_positions(pos, offsets), dipM, phi, basis)

    def _dipole(self, dipM, B, dev, dtype):
        Mt = dipole_tensor(dipM if dipM is not None else self.dipM, None, dev)
        if Mt is None:
            Mt = torch.eye(3, device=dev) / 3
        if Mt.dim() == 2:
            Mt = Mt[None].expand(B, 3, 3)
        return ((Mt + Mt.transpose(-1, -2)) * .5).to(dtype)

    def _channel_pixels(self, pairs, pixels, energy, dipM, phi, basis):
        """Coherently combine each channel's polarized arm fields, then integrate its intensity."""
        a, g = self.a, self.gamma
        out = []
        for c, (et, eb) in enumerate(pairs):
            ph = self.phi[c] + (phi if phi is not None else 0.)
            phase = torch.exp(1j * ph.to(torch.float32)).reshape(-1, 1, 1, 1)
            top = contract_fields(et, dipM, basis=basis)
            bottom = contract_fields(eb, dipM, basis=basis)
            cross = contract_fields(et, dipM, y=eb * phase, basis=basis)
            out.append(top + a * a * bottom + 2 * a * g * cross)
        intensity = torch.stack(out, 1) / (1 + a * a)
        image = pixels(intensity) * pixel_area(self.pixel_um) / energy
        return image * self.tau.reshape(1, -1, *([1] * (image.ndim - 2))) * 2

    def _forward_ref(self, pos, dipM=None, phi=None, basis=False):
        """All channels at one position per emitter."""
        pixels = self._integrator()
        Et, Eb, energy = self.fields(pos, pixels)
        Mc = None if basis else self._dipole(dipM, len(pos), pos.device, Et.real.dtype)
        w = self.channel_field_weights()
        if w is None:
            pairs = ((Et[:, :, int(p)], Eb[:, :, int(p)]) for p in self.pol)
        else:                       # linear detection of the camera-frame fields (channel_field_weights)
            pairs = ((torch.einsum('j,bdjuv->bduv', w[0, c], Et), torch.einsum('j,bdjuv->bduv', w[1, c], Eb))
                     for c in range(len(self.pol)))
        return self._channel_pixels(pairs, pixels, energy, Mc, phi, basis)

    def _forward_channels(self, pos_c, dipM=None, phi=None, basis=False):
        """Reuse each pupil across channels. Apply the channel shift and select its
        polarization before transforming the fields into the image plane."""
        B, C, dev = pos_c.shape[0], pos_c.shape[1], pos_c.device
        pos = pos_c[:, 0]
        pu = self.pixel_um
        sh = (pos_c[:, :, :2] - pos[:, None, :2]) * (pos.new_tensor(pu) if isinstance(pu, tuple) else pu)  # [B, C, 2] um, camera frame
        pixels = self._integrator()
        top = self._pupil_at_depth(self.top, pos, pixels)
        pb = pos + self.d_bot[None]
        if self.mirror == 'y':      # bottom frame: (x, -y, -z), dipole (1, -1, -1); image flipped in y, E_y negated
            pb = torch.stack([pb[:, 0], -pb[:, 1], -pb[:, 2]], 1)
            sgn_b, sb, flip_dim, neg_pol = torch.tensor([1.0, -1.0, -1.0], device=dev), (1.0, -1.0), -2, 1
        else:                       # (-x, y, -z), dipole (-1, 1, -1); flipped in x, E_x negated
            pb = torch.stack([-pb[:, 0], pb[:, 1], -pb[:, 2]], 1)
            sgn_b, sb, flip_dim, neg_pol = torch.tensor([-1.0, 1.0, -1.0], device=dev), (-1.0, 1.0), -1, 0
        bot = self._pupil_at_depth(self.bot, pb, pixels)
        Pt, Pb = (arm.fields.unflatten(1, (3, 2)) for arm in (top, bot))
        Mc = None if basis else self._dipole(dipM, B, dev, Pt.real.dtype)
        w = self.channel_field_weights()
        if w is not None:           # bottom weights in its own frame: the camera-frame E_neg_pol is -flip(E)
            w = w.clone()
            w[1, :, neg_pol] = -w[1, :, neg_pol]

        def channel_fields():
            for c in range(C):
                p = int(self.pol[c])
                sx, sy = sh[:, c, 0, None, None], sh[:, c, 1, None, None]
                rt = torch.exp(-1j * (top.kx * sx + top.ky * sy))[:, None]
                rb = torch.exp(-1j * (bot.kx * (sb[0] * sx) + bot.ky * (sb[1] * sy)))[:, None]
                if w is None:
                    et = top.image_fields(Pt[:, :, p] * rt) * top.amplitude_scale
                    eb = bot.image_fields(Pb[:, :, p] * rb) * bot.amplitude_scale
                    sign = -1.0 if p == neg_pol else 1.0
                else:
                    et = top.image_fields(torch.einsum('j,bdjuv->bduv', w[0, c], Pt) * rt) * top.amplitude_scale
                    eb = bot.image_fields(torch.einsum('j,bdjuv->bduv', w[1, c], Pb) * rb) * bot.amplitude_scale
                    sign = 1.0
                eb = eb * sgn_b.to(eb.real.dtype)[None, :, None, None]
                yield et, torch.flip(eb, dims=[flip_dim]) * sign

        return self._channel_pixels(channel_fields(), pixels, top.energy, Mc, phi, basis)

    # ------------------------------------------------------------------ PSF interface (multi-channel)
    @property
    def xy_limit(self):
        return self.roisize / 2 - 1.0

    def render_channels(self, pos_c, extra=None, grad=False, dipM=None):
        """MultiChannelPSF.render_channels: extra [N, 1] = the 4Pi phase."""
        Phi = _phase(pos_c, None if extra is None else extra[:, 0])
        return self._render_c(pos_c, Phi, grad, dipM)

    def _render_c(self, pos_c, Phi, grad, dipM):
        if not self.channel_offsets.requires_grad and bool((pos_c[:, 1:] == pos_c[:, :1]).all()):
            # one position for all channels: one evaluation
            pos = pos_c[:, 0]
            fn = lambda p, phi: self._forward_ref(p, dipM=dipM, phi=phi)
        else:
            pos = pos_c
            fn = lambda p, phi: self._forward_channels(p, dipM=dipM, phi=phi)
        return render_with_grad(fn, pos, grad, phi=Phi)

    def default_extra_starts(self):
        """Four phase starts for the multi-start fit: 0, pi / 2, pi, 3 pi / 2."""
        return [[0.5 * math.pi * k] for k in range(4)]

    def wrap_extra(self, theta):
        """The phase column of theta to (-pi, pi] (in place)."""
        i = 4 + self.n_channels
        theta[:, i] = math.pi - torch.remainder(math.pi - theta[:, i], 2 * math.pi)
        return theta

    to = PSF.to

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        if torch.is_tensor(self.dipM):
            self.dipM = fn(self.dipM)
        self.dev = self.phi.device
        return self

    def channel_field_weights(self):
        """Complex detection weights [2 (top, bottom arm), C, 2 (E_x, E_y)]: channel c images the field
        w[k, c] . (E_x, E_y) of arm k (camera frame), or None (default) for the pure E_x / E_y channel of pol_c.
        From self.detection (a torch.nn.Module with weights(pol) -> [2, C, 2] complex) when one is set. A detection
        model is stored in files only through a model extension: its extension_entry() (torch_psf.extensions)."""
        if self.detection is None:
            return None
        return self.detection.weights(self.pol).to(self.pol.device)     # also a detection module on another device

    def file_extensions(self):
        """Extension entries of this model (torch_psf.extensions): the detection model's, if it has one."""
        if self.detection is None:
            return []
        fn = getattr(self.detection, 'extension_entry', None)
        if fn is None:
            raise ValueError(f'detection model {type(self.detection).__name__} has no extension_entry(): it cannot be '
                             'stored in a state or file (torch_psf.extensions)')
        return [fn()]

    def resized(self, roisize):
        """A copy for roisize x roisize patches (same parameters)."""
        new = copy.deepcopy(self)
        new.roisize = int(roisize)
        new.top, new.bot = self.top.resized(roisize), self.bot.resized(roisize)
        return new

    def __repr__(self):
        return (f'FourPiPSF(roisize={self.roisize}, pixel_um={self.pixel_um}, NA=({float(self.top.NA):.3f}, '
                f'{float(self.bot.NA):.3f}), n_imm=({self.top.n_imm}, {self.bot.n_imm}), n_med={self.n_med:.3f}, '
                f'channels={self.n_channels}, mirror={self.mirror!r}, asym={self.asym}, zmode={self.axial.mode!r})')

    # ------------------------------------------------------------------ parameter dict
    def state(self):
        """Parameter dict {meta, zt, zb, phi, tau, a, gamma, d_bot, blur_px2, (depth), raw, arms, (channel_offsets)}
        (the fitted-state format). meta: R, px, NA, n, lam, npupil, os, mirror, zidx, pol, NA_t, n_imm_t, NA_b,
        n_imm_b, n_med, zmode, and asym=True for the asymmetric-arm geometry. channel_offsets [C, 2] (px) only when
        nonzero (from_state: absent = zeros), so states of models without offsets are unchanged."""
        t, b = self.top, self.bot
        meta = dict(R=self.roisize, px=pixel_json(self.pixel_um), NA=float(t.NA), n=t.n_imm, lam=t.wavelength_um, npupil=t.npupil,
                    os=self.os, mirror=self.mirror, zidx=list(self.zidx), pol=[int(v) for v in self.pol],
                    NA_t=float(t.NA), n_imm_t=t.n_imm, NA_b=float(b.NA), n_imm_b=b.n_imm, n_med=self.n_med,
                    zmode=self.axial.mode)
        if self.pixel_integration != 'midpoint':
            meta['pixel_integration'] = self.pixel_integration
        if self.asym:
            meta['asym'] = True
        s = dict(meta=meta, zt=_list(t.zcoef), zb=_list(b.zcoef), phi=_list(self.phi), tau=_list(self.tau),
                 a=float(self.a.detach()), gamma=float(self.gamma.detach()), d_bot=_list(self.d_bot),
                 blur_px2=float(self.blur_px2.detach()))
        if self.asym:
            s['depth'] = _list(self.depth)
        s['raw'] = {k: _list(v) if v.dim() else float(v.detach()) for k, v in self.named_parameters()
                    if not k.startswith(('top.', 'bot.', 'detection.')) and k != 'channel_offsets'}
        ex = self.file_extensions()
        if ex:
            s['extensions'] = ex
        s['arms'] = {'top': _arm_state(t), 'bottom': _arm_state(b)}
        if self.has_offsets():
            s['channel_offsets'] = _list(self.channel_offsets)
        if self.dipM is not None:
            s['psf_dipole_M'] = _list(self.dipM)
        return s

    @classmethod
    def from_state(cls, s, device=None, roisize=None, z_range=(-0.6, 0.6)):
        """FourPiPSF from a parameter dict (state()). Symmetric states (no meta.asym) have one NA and index n for both
        arms and the medium (unless NA_t / n_imm_t / NA_b / n_imm_b / n_med are given); asymmetric states
        (meta.asym) per-arm NA_t, n_imm_t, NA_b, n_imm_b, the medium n_med and depth. roisize / zmode override the
        stored ones; channel_offsets: absent = zeros."""
        m = s['meta']
        asym = bool(m.get('asym'))
        NA = (m.get('NA_t', m['NA']), m.get('NA_b', m['NA']))
        ni = (m.get('n_imm_t', m['n']), m.get('n_imm_b', m['n']))
        p = cls(m['R'] if roisize is None else roisize, pixel_um=m['px'], wavelength_um=m['lam'], NA=NA, n_imm=ni,
                n_med=m.get('n_med', m['n']), zernike_indices=tuple(m['zidx']), npupil=m['npupil'], oversample=m['os'],
                mirror=m['mirror'], pol=tuple(m['pol']), depth_um=tuple(s['depth']) if asym else None,
                z_range=z_range, device=device,
                pixel_integration=pixel_integration_mode(m.get('pixel_integration'), from_file=True))
        from torch_psf.composition import ZMode
        p = ZMode(p, m.get('zmode', 'emitter'))
        t = lambda v: torch.tensor(v, device=p.dev)
        if s.get('psf_dipole_M') is not None:
            p.dipM = t(s['psf_dipole_M'])
        with torch.no_grad():
            p.top.zcoef.copy_(t(s['zt'])); p.bot.zcoef.copy_(t(s['zb']))
            p.phi.copy_(t(s['phi'])); p.tau_logit.copy_(torch.log(torch.tensor(s['tau'])).to(p.dev))
            p.a_log.fill_(math.log(s['a'])); p.gamma_logit.fill_(math.log(s['gamma'] / max(1 - s['gamma'], 1e-6)))
            p.d_bot.copy_(t(s['d_bot'])); p.logblur.fill_(math.log(s['blur_px2']))
            params = dict(p.named_parameters())
            for k, v in (s.get('raw') or {}).items():      # the exact parameters (state() of this class)
                if k in params:
                    params[k].copy_(torch.as_tensor(v, dtype=params[k].dtype).reshape(params[k].shape))
            arms = s.get('arms') or {}
            for name, arm in (('top', p.top), ('bottom', p.bot)):
                if arms.get(name):
                    _arm_load(arm, arms[name])
            if s.get('channel_offsets') is not None:
                p.channel_offsets.copy_(torch.as_tensor(s['channel_offsets'], dtype=torch.float32).reshape(-1, 2))
        return _ext.apply(p, s.get('extensions'), 'FourPiPSF state')

    # ------------------------------------------------------------------ files
    @staticmethod
    def integration_version(mode):
        return (0, 2) if mode == 'midpoint' else (0, 4)

    def format_version(self, source=None):
        version = pixel_version(self.integration_version(self.pixel_integration), self.pixel_um)
        return max(version, _ext.EXTENSIONS_VERSION) if self.file_extensions() else version

    def _file_dict(self, source=None):
        o = Optics(pixel_um=self.pixel_um, NA=float(self.top.NA), wavelength_um=self.top.wavelength_um,
                   n_imm=self.top.n_imm, n_med=self.n_med)
        return dict(optics=o.to_dict(), roisize=self.roisize, z_range_um=None if self.z_range is None else list(self.z_range),
                    psf_dipole_M=_list(self.dipM), fourpi=self.state()), {}

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        zr = d.get('z_range_um')
        p = cls.from_state(d['fourpi'], device=device, roisize=roisize or d.get('roisize'),
                           z_range=tuple(zr) if zr else None)
        if d.get('psf_dipole_M') is not None:
            p.dipM = torch.tensor(d['psf_dipole_M'], device=p.dev, dtype=torch.float32)
        return p


class FourPiSplinePSF(_CoupledChannels):
    """MultiChannelPSF view of one spline bank indexed by phase, channel and optional dipole component."""
    type_name = 'fourpi_spline'
    dtype = property(lambda self: self.spline.dtype)
    extra_params = ('phi',)

    def __init__(self, roisize, z_range, knots, oversample=4, dz_um=0.01, zpad=3, xy_limit=None, tau=None, device=None,
                 pixel_um=None, source_state=None, channel_offsets=None, integrated_grid=None, default_dipM=None,
                 kernel=None, backend=None):
        self.roisize = self.P = int(roisize)
        self.S, self.hz, self.zpad = int(oversample), float(dz_um), int(zpad)
        self.zmin, self.zmax = float(z_range[0]), float(z_range[1])
        self.dev = resolve_device(device)
        self.pixel_um = pixel_size(pixel_um, floats=True)
        self.source_state = source_state
        xl = self.P / 2 + 1 if xy_limit is None else float(xy_limit)
        K = torch.as_tensor(knots)
        self.dipole_components = K.ndim == 6
        self.default_dipM = _list(default_dipM)
        weights = None
        if self.dipole_components:
            M = torch.eye(3) / 3 if default_dipM is None else torch.as_tensor(default_dipM)
            if M.shape != (3, 3) or K.shape[2] != 6:
                raise ValueError('expected six dipole splines per channel and a [3,3] default tensor')
            weights = M_to_m6(M.to(device=K.device, dtype=torch.float32))

        density = integrated_grid is not None
        g = integrated_grid or {}
        if 'basis' in g or 'kernel' in g:
            grid_kernel, grid_integrated = SplinePSF.grid_interpolant(g)
            if not grid_integrated or kernel not in (None, grid_kernel):
                raise ValueError('inconsistent kernel / integrated in phase grid')
            kernel = grid_kernel
        if K.shape[0] != 3 or K.ndim not in (5, 6):
            raise ValueError('expected phase splines [3,C,(6),nz,n,n]')
        axes = ('phase', 'channel') + (('dipole',) if self.dipole_components else ())
        ids = torch.arange(math.prod(K.shape[:-3])).reshape(K.shape[:-3])
        self.spline = SplinePSF(self.P, self.z_range, self.S, self.hz, self.zpad, xl, device=self.dev,
                               knots=K, knot_u0_px=g.get('u0_px'), knot_z0_um=g.get('z0_um'),
                               pixel_um=self.pixel_um if density else None, kernel=kernel or ('bspline' if density else 'catmull'),
                               integrated=density, backend=backend or ('sep' if density else 'conv'),
                               axes=axes, spline_index=ids, component_weights=weights)
        C = K.shape[1]
        self.tau = (torch.full((C,), 1.0 / C) if tau is None else torch.as_tensor(tau, dtype=torch.float32)).to(self.dev)
        self._init_channels(C, channel_offsets)

    # ------------------------------------------------------------------ geometry
    @property
    def n_channels(self):
        return self.spline.spline_index.shape[1]

    @property
    def knots(self):
        return self.spline.knots

    @property
    def z_range(self):
        return (self.zmin, self.zmax)

    @property
    def z_limits(self):
        return self.spline.z_limits

    @property
    def xy_limit(self):
        return self.spline.xmax

    def __repr__(self):
        L = self.spline
        return (f'FourPiSplinePSF(roisize={self.P}, channels={self.n_channels}, z_range=({self.zmin}, {self.zmax}), '
                f'oversample={self.S}, dz_um={self.hz}, zpad={self.zpad}, xy_limit={L.xmax}, knots {(L.nz, L.nv, L.nu)})')

    # ------------------------------------------------------------------ evaluation
    def basis(self, pos, grad=False, *, dipM=None, xy=None, phi=None):
        if self.mapped and grad:
            return render_with_grad(lambda p: self.basis(p, dipM=dipM, xy=xy), pos, True)
        corners = None if xy is None else xy - (self.roisize - 1) / 2
        pc = self.channel_positions(pos, corners=corners)
        images = self._phase_components(pc, grad, None, dipM)
        ratio = self.intensity_ratio[None, :, None, None, None]
        return tuple(v * ratio for v in images) if grad else images * ratio

    def components(self, pos, grad=False):
        """[3 (inc, cr, ci), N, C, P, P]; grad: [value, d/dx, d/dy, d/dz], each of that shape."""
        pc = pos[:, None, :].expand(-1, self.n_channels, -1)
        out = self._phase_components(pc, grad, None, None)
        return tuple(v.movedim(2, 0) for v in out) if grad else out.movedim(2, 0)

    def evaluate(self, pos, Phi, grad=False):
        return self._render_c(pos[:, None, :].expand(-1, self.n_channels, -1), Phi, grad)

    def _phase_components(self, pc, grad, knots, dipM, basis=False):
        ids = self.spline.spline_index.transpose(0, 1)
        if (dipM is not None or basis) and not self.dipole_components:
            raise ValueError('build with components="dipole" (dipole_components=True)')
        p = pc[:, :, None, :]
        if self.dipole_components and dipM is None and not basis:
            # Fixed dipoles in phase files are defined on the coefficients. Rounding
            # these combinations before interpolation also preserves the archive TIFF.
            bank = self._stored_splines() if knots is None else knots.reshape_as(self._stored_splines())
            fixed = torch.einsum('d,fcdzyx->fczyx', self.spline.component_weights, bank)
            return self.spline.evaluate_splines(p, ids[..., 0] // 6, grad, fixed)
        if self.dipole_components:
            p = p.unsqueeze(-2)
        images = self.spline.evaluate_splines(p, ids, grad, knots)
        if basis or not self.dipole_components:
            return images
        w = self.spline.component_weights if dipM is None else M_to_m6(torch.as_tensor(dipM).to(self.knots))
        def contract(v):
            return torch.einsum('nd,ncfdij->ncfdij', w.reshape(-1, 6), v).sum(3)
        return tuple(contract(v) for v in images) if grad else contract(images)

    @staticmethod
    def _phase_sum(images, phi):
        # Multiply each component first, preserving the arithmetic of the phase file format.
        w = torch.stack((torch.ones_like(phi), phi.cos(), -phi.sin()), -1)
        return torch.einsum('nf,ncf...->ncf...', w.to(images), images).sum(2)

    def _dipole_basis(self, pos, grad=False, *, phi=None, xy=None):
        if xy is not None and not self.mapped:
            raise ValueError('xy requires channel mappings')
        corners = None if xy is None else xy - (self.roisize - 1) / 2
        if grad:
            return render_with_grad(lambda p, phi: self._dipole_basis(p, phi=phi, xy=xy), pos, True, phi=_phase(pos, phi))
        raw = pos.ndim == 3
        pc = pos if raw else self.channel_positions(pos, corners=corners)
        images = self._phase_components(pc, False, None, None, basis=True)
        basis = self._phase_sum(images, _phase(pc, phi))
        return basis if raw else basis * self.intensity_ratio[None, :, None, None, None]

    def render_splines(self, pos, grad=False, *, phi=None, knots=None, offsets=None, dipM=None, corners=None):
        Phi = _phase(pos, phi)
        if self.mapped:
            return self._ratio(self._render_mapped(pos, Phi[:, None], offsets, corners, grad, dipM, knots=knots))
        return self._ratio(self._render_c(self.channel_positions(pos, offsets, corners), Phi, grad, knots, dipM))

    def render_channels(self, pos_c, extra=None, grad=False, knots=None, dipM=None):
        return self._render_c(pos_c, _phase(pos_c, None if extra is None else extra[:, 0]), grad, knots, dipM)

    def _render_c(self, pos_c, Phi, grad, knots=None, dipM=None):
        parts = self._phase_components(pos_c, grad, knots, dipM)
        if not grad:
            return self._phase_sum(parts, Phi)
        cr, ci = parts[0][:, :, 1], parts[0][:, :, 2]
        dphi = -Phi.sin()[:, None, None, None] * cr - Phi.cos()[:, None, None, None] * ci
        return (*[self._phase_sum(v, Phi) for v in parts], dphi)

    default_extra_starts = FourPiPSF.default_extra_starts
    wrap_extra = FourPiPSF.wrap_extra

    def to(self, device=None, dtype=None):
        self.spline.to(device, dtype)
        self.tau = self.tau.to(device=self.spline.dev, dtype=dtype)
        self.dev = self.spline.dev
        self.channel_offsets = self.channel_offsets.to(device=self.dev, dtype=dtype)
        self.intensity_ratio = self.intensity_ratio.to(device=self.dev, dtype=dtype)
        return self

    def resized(self, roisize):
        """The same spline (knots shared) for roisize x roisize patches (a larger ROI reduces xy_limit)."""
        new = copy.copy(self)
        new.roisize = new.P = int(roisize)
        new.spline = self.spline.resized(roisize)
        return new

    def to_spline(self, roisize=None, z_range=None, **grid):
        if (z_range is not None and tuple(z_range) != self.z_range) or grid:
            return self.resampled(roisize=roisize, z_range=z_range, **grid)
        return self if roisize in (None, self.roisize) else self.resized(roisize)

    def resampled(self, **grid):
        """The 4Pi splines on another knot grid or with another kernel (arguments as SplinePSF.resampled): the one
        phase / channel / dipole knot bank is re-tabulated from its own interpolant."""
        table = self.spline.resampled(**grid)
        new = copy.copy(self)
        new.spline = table
        new.roisize = new.P = table.roisize
        new.S, new.hz, new.zpad = table.S, table.hz, table.zpad
        new.zmin, new.zmax = table.z_range
        return new

    # ------------------------------------------------------------------ files
    def format_version(self, source=None):
        version = pixel_version(self.spline.format_version(), self.pixel_um,
                                (self.source_state or {}).get('meta', {}).get('px'))
        if self.spline.integrated:
            version = max(version, (0, 4))
        if source is not None:
            return max(version, source.format_version())
        if self.source_state is not None:
            mode = pixel_integration_mode(self.source_state['meta'].get('pixel_integration'), from_file=True)
            version = max(version, FourPiPSF.integration_version(mode))
            if _ext.used(self.source_state):
                version = max(version, _ext.EXTENSIONS_VERSION)
        return version

    def _stored_splines(self):
        return self.spline._volumes()

    def _file_dict(self, source=None):
        L = self.spline
        g = dict(L.to_dict(), roisize=self.P, n_channels=self.n_channels, tau=_list(self.tau), xy_limit=L.xmax)
        if L.type_name == 'cubic_spline':
            g['pixel_um'] = pixel_json(self.pixel_um)
        if self.default_dipM is not None:
            g['default_dipM'] = self.default_dipM
        if self.has_offsets():
            g['channel_offsets'] = _list(self.channel_offsets)
        d = dict(fourpi_spline=g)
        st = source.state() if source is not None else self.source_state
        if st is not None:
            d['fourpi'] = st
        key = 'dipole_knots' if self.dipole_components else 'knots'
        return d, {key: self._stored_splines().detach().cpu().numpy()}

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        g = d['fourpi_spline']
        kernel, integrated = SplinePSF.grid_interpolant(g)
        key = 'dipole_knots' if g.get('dipole_components') else 'knots'
        tables = torch.as_tensor(arrays[key])
        if g.get('dipole_components') and (tables.ndim != 6 or tables.shape[2] != 6):
            raise ValueError('expected 4Pi dipole splines [3,C,6,nz,n,n]')
        s = cls(g['roisize'], tuple(g['z_range']), tables, oversample=g['oversample'],
                dz_um=g['dz_um'], zpad=g['zpad'], xy_limit=g['xy_limit'], tau=g.get('tau'), device=device,
                pixel_um=g.get('pixel_um'), source_state=d.get('fourpi'), channel_offsets=g.get('channel_offsets'),
                integrated_grid=g if integrated else None, kernel=kernel,
                default_dipM=g.get('default_dipM'), backend=g.get('backend'))
        if s.spline.nz != g.get('nz', s.spline.nz) or s.spline.c != g.get('c', s.spline.c):
            raise ValueError('4Pi coefficient volumes do not match the stored grid')
        return s if roisize in (None, s.roisize) else s.resized(roisize)
