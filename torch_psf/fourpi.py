"""Coherent two-objective (4Pi) PSF with C polarization / phase-split interference channels, and its spline table.

FourPiPSF: the vectorial model, built on VectorialPSF.pupil_fields() of two arms (top, bottom objective).
FourPiSplinePSF: its multi-channel spline table, with the global 4Pi phase as an exact analytic dependence.

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
               global 4Pi phase of the emitter (forward(..., phi_extra=Phi), render(..., phi=Phi)).
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
from torch_psf.optics import resolve_device, Optics
from torch_psf.psf import PSF
from torch_psf.multichannel import MultiChannelPSF
from torch_psf.vectorial import VectorialPSF, gauss_blur_os, dipole_tensor, _list
from torch_psf.spline import CubicSplinePSF

DEFAULT_ZIDX = tuple(range(5, 22))


def _pair(v):
    return (float(v[0]), float(v[1])) if isinstance(v, (tuple, list)) else (float(v), float(v))


def _quad(M, x, y):
    """sum_ij M_ij Re[x_i conj(y_j)] over the dipole axis 1: M [B, 3, 3], x, y [B, 3, n, n] -> [B, n, n]"""
    yr = torch.einsum('bij,bjuv->biuv', M, y.real)
    yi = torch.einsum('bij,bjuv->biuv', M, y.imag)
    return (x.real * yr + x.imag * yi).sum(1)


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


class FourPiPSF(torch.nn.Module, MultiChannelPSF):
    """Coherent 4Pi PSF on a roisize x roisize patch with C channels -> images [N, C, P, P] (module docstring).

    Constructor: roisize (px), pixel_um, wavelength_um, NA, n_imm (a number, or (top, bottom) per arm), n_med (sample
    medium; default n_imm of the top arm), zernike_indices (Noll, both arms), npupil, oversample, mirror ('y' | 'x'),
    pol (polarization per channel, 0 = E_x, 1 = E_y; its length is the channel count C), depth_um (None or
    (top, bottom) um: asymmetric-arm geometry), zmode ('emitter' | 'stage'), z_range (um; default range of to_spline),
    device. Initial values: phi [C] (default -2 pi c / C mod 2 pi), tau [C] (default 1 / C), a (1), gamma
    (sigmoid(2)), d_bot (0, 0, 0), blur_px2 (0.1), channel_offsets [C, 2] px (zeros).
    Optional plain attribute dipM: emission-dipole tensor (3 x 3 or [N, 3, 3]) used when forward gets M=None.
    A MultiChannelPSF with extra_params ('phi',); tau plays the role of the channel intensity ratio (intensity_ratio
    stays ones and is not stored)."""
    type_name = 'fourpi'
    extra_params = ('phi',)
    BLUR_PAD = VectorialPSF.BLUR_PAD

    def __init__(self, roisize, pixel_um=0.10833, wavelength_um=0.68, NA=1.35, n_imm=1.406, n_med=None,
                 zernike_indices=DEFAULT_ZIDX, npupil=48, oversample=2, mirror='y', pol=(0, 1, 0, 1), depth_um=None,
                 zmode='emitter', z_range=(-0.6, 0.6), device=None, phi=None, tau=None, a=1.0, gamma=None,
                 d_bot=(0.0, 0.0, 0.0), blur_px2=0.1, channel_offsets=None):
        super().__init__()
        if mirror not in ('x', 'y'):
            raise ValueError("mirror must be 'x' or 'y'")
        if zmode not in ('emitter', 'stage'):
            raise ValueError("zmode must be 'emitter' or 'stage'")
        self.roisize, self.pixel_um, self.os, self.mirror, self.zmode = int(roisize), float(pixel_um), int(oversample), mirror, zmode
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
        self.register_buffer('pol', torch.tensor(list(pol), device=self.dev))
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
    def _arm_pupil(self, arm, pos):
        """Pupil fields of one arm for emitters at pos [B, 3] (its own frame): (Pb [B, 6, n, n], kx, ky, Fx, Fy,
        dk, pupil energy); arm_fields transforms them to the image plane."""
        P, kz_med, kz_imm, kx, ky = arm.pupil_fields()
        W = (torch.einsum('j,jnm->nm', arm.zcoef, arm.Z) + arm.pupil_phase) * (2 * math.pi)
        x = pos[:, 0, None, None] * self.pixel_um
        y = pos[:, 1, None, None] * self.pixel_um
        z = pos[:, 2, None, None]
        if self.depth is None:
            zph = torch.exp(1j * kz_med * z)
        else:
            dep = self.depth[0] if arm is self.top else self.depth[1]
            if self.zmode == 'emitter':
                zph = torch.exp(1j * kz_med * (dep + z).clamp(min=0.0))
            else:
                zph = torch.exp(1j * kz_med * dep.clamp(min=0.0)) * torch.exp(-1j * kz_imm * z)
        Pb = P[None] * (torch.exp(1j * (W[None] - kx * x - ky * y)) * zph)[:, None]
        ns = (self.roisize + 2 * self.BLUR_PAD) * self.os
        u = (torch.arange(ns, device=P.device) - (ns - 1) / 2) / self.os * self.pixel_um
        Fx = torch.exp(1j * u[:, None] * kx[0][None, :])
        Fy = torch.exp(1j * u[:, None] * ky[:, 0][None, :])
        dk = (kx[0, 1] - kx[0, 0]) * (ky[1, 0] - ky[0, 0])
        energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        return Pb, kx, ky, Fx, Fy, dk, energy

    def arm_fields(self, arm, pos, sgn_mu):
        """Image fields of one arm (self.top or self.bot) on the blur-padded oversampled grid: (E [B, 3 (dipole),
        2 (E_x, E_y), ns, ns], pupil energy). pos [B, 3] in the arm's own frame (x, y px, z um); sgn_mu [3] = sign of
        the dipole components in that frame."""
        B = pos.shape[0]
        Pb, kx, ky, Fx, Fy, dk, energy = self._arm_pupil(arm, pos)
        ns = Fx.shape[0]
        E = (Fy @ Pb.to(torch.complex64) @ Fx.T).reshape(B, 3, 2, ns, ns)
        E = E * sgn_mu.to(E.real.dtype)[None, :, None, None, None]
        return E * (dk / (2 * math.pi) ** 2), energy

    def fields(self, pos):
        """(E_top, E_bottom, top pupil energy): fields [B, 3, 2, ns, ns] of both arms in the top (camera) frame."""
        dev = pos.device
        Et, energy = self.arm_fields(self.top, pos, torch.ones(3, device=dev))
        pb = pos + self.d_bot[None]
        if self.mirror == 'y':
            pb = torch.stack([pb[:, 0], -pb[:, 1], -pb[:, 2]], 1)
            Eb, _ = self.arm_fields(self.bot, pb, torch.tensor([1.0, -1.0, -1.0], device=dev))
            Eb = torch.flip(Eb, dims=[-2])
            Eb = torch.stack([Eb[:, :, 0], -Eb[:, :, 1]], 2)
        else:
            pb = torch.stack([-pb[:, 0], pb[:, 1], -pb[:, 2]], 1)
            Eb, _ = self.arm_fields(self.bot, pb, torch.tensor([-1.0, 1.0, -1.0], device=dev))
            Eb = torch.flip(Eb, dims=[-1])
            Eb = torch.stack([-Eb[:, :, 0], Eb[:, :, 1]], 2)
        return Et, Eb, energy

    def forward(self, pos, M=None, phi_extra=None, intensity=None, bg=None, offsets=None):
        """pos [B, 3] (x, y px relative to the ROI centre, z um), M: emission-dipole tensor [3, 3] or [B, 3, 3] (default
        self.dipM, else isotropic), phi_extra: global 4Pi phase Phi [B] (or a number), intensity [B], bg [B] or [B, C],
        offsets [B, C, 2]: per-spot channel offsets (px; added to channel_offsets)
        -> [B, C, P, P] expected photons per emitted photon (times intensity, plus bg, when given). Channel c is
        rendered at (x, y) + channel_offsets[c] (+ offsets[:, c]); without offsets all channels share one evaluation."""
        if self._shared_position(offsets):
            I = self._forward_ref(pos, M, phi_extra)
        else:
            I = self._forward_channels(self.channel_positions(pos, offsets), M, phi_extra)
        if intensity is not None:
            I = I * intensity[:, None, None, None]
        if bg is not None:
            I = I + (bg[:, :, None, None] if bg.dim() == 2 else bg[:, None, None, None])
        return I

    def _dipole(self, M, B, dev, dtype):
        Mt = dipole_tensor(M if M is not None else self.dipM, None, dev)
        if Mt is None:
            Mt = torch.eye(3, device=dev) / 3
        if Mt.dim() == 2:
            Mt = Mt[None].expand(B, 3, 3)
        return Mt.to(dtype)

    def _forward_ref(self, pos, M=None, phi_extra=None):
        """[B, C, P, P] of all channels at one position pos [B, 3] (no channel offsets)."""
        B, dev = pos.shape[0], pos.device
        Et, Eb, energy = self.fields(pos)
        a, g = self.a, self.gamma
        Mt = dipole_tensor(M if M is not None else self.dipM, None, dev)
        if Mt is None:
            Mt = torch.eye(3, device=dev) / 3
        if Mt.dim() == 2:
            Mt = Mt[None].expand(B, 3, 3)
        Mc = Mt.to(Et.real.dtype)
        C = self.n_channels
        out = []
        for c in range(C):
            p = int(self.pol[c])
            et, eb = Et[:, :, p], Eb[:, :, p]
            ph = torch.as_tensor(self.phi[c] + (phi_extra if phi_extra is not None else 0.0), device=dev)
            e_ph = torch.exp(1j * ph.to(torch.float32))
            e_ph = e_ph.reshape(-1, 1, 1, 1) if e_ph.dim() > 0 else e_ph
            out.append(_quad(Mc, et, et) + a * a * _quad(Mc, eb, eb) + 2 * a * g * _quad(Mc, et, eb * e_ph))
        I = torch.stack(out, 1) / (1 + a * a)
        bv = self.blur_px2
        I = gauss_blur_os(I.reshape(B * C, *I.shape[-2:]), torch.stack([bv, bv, torch.zeros_like(bv)]), self.os,
                          self.BLUR_PAD)
        R, o = self.roisize, self.os
        I = I.reshape(B, C, R, o, R, o).mean((3, 5)) * self.pixel_um ** 2 / energy
        I = I * self.tau[None, :, None, None] * 2          # channel weights; x 2: one polarization per channel
        return I

    def _forward_channels(self, pos_c, M=None, phi_extra=None):
        """[B, C, P, P] with channel c at its own position pos_c [B, C, 3] (one z for all channels). The pupil fields of
        both arms are computed once per spot (at channel 0's position); channel c's lateral shift relative to it is a
        phase ramp on both arms' pupil fields (in each arm's frame) before that channel's image transform (the
        dipole components of its polarization only)."""
        B, C, dev = pos_c.shape[0], pos_c.shape[1], pos_c.device
        pos = pos_c[:, 0]
        sh = (pos_c[:, :, :2] - pos[:, None, :2]) * self.pixel_um          # [B, C, 2] um, camera frame
        Pt, kx, ky, Fx, Fy, dk, energy = self._arm_pupil(self.top, pos)
        pb = pos + self.d_bot[None]
        if self.mirror == 'y':      # bottom frame: (x, -y, -z), dipole (1, -1, -1); image flipped in y, E_y negated
            pb = torch.stack([pb[:, 0], -pb[:, 1], -pb[:, 2]], 1)
            sgn_b, sb, flip_dim, neg_pol = torch.tensor([1.0, -1.0, -1.0], device=dev), (1.0, -1.0), -2, 1
        else:                       # (-x, y, -z), dipole (-1, 1, -1); flipped in x, E_x negated
            pb = torch.stack([-pb[:, 0], pb[:, 1], -pb[:, 2]], 1)
            sgn_b, sb, flip_dim, neg_pol = torch.tensor([-1.0, 1.0, -1.0], device=dev), (-1.0, 1.0), -1, 0
        Pbt, kxb, kyb, Fxb, Fyb, dkb, _ = self._arm_pupil(self.bot, pb)
        n = Pt.shape[-1]
        ns = Fx.shape[0]
        Pt, Pbt = Pt.reshape(B, 3, 2, n, n), Pbt.reshape(B, 3, 2, n, n)
        Mc = self._dipole(M, B, dev, torch.float32)
        a, g = self.a, self.gamma
        Phi = 0.0 if phi_extra is None else phi_extra
        out = []
        for c in range(C):
            p = int(self.pol[c])
            sx, sy = sh[:, c, 0, None, None], sh[:, c, 1, None, None]
            rt = torch.exp(-1j * (kx * sx + ky * sy))[:, None]
            rb = torch.exp(-1j * (kxb * (sb[0] * sx) + kyb * (sb[1] * sy)))[:, None]
            et = (Fy @ (Pt[:, :, p] * rt).to(torch.complex64) @ Fx.T) * (dk / (2 * math.pi) ** 2)
            eb = (Fyb @ (Pbt[:, :, p] * rb).to(torch.complex64) @ Fxb.T) * (dkb / (2 * math.pi) ** 2)
            eb = eb * sgn_b.to(eb.real.dtype)[None, :, None, None]
            eb = torch.flip(eb, dims=[flip_dim]) * (-1.0 if p == neg_pol else 1.0)
            ph = torch.as_tensor(self.phi[c] + Phi, device=dev)
            e_ph = torch.exp(1j * ph.to(torch.float32))
            e_ph = e_ph.reshape(-1, 1, 1, 1) if e_ph.dim() > 0 else e_ph
            out.append(_quad(Mc, et, et) + a * a * _quad(Mc, eb, eb) + 2 * a * g * _quad(Mc, et, eb * e_ph))
        I = torch.stack(out, 1) / (1 + a * a)
        bv = self.blur_px2
        I = gauss_blur_os(I.reshape(B * C, ns, ns), torch.stack([bv, bv, torch.zeros_like(bv)]), self.os, self.BLUR_PAD)
        R, o = self.roisize, self.os
        I = I.reshape(B, C, R, o, R, o).mean((3, 5)) * self.pixel_um ** 2 / energy
        return I * self.tau[None, :, None, None] * 2

    # ------------------------------------------------------------------ PSF interface (multi-channel)
    @property
    def xy_limit(self):
        return self.roisize / 2 - 1.0

    def render(self, pos, grad=False, phi=None, M=None, offsets=None):
        """Normalized channel images [N, C, P, P] at pos [N, 3] (x, y px, z um) and 4Pi phase phi [N] (default 0), with
        the channel offsets (+ per-spot offsets [N, C, 2]). grad=True: (value, d/dx, d/dy (per px), d/dz (per um),
        d/dphi (per rad)), by forward-mode autodiff."""
        Phi = torch.zeros(len(pos), device=pos.device) if phi is None else torch.as_tensor(phi, device=pos.device,
                                                                                           dtype=torch.float32)
        return self._ratio(self._render_c(self.channel_positions(pos, offsets), Phi, grad, M))

    def render_channels(self, pos_c, extra=None, grad=False, M=None):
        """MultiChannelPSF.render_channels: extra [N, 1] = the 4Pi phase."""
        Phi = torch.zeros(pos_c.shape[0], device=pos_c.device) if extra is None else extra[:, 0]
        return self._render_c(pos_c, Phi, grad, M)

    def _render_c(self, pos_c, Phi, grad, M):
        if not self.channel_offsets.requires_grad and bool((pos_c[:, 1:] == pos_c[:, :1]).all()):
            # one position for all channels: one evaluation
            pos = pos_c[:, 0]
            fn = lambda p, f: self._forward_ref(p, M=M, phi_extra=f)
        else:
            pos = pos_c
            fn = lambda p, f: self._forward_channels(p, M=M, phi_extra=f)
        if not grad:
            return fn(pos, Phi)
        out, val = [], None
        for k in range(4):
            tp, tf = torch.zeros_like(pos), torch.zeros_like(Phi)
            if k < 3:
                tp[..., k] = 1
            else:
                tf[:] = 1
            v, d = torch.func.jvp(fn, (pos, Phi), (tp, tf))
            val = v if val is None else val
            out.append(d)
        return (val, *out)

    def default_extra_starts(self):
        """Four phase starts for the multi-start fit: 0, pi / 2, pi, 3 pi / 2."""
        return [[0.5 * math.pi * k] for k in range(4)]

    def wrap_extra(self, theta):
        """The phase column of theta to (-pi, pi] (in place)."""
        i = 4 + self.n_channels
        theta[:, i] = math.pi - torch.remainder(math.pi - theta[:, i], 2 * math.pi)
        return theta

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        if torch.is_tensor(self.dipM):
            self.dipM = fn(self.dipM)
        self.dev = self.phi.device
        return self

    def resized(self, roisize):
        """A copy for roisize x roisize patches (same parameters)."""
        new = copy.deepcopy(self)
        new.roisize = int(roisize)
        new.top, new.bot = self.top.resized(roisize), self.bot.resized(roisize)
        return new

    def __repr__(self):
        return (f'FourPiPSF(roisize={self.roisize}, pixel_um={self.pixel_um}, NA=({float(self.top.NA):.3f}, '
                f'{float(self.bot.NA):.3f}), n_imm=({self.top.n_imm}, {self.bot.n_imm}), n_med={self.n_med:.3f}, '
                f'channels={self.n_channels}, mirror={self.mirror!r}, asym={self.asym}, zmode={self.zmode!r})')

    # ------------------------------------------------------------------ parameter dict
    def state(self):
        """Parameter dict {meta, zt, zb, phi, tau, a, gamma, d_bot, blur_px2, (depth), raw, arms, (channel_offsets)}
        (the fitted-state format). meta: R, px, NA, n, lam, npupil, os, mirror, zidx, pol, NA_t, n_imm_t, NA_b,
        n_imm_b, n_med, zmode, and asym=True for the asymmetric-arm geometry. channel_offsets [C, 2] (px) only when
        nonzero (from_state: absent = zeros), so states of models without offsets are unchanged."""
        t, b = self.top, self.bot
        meta = dict(R=self.roisize, px=self.pixel_um, NA=float(t.NA), n=t.n_imm, lam=t.wavelength_um, npupil=t.npupil,
                    os=self.os, mirror=self.mirror, zidx=list(self.zidx), pol=[int(v) for v in self.pol],
                    NA_t=float(t.NA), n_imm_t=t.n_imm, NA_b=float(b.NA), n_imm_b=b.n_imm, n_med=self.n_med,
                    zmode=self.zmode)
        if self.asym:
            meta['asym'] = True
        s = dict(meta=meta, zt=_list(t.zcoef), zb=_list(b.zcoef), phi=_list(self.phi), tau=_list(self.tau),
                 a=float(self.a.detach()), gamma=float(self.gamma.detach()), d_bot=_list(self.d_bot),
                 blur_px2=float(self.blur_px2.detach()))
        if self.asym:
            s['depth'] = _list(self.depth)
        s['raw'] = {k: _list(v) if v.dim() else float(v.detach()) for k, v in self.named_parameters()
                    if not k.startswith(('top.', 'bot.')) and k != 'channel_offsets'}
        s['arms'] = {'top': _arm_state(t), 'bottom': _arm_state(b)}
        if self.has_offsets():
            s['channel_offsets'] = _list(self.channel_offsets)
        return s

    @classmethod
    def from_state(cls, s, device=None, roisize=None, zmode=None, z_range=(-0.6, 0.6)):
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
                zmode=zmode or m.get('zmode', 'emitter'), z_range=z_range, device=device)
        t = lambda v: torch.tensor(v, device=p.dev)
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
        return p

    # ------------------------------------------------------------------ files
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

    # ------------------------------------------------------------------ spline table
    def to_spline(self, roisize=None, z_range=None, oversample=4, dz_um=0.01, zpad=3, xy_limit=None, chunk=128):
        """FourPiSplinePSF of this model for roisize x roisize patches over z_range (um; default self.z_range): knots
        every 1 / oversample px and dz_um um, offset limit xy_limit (default roisize / 2 + 1 px). The three phase
        components per channel are tabulated from the images at Phi = 0, pi, pi / 2 (rendered on the larger ROI the
        knots need, chunk positions at a time). The channel offsets are baked into the tables (channel c tabulated at
        (x, y) + channel_offsets[c]; the spline's own channel_offsets are zero); keep them within ~1 px, as the
        shifted tables lose that much of the margin of the stored support."""
        P = roisize or self.roisize
        zr = self.z_range if z_range is None else z_range
        geo = CubicSplinePSF(P, zr, oversample=oversample, dz_um=dz_um, zpad=zpad,
                             xy_limit=P / 2 + 1 if xy_limit is None else xy_limit, device='cpu')
        big = self if self.roisize == geo.Pl else self.resized(geo.Pl)
        S, Pl, nz, dev = oversample, geo.Pl, geo.nz, self.dev
        zs = geo.z0 + dz_um * torch.arange(nz, device=dev, dtype=torch.float64)
        a = torch.arange(S, device=dev, dtype=torch.float64) / S
        Zg, B, A = torch.meshgrid(zs, a, a, indexing='ij')
        pos = torch.stack([A.reshape(-1), B.reshape(-1), Zg.reshape(-1)], 1).float()
        comps = []
        with torch.no_grad():
            for ph in (0.0, math.pi, 0.5 * math.pi):
                comps.append(torch.cat([big(p, phi_extra=torch.full((len(p),), ph, device=dev)) for p in pos.split(chunk)]))
        g0, gpi, g90 = comps
        inc = 0.5 * (g0 + gpi)
        cr = 0.5 * (g0 - gpi)
        ci = inc - g90
        C = self.n_channels
        # g at u = (j - c) - a / S -> knot index j S + (S - 1 - a), as CubicSplinePSF.build_planes
        K = torch.stack([comp.reshape(nz, S, S, C, Pl, Pl).flip(1).flip(2).permute(3, 0, 4, 1, 5, 2)
                         .reshape(C, nz, Pl * S, Pl * S) for comp in (inc, cr, ci)])
        return FourPiSplinePSF(P, zr, K, oversample=S, dz_um=dz_um, zpad=zpad, xy_limit=xy_limit, tau=self.tau.detach(),
                               device=dev, pixel_um=self.pixel_um, source_state=self.state())


class FourPiSplinePSF(MultiChannelPSF):
    """Multi-channel spline table of a FourPiPSF with the 4Pi phase Phi as an exact analytic dependence:
        g_c(x, y, z; Phi) = inc_c(x, y, z) + cos(Phi) cr_c(x, y, z) - sin(Phi) ci_c(x, y, z)
    (inc = incoherent part, cr + i ci = coherent cross term), each component a CubicSplinePSF (Catmull-Rom knots).

    FourPiSplinePSF(roisize, z_range, knots [3 (inc, cr, ci), C, nz, n, n], oversample, dz_um, zpad, xy_limit (default
    roisize / 2 + 1), tau [C] (channel weights of the source model, for reference), device, pixel_um, source_state
    (the FourPiPSF state it was built from, stored in the file), channel_offsets [C, 2] px (default zeros: channel c
    evaluates its tables at (x, y) + channel_offsets[c]; FourPiPSF.to_spline bakes the source's offsets into the
    tables instead)). Usually made by FourPiPSF.to_spline.
    luts[comp][c]: the CubicSplinePSF of component comp (0 inc, 1 cr, 2 ci) of channel c.
    A MultiChannelPSF with extra_params ('phi',): theta = [x, y, z, I, bg_0 .. bg_{C-1}, phi]."""
    type_name = 'fourpi_spline'
    extra_params = ('phi',)

    def __init__(self, roisize, z_range, knots, oversample=4, dz_um=0.01, zpad=3, xy_limit=None, tau=None, device=None,
                 pixel_um=None, source_state=None, channel_offsets=None):
        self.roisize = self.P = int(roisize)
        self.S, self.hz, self.zpad = int(oversample), float(dz_um), int(zpad)
        self.zmin, self.zmax = float(z_range[0]), float(z_range[1])
        self.dev = resolve_device(device)
        self.pixel_um = None if pixel_um is None else float(pixel_um)
        self.source_state = source_state
        xl = self.P / 2 + 1 if xy_limit is None else float(xy_limit)
        K = torch.as_tensor(knots)
        self.luts = [[CubicSplinePSF(self.P, (self.zmin, self.zmax), oversample=self.S, dz_um=self.hz, zpad=self.zpad,
                                     xy_limit=xl, device=self.dev, knots=K[k, c]) for c in range(K.shape[1])]
                     for k in range(3)]
        C = K.shape[1]
        self.tau = (torch.full((C,), 1.0 / C) if tau is None else torch.as_tensor(tau, dtype=torch.float32)).to(self.dev)
        self._init_channels(C, channel_offsets)

    # ------------------------------------------------------------------ geometry
    @property
    def n_channels(self):
        return len(self.luts[0])

    @property
    def knots(self):
        """[3, C, nz, n, n] (a stacked copy)."""
        return torch.stack([torch.stack([L.knots for L in row]) for row in self.luts])

    @property
    def z_range(self):
        return (self.zmin, self.zmax)

    @property
    def z_limits(self):
        return self.luts[0][0].z_limits

    @property
    def xy_limit(self):
        return self.luts[0][0].xmax

    def __repr__(self):
        L = self.luts[0][0]
        return (f'FourPiSplinePSF(roisize={self.P}, channels={self.n_channels}, z_range=({self.zmin}, {self.zmax}), '
                f'oversample={self.S}, dz_um={self.hz}, zpad={self.zpad}, xy_limit={L.xmax}, knots {tuple(L.knots.shape)})')

    # ------------------------------------------------------------------ evaluation
    def components(self, pos, grad=False):
        """[3 (inc, cr, ci), N, C, P, P]; grad: [value, d/dx, d/dy, d/dz], each of that shape."""
        out = [[L.render(pos, grad) for L in row] for row in self.luts]
        if not grad:
            return torch.stack([torch.stack(r, 1) for r in out])
        return [torch.stack([torch.stack([o[k] for o in r], 1) for r in out]) for k in range(4)]

    def evaluate(self, pos, Phi, grad=False):
        """[N, C, P, P] at pos [N, 3] and 4Pi phase Phi [N]; grad: (value, d/dx, d/dy, d/dz, d/dPhi)."""
        cP, sP = torch.cos(Phi)[:, None, None, None], torch.sin(Phi)[:, None, None, None]
        if not grad:
            inc, cr, ci = self.components(pos)
            return inc + cP * cr - sP * ci
        v, dx, dy, dz = self.components(pos, True)
        f = lambda c: c[0] + cP * c[1] - sP * c[2]
        return f(v), f(dx), f(dy), f(dz), -sP * v[1] - cP * v[2]

    def render(self, pos, grad=False, phi=None, knots=None, offsets=None):
        """[N, C, P, P] at pos [N, 3] and phi [N] (default 0), with the channel offsets (+ per-spot offsets
        [N, C, 2] px); grad: (value, d/dx, d/dy, d/dz, d/dphi). knots: optional [3, C, nz, n, n] used instead of the
        stored knots (e.g. a tensor that requires grad; as CubicSplinePSF.render(knots=))."""
        Phi = torch.zeros(len(pos), device=pos.device) if phi is None else torch.as_tensor(phi, device=pos.device,
                                                                                           dtype=torch.float32)
        if knots is None and self._shared_position(offsets):
            return self._ratio(self.evaluate(pos, Phi, grad))
        return self._ratio(self._render_c(self.channel_positions(pos, offsets), Phi, grad, knots))

    def render_channels(self, pos_c, extra=None, grad=False, knots=None):
        """MultiChannelPSF.render_channels: channel c's tables evaluated at pos_c[:, c]; extra [N, 1] = the phase."""
        Phi = torch.zeros(pos_c.shape[0], device=pos_c.device) if extra is None else extra[:, 0]
        return self._render_c(pos_c, Phi, grad, knots)

    def _render_c(self, pos_c, Phi, grad, knots=None):
        cP, sP = torch.cos(Phi)[:, None, None], torch.sin(Phi)[:, None, None]
        vals = [[] for _ in range(5 if grad else 1)]
        for c in range(self.n_channels):
            p = pos_c[:, c]
            a, b, e = (self.luts[k][c].render(p, grad, knots=None if knots is None else knots[k, c]) for k in range(3))
            if not grad:
                vals[0].append(a + cP * b - sP * e)
                continue
            for j in range(4):
                vals[j].append(a[j] + cP * b[j] - sP * e[j])
            vals[4].append(-sP * b[0] - cP * e[0])
        out = tuple(torch.stack(v, 1) for v in vals)
        return out if grad else out[0]

    default_extra_starts = FourPiPSF.default_extra_starts
    wrap_extra = FourPiPSF.wrap_extra

    def to(self, device=None, dtype=None):
        for row in self.luts:
            for L in row:
                L.to(device, dtype)
        self.tau = self.tau.to(self.luts[0][0].dev)
        self.dev = self.luts[0][0].dev
        self.channel_offsets = self.channel_offsets.to(self.dev)
        self.intensity_ratio = self.intensity_ratio.to(self.dev)
        return self

    def resized(self, roisize):
        """The same table (knots shared) for roisize x roisize patches (a larger ROI reduces xy_limit)."""
        new = copy.copy(self)
        new.roisize = new.P = int(roisize)
        new.luts = [[L.resized(roisize) for L in row] for row in self.luts]
        return new

    def to_spline(self, roisize=None, z_range=None, **kw):
        if (z_range is not None and tuple(z_range) != self.z_range) or kw:
            raise ValueError('re-tabulating a spline on another grid is not supported; build it from the source model')
        return self if roisize in (None, self.roisize) else self.resized(roisize)

    # ------------------------------------------------------------------ files
    def _file_dict(self, source=None):
        L = self.luts[0][0]
        g = dict(L.to_dict(), roisize=self.P, n_channels=self.n_channels, tau=_list(self.tau), xy_limit=L.xmax)
        if self.has_offsets():
            g['channel_offsets'] = _list(self.channel_offsets)
        d = dict(fourpi_spline=g)
        st = source.state() if source is not None else self.source_state
        if st is not None:
            d['fourpi'] = st
        return d, dict(knots=self.knots.detach().cpu().numpy())

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        g = d['fourpi_spline']
        s = cls(g['roisize'], tuple(g['z_range']), torch.as_tensor(arrays['knots']), oversample=g['oversample'],
                dz_um=g['dz_um'], zpad=g['zpad'], xy_limit=g['xy_limit'], tau=g.get('tau'), device=device,
                pixel_um=g.get('pixel_um'), source_state=d.get('fourpi'), channel_offsets=g.get('channel_offsets'))
        return s if roisize in (None, s.roisize) else s.resized(roisize)
