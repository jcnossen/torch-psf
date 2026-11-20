"""Phase-diversity PSF estimation from z-stacks of a thin object (beads on a coverslip at any density, debris
included).

Every plane k of a z-stack images the same 2D object O through the PSF h_k at a known focus f_k:
    I_k(x) = a_k (O * h_k)(x - d_k) + b_k(x)    (+ noise)
(* = convolution; a_k a per-plane intensity factor (bleaching), d_k a lateral drift, b_k the background). The object
is eliminated, so no bead positions, bead model or bead isolation are needed: overlapping beads, aggregates and debris
on the coverslip are all part of O. The PSF h_k(theta) comes from any differentiable model (a VectorialPSF: Zernikes,
NA, pupil apodization, n_med, free pupil maps; or a callable focus -> PSF stack with its own parameters) and theta is
fitted by autograd. Run per patch (patches small enough that the PSF is about constant over one): one pupil per patch
gives the field dependence, a shared pupil over all patches the field average.

Two stages:

1. Least squares, Gonsalves' reduced form (Fourier domain, windowed patches; `gonsalves_metric`). With D_k, H_k the
   spectra of the windowed, background-subtracted data and of the PSF (with the drift phase ramp and a_k), and per
   plane noise weights w_k = 1 / sigma_k^2 (Poisson: sigma_k^2 = sum_x window^2 I_k),
       O     = sum_k w_k conj(H_k) D_k / (sum_k w_k |H_k|^2 + eps)
       E(th) = sum_f [ sum_k w_k |D_k|^2 - |sum_k w_k conj(H_k) D_k|^2 / (sum_k w_k |H_k|^2 + eps) ]
   E = min_O sum_k w_k |D_k - H_k O|^2 (plus the eps term): it depends only on the PSF and the nuisance terms.
2. Poisson refinement (method='poisson', started from stage 1): the object on an extended grid (the patch plus a
   margin of the PSF half width on every side: light of beads outside the patch and the patch edges are modelled, no
   window), mu_k = a_k crop(O * h_k(. - d_k)) + b_k on the valid pixels (inside the field stop: a field stop in an
   image plane clips the image, so leaving its outside pixels out is exact), Poisson likelihood. The object is
   profiled out: started from the unwindowed least-squares object, then Fisher-scoring (Newton) steps solved by
   preconditioned conjugate gradients, warm-started, one per iteration (a free, band-limited object; a non-negative
   pixel object (Richardson-Lucy, object_solver='rl') cannot represent a bead between pixel centres and biases the
   PSF); the background by EM steps. The PSF and nuisance terms follow the gradient at the current object (at the
   object's optimum this is the gradient of the profile likelihood). The reported deviance per pixel is that of the
   profiled model (a little below 1 for a correct model: the object takes ~G^2 / (N^2 K) of the degrees of freedom).

Nuisance terms: per-plane background (const, or + a linear gradient over the patch), per-plane intensity factor
(per patch or per stack, mean log 0), lateral drift (per stack, a polynomial over the planes without constant term,
or free per plane), a focus offset (per patch or per stack: the object's height relative to the stack centre, e.g. a
tilted coverslip), optionally the z-step scale. The overall object brightness and position absorb the constant
terms.

Limits: the object must be thin (one height per patch); beads off the coverslip, thick samples or a strongly tilted
coverslip within a patch break the model. Plane-to-plane intensity / drift changes are modelled, an object change
(beads moving / appearing) is not.
"""
import math
import time
import numpy as np
import torch
from scipy import ndimage

PSF_PARAMS = ('zcoef', 'NA', 'apod', 'n_med', 'saf', 'pupil_phase', 'pupil_amp')


# ---------------------------------------------------------------------------------------------------- data
def tukey2d(N, alpha=0.25, device=None):
    """2D Tukey window [N, N] (cosine taper over alpha / 2 of the width on each side)."""
    n = torch.arange(N, device=device, dtype=torch.float32)
    w = torch.ones(N, device=device)
    if alpha > 0:
        L = alpha * (N - 1) / 2
        lo, hi = n < L, n > (N - 1) - L
        w[lo] = 0.5 * (1 - torch.cos(math.pi * n[lo] / L))
        w[hi] = 0.5 * (1 - torch.cos(math.pi * (N - 1 - n[hi]) / L))
    return w[:, None] * w[None, :]


def patch_grid(frame_shape, size, step=None, mask=None, min_valid=0.5, signal=None, min_signal=0.0):
    """Corners (x0, y0) of size x size patches on a grid with spacing `step` (default size: no overlap) that covers
    the frame [H, W] (the last row / column moved inside), keeping those with at least min_valid of their pixels
    inside `mask` (bool [H, W], e.g. the field stop) and, with `signal` ([H, W], e.g. the background-subtracted maximum
    projection), max(signal) >= min_signal in the patch. Returns int numpy [n, 2]."""
    H, W = frame_shape
    step = size if step is None else int(step)

    def starts(L):
        s = list(range(0, max(L - size, 0) + 1, step))
        if s[-1] != L - size and L >= size:
            s.append(L - size)
        return s
    out = []
    for y0 in starts(H):
        for x0 in starts(W):
            if mask is not None and mask[y0:y0 + size, x0:x0 + size].mean() < min_valid:
                continue
            if signal is not None:
                sv = signal[y0:y0 + size, x0:x0 + size]
                sv = sv[mask[y0:y0 + size, x0:x0 + size]] if mask is not None else sv
                if sv.size == 0 or sv.max() < min_signal:
                    continue
            out.append((x0, y0))
    return np.array(out, int).reshape(-1, 2)


def signal_map(stack, sigma=1.0, bg_size=21):
    """Background-subtracted maximum projection of a z-stack [nk, H, W] in robust sd units (for patch_grid)."""
    a = np.asarray(stack, np.float32)
    mp = ndimage.gaussian_filter(a, (0, sigma, sigma)).max(0)
    bg = ndimage.uniform_filter(ndimage.grey_opening(mp, size=(bg_size, bg_size)), bg_size)
    det = mp - bg
    sd = 1.4826 * np.median(np.abs(det - np.median(det)))
    return det / max(sd, 1e-6)


class PDPatches:
    """z-stack patches for phase diversity: d [n, K, N, N] (photons), valid [n, N, N] (1 = pixel used, e.g. inside
    the field stop), corner [n, 2] (x0, y0 frame px of the patch's first pixel), stack [n] (stack index of each patch),
    frame_shapes (per stack)."""

    def __init__(self, d, valid, corner, stack, frame_shapes=None):
        self.d = d
        self.valid = valid.float()
        self.corner = np.asarray(corner, int).reshape(-1, 2)
        self.stack = torch.as_tensor(np.asarray(stack), dtype=torch.long, device=d.device)
        self.frame_shapes = frame_shapes
        self.n, self.K, self.N = d.shape[0], d.shape[1], d.shape[-1]
        self.dev = d.device
        self.n_stacks = int(self.stack.max()) + 1 if self.n else 0

    @classmethod
    def from_stacks(cls, stacks, size, step=None, masks=None, corners=None, min_valid=0.5, min_signal=0.0,
                    device=None):
        """stacks: list of [K, H, W] tensors / arrays (photons). masks: per stack a bool [H, W] (pixels to use; None =
        all). corners: per stack an [n, 2] array of patch corners (default patch_grid(..., size, step, mask,
        min_valid, signal_map, min_signal))."""
        ds, vs, cs, ss, shapes = [], [], [], [], []
        for i, s in enumerate(stacks):
            a = s if torch.is_tensor(s) else torch.as_tensor(np.asarray(s, np.float32))
            a = a.to(device or a.device).float()
            shapes.append(tuple(a.shape[-2:]))
            m = None if masks is None else masks[i]
            if corners is not None:
                cc = np.asarray(corners[i], int).reshape(-1, 2)
            else:
                sig = signal_map(a.cpu().numpy()) if min_signal > 0 else None
                cc = patch_grid(a.shape[-2:], size, step, m, min_valid, sig, min_signal)
            mt = None if m is None else torch.as_tensor(np.asarray(m), device=a.device)
            for x0, y0 in cc:
                ds.append(a[:, y0:y0 + size, x0:x0 + size])
                vs.append(torch.ones(size, size, device=a.device) if mt is None else
                          mt[y0:y0 + size, x0:x0 + size].float())
                cs.append((x0, y0))
                ss.append(i)
        if not ds:
            raise ValueError('no patches')
        return cls(torch.stack(ds), torch.stack(vs), np.array(cs), np.array(ss), shapes)

    def subset(self, ids):
        ids = np.asarray(ids, int).reshape(-1)
        it = torch.as_tensor(ids, device=self.dev)
        p = PDPatches(self.d[it], self.valid[it], self.corner[ids], self.stack[it].cpu().numpy(), self.frame_shapes)
        p.n_stacks = self.n_stacks
        return p

    def planes(self, sl):
        p = PDPatches(self.d[:, sl], self.valid, self.corner, self.stack.cpu().numpy(), self.frame_shapes)
        p.n_stacks = self.n_stacks
        return p

    @property
    def centers(self):
        """patch centres [n, 2] (x, y frame px)"""
        return self.corner + (self.N - 1) / 2


# ---------------------------------------------------------------------------------------------------- Fourier tools
def _freqs(G, device):
    return torch.fft.fftfreq(G, device=device), torch.fft.rfftfreq(G, device=device)      # fy [G], fx [G // 2 + 1]


def rfft_weights(G, device=None):
    """[G, G // 2 + 1] weights of the rfft2 half spectrum (2 for the columns that stand for +-f, else 1), so that
    sum(w |X|^2) = sum over the full spectrum."""
    w = torch.full((G, G // 2 + 1), 2.0, device=device)
    w[:, 0] = 1
    if G % 2 == 0:
        w[:, -1] = 1
    return w


def psf_otf(h, G, sinc=False, drift=None):
    """Spectra of PSF images h [..., R, R] (centre at pixel (R - 1) // 2, R <= G): zero-padded to G x G with the centre
    moved to index 0, rfft2 -> [..., G, G // 2 + 1] complex. sinc: times sinc(fx) sinc(fy) (pixel integration of point
    samples of a band-limited PSF). drift [..., 2] (x, y px): the PSF shifted by drift (phase ramp)."""
    R = h.shape[-1]
    c = (R - 1) // 2
    hp = torch.nn.functional.pad(h, (0, G - R, 0, G - R))
    hp = torch.roll(hp, (-c, -c), (-2, -1))
    H = torch.fft.rfft2(hp)
    fy, fx = _freqs(G, h.device)
    if sinc:
        H = H * (torch.sinc(fy)[:, None] * torch.sinc(fx)[None, :])
    if drift is not None:
        ph = -2 * math.pi * (drift[..., 0, None, None] * fx[None, :] + drift[..., 1, None, None] * fy[:, None])
        H = H * torch.polar(torch.ones_like(ph), ph)
    return H


def gonsalves_metric(D, H, w=None, eps=0.0, wf=None, reduce=True):
    """Gonsalves' reduced least-squares error with the object eliminated:
        E = sum_f [sum_k w_k |D_k|^2 - |sum_k w_k conj(H_k) D_k|^2 / (sum_k w_k |H_k|^2 + eps)]
    D, H [..., K, F1, F2] complex: data and PSF spectra; w [..., K] plane weights (1 / noise variance; default 1); eps
    (scalar or [...]): Wiener regularisation; wf [F1, F2]: frequency weights (rfft half spectrum, OTF support mask).
    Returns [...] (summed over the frequencies) or the per-frequency error [..., F1, F2] (reduce=False).
    E equals min_O sum_k w_k |D_k - H_k O|^2 + eps |O|^2, at O = object_estimate(D, H, w, eps)."""
    wk = torch.ones(D.shape[:-2], device=D.device) if w is None else w
    wk = wk[..., None, None]
    num = (wk * H.conj() * D).sum(-3)
    den = (wk * (H.real ** 2 + H.imag ** 2)).sum(-3)
    den = den + (eps[..., None, None] if torch.is_tensor(eps) and eps.dim() > 0 else eps)
    E = (wk * (D.real ** 2 + D.imag ** 2)).sum(-3) - (num.real ** 2 + num.imag ** 2) / den
    if wf is not None:
        E = E * wf
    return E.sum((-2, -1)) if reduce else E


def object_estimate(D, H, w=None, eps=0.0):
    """The least-squares object spectrum sum_k w_k conj(H_k) D_k / (sum_k w_k |H_k|^2 + eps) ([..., F1, F2])."""
    wk = torch.ones(D.shape[:-2], device=D.device) if w is None else w
    wk = wk[..., None, None]
    den = (wk * (H.real ** 2 + H.imag ** 2)).sum(-3)
    den = den + (eps[..., None, None] if torch.is_tensor(eps) and eps.dim() > 0 else eps)
    return (wk * H.conj() * D).sum(-3) / den


# ---------------------------------------------------------------------------------------------------- PSF models
class VectorialStack:
    """A VectorialPSF as a PSF-stack model for phase diversity: a private copy with roisize R (odd) and `oversample`
    (1: point samples at the pixel centres, the pixel integration is then a sinc in the Fourier domain; exact when the
    PSF is band-limited below the pixel Nyquist frequency, 2 NA / lambda < 1 / (2 pixel)). __call__(focus [n, K]) ->
    [n, K, R, R]; emitters at height emitter_z_um (or the mean over bead_points [m, 3] um, a finite bead: the
    lateral bead size is part of the object, the axial one is not)."""

    def __init__(self, vec, R, oversample=1, emitter_z_um=0.0, bead_points=None):
        R = int(R) | 1
        self.vec = vec.resampled(R, oversample=oversample)
        self.R, self.sinc = R, oversample == 1
        self.emitter_z_um = float(emitter_z_um)
        self.bead_points = bead_points
        self.support = 2 * float(vec.n_imm) / vec.wavelength_um * vec.pixel_um     # cycles / px (NA <= n_imm)
        if self.sinc and 2 * float(vec.NA) / vec.wavelength_um * vec.pixel_um > 0.5:
            raise ValueError('oversample=1 needs 2 NA / lambda < 1 / (2 pixel); use oversample >= 2')

    def params(self, fit):
        v = self.vec
        self.freeze()
        out = []
        for k in fit:
            t = getattr(v, k)
            t.requires_grad_(True)
            out.append((k, t))
        return out

    def freeze(self):
        for k in PSF_PARAMS:
            getattr(self.vec, k).requires_grad_(False)

    def __call__(self, focus, zcoef_offset=None):
        """focus [n, K] -> [n, K, R, R]; zcoef_offset [n, nz]: per-patch Zernike offsets (field dependence)"""
        n, K = focus.shape
        R = self.R
        zo = None if zcoef_offset is None else zcoef_offset[:, None].expand(-1, K, -1).reshape(n * K, -1)
        if self.bead_points is None:
            pos = torch.zeros(n * K, 3, device=focus.device)
            pos[:, 2] = self.emitter_z_um
            return self.vec(pos, focus.reshape(-1), zcoef_offset=zo).reshape(n, K, R, R)
        b = torch.as_tensor(self.bead_points, dtype=torch.float32, device=focus.device)
        m = len(b)
        pos = torch.stack([b[:, 0] / self.vec.pixel_um, b[:, 1] / self.vec.pixel_um, self.emitter_z_um + b[:, 2]], 1)
        pos = pos[None].expand(n * K, -1, -1).reshape(-1, 3)
        img = self.vec(pos, focus.reshape(-1).repeat_interleave(m),
                       zcoef_offset=None if zo is None else zo.repeat_interleave(m, 0))
        return img.reshape(n, K, m, R, R).mean(2)

    def export(self, like):
        """the fitted parameters copied into a copy of the VectorialPSF `like` (its roisize / sampling)"""
        out = like.resized(like.roisize)
        with torch.no_grad():
            for k in PSF_PARAMS:
                getattr(out, k).copy_(getattr(self.vec, k).detach())
        out.NA.requires_grad_(False)
        out.n_med.requires_grad_(False)
        return out


class CallableStack:
    """Any differentiable PSF-stack model: fn(focus [n, K]) -> [n, K, R, R] (odd R, centre pixel (R - 1) // 2,
    pixel-integrated, sum ~1), params: dict name -> leaf tensor (requires_grad) to fit; support: OTF support radius in
    cycles / px (None = all frequencies)."""

    def __init__(self, fn, params, R, support=None, sinc=False):
        self.fn, self._params, self.R, self.support, self.sinc = fn, dict(params), int(R), support, sinc

    def params(self, fit):
        names = list(self._params) if fit is None or fit == 'all' else list(fit)
        for k in names:
            self._params[k].requires_grad_(True)
        return [(k, self._params[k]) for k in names]

    def freeze(self):
        for t in self._params.values():
            t.requires_grad_(False)

    def __call__(self, focus):
        return self.fn(focus)


def spline_stack(spline, R=None, knots=None):
    """CallableStack of a SplinePSF (bead-stack convention z = -focus, see ZStackEstimator): fits its knots
    (a copy, `knots` param; the knots are the PSF, so this refines a spline with the object profiled out)."""
    K0 = (spline.knots if knots is None else knots).detach().clone()
    Rr = int(R or spline.roisize)
    if Rr % 2 == 0 or Rr != spline.roisize:
        raise ValueError('spline_stack: odd R equal to the spline roisize')

    def fn(focus):
        n, K = focus.shape
        pos = torch.zeros(n * K, 3, device=focus.device)
        pos[:, 2] = -focus.reshape(-1)
        lo, hi = spline.z_limits
        pos[:, 2] = pos[:, 2].clamp(lo, hi)
        return spline.render_splines(pos, knots=K0).reshape(n, K, Rr, Rr)
    return CallableStack(fn, dict(knots=K0), Rr)


# ---------------------------------------------------------------------------------------------------- fit
def _lr_factor(it, iters):
    n_flat = int(iters * 0.7)
    return 1.0 if it < n_flat else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (it - n_flat) / max(iters - n_flat, 1)))


def fit_phase_diversity(patches, focus_um, psf, fit=('zcoef',), method='poisson', ls_iters=150, iters=150,
                        lr=3e-3, lr_apod=1e-2, lr_pupil=1e-3, focus_offset='patch', drift='poly', drift_order=3,
                        intensity='plane', bg='const', fit_dz_scale=False, emitter_z_um=0.0, bead_points=None,
                        render_size=41, oversample=1, window=0.25, eps=1e-4, object_solver='newton', obj_start=None,
                        obj_inner=None, cg_iters=6, warmup=20,
                        lr_focus=1e-2, lr_drift=1e-2, pupil_l2=0.0, max_planes=600, field=None, xy=None,
                        lr_field=None, log=None, stop=None):
    """Phase-diversity fit of a PSF model to z-stack patches (see the module docstring).

    patches: PDPatches (photons); focus_um [K]: focus of every plane (um, focal plane relative to the coverslip, e.g.
    stage_focus); psf: a VectorialPSF (copied; rendered by VectorialStack with render_size x render_size px,
    `oversample`, emitters at emitter_z_um or averaged over bead_points) or a CallableStack / spline_stack (any
    differentiable PSF stack). fit: the parameters to fit (VectorialPSF: 'zcoef', 'NA', 'apod', 'n_med', 'saf',
    'pupil_phase', 'pupil_amp'; a CallableStack: its parameter names); lr / lr_apod (apod, saf, pupil_amp) / lr_pupil
    (pupil_phase): Adam steps; pupil_l2: L2 penalty (per pupil sample, x the number of data pixels / 1e4) on the free
    pupil maps.
    method: 'ls' (stage 1 only) or 'poisson' (stage 1, then stage 2). ls_iters / iters: Adam iterations of the two
    stages (cosine decay of all steps over the last 30 %; the PSF fixed for the first `warmup` of stage 1).
    focus_offset: 'patch' (one per patch), 'stack', or None; drift: 'poly' (per stack, polynomial of degree drift_order
    over the planes, no constant term), 'free' (per stack and plane, mean 0) or None; intensity: 'plane' (per patch
    and plane), 'stack' (per stack and plane) or None; bg: 'const' or 'linear' per patch and plane; fit_dz_scale: the
    z-step scale. window: Tukey taper of stage 1; eps: stage-1 regularisation relative to sum_k w_k |H_k(0)|^2.
    object_solver: 'newton' (Fisher scoring, cg_iters conjugate-gradient steps each; a free, band-limited object)
    or 'rl' (Richardson-Lucy, a non-negative pixel object: biased for beads between pixels, see the notes);
    obj_start / obj_inner: object steps at the start of stage 2 / per iteration (default newton 8 / 1, rl 50 / 3).
    max_planes:
    patch-planes per backward pass (memory). field: a torch_psf.field.FieldZernikes (copied; VectorialPSF only): every
    patch gets the Zernikes zcoef + field(x, y) at its field position xy [n, 2] (camera px; default the patch
    centres); with 'field' in fit the polynomial coefficients are fitted (Adam step lr_field, default lr; the ridge
    prior is added to the loss), else held fixed. psf may also be a torch_psf.field.FieldPSF.

    Returns dict(psf (VectorialPSF copy with the fitted parameters, or the CallableStack), report, history,
    focus_offset [n] um, drift [S, K, 2] px, scale [n, K], bg [n, K], object [n, G, G] (stage 2; frame px
    corner - margin), margin, mu [n, K, N, N] (stage 2 model, or None), deviance [n] (per pixel, stage 2), nll [n],
    metric [n] (stage 1))."""
    t_start = time.time()
    if object_solver not in ('newton', 'rl'):
        raise ValueError(object_solver)
    obj_start = (8 if object_solver == 'newton' else 50) if obj_start is None else int(obj_start)
    obj_inner = (1 if object_solver == 'newton' else 3) if obj_inner is None else int(obj_inner)
    dev = patches.dev
    n, K, N = patches.n, patches.K, patches.N
    S = patches.n_stacks
    focus = torch.as_tensor(focus_um, dtype=torch.float32, device=dev).reshape(-1)
    assert len(focus) == K, (len(focus), K)
    if hasattr(psf, 'field') and hasattr(psf, 'base'):                      # a FieldPSF
        field = psf.field if field is None else field
        psf = psf.base
    model_in = psf
    if hasattr(psf, 'zcoef'):
        model = VectorialStack(psf, render_size, oversample, emitter_z_um, bead_points)
    else:
        model = psf
    R = model.R
    m = (R - 1) // 2
    G = int(math.ceil((N + 2 * m) / 8) * 8)
    sid = patches.stack
    d, valid = patches.d, patches.valid

    # ---- parameters
    fz = None
    if field is not None:
        if not isinstance(model, VectorialStack):
            raise ValueError('field dependence needs a VectorialPSF')
        fz = field.copy().to(dev)
        fz.coef.requires_grad_('field' in fit)
        fxy = torch.as_tensor(patches.centers if xy is None else xy, dtype=torch.float32,
                              device=dev).reshape(-1, 2)
    named = model.params([k for k in fit if k != 'field'])
    groups = []
    for k, t in named:
        lr_k = lr_apod if k in ('apod', 'saf', 'pupil_amp') else lr_pupil if k == 'pupil_phase' else lr
        groups.append(dict(params=[t], lr=lr_k, name='psf'))
    if fz is not None and 'field' in fit:
        groups.append(dict(params=[fz.coef], lr=lr if lr_field is None else lr_field, name='psf'))
    nuis = []
    with torch.no_grad():
        dd = torch.where(valid[:, None] > 0, d, torch.full_like(d, float('nan'))).reshape(n, K, -1)
        bg0 = torch.nanquantile(dd, 0.3, dim=-1).clamp_min(1e-3)                    # [n, K]
    # the constant background per patch and plane is not a gradient parameter: it is fixed in stage 1 (where it is
    # degenerate with the eliminated object's mean) and updated by EM steps with the object in stage 2
    bgc = bg0.clone()
    bgg = None
    if bg == 'linear':
        bgg = torch.zeros(n, K, 2, device=dev, requires_grad=True)
        nuis2 = [dict(params=[bgg], lr=0.1)]
    elif bg != 'const':
        raise ValueError(bg)
    else:
        nuis2 = []
    ls = None
    if intensity == 'plane':
        ls = torch.zeros(n, K, device=dev, requires_grad=True)
    elif intensity == 'stack':
        ls = torch.zeros(S, K, device=dev, requires_grad=True)
    elif intensity is not None:
        raise ValueError(intensity)
    if ls is not None:
        nuis.append(dict(params=[ls], lr=1e-2))
    kk = torch.arange(K, device=dev, dtype=torch.float32) - (K - 1) / 2
    tpow = torch.stack([(kk / max((K - 1) / 2, 1.0)) ** (j + 1) for j in range(int(drift_order))], 0)   # [order, K]
    dr = None
    if drift == 'poly':
        dr = torch.zeros(S, int(drift_order), 2, device=dev, requires_grad=True)
    elif drift == 'free':
        dr = torch.zeros(S, K, 2, device=dev, requires_grad=True)
    elif drift not in (None, False):
        raise ValueError(drift)
    if dr is not None:
        nuis.append(dict(params=[dr], lr=lr_drift))
    fo = None
    if focus_offset == 'patch':
        fo = torch.zeros(n, device=dev, requires_grad=True)
    elif focus_offset == 'stack':
        fo = torch.zeros(S, device=dev, requires_grad=True)
    elif focus_offset is not None:
        raise ValueError(focus_offset)
    if fo is not None:
        nuis.append(dict(params=[fo], lr=lr_focus))
    log_s = None
    if fit_dz_scale:
        log_s = torch.zeros((), device=dev, requires_grad=True)
        nuis.append(dict(params=[log_s], lr=1e-3))

    def drift_of(ids):
        if dr is None:
            return None
        s = sid[ids]
        if drift == 'poly':
            return torch.einsum('sjc,jk->skc', dr[s], tpow)                     # [n_, K, 2]
        return dr[s] - dr[s].mean(1, keepdim=True)

    def scale_of(ids):
        if ls is None:
            return torch.ones(len(ids), K, device=dev)
        v = ls[ids] if intensity == 'plane' else ls[sid[ids]]
        return torch.exp(v - v.mean(1, keepdim=True))

    def focus_of(ids):
        f = focus if log_s is None else focus * torch.exp(log_s)
        f = f[None].expand(len(ids), -1)
        if fo is None:
            return f
        off = fo[ids] if focus_offset == 'patch' else fo[sid[ids]]
        return f + off[:, None]

    uv = (torch.arange(N, device=dev, dtype=torch.float32) - (N - 1) / 2) / ((N - 1) / 2)

    def bg_of(ids):
        b = bgc[ids][..., None, None]
        if bgg is not None:
            g_ = bgg[ids]
            b = b + g_[..., 0, None, None] * uv[None, None, None, :] + g_[..., 1, None, None] * uv[None, None, :, None]
        return b                                                                  # [n_, K, 1 | N, 1 | N]

    def otf(ids):
        if fz is not None:
            it_ = torch.as_tensor(ids, device=dev)
            h = model(focus_of(ids), fz.offsets(fxy[it_], model.vec.zidx))      # [n_, K, R, R]
        else:
            h = model(focus_of(ids))                                              # [n_, K, R, R]
        H = psf_otf(h, G, model.sinc, drift_of(ids))
        return H * scale_of(ids)[..., None, None]

    def penalty():
        pf = fz.penalty() if (fz is not None and 'field' in fit) else 0.0
        if pupil_l2 <= 0:
            return pf
        p = 0.0
        for k, t in named:
            if k == 'pupil_phase':
                p = p + (t ** 2).sum()
            elif k == 'pupil_amp':
                p = p + ((t - 1) ** 2).sum()
        return pf + pupil_l2 * float(valid.sum()) * K / 1e4 * p

    per = max(1, int(max_planes) // K)
    chunks = [np.arange(s, min(s + per, n)) for s in range(0, n, per)]
    wf = rfft_weights(G, dev)
    fy, fx = _freqs(G, dev)
    if model.support is not None:
        wf = wf * ((fy[:, None] ** 2 + fx[None, :] ** 2) <= (1.02 * model.support) ** 2).float()
    win = tukey2d(N, window, dev)
    hist = []
    report = dict(method=method, n_patches=n, n_planes=K, patch=N, render_size=R, grid=G, fit=list(fit))

    def safe_step(o):
        """Adam step that skips non-finite gradients (e.g. the vectorial model's sqrt(1 - s^2) at the critical
        angle when NA is fitted: an infinite derivative on one pupil sample)"""
        for g_ in o.param_groups:
            for t in g_['params']:
                if t.grad is not None and not bool(torch.isfinite(t.grad).all()):
                    t.grad = torch.nan_to_num(t.grad, nan=0.0, posinf=0.0, neginf=0.0)
        o.step()

    def make_opt(stage2=False):
        o = torch.optim.Adam(groups + nuis + (nuis2 if stage2 else []))
        for g in o.param_groups:
            g['lr0'] = g['lr']
        return o

    def set_lr(o, it, n_it, warm):
        f = _lr_factor(it, n_it)
        for g in o.param_groups:
            g['lr'] = g['lr0'] * f * (0.0 if (g.get('name') == 'psf' and it < warm) else 1.0)

    # ---- stage 1: least squares, Gonsalves' reduced form (windowed patches)
    def ls_terms(ids):
        it_ = torch.as_tensor(ids, device=dev)
        wv = win[None] * valid[it_]                                               # [n_, N, N]
        x = (d[it_] - bg_of(it_)) * wv[:, None]
        D = torch.fft.rfft2(torch.nn.functional.pad(x, (m, G - N - m, m, G - N - m)))
        with torch.no_grad():
            var = (d[it_].clamp_min(1.0) * wv[:, None] ** 2).sum((-2, -1))          # [n_, K]
        w = 1.0 / var
        H = otf(it_)
        with torch.no_grad():
            e_abs = eps * (w * (H[..., 0, 0].abs() ** 2)).sum(-1)                  # [n_]
        return D, H, w, e_abs

    metric = np.zeros(n)
    t0 = time.time()
    if ls_iters > 0:
        opt = make_opt()
        for it in range(ls_iters):
            set_lr(opt, it, ls_iters, warmup)
            opt.zero_grad()
            tot = 0.0
            for ids in chunks:
                D, H, w, e_abs = ls_terms(ids)
                E = gonsalves_metric(D, H, w, e_abs, wf)
                metric[ids] = E.detach().cpu().numpy()
                l = E.sum()
                l.backward()
                tot += float(l.detach())
            p = penalty()
            if torch.is_tensor(p):
                p.backward()
            safe_step(opt)
            hist.append(dict(stage='ls', it=it, value=tot))
            if log is not None:
                log(it, tot, time.time() - t_start)
            if stop is not None and stop():
                break
    report.update(ls_metric=float(metric.sum()), ls_metric_per_px=float(metric.sum() / (n * K * N * N)),
                  ls_seconds=time.time() - t0)

    # ---- stage 2: Poisson, extended object profiled out by Richardson-Lucy
    obj = mu_out = None
    dev_p = np.full(n, np.nan)
    nll_p = np.full(n, np.nan)
    if method == 'poisson':
        t0 = time.time()
        pad = lambda x: torch.nn.functional.pad(x, (m, G - N - m, m, G - N - m))
        crop = lambda x: x[..., m:m + N, m:m + N]
        vpad = pad(valid)                                                          # [n, G, G]
        Vh = torch.fft.rfft2(vpad)
        obj = torch.zeros(n, G, G, device=dev)

        def forward(ids, H, O):
            it_ = torch.as_tensor(ids, device=dev)
            img = torch.fft.irfft2(torch.fft.rfft2(O)[:, None] * H, s=(G, G))     # [n_, K, G, G]
            return crop(img) + bg_of(it_)

        def conv(H, O):                                     # [n_, G, G] -> [n_, K, N, N] (no background)
            return crop(torch.fft.irfft2(torch.fft.rfft2(O)[:, None] * H, s=(G, G)))

        def conv_t(Hc, q):                                  # adjoint: [n_, K, N, N] -> [n_, G, G]
            return torch.fft.irfft2((torch.fft.rfft2(pad(q)) * Hc).sum(1), s=(G, G))

        def rl_steps(ids, H, nsteps):
            """multi-frame Richardson-Lucy (EM) steps of the object, non-negative; EM steps of the background"""
            it_ = torch.as_tensor(ids, device=dev)
            Hc = H.conj()
            den = torch.fft.irfft2(Vh[it_] * Hc.sum(1), s=(G, G))
            ok = den > 1e-3 * den.amax((-2, -1), keepdim=True)
            O = obj[it_]
            vv = valid[it_][:, None]
            nv = vv.sum((-2, -1)).clamp_min(1.0)
            for _ in range(nsteps):
                mu = forward(ids, H, O).clamp_min(1e-6)
                q = vv * d[it_] / mu
                num = conv_t(Hc, q)
                O = torch.where(ok, O * (num / den.clamp_min(1e-12)).clamp_min(0), torch.zeros_like(O))
                bgc[it_] = bgc[it_] * q.sum((-2, -1)) / nv                       # EM step of the background
            obj[it_] = O

        def newton_steps(ids, H, nsteps):
            """Fisher-scoring (Newton) steps of the object for the Poisson likelihood, solved by preconditioned
            conjugate gradients (preconditioner: the Fourier-diagonal approximation sum_k |H_k|^2 mean(w_k)); the
            object is free (a band-limited object: a bead between pixels is represented exactly, which a
            non-negative pixel object cannot do); EM steps of the background."""
            it_ = torch.as_tensor(ids, device=dev)
            Hc = H.conj()
            H2 = H.real ** 2 + H.imag ** 2
            O = obj[it_]
            vv = valid[it_][:, None]
            nv = vv.sum((-2, -1)).clamp_min(1.0)
            dk = d[it_]
            dot = lambda a, b: (a * b).sum((-2, -1), keepdim=True)
            for _ in range(nsteps):
                mu = forward(ids, H, O).clamp_min(1e-3)
                g = conv_t(Hc, vv * (1 - dk / mu))
                W = vv / mu                                                           # Fisher weights
                Pd = (H2 * (W.sum((-2, -1)) / (G * G))[..., None, None]).sum(1)      # [n_, G, Gf]
                lam = 1e-6 * Pd.amax((-2, -1), keepdim=True)
                Minv = lambda r: torch.fft.irfft2(torch.fft.rfft2(r) / (Pd + lam), s=(G, G))
                lam_r = lam[..., :1, :1]
                Aop = lambda p: conv_t(Hc, W * conv(H, p)) + lam_r * p
                x = torch.zeros_like(O)
                r = -g
                z = Minv(r)
                pdir = z
                rz = dot(r, z)
                for _c in range(cg_iters):
                    Ap = Aop(pdir)
                    alpha = rz / dot(pdir, Ap).clamp_min(1e-30)
                    x = x + alpha * pdir
                    r = r - alpha * Ap
                    z = Minv(r)
                    rz_new = dot(r, z)
                    pdir = z + (rz_new / rz.clamp_min(1e-30)) * pdir
                    rz = rz_new
                # keep the model positive on the used pixels (halve the step where it is not)
                step = torch.ones(len(ids), 1, 1, device=dev)
                for _b in range(4):
                    mn = torch.where(vv > 0, conv(H, O + step * x) + bg_of(it_), torch.full_like(dk, 1e9))
                    bad = (mn.amin((1, 2, 3)) <= 0.05 * bgc[it_].mean(1).clamp_min(1e-3))[:, None, None]
                    if not bool(bad.any()):
                        break
                    step = torch.where(bad, step * 0.5, step)
                x = torch.where(torch.isfinite(x).all((-2, -1), keepdim=True), x, torch.zeros_like(x))
                O = O + step * x
                mu = forward(ids, H, O).clamp_min(1e-3)
                bgc[it_] = bgc[it_] * (vv * dk / mu).sum((-2, -1)) / nv
            obj[it_] = O

        def obj_steps(ids, H, nsteps):
            (newton_steps if object_solver == 'newton' else rl_steps)(ids, H, nsteps)

        # start: the least-squares object of the unwindowed (masked, background-subtracted) data, clipped, then RL
        with torch.no_grad():
            for ids in chunks:
                it_ = torch.as_tensor(ids, device=dev)
                H = otf(it_)
                x = pad((d[it_] - bg_of(it_)) * valid[it_][:, None])
                var = (d[it_].clamp_min(1.0) * valid[it_][:, None]).sum((-2, -1))
                w = 1.0 / var
                e_abs = eps * (w * (H[..., 0, 0].abs() ** 2)).sum(-1)
                O = torch.fft.irfft2(object_estimate(torch.fft.rfft2(x), H, w, e_abs), s=(G, G))
                if object_solver == 'rl':
                    O = O.clamp_min(0)
                    O = O + 1e-3 * O.mean((-2, -1), keepdim=True)
                obj[it_] = O
                obj_steps(ids, H, obj_start)
        opt = make_opt(True)
        npx = float(valid.sum()) * K
        for it in range(iters):
            set_lr(opt, it, iters, warmup if ls_iters <= 0 else 0)
            opt.zero_grad()
            tot = 0.0
            for ids in chunks:
                it_ = torch.as_tensor(ids, device=dev)
                H = otf(it_)
                with torch.no_grad():
                    obj_steps(ids, H.detach(), obj_inner)
                mu = forward(ids, H, obj[it_]).clamp_min(1e-6)
                l = (valid[it_][:, None] * (mu - d[it_] * torch.log(mu))).sum()
                l.backward()
                tot += float(l.detach())
            p = penalty()
            if torch.is_tensor(p):
                p.backward()
            safe_step(opt)
            hist.append(dict(stage='poisson', it=it, value=tot / npx))
            if log is not None:
                log(ls_iters + it, tot / npx, time.time() - t_start)
            if stop is not None and stop():
                break
        # final object at the final PSF, then the statistics
        mu_out = torch.zeros_like(d)
        with torch.no_grad():
            for ids in chunks:
                it_ = torch.as_tensor(ids, device=dev)
                H = otf(it_)
                obj_steps(ids, H, 2 * obj_inner + 1)
                mu = forward(ids, H, obj[it_]).clamp_min(1e-6)
                mu_out[it_] = mu
                dv = 2 * (torch.where(d[it_] > 0, d[it_] * torch.log(d[it_].clamp_min(1e-9) / mu),
                                      torch.zeros_like(mu)) - (d[it_] - mu))
                vv = valid[it_][:, None]
                npx_i = vv.sum((1, 2, 3)) * K
                dev_p[ids] = ((dv * vv).sum((1, 2, 3)) / npx_i.clamp_min(1)).cpu().numpy()
                dc = d[it_]
                nll = (mu - dc * torch.log(mu) + torch.lgamma(dc + 1)) * vv
                nll_p[ids] = (nll.sum((1, 2, 3)) / npx_i.clamp_min(1)).cpu().numpy()
        w_p = valid.sum((-2, -1)).cpu().numpy() * K
        report.update(deviance_per_px=float((dev_p * w_p).sum() / w_p.sum()),
                      nll_per_px=float((nll_p * w_p).sum() / w_p.sum()), poisson_seconds=time.time() - t0)
    elif method != 'ls':
        raise ValueError(method)
    model.freeze()
    if fz is not None:
        fz.coef.requires_grad_(False)
        report['field'] = dict(modes=fz.noll, orders=fz.orders, n_params=fz.n_params, prior_sd=fz.prior_sd,
                               penalty=float(fz.penalty()))
    for g in nuis:
        for t in g['params']:
            t.requires_grad_(False)
    out_psf = model.export(model_in) if isinstance(model, VectorialStack) else model
    ids_all = np.arange(n)
    with torch.no_grad():
        dfo = (fo if focus_offset == 'patch' else fo[sid]) if fo is not None else torch.zeros(n, device=dev)
        drift_all = drift_of(ids_all) if dr is not None else torch.zeros(n, K, 2, device=dev)
        rep_fit = {}
        if hasattr(out_psf, 'NA'):
            rep_fit = dict(NA=float(out_psf.NA), apod=out_psf.apod.detach().cpu().tolist(), n_med=float(out_psf.n_med))
        report.update(rep_fit, seconds=time.time() - t_start)
        if log_s is not None:
            report['dz_scale'] = float(torch.exp(log_s))
        res = dict(psf=out_psf, report=report, history=hist, focus_offset=dfo.detach().clone(),
                   drift=drift_all.detach().clone(), scale=scale_of(ids_all).detach(), bg=bgc.detach().clone(),
                   object=obj, margin=m, grid=G, mu=mu_out, deviance=dev_p, nll=nll_p, metric=metric,
                   dz_scale=None if log_s is None else float(torch.exp(log_s)), field=fz)
    return res


# ---------------------------------------------------------------------------------------------------- pupil helpers
def radial_amplitude(psf, s=None, n=200):
    """Radial pupil transmission of a VectorialPSF over the physical pupil coordinate s = n_imm sin(theta_imm) = NA rho
    (0 .. n_imm): exp(sum_n apod_n rho^(2n)) for s <= NA, 0 beyond (the fitted transmission, without the fixed
    aplanatic / Fresnel factors). Returns (s [n], amplitude [n]) numpy, 1 at the centre."""
    na = torch.linspace(0, float(psf.n_imm), n) if s is None else torch.as_tensor(s, dtype=torch.float32)
    NA = float(psf.NA)
    rho = na / NA
    a = psf.apod.detach().cpu().float()
    r2 = rho ** 2
    amp = torch.exp(a[0] * r2 + a[1] * r2 ** 2 + a[2] * r2 ** 3) * (rho <= 1).float()
    return na.numpy(), amp.numpy()
