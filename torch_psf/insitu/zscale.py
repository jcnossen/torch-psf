"""z scale of one PSF relative to another, and the free-knot warning text (used in the in-situ reports)."""
import torch


@torch.no_grad()
def z_scale(psf_new, psf_ref, z_range=None, n=33, photons=1e5, bg=1.0):
    """z axis of psf_new relative to psf_ref: noise-free bright emitters at x = y = 0 and n heights in z_range rendered
    with psf_new are localized with psf_ref; returns the slope d z_ref / d z_new of a straight line through the inner
    80 % of the range (1 = same z scale) and the offset (um) at the range centre."""
    zr = z_range or psf_new.z_range
    z = torch.linspace(zr[0], zr[1], n, device=psf_ref.dev if hasattr(psf_ref, 'dev') else None)
    pos = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], 1)
    mu = photons * psf_new.render(pos) + bg
    from torch_psf.fit import localize
    th, _ = localize(psf_ref, mu, None)
    k = slice(int(0.1 * n), n - int(0.1 * n))
    A = torch.stack([torch.ones_like(z[k]), z[k] - z[k].mean()], 1).double()
    co = torch.linalg.lstsq(A, th[k, 2:3].double()).solution[:, 0]
    return float(co[1]), float(co[0] - z[k].mean())


FREE_KNOT_WARNING = ('free knots can improve the held-out likelihood and the repeat precision while warping the z '
                     'axis (blinking data do not fix the z scale; see docs/paper, section 4.7): check report["z_scale"] '
                     'against a reference')
