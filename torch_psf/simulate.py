"""Simulated data with known truth: isolated emitters (patches), bead z-stacks and blinking SMLM movies.

render functions: render(pos [n, 3]) -> normalized PSF [n, P, P] (e.g. ConvSplinePSF.render, or
lambda p: vec.forward_batched(p, focus)); for bead stacks render(pos, focus [n]) -> [n, P, P].
All random numbers come from the given torch.Generator (reproducible)."""
import math
import torch
from torch_psf.config import resolve_device


def _gen(generator, device, seed):
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed)
    return generator


def simulate_emitters(render, N, bg=30.0, photons=4000.0, photons_logsd=0.4, z_range=(0.05, 0.75), xy_range=1.0,
                      generator=None, seed=0, device=None):
    """N isolated emitters, uniform x, y in +-xy_range / 2 px around the patch centre, z uniform in z_range (um),
    log-normal photons (median `photons`, log sd photons_logsd), constant bg (photons / px), Poisson noise.
    Returns dict(d [N, P, P], th [N, 5] = (x, y, z, photons, bg), mu [N, P, P])."""
    device = resolve_device(device)
    g = _gen(generator, device, seed)
    r = lambda *s: torch.rand(*s, generator=g, device=device)
    th = torch.stack([(r(N) - 0.5) * xy_range, (r(N) - 0.5) * xy_range, z_range[0] + (z_range[1] - z_range[0]) * r(N),
                      torch.exp(math.log(photons) + photons_logsd * torch.randn(N, generator=g, device=device)),
                      torch.full((N,), float(bg), device=device)], 1)
    with torch.no_grad():
        mu = th[:, 3, None, None] * render(th[:, :3]) + bg
    return dict(d=torch.poisson(mu, generator=g), th=th, mu=mu)


def simulate_bead_stack(render, focus, nbeads=6, photons=40000.0, bg=30.0, emitter_z=0.0, generator=None, seed=0,
                        device=None):
    """Bead z-stack: nbeads beads at random sub-pixel positions (+-0.5 px) and height emitter_z, imaged at the focus
    positions focus [nk] (um). render(pos, focus) -> [n, P, P]. Returns dict(d [nbeads, nk, P, P] photon counts,
    xy [nbeads, 2], focus [nk], photons, bg)."""
    device = resolve_device(device)
    g = _gen(generator, device, seed)
    nk = len(focus)
    xy = torch.rand(nbeads, 2, generator=g, device=device) - 0.5
    pos = torch.cat([xy[:, None].expand(-1, nk, -1), torch.full((nbeads, nk, 1), float(emitter_z), device=device)],
                    -1).reshape(-1, 3)
    f = focus[None].expand(nbeads, -1).reshape(-1)
    with torch.no_grad():
        mu = photons * render(pos, f) + bg
    d = torch.poisson(mu, generator=g)
    P = d.shape[-1]
    return dict(d=d.reshape(nbeads, nk, P, P), xy=xy, focus=focus, photons=photons, bg=bg)


@torch.no_grad()
def simulate_movie(render, roisize, T=6000, H=128, W=128, nsites=700, p_on=0.002, mean_on=3.0, bg=30.0,
                   photons=4000.0, photons_logsd=0.4, z_range=(0.0, 0.8), margin=4, generator=None, seed=0,
                   device=None, chunk=500, sites=None):
    """SMLM movie: fixed binding sites (uniform in the frame minus `margin` px, z uniform in z_range; or given as
    sites = (x [n], y [n], z [n]) tensors in px / px / um, e.g. from filament_sites; nsites is then ignored) that blink with a
    two-state Markov chain (off -> on with p_on per frame, on -> off with 1 / mean_on), log-normal photons per event,
    constant background, Poisson noise. Each event is rendered on a roisize x roisize patch around its pixel.
    Returns movie [T, H, W] (uint16, cpu) and the truth table [n, 6] = (frame, x, y, z, photons, site) (x = column,
    y = row, px; z um)."""
    device = resolve_device(device)
    g = _gen(generator, device, seed)
    r = lambda *s: torch.rand(*s, generator=g, device=device)
    PR = roisize
    if sites is None:
        sx, sy = margin + (W - 2 * margin) * r(nsites), margin + (H - 2 * margin) * r(nsites)
        sz = z_range[0] + (z_range[1] - z_range[0]) * r(nsites)
    else:
        sx, sy, sz = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in sites]
        nsites = len(sx)
    on = torch.zeros(T, nsites, dtype=torch.bool, device=device)
    state = torch.zeros(nsites, dtype=torch.bool, device=device)
    for t in range(T):                                     # sequential in time (Markov chain)
        u = r(nsites)
        state = torch.where(state, u > 1 / mean_on, u < p_on)
        on[t] = state
    tt, ss = torch.nonzero(on, as_tuple=True)
    n = len(tt)
    phot = torch.exp(math.log(photons) + photons_logsd * torch.randn(n, generator=g, device=device))
    X, Y, Z = sx[ss], sy[ss], sz[ss]
    hr = (PR - 1) // 2
    ox, oy = torch.round(X).long() - hr, torch.round(Y).long() - hr
    pos = torch.stack([X - torch.round(X), Y - torch.round(Y), Z], 1)
    movie = torch.empty(T, H, W, dtype=torch.uint16)
    ii = torch.arange(PR, device=device)
    for t0 in range(0, T, chunk):
        m = (tt >= t0) & (tt < t0 + chunk)
        fr = torch.full((min(chunk, T - t0), H, W), float(bg), device=device)
        if m.any():
            pat = phot[m, None, None] * render(pos[m])
            yy = (oy[m, None, None] + ii[None, :, None]).expand(-1, PR, PR)
            xx = (ox[m, None, None] + ii[None, None, :]).expand(-1, PR, PR)
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            flat = ((tt[m] - t0)[:, None, None] * H + yy.clamp(0, H - 1)) * W + xx.clamp(0, W - 1)
            fr.view(-1).index_add_(0, flat[ok], pat[ok])
        movie[t0:t0 + chunk] = torch.poisson(fr, generator=g).clamp(max=65535).to(torch.uint16).cpu()
    truth = torch.stack([tt.float(), X, Y, Z, phot, ss.float()], 1)
    return movie, truth


def filament_sites(n_filaments=6, sites_per_um=60.0, H=128, W=128, pixel_um=0.108, z_range=(0.0, 0.8), margin=6,
                   curvature=0.6, generator=None, seed=0, device=None):
    """Binding sites along smooth random filaments (a microtubule-like test structure) for simulate_movie(sites=...).
    Each filament is a curve through the frame (minus margin px) whose direction changes smoothly (curvature: rms
    change of the direction in rad per um), with its height varying smoothly within z_range. Sites are uniform along
    the curves (sites_per_um per um). Returns (x px, y px, z um) tensors."""
    device = resolve_device(device)
    g = _gen(generator, device, seed)
    r = lambda *s: torch.rand(*s, generator=g, device=device)
    xs, ys, zs = [], [], []
    step = 0.05                                             # um
    for _ in range(n_filaments):
        x, y = margin + (W - 2 * margin) * r(1), margin + (H - 2 * margin) * r(1)
        ang = 2 * math.pi * r(1)
        z0, z1, ph = r(1), r(1), 2 * math.pi * r(1)
        pts = []
        for k in range(4000):
            ang = ang + curvature * math.sqrt(step) * torch.randn(1, generator=g, device=device)
            x = x + step / pixel_um * torch.cos(ang); y = y + step / pixel_um * torch.sin(ang)
            if not (margin <= float(x) < W - margin and margin <= float(y) < H - margin):
                break
            pts.append(torch.cat([x, y]))
        if len(pts) < 2:
            continue
        P = torch.stack(pts)
        L = len(P) * step
        n = int(sites_per_um * L)
        idx = (r(n) * (len(P) - 1)).long()
        t = idx.float() / max(len(P) - 1, 1)
        zf = z0 + (z1 - z0) * t + 0.25 * torch.sin(2 * math.pi * t + ph)
        zf = zf.clamp(0, 1)
        xs.append(P[idx, 0]); ys.append(P[idx, 1]); zs.append(z_range[0] + (z_range[1] - z_range[0]) * zf)
    return torch.cat(xs), torch.cat(ys), torch.cat(zs)


PRESETS = dict(
    astigmatic=dict(zernike={6: 0.10, 5: 0.02, 11: 0.03}, focus_um=0.4, z_range=(0.0, 0.8), roisize=19),
    tetrapod=dict(zernike={6: 0.20, 12: -0.20}, focus_um=1.0, z_range=(0.0, 2.0), roisize=31),
)


@torch.no_grad()
def simulate_reference_movie(psf='astigmatic', T=200, H=64, W=64, emitters_per_frame=5.0, mean_on=3.0,
                             photons=3000.0, photons_logsd=0.3, bg=20.0, gain_e_per_adu=0.5, offset_adu=100.0,
                             readnoise_e=1.5, optics=None, seed=0, device=None):
    """Small deterministic SMLM movie for reference cases (e.g. browser tests against torch_psf).

    psf: 'astigmatic' | 'tetrapod' (vectorial presets, see PRESETS: Zernikes in waves, focus_um, z_range, render
    roisize) or a VectorialPSF (its focus_um and z_range are used). Blinking binding sites (two-state Markov chain,
    mean on-time mean_on frames, on-probability chosen so that on average emitters_per_frame are on), z uniform in the
    z range, log-normal photons, constant bg (photons / px), Poisson noise, then the camera:
    adu = round(photons / gain + offset + N(0, readnoise_e / gain)) clipped to 0..65535.
    Returns dict(movie_adu [T, H, W] uint16 numpy, truth [n, 6] numpy = (frame, x px, y px, z um, photons, site),
    psf (the VectorialPSF), camera (Camera), params (dict of the settings))."""
    from torch_psf.vectorial import VectorialPSF
    from torch_psf.config import Optics, Camera
    device = resolve_device(device)
    if isinstance(psf, str):
        pr = PRESETS[psf]
        o = optics if isinstance(optics, Optics) else Optics.from_dict(optics or dict(pixel_um=0.108, NA=1.49,
                                                                                       wavelength_um=0.68))
        vec = VectorialPSF.from_optics(o, pr['roisize'], focus_um=pr['focus_um'], z_range=pr['z_range'],
                                       zernike_indices=tuple(range(5, 23)), device=device).set_zernike(pr['zernike'])
    else:
        vec = psf
    nsites = max(1, int(round(4 * emitters_per_frame)))
    frac = emitters_per_frame / nsites                    # stationary on-fraction p_on / (p_on + 1 / mean_on)
    p_on = frac / (1 - frac) / mean_on
    g = torch.Generator(device=device).manual_seed(seed)
    fz = torch.full((1,), vec.focus_um, device=device)
    movie, truth = simulate_movie(lambda p: vec.forward_batched(p, fz.expand(len(p))), vec.roisize, T=T, H=H, W=W,
                                  nsites=nsites, p_on=p_on, mean_on=mean_on, bg=bg, photons=photons,
                                  photons_logsd=photons_logsd, z_range=vec.z_range, margin=4, generator=g,
                                  device=device)
    ph = movie.to(device).float()
    adu = ph / gain_e_per_adu + offset_adu + (readnoise_e / gain_e_per_adu) * torch.randn(ph.shape, generator=g,
                                                                                         device=device)
    adu = torch.round(adu).clamp(0, 65535).to(torch.int32).cpu().numpy().astype('uint16')
    cam = Camera(gain_e_per_adu=gain_e_per_adu, offset_adu=offset_adu, readnoise_e=readnoise_e)
    params = dict(psf=psf if isinstance(psf, str) else 'custom', T=T, H=H, W=W, emitters_per_frame=emitters_per_frame,
                  nsites=nsites, p_on=p_on, mean_on=mean_on, photons=photons, photons_logsd=photons_logsd, bg=bg,
                  gain_e_per_adu=gain_e_per_adu, offset_adu=offset_adu, readnoise_e=readnoise_e, seed=seed)
    return dict(movie_adu=adu, truth=truth.cpu().numpy(), psf=vec, camera=cam, params=params)
