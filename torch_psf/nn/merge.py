"""Network maps -> emitters: the offset-aware merge (default) and DECODE's 3 x 3 non-maximum suppression.

Grid G (dict of tensors [B, H, W], from grid()): p, x, y (px, image coordinates), z (um; 0 for 2D models), sx, sy,
sz, photons, s_photons, bg, s_bg, and cov [3, 3, B, H, W] (the x, y, z covariance, px / px / um) for full_xyz
models. Both rules return rows: dict of tensors [m] b (image index in the batch), iy, ix (the pixel that represents the
emitter), p, x, y, z, sx, sy, sz, photons, s_photons, bg (+ cov [m, 3, 3]).

merge (offset-aware merging; SIMCODE's default rule): every pixel with p > cand_th is a candidate with its own
predicted position. Two candidates in a (2 R + 1)^2 neighbourhood are linked when their predicted positions coincide:
lateral distance^2 / (r0^2 + k^2 (s_i^2 + s_j^2)) + dz^2 / (rz^2 + k^2 (sz_i^2 + sz_j^2)) <= 1, with s^2 the mean of
sx^2 and sy^2. Greedy in decreasing p, a seed (a candidate without a not-yet-taken linked candidate of higher p) takes
all its not-yet-taken linked candidates. Cluster p = sum of the members' p (clamped to 1); position, photons and bg =
p-weighted means (weight='prec': precision-weighted, bg p-weighted); the sigma of a merged value = the same weighted
mean of the members' sigmas (the members are predictions from overlapping inputs, fully correlated: averaging does
not shrink the uncertainty). The covariance of a merged full_xyz emitter = the weighted mean of the members'
covariances plus the weighted scatter of their positions (parallel-axis update). Candidates with distinct positions
stay separate even in adjacent pixels.

nms (DECODE's SpatialIntegration): 3 x 3 non-maximum suppression of p above raw_th, p summed over the cross
neighbourhood, clamped to 1; parameters of the maximum pixel.
"""
import torch
import torch.nn.functional as F
from torch_psf.nn.model import xyz_cholesky

MERGE_DEFAULTS = dict(cand_th=0.05, r0=0.6, k=1.0, rz=0.3, R=2, weight='p', iters=12)
P_MIN = 0.2              # default lower bound of the rows' p (the final threshold is the user's choice)
_NAMES = ('x', 'y', 'z', 'sx', 'sy', 'sz', 'photons', 's_photons', 'bg')


def grid(out, n_params=None, cov=None):
    """SMLMNet output [B, C, H, W] -> grid G (module docstring). n_params / cov: of the model (default: from C, with
    full_xyz if C = 1 + 2 n + 3 for n = 5)."""
    C = out.shape[1]
    if n_params is None:
        cov = cov if cov is not None else (C == 14)
        n_params = (C - 1 - (3 if cov else 0)) // 2
    n = n_params
    z3 = n == 5
    names = ['x', 'y', 'z', 'bg', 'photons'] if z3 else ['x', 'y', 'bg', 'photons']
    G = dict(p=out[:, 0])
    for i, nm in enumerate(names):
        G[nm] = out[:, 1 + i]
        G['s_' + nm if nm in ('bg', 'photons') else 's' + nm] = out[:, 1 + n + i]
    if not z3:
        G['z'] = torch.zeros_like(G['p']); G['sz'] = torch.zeros_like(G['p'])
    if cov:
        sigma = out[:, 1 + n:4 + n].movedim(1, -1).float()
        L = xyz_cholesky(sigma, out[:, -3:].movedim(1, -1).float())
        with torch.autocast(out.device.type, enabled=False):
            G['cov'] = (L @ L.mT).movedim((-2, -1), (0, 1))
    return G


def _shift(t, dy, dx, fill=0):
    """out[..., y, x] = t[..., y + dy, x + dx], `fill` outside."""
    H, W = t.shape[-2:]
    out = torch.full_like(t, fill)
    y0, y1, x0, x1 = max(0, -dy), min(H, H - dy), max(0, -dx), min(W, W - dx)
    if y1 > y0 and x1 > x0:
        out[..., y0:y1, x0:x1] = t[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def aggregate_p(p, raw_th, split_th=0.6):
    """DECODE's SpatialIntegration (norm_sum): p [B, H, W] -> aggregated p [B, H, W]."""
    clip = torch.where(p > raw_th, p, torch.zeros_like(p))[:, None]
    mask1 = torch.eq(p[:, None], F.max_pool2d(clip, 3, 1, padding=1)).float()
    filt = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=p.device)[None, None]
    conv = F.conv2d(p[:, None], filt, padding=1)
    mask2 = (p * (1 - mask1[:, 0]) > split_th).float()[:, None]
    return torch.clamp(mask1 * conv + mask2 * conv, 0., 1.)[:, 0]


def _rows(G, pa, mask, vals=None):
    b, iy, ix = torch.nonzero(mask, as_tuple=True)
    V = G if vals is None else vals
    r = dict(b=b, iy=iy, ix=ix, p=pa[b, iy, ix])
    for nm in _NAMES:
        r[nm] = V[nm][b, iy, ix]
    if 'cov' in G:
        r['cov'] = V['cov'][:, :, b, iy, ix].permute(2, 0, 1)
    return r


@torch.no_grad()
def nms(G, raw_th=0.05, p_min=None, split_th=0.6):
    """DECODE's rule (module docstring); raw_th 0.05 is SIMCODE's 'v1' rule, 0.5 DECODE's native one."""
    pa = aggregate_p(G['p'], raw_th, split_th)
    return _rows(G, pa, pa > raw_th if p_min is None else (pa > raw_th) & (pa >= p_min))


def _gate(G, R, r0, k, rz):
    """-> offsets (dy, dx), d2 [n_off, B, H, W] (normalised squared distance to the neighbour; 1 = gate edge)."""
    offs = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if (dy, dx) != (0, 0)]
    s2 = 0.5 * (G['sx'] ** 2 + G['sy'] ** 2)
    sz2 = G['sz'] ** 2
    d2 = []
    for dy, dx in offs:
        xj, yj, zj = (_shift(G[n], dy, dx) for n in ('x', 'y', 'z'))
        s2j, sz2j = _shift(s2, dy, dx), _shift(sz2, dy, dx)
        d2.append(((G['x'] - xj) ** 2 + (G['y'] - yj) ** 2) / (r0 ** 2 + k ** 2 * (s2 + s2j))
                  + (G['z'] - zj) ** 2 / (rz ** 2 + k ** 2 * (sz2 + sz2j)))
    return offs, torch.stack(d2)


@torch.no_grad()
def merge(G, cand_th=0.05, r0=0.6, k=1.0, rz=0.3, R=2, weight='p', iters=12, p_min=None):
    """Offset-aware merging (module docstring). r0 px / rz um: gate widths at zero sigma, k: sigma multiplier,
    R: neighbourhood half-width (px), iters: greedy rounds (left-overs become their own clusters), p_min: lower
    bound of the clusters' p."""
    p = G['p']
    B, H, W = p.shape
    cand = p > cand_th
    offs, d2 = _gate(G, R, r0, k, rz)
    link = [cand & _shift(cand, dy, dx, False) & (d2[i] <= 1.) for i, (dy, dx) in enumerate(offs)]
    ids = torch.arange(B * H * W, device=p.device).view(B, H, W)
    key = torch.where(cand, p.double() + 1e-9 * (1. - ids.double() / (B * H * W)),      # deterministic tie-break
                      torch.full_like(p, -1., dtype=torch.double))
    U, owner, seeds = cand.clone(), torch.full_like(ids, -1), torch.zeros_like(cand)
    for _ in range(iters):
        if not bool(U.any()):
            break
        higher = torch.zeros_like(cand)
        for i, (dy, dx) in enumerate(offs):
            higher |= link[i] & _shift(U, dy, dx, False) & (_shift(key, dy, dx, -1.) > key)
        S = U & ~higher
        bk = torch.where(S, key, torch.full_like(key, -1.))
        bid = torch.where(S, ids, torch.full_like(ids, -1))
        for i, (dy, dx) in enumerate(offs):
            sj, kj = _shift(S, dy, dx, False), _shift(key, dy, dx, -1.)
            m = link[i] & sj & (kj > bk)
            bk = torch.where(m, kj, bk)
            bid = torch.where(m, _shift(ids, dy, dx, -1), bid)
        new = U & (bid >= 0)
        owner = torch.where(new, bid, owner)
        U = U & ~new
        seeds |= S
    seeds |= U                                     # left-overs (iteration cap): own clusters
    owner = torch.where(U, ids, owner)
    pw = p * cand.float()
    prec = weight == 'prec'
    sig = dict(x='sx', y='sy', z='sz', photons='s_photons')
    psum = torch.zeros_like(p)
    acc = {nm: [pw * 0, pw * 0, pw * 0] for nm in ('x', 'y', 'z', 'photons', 'bg')}    # sum w, sum w v, sum w s
    cov = G.get('cov')
    if cov is not None:
        centre = torch.stack([G[nm] for nm in 'xyz'])
        moment = torch.zeros_like(cov)
        mean_offset = torch.zeros_like(centre)
        cw = torch.zeros_like(p)

    def add(wij, shifted):
        """seed perspective: wij [B, H, W] = membership weight x p of the neighbour; shifted(t) = its value."""
        nonlocal psum, moment, mean_offset, cw
        if cov is not None:
            w = wij / shifted(cov.diagonal(dim1=0, dim2=1).sum(-1)).clamp(min=1e-8) if prec else wij
            delta = (shifted(centre) - centre) - mean_offset
            total = cw + w
            fraction = w / total.clamp_min(1e-12)
            moment += w * shifted(cov) + (cw * fraction) * delta[:, None] * delta[None, :]
            mean_offset += fraction * delta
            cw = total
        psum = psum + wij
        for nm, a in acc.items():
            sg = shifted(G[sig[nm]]) if nm in sig else torch.zeros_like(wij)
            w = wij / sg.clamp(min=1e-4) ** 2 if (prec and nm in sig) else wij
            v = shifted(G[nm])
            a[0] = a[0] + w; a[1] = a[1] + w * v; a[2] = a[2] + w * sg

    add(pw * seeds.float(), lambda t: t)
    for i, (dy, dx) in enumerate(offs):
        wij = (link[i] & (_shift(owner, dy, dx, -1) == ids)).float() * _shift(pw, dy, dx)
        add(wij * seeds.float(), lambda t, dy=dy, dx=dx: _shift(t, dy, dx))
    pc = psum.clamp(0., 1.)
    mask = seeds & (pc >= (0. if p_min is None else p_min)) & (psum > 0)
    vals = {}
    for nm, (w0, w1, w2) in acc.items():
        w0 = w0.clamp(min=1e-12)
        vals[nm] = w1 / w0
        if nm in sig:
            vals[sig[nm]] = w2 / w0
    if cov is not None:
        c = moment / cw.clamp(min=1e-12)
        vals['cov'] = c
        for i, nm in enumerate('xyz'):
            vals[nm] = centre[i] + mean_offset[i]
            vals['s' + nm] = c[i, i].clamp(min=0).sqrt()
    for nm in _NAMES:
        vals.setdefault(nm, G[nm])
    return _rows(G, pc, mask, vals)


def detect(out, rule='merge', p_min=P_MIN, n_params=None, cov=None, **kw):
    """SMLMNet output [B, C, H, W] -> rows (module docstring). rule 'merge' (keywords: MERGE_DEFAULTS) or 'nms'
    (raw_th, split_th)."""
    G = grid(out.detach().float(), n_params, cov)
    if rule == 'merge':
        return merge(G, p_min=p_min, **dict(MERGE_DEFAULTS, **kw))
    if rule == 'nms':
        return nms(G, p_min=p_min, **kw)
    raise ValueError(f"rule {rule!r}: 'merge' or 'nms'")
