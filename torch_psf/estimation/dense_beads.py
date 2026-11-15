"""PSF estimation from bead z-stacks with dense, overlapping beads (a joint fit of bead clusters).

The isolated-bead route (ZStackEstimator, estimation.fit_beads) cuts one ROI per bead and needs beads that are far
apart. In a dense bead field most beads have a neighbour within the PSF extent; here they are all used:

1. detection (find_all_beads): local maxima of the blurred maximum projection over z, minus a local background, above
   `threshold` robust standard deviations (close pairs and clusters included, min_dist px apart);
2. grouping (find_bead_clusters): beads whose patch x patch boxes overlap (patch = the PSF extent over the scanned
   range, psf_extent) form a cluster (connected components). A cluster is dropped when a member box leaves the frame
   or the field stop (field_stop_mask), when the union box is larger than max_box px or it has more than max_members
   beads;
3. model (fit_dense_beads), per cluster c and plane k on the pixels of the union of its members' boxes:
       mu = bg_ck(x, y) + sum_b I_bk PSF(x - x_b - drift_s(k), y - y_b - drift_s(k), z_e; focus_k * scale + df_b)
   with one shared VectorialPSF (Zernikes etc.), per bead x_b, y_b and photons I_bk (or I_b times a per-plane factor),
   per cluster and plane a constant or linear background, per stack a polynomial lateral drift over the planes, and a
   focus offset df_b per bead ('bead'), per stack ('stack') or a plane a + b x + c y per stack ('tilt', a tilted
   coverslip). Poisson likelihood, Adam, gradients accumulated over chunks of clusters.

Every bead is rendered into its own patch x patch box around its starting pixel and the boxes are scattered (summed)
into the flat pixel list of its cluster (BeadClusters), so clusters of any shape share one batched PSF evaluation.

Held-out check: clusters with train=False are left out of the fit and afterwards refitted with the PSF fixed (their own
positions, photons, background; drift and focus offsets of their stacks); report heldout_nll_per_px / deviance.
dense_nll() scores any PSF on a set of clusters the same way (e.g. the isolated-bead estimate on held-out clusters).

simulate_dense_bead_stack and dense_bead_layout make test data with known truth.
"""
import math
import time
import numpy as np
import torch
from scipy import ndimage
from torch_psf.fit import poisson_nll


# ---------------------------------------------------------------------------------------------------- detection
def _otsu(v, nbins=256):
    """Otsu threshold of the values v (1-D)."""
    h, e = np.histogram(v, nbins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]


def field_stop_mask(stack, margin=2, sigma=2.0, min_dark=0.02, min_contrast=0.3):
    """Inside of a field stop (a dark frame region around the illuminated aperture) of a z-stack [nk, H, W] (or its
    mean [H, W]): the blurred mean image split by an Otsu threshold (of its 0.5 - 98th percentile range), holes filled, the largest bright component,
    eroded by `margin` px. Returns a bool [H, W]; all True when there is no field stop (less than min_dark of the
    frame dark, the dark part not touching the border, or a contrast (inside - dark) / inside below min_contrast,
    inside / dark = the median of each part)."""
    a = np.asarray(stack, np.float32)
    m = ndimage.gaussian_filter(a.mean(0) if a.ndim == 3 else a, sigma)
    full = np.ones(m.shape, bool)
    t = _otsu(np.clip(m.ravel(), *np.percentile(m, [0.5, 98])))          # clipped: the bright beads do not count
    ins = m > t
    if ins.all() or not ins.any():
        return full
    dark, mid = np.median(m[~ins]), np.median(m[ins])
    if mid <= 0 or (mid - dark) / abs(mid) < min_contrast:
        return full
    ins = ndimage.binary_fill_holes(ins)
    lab, n = ndimage.label(ins)
    if n == 0:
        return full
    ins = lab == (1 + int(np.argmax(ndimage.sum(ins, lab, range(1, n + 1)))))
    out = ~ins
    border = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    if out.mean() < min_dark or not border.any():
        return full
    return ndimage.binary_erosion(ins, iterations=int(margin)) if margin > 0 else ins


def find_all_beads(stack, sigma=1.0, threshold=6.0, min_dist=3, bg_size=21, mask=None, rel_threshold=0.0):
    """All beads in a z-stack [nk, H, W] (photons or ADU), close pairs and clusters included: local maxima (at least
    min_dist px apart) of the maximum projection of the sigma-blurred planes minus its local background (a grey
    opening of size bg_size, smoothed), above threshold robust sd (MAD of the background-subtracted projection) and
    above rel_threshold x the brightest. mask [H, W] (bool): only maxima inside it. Returns (xy [n, 2] float image px
    (x = column), amp [n] = the background-subtracted peak), brightest first."""
    a = np.asarray(stack, np.float32)
    mp = ndimage.gaussian_filter(a, (0, sigma, sigma)).max(0) if sigma > 0 else a.max(0)
    bg = ndimage.uniform_filter(ndimage.grey_opening(mp, size=(bg_size, bg_size)), bg_size)
    det = mp - bg
    sd = 1.4826 * np.median(np.abs(det - np.median(det)))
    pk = (det == ndimage.maximum_filter(det, size=2 * int(min_dist) + 1)) & (det > threshold * max(sd, 1e-6))
    pk &= det > rel_threshold * det.max()
    if mask is not None:
        pk &= mask
    y, x = np.nonzero(pk)
    # sub-pixel: centroid of the positive part of det in a 3 x 3 neighbourhood
    H, W = det.shape
    xs, ys = np.empty(len(x)), np.empty(len(y))
    for i in range(len(x)):
        y0, y1, x0, x1 = max(y[i] - 1, 0), min(y[i] + 2, H), max(x[i] - 1, 0), min(x[i] + 2, W)
        w = np.clip(det[y0:y1, x0:x1], 0, None)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        s = w.sum()
        xs[i], ys[i] = ((gx * w).sum() / s, (gy * w).sum() / s) if s > 0 else (x[i], y[i])
    amp = det[y, x]
    o = np.argsort(-amp)
    return np.stack([xs[o], ys[o]], 1), amp[o]


def find_bead_clusters(xy, amp, patch, frame_shape, mask=None, max_members=6, max_box=None, faint=0.0):
    """Group beads (xy [n, 2] image px, amp [n]) whose patch x patch boxes (centred on the rounded positions)
    overlap. faint: detections with amp below faint x the median amplitude of the brightest 20 % do not take part in
    the grouping (debris and faint specks would chain clusters together); one that lies in a member box of a kept
    cluster joins it as a 'faint' member (modelled, but it does not grow the cluster's pixels), the others are ignored.
    Returns (kept, dropped): lists of dict(xy [m, 2], amp [m], box (x0, y0, x1, y1), faint [m] bool), dropped ones
    also with 'why' ('frame', 'field_stop', 'members', 'box'). max_box (px, default 3 patch): largest union box side.
    max_members counts the bright members."""
    xy_all, amp_all = np.asarray(xy, float), np.asarray(amp, float)
    if faint > 0 and len(amp_all):
        top = np.sort(amp_all)[::-1][:max(1, int(np.ceil(0.2 * len(amp_all))))]
        bright = amp_all >= faint * np.median(top)
    else:
        bright = np.ones(len(amp_all), bool)
    xy, amp = xy_all[bright], amp_all[bright]
    n = len(xy)
    max_box = 3 * patch if max_box is None else max_box
    r = np.round(xy).astype(int)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        near = np.nonzero((np.abs(r[i + 1:] - r[i]).max(1) < patch))[0] + i + 1
        for j in near:
            a, b = root(i), root(j)
            if a != b:
                parent[b] = a
    roots = np.array([root(i) for i in range(n)])
    H, W = frame_shape
    h = patch // 2
    kept, dropped = [], []
    for g in np.unique(roots):
        idx = np.nonzero(roots == g)[0]
        a0 = r[idx] - h                                  # box corners (x0, y0) of the members
        box = (int(a0[:, 0].min()), int(a0[:, 1].min()), int(a0[:, 0].max()) + patch, int(a0[:, 1].max()) + patch)
        c = dict(xy=xy[idx], amp=np.asarray(amp)[idx], box=box, faint=np.zeros(len(idx), bool))
        why = None
        if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
            why = 'frame'
        elif mask is not None and not all(mask[y0:y0 + patch, x0:x0 + patch].all() for x0, y0 in a0):
            why = 'field_stop'
        elif len(idx) > max_members:
            why = 'members'
        elif max(box[2] - box[0], box[3] - box[1]) > max_box:
            why = 'box'
        if why is None:
            kept.append(c)
        else:
            c['why'] = why
            dropped.append(c)
    # faint detections inside a member box of a kept cluster join it
    for f in np.nonzero(~bright)[0]:
        rf = np.round(xy_all[f]).astype(int)
        for c in kept:
            rb = np.round(c['xy'][~c['faint']]).astype(int)
            if (np.abs(rb - rf).max(1) <= h).any():
                c['xy'] = np.concatenate([c['xy'], xy_all[f][None]])
                c['amp'] = np.append(c['amp'], amp_all[f])
                c['faint'] = np.append(c['faint'], True)
                break
    return kept, dropped


@torch.no_grad()
def psf_extent(vec, focus_um, emitter_z_um=0.0, rel=0.02, pmin=11, pmax=41, big=81):
    """Smallest odd patch size (px) whose border stays below rel x the brightest peak (over all planes) of the PSF at
    every focus in focus_um (pass the scanned focus positions). An energy criterion does not work for beads on the
    coverslip: their supercritical light spreads into a wide, faint ring when defocused (at 0.75 um defocus, NA 1.49,
    a 41 px box still misses ~20 % of the light); this one asks where a neighbour's light becomes small compared to an
    in-focus bead."""
    v = vec.resized(big)
    f = torch.as_tensor(focus_um, dtype=torch.float32, device=v.dev).reshape(-1)
    pos = torch.zeros(len(f), 3, device=v.dev)
    pos[:, 2] = float(emitter_z_um)
    img = torch.cat([v(pos[s:s + 64], f[s:s + 64]) for s in range(0, len(f), 64)])
    peak = img.max()
    c = big // 2
    for P in range(pmin | 1, pmax + 1, 2):
        h = P // 2
        box = img[:, c - h:c + h + 1, c - h:c + h + 1]
        edge = torch.stack([box[:, 0].max(1).values, box[:, -1].max(1).values, box[:, :, 0].max(1).values,
                            box[:, :, -1].max(1).values], 1).max()
        if bool(edge <= rel * peak):
            return P
    return pmax | 1


# ---------------------------------------------------------------------------------------------------- data
class BeadClusters:
    """Clusters of beads in z-stacks, flattened for the joint fit.

    clusters: list of dict(stack (index), xy [m, 2] image px, amp [m], optional faint [m] bool: members whose box does
    not add pixels); stacks: list of [nk, H, W] tensors (photons);
    patch (odd, px). Every bead gets a patch x patch box around its rounded position; a cluster's pixels are the union of
    its members' boxes. Flat layout: d [nk, Npix] (the data of all clusters, cluster after cluster), pix_cluster
    [Npix], pix_uv [Npix, 2] (pixel position in its cluster box, -1..1), per bead: cluster, stack, idx [B, R^2]
    (flat pixel index of every pixel of its render box, R = patch + 2 render_pad; -1 = not a pixel of the cluster),
    xy0 [B, 2] (start offset from the box centre, px), corner [B, 2] (patch box corner (x0, y0), image px).
    cl_bead / cl_pix [C + 1]: first bead / pixel of each cluster. render_pad > 0 renders every bead in a box larger
    than the grouping patch, so the PSF tails of a member that fall on the other members' pixels are modelled.
    subset(ids) -> BeadClusters of some clusters; planes(sl) -> only some planes."""

    def __init__(self, clusters, crops, patch, frame_shapes, device, render_pad=0):
        self.clusters, self.crops, self.patch, self.frame_shapes = clusters, crops, int(patch), frame_shapes
        self.dev = torch.device(device)
        P = self.patch
        h = P // 2
        self.pad = int(render_pad)
        self.R = R = P + 2 * self.pad
        jj = np.arange(R) - self.pad
        d, pc, uv, b_cl, b_st, b_idx, b_xy0, b_cor, b_img = [], [], [], [], [], [], [], [], []
        cl_bead, cl_pix = [0], [0]
        self.masks = []
        npx = 0
        for ci, (c, crop) in enumerate(zip(clusters, crops)):
            x0, y0, x1, y1 = c['box']
            hh, ww = y1 - y0, x1 - x0
            r = np.round(c['xy']).astype(int)
            m = np.zeros((hh, ww), bool)
            grow = ~np.asarray(c.get('faint', np.zeros(len(r), bool)), bool)
            for rx, ry in r[grow]:
                m[ry - h - y0:ry - h - y0 + P, rx - h - x0:rx - h - x0 + P] = True
            lid = -np.ones((hh, ww), np.int64)
            lid[m] = np.arange(m.sum()) + npx
            self.masks.append(m)
            mt = torch.as_tensor(m, device=self.dev)
            d.append(crop[:, mt])
            pc.append(np.full(m.sum(), ci))
            gy, gx = np.nonzero(m)
            uv.append(np.stack([(gx - (ww - 1) / 2) / max((ww - 1) / 2, 1), (gy - (hh - 1) / 2) / max((hh - 1) / 2, 1)], 1))
            for (rx, ry), (fx, fy) in zip(r, c['xy']):
                yy, xx = ry - h - y0 + jj[:, None] + 0 * jj[None, :], rx - h - x0 + jj[None, :] + 0 * jj[:, None]
                inb = (yy >= 0) & (yy < hh) & (xx >= 0) & (xx < ww)
                v = -np.ones((R, R), np.int64)
                v[inb] = lid[yy[inb], xx[inb]]
                b_idx.append(v.ravel())                     # -1: outside the cluster's pixels (dropped)
                b_cl.append(ci)
                b_st.append(c['stack'])
                b_xy0.append((fx - rx, fy - ry))
                b_cor.append((rx - h, ry - h))
                b_img.append((fx, fy))
            npx += int(m.sum())
            cl_bead.append(len(b_cl))
            cl_pix.append(npx)
        t = lambda v, dt=torch.float32: torch.as_tensor(np.asarray(v), dtype=dt, device=self.dev)
        self.d = torch.cat(d, 1) if d else torch.zeros(0, 0, device=self.dev)
        self.pix_cluster = t(np.concatenate(pc), torch.long)
        self.pix_uv = t(np.concatenate(uv))
        self.bead_cluster, self.bead_stack = t(b_cl, torch.long), t(b_st, torch.long)
        self.idx = t(np.stack(b_idx), torch.long)
        self.xy0, self.corner, self.xy_image = t(b_xy0), t(b_cor), t(b_img)
        self.cl_bead, self.cl_pix = cl_bead, cl_pix
        self.cl_stack = t([c['stack'] for c in clusters], torch.long)
        self.n_clusters, self.n_beads, self.n_px = len(clusters), len(b_cl), npx
        self.nk = self.d.shape[0]
        self.n_stacks = len(frame_shapes)

    @classmethod
    def from_stacks(cls, stacks, clusters, patch, device=None, render_pad=0):
        """stacks: list of [nk, H, W] arrays / tensors (photons); clusters: list of dict(stack, xy, amp, box)."""
        dev = torch.device(device) if device is not None else (stacks[0].device if torch.is_tensor(stacks[0])
                                                                  else torch.device('cpu'))
        st = [s if torch.is_tensor(s) else torch.as_tensor(np.asarray(s, np.float32)) for s in stacks]
        crops = [st[c['stack']][:, c['box'][1]:c['box'][3], c['box'][0]:c['box'][2]].to(dev).float() for c in clusters]
        return cls(clusters, crops, patch, [tuple(s.shape[-2:]) for s in st], dev, render_pad)

    def subset(self, ids):
        ids = [int(i) for i in ids]
        return BeadClusters([self.clusters[i] for i in ids], [self.crops[i] for i in ids], self.patch,
                            self.frame_shapes, self.dev, self.pad)

    def planes(self, sl):
        """A copy with only the planes sl (slice or index list) of every crop."""
        return BeadClusters(self.clusters, [c[sl] for c in self.crops], self.patch, self.frame_shapes, self.dev, self.pad)

    def unflatten(self, v, c):
        """[nk, Npix] (flat, all clusters) -> [nk, h, w] image of cluster c (NaN outside its mask)."""
        m = self.masks[c]
        out = torch.full((v.shape[0],) + m.shape, float('nan'), device=v.device)
        out[:, torch.as_tensor(m, device=v.device)] = v[:, self.cl_pix[c]:self.cl_pix[c + 1]]
        return out

    def cluster_sizes(self):
        return np.diff(self.cl_bead)


def _chunks(data, active, max_renders):
    """Runs of consecutive active clusters with at most max_renders bead-planes each: list of (c0, c1)."""
    out, c0, n = [], None, 0
    for c in range(data.n_clusters + 1):
        nb = 0 if c == data.n_clusters else (data.cl_bead[c + 1] - data.cl_bead[c]) * data.nk
        if c == data.n_clusters or not active[c] or (c0 is not None and n + nb > max_renders):
            if c0 is not None:
                out.append((c0, c))
            c0, n = None, 0
        if c < data.n_clusters and active[c]:
            if c0 is None:
                c0 = c
            n += nb
    return out


# ---------------------------------------------------------------------------------------------------- fit
def fit_dense_beads(data, focus_um, vectorial, iters=300, lr=3e-3, fit=('zcoef',), emitter_z_um=0.0, train=None,
                    drift=True, drift_order=3, focus_offset='tilt', bg='linear', intensity='free',
                    fit_dz_scale=False, warmup=50, heldout_iters=150, max_renders=3000, reject=None, reject_at=None,
                    log=None, stop=None):
    """Joint fit of a shared VectorialPSF (copied, roisize = data.patch) to the bead clusters `data` (BeadClusters),
    imaged at the focus positions focus_um [nk] (um, focal plane relative to the coverslip, see stage_focus). Poisson
    likelihood, Adam (lr for the PSF parameters; cosine decay to 0.1 lr over the last 30 % of the iterations).

    fit: vectorial parameters to fit ('zcoef', 'NA', 'n_med'); () = PSF fixed (only the bead / cluster / stack terms:
    a refit, e.g. to score a PSF). emitter_z_um: bead (centre) height. train [C] bool: clusters in the fit (default
    all); the others are refitted afterwards with the PSF fixed and scored (heldout_iters). drift / drift_order: per
    stack polynomial xy drift over the planes (no constant term). focus_offset: None, 'stack' (one per stack), 'tilt'
    (a + b x + c y per stack; x, y = image position / half the frame) or 'bead' (one per bead), in um, added to the
    focus. bg: 'const' or 'linear' (per cluster and plane: const + gradients over the cluster box). intensity: 'free'
    (photons per bead and plane) or 'bead_x_plane' (per bead times a per-plane factor per stack). fit_dz_scale: a scale
    of the focus steps. warmup: first iterations with the PSF fixed (the per-bead terms settle first). reject: drop
    (from training) the clusters whose mean chi2 per pixel ((d - mu)^2 / mu) is above reject x the median, at
    iteration reject_at (default iters // 3). max_renders: bead-planes per backward pass (memory).

    Returns dict(psf, report, history, xy [B, 2] (image px), photons [B, nk], bg [C, nk], focus_offset [B] (um),
    drift, mu [nk, Npix] (the fitted expected images; data.unflatten(mu, c) for one cluster), chi2 [C], deviance [C]
    (Poisson deviance per pixel, ~1 for a model that explains the data), train [C] (after rejection), dz_scale)."""
    dev = data.dev
    P, R, nk, B, C, S = data.patch, data.R, data.nk, data.n_beads, data.n_clusters, data.n_stacks
    focus = torch.as_tensor(focus_um, dtype=torch.float32, device=dev)
    assert len(focus) == nk, (len(focus), nk)
    psf = vectorial.resized(R)
    for k in ('zcoef', 'NA', 'n_med'):
        getattr(psf, k).requires_grad_(k in fit)
    train = np.ones(C, bool) if train is None else np.asarray(train, bool).copy()
    d = data.d

    # ---- parameters
    xy = data.xy0.clone().requires_grad_(True)
    pcl = data.pix_cluster
    with torch.no_grad():
        bg0 = torch.stack([d[:, data.cl_pix[c]:data.cl_pix[c + 1]].quantile(0.1, 1) for c in range(C)])   # [C, nk]
        flux = torch.zeros(C, nk, device=dev).index_add_(0, pcl, (d - bg0[pcl].T).T).clamp_min(100.0)
        amp = torch.as_tensor(np.concatenate([np.clip(c['amp'], 1e-3, None) for c in data.clusters]), device=dev,
                              dtype=torch.float32)
        share = amp / torch.zeros(C, device=dev).index_add_(0, data.bead_cluster, amp)[data.bead_cluster]
        I0 = flux.median(1).values[data.bead_cluster] * share                                       # [B]
    if intensity == 'free':
        logI = torch.log(I0)[:, None].repeat(1, nk).requires_grad_(True)
        int_params = [logI]
        photons = lambda b0, b1: torch.exp(logI[b0:b1])
    elif intensity == 'bead_x_plane':
        logIb = torch.log(I0).requires_grad_(True)
        logIk = torch.zeros(S, nk, device=dev, requires_grad=True)
        int_params = [logIb, logIk]
        photons = lambda b0, b1: torch.exp(logIb[b0:b1, None] + logIk[data.bead_stack[b0:b1]])
    else:
        raise ValueError(intensity)
    bgc = bg0.clone().requires_grad_(True)
    bg_params = [bgc]
    if bg == 'linear':
        bgg = torch.zeros(C, nk, 2, device=dev, requires_grad=True)
        bg_params.append(bgg)
    elif bg != 'const':
        raise ValueError(bg)
    groups = [dict(params=[getattr(psf, k) for k in fit], lr=lr, name='psf'), dict(params=[xy], lr=1e-2),
              dict(params=int_params, lr=1e-2), dict(params=bg_params, lr=0.3)]
    kk = torch.arange(nk, device=dev, dtype=torch.float32) - (nk - 1) / 2
    tpow = torch.stack([(kk / max((nk - 1) / 2, 1.0)) ** (k + 1) for k in range(int(drift_order))], 0)   # [order, nk]
    dr = None
    if drift:
        dr = torch.zeros(S, int(drift_order), 2, device=dev, requires_grad=True)
        groups.append(dict(params=[dr], lr=2e-3))
    fo = None
    if focus_offset == 'stack':
        fo = torch.zeros(S, device=dev, requires_grad=True)
    elif focus_offset == 'tilt':
        fo = torch.zeros(S, 3, device=dev, requires_grad=True)
        fs = torch.tensor([[w / 2, h / 2] for h, w in data.frame_shapes], device=dev)
        xn = (data.xy_image - fs[data.bead_stack]) / fs[data.bead_stack]                       # [B, 2] in -1..1
        tilt_basis = torch.cat([torch.ones(B, 1, device=dev), xn], 1)                          # [B, 3]
    elif focus_offset == 'bead':
        fo = torch.zeros(B, device=dev, requires_grad=True)
    elif focus_offset is not None:
        raise ValueError(focus_offset)
    if fo is not None:
        groups.append(dict(params=[fo], lr=2e-3))
    log_s = None
    if fit_dz_scale:
        log_s = torch.zeros((), device=dev, requires_grad=True)
        groups.append(dict(params=[log_s], lr=1e-3, name='psf'))
    if not fit:
        groups = groups[1:]
    opt = torch.optim.Adam(groups)
    for g in opt.param_groups:
        g['lr0'] = g['lr']

    def df_bead(b0, b1):
        if fo is None:
            return torch.zeros(b1 - b0, device=dev)
        if focus_offset == 'stack':
            return fo[data.bead_stack[b0:b1]]
        if focus_offset == 'tilt':
            return (fo[data.bead_stack[b0:b1]] * tilt_basis[b0:b1]).sum(1)
        return fo[b0:b1]

    def model(c0, c1):
        b0, b1, p0, p1 = data.cl_bead[c0], data.cl_bead[c1], data.cl_pix[c0], data.cl_pix[c1]
        nb = b1 - b0
        pxy = xy[b0:b1, None].expand(-1, nk, -1)
        if dr is not None:
            pxy = pxy + torch.einsum('bkc,kn->bnc', dr[data.bead_stack[b0:b1]], tpow)
        fk = focus if log_s is None else focus * torch.exp(log_s)
        foc = fk[None] + df_bead(b0, b1)[:, None]                                             # [nb, nk]
        pos = torch.cat([pxy, torch.full((nb, nk, 1), float(emitter_z_um), device=dev)], -1).reshape(-1, 3)
        img = psf(pos, foc.reshape(-1)).reshape(nb, nk, R * R) * photons(b0, b1)[:, :, None]
        ix = data.idx[b0:b1]
        ix = torch.where(ix < 0, p1 - p0, ix - p0).reshape(-1)                               # sink: p1 - p0
        mu = torch.zeros(nk, p1 - p0 + 1, device=dev).index_add(
            1, ix, img.permute(1, 0, 2).reshape(nk, nb * R * R))[:, :-1]
        pc = pcl[p0:p1]
        b = bgc[pc].T
        if bg == 'linear':
            b = b + (bgg[pc] * data.pix_uv[p0:p1, None, :]).sum(2).T
        return mu + b                                                                         # [nk, p1 - p0]

    def loss_of(c0, c1):
        p0, p1 = data.cl_pix[c0], data.cl_pix[c1]
        return poisson_nll(d[:, p0:p1].T, model(c0, c1).T).sum()

    hist, t0, tot = [], time.time(), float('nan')
    reject_at = iters // 3 if reject_at is None else reject_at
    rejected = []
    chunks = _chunks(data, train, max_renders)
    n_flat = int(iters * 0.7)
    for it in range(iters):
        fac = 1.0 if it < n_flat else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (it - n_flat) / max(iters - n_flat, 1)))
        for g in opt.param_groups:
            g['lr'] = g['lr0'] * fac * (0.0 if (g.get('name') == 'psf' and it < warmup) else 1.0)
        opt.zero_grad()
        tot = 0.0
        for c0, c1 in chunks:
            l = loss_of(c0, c1)
            l.backward()
            tot += float(l.detach())
        opt.step()
        hist.append(dict(it=it, nll=tot))
        if log is not None:
            log(it, tot, time.time() - t0, psf.zcoef.detach())
        if reject is not None and it == reject_at:
            c2, _ = _cluster_stats(data, model, train, max_renders)
            med = float(np.median(c2[train]))
            bad = train & (c2 > reject * med)
            rejected = np.nonzero(bad)[0].tolist()
            train &= ~bad
            chunks = _chunks(data, train, max_renders)
        if stop is not None and stop():
            break
    ntrain_px = sum(data.cl_pix[c1] - data.cl_pix[c0] for c0, c1 in chunks) * nk
    rep = dict(method='dense_beads', n_beads=int(sum(data.cl_bead[c + 1] - data.cl_bead[c] for c in range(C) if train[c])),
               n_clusters=int(train.sum()), n_planes=nk, patch=P, iterations=len(hist), nll_per_px=tot / max(ntrain_px, 1),
               rejected_clusters=rejected, seconds=time.time() - t0)
    for p in [psf.zcoef, psf.NA, psf.n_med] + ([log_s] if log_s is not None else []):
        p.requires_grad_(False)
    # statistics of the training clusters (before the held-out refit touches the shared drift / focus terms)
    rej = np.zeros(C, bool)
    rej[rejected] = True
    c2, dv, mu = _cluster_stats(data, model, train | rej, max_renders, keep_mu=True)
    # ---- held-out clusters: refit their own terms with the PSF (and the dz scale) fixed, then score them
    ho = ~train
    ho[rejected] = False
    if ho.any():
        opt2 = torch.optim.Adam([dict(params=[xy], lr=1e-2), dict(params=int_params[:1], lr=1e-2),
                                 dict(params=bg_params, lr=0.3)] + ([dict(params=[dr], lr=2e-3)] if dr is not None else [])
                                + ([dict(params=[fo], lr=2e-3)] if fo is not None else []))
        hchunks = _chunks(data, ho, max_renders)
        for it in range(heldout_iters):
            opt2.zero_grad()
            for c0, c1 in hchunks:
                loss_of(c0, c1).backward()
            opt2.step()
        c2h, dvh, muh = _cluster_stats(data, model, ho, max_renders, keep_mu=True)
        c2[ho], dv[ho] = c2h[ho], dvh[ho]
        hp = torch.as_tensor(np.concatenate([np.arange(data.cl_pix[c], data.cl_pix[c + 1]) for c in np.nonzero(ho)[0]]),
                             device=dev)
        mu[:, hp] = muh[:, hp]
    with torch.no_grad():
        nll_c = np.zeros(C)
        for c in range(C):
            p0, p1 = data.cl_pix[c], data.cl_pix[c + 1]
            nll_c[c] = float(poisson_nll(d[:, p0:p1].T, mu[:, p0:p1].T.clamp_min(1e-6)).sum())
        npx_c = np.diff(data.cl_pix).astype(float) * nk
    if ho.any():
        rep['heldout_nll_per_px'] = float(nll_c[ho].sum() / npx_c[ho].sum())
        rep['heldout_deviance_per_px'] = float((dv[ho] * npx_c[ho]).sum() / npx_c[ho].sum())
        rep['heldout_n_clusters'] = int(ho.sum())
        rep['heldout_n_beads'] = int(sum(data.cl_bead[c + 1] - data.cl_bead[c] for c in np.nonzero(ho)[0]))
    rep['deviance_per_px'] = float((dv[train] * npx_c[train]).sum() / max(npx_c[train].sum(), 1))
    rep['chi2_median'] = float(np.median(c2[train])) if train.any() else float('nan')
    psf.zcoef.requires_grad_(True)
    with torch.no_grad():
        dz = None if log_s is None else float(torch.exp(log_s))
        if dz is not None:
            rep['dz_scale'] = dz
        out = dict(xy=(xy + data.corner + P // 2).detach(), photons=photons(0, B).detach(), bg=bgc.detach(),
                   focus_offset=df_bead(0, B).detach(), drift=None if dr is None else dr.detach(),
                   focus_offset_params=None if fo is None else fo.detach(), mu=mu, chi2=c2, deviance=dv,
                   nll=nll_c, train=train, heldout=ho, dz_scale=dz)
    return dict(psf=psf, report=rep, history=hist, **out)


@torch.no_grad()
def _cluster_stats(data, model, active, max_renders, keep_mu=False):
    """Per cluster mean chi2 ((d - mu)^2 / mu) and Poisson deviance per pixel; optionally mu [nk, Npix]."""
    C = data.n_clusters
    c2, dv = np.full(C, np.nan), np.full(C, np.nan)
    mu_all = torch.zeros_like(data.d) if keep_mu else None
    for c0, c1 in _chunks(data, active, max_renders):
        mu = model(c0, c1)
        p0 = data.cl_pix[c0]
        if keep_mu:
            mu_all[:, p0:data.cl_pix[c1]] = mu
        for c in range(c0, c1):
            a, b = data.cl_pix[c] - p0, data.cl_pix[c + 1] - p0
            m, dd = mu[:, a:b].clamp_min(1e-6), data.d[:, data.cl_pix[c]:data.cl_pix[c + 1]]
            c2[c] = float(((dd - m) ** 2 / m).mean())
            dev_ = 2 * (torch.where(dd > 0, dd * torch.log(dd.clamp_min(1e-9) / m), torch.zeros_like(dd)) - (dd - m))
            dv[c] = float(dev_.mean())
    return (c2, dv, mu_all) if keep_mu else (c2, dv)


def dense_nll(data, focus_um, psf, iters=150, **kw):
    """Score a fixed PSF on bead clusters: refit every cluster's own terms (positions, photons, background, drift and
    focus offsets as configured in **kw, see fit_dense_beads) with the PSF fixed. Returns the fit_dense_beads
    result; result['report']['nll_per_px'] / ['deviance_per_px'] (lower is better)."""
    kw = dict(kw)
    kw.pop('warmup', None)
    return fit_dense_beads(data, focus_um, psf, iters=iters, fit=(), warmup=0, **kw)


# ---------------------------------------------------------------------------------------------------- simulation
def dense_bead_layout(H, W, cluster_sizes, sep_px=(10.0, 20.0), margin=12, gap_px=30.0, seed=0, max_tries=20000):
    """Bead positions [n, 2] (image px) for a simulation: clusters of the given sizes (e.g. [1, 1, 2, 2, 3]) whose
    members are sep_px (min, max) apart from an earlier member (a random chain / tree), clusters at least gap_px
    apart (nearest beads), all at least `margin` px from the border. Returns (xy [n, 2] numpy, cluster id [n])."""
    rng = np.random.default_rng(seed)
    pts, cid = [], []
    for ci, m in enumerate(cluster_sizes):
        for _ in range(max_tries):
            c = [rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)]
            mem = [c]
            ok = True
            for _k in range(m - 1):
                for _t in range(200):
                    a = mem[rng.integers(len(mem))]
                    r, t = rng.uniform(*sep_px), rng.uniform(0, 2 * np.pi)
                    p = [a[0] + r * np.cos(t), a[1] + r * np.sin(t)]
                    if (margin <= p[0] <= W - margin and margin <= p[1] <= H - margin and
                            min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in mem) >= sep_px[0]):
                        mem.append(p)
                        break
                else:
                    ok = False
                    break
            if ok and (not pts or all(min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in pts) >= gap_px for p in mem)):
                pts += mem
                cid += [ci] * len(mem)
                break
        else:
            raise RuntimeError('dense_bead_layout: no room for all clusters')
    return np.array(pts), np.array(cid)


@torch.no_grad()
def simulate_dense_bead_stack(vec, focus_um, xy, H, W, photons=40000.0, bg=30.0, emitter_z_um=0.0, focus_offset=None,
                              drift=None, render_size=None, generator=None, seed=0):
    """Bead z-stack [nk, H, W] (Poisson photon counts) of beads at image positions xy [n, 2] (px), each rendered with
    the VectorialPSF vec in a render_size box (default vec.roisize; take it larger than the fit patch so the PSF tails
    are in the data) at the focus positions focus_um [nk] (+ focus_offset [n] um per bead) and summed. photons: scalar
    or [n] (or [n, nk]); bg: scalar or [H, W]; drift: [nk, 2] px (shared lateral drift). Returns dict(d, mu, xy,
    photons, focus_offset)."""
    dev = vec.dev
    R = render_size or vec.roisize
    v = vec if vec.roisize == R else vec.resized(R)
    g = generator or torch.Generator(device=dev).manual_seed(seed)
    focus = torch.as_tensor(focus_um, dtype=torch.float32, device=dev)
    nk, n = len(focus), len(xy)
    xy = torch.as_tensor(np.asarray(xy), dtype=torch.float32, device=dev)
    fo = torch.zeros(n, device=dev) if focus_offset is None else torch.as_tensor(focus_offset, dtype=torch.float32,
                                                                                    device=dev)
    ph = torch.as_tensor(photons, dtype=torch.float32, device=dev)
    ph = ph.reshape(-1, 1).expand(n, nk) if ph.dim() < 2 else ph
    dr = torch.zeros(nk, 2, device=dev) if drift is None else torch.as_tensor(drift, dtype=torch.float32, device=dev)
    mu = torch.zeros(nk, H + 2 * R, W + 2 * R, device=dev)
    h = R // 2
    for i in range(n):
        p = xy[i][None] + dr                                                           # [nk, 2]
        r = torch.round(xy[i]).long()
        pos = torch.cat([p - r[None].float(), torch.full((nk, 1), float(emitter_z_um), device=dev)], 1)
        img = v(pos, focus + fo[i]) * ph[i][:, None, None]
        y0, x0 = int(r[1]) - h + R, int(r[0]) - h + R
        mu[:, y0:y0 + R, x0:x0 + R] += img
    mu = mu[:, R:R + H, R:R + W] + torch.as_tensor(bg, dtype=torch.float32, device=dev)
    return dict(d=torch.poisson(mu, generator=g), mu=mu, xy=xy, photons=ph, focus_offset=fo)
