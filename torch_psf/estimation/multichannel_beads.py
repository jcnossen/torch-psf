"""PSF estimation from bead z-stacks seen in several channels (multicolour / biplane / dual view on one camera), with
shared bead positions, a channel mapping and a field-dependent PSF per channel.

Every bead has one position (x0, y0) in the reference channel 0 (crop px) and one focus offset (its height); channel c
sees it at from_ref_c(x0, y0) (torch_psf.mapping.ChannelMap, fitted jointly), with its own PSF there: a VectorialPSF
per channel (Zernikes, NA, apod; NA / apod optionally shared) and optionally a field polynomial per channel in channel
coordinates (torch_psf.field.FieldZernikes). Per channel and bead the photons per plane (or a per-bead intensity
ratio to a shared per-plane intensity), per channel, cluster and plane a linear background, per stack a lateral drift
(shared: one camera, the drift is in the sample, mapped into every channel), per channel a focus offset (chromatic focal
shift; channel 0 fixed). Per channel c and cluster k

    mu_ck = bg_ck(x, y) + sum_b I_cbk PSF_c(from_ref_c(X_b + drift_s(k)) - box_cb, z_e;
                                                focus_k + df_b + dfc_c; Zernikes_c + field_c(x_cb, y_cb))

on the pixels of the cluster in that channel: the union of its members' boxes at their mapped positions, when all of
them lie inside channel c's frame and field stop (else the cluster is not used in that channel). The data structure is
one estimation.dense_beads.BeadClusters per channel plus the global bead / cluster ids (MultiChannelBeads), so the
render / scatter machinery of fit_dense_beads is reused per channel.

    data = build_multichannel_beads(stacks, maps, patch, masks)          # stacks[c][s] [nk, H, W] photons
    r = fit_multichannel_beads(data, focus_um, [vec0, vec1], maps, fit=('zcoef', 'NA', 'apod', 'field'),
                               fields=[fz0, fz1], fit_mapping=True, train=train)
    r['psfs'], r['maps'], r['report']['channels'][c]['heldout_deviance_per_px']
    multichannel_nll(heldout_data, focus_um, psfs, maps)                 # score fixed PSFs (+ mapping) on clusters

The independent-channel fit is the same function on one channel (data.channel(c)) with the mapping fixed: the bead
positions are then free per channel.
"""
import math
import time
import numpy as np
import torch
from scipy import ndimage
from torch_psf.fit import poisson_nll
from torch_psf.estimation.dense_beads import find_all_beads, BeadClusters, _chunks
from torch_psf.mapping import ChannelMap, match_pairs, register_points


# ---------------------------------------------------------------------------------------------------- field stop
def channel_stop_mask(stack, frac=0.35, open_px=15, sigma=2.0, margin=2, min_dark=0.02, min_contrast=0.3):
    """Field stop of one channel of a bead z-stack [nk, H, W] (photons): the median over the planes (the beads are in
    focus in a few planes only), a grey opening of open_px (removes the beads), blurred; inside = above lo + frac (hi -
    lo) (lo, hi = the 1st / 90th percentiles; robust to a background gradient inside the stop, where an Otsu split
    fails), holes filled, the largest component, eroded by margin px. All True when there is no dark region touching
    the border (less than min_dark of the frame) or the contrast (inside - dark) / inside is below min_contrast."""
    a = np.asarray(stack.detach().cpu() if torch.is_tensor(stack) else stack, np.float32)
    med = np.median(a, 0) if a.ndim == 3 else a
    m = ndimage.gaussian_filter(ndimage.grey_opening(med, size=(open_px, open_px)), sigma)
    full = np.ones(m.shape, bool)
    lo, hi = np.percentile(m, [1, 90])
    ins = m > lo + frac * (hi - lo)
    if ins.all() or not ins.any():
        return full
    ins = ndimage.binary_fill_holes(ins)
    lab, n = ndimage.label(ins)
    ins = lab == (1 + int(np.argmax(ndimage.sum(ins, lab, range(1, n + 1)))))
    out = ~ins
    dark, mid = np.median(m[out]) if out.any() else 0.0, np.median(m[ins])
    border = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    if out.mean() < min_dark or not border.any() or mid <= 0 or (mid - dark) / abs(mid) < min_contrast:
        return full
    return ndimage.binary_erosion(ins, iterations=int(margin)) if margin > 0 else ins


# ---------------------------------------------------------------------------------------------------- detection
def detect_channels(stacks, threshold=6.0, min_dist=3, faint=0.1, sigma=1.0, rel_threshold=0.0):
    """Bead detections of every channel and stack: [c][s] dict(xy [n, 2] channel px, amp [n], bright [n] bool (amp
    >= faint x the median of the brightest 20 % of that channel and stack)); rel_threshold: only maxima above that
    fraction of the brightest (find_all_beads; for a few bright beads on a structured background, where the robust
    threshold alone gives hundreds of noise maxima that chain the beads into clusters)"""
    out = []
    for cs in stacks:
        row = []
        for st in cs:
            a = st.detach().cpu().numpy() if torch.is_tensor(st) else np.asarray(st, np.float32)
            xy, amp = find_all_beads(a, sigma=sigma, threshold=threshold, min_dist=min_dist,
                                     rel_threshold=rel_threshold)
            top = np.sort(amp)[::-1][:max(1, int(np.ceil(0.2 * len(amp))))] if len(amp) else np.ones(1)
            row.append(dict(xy=xy, amp=amp, bright=amp >= faint * np.median(top)))
        out.append(row)
    return out


def register_channels(dets, frame_shapes, order=1, stacks=None, max_points=None, **kw):
    """ChannelMap per channel (channel 0 identity) from the bright detections: register_points of channel c against
    channel 0, the detections of the stacks (default all) pooled (positions of different stacks are not matched with
    each other: each stack is registered with the same map; the pairs are pooled for the least-squares fit). Returns
    (maps, info per channel: flip, n_pairs, rms_px, residuals, pairs [(stack, i_ref, i_ch)])."""
    C = len(dets)
    S = len(dets[0])
    stacks = list(range(S)) if stacks is None else list(stacks)
    H, W = frame_shapes[0]
    maps, infos = [ChannelMap.identity_map(0, norm=(W / 2, H / 2, W / 2, H / 2))], [None]
    for c in range(1, C):
        # coarse map from the stack with the most bright detections, then all stacks pooled
        s0 = max(stacks, key=lambda s: int(dets[0][s]['bright'].sum()) + int(dets[c][s]['bright'].sum()))
        r = dets[0][s0]['xy'][dets[0][s0]['bright']]
        q = dets[c][s0]['xy'][dets[c][s0]['bright']]
        m, _, info0 = register_points(r, q, frame_shapes[c], order=1, index=c, max_points=max_points,
                                      ref_amp=dets[0][s0]['amp'][dets[0][s0]['bright']],
                                      ch_amp=dets[c][s0]['amp'][dets[c][s0]['bright']], **kw)
        for tol in (2.0, 1.5):
            R, Q, pairs = [], [], []
            for s in stacks:
                br, bq = dets[0][s]['bright'], dets[c][s]['bright']
                ir, iq = match_pairs(m.from_ref(dets[0][s]['xy'][br]), dets[c][s]['xy'][bq], tol)
                R.append(dets[0][s]['xy'][br][ir]); Q.append(dets[c][s]['xy'][bq][iq])
                pairs += [(s, int(np.nonzero(br)[0][i]), int(np.nonzero(bq)[0][j])) for i, j in zip(ir, iq)]
            R, Q = np.concatenate(R), np.concatenate(Q)
            m = ChannelMap.fit_points(R, Q, order, norm=maps[0].norm, index=c)
        res = m.from_ref(R) - Q
        maps.append(m)
        infos.append(dict(flip=info0['flip'], coarse_shift=info0['coarse_shift'], histogram_peak=info0['histogram_peak'],
                          n_pairs=int(len(R)), rms_px=float(np.sqrt((res ** 2).sum(1).mean())), residuals=res,
                          ref_xy=R, ch_xy=Q, pairs=pairs, order=int(order), describe=m.describe()))
    return maps, infos


# ---------------------------------------------------------------------------------------------------- data
class MultiChannelBeads:
    """Beads of bead stacks seen in C channels (module docstring). Global beads: xy_ref [B, 2] (start positions,
    reference channel px), stack [B], cluster [B], amp [B, C] (detection amplitude per channel, 0 = not detected),
    faint [B]; global clusters: cl_stack [K], usable [K, C] bool (the cluster is fitted in channel c), why [K][C]
    (reason a channel is not used: 'frame', 'field_stop', 'box' or None). Per channel: channels[c] = BeadClusters of
    the clusters usable in c (or None), cl_gid[c] [K_c] (global cluster id of each), gid[c] [B_c] (global bead id of
    each bead of that BeadClusters, long tensor on the device). subset(cluster ids), channel(c) (one channel only)."""

    def __init__(self, channels, cl_gid, gid, xy_ref, stack, cluster, amp, faint, cl_stack, usable, why, patch,
                 nk, frame_shapes, index=None, device=None):
        self.channels, self.cl_gid, self.gid = channels, cl_gid, gid
        self.xy_ref, self.stack, self.cluster, self.amp, self.faint = xy_ref, stack, cluster, amp, faint
        self.cl_stack, self.usable, self.why = cl_stack, usable, why
        self.patch, self.nk, self.frame_shapes = int(patch), int(nk), frame_shapes
        self.index = list(range(len(channels))) if index is None else list(index)     # original channel numbers
        self.dev = torch.device(device) if device is not None else next(
            d.dev for d in channels if d is not None)

    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def n_beads(self):
        return len(self.xy_ref)

    @property
    def n_clusters(self):
        return len(self.cl_stack)

    @property
    def n_stacks(self):
        return int(max(self.cl_stack.max() + 1, 1)) if len(self.cl_stack) else 0

    @property
    def R(self):
        return next(d.R for d in self.channels if d is not None)

    def subset(self, ids):
        """the clusters ids (global) only; beads renumbered"""
        ids = np.asarray(ids, int)
        keep_b = np.isin(self.cluster, ids)
        bmap = -np.ones(self.n_beads, int)
        bmap[keep_b] = np.arange(keep_b.sum())
        cmap = -np.ones(self.n_clusters, int)
        cmap[ids] = np.arange(len(ids))
        chans, clg, gids = [], [], []
        for c, d in enumerate(self.channels):
            if d is None:
                chans.append(None); clg.append(np.zeros(0, int)); gids.append(None)
                continue
            loc = [i for i, g in enumerate(self.cl_gid[c]) if cmap[g] >= 0]
            if not loc:
                chans.append(None); clg.append(np.zeros(0, int)); gids.append(None)
                continue
            sub = d.subset(loc)
            chans.append(sub)
            clg.append(cmap[np.asarray(self.cl_gid[c])[loc]])
            g = np.concatenate([self.gid[c].cpu().numpy()[d.cl_bead[i]:d.cl_bead[i + 1]] for i in loc])
            gids.append(torch.as_tensor(bmap[g], device=self.dev))
        return MultiChannelBeads(chans, clg, gids, self.xy_ref[keep_b], self.stack[keep_b], cmap[self.cluster[keep_b]],
                                 self.amp[keep_b], self.faint[keep_b], self.cl_stack[ids], self.usable[ids],
                                 [self.why[i] for i in ids], self.patch, self.nk, self.frame_shapes, self.index,
                                 self.dev)

    def channel(self, c):
        """MultiChannelBeads of channel c alone (its usable clusters; beads renumbered)"""
        ids = np.nonzero(self.usable[:, c])[0]
        sub = self.subset(ids)
        return MultiChannelBeads([sub.channels[c]], [sub.cl_gid[c]], [sub.gid[c]], sub.xy_ref, sub.stack, sub.cluster,
                                 sub.amp[:, [c]], sub.faint, sub.cl_stack, sub.usable[:, [c]],
                                 [[w[c]] for w in sub.why], self.patch, self.nk, [self.frame_shapes[c]],
                                 [self.index[c]], self.dev)

    def planes(self, sl):
        return MultiChannelBeads([None if d is None else d.planes(sl) for d in self.channels], self.cl_gid, self.gid,
                                 self.xy_ref, self.stack, self.cluster, self.amp, self.faint, self.cl_stack,
                                 self.usable, self.why, self.patch,
                                 len(range(self.nk)[sl]) if isinstance(sl, slice) else len(sl), self.frame_shapes,
                                 self.index, self.dev)

    def summary(self):
        return dict(n_beads=int(self.n_beads), n_clusters=int(self.n_clusters),
                    clusters_per_channel=[int(u) for u in self.usable.sum(0)],
                    beads_per_channel=[0 if d is None else int(d.n_beads) for d in self.channels],
                    clusters_all_channels=int(self.usable.all(1).sum()))


def build_multichannel_beads(stacks, maps, patch, masks=None, dets=None, threshold=6.0, min_dist=3, faint=0.1,
                             merge_px=2.0, max_members=8, max_box=None, render_pad=0, device=None):
    """MultiChannelBeads of the stacks[c][s] ([nk, H, W] photons, channel crops) with the mappings maps[c]
    (ChannelMap, reference -> channel c). Detections (detect_channels, or dets) of every channel are mapped to the
    reference channel and merged (mutual nearest neighbours within merge_px); a bead is bright when it is bright in
    any channel (faint ones do not take part in the grouping and join a cluster only as non-growing members, as in
    find_bead_clusters). Beads whose patch x patch boxes overlap in any channel form a cluster; a cluster is used in
    channel c when all its bright members' boxes are inside that channel's frame and field stop (masks[c][s] or
    masks[c], bool [H, W]; None = no stop) and its box is at most max_box (default 3 patch); clusters with more than
    max_members bright beads are dropped."""
    from scipy.spatial import cKDTree
    C, S = len(stacks), len(stacks[0])
    dev = torch.device(device) if device is not None else (stacks[0][0].device if torch.is_tensor(stacks[0][0])
                                                           else torch.device('cpu'))
    if dets is None:
        dets = detect_channels(stacks, threshold, min_dist, faint)
    shapes = [tuple(stacks[c][0].shape[-2:]) for c in range(C)]
    max_box = 3 * patch if max_box is None else max_box
    h = patch // 2
    nk = int(stacks[0][0].shape[0])
    g_xy, g_st, g_amp, g_bright = [], [], [], []
    for s in range(S):
        xy = np.zeros((0, 2)); amp = np.zeros((0, C)); bright = np.zeros(0, bool)
        for c in range(C):
            d = dets[c][s]
            ref = maps[c].to_ref(np.asarray(d['xy'], np.float64)) if c > 0 else np.asarray(d['xy'], np.float64)
            ia, ib = match_pairs(ref, xy, merge_px) if len(xy) else (np.zeros(0, int), np.zeros(0, int))
            amp[ib, c] = d['amp'][ia]
            bright[ib] |= d['bright'][ia]
            new = np.setdiff1d(np.arange(len(ref)), ia)
            xy = np.concatenate([xy, ref[new]])
            a = np.zeros((len(new), C)); a[:, c] = d['amp'][new]
            amp = np.concatenate([amp, a]); bright = np.concatenate([bright, d['bright'][new]])
        g_xy.append(xy); g_amp.append(amp); g_bright.append(bright); g_st.append(np.full(len(xy), s))
    # fill undetected amplitudes from the other channels (median ratio)
    amp_all = np.concatenate(g_amp)
    both = (amp_all > 0).all(1)
    ratio = np.array([np.median(amp_all[both, c] / amp_all[both, 0]) if both.any() else 1.0 for c in range(C)])
    clusters, why_all = [], []
    beads = dict(xy=[], stack=[], amp=[], faint=[], cluster=[])
    for s in range(S):
        xy, amp, bright = g_xy[s], g_amp[s].copy(), g_bright[s]
        for i in range(len(xy)):
            if (amp[i] <= 0).any():
                k = int(np.argmax(amp[i] / ratio))
                amp[i, amp[i] <= 0] = amp[i, k] / ratio[k] * ratio[amp[i] <= 0]
        chpos = [maps[c].from_ref(xy) for c in range(C)]
        rch = [np.round(p).astype(int) for p in chpos]
        bi = np.nonzero(bright)[0]
        parent = np.arange(len(xy))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for c in range(C):
            if len(bi) < 2:
                break
            for i, j in cKDTree(rch[c][bi]).query_pairs(patch - 0.5, p=np.inf):
                a, b = root(bi[i]), root(bi[j])
                if a != b:
                    parent[b] = a
        roots = np.array([root(i) for i in range(len(xy))])
        members = {g: list(np.nonzero((roots == g) & bright)[0]) for g in np.unique(roots[bi])}
        # faint beads inside a member box (any channel) of a group join it as non-growing members
        faint_of = {g: [] for g in members}
        for f in np.nonzero(~bright)[0]:
            for g, mem in members.items():
                if any((np.abs(rch[c][mem] - rch[c][f]).max(1) <= h).any() for c in range(C)):
                    faint_of[g].append(f)
                    break
        for g, mem in members.items():
            usable, why = [], []
            for c in range(C):
                a0 = rch[c][mem] - h
                H, W = shapes[c]
                box = (int(a0[:, 0].min()), int(a0[:, 1].min()), int(a0[:, 0].max()) + patch,
                       int(a0[:, 1].max()) + patch)
                mk = None if masks is None or masks[c] is None else (masks[c][s] if isinstance(masks[c], (list, tuple))
                                                                       else masks[c])
                w = None
                if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
                    w = 'frame'
                elif mk is not None and not all(mk[y0:y0 + patch, x0:x0 + patch].all() for x0, y0 in a0):
                    w = 'field_stop'
                elif max(box[2] - box[0], box[3] - box[1]) > max_box:
                    w = 'box'
                usable.append(w is None)
                why.append(w)
            if len(mem) > max_members:
                usable = [False] * C
                why = ['members'] * C
            k = len(clusters)
            allm = list(mem) + list(faint_of[g])
            clusters.append(dict(stack=s, beads=allm, faint=[False] * len(mem) + [True] * len(faint_of[g]),
                                 usable=usable))
            why_all.append(why)
    # keep clusters used in at least one channel; global bead list
    kept = [k for k, c in enumerate(clusters) if any(c['usable'])]
    for kk, k in enumerate(kept):
        c = clusters[k]
        s = c['stack']
        for b, fa in zip(c['beads'], c['faint']):
            beads['xy'].append(g_xy[s][b]); beads['stack'].append(s); beads['amp'].append(g_amp[s][b])
            beads['faint'].append(fa); beads['cluster'].append(kk)
    xy_ref = np.array(beads['xy']).reshape(-1, 2)
    b_stack, b_cl = np.array(beads['stack'], int), np.array(beads['cluster'], int)
    b_amp, b_faint = np.array(beads['amp']).reshape(-1, C), np.array(beads['faint'], bool)
    for i in range(len(b_amp)):                                  # fill amplitudes (as above)
        if (b_amp[i] <= 0).any():
            k = int(np.argmax(b_amp[i] / ratio))
            b_amp[i, b_amp[i] <= 0] = b_amp[i, k] / ratio[k] * ratio[b_amp[i] <= 0]
    usable = np.array([clusters[k]['usable'] for k in kept], bool).reshape(-1, C)
    cl_stack = np.array([clusters[k]['stack'] for k in kept], int)
    why = [why_all[k] for k in kept]
    chans, clg, gids = [], [], []
    for c in range(C):
        cl_c, ids = [], []
        for kk in range(len(kept)):
            if not usable[kk, c]:
                continue
            bi = np.nonzero(b_cl == kk)[0]
            pc = maps[c].from_ref(xy_ref[bi])
            r = np.round(pc[~b_faint[bi]]).astype(int)
            a0 = r - h
            box = (int(a0[:, 0].min()), int(a0[:, 1].min()), int(a0[:, 0].max()) + patch, int(a0[:, 1].max()) + patch)
            cl_c.append(dict(stack=int(cl_stack[kk]), xy=pc, amp=b_amp[bi, c], box=box, faint=b_faint[bi],
                             gid=bi))
            ids.append(kk)
        if not cl_c:
            chans.append(None); clg.append(np.zeros(0, int)); gids.append(None)
            continue
        bc = BeadClusters.from_stacks(stacks[c], cl_c, patch, dev, render_pad)
        chans.append(bc)
        clg.append(np.array(ids, int))
        gids.append(torch.as_tensor(np.concatenate([d['gid'] for d in cl_c]), device=dev))
    return MultiChannelBeads(chans, clg, gids, xy_ref, b_stack, b_cl, b_amp, b_faint, cl_stack, usable, why, patch,
                             nk, shapes, None, dev)


# ---------------------------------------------------------------------------------------------------- fit
def _photons_init(d, amp):
    """per-bead starting photons of one channel's BeadClusters (as fit_dense_beads)"""
    dev = d.dev
    C, nk = d.n_clusters, d.nk
    with torch.no_grad():
        bg0 = torch.stack([d.d[:, d.cl_pix[c]:d.cl_pix[c + 1]].quantile(0.1, 1) for c in range(C)])
        flux = torch.zeros(C, nk, device=dev).index_add_(0, d.pix_cluster, (d.d - bg0[d.pix_cluster].T).T).clamp_min(100.0)
        a = torch.as_tensor(np.clip(amp, 1e-3, None), dtype=torch.float32, device=dev)
        share = a / torch.zeros(C, device=dev).index_add_(0, d.bead_cluster, a)[d.bead_cluster]
        I0 = flux.median(1).values[d.bead_cluster] * share
    return bg0, I0


def fit_multichannel_beads(data, focus_um, psfs, maps, iters=300, lr=3e-3, fit=('zcoef',), share=(), fields=None,
                           fit_mapping=True, map_mode='all', train=None, emitter_z_um=0.0, drift=True, drift_order=3,
                           focus_offset='bead', channel_focus=True, fit_channel_focus=True, bg='linear',
                           intensity='free', warmup=50,
                           heldout_iters=200, max_renders=3000, reject=None, reject_at=None, exclude=None,
                           lr_focus=1e-2, lr_drift=1e-2, lr_apod=1e-2, lr_field=None, lr_map=1e-2, lr_xy=1e-2,
                           log=None, stop=None):
    """Joint fit of one PSF per channel (VectorialPSF, or FieldPSF / fields[c] for a field-dependent one), the channel
    mappings and the per-bead / cluster / stack terms to MultiChannelBeads (module docstring). Poisson likelihood
    summed over the channels, Adam (as estimation.fit_dense_beads: the PSF parameters held for `warmup` iterations,
    cosine decay over the last 30 %).

    psfs: per channel the start model (copied; roisize = data.R); maps: per channel a ChannelMap (copied; channel 0
    of the data is the reference and should be the identity). fit: PSF parameters fitted in every channel ('zcoef',
    'NA', 'n_med', 'apod', 'field'); share: of 'NA', 'n_med', 'apod' those that are one parameter for all channels
    (else one per channel). fields: per channel a FieldZernikes (channel coordinates) or None. fit_mapping: fit the
    mappings of channels 1.. (map_mode 'all' / 'affine' / 'shift': which coefficients; ChannelMap.set_fit; Adam step
    lr_map px). channel_focus: a focus offset per channel (um, channel 0 fixed at 0; the chromatic focal shift; True:
    start 0, or start values [C]; fitted when fit_channel_focus; None / False: none). focus_offset: 'bead' (one per bead,
    shared by the channels), 'stack', 'tilt' or None. intensity: 'free' (photons per channel, bead and plane) or
    'ratio' (a shared photons per bead and plane times a per-bead intensity ratio per channel, channel 0 = 1).
    train [K] bool: clusters in the fit (default all); the others are refitted afterwards with the PSFs, mappings and
    channel focus fixed and scored per channel. exclude: clusters neither fitted nor scored. reject: at reject_at
    (default iters // 3) clusters whose chi2 in any channel is above reject x that channel's median leave the fit.

    Returns dict(psfs (per channel VectorialPSF, roisize R), fields (FieldZernikes or None per channel), maps (fitted
    ChannelMaps), xy_ref [B, 2], focus_offset [B], channel_focus [C], photons (per channel [B_c, nk]), drift,
    report (per channel: deviance_per_px, nll_per_px, chi2_median, heldout_*; totals; mapping), history, train,
    heldout, per-channel results 'channel' [c] = dict(mu, chi2, deviance, nll per cluster of that channel's
    BeadClusters))."""
    dev = data.dev
    C, B, K, S, nk = data.n_channels, data.n_beads, data.n_clusters, max(data.n_stacks, 1), data.nk
    R, P = data.R, data.patch
    focus = torch.as_tensor(focus_um, dtype=torch.float32, device=dev)
    assert len(focus) == nk, (len(focus), nk)
    fields = list(fields) if fields is not None else [None] * C
    vecs = []
    for c in range(C):
        v = psfs[c]
        if hasattr(v, 'field') and hasattr(v, 'base'):
            fields[c] = v.field if fields[c] is None else fields[c]
            v = v.base
        vecs.append(v.resized(R))
    for k in ('NA', 'n_med', 'apod'):
        if k in share:
            for v in vecs[1:]:
                setattr(v, k, getattr(vecs[0], k))                       # one Parameter for all channels
    for v in vecs:
        for k in ('zcoef', 'NA', 'n_med', 'apod'):
            getattr(v, k).requires_grad_(k in fit)
    fz = [None if f is None else f.copy().to(dev) for f in fields]
    for f in fz:
        if f is not None:
            f.coef.requires_grad_('field' in fit)
    maps = [m.copy().to(dev) for m in maps]
    fit_map = [bool(fit_mapping) and not maps[c].identity and c > 0 for c in range(C)]
    for c, m in enumerate(maps):
        m.inv.requires_grad_(fit_map[c])
        if fit_map[c]:
            m.set_fit(map_mode)
    train = np.ones(K, bool) if train is None else np.asarray(train, bool).copy()
    excl = np.zeros(K, bool)
    if exclude is not None:
        excl[np.asarray(exclude, int)] = True
    train &= ~excl

    # ---- parameters
    X = torch.as_tensor(data.xy_ref, dtype=torch.float64, device=dev).clone().requires_grad_(True)
    chans = []
    for c in range(C):
        d = data.channels[c]
        if d is None:
            chans.append(None)
            continue
        bg0, I0 = _photons_init(d, data.amp[data.gid[c].cpu().numpy(), c])
        ch = dict(d=d, gid=data.gid[c], clg=np.asarray(data.cl_gid[c], int), bgc=bg0.clone().requires_grad_(True),
                  I0=I0, center=(d.corner + P // 2).double(), fxy=d.xy_image)
        ch['bgg'] = torch.zeros(d.n_clusters, nk, 2, device=dev, requires_grad=True) if bg == 'linear' else None
        if intensity == 'free':
            ch['logI'] = torch.log(I0)[:, None].repeat(1, nk).requires_grad_(True)
        chans.append(ch)
    if intensity == 'ratio':
        I0g = torch.ones(B, device=dev)
        cnt = torch.zeros(B, device=dev)
        lr0 = torch.zeros(B, C, device=dev)
        for c, ch in enumerate(chans):
            if ch is not None:
                I0g.index_put_((ch['gid'],), ch['I0'])
        for c, ch in enumerate(chans):
            if ch is not None:
                lr0[ch['gid'], c] = torch.log(ch['I0'] / I0g[ch['gid']]).clamp(-5, 5)
        logIg = torch.log(I0g)[:, None].repeat(1, nk).requires_grad_(True)
        logR = lr0.clone().requires_grad_(True)
        rmask = torch.ones(B, C, device=dev)
        rmask[:, 0] = 0.0
        logR.register_hook(lambda g: g * rmask)
        with torch.no_grad():
            logR[:, 0] = 0.0
    elif intensity != 'free':
        raise ValueError(intensity)

    def photons(c, b0, b1):
        ch = chans[c]
        if intensity == 'free':
            return torch.exp(ch['logI'][b0:b1])
        g = ch['gid'][b0:b1]
        return torch.exp(logIg[g] + logR[g, c][:, None])

    psf_params, seen = [], set()
    for v in vecs:
        for k in fit:
            if k in ('zcoef', 'NA', 'n_med'):
                p = getattr(v, k)
                if id(p) not in seen:
                    psf_params.append(p); seen.add(id(p))
    groups = [dict(params=psf_params, lr=lr, name='psf'), dict(params=[X], lr=lr_xy)]
    if 'apod' in fit:
        ap = []
        for v in vecs:
            if id(v.apod) not in seen:
                ap.append(v.apod); seen.add(id(v.apod))
        groups.append(dict(params=ap, lr=lr_apod, name='psf'))
    if 'field' in fit and any(f is not None for f in fz):
        groups.append(dict(params=[f.coef for f in fz if f is not None], lr=lr if lr_field is None else lr_field,
                           name='psf'))
    int_params = [ch['logI'] for ch in chans if ch is not None] if intensity == 'free' else [logIg, logR]
    bg_params = [ch['bgc'] for ch in chans if ch is not None] + [ch['bgg'] for ch in chans
                                                                if ch is not None and ch['bgg'] is not None]
    groups += [dict(params=int_params, lr=1e-2), dict(params=bg_params, lr=0.3)]
    mparams = [maps[c].inv for c in range(C) if fit_map[c]]
    if mparams:
        groups.append(dict(params=mparams, lr=lr_map))
    kk = torch.arange(nk, device=dev, dtype=torch.float32) - (nk - 1) / 2
    tpow = torch.stack([(kk / max((nk - 1) / 2, 1.0)) ** (k + 1) for k in range(int(drift_order))], 0)
    dr = None
    if drift:
        dr = torch.zeros(S, int(drift_order), 2, device=dev, requires_grad=True)
        groups.append(dict(params=[dr], lr=lr_drift))
    fo = None
    bstack = torch.as_tensor(data.stack, device=dev)
    if focus_offset == 'stack':
        fo = torch.zeros(S, device=dev, requires_grad=True)
    elif focus_offset == 'tilt':
        fo = torch.zeros(S, 3, device=dev, requires_grad=True)
        H0, W0 = data.frame_shapes[0]
        fs = torch.tensor([W0 / 2, H0 / 2], device=dev)
        tilt_basis = torch.cat([torch.ones(B, 1, device=dev), (X.detach().float() - fs) / fs], 1)
    elif focus_offset == 'bead':
        fo = torch.zeros(B, device=dev, requires_grad=True)
    elif focus_offset is not None:
        raise ValueError(focus_offset)
    if fo is not None:
        groups.append(dict(params=[fo], lr=lr_focus))
    cf = None
    if channel_focus is not None and channel_focus is not False and C > 1:
        cf0 = torch.zeros(C) if channel_focus is True else (channel_focus.detach().float().cpu() if torch.is_tensor(
            channel_focus) else torch.as_tensor(np.asarray(channel_focus, np.float32)))
        cf = cf0.reshape(C).to(dev).clone().requires_grad_(bool(fit_channel_focus))
        if fit_channel_focus:
            cmask = torch.ones(C, device=dev)
            cmask[0] = 0.0
            cf.register_hook(lambda g: g * cmask)
            groups.append(dict(params=[cf], lr=lr_focus))
    groups = [g for g in groups if g['params']]
    opt = torch.optim.Adam(groups)
    for g in opt.param_groups:
        g['lr0'] = g['lr']

    def df(gid):
        if fo is None:
            return torch.zeros(len(gid), device=dev)
        if focus_offset == 'stack':
            return fo[bstack[gid]]
        if focus_offset == 'tilt':
            return (fo[bstack[gid]] * tilt_basis[gid]).sum(1)
        return fo[gid]

    def model(c, c0, c1):
        ch = chans[c]
        d = ch['d']
        b0, b1, p0, p1 = d.cl_bead[c0], d.cl_bead[c1], d.cl_pix[c0], d.cl_pix[c1]
        nb = b1 - b0
        g = ch['gid'][b0:b1]
        Xr = X[g][:, None].expand(-1, nk, -1)
        if dr is not None:
            Xr = Xr + torch.einsum('bkc,kn->bnc', dr[bstack[g]], tpow).double()
        xc = maps[c].from_ref(Xr.reshape(-1, 2)).reshape(nb, nk, 2) - ch['center'][b0:b1, None]
        foc = focus[None] + df(g)[:, None] + (0.0 if cf is None else cf[c])
        pos = torch.cat([xc.float(), torch.full((nb, nk, 1), float(emitter_z_um), device=dev)], -1).reshape(-1, 3)
        zoff = None
        if fz[c] is not None:
            zoff = fz[c].offsets(ch['fxy'][b0:b1], vecs[c].zidx)[:, None].expand(-1, nk, -1).reshape(nb * nk, -1)
        img = vecs[c](pos, foc.reshape(-1), zcoef_offset=zoff).reshape(nb, nk, R * R) * photons(c, b0, b1)[:, :, None]
        ix = d.idx[b0:b1]
        ix = torch.where(ix < 0, p1 - p0, ix - p0).reshape(-1)
        mu = torch.zeros(nk, p1 - p0 + 1, device=dev).index_add(
            1, ix, img.permute(1, 0, 2).reshape(nk, nb * R * R))[:, :-1]
        pc = d.pix_cluster[p0:p1]
        b = ch['bgc'][pc].T
        if ch['bgg'] is not None:
            b = b + (ch['bgg'][pc] * d.pix_uv[p0:p1, None, :]).sum(2).T
        return mu + b

    def chunks_of(mask_global):
        out = []
        for c, ch in enumerate(chans):
            if ch is None:
                continue
            act = mask_global[ch['clg']]
            out += [(c, c0, c1) for c0, c1 in _chunks(ch['d'], act, max_renders)]
        return out

    def stats(mask_global, keep_mu=False):
        """per channel: chi2 / deviance per cluster (channel-local index), nll per cluster, mu"""
        res = []
        with torch.no_grad():
            for c, ch in enumerate(chans):
                if ch is None:
                    res.append(None)
                    continue
                d = ch['d']
                Kc = d.n_clusters
                c2, dv, nl = np.full(Kc, np.nan), np.full(Kc, np.nan), np.full(Kc, np.nan)
                mu_all = torch.zeros_like(d.d) if keep_mu else None
                act = mask_global[ch['clg']]
                for c0, c1 in _chunks(d, act, max_renders):
                    mu = model(c, c0, c1)
                    q0 = d.cl_pix[c0]
                    if keep_mu:
                        mu_all[:, q0:d.cl_pix[c1]] = mu
                    for k in range(c0, c1):
                        a, b = d.cl_pix[k] - q0, d.cl_pix[k + 1] - q0
                        m, dd = mu[:, a:b].clamp_min(1e-6), d.d[:, d.cl_pix[k]:d.cl_pix[k + 1]]
                        c2[k] = float(((dd - m) ** 2 / m).mean())
                        dvv = 2 * (torch.where(dd > 0, dd * torch.log(dd.clamp_min(1e-9) / m), torch.zeros_like(dd))
                                   - (dd - m))
                        dv[k] = float(dvv.mean())
                        nl[k] = float(poisson_nll(dd.T, m.T).sum())
                res.append(dict(chi2=c2, deviance=dv, nll=nl, mu=mu_all,
                                npx=np.diff(d.cl_pix).astype(float) * nk))
        return res

    def penalty():
        return sum(f.penalty() for f in fz if f is not None)

    hist, t0, tot = [], time.time(), float('nan')
    reject_at = iters // 3 if reject_at is None else reject_at
    rejected = []
    chunks = chunks_of(train)
    n_flat = int(iters * 0.7)
    for it in range(iters):
        fac = 1.0 if it < n_flat else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (it - n_flat) / max(iters - n_flat, 1)))
        for g in opt.param_groups:
            g['lr'] = g['lr0'] * fac * (0.0 if (g.get('name') == 'psf' and it < warmup) else 1.0)
        opt.zero_grad()
        tot = 0.0
        for c, c0, c1 in chunks:
            d = chans[c]['d']
            l = poisson_nll(d.d[:, d.cl_pix[c0]:d.cl_pix[c1]].T, model(c, c0, c1).T).sum()
            l.backward()
            tot += float(l.detach())
        if 'field' in fit and any(f is not None for f in fz):
            pen = penalty()
            pen.backward()
            tot += float(pen.detach())
        opt.step()
        hist.append(dict(it=it, nll=tot))
        if log is not None:
            log(it, tot, time.time() - t0, vecs[0].zcoef.detach())
        if reject is not None and it == reject_at:
            st = stats(train)
            bad = np.zeros(K, bool)
            for c, ch in enumerate(chans):
                if ch is None:
                    continue
                c2 = st[c]['chi2']
                tr = train[ch['clg']]
                if tr.any():
                    med = float(np.nanmedian(c2[tr]))
                    bad[ch['clg'][tr & (c2 > reject * med)]] = True
            rejected = np.nonzero(bad & train)[0].tolist()
            train &= ~bad
            chunks = chunks_of(train)
        if stop is not None and stop():
            break
    for v in vecs:
        for k in ('zcoef', 'NA', 'n_med', 'apod'):
            getattr(v, k).requires_grad_(False)
    for f in fz:
        if f is not None:
            f.coef.requires_grad_(False)
    for m in maps:
        m.inv.requires_grad_(False)
    if cf is not None:
        cf = cf.detach()
    rej = np.zeros(K, bool)
    rej[rejected] = True
    st = stats(train | rej, keep_mu=True)
    # ---- held-out clusters: refit their own terms with the PSFs, mappings and channel focus fixed
    ho = ~train & ~rej & ~excl
    if ho.any():
        hparams = [dict(params=[X], lr=lr_xy), dict(params=int_params, lr=1e-2), dict(params=bg_params, lr=0.3)]
        if dr is not None:
            hparams.append(dict(params=[dr], lr=lr_drift))
        if fo is not None:
            hparams.append(dict(params=[fo], lr=lr_focus))
        opt2 = torch.optim.Adam(hparams)
        hchunks = chunks_of(ho)
        for it in range(heldout_iters):
            opt2.zero_grad()
            for c, c0, c1 in hchunks:
                d = chans[c]['d']
                poisson_nll(d.d[:, d.cl_pix[c0]:d.cl_pix[c1]].T, model(c, c0, c1).T).sum().backward()
            opt2.step()
            if stop is not None and stop():
                break
        sh = stats(ho, keep_mu=True)
        for c, ch in enumerate(chans):
            if ch is None:
                continue
            loc = ho[ch['clg']]
            for k in ('chi2', 'deviance', 'nll'):
                st[c][k][loc] = sh[c][k][loc]
            if loc.any():
                d = ch['d']
                hp = torch.as_tensor(np.concatenate([np.arange(d.cl_pix[k], d.cl_pix[k + 1])
                                                     for k in np.nonzero(loc)[0]]), device=dev)
                st[c]['mu'][:, hp] = sh[c]['mu'][:, hp]
    # ---- report
    chrep, ho_tot = [], None
    for c, ch in enumerate(chans):
        if ch is None:
            chrep.append(None)
            continue
        s_, tr, hl = st[c], train[ch['clg']], ho[ch['clg']]
        npx = s_['npx']
        r = dict(channel=int(data.index[c]), n_clusters=int(tr.sum()), n_beads=int(sum(
            ch['d'].cl_bead[k + 1] - ch['d'].cl_bead[k] for k in np.nonzero(tr)[0])),
            deviance_per_px=float((s_['deviance'][tr] * npx[tr]).sum() / max(npx[tr].sum(), 1)),
            nll_per_px=float(s_['nll'][tr].sum() / max(npx[tr].sum(), 1)),
            chi2_median=float(np.median(s_['chi2'][tr])) if tr.any() else float('nan'))
        if hl.any():
            ho_tot = (ho_tot or 0.0) + float(s_['nll'][hl].sum())
            r.update(heldout_deviance_per_px=float((s_['deviance'][hl] * npx[hl]).sum() / npx[hl].sum()),
                     heldout_nll_per_px=float(s_['nll'][hl].sum() / npx[hl].sum()),
                     heldout_n_clusters=int(hl.sum()),
                     heldout_n_beads=int(sum(ch['d'].cl_bead[k + 1] - ch['d'].cl_bead[k] for k in np.nonzero(hl)[0])))
        if 'NA' in fit:
            r['NA'] = float(vecs[c].NA)
        if 'apod' in fit:
            r['apod'] = vecs[c].apod.detach().cpu().tolist()
        if fz[c] is not None:
            r['field'] = dict(modes=fz[c].noll, orders=fz[c].orders, n_params=fz[c].n_params,
                              prior_sd=fz[c].prior_sd, penalty=float(fz[c].penalty()))
        chrep.append(r)
    rep = dict(method='multichannel_beads', n_channels=C, channels=chrep, iterations=len(hist), loss=tot,
               rejected_clusters=rejected, seconds=time.time() - t0, n_train_clusters=int(train.sum()),
               n_heldout_clusters=int(ho.sum()), shared=list(share), fit=list(fit),
               fit_mapping=[bool(f) for f in fit_map], intensity=intensity, heldout_nll_total=ho_tot)
    if cf is not None:
        rep['channel_focus_um'] = cf.detach().cpu().tolist()
    rep['mapping'] = [m.describe() if not m.identity else None for m in maps]
    for k in ('deviance_per_px', 'heldout_deviance_per_px', 'heldout_nll_per_px', 'nll_per_px'):
        vals = [(r[k], r.get('heldout_n_clusters' if 'heldout' in k else 'n_clusters', 0)) for r in chrep
                if r is not None and k in r]
        if vals:
            rep[k] = float(np.mean([v for v, _ in vals]))
    with torch.no_grad():
        out = dict(psfs=vecs, fields=fz, maps=maps, xy_ref=X.detach(), focus_offset=df(torch.arange(B, device=dev)).detach(),
                   channel_focus=None if cf is None else cf.detach(), drift=None if dr is None else dr.detach(),
                   photons=[None if ch is None else torch.cat([photons(c, 0, ch['d'].n_beads)]).detach()
                            for c, ch in enumerate(chans)],
                   gid=[None if ch is None else ch['gid'] for ch in chans],
                   channel=st, train=train, heldout=ho, report=rep, history=hist)
    return out


def multichannel_nll(data, focus_um, psfs, maps, iters=200, channel_focus=None, fields=None, fit_channel_focus=False,
                     **kw):
    """Score fixed PSFs, mappings and channel focus offsets (channel_focus [C] um, e.g. result['channel_focus'];
    None: 0; fit_channel_focus=True: fitted as well, e.g. for channels whose PSFs were fitted independently) on
    clusters: refit every cluster's own terms (shared positions and focus offsets, photons, backgrounds,
    drift) with the PSFs and mappings fixed. Returns the fit_multichannel_beads result;
    report['channels'][c]['deviance_per_px'] / ['nll_per_px'] per channel (the clusters are all 'training' clusters
    of this refit)."""
    kw = dict(kw)
    kw.pop('warmup', None)
    C = data.n_channels
    cf = None if C == 1 else (np.zeros(C) if channel_focus is None else
                              np.asarray(channel_focus.detach().cpu() if torch.is_tensor(channel_focus)
                                         else channel_focus, np.float32))
    return fit_multichannel_beads(data, focus_um, psfs, maps, iters=iters, fit=(), fit_mapping=False, warmup=0,
                                  channel_focus=cf, fit_channel_focus=fit_channel_focus, fields=fields, **kw)


def intensity_ratios(result, data):
    """per channel the median over beads seen in channel 0 and c of (photons summed over planes in c) / (in 0)"""
    ph = result['photons']
    gid = result['gid']
    tot = []
    for c in range(data.n_channels):
        t = np.full(data.n_beads, np.nan)
        if ph[c] is not None:
            t[gid[c].cpu().numpy()] = ph[c].sum(1).cpu().numpy()
        tot.append(t)
    return [1.0] + [float(np.nanmedian(tot[c] / tot[0])) for c in range(1, data.n_channels)]


# ---------------------------------------------------------------------------------------------------- simulation
@torch.no_grad()
def simulate_multichannel_beads(vecs, maps, focus_um, xy_ref, shape, fields=None, photons=40000.0, ratio=None,
                                bg=30.0, emitter_z_um=0.0, focus_offset=None, channel_focus=None, drift=None,
                                render_size=None, seed=0):
    """Bead stacks of C channels: beads at reference positions xy_ref [n, 2] seen in channel c at
    maps[c].from_ref(xy_ref) with the VectorialPSF vecs[c] (and the FieldZernikes fields[c] at the channel position),
    focus focus_um + channel_focus[c] + focus_offset [n]; photons [n] (or scalar) times ratio[c]; drift [nk, 2] px
    in the reference frame (mapped through the linear part of each map). Returns per channel the
    simulate_dense_bead_stack dict (d [nk, H, W] photons, mu, xy channel positions)."""
    from torch_psf.estimation.dense_beads import simulate_dense_bead_stack
    C = len(vecs)
    xy_ref = np.asarray(xy_ref, np.float64)
    H, W = shape
    out = []
    for c in range(C):
        xy = maps[c].from_ref(xy_ref)
        dr = None
        if drift is not None:
            A, _ = maps[c].linear()
            dr = np.asarray(drift, np.float64) @ A.T
        f = torch.as_tensor(focus_um, dtype=torch.float32) + (0.0 if channel_focus is None else float(channel_focus[c]))
        ph = photons if ratio is None else (np.asarray(photons, np.float64) * float(ratio[c]))
        out.append(simulate_dense_bead_stack(vecs[c], f.to(vecs[c].dev), xy, H, W, photons=ph, bg=bg,
                                             emitter_z_um=emitter_z_um, focus_offset=focus_offset, drift=dr,
                                             render_size=render_size, seed=seed + 101 * c,
                                             field=None if fields is None else fields[c]))
    return out
