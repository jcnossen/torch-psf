"""MultiChannelBeadEstimator: PSFs of C channels (multicolour / biplane / dual view on one camera) from bead z-stacks,
with shared bead positions, a fitted channel mapping and an optional field polynomial per channel
(torch_psf.estimation.multichannel_beads). The result is a ChannelsPSF (one VectorialPSF or FieldPSF per channel,
the ChannelMaps as its mappings) that save_zip writes as one psf.zip with C channels (mapping, crop, flip, intensity
ratio and field block per channel; docs/PSF_FORMAT.md, format unchanged).

    est = MultiChannelBeadEstimator(regions=[(0, 0, 475, 400), (475, 0, 950, 400)], names=['left', 'right'],
                                    optics=dict(pixel_um=0.097, NA=1.49, wavelength_um=0.67), camera=cam, holdout=1,
                                    fit_na=True, fit_apod=True, field=dict(order={5: 2, 6: 2, 11: 2}))
    psf = est.estimate(['beads.tif', 'beads2.tif'], dz_um=0.01)        # ChannelsPSF with mappings
    est.save_zip('psf.zip', camera=cam)                                 # one zip, both channels
    PSFZip.read('psf.zip').multichannel(field=True, mapping=True)      # the same model back
"""
import time
import numpy as np
import tifffile
import torch
from torch_psf.estimators import Estimator, _camera, _start_vectorial
from torch_psf.estimation import stage_focus


class MultiChannelBeadEstimator(Estimator):
    """PSF per channel from bead z-stacks of a split camera frame (module docstring).

    regions: per channel the crop (x0, y0, x1, y1) of the camera frame (channel crop coordinates are relative to it);
    names: channel names (default ch0, ch1, ...); channel 0 is the reference. start: one PSF (file / VectorialPSF) for
    all channels or one per channel; optics: Optics / dict (one, or one per channel, e.g. per-channel wavelengths).
    mapping_order: order of the channel mapping polynomial (1 = affine); fit_mapping: fit it jointly ('all' /
    'affine' / 'shift' coefficients: map_mode) from the registration of the bead detections (flips found
    automatically); joint: shared bead positions across the channels (False: every channel fitted on its own, the
    mapping only from the registration). share: of 'NA', 'apod', 'n_med' those that are one parameter for all
    channels. field: field polynomial per channel (field_from_spec dict, one for all channels or a list per channel):
    after the global fit a second fit from it with the polynomials (channel coordinates), kept when the held-out NLL
    summed over the channels is lower. field_stop: 'auto' (estimation.multichannel_beads.channel_stop_mask per
    channel and stack), False, or masks[c] (bool [H, W] per channel). holdout: index of a stack kept out of the fit
    (scored per channel), or holdout_fraction. Other settings as DenseBeadEstimator (patch, render_pad, plane_step,
    threshold, faint, max_members, max_box, reject, iters, warmup, drift, focus_offset, bg, intensity,
    emitter_z_um).

    estimate(stacks, dz_um) -> ChannelsPSF. report: registration (per channel: flip, pairs, rms), data summary,
    global / field fit reports (per channel deviance, held-out deviance / NLL, NA, apod, channel focus, mapping),
    chosen. data (MultiChannelBeads), result / result_field (fit_multichannel_beads dicts), maps_registered,
    channel_stacks, masks."""

    def __init__(self, regions, names=None, start=None, optics=None, camera=None, patch=None, render_pad=None,
                 sign=1.0, iters=300, warmup=50, holdout=None, holdout_fraction=0.0, holdout_seed=0,
                 field_stop='auto', threshold=6.0, min_dist=3, faint=0.1, max_members=8, max_box=None, merge_px=2.0,
                 drift=True, drift_order=3, focus_offset='bead', bg='linear', intensity='free', fit_na=False,
                 fit_n_med=False, fit_apod=False, share=(), mapping_order=1, fit_mapping=True, map_mode='all',
                 joint=True, channel_focus=True, emitter_z_um=0.0, plane_step=1, reject=None, extent_rel=0.02,
                 max_patch=31, zernike_indices=tuple(range(5, 22)), npupil=64, oversample_pupil=3,
                 max_renders=3000, field=None, lr=3e-3, lr_map=1e-2, device=None, progress=None, stop=None):
        super().__init__(device, progress, stop)
        self.regions = [tuple(int(v) for v in r) for r in regions]
        C = len(self.regions)
        self.names = [f'ch{c}' for c in range(C)] if names is None else list(names)
        self.start, self.optics, self.camera = start, optics, _camera(camera)
        self.patch, self.render_pad, self.sign, self.iters, self.warmup = patch, render_pad, sign, iters, warmup
        self.holdout, self.holdout_fraction, self.holdout_seed = holdout, holdout_fraction, holdout_seed
        self.field_stop, self.threshold, self.min_dist, self.faint = field_stop, threshold, min_dist, faint
        self.max_members, self.max_box, self.merge_px = max_members, max_box, merge_px
        self.fit_kw = dict(drift=drift, drift_order=drift_order, focus_offset=focus_offset, bg=bg,
                           intensity=intensity, emitter_z_um=float(emitter_z_um), max_renders=max_renders)
        self.fit = (('zcoef',) + (('NA',) if fit_na else ()) + (('n_med',) if fit_n_med else ())
                    + (('apod',) if fit_apod else ()))
        self.share, self.mapping_order, self.fit_mapping, self.map_mode = tuple(share), mapping_order, fit_mapping, map_mode
        self.joint, self.channel_focus, self.plane_step, self.reject = joint, channel_focus, plane_step, reject
        self.extent_rel, self.max_patch = extent_rel, max_patch
        self.zernike_indices, self.npupil, self.oversample_pupil = zernike_indices, npupil, oversample_pupil
        self.field, self.lr, self.lr_map = field, lr, lr_map

    # ------------------------------------------------------------------ input
    def _channel_stacks(self, stacks):
        """[c][s] channel crops (photons, every plane_step-th plane)"""
        if isinstance(stacks, (str, np.ndarray)) or torch.is_tensor(stacks):
            stacks = [stacks]
        out = [[] for _ in self.regions]
        for s in stacks:
            a = tifffile.imread(s) if isinstance(s, str) else s
            a = (a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a, np.float32))).to(self.dev).float()
            if self.camera is not None:
                a = self.camera.to_photons(a)
            a = a[::self.plane_step]
            for c, (x0, y0, x1, y1) in enumerate(self.regions):
                out[c].append(a[:, y0:y1, x0:x1].contiguous())
        return out

    def _starts(self, R):
        C = len(self.regions)
        st = self.start if isinstance(self.start, (list, tuple)) else [self.start] * C
        op = self.optics if isinstance(self.optics, (list, tuple)) else [self.optics] * C
        out = []
        for c in range(C):
            v = _start_vectorial(st[c], op[c], R, self.dev, self.zernike_indices, self.npupil, self.oversample_pupil)
            out.append(v if v is not st[c] else v.resized(R))
        return out

    def _field_spec(self, c):
        f = self.field
        return f[c] if isinstance(f, (list, tuple)) else f

    # ------------------------------------------------------------------ estimation
    def estimate(self, stacks, dz_um):
        """stacks: list of [nk, H, W] camera frames (arrays / tensors / TIFF paths; all channels side by side). Returns
        the ChannelsPSF (models at roisize patch, focus_um = the channel focus offset; mappings = the fitted maps)."""
        from torch_psf.estimation.dense_beads import psf_extent
        from torch_psf.estimation.multichannel_beads import (channel_stop_mask, detect_channels, register_channels,
                                                             build_multichannel_beads, fit_multichannel_beads)
        from torch_psf.multichannel import ChannelsPSF
        self._reset()
        C = len(self.regions)
        cs = self._channel_stacks(stacks)
        S = len(cs[0])
        nk_full = (cs[0][0].shape[0] - 1) * self.plane_step + 1
        focus = stage_focus(nk_full, dz_um, self.sign).to(self.dev)[::self.plane_step]
        vecs = self._starts(21)
        P = self.patch or psf_extent(vecs[0], focus, self.fit_kw['emitter_z_um'], rel=self.extent_rel,
                                     pmax=self.max_patch)
        pad = P // 3 if self.render_pad is None else int(self.render_pad)
        # field stops, detection, registration
        if isinstance(self.field_stop, (list, tuple)):
            masks = list(self.field_stop)
        elif self.field_stop == 'auto':
            masks = [[channel_stop_mask(cs[c][s]) for s in range(S)] for c in range(C)]
        else:
            masks = None
        dets = detect_channels(cs, self.threshold, self.min_dist, self.faint)
        shapes = [tuple(cs[c][0].shape[-2:]) for c in range(C)]
        maps, reg = register_channels(dets, shapes, order=self.mapping_order)
        self.maps_registered = [m.copy() for m in maps]
        data = build_multichannel_beads(cs, maps, P, masks, dets=dets, merge_px=self.merge_px,
                                        max_members=self.max_members, max_box=self.max_box, render_pad=pad,
                                        device=self.dev)
        if data.n_clusters == 0:
            raise ValueError('no usable bead clusters found')
        train = np.ones(data.n_clusters, bool)
        if self.holdout is not None:
            train = data.cl_stack != self.holdout
        elif self.holdout_fraction > 0:
            rng = np.random.default_rng(self.holdout_seed)
            train[rng.permutation(data.n_clusters)[:int(round(self.holdout_fraction * data.n_clusters))]] = False
        self.report.update(n_channels=C, names=self.names, regions=self.regions, n_stacks=S, n_planes=len(focus),
                           dz_um=dz_um * self.plane_step, patch=P, render_pad=pad, data=data.summary(),
                           registration=[None if r is None else {k: v for k, v in r.items()
                                                                  if k in ('flip', 'n_pairs', 'rms_px', 'order',
                                                                           'coarse_shift', 'describe')} for r in reg],
                           field_stop=[None if masks is None else float(np.mean([m.mean() for m in masks[c]]))
                                       for c in range(C)])
        self.data, self.focus_um, self.channel_stacks, self.masks, self.detections = data, focus, cs, masks, dets
        kw = dict(self.fit_kw, lr=self.lr, lr_map=self.lr_map, warmup=self.warmup, reject=self.reject)
        res = self._fit(data, focus, vecs, maps, train, None, kw, 'global')
        self.result = res
        self.report['global'] = self._summ(res)
        chosen = res
        self.result_field = None
        if self.field and not self._stopped():
            from torch_psf.field import field_from_spec
            fzs = []
            for c in range(C):
                d = data.channels[c]
                if d is None:
                    fzs.append(None)
                    continue
                btr = torch.as_tensor(res['train'][data.cl_gid[c]], device=self.dev)[d.bead_cluster]
                fz, _ = field_from_spec(self._field_spec(c), res['psfs'][c].zidx, d.xy_image[btr], device=self.dev)
                fzs.append(fz)
            it_f = (int(self.field.get('iters', self.iters)) if isinstance(self.field, dict) else self.iters)
            kwf = dict(kw, reject=None, exclude=res['report']['rejected_clusters'] or None)
            rf = self._fit(data, focus, res['psfs'], res['maps'], res['train'] | np.isin(
                np.arange(data.n_clusters), res['report']['rejected_clusters']), fzs, kwf, 'field', iters=it_f,
                channel_focus=res['channel_focus'])
            self.result_field = rf
            self.report['field'] = self._summ(rf)
            hg, hf = res['report'].get('heldout_nll_total'), rf['report'].get('heldout_nll_total')
            better = hg is None or hf is None or hf < hg
            self.report['field']['kept'] = bool(better)
            if better:
                chosen = rf
        psf = self.model_of(chosen)
        self.intensity_ratio = psf.intensity_ratio.cpu().tolist()
        self.report['intensity_ratio'] = self.intensity_ratio
        self.report['chosen'] = 'field' if chosen is self.result_field else 'global'
        self.report['seconds'] = time.time() - self._t0
        psf.provenance = dict(estimator='MultiChannelBeadEstimator', dz_um=float(dz_um), report=self.report)
        for m in psf.models:
            m.provenance = psf.provenance
        self.psf = psf
        return psf

    def model_of(self, r):
        """ChannelsPSF of a fit result (self.result: the global fit, self.result_field: the field fit): per channel
        the VectorialPSF at roisize patch with focus_um = the channel focus offset (or its FieldPSF), the fitted
        mappings, the intensity ratios of the fitted photons"""
        from torch_psf.multichannel import ChannelsPSF
        from torch_psf.field import FieldPSF
        P = self.report['patch']
        cf = r['channel_focus']
        models = []
        for c in range(len(self.regions)):
            v = r['psfs'][c].resized(P)
            v.focus_um = 0.0 if cf is None else float(cf[c])
            models.append(v if r['fields'][c] is None else FieldPSF(v, r['fields'][c]))
        psf = ChannelsPSF(models, names=self.names, mappings=[m.copy() for m in r['maps']])
        psf.intensity_ratio = torch.as_tensor(self._ratios(r, self.data), dtype=torch.float32, device=self.dev)
        psf.provenance = dict(estimator='MultiChannelBeadEstimator', report=self.report)
        return psf

    def _fit(self, data, focus, vecs, maps, train, fields, kw, stage, iters=None, channel_focus=True):
        from torch_psf.estimation.multichannel_beads import fit_multichannel_beads
        C = data.n_channels
        fit = self.fit + (('field',) if fields is not None else ())
        it = self.iters if iters is None else iters
        if self.joint or C == 1:
            r = fit_multichannel_beads(data, focus, vecs, maps, iters=it, fit=fit, share=self.share, fields=fields,
                                       fit_mapping=self.fit_mapping, map_mode=self.map_mode, train=train,
                                       channel_focus=(channel_focus if self.channel_focus else None),
                                       log=self._log(f'multichannel {stage}'), stop=self.stop, **kw)
            return r
        # independent channels: one fit per channel (bead positions free per channel, the mapping fixed)
        parts = []
        for c in range(C):
            dc = data.channel(c)
            ids = np.nonzero(data.usable[:, c])[0]
            ex = kw.get('exclude')
            kwc = dict(kw)
            if ex is not None:
                kwc['exclude'] = [int(np.nonzero(ids == e)[0][0]) for e in ex if e in ids] or None
            r = fit_multichannel_beads(dc, focus, [vecs[c]], [maps[c]], iters=it, fit=fit, share=(),
                                       fields=None if fields is None else [fields[c]], fit_mapping=False,
                                       train=np.asarray(train)[ids], channel_focus=None,
                                       log=self._log(f'channel {c} {stage}'), stop=self.stop, **kwc)
            parts.append((ids, r))
        return _combine(parts, data, maps)

    @staticmethod
    def _summ(r):
        rep = r['report']
        keys = ('deviance_per_px', 'nll_per_px', 'chi2_median', 'heldout_deviance_per_px', 'heldout_nll_per_px',
                'heldout_n_clusters', 'heldout_n_beads', 'n_clusters', 'n_beads', 'NA', 'apod', 'field')
        return dict(channels=[None if c is None else {k: c[k] for k in keys if k in c} for c in rep['channels']],
                    mapping=rep.get('mapping'), channel_focus_um=rep.get('channel_focus_um'),
                    rejected_clusters=rep.get('rejected_clusters'), heldout_nll_total=rep.get('heldout_nll_total'),
                    seconds=rep.get('seconds'))

    @staticmethod
    def _ratios(r, data):
        from torch_psf.estimation.multichannel_beads import intensity_ratios
        try:
            return intensity_ratios(r, data)
        except Exception:
            return [1.0] * data.n_channels

    # ------------------------------------------------------------------ output
    def channel_entries(self, psf=None):
        """psf.zip channel extras of the result: name, crop (the region), mapping block (ChannelMap.to_block over the
        channel crop), flip, intensity_ratio"""
        psf = psf or self.psf
        out = []
        for c, (x0, y0, x1, y1) in enumerate(self.regions):
            m = psf.mappings[c]
            ext = (0, 0, x1 - x0 - 1, y1 - y0 - 1)
            out.append(dict(name=self.names[c], crop=dict(x0=x0, y0=y0, w=x1 - x0, h=y1 - y0),
                            mapping=m.to_block(c, extent=ext), flip=m.flip() if not m.identity else 'none',
                            intensity_ratio=float(psf.intensity_ratio[c])))
        return out

    def save_zip(self, path, camera=None, psf=None, **kw):
        """io.save_zip of the result (or of psf, e.g. model_of(result_field)): one psf.zip with the C channels
        (model, mapping, crop, flip, intensity ratio, field block)"""
        from torch_psf.io import save_zip
        psf = psf or self.psf
        return save_zip(path, list(psf.models), camera=camera or self.camera, channels=self.channel_entries(psf),
                        provenance=psf.provenance, **kw)


def _combine(parts, data, maps):
    """fit_multichannel_beads results of single channels -> one result dict of the same layout (per channel lists)"""
    C = data.n_channels
    psfs, fields, chans, chrep, photons, gid = [], [], [], [], [], []
    tot, any_ = 0.0, False
    for c, (ids, r) in enumerate(parts):
        psfs.append(r['psfs'][0]); fields.append(r['fields'][0]); chans.append(r['channel'][0])
        rr = r['report']['channels'][0]
        if rr is not None:
            rr = dict(rr, channel=c)
        chrep.append(rr)
        photons.append(r['photons'][0])
        gid.append(None if r['gid'][0] is None else torch.as_tensor(
            np.nonzero(np.isin(data.cluster, ids))[0], device=data.dev)[r['gid'][0]])
        h = r['report'].get('heldout_nll_total')
        if h is not None:
            tot += h
            any_ = True
    rep = dict(method='multichannel_beads (independent channels)', n_channels=C, channels=chrep,
               rejected_clusters=sorted({int(parts[c][0][k]) for c in range(C)
                                         for k in parts[c][1]['report']['rejected_clusters']}),
               heldout_nll_total=tot if any_ else None,
               seconds=float(sum(p[1]['report']['seconds'] for p in parts)))
    train = np.zeros(data.n_clusters, bool)
    for ids, r in parts:
        train[ids] |= r['train']
    # the mapping from the fitted bead positions of the independent fits (beads fitted in channel 0 and channel c)
    xy = []
    for c, (ids, r) in enumerate(parts):
        g = np.nonzero(np.isin(data.cluster, ids))[0]
        p = np.full((data.n_beads, 2), np.nan)
        p[g] = maps[c].from_ref(r['xy_ref'].cpu().numpy()) if r['xy_ref'] is not None else np.nan
        xy.append(p)
    new_maps, fit_info = [maps[0].copy()], [None]
    for c in range(1, C):
        ok = np.isfinite(xy[0]).all(1) & np.isfinite(xy[c]).all(1)
        if ok.sum() >= max(6, 2 * maps[c].inv.shape[0]):
            m = type(maps[c]).fit_points(xy[0][ok], xy[c][ok], maps[c].order, maps[c].norm, c, device=maps[c].device)
            res = m.from_ref(xy[0][ok]) - xy[c][ok]
            res0 = maps[c].from_ref(xy[0][ok]) - xy[c][ok]
            fit_info.append(dict(n_beads=int(ok.sum()), rms_px=float(np.sqrt((res ** 2).sum(1).mean())),
                                 rms_px_registration=float(np.sqrt((res0 ** 2).sum(1).mean()))))
            new_maps.append(m)
        else:
            new_maps.append(maps[c].copy())
            fit_info.append(None)
    rep['mapping'] = [m.describe() if not m.identity else None for m in new_maps]
    rep['mapping_from_positions'] = fit_info
    return dict(psfs=psfs, fields=fields, maps=new_maps, channel=chans, report=rep, photons=photons,
                gid=gid, channel_focus=None, train=train, parts=parts, xy_channels=xy)
