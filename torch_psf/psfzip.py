"""psf.zip: the shared PSF export (format 'torch-psf', version 0.2.0; the specification is docs/PSF_FORMAT.md).

One zip holds every channel of a setup:
    psf.json         format / version, optics, camera, provenance, emitter_params, channels (model, arrays, mapping to
                     the reference channel, field dependence, crop / flip / intensity ratio), source, patterns
    <owner>/<name>.npy   the arrays (pupil maps, spline knots, camera maps), referenced by member name from psf.json
    zstack.tif       the model rendered through z (ImageJ hyperstack, float32, axes ZCYX)
    preview.png      xy slices through z and an xz section
    fit_report.pdf   an overview of the estimation (torch_psf.psfreport); the model pages only without estimator data

    save_zip('psf.zip', psf, camera=cam, report=est)       # est: the ZStackEstimator / InSituEstimator that made psf
    psf = PSF.load('psf.zip')                               # single-channel or coupled (4Pi) file
    pz = PSFZip.read('psf.zip'); pz.channels[1].to_ref(x, y); pz.channels[0].model_at(x, y)

The reader turns a single-channel (or coupled) zip into the 0.1.0 json dict of that model (read_legacy), so every
model class reads it with its own _from_file; this module therefore needs no model class (the registry of PSF
supplies them) and PSF.read_file imports it lazily.
"""
import copy
import dataclasses
import io as _io
import json
import warnings
import zipfile
import numpy as np
import torch
from torch_psf.psf import PSF, FORMAT
from torch_psf.dataset import _plain

ZIP_VERSION = '0.2.0'
JSON_NAME, ZSTACK_NAME, PREVIEW_NAME, REPORT_NAME = 'psf.json', 'zstack.tif', 'preview.png', 'fit_report.pdf'
COUPLED = ('fourpi', 'fourpi_spline')
UNITS = dict(x='px', y='px', z='um', I='photons', bg='photons / px', phi='rad')


def is_psf_zip(path):
    """True if path (not a parsed dict) is a zip file."""
    if isinstance(path, dict):
        return False
    try:
        return zipfile.is_zipfile(str(path))
    except OSError:
        return False


# ====================================================================================================== features
def poly_features(x, y, order, norm):
    """PolyMap features u^i v^j for i + j <= order (i outer loop, j inner), u = (x - cx) / sx, v = (y - cy) / sy;
    norm = (cx, cy, sx, sy). x, y: arrays of one shape -> [..., n_feat] (float64)."""
    cx, cy, sx, sy = (float(v) for v in norm)
    u, v = (np.asarray(x, float) - cx) / sx, (np.asarray(y, float) - cy) / sy
    return np.stack([u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i)], -1)


def n_features(order):
    return (order + 1) * (order + 2) // 2


def _feats_norm(d, crop):
    """(cx, cy, sx, sy) of a block with feats_norm, else the crop centre (W / 2, H / 2, W / 2, H / 2) as SIMCODE's
    PolyMap.from_dict(size=(W, H))."""
    n = d.get('feats_norm')
    if n:
        return (n['cx'], n['cy'], n['sx'], n['sy'])
    if crop and crop.get('w') and crop.get('h'):
        W, H = float(crop['w']), float(crop['h'])
        return (W / 2, H / 2, W / 2, H / 2)
    raise ValueError("a 'poly' block needs feats_norm (or the channel crop w / h)")


def _array(v, arrays):
    """inline list or the member name of an .npy in the zip -> numpy"""
    if isinstance(v, str):
        return np.asarray(arrays[v])
    return np.asarray(v, float)


# ====================================================================================================== mapping
class Mapping:
    """Coordinates of channel `index` (channel crop px: x = column, y = row, pixel centres at integers) <-> the
    reference channel 0. Block types: identity, affine ({matrix: 2 x 3}: channel -> reference), poly (SIMCODE PolyMap
    dict: order, fwd {c: [nf, 2]} channel c -> 0, inv {c: [nf, 2]} 0 -> c, feats_norm), grid (fwd / inv: member names
    of [2, H, W] displacement fields (dx, dy) sampled on the channel / reference pixel grid)."""

    def __init__(self, d=None, index=0, arrays=None, crop=None):
        self.d = dict(d or {'type': 'identity'})
        self.index = int(index)
        t = self.d.get('type') or ('poly' if 'fwd' in self.d and 'order' in self.d else 'identity')
        self.type = t
        if t == 'poly':
            fwd, inv = self.d['fwd'], self.d['inv']
            key = str(index) if str(index) in fwd else (next(iter(fwd)) if len(fwd) == 1 else None)
            if key is None:                      # channel 0 of a SIMCODE map (no own entry): identity
                if index != 0:
                    raise ValueError(f'poly mapping without an entry for channel {index}')
                self.type = 'identity'
            else:
                self.order = int(self.d['order'])
                self.norm = _feats_norm(self.d, crop)
                self.cf, self.ci = np.asarray(fwd[key], float), np.asarray(inv[key], float)
        elif t == 'affine':
            A = np.asarray(self.d['matrix'], float).reshape(2, 3)
            self.A = A
            M = np.vstack([A, [0, 0, 1]])
            self.Ainv = np.linalg.inv(M)[:2]
        elif t == 'grid':
            self.gf = np.asarray(arrays[self.d['fwd']], float)
            self.gi = None if not self.d.get('inv') else np.asarray(arrays[self.d['inv']], float)
        elif t != 'identity':
            raise ValueError(f'unknown mapping type {t!r}')

    @staticmethod
    def _sample(g, x, y):
        from scipy.ndimage import map_coordinates
        c = np.stack([np.ravel(y), np.ravel(x)])
        return [map_coordinates(g[k], c, order=1, mode='nearest').reshape(np.shape(x)) for k in range(2)]

    def to_ref(self, x, y):
        """channel coordinates -> reference-channel coordinates (x0, y0)"""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.type == 'identity':
            return x, y
        if self.type == 'affine':
            return self.A[0, 0] * x + self.A[0, 1] * y + self.A[0, 2], self.A[1, 0] * x + self.A[1, 1] * y + self.A[1, 2]
        if self.type == 'poly':
            F = poly_features(x, y, self.order, self.norm) @ self.cf
            return F[..., 0], F[..., 1]
        dx, dy = self._sample(self.gf, x, y)
        return x + dx, y + dy

    def from_ref(self, x0, y0):
        """reference-channel coordinates -> channel coordinates"""
        x0, y0 = np.asarray(x0, float), np.asarray(y0, float)
        if self.type == 'identity':
            return x0, y0
        if self.type == 'affine':
            B = self.Ainv
            return B[0, 0] * x0 + B[0, 1] * y0 + B[0, 2], B[1, 0] * x0 + B[1, 1] * y0 + B[1, 2]
        if self.type == 'poly':
            F = poly_features(x0, y0, self.order, self.norm) @ self.ci
            return F[..., 0], F[..., 1]
        if self.gi is not None:
            dx, dy = self._sample(self.gi, x0, y0)
            return x0 + dx, y0 + dy
        x, y = x0.copy(), y0.copy()                # fixed-point inversion of the forward field
        for _ in range(20):
            dx, dy = self._sample(self.gf, x, y)
            x, y = x0 - dx, y0 - dy
        return x, y


# ====================================================================================================== field
class FieldZernikePoly:
    """Field dependence 'zernike_poly': waves of the Noll indices zernike_noll as polynomials in the channel
    coordinates, coef [n_zernike, n_feat] (PolyMap features, feats_norm). The channel model's zernike_waves are the
    value at reference_point (default the feats_norm centre, where the polynomial is coef[:, 0])."""

    def __init__(self, d, arrays=None, crop=None):
        self.d = d
        self.noll = [int(j) for j in d['zernike_noll']]
        self.order = int(d['order'])
        self.norm = _feats_norm(d, crop)
        self.coef = _array(d['coef'], arrays or {}).reshape(len(self.noll), n_features(self.order))
        rp = d.get('reference_point')
        self.ref = (self.norm[0], self.norm[1]) if not rp else (float(rp['x']), float(rp['y']))

    def zernike_at(self, x, y):
        """[..., n_zernike] waves at channel coordinates (x, y)."""
        return poly_features(x, y, self.order, self.norm) @ self.coef.T

    def model_at(self, model, x, y):
        """A copy of the vectorial model with the Zernikes of this block evaluated at the field point (x, y)."""
        if not hasattr(model, 'zidx'):
            raise TypeError(f'zernike_poly field dependence needs a vectorial model, not {type(model).__name__}')
        m = model.resized(model.roisize)
        w = self.zernike_at(float(x), float(y))
        with torch.no_grad():
            for j, v in zip(self.noll, w):
                m.zcoef[m.zidx.index(j)] = float(v)
        return m


def _field(d, arrays, crop):
    if not d:
        return None
    if d.get('type') == 'zernike_poly':
        return FieldZernikePoly(d, arrays, crop)
    return d                                   # 'grid' (spec only) and unknown types: the raw block


# ====================================================================================================== reading
def _check_version(meta):
    if meta.get('format') != FORMAT:
        raise ValueError(f"not a torch-psf psf.json (format {meta.get('format')!r})")
    v = str(meta.get('version', '0'))
    major_minor = tuple(int(p) for p in v.split('.')[:2])
    if major_minor != (0, 2):
        raise ValueError(f'psf.zip version {v}: this reader supports 0.2.x')


def read_zip(path):
    """(psf.json dict, {member name: numpy array} of every .npy in the zip)."""
    with zipfile.ZipFile(str(path)) as z:
        meta = json.loads(z.read(JSON_NAME).decode('utf-8'))
        arrays = {n: np.load(_io.BytesIO(z.read(n)), allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    _check_version(meta)
    return meta, arrays


def _arrays_of(block, arrays):
    return {k: arrays[m] for k, m in (block.get('arrays') or {}).items()}


def _channel_index(meta, channel):
    names = [c['name'] for c in meta['channels']]
    if isinstance(channel, str):
        return names.index(channel)
    return int(channel)


def legacy(meta, arrays, channel=None):
    """(0.1.0-style json dict, arrays, type) of the single-channel / coupled model of a psf.json, or of one channel
    (index or name) of an independent multi-channel file: what PSF.read_file returns for a json + npz."""
    chans = meta['channels']
    types = {c['type'] for c in chans}
    src = meta.get('source')
    if types == {'phase_spline'}:
        g = dict(chans[0]['model']['spline'], n_channels=len(chans), tau=[c['model'].get('tau') for c in chans])
        if any(t is None for t in g['tau']):
            g['tau'] = None
        knots = np.stack([np.stack([arrays[c['arrays'][k]] for c in chans]) for k in ('inc', 'cr', 'ci')])
        d, arr, typ = dict(fourpi_spline=g), dict(knots=knots), 'fourpi_spline'
        if src and src.get('type') == 'fourpi':
            d.update({k: v for k, v in src['model'].items() if k == 'fourpi'})
    elif types == {'source_output'}:
        d, arr, typ = dict(src['model']), _arrays_of(src, arrays), src['type']
    else:
        if channel is None and len(chans) > 1:
            raise ValueError(f'{len(chans)} independent channels: read one with torch_psf.io.load(path, channel=c) '
                             f'or all with torch_psf.psfzip.PSFZip.read(path)')
        ch = chans[_channel_index(meta, channel or 0)]
        d, arr, typ = dict(ch['model']), _arrays_of(ch, arrays), ch['type']
        if ch.get('source'):
            s = ch['source']
            d = {**s['model'], **d}
            arr = {**_arrays_of(s, arrays), **arr}
    if meta.get('optics'):
        d.setdefault('optics', meta['optics'])
    for k in ('camera', 'provenance'):
        if meta.get(k) is not None:
            d[k] = meta[k]
    d.update(format=FORMAT, version=meta['version'], type=typ)
    return d, (arr or None), typ


def read_legacy(path, channel=None):
    """PSF.read_file for a psf.zip."""
    meta, arrays = read_zip(path)
    return legacy(meta, arrays, channel)


def camera_from_meta(meta, arrays):
    """Camera of a psf.json (per-pixel maps from the zip), or None."""
    from torch_psf.camera import Camera
    c = meta.get('camera')
    if not c:
        return None
    cam = Camera.from_dict({k: v for k, v in c.items() if k not in ('arrays', 'maps')}, load_maps=False)
    for k, m in (c.get('arrays') or {}).items():
        setattr(cam, k, np.asarray(arrays[m], np.float32))
    return cam


def _model_from_block(block, arrays, roisize=None, device=None):
    d = dict(block['model'])
    return PSF._registry[block['type']]._from_file(d, _arrays_of(block, arrays), roisize=roisize, device=device)


@dataclasses.dataclass
class Channel:
    """One channel of a psf.zip. model: its PSF (None for the outputs of a coupled model: then PSFZip.psf renders all
    channels), source: the model it was derived from (e.g. the vectorial model of a spline) or None, mapping:
    Mapping to the reference channel 0, field: FieldZernikePoly / raw field block / None, entry: the psf.json entry."""
    name: str
    index: int
    type: str
    entry: dict
    model: object = None
    source: object = None
    mapping: Mapping = None
    field: object = None

    @property
    def crop(self):
        return self.entry.get('crop')

    @property
    def intensity_ratio(self):
        return float(self.entry.get('intensity_ratio', 1.0))

    def to_ref(self, x, y):
        return self.mapping.to_ref(x, y)

    def from_ref(self, x0, y0):
        return self.mapping.from_ref(x0, y0)

    def model_at(self, x, y):
        """The channel model at the field point (x, y) (channel coordinates): with a zernike_poly field block the
        vectorial model (the channel model, or its source for a spline) with the Zernikes evaluated there; without
        field dependence the channel model itself."""
        if self.field is None:
            return self.model
        if not isinstance(self.field, FieldZernikePoly):
            raise NotImplementedError(f"field type {self.field.get('type')!r} is not evaluated by this reader")
        base = self.model if hasattr(self.model, 'zidx') else self.source
        return self.field.model_at(base, x, y)


@dataclasses.dataclass
class PSFZip:
    """A read psf.zip: meta (psf.json), psf (the model of a single-channel or coupled file; None for independent
    multi-channel files), channels (list of Channel), camera, arrays ({member: numpy})."""
    meta: dict
    psf: object
    channels: list
    camera: object
    arrays: dict

    @classmethod
    def read(cls, path, roisize=None, device=None):
        meta, arrays = read_zip(path)
        chans = meta['channels']
        types = {c['type'] for c in chans}
        coupled = types <= {'phase_spline', 'source_output'}
        prov = meta.get('provenance') or None
        psf = None
        if coupled or len(chans) == 1:
            d, arr, typ = legacy(meta, arrays)
            psf = PSF._registry[typ]._from_file(d, arr, roisize=roisize, device=device)
            psf.provenance = prov
        out = []
        for i, c in enumerate(chans):
            model = source = None
            if not coupled:
                if psf is not None:
                    model = psf
                else:
                    d, arr, typ = legacy(meta, arrays, i)
                    model = PSF._registry[typ]._from_file(d, arr, roisize=roisize, device=device)
                    model.provenance = prov
                if c.get('source'):
                    source = _model_from_block(c['source'], arrays, device=device)
            out.append(Channel(name=c['name'], index=i, type=c['type'], entry=c, model=model, source=source,
                               mapping=Mapping(c.get('mapping'), i, arrays, c.get('crop')),
                               field=_field(c.get('field'), arrays, c.get('crop'))))
        return cls(meta=meta, psf=psf, channels=out, camera=camera_from_meta(meta, arrays), arrays=arrays)

    def channel(self, c):
        return self.channels[_channel_index(self.meta, c)]


# ====================================================================================================== rendering
def _channel_models(psf):
    return list(psf) if isinstance(psf, (list, tuple)) else [psf]


@torch.no_grad()
def render_zstack(psf, planes=25, sample_px=1.0, chunk=64):
    """The model rendered at x = y = 0 through z: (stack [Z, C, Y, X] float32 numpy, z [Z] um). planes: number of
    planes evenly over the z range of the (first) model, ends included (one plane at z = 0 for a 2D model).
    sample_px: lateral sample spacing in camera px, 1 / k for an integer k: sample (m, n) holds the pixel-integrated
    PSF of a camera pixel centred at ((n - (k P - 1) / 2) / k, (m - (k P - 1) / 2) / k) px from the emitter (rendered
    with k x k sub-pixel emitter shifts). psf: one model (a coupled model gives its C channels; 4Pi at phi = 0) or a
    list of independent channel models (rendered at the largest ROI size)."""
    k = int(round(1.0 / float(sample_px)))
    if k < 1 or abs(k * float(sample_px) - 1) > 1e-6:
        raise ValueError(f'sample_px must be 1 / k for an integer k >= 1, not {sample_px}')
    models = _channel_models(psf)
    P = max(int(m.roisize) for m in models)
    models = [m if m.roisize == P else m.resized(P) for m in models]
    zr = models[0].z_range
    zs = np.zeros(1) if zr is None else np.linspace(float(zr[0]), float(zr[1]), int(planes))
    a = (np.arange(k) - (k - 1) / 2) / k
    Zg, B, A = np.meshgrid(zs, a, a, indexing='ij')
    pos0 = torch.tensor(np.stack([A.ravel(), B.ravel(), Zg.ravel()], 1), dtype=torch.float32)
    outs = []
    for m in models:
        pos = pos0.to(m.dev)
        parts = []
        for p in pos.split(chunk):
            if m.type_name in COUPLED:
                r = m.render(p, phi=torch.zeros(len(p), device=m.dev))
            else:
                r = m.render(p)
            parts.append((r if r.dim() == 4 else r[:, None]).float().cpu())
        outs.append(torch.cat(parts))
    img = torch.cat(outs, 1)                                       # [Z k k, C, P, P]
    C = img.shape[1]
    img = img.reshape(len(zs), k, k, C, P, P).flip(1).flip(2)      # sample index j k + (k - 1 - shift index)
    img = img.permute(0, 3, 4, 1, 5, 2).reshape(len(zs), C, P * k, P * k)
    return img.numpy().astype(np.float32), zs


def zstack_tif_bytes(stack, z_um, sample_um=None, info=None):
    """ImageJ hyperstack (axes ZCYX, float32) with the sample spacing (um) as resolution, the z step as spacing and
    info (dict, as json) in the ImageJ Info field."""
    import tifffile
    md = dict(axes='ZCYX', unit='um')
    if len(z_um) > 1:
        md['spacing'] = float(z_um[1] - z_um[0])
    if info is not None:
        md['Info'] = json.dumps(_plain(info))
    kw = {} if not sample_um else dict(resolution=(1.0 / sample_um, 1.0 / sample_um))
    bio = _io.BytesIO()
    tifffile.imwrite(bio, np.ascontiguousarray(stack, np.float32), imagej=True, metadata=md, **kw)
    return bio.getvalue()


def psf_figure(stack, z_um, names=None, sample_px=1.0, n_xy=5, title=None, fig=None, rect=None):
    """matplotlib Figure: per channel (row) n_xy xy slices through z and an xz section (y = the centre row). fig /
    rect (left, bottom, right, top): draw into that region of an existing figure."""
    from matplotlib.figure import Figure
    Z, C, H, W = stack.shape
    names = names or [f'ch{c}' for c in range(C)]
    zi = np.unique(np.round(np.linspace(0, Z - 1, min(n_xy, Z))).astype(int))
    has_xz = Z > 1
    if fig is None:
        fig = Figure(figsize=(11.0, 0.7 + 2.1 * C))
        rect = (0.05, 0.3 / (0.7 + 2.1 * C), 0.98, 1 - (0.45 if title else 0.25) / (0.7 + 2.1 * C))
    ratios = [1] * len(zi) + ([0.45, 2.2] if has_xz else [])
    gs = fig.add_gridspec(C, len(ratios), width_ratios=ratios, left=rect[0], bottom=rect[1], right=rect[2],
                          top=rect[3], wspace=0.12, hspace=0.45)
    ext = np.array([-(W / 2), W / 2]) * sample_px
    for c in range(C):
        for j, i in enumerate(zi):
            ax = fig.add_subplot(gs[c, j])
            ax.imshow(stack[i, c], cmap='gray', extent=(ext[0], ext[1], ext[1], ext[0]), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'z {z_um[i]:.2f} um', fontsize=7, color='#52514e')
            if j == 0:
                ax.set_ylabel(names[c], fontsize=8)
        if has_xz:
            ax = fig.add_subplot(gs[c, -1])
            dz = (z_um[1] - z_um[0]) / 2
            ax.imshow(stack[:, c, H // 2, :], cmap='gray', aspect='auto', origin='lower', interpolation='nearest',
                      extent=(ext[0], ext[1], z_um[0] - dz, z_um[-1] + dz))
            ax.set_title('xz section (y = 0)', fontsize=7, color='#52514e')
            ax.tick_params(labelsize=6)
            ax.set_xlabel('x (px)', fontsize=7); ax.set_ylabel('z (um)', fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def _png_bytes(fig, dpi=110):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    bio = _io.BytesIO()
    fig.savefig(bio, format='png', dpi=dpi)
    return bio.getvalue()


# ====================================================================================================== writing
def _model_block(model, prefix):
    """{type, model, arrays} of a PSF model (its 0.1.0 json keys) and {member: array}."""
    d, arrs = model._file_dict(None)
    members = {k: f'{prefix}/{k}.npy' for k in arrs}
    return (dict(type=model.type_name, model=_plain(d), arrays=members),
            {members[k]: np.asarray(v) for k, v in arrs.items()})


def _emitter_params(model):
    p = ['x', 'y'] + (['z'] if model.z_range is not None else []) + ['I', 'bg']
    return p + (['phi'] if model.type_name in COUPLED else [])


def _channel_blocks(psf, source, names):
    """(channels, top-level source block or None, {member: array}) of the model(s)"""
    arrays = {}
    t = getattr(psf, 'type_name', None)
    if t == 'fourpi':
        sb, a = _model_block(psf, 'source')
        arrays.update(a)
        chans = [dict(name=names(c), type='source_output', index=c,
                      model=dict(pol=int(psf.pol[c]), phi_rad=float(psf.phi[c].detach()),
                                 tau=float(psf.tau[c].detach())))
                 for c in range(psf.n_channels)]
        return chans, sb, arrays
    if t == 'fourpi_spline':
        if source is not None:
            sb, a = _model_block(source, 'source')
            arrays.update(a)
        elif psf.source_state is not None:
            sb = dict(type='fourpi', model=_plain(dict(fourpi=psf.source_state)), arrays={})
        else:
            sb = None
        chans = []
        for c in range(psf.n_channels):
            members = {k: f'{names(c)}/{k}.npy' for k in ('inc', 'cr', 'ci')}
            for comp, k in enumerate(('inc', 'cr', 'ci')):
                arrays[members[k]] = psf.luts[comp][c].knots.detach().cpu().numpy()
            g = dict(psf.luts[0][c].to_dict(), pixel_um=psf.pixel_um)
            chans.append(dict(name=names(c), type='phase_spline',
                              model=_plain(dict(spline=g, tau=float(psf.tau[c].detach()))), phase_param='phi',
                              arrays=members))
        return chans, sb, arrays
    models = _channel_models(psf)
    sources = list(source) if isinstance(source, (list, tuple)) else [source] + [None] * (len(models) - 1)
    if len(sources) != len(models):
        raise ValueError('source: one per channel')
    chans = []
    for c, (m, s) in enumerate(zip(models, sources)):
        if m.type_name in COUPLED:
            raise ValueError('a coupled (4Pi) model is one psf.zip of its own, not one of several channels')
        blk, a = _model_block(m, names(c))
        arrays.update(a)
        if s is not None:
            sblk, a = _model_block(s, f'{names(c)}/source')
            arrays.update(a)
            blk['source'] = sblk
        chans.append(dict(name=names(c), **blk))
    return chans, None, arrays


def _np_cpu(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


def _camera_block(camera, arrays):
    d = camera.to_dict()
    amap = {}
    for k in ('gain_e_per_adu', 'offset_adu', 'readnoise_e'):
        v = getattr(camera, k)
        if (torch.is_tensor(v) or isinstance(v, np.ndarray)) and np.ndim(v) > 0:
            arr = np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, np.float32)
            amap[k] = f'camera/{k}.npy'
            arrays[amap[k]] = arr
            d[k] = float(np.median(arr))
    if amap:
        d['arrays'], d['maps'] = amap, None
    return _plain(d)


def _normalize_mapping(m, name, arrays):
    """JSON mapping block; the displacement fields of a grid mapping given as arrays go into the zip"""
    if m is None:
        return None
    m = dict(m)
    if m.get('type') == 'grid':
        for k in ('fwd', 'inv'):
            if m.get(k) is not None and not isinstance(m[k], str):
                arrays[f'{name}/mapping_{k}.npy'] = np.asarray(_np_cpu(m[k]), np.float32)
                m[k] = f'{name}/mapping_{k}.npy'
    m = _plain(m)
    if 'type' not in m:
        m['type'] = 'poly' if ('fwd' in m and 'order' in m) else 'identity'
    return m


def _check_field(field, block, crop, arrays):
    """warn when the channel model's Zernikes differ from the field polynomial at its reference point"""
    if field.get('type') != 'zernike_poly':
        return
    f = FieldZernikePoly(field, arrays, crop)
    m = block['model']
    if 'zernike_waves' not in m and block.get('source'):
        m = block['source']['model']
    if 'zernike_waves' not in m:
        warnings.warn('zernike_poly field block on a channel without a vectorial model (or source)')
        return
    w = dict(zip(m['zernike_noll'], m['zernike_waves']))
    at = f.zernike_at(*f.ref)
    bad = [j for j, v in zip(f.noll, at) if j not in w or abs(w[j] - v) > 1e-5]
    if bad:
        warnings.warn(f'field block: zernike_waves of the model differ from the polynomial at the reference point for '
                      f'Noll {bad}')


def _optics_of(chans, sb, models):
    for blk in ([sb] if sb else []) + [c.get('source') for c in chans if c.get('source')] + chans:
        if blk and isinstance(blk.get('model'), dict) and blk['model'].get('optics'):
            return blk['model']['optics']
        if blk and blk.get('model', {}).get('fourpi'):
            m = blk['model']['fourpi']['meta']
            return dict(pixel_um=m['px'], NA=m['NA'], wavelength_um=m['lam'], n_imm=m['n'], n_med=m.get('n_med', m['n']))
    px = getattr(models[0], 'pixel_um', None)
    return None if px is None else dict(pixel_um=float(px))


def save_zip(path, psf, camera=None, provenance=None, source=None, report=None, zstack_px=1.0, zstack_planes=25,
             channels=None, patterns=None, report_max_spots=1000):
    """Write a psf.zip (docs/PSF_FORMAT.md). Returns the psf.json dict.

    psf: a PSF model, or a list of independent channel models (biplane, multicolour, dual view: one zip for the
    setup; channel 0 is the reference). A FourPiPSF is written as one coupled model (source block, C output
    channels), a FourPiSplinePSF as C phase_spline channels plus its source model. source: the model a spline was
    built from (one per channel for a list). camera: Camera (per-pixel maps go into the zip). provenance: default
    psf.provenance. report: the estimator that made the PSF (ZStackEstimator / InSituEstimator: its report, history
    and data), or a dict (an estimator report, or {report, history, data, settings}); None: the report pages come from
    provenance['report'] when present, else the model pages only. zstack_px: sample spacing of zstack.tif (camera px,
    1 / integer), zstack_planes: planes over the z range. channels: per-channel extras, a list of dicts (name,
    mapping, field, crop, flip, intensity_ratio, or other keys kept as given). patterns: optional pattern /
    modulation metadata block (SIMFLUX / SIMCODE). report_max_spots: spots refitted for the report histograms."""
    from torch_psf import psfreport
    models = _channel_models(psf)
    extras = list(channels or [])
    chans, sb, arrays = _channel_blocks(
        psf, source, lambda c: str(extras[c].get('name') or f'ch{c}') if c < len(extras) else f'ch{c}')
    if len(extras) > len(chans):
        raise ValueError(f'channels: {len(extras)} entries for {len(chans)} channels')
    for i, ch in enumerate(chans):
        ch.setdefault('mapping', {'type': 'identity'})
        ch.setdefault('intensity_ratio', 1.0)
        if i < len(extras):
            ex = dict(extras[i])
            for k, v in ex.items():
                if k == 'mapping':
                    ch['mapping'] = _normalize_mapping(v, ch['name'], arrays) or {'type': 'identity'}
                elif k == 'field':
                    if v is not None:
                        ch['field'] = _plain(v)
                        _check_field(ch['field'], ch, ch.get('crop') or ex.get('crop'), arrays)
                else:
                    ch[k] = _plain(v)
    names = [c['name'] for c in chans]
    if len(set(names)) != len(names):
        raise ValueError(f'channel names must be unique: {names}')
    meta = dict(format=FORMAT, version=ZIP_VERSION, psf_type=models[0].type_name if len(models) == 1 else 'channels',
                n_dims=models[0].n_dims, emitter_params=_emitter_params(models[0]), units=UNITS)
    if len(chans) > 1:
        meta['per_channel_params'] = ['bg']
    optics = _optics_of(chans, sb, models)
    if optics:
        meta['optics'] = optics
    if camera is not None:
        meta['camera'] = _camera_block(camera, arrays)
    prov = provenance if provenance is not None else (models[0].provenance or {})
    meta['provenance'] = _plain(prov)
    meta['channels'] = chans
    if sb is not None:
        meta['source'] = sb
    if patterns is not None:
        meta['patterns'] = _plain(patterns)
    # zstack, preview, report
    stack, zs = render_zstack(psf, zstack_planes, zstack_px)
    px = (optics or {}).get('pixel_um')
    sample_um = None if not px else float(px) * float(zstack_px)
    zinfo = dict(file=ZSTACK_NAME, axes='ZCYX', shape=list(stack.shape), dtype='float32', z_um=[float(z) for z in zs],
                 sample_px=float(zstack_px), sample_um=sample_um, channels=names, position=dict(x=0.0, y=0.0),
                 params=dict(phi=0.0) if models[0].type_name in COUPLED else {},
                 value='pixel-integrated normalized PSF (render): fraction of the photons in a camera pixel centred '
                       'at the sample')
    meta['zstack'] = zinfo
    title = f'{meta["psf_type"]} PSF' + (f' ({len(chans)} channels)' if len(chans) > 1 else '')
    png = _png_bytes(psf_figure(stack, zs, names, zstack_px, title=title))
    meta['preview'] = dict(file=PREVIEW_NAME)
    meta['report'] = dict(file=REPORT_NAME)          # sections filled in below
    pdf, sections = psfreport.write_report(psf, meta=meta, stack=stack, z_um=zs, report=report, source=source,
                                           max_spots=report_max_spots)
    meta['report']['sections'] = sections
    tif = zstack_tif_bytes(stack, zs, sample_um, info=dict(format=FORMAT, version=ZIP_VERSION, psf_type=meta['psf_type'],
                                                          z_um=zinfo['z_um'], channels=names, sample_px=zinfo['sample_px']))
    with zipfile.ZipFile(str(path), 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(JSON_NAME, json.dumps(meta, indent=1))
        for m, a in arrays.items():
            b = _io.BytesIO()
            np.save(b, np.ascontiguousarray(a), allow_pickle=False)
            z.writestr(m, b.getvalue())
        z.writestr(ZSTACK_NAME, tif)
        z.writestr(PREVIEW_NAME, png, compress_type=zipfile.ZIP_STORED)
        z.writestr(REPORT_NAME, pdf)
    return meta

