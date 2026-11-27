"""Training of SMLMNet on live simulated scenes (torch_psf.nn.scenes), with the optimiser and schedule of SIMCODE's
plain-SMLM trainer (simcode.v1net.train): Lion, a linear warm-up from lr / 100 over `warmup` steps, then lr x
decay_gamma^(step // decay_every); one fresh scene batch per step.

    python -m torch_psf nn train config.json [--out DIR] [--steps N]
    torchrun --standalone --nproc_per_node 2 -m torch_psf nn train config.json     (optional data parallel)

Config (one JSON dict per run; DEFAULTS below, 'model' and 'scenes' merged key by key):
  out              run directory (also the model directory: weights.pt, model.json at the end)
  seed             seed of the training stream; the network initialisation uses seed + 1, the validation batches
                   seed + 7; the stream after a resume at step s restarts at seed + 1000 s
  steps, batch     updates and fields per update
  size, halo       supervised square (px) and context simulated around it (px; the loss sees the square only)
  lr, weight_decay, warmup, decay_every, decay_gamma        Lion (betas 0.9, 0.99) and the schedule
  count_loss_weight                                          weight of the count term of the loss
  val_batches, log_every, save_every                         validation batches, log / checkpoint intervals
  resume           continue from last.ckpt in out
  deterministic    torch.use_deterministic_algorithms (and cuDNN deterministic, CUBLAS workspace :4096:8)
  device           cuda / cpu (default: cuda if available)
  tensorboard      also write the log scalars to out/tb (needs the tensorboard package)
  model            SMLMNet options (window, z, cov, max_photons, max_bg, xyz_scale, input_scale, input_offset,
                   frame_features, window_features, window_out, head_features, batch_norm)
  scenes           the scene settings (torch_psf.nn.scenes.SCENE_DEFAULTS: psf, psf_weights, roisize, window,
                   pixel_nm, pixel_nm_y, camera, label_min, min_on, ranges)

Outputs in out: config.json (the resolved config with the torch settings), train.log (one JSON line per log
interval: train / validation loss, lr, s per step, validation metrics), last.ckpt (network, optimiser, step),
weights.step<N>.pt, and at the end weights.pt + model.json.

Data parallel (WORLD_SIZE > 1 under torchrun; nothing changes for one process): each rank simulates its own stream
(seed + rank x 1000003), batch / world size fields per rank ('ddp_batch': 'split', default) or the full batch
('per_rank'); BatchNorm -> SyncBatchNorm, DistributedDataParallel; rank 0 validates, logs and saves.
"""
import copy
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from torch_psf.nn.model import SMLMNet
from torch_psf.nn.scenes import Scenes, scene_config, RANGES
from torch_psf.nn import files

DEFAULTS = dict(out=None, seed=20261122, steps=100000, batch=16, size=40, halo=0, lr=1e-4, weight_decay=1e-3,
                warmup=512, decay_every=40000, decay_gamma=0.5, count_loss_weight=0.01, val_batches=8,
                save_every=10000, log_every=1000, resume=True, deterministic=True, device=None, tensorboard=False,
                ddp_batch='split', model={}, scenes={})
RANK_SEED_STRIDE = 1000003


def load_config(cfg):
    """A config path or dict -> the resolved config (DEFAULTS <- cfg; scenes completed by scene_config; relative PSF
    paths are resolved against the config file's directory)."""
    base = None
    if not isinstance(cfg, dict):
        base = os.path.dirname(os.path.abspath(str(cfg)))
        with open(cfg) as f:
            cfg = json.load(f)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown config keys {sorted(unknown)}')
    c = dict(copy.deepcopy(DEFAULTS), **copy.deepcopy(cfg))
    sc = dict(c['scenes'])
    if base:
        sc['psf'] = [p if os.path.isabs(p) else os.path.join(base, p) for p in sc.get('psf', [])]
    c['scenes'] = scene_config(sc)
    win = int(c['model'].get('window', 3))
    if win != int(c['scenes']['window']):
        raise ValueError(f"model window {win} != scenes window {c['scenes']['window']}")
    return c


def lr_at(step, cfg):
    base = cfg['lr'] * cfg['decay_gamma'] ** (step // cfg['decay_every'])
    w = cfg['warmup']
    return base * (0.01 + 0.99 * step / w) if step < w else base


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): update = sign(beta1 m + (1 - beta1) g), decoupled weight decay p *= 1 - lr wd,
    p -= lr update, then m = beta2 m + (1 - beta2) g (the update of the lion-pytorch package)."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            b1, b2 = g['betas']
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['exp_avg'] = torch.zeros_like(p)
                m = st['exp_avg']
                p.data.mul_(1 - g['lr'] * g['weight_decay'])
                update = m.clone().mul_(b1).add(p.grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-g['lr'])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


class Context:
    """Rank / world size of this process (on = torchrun with WORLD_SIZE > 1)."""
    def __init__(self):
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.on = self.world > 1
        self.rank = int(os.environ.get('RANK', '0')) if self.on else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0')) if self.on else 0
        self.main = self.rank == 0

    def init(self):
        if self.on:
            import torch.distributed as dist
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo',
                                        timeout=datetime.timedelta(hours=2))
        return self

    def close(self):
        if self.on:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier(); dist.destroy_process_group()

    def device(self, cfg):
        if self.on and torch.cuda.is_available():
            return torch.device('cuda', self.local_rank)
        from torch_psf.optics import resolve_device
        return resolve_device(cfg['device'])

    def seed(self, seed):
        return seed + self.rank * RANK_SEED_STRIDE

    def batch(self, cfg):
        if not self.on or cfg['ddp_batch'] == 'per_rank':
            return cfg['batch']
        if cfg['ddp_batch'] != 'split':
            raise ValueError(f"ddp_batch {cfg['ddp_batch']!r}: 'split' or 'per_rank'")
        if cfg['batch'] % self.world:
            raise ValueError(f"batch {cfg['batch']} is not divisible by the world size {self.world}")
        return cfg['batch'] // self.world

    def mean(self, x):
        if not self.on:
            return float(x)
        import torch.distributed as dist
        t = torch.tensor(float(x), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
        dist.all_reduce(t)
        return float(t) / self.world


def set_deterministic(on):
    if on:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def torch_backend():
    return dict(torch_version=str(torch.__version__), deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32)


def supervised(out, halo):
    """Network output on the simulated frame -> the supervised square (x, y in its coordinates)."""
    if not halo:
        return out
    o = out[..., halo:out.shape[-2] - halo, halo:out.shape[-1] - halo].clone()
    o[:, 1:3] -= halo
    return o


@torch.no_grad()
def validation_metrics(net, outputs, labels, pixel_nm, pixel_nm_y=None, p_min=0.5):
    """Detections (nms, raw_th 0.05, p >= p_min) matched to the labels: recall, precision, jaccard, rmse x / y / z
    (nm), z-score sd per axis and, for full_xyz, the 1-sigma coverage of the whitened errors per axis."""
    from torch_psf.nn.merge import detect
    from torch_psf.nn.calibrate import match_window
    err, zs, white = {k: [] for k in 'xyz'}, {k: [] for k in 'xyz'}, []
    ne = ng = nm = 0
    axes = 'xyz' if net.z else 'xy'
    for out, lab in zip(outputs, labels):
        d = detect(out, 'nms', p_min=p_min, n_params=net.n_params, cov=net.cov, raw_th=0.05)
        d = {k: v.float().cpu().numpy() for k, v in d.items()}
        L = {k: v.float().cpu().numpy() for k, v in lab.items()}
        ne += len(d['x']); ng += len(L['x'])
        for w in np.unique(L['fov']):
            gi = np.nonzero(L['fov'] == w)[0]; ri = np.nonzero(d['b'] == w)[0]
            if not len(ri):
                continue
            gz = L['z'][gi] if net.z else np.zeros(len(gi))
            a, b = match_window(np.stack([L['x'][gi], L['y'][gi], gz], 1),
                                np.stack([d['x'][ri], d['y'][ri], d['z'][ri]], 1), pixel_nm, pixel_nm_y)
            g, r = gi[a], ri[b]
            nm += len(g)
            for k in axes:
                e = d[k][r] - L[k][g]
                err[k].append(e); zs[k].append(e / d['s' + k][r])
            if net.cov and len(g):
                e = np.stack([d[k][r] - L[k][g] for k in 'xyz'], -1)
                white.append(np.linalg.solve(np.linalg.cholesky(d['cov'][r].astype(np.float64)), e[..., None])[..., 0])
    c = lambda v: np.concatenate(v) if v else np.zeros(0)
    py = pixel_nm if pixel_nm_y is None else pixel_nm_y
    m = dict(recall=nm / max(ng, 1), precision=nm / max(ne, 1), jaccard=nm / max(ne + ng - nm, 1), matched=nm)
    for k, s in zip(axes, (pixel_nm, py, 1000.)):
        v = c(err[k])
        m[f'rmse_{k}_nm'] = float(np.sqrt(np.mean(v ** 2)) * s) if len(v) else float('nan')
        m[f'zscore_sd_{k}'] = float(np.std(c(zs[k]))) if len(v) else float('nan')
    if white:
        W = c(white)
        for i, k in enumerate('xyz'):
            m[f'coverage_1sd_{k}'] = float(np.mean(np.abs(W[:, i]) <= 1))
    return m


def build_model(cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed'] + 1)
        return SMLMNet(**cfg['model'])


def train_model(cfg, out=None, steps=None, progress=print):
    """Train a model (module docstring). cfg: config path or dict; out / steps: overrides. Returns the run directory."""
    cfg = load_config(cfg)
    if out is not None:
        cfg['out'] = str(out)
    if steps is not None:
        cfg['steps'] = int(steps)
    if not cfg['out']:
        raise ValueError('config: out (the run directory) is required')
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        return _run(cfg, progress)
    finally:                                         # the process's torch settings as before
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[2], saved[3]


def _run(cfg, progress):
    ctx = Context().init()
    set_deterministic(cfg['deterministic'])
    dev = ctx.device(cfg)
    outp = Path(cfg['out'])
    if ctx.main:
        outp.mkdir(parents=True, exist_ok=True)
    seed = ctx.seed(cfg['seed'])
    sc = cfg['scenes']
    net = build_model(cfg).to(dev)
    start = 0
    ck = outp / 'last.ckpt'
    s = torch.load(ck, map_location=dev, weights_only=True) if cfg['resume'] and ck.exists() else None
    if s is not None:
        net.load_state_dict(s['net'])
    dnet = net
    if ctx.on:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        dnet = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index] if dev.type == 'cuda' else None,
                                                         broadcast_buffers=False)
    loss_fn = net.loss(cfg['count_loss_weight'])
    opt = Lion(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    if s is not None:
        opt.load_state_dict(s['opt'])
        start = int(s['step'])
        if ctx.main:
            progress(f'resumed at step {start}')
    del s
    scenes = Scenes(sc, ctx.batch(cfg), cfg['size'], cfg['halo'], dev, seed=seed + 1000 * start)
    pix, pix_y = sc['pixel_nm'], sc['pixel_nm_y']
    if ctx.main:
        val = Scenes(sc, cfg['batch'], cfg['size'], cfg['halo'], dev, seed=cfg['seed'] + 7)
        vb = list(val.batches(cfg['val_batches'], z=net.z))
        resolved = dict(cfg, torch_backend=torch_backend(),
                        ddp=dict(world_size=ctx.world, batch_per_rank=ctx.batch(cfg)) if ctx.on else None)
        (outp / 'config.json').write_text(json.dumps(resolved, indent=1))
        tb = None
        if cfg['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(outp / 'tb'), purge_step=start or None)
        log = open(outp / 'train.log', 'a')

    def emit(d):
        line = json.dumps(d)
        progress(line)
        log.write(line + '\n'); log.flush()
        if tb is not None:
            for k, v in d.items():
                if isinstance(v, (int, float)) and k != 'step':
                    tb.add_scalar(k, v, d['step'])
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            tb.add_scalar(f'{k}/{kk}', vv, d['step'])

    def validate():
        net.eval()
        with torch.no_grad():
            outs = [supervised(net(f), cfg['halo']) for f, _, _ in vb]
            v = float(np.mean([float(loss_fn(o, t)) for o, (_, t, _) in zip(outs, vb)]))
            m = validation_metrics(net, outs, [lab for _, _, lab in vb], pix, pix_y)
        net.train()
        return v, m

    if ctx.main:
        v, m = validate()
        emit(dict(step=start, validation_loss=v, validation=m, params=sum(p.numel() for p in net.parameters())))
    net.train()
    t0, run = time.time(), []
    for step, (f, t, _) in enumerate(scenes.batches(cfg['steps'] - start, z=net.z), start):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, cfg)
        loss = loss_fn(supervised(dnet(f), cfg['halo']), t)
        lv = ctx.mean(loss.detach()) if ctx.on else float(loss.detach())
        if not np.isfinite(lv):
            raise ValueError(f'non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run.append(lv)
        if not ctx.main:
            continue
        if (step + 1) % cfg['log_every'] == 0:
            v, m = validate()
            emit(dict(step=step + 1, train_loss=float(np.mean(run)), validation_loss=v, lr=opt.param_groups[0]['lr'],
                      s_per_step=(time.time() - t0) / len(run), validation=m))
            run, t0 = [], time.time()
        if (step + 1) % cfg['save_every'] == 0 or step + 1 == cfg['steps']:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step + 1), ck.with_suffix('.tmp'))
            ck.with_suffix('.tmp').replace(ck)
            torch.save(net.state_dict(), outp / f'weights.step{step + 1}.pt')
    if ctx.main:
        meta = files.model_meta(net, pix, pix_y, sc['psf'], files.camera_dict(sc['camera']),
                                training=dict(steps=cfg['steps'], seed=cfg['seed'], config='config.json'),
                                source=dict(tool='torch_psf.nn.train', date=datetime.date.today().isoformat()))
        files.save_model(outp, net, meta)
        log.close()
        if tb is not None:
            tb.close()
    ctx.close()
    return str(outp)


# ---------------------------------------------------------------------------------------------- SIMCODE configs
SIMCODE_MODEL_KEYS = dict(window='window', max_intensity='max_photons', max_bg='max_bg', xyz_scale='xyz_scale',
                          enable3D='z', unet_shared_features='frame_features',
                          unet_combiner_features='window_features', unet_combiner_output_features='window_out',
                          output_head_features='head_features', unet_batch_norm='batch_norm',
                          input_scale='input_scale', input_offset='input_offset', cov='cov')
SIMCODE_RUN_DEFAULTS = dict(seed=20261122, batch=16, size=40, halo=0, lr=1e-4, weight_decay=1e-3, warmup=512,
                            decay_every=40000, decay_gamma=0.5, count_loss_weight=0.01, label_min=100.,
                            val_batches=8, save_every=10000, log_every=1000, model={}, resume=True)


def simcode_model_options(m):
    """SIMCODE V1Net keyword arguments -> SMLMNet options (only the plain-SMLM form: bg_frames 1, default act)."""
    m = dict(m)
    if int(m.pop('bg_frames', 1)) != 1:
        raise ValueError('bg_frames > 1 (per-frame background maps) is not part of torch_psf.nn')
    if m.pop('unet_act', None) is not None:
        raise ValueError('unet_act: only the default ELU')
    m.pop('bg_init', None)
    bad = set(m) - set(SIMCODE_MODEL_KEYS)
    if bad:
        raise ValueError(f'V1Net options without a torch_psf.nn equivalent: {sorted(bad)}')
    return {SIMCODE_MODEL_KEYS[k]: v for k, v in m.items()}


def convert_simcode_config(run_json, args_json, psf=None, out=None):
    """SIMCODE's plain-SMLM V1Net run (run.json of simcode.v1net.train + the args.json of the stage it reads its
    scenes from) -> a torch_psf.nn training config that repeats it. psf: local PSF file(s) replacing the stored
    paths (same order). out: the run directory (default: none)."""
    rd = json.load(open(run_json)) if not isinstance(run_json, dict) else run_json
    ad = json.load(open(args_json)) if not isinstance(args_json, dict) else args_json
    r = dict(SIMCODE_RUN_DEFAULTS, **rd)
    c = ad['config']
    model = simcode_model_options(r['model'])
    det = c['detector']
    if float(det['min_phot']) != float(r['label_min']):
        raise ValueError(f"label_min {r['label_min']} != the stage's detector min_phot {det['min_phot']} "
                         '(SIMCODE refuses such a run)')
    sim = dict(c.get('sim') or {})
    for k in ('adu_per_e', 'offset', 'rn_adu', 'pixel_nm'):        # the simulator takes these from camera / pixel_nm
        sim.pop(k, None)
    if 'zr' in sim:
        sim['z_range_um'] = sim.pop('zr')
    bad = set(sim) - set(RANGES)
    if bad:
        raise ValueError(f'scene ranges without a torch_psf.nn equivalent: {sorted(bad)}')
    paths = c['psf'] if psf is None else ([psf] if isinstance(psf, (str, os.PathLike)) else list(psf))
    if len(paths) != len(c['psf']):
        raise ValueError(f"{len(c['psf'])} PSF files in the stored config, {len(paths)} given")
    cam = c['camera']
    scenes = dict(psf=[str(p) for p in paths], psf_weights=c.get('psf_weights'), roisize=23, window=int(c['window']),
                  pixel_nm=float(c['pixel_nm']), pixel_nm_y=c.get('pixel_nm_y'),
                  camera=dict(adu_per_e=float(cam['adu_per_e']), offset_adu=float(cam['offset']),
                              readnoise_adu=float(cam['rn_adu'])),
                  label_min=float(r['label_min']), min_on=float(det['min_on']), ranges=sim)
    if int(model.get('window', 3)) != scenes['window']:
        raise ValueError(f"model window {model.get('window', 3)} != the stage's window {scenes['window']}")
    keys = ('seed', 'steps', 'batch', 'size', 'halo', 'lr', 'weight_decay', 'warmup', 'decay_every', 'decay_gamma',
            'count_loss_weight', 'val_batches', 'save_every', 'log_every', 'resume')
    cfg = {k: r[k] for k in keys if k in r}
    cfg.update(model=model, scenes=scenes, out=None if out is None else str(out), deterministic=True)
    return cfg
