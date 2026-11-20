"""Single-PSF benchmark: accuracy and speed of the vectorial model (box / spectral / Gauss-Legendre pixel
integration), the 2D and astigmatic Gaussians fitted to it, and the spline definitions. From the repository root run
python study/psf_unify/bench.py, then python study/psf_unify/report.py. --phase timing updates an existing precision run.
GPU utilization is sampled around each timing group. Desktop-active times are marked separately.
Use --phase retry for a quiet-window retry of contaminated cases and --phase paired to repeat
the interleaved S=2 comparison. Both keep the original measurements in the JSON.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from torch_psf import AstigmaticGaussianPSF, GaussianPSF, PSF
from torch_psf import fit
from torch_psf.gaussian import _axis
from torch_psf.spline import to_spline

OUT = Path(__file__).resolve().parent / 'out'
MODEL = Path('C:/dev/simcode2-inspect/models/m1809/calib_1809_2_psf_v2.psf.zip')
P, ZR, DZ, XL = 13, (0., .4), .01, 2.
N, BG, REPEATS = 3000, 10., 10
PHOTONS = (1000, 5000, 20000)
TABLES = [(name, s) for s in (2, 1, 3, 4) for name in ('cr_int', 'bs_int', 'cr_pix')]
OPTICS = [(name, s) for s in (3, 2, 4) for name in ('box', 'spectral', 'gauss_legendre')]   # s = samples / pixel
OPTICS.append(('box', 1))   # point sampling: the PSF value at the pixel centre
GAUSS = [('astig', None), ('gauss2d', None)]
GAUSS_PARAMS = {}      # fitted Gaussian parameters (accuracy run), passed to the timing workers through the JSON
BATCHES = (1000, 10000, 100000)
OPS = ('forward', 'grad', 'backward')


def save(out, data):
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False), encoding='utf-8')
    tmp.replace(out)


MAX_MEM_GB = None   # --max-mem-gb: cap of this process's CUDA allocator (shared GPUs)
_GPU_INDEX = None


def physical_gpu_index():
    """nvidia-smi index of the card torch uses (CUDA_VISIBLE_DEVICES remaps torch's device 0): matched by UUID,
    else the first entry of CUDA_VISIBLE_DEVICES, else 0."""
    global _GPU_INDEX
    if _GPU_INDEX is not None:
        return _GPU_INDEX
    idx = None
    try:
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                       text=True, timeout=10).strip().splitlines()
        for row in rows:
            i, u = [x.strip() for x in row.split(',')]
            if u.removeprefix('GPU-') == uuid.removeprefix('GPU-'):
                idx = int(i)
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        pass
    if idx is None:
        first = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
        idx = int(first) if first.isdigit() else 0
    _GPU_INDEX = idx
    return idx


def limit_memory():
    """Apply --max-mem-gb to the allocator (a no-op without it); returns the allowed fraction of the card."""
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    fraction = 1.0 if MAX_MEM_GB is None else min(1.0, MAX_MEM_GB * 2**30 / total)
    if MAX_MEM_GB is not None:
        torch.cuda.set_per_process_memory_fraction(fraction)
    return fraction


def gpu_sample():
    cmd = ['nvidia-smi', f'--id={physical_gpu_index()}', '--query-gpu=utilization.gpu,memory.used,temperature.gpu',
           '--format=csv,noheader,nounits']
    try:
        u, m, t = map(float, subprocess.check_output(cmd, text=True, timeout=10).strip().split(','))
        return dict(util_percent=u, used_MiB=m, temperature_C=t)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return dict(error=str(e))


def idle_check():
    # Sample after our last kernels have drained and the driver's utilization window has expired.
    torch.cuda.synchronize()
    time.sleep(1.5)
    samples = []
    for _ in range(3):
        samples.append(gpu_sample())
        time.sleep(.5)
    return samples


def idle(samples):
    utilization = [x.get('util_percent', 100) for x in samples]
    return np.median(utilization) <= 5 and max(utilization) <= 10


def positions(n):
    u = torch.rand(n, 3, generator=torch.Generator().manual_seed(1809))
    # Leave room for the localizer's lateral support margin and the perturbed initialization.
    return (u * torch.tensor([3.8, 3.8, .4]) + torch.tensor([-1.9, -1.9, 0.])).cuda()


class ShiftedAstigmaticGaussianPSF(AstigmaticGaussianPSF):
    """The astigmatic Gaussian with its z origin at z0_um (the calibration's focus need not sit at z = 0)."""
    def __init__(self, roisize, sigma_px, gamma_um, d_um, z0_um, z_range, device=None, pixel_um=None):
        super().__init__(roisize, sigma_px, gamma_um, d_um, z_range, device, pixel_um)
        self.z0 = float(z0_um)

    def _widths(self, z):
        return super()._widths(z - self.z0)


def _gauss_images(sx, sy):
    """Pixel-integrated Gaussian images [n, P, P] centred in the ROI, widths sx, sy [n] (px), differentiable."""
    o = torch.arange(P, device=sx.device, dtype=sx.dtype) - (P - 1) / 2
    zero = torch.zeros_like(sx)
    return _axis(o, zero, sy)[0][:, :, None] * _axis(o, zero, sx)[0][:, None, :]


def _scaled_misfit(images, target):
    # best amplitude per plane in closed form: the fit's photon count absorbs it
    a = (images * target).sum((-2, -1)) / images.square().sum((-2, -1))
    return (a[:, None, None] * images - target).square().sum()


def fit_gaussians(ref):
    """Least-squares Gaussian calibrations of the reference PSF on a z grid through ZR (emitter at the ROI centre):
    the astigmatic model (sx0, sy0, gamma, d, z0) and, at the reference's in-focus plane (highest peak), the 2D
    model (sx, sy)."""
    z = torch.linspace(ZR[0], ZR[1], 41, device='cuda')
    with torch.no_grad():
        target = ref.render(torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), 1)).double()
    zf = float(z[target.amax((-2, -1)).argmax()])
    q = torch.tensor([1.3, 1.3, .1, .3, zf], dtype=torch.float64, device='cuda').log1p().requires_grad_()
    def astig(v):
        sx0, sy0, gamma, d, z0 = v.expm1()
        zz = z.double() - z0
        return _gauss_images(sx0 * torch.sqrt(1 + ((zz - gamma) / d) ** 2), sy0 * torch.sqrt(1 + ((zz + gamma) / d) ** 2))
    opt = torch.optim.LBFGS([q], max_iter=500, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = _scaled_misfit(astig(q), target)
        loss.backward()
        return loss
    opt.step(closure)
    sx0, sy0, gamma, d, z0 = [float(x) for x in q.detach().expm1()]
    if gamma < 0:   # the model is symmetric under gamma -> -gamma with sx0 <-> sy0 swapped; keep gamma >= 0
        sx0, sy0, gamma = sy0, sx0, -gamma
    k = int(z.sub(zf).abs().argmin())
    w = torch.tensor([1.3, 1.3], dtype=torch.float64, device='cuda').log().requires_grad_()
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn='strong_wolfe')
    def closure2():
        opt.zero_grad()
        e = w.exp()
        loss = _scaled_misfit(_gauss_images(e[:1], e[1:]), target[k:k + 1])
        loss.backward()
        return loss
    opt.step(closure2)
    sx, sy = [float(x) for x in w.detach().exp()]
    return dict(astig=dict(sigma_px=[sx0, sy0], gamma_um=gamma, d_um=d, z0_um=z0),
                gauss2d=dict(sigma_px=[sx, sy], z_um=zf))


@torch.no_grad()
def model(ref, name, s):
    if name == 'astig':
        g = GAUSS_PARAMS['astig']
        return ShiftedAstigmaticGaussianPSF(P, tuple(g['sigma_px']), g['gamma_um'], g['d_um'], g['z0_um'], ZR,
                                            device='cuda', pixel_um=ref.pixel_um)
    if name == 'gauss2d':
        return GaussianPSF(P, tuple(GAUSS_PARAMS['gauss2d']['sigma_px']), device='cuda', pixel_um=ref.pixel_um)
    if name in ('bs_int', 'cr_int'):
        # The five-tap integral at S=1 needs another pixel of lateral padding.
        m = to_spline(ref, ZR, roisize=P, oversample=s, dz_um=DZ, value='density', zpad=6, xy_limit=XL + (s == 1),
                               kernel='bspline' if name == 'bs_int' else 'catmull', chunk=2)
        assert m.xmax >= XL
        m.xmax = XL
        return m
    if name == 'cr_pix':
        # explicit kernel / value: the plain default of to_spline is the integrated B-spline
        return ref.to_spline(P, ZR, kernel='catmull', value='pixel', oversample=s, dz_um=DZ, zpad=3, xy_limit=XL,
                             chunk=2)
    return ref.resampled(P, oversample=s, pixel_integration=name)


def variants(name):
    if name not in ('cr_int', 'bs_int', 'cr_pix'):
        return [('native', False)]
    return [('sep', False), ('conv', False), ('conv', True)]


def configure(m, backend, tf32, autotune=False):
    if hasattr(m, 'backend'):
        m.backend = backend
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = autotune


def theta(p, photons):
    return torch.cat((p, torch.full_like(p[:, :1], photons), torch.full_like(p[:, :1], BG)), 1)


@torch.no_grad()
def renders(m, p, chunk):
    chunks = p.split(chunk)
    value = torch.cat([m.render(q) for q in chunks])
    derivatives = zip(*(m.render(q, grad=True)[1:] for q in chunks))
    return (value, *(torch.cat(parts) for parts in derivatives))


@torch.no_grad()
def precision(m, pos, ref, nm, fixed=()):
    # Normalise each emitter and each derivative by its own reference absolute peak.
    v = renders(m, pos, 200)
    out = {}
    for key, a, b in zip(('value', 'dx', 'dy', 'dz'), v, ref):
        if key == 'dz' and 2 in fixed:
            continue
        e = (a - b).abs().amax((-2, -1)) / b.abs().amax((-2, -1))
        out[key] = dict(max=float(e.max()), median=float(e.median()))
    diff = (v[0] - ref[0]).double()
    out['chi2'] = {}
    for photons in (50000, 200000):
        roi = ((photons * diff).square() / (photons * ref[0].double() + BG)).sum((-2, -1))
        out['chi2'][str(photons)] = dict(mean_roi=float(roi.mean()), mean_pixel=float(roi.mean()/P**2),
                                        max_pixel_mean=float(roi.max()/P**2))
    out['bias_nm'] = {}
    p = pos
    for photons in PHOTONS:
        # Keep the Poisson objective in double precision to resolve sub-nm changes in expected images.
        start = theta(p, photons).double()
        start[:, :3] += start.new_tensor([.03, -.03, 0. if 2 in fixed else .008])
        data = photons * ref[0].double() + BG
        fitted = torch.cat([fit.lm_poisson(lambda t: fit.model_and_jacobian(m, t), d, t0, fixed=fixed,
                                           clamp=lambda t: fit.clamp_theta(t, m, z_limits=ZR), iters=25)[0]
                            for d, t0 in zip(data.split(500), start.split(500))])
        delta = (fitted[:, :3] - p) * nm
        out['bias_nm'][str(photons)] = dict(max=delta.abs().amax(0).tolist(),
                                            rms=delta.square().mean(0).sqrt().tolist())
    return out


def accuracy_run(ref, data, out):
    configure(ref, 'native', False)
    p = positions(N)
    r = renders(ref, p, 24)
    nm = p.new_tensor([1000 * ref.pixel_um, 1000 * ref.pixel_um, 1000])
    data['crlb_nm'] = {}
    for photons in PHOTONS:
        sd = torch.cat([fit.crlb(ref, theta(q, photons)) for q in p.split(24)])
        data['crlb_nm'][str(photons)] = (sd[:, :3] * nm).median(0).values.tolist()
    # the reference (spectral pixel integral) against an independent quadrature without FFTs
    with torch.no_grad():
        gl = ref.resampled(P, oversample=6, pixel_integration='gauss_legendre')
        q = p[:300]
        a, b = renders(gl, q, 24), renders(ref, q, 24)
    data['reference_check'] = dict(
        method='gauss_legendre, 6 nodes / pixel', n=len(q),
        value_max=float(((a[0] - b[0]).abs().amax((-2, -1)) / b[0].abs().amax((-2, -1))).max()))
    GAUSS_PARAMS.update(fit_gaussians(ref))
    data['gaussian_fit'] = dict(GAUSS_PARAMS)
    zf = GAUSS_PARAMS['gauss2d']['z_um']
    p2 = p.clone()
    p2[:, 2] = zf
    r2 = renders(ref, p2, 24)
    data['crlb_nm_focus'] = {}
    for photons in PHOTONS:
        sd = torch.cat([fit.crlb(ref, theta(q, photons)) for q in p2.split(24)])
        data['crlb_nm_focus'][str(photons)] = (sd[:, :3] * nm).median(0).values.tolist()
    data['accuracy'] = []
    for name, s in OPTICS + GAUSS + TABLES:
        configure(ref, 'native', False)
        m = model(ref, name, s)
        assert p[:, :2].abs().max() < m.xy_limit - .05
        for backend, tf32 in variants(name):
            row = dict(name=name, S=s, backend=backend, tf32=tf32)
            configure(m, backend, tf32)
            if name == 'gauss2d':
                row.update(precision(m, p2, r2, nm, fixed=(2,)), z_um=zf)
            else:
                row.update(precision(m, p, r, nm))
            data['accuracy'].append(row)
            save(out, data)
            print('precision', name, s, backend, tf32, row['value'], flush=True)


def timing_grid():
    for name, s in TABLES + OPTICS + GAUSS:
        tunes = (False, True) if name in ('cr_int', 'bs_int', 'cr_pix') else (False,)   # cuDNN: conv splines only
        ops = OPS if name not in ('astig', 'gauss2d') else OPS[:2]   # Gaussian shape parameters are plain floats (no gradient)
        for backend, tf32 in variants(name):
            for tune, n, op in itertools.product(tunes, BATCHES, ops):
                yield dict(name=name, S=s, backend=backend, tf32=tf32, cudnn_benchmark=tune, batch=n, op=op)


def operation(m, p, op):
    # One native batch per call. An OOM is a result for this batch size.
    table = hasattr(m, 'knots')
    if op == 'backward':
        if table:
            params = [m.knots.detach().requires_grad_()]
        else:
            params = [m.zcoef.requires_grad_()]

    def run():
        if op == 'backward':
            images = m.render_tables(p, knots=params[0]) if table else m.render(p)
            torch.autograd.grad(images.square().sum(), params)
        else:
            with torch.no_grad():
                m.render(p, grad=(op == 'grad'))

    return run


def stats(elapsed, n, peak):
    speed = n / np.asarray(elapsed)
    q1, med, q3 = np.percentile(speed, [25, 50, 75])
    return dict(seconds=elapsed, spots_s=med, q1=q1, q3=q3, iqr=q3-q1, peak_MiB=peak)


def timed_case(m, p, op):
    run = operation(m, p, op)
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        run()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - t)
    return stats(elapsed, len(p), torch.cuda.max_memory_allocated() / 2**20)


def paired_run(ref):
    # Rotate the three S=2 splines through the same timing window for the decision, all on the sep backend.
    p = positions(10000)
    names = ['cr_int', 'bs_int', 'cr_pix']
    configure(ref, 'native', False)
    models = [model(ref, name, 2) for name in names]
    for m in models:
        configure(m, 'sep', False)
    attempts = []
    for _ in range(12):
        before = idle_check()
        attempt = dict(before=before, rows=[])
        attempts.append(attempt)
        if not idle(before):
            time.sleep(3)
            continue
        for op in OPS:
            runs = [operation(m, p, op) for m in models]
            for run in runs:
                run()
                run()
            torch.cuda.synchronize()
            elapsed = [[] for _ in names]
            peaks = [0.] * len(names)
            for repeat in range(REPEATS):
                for j in range(len(names)):
                    i = (j + repeat) % len(names)
                    torch.cuda.reset_peak_memory_stats()
                    t = time.perf_counter()
                    runs[i]()
                    torch.cuda.synchronize()
                    elapsed[i].append(time.perf_counter() - t)
                    peaks[i] = max(peaks[i], torch.cuda.max_memory_allocated() / 2**20)
            for name, times, peak in zip(names, elapsed, peaks):
                attempt['rows'].append(dict(name=name, op=op, **stats(times, len(p), peak)))
        attempt['after'] = idle_check()
        attempt['idle'] = bool(idle(attempt['after']))
        if attempt['idle']:
            break
    return attempts


def timing_run(ref, data, out, tune, retry=False):
    rows = [r for r in data['timing'] if r['cudnn_benchmark'] == tune]
    if retry:
        rows = [r for r in rows if ('spots_s' in r and not r.get('idle'))
                or r.get('unavailable', '').startswith('Preflight')]
    p = positions(max(BATCHES))
    # On WDDM, large native batches can spill into shared system RAM. Limit PyTorch's
    # allocator to the physical memory available to this process at the start of timing.
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    fraction = min(1., (free + torch.cuda.memory_reserved()) / total, limit_memory())
    torch.cuda.set_per_process_memory_fraction(fraction)
    budget = total * fraction / 2**20
    data.setdefault('allocator_budget_MiB', {}).setdefault(str(tune), budget)
    checks = data['timing_idle_checks']
    for (name, s), group in itertools.groupby(rows, key=lambda r: (r['name'], r['S'])):
        group = list(group)
        samples = idle_check()
        check = dict(name=name, S=s, cudnn_benchmark=tune, samples=samples, allocator_budget_MiB=budget)
        checks.append(check)
        if retry:
            check['retry'] = True
            if not idle(samples):
                save(out, data)
                continue
            # Preserve earlier measurements when trying again during a quieter window.
            originals = group
            group = [dict(r) for r in originals]
            for row in group:
                row.pop('unavailable', None)
        if any(x.get('util_percent', 100) >= 40 for x in samples):
            for row in group:
                row['unavailable'] = 'Preflight GPU utilization >=40% or unavailable'
        else:
            configure(ref, 'native', False, tune)
            m = model(ref, name, s)
            for row in group:
                configure(m, row['backend'], row['tf32'], row['cudnn_benchmark'])
                try:
                    row.update(timed_case(m, p[:row['batch']], row['op']))
                except (RuntimeError, MemoryError) as e:
                    row['unavailable'] = str(e).splitlines()[0]
                    for key in ('spots_s', 'seconds', 'q1', 'q3', 'iqr', 'peak_MiB'):
                        row.pop(key, None)
                torch.cuda.empty_cache()
            del m
            after = check['after'] = idle_check()
            for row in group:
                row['idle'] = bool(idle(samples) and idle(after))
        if retry:
            for old, new in zip(originals, group):
                if 'spots_s' in new and (new.get('idle') or 'spots_s' not in old):
                    data.setdefault('previous_timings', []).append(old.copy())
                    old.clear()
                    old.update(new)
                else:
                    data.setdefault('previous_timings', []).append(new)
        save(out, data)
        print('timing', name, s, samples, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--out', type=Path, default=OUT / 'psf_bench.json')
    parser.add_argument('--phase', choices=['all', 'precision', 'timing', 'paired', 'retry'], default='all')
    parser.add_argument('--timing-worker', choices=['on', 'off'], help=argparse.SUPPRESS)
    parser.add_argument('--max-mem-gb', type=float, default=None,
                        help='cap of the CUDA memory this process may use (GB; shared GPUs); batches that do not fit '
                             'are recorded as out of memory')
    args = parser.parse_args()
    global MAX_MEM_GB
    MAX_MEM_GB = args.max_mem_gb
    limit_memory()
    mem_args = [] if MAX_MEM_GB is None else ['--max-mem-gb', str(MAX_MEM_GB)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.timing_worker or args.phase == 'paired':
        data = json.loads(args.out.read_text())
        if (data['sha256'] != hashlib.sha256(args.model.read_bytes()).hexdigest()
                or data['gpu'] != torch.cuda.get_device_name() or data['torch'] != str(torch.__version__)):
            raise ValueError('Timing requires the same calibration, device and PyTorch build')
        GAUSS_PARAMS.update(data.get('gaussian_fit', {}))
        ref = PSF.load(args.model, roisize=P, device='cuda').resampled(P, pixel_integration='spectral')
        if args.timing_worker:
            timing_run(ref, data, args.out, args.timing_worker == 'on', retry=args.phase == 'retry')
        else:
            data.setdefault('paired', []).extend(paired_run(ref))
            save(args.out, data)
        return
    if args.phase == 'retry':
        for tune in ('off', 'on'):
            subprocess.run([sys.executable, __file__, '--model', str(args.model), '--out', str(args.out),
                            '--phase', 'retry', '--timing-worker', tune] + mem_args, check=True)
        return
    model_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    ref = PSF.load(args.model, roisize=P, device='cuda').resampled(P, pixel_integration='spectral')
    try:
        revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                            'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    data = dict(revision=revision, measured_utc=datetime.now(timezone.utc).isoformat(), model=args.model.name, sha256=model_sha, NA=float(ref.NA), pixel_um=ref.pixel_um,
                focus_um=ref.focus_um, npupil=ref.npupil, optical_os=ref.os, dz_um=DZ, z_range=ZR,
                xy_sample=[-1.9, 1.9], xy_limit=XL, roisize=P, N=N, n_fit=N, seed=1809, bg=BG,
                torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                gpu_total_MiB=torch.cuda.get_device_properties(0).total_memory / 2**20,
                cudnn=torch.backends.cudnn.version(), repeats=REPEATS, preflight=gpu_sample())
    if args.phase in ('timing', 'precision') and args.out.exists():
        prev = json.loads(args.out.read_text())
        if prev['sha256'] != model_sha or prev.get('revision', revision) != revision:
            raise ValueError('Existing results belong to a different calibration or revision; use --phase all')
        keep = ('accuracy', 'crlb_nm', 'crlb_nm_focus', 'gaussian_fit', 'reference_check') if args.phase == 'timing' else (
            'timing', 'timing_idle_checks', 'allocator_budget_MiB', 'paired', 'previous_timings')
        if any(prev.get(key) != data[key] for key in ('gpu', 'torch', 'cudnn', 'seed', 'N', 'roisize')):
            raise ValueError('Existing results have different hardware, software or sampling settings')
        data.update({k: prev[k] for k in keep if k in prev})
    expected = sum(len(variants(name)) for name, s in OPTICS + GAUSS + TABLES)
    if args.phase in ('all', 'precision') or len(data.get('accuracy', [])) != expected:
        accuracy_run(ref, data, args.out)
    if args.phase == 'precision':
        save(args.out, data)
        return
    data['timing'] = list(timing_grid())
    data['timing_idle_checks'] = []
    save(args.out, data)
    # cuDNN caches an algorithm for a shape. Give each benchmark setting its own
    # process so the second policy cannot reuse the first policy's warmed plans.
    del ref
    torch.cuda.empty_cache()
    for tune in ('off', 'on'):
        subprocess.run([sys.executable, __file__, '--model', str(args.model), '--out', str(args.out),
                        '--timing-worker', tune] + mem_args, check=True)
    subprocess.run([sys.executable, __file__, '--model', str(args.model), '--out', str(args.out),
                    '--phase', 'paired'] + mem_args, check=True)


if __name__ == '__main__':
    main()
