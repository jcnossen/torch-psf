"""torch_psf.nn against SIMCODE (the source of the port): network outputs, merge, loss, sigma calibration, scenes, and a
GPU smoke training on the Tubulin-A647-3D scene settings.

Needs two things from outside the repository (the tests skip without them):
  TORCH_PSF_NN_REF       a directory with the reference files of the Tubulin full-covariance run: v1net.pt, v1net.json,
                         run.json, args.json, sigma_scale_quantile.json, tub3d_bead.psf.zip
  TORCH_PSF_NN_SIMCODE   the SIMCODE source: a directory that holds the simcode package (and examples/), or
                         '<git repo>@<ref>' (e.g. C:/dev/simcode-dev@refs/tmp/ddp_stack), exported with git archive
SIMCODE is imported here only. Its scene simulator runs on CUDA only, so the scene comparison needs a GPU.
TORCH_PSF_NN_STATS_BATCHES (default 100): scene batches per simulator in the scene comparison (pytest -s prints the table).
"""
import json
import os
import subprocess
import sys
import tarfile
import io
import numpy as np
import pytest
import torch
import torch_psf.nn as tn
from torch_psf.nn import merge as M, calibrate as C, files
from conftest import needs_gpu

REF = os.environ.get('TORCH_PSF_NN_REF')
REF_FILES = ('v1net.pt', 'v1net.json', 'run.json', 'args.json', 'sigma_scale_quantile.json', 'tub3d_bead.psf.zip')


def ref(name):
    if not REF or not os.path.isfile(os.path.join(REF, name)):
        pytest.skip(f'TORCH_PSF_NN_REF with {name} not available')
    return os.path.join(REF, name)


@pytest.fixture(scope='module')
def simcode(tmp_path_factory):
    spec = os.environ.get('TORCH_PSF_NN_SIMCODE')
    if not spec:
        pytest.skip('TORCH_PSF_NN_SIMCODE not set')
    if '@' in spec and not os.path.isdir(spec):
        repo, rev = spec.rsplit('@', 1)
        root = str(tmp_path_factory.mktemp('simcode'))
        tar = subprocess.run(['git', '-C', repo, 'archive', rev, 'simcode', 'examples'], capture_output=True, check=True)
        with tarfile.open(fileobj=io.BytesIO(tar.stdout)) as t:
            t.extractall(root)
    else:
        root = spec
    if not os.path.isdir(os.path.join(root, 'simcode')):
        pytest.skip(f'{root}: no simcode package')
    sys.path.insert(0, root)
    yield root
    sys.path.remove(root)


@pytest.fixture(scope='module')
def tub(tmp_path_factory):
    """(config dict, converted model directory) of the Tubulin run."""
    d = tmp_path_factory.mktemp('tub')
    cfg = tn.convert_simcode_config(ref('run.json'), ref('args.json'), psf=ref('tub3d_bead.psf.zip'))
    cp = d / 'config.json'
    cp.write_text(json.dumps(cfg))
    m = tn.convert_simcode_model(ref('v1net.pt'), str(d / 'model'), meta=ref('v1net.json'),
                                 sigma_scale=ref('sigma_scale_quantile.json'), config=str(cp))
    return tn.load_config(str(cp)), m


def simcode_net(**kw):
    from simcode.v1net.model import V1Net
    return V1Net(**kw)


def tub_windows(cfg, n=2, seed=5):
    S = tn.Scenes(cfg['scenes'], 8, 40, 0, 'cpu', seed=seed)
    return [f for f, _, _ in S.batches(n)]


# ---------------------------------------------------------------------------------------------- 1. network
def test_network_equals_simcode(simcode, tub):
    cfg, mdir = tub
    net, meta, _ = tn.load_model(mdir, 'cpu')
    M_ = json.load(open(ref('v1net.json')))
    r = simcode_net(**M_['model'])
    r.load_state_dict(torch.load(ref('v1net.pt'), map_location='cpu', weights_only=True))
    r.eval()
    torch.manual_seed(0)
    inputs = [torch.rand(4, 3, 40, 40) * 100 + 5, torch.rand(2, 3, 64, 48) * 300, *tub_windows(cfg)]
    worst = 0.
    with torch.no_grad():
        for x in inputs:
            a, b = net(x), r(x)[:, 0]
            worst = max(worst, float((a - b).abs().max()))
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-5)
    print('network: max |torch_psf.nn - simcode| =', worst)


@pytest.mark.parametrize('opts', [dict(cov=None), dict(enable3D=False), dict(cov='full_xyz', max_bg=200.)])
def test_random_network_equals_simcode(simcode, opts):
    torch.manual_seed(1)
    r = simcode_net(unet_shared_features=(8, 16), unet_combiner_features=(8, 16, 32), unet_combiner_output_features=16,
                    output_head_features=4, **opts)
    for m in r.modules():                    # nonzero correlation head and BN statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-1, 1); m.running_var.uniform_(0.5, 2)
    if opts.get('cov'):
        torch.nn.init.normal_(r.cov_head.weight, std=0.1)
    r.eval()
    from torch_psf.nn.train import simcode_model_options
    kw = dict(unet_shared_features=(8, 16), unet_combiner_features=(8, 16, 32), unet_combiner_output_features=16,
              output_head_features=4, **opts)
    net = tn.SMLMNet(**simcode_model_options(kw))
    net.load_state_dict(files.simcode_state_dict(r.state_dict()))
    net.eval()
    x = torch.rand(3, 3, 24, 24) * 80
    with torch.no_grad():
        assert torch.allclose(net(x), r(x)[:, 0], rtol=1e-6, atol=1e-5)


# ---------------------------------------------------------------------------------------------- 2. decode
def _same_rows(a, b, keys):
    assert torch.equal(a['b'], b['b']) and torch.equal(a['iy'], b['iy']) and torch.equal(a['ix'], b['ix'])
    for k, kb in keys.items():
        assert torch.allclose(a[k], b[kb], rtol=1e-6, atol=1e-6), k


KEYS = dict(p='p', x='x', y='y', z='z', sx='sx', sy='sy', sz='sz', photons='N', s_photons='sN', bg='bg')


def test_merge_equals_simcode(simcode, tub):
    from simcode.v1net import decode_rules as DR
    from simcode.v1net.infer import DEFAULT_RULE_KW
    cfg, mdir = tub
    net, _, _ = tn.load_model(mdir, 'cpu')
    n_rows = 0
    with torch.no_grad():
        outs = [net(x) for x in tub_windows(cfg, 3, seed=9)]
    for o in outs:
        G = DR.grid_from_v1net(o[:, None], cov='full_xyz')
        a = tn.detect(o, 'merge', p_min=0.2)
        b = DR.detect(G, 'merge', p_min=0.2, **DEFAULT_RULE_KW)
        _same_rows(a, b, dict(KEYS, cov='cov_xyz'))
        n_rows += len(a['p'])
        a = tn.detect(o, 'nms', p_min=0.5, raw_th=0.05)
        b = DR.detect(G, 'v1', p_min=0.5)
        _same_rows(a, b, dict(KEYS, cov='cov_xyz'))
        # the diagonal layout (no correlation channels) and precision weighting
        od = o[:, :-3]
        a = tn.detect(od, 'merge', p_min=0.2, weight='prec')
        b = DR.detect(DR.grid_from_v1net(od[:, None]), 'merge', p_min=0.2, **dict(DEFAULT_RULE_KW, weight='prec'))
        _same_rows(a, b, KEYS)
    assert n_rows > 50
    print('merge: identical rows,', n_rows, 'localizations')


# ---------------------------------------------------------------------------------------------- 3. loss
@pytest.mark.parametrize('cov', [None, 'full_xyz'])
def test_loss_equals_simcode(simcode, tub, cov):
    cfg, mdir = tub
    from simcode.v1net.model import V1NetLoss
    net, _, _ = tn.load_model(mdir, 'cpu')
    S = tn.Scenes(cfg['scenes'], 4, 40, 0, 'cpu', seed=21)
    f, t, _ = next(S.batches(1))
    with torch.no_grad():
        o = net(f)
    if not cov:
        o = o[:, :-3]
    names = ['x', 'y', 'z', 'bg', 'N']
    oa, ob = o.clone().requires_grad_(True), o.clone().requires_grad_(True)
    la = tn.GMMLoss(5, cov=cov, count_loss_weight=0.01)(oa, t)
    lb = V1NetLoss(names, cov=cov, count_loss_weight=0.01)(ob[:, None], t[:, None])
    la.backward(); lb.backward()
    assert torch.allclose(la, lb, rtol=1e-6, atol=1e-4)
    assert torch.allclose(oa.grad, ob.grad, rtol=1e-5, atol=1e-6)
    print(f'loss {cov}: {float(la):.6f} vs {float(lb):.6f}, max grad diff {float((oa.grad - ob.grad).abs().max()):.3g}')


# ---------------------------------------------------------------------------------------------- 4. calibration
def _errors(rng, n):
    A = rng.normal(size=(n, 3, 3)) * 0.3 + np.eye(3)
    cov = A @ A.transpose(0, 2, 1) * np.array([0.04, 0.04, 0.01])[None, :, None] ** 0.5 \
        * np.array([0.04, 0.04, 0.01])[None, None, :] ** 0.5
    L = np.linalg.cholesky(cov)
    w = rng.standard_t(4, size=(n, 3)) * 1.3
    e = (L @ w[..., None])[..., 0]
    return dict(photons=rng.uniform(50, 4000, n), z_true=rng.uniform(-0.5, 0.5, n), ex=e[:, 0], ey=e[:, 1],
                ez=e[:, 2], cov=cov, sx=np.sqrt(cov[:, 0, 0]), sy=np.sqrt(cov[:, 1, 1]), sz=np.sqrt(cov[:, 2, 2]))


@pytest.mark.parametrize('method', ['rms', 'quantile'])
def test_xyz_matrix_equals_simcode(simcode, method):
    from simcode.v1net import sigma_scale as SS
    rng = np.random.default_rng(4)
    Ef, Ec = _errors(rng, 20000), _errors(rng, 20000)
    sc = lambda E: dict(N=E['photons'], zt=E['z_true'], ex=E['ex'], ey=E['ey'], ez=E['ez'], cov_xyz=E['cov'])
    edges = list(C.DEFAULT_EDGES)
    a, ra = C.fit_xyz_matrix(Ef, Ec, edges, (-0.4, 0.4), method)
    b, rb = SS.matrix_calibration(sc(Ef), sc(Ec), edges, (-0.4, 0.4), method)
    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)
    assert ra['after']['coverage'] == rb['after']['coverage']
    # application to merged rows (float32 rows; simcode factors in float32, torch_psf in float64)
    rows = dict(photons=torch.tensor(Ef['photons'][:500], dtype=torch.float32),
                cov=torch.tensor(Ef['cov'][:500], dtype=torch.float32),
                **{k: torch.tensor(Ef[k][:500], dtype=torch.float32) for k in ('sx', 'sy', 'sz')})
    S = dict(bin='N', edges=edges, xyz_matrix=b)
    ra = C.apply(dict(bin='photons', edges=edges, xyz_matrix=a), rows)
    rb = SS.apply(S, dict(N=rows['photons'], cov_xyz=rows['cov'], sx=rows['sx'], sy=rows['sy'], sz=rows['sz']))
    for k in ('sx', 'sy', 'sz'):
        assert torch.allclose(ra[k], rb[k], rtol=1e-5)
    assert torch.allclose(ra['cov'], rb['cov_xyz'], rtol=1e-4, atol=1e-9)


def test_converted_calibration(tub):
    _, mdir = tub
    _, _, cal = tn.load_model(mdir, 'cpu')
    S = json.load(open(ref('sigma_scale_quantile.json')))
    assert cal['xyz_matrix'] == S['xyz_matrix'] and cal['edges'] == S['edges'] and cal['method'] == 'quantile'


# ---------------------------------------------------------------------------------------------- 5. scenes
def _stats_simcode(q):
    """per-batch statistics of a SIMCODE Live packet (dict of numpy arrays)."""
    nb = q['frames'].shape[0]
    on = q['gt_onf'][:, 1] > 0
    return dict(emitters=len(q['gt_x']) / nb, on_centre=on.sum() / nb, labels=q['gt_label'].sum() / nb,
                photons_on=q['gt_phot'][on, 1], bright=q['gt_bright'], on_time=q['gt_t1'] - q['gt_t0'],
                onf_sum=q['gt_onf'].sum(1), bg=q['bg'].mean((1, 2, 3)), frame_mean=q['frames'].mean(),
                frame_sd=q['frames'].std(), z_label=q['gt_z'][q['gt_label']])


def _stats_torch_psf(sc, lab):
    g = {k: v.cpu().numpy() for k, v in sc['gt'].items()}
    nb = sc['frames'].shape[0]
    on = g['onf'][:, 1] > 0
    fr = sc['frames'].cpu().numpy()
    return dict(emitters=len(g['x']) / nb, on_centre=on.sum() / nb, labels=len(lab['x']) / nb,
                photons_on=g['phot'][on, 1], bright=g['bright'], on_time=g['t1'] - g['t0'], onf_sum=g['onf'].sum(1),
                bg=sc['bg'].cpu().numpy().mean((1, 2, 3)), frame_mean=fr.mean(), frame_sd=fr.std(),
                z_label=lab['z'].cpu().numpy())


def _summary(batches):
    """-> {statistic: (value, standard error over batches)} (per-batch means; pooled quantiles for distributions)."""
    out = {}
    for k in ('emitters', 'on_centre', 'labels', 'frame_mean', 'frame_sd'):
        v = np.array([b[k] for b in batches], np.float64)
        out[k] = (v.mean(), v.std() / np.sqrt(len(v)))
    for k in ('photons_on', 'bright', 'on_time', 'onf_sum', 'bg', 'z_label'):
        per = np.array([np.mean(b[k]) if len(b[k]) else np.nan for b in batches])
        per = per[np.isfinite(per)]
        allv = np.concatenate([b[k] for b in batches])
        out[k + '_mean'] = (per.mean(), per.std() / np.sqrt(len(per)))
        for q in (10, 50, 90):
            out[f'{k}_q{q}'] = (float(np.percentile(allv, q)), None)
    return out


@needs_gpu
def test_scene_statistics_match_simcode(simcode, tub, tmp_path):
    from examples.baselines.decode_replay import Live
    cfg, _ = tub
    a = json.load(open(ref('args.json')))
    a['config']['psf'] = [ref('tub3d_bead.psf.zip')]
    ap = tmp_path / 'args.json'
    ap.write_text(json.dumps(a))
    n = int(os.environ.get('TORCH_PSF_NN_STATS_BATCHES', '100'))
    live = Live(str(ap), 1234, 16, size=40, halo=0)
    A = _summary([_stats_simcode(q) for q in live.packets(n, seed=1234)])
    S = tn.Scenes(cfg['scenes'], 16, 40, 0, 'cuda', seed=1234)
    B = []
    for _ in range(n):
        sc = S.scene()
        B.append(_stats_torch_psf(sc, S.labels(sc)))
    B = _summary(B)
    print(f'\nscene statistics, {n} batches of 16 fields of 40 x 40 px each (value +- standard error over batches)')
    print(f"{'statistic':22s} {'SIMCODE Live':>22s} {'torch_psf.nn.Scenes':>22s} {'diff / se':>10s}")
    bad = []
    for k in A:
        (va, sa), (vb, sb) = A[k], B[k]
        if sa is None:
            print(f'{k:22s} {va:22.4g} {vb:22.4g} {"":>10s}')
            continue
        z = (vb - va) / np.hypot(sa, sb)
        print(f'{k:22s} {va:14.4g} +- {sa:<6.2g}{vb:14.4g} +- {sb:<6.2g}{z:10.2f}')
        if abs(z) > 4.5:
            bad.append((k, z))
    assert not bad, bad


# ---------------------------------------------------------------------------------------------- 6. training
@needs_gpu
def test_tubulin_smoke_training(tub, tmp_path):
    """The Tubulin network (full size, full_xyz) from scratch on the Tubulin scene settings for <= 1 minute."""
    cfg, _ = tub
    c = dict(cfg, out=str(tmp_path / 'run'), steps=250, log_every=50, val_batches=4, save_every=10 ** 6,
             device='cuda', resume=False)
    tn.train_model(c, progress=lambda s: None)
    L = [json.loads(s) for s in open(tmp_path / 'run' / 'train.log')]
    print('\n'.join(json.dumps({k: d.get(k) for k in ('step', 'train_loss', 'validation_loss', 's_per_step')})
                    for d in L))
    assert L[-1]['train_loss'] < L[1]['train_loss']
    assert L[-1]['validation_loss'] < L[0]['validation_loss']
