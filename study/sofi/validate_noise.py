"""Step 3a: the analytic covariance Sigma of the cross-cumulant estimates (torch_psf.sofi.fisher.noise_cov) against
(a) Monte-Carlo: K independent movies of the same sample (independent blinking and noise), and
(b) a block bootstrap on one long movie (what one would do on real data).
Tests: per-entry variance ratio, the whitened chi^2 (tests the full matrix: E = M (1 - 1/K) if Sigma is right) and
the variance along the leading eigenvectors of Sigma. Scenarios: dense SOFI-like (duty 0.3) and dense SMLM-like
(duty 0.02, bright, long dark state).

python validate_noise.py
"""
import argparse
from common import *
from torch_psf.sofi.fisher import patch_matrix, pairs, noise_cov

ap = argparse.ArgumentParser()
ap.add_argument('--K', type=int, default=256)
ap.add_argument('--T', type=int, default=2000)
ap.add_argument('--P', type=int, default=10)
a = ap.parse_args()
torch.manual_seed(0)
Q, os_, R = 13, 2, 5

SCEN = {
    'sofi': dict(blink=BlinkModel(duty=0.3, t_on=2.0), eps=300.0, density=80.0, bg=60.0, rn=1.5, emccd=False),
    'smlm_dense': dict(blink=BlinkModel(duty=0.02, t_on=3.0, k_dark=0.05, t_dark=200.0), eps=3000.0, density=400.0,
                       bg=40.0, rn=1.5, emccd=False),
    'sofi_emccd': dict(blink=BlinkModel(duty=0.3, t_on=0.5), eps=300.0, density=80.0, bg=60.0, rn=0.0, emccd=True),
}


def run(name, sc, P, T, K):
    psf = make_psf(Q, random_aberrations(0.05, seed=2), astig=0.08)
    h = GridPSF(psf, Q, os_, (0.0,)).h().double()
    Hm = patch_matrix(h, P)                                   # [P*P, Ng]
    Ng = Hm.shape[1]
    px_um2 = psf.pixel_um ** 2 / os_ ** 2
    g = torch.Generator(device=DEV).manual_seed(11)
    cnt = torch.poisson(torch.full((Ng,), sc['density'] * px_um2, device=DEV, dtype=torch.float64), generator=g)
    em = torch.repeat_interleave(torch.arange(Ng, device=DEV), cnt.long())          # grid index per emitter
    n = len(em)
    st = sofi.trace_stats(sc['blink'], n=2000, T=8000)
    eps = sc['eps']
    rho = cnt * eps ** 2 * st['var']
    k3 = cnt * eps ** 3 * st['k3']
    c4 = cnt * eps ** 4 * st['c4']
    F2 = 2.0 if sc['emccd'] else 1.0
    mean = sc['bg'] + (Hm * (cnt * eps * st['mean'])[None]).sum(1)
    Nvar = F2 * mean + sc['rn'] ** 2
    I, J, _, _ = pairs(P, R)
    M = len(I)
    Sig, _, _ = noise_cov(Hm, rho, k3, c4, Nvar, I, J, T, st['alpha'], F2)
    SigG, _, _ = noise_cov(Hm, rho, None, None, Nvar, I, J, T, st['alpha'], F2)     # Gaussian part only
    L = torch.linalg.cholesky(Sig)
    Hn = Hm[:, em].float()                                    # [P*P, n] footprints of the emitters
    # (a) Monte Carlo: K movies, each T frames, same emitters
    gs = []
    for k0 in range(0, K, 32):
        kb = min(32, K - k0)
        s1 = torch.zeros(kb, P * P, device=DEV, dtype=torch.float64)
        s2 = torch.zeros(kb, M, device=DEV, dtype=torch.float64)
        state = None
        for t0 in range(0, T, 500):
            tn = min(500, T - t0)
            tr, state = sofi.simulate_traces(sc['blink'], kb * n, tn, generator=g, device=DEV, state=state,
                                              return_state=True)
            mu = sc['bg'] + (tr.view(tn, kb, n) * eps) @ Hn.T                       # [t, kb, P*P]
            ph = torch.poisson(mu, generator=g)
            if sc['emccd']:
                ph = torch.where(ph > 0, torch._standard_gamma(ph.clamp(min=1e-6)), torch.zeros_like(ph))
            F = (ph + sc['rn'] * torch.randn(ph.shape, generator=g, device=DEV)).double()
            s1 += F.sum(0)
            s2 += (F[:, :, I] * F[:, :, J]).sum(0)
        m = s1 / T
        gs.append((s2 / T - m[:, I] * m[:, J]) * T / (T - 1))
    gk = torch.cat(gs)                                        # [K, M]
    dev_ = gk - gk.mean(0)
    v_mc = (dev_ ** 2).sum(0) / (K - 1)
    ratio = v_mc / torch.diag(Sig)
    ratioG = v_mc / torch.diag(SigG)
    w = torch.linalg.solve_triangular(L, dev_.T, upper=False)
    chi2 = float((w ** 2).sum() / (K - 1))                    # E = M
    e, V = torch.linalg.eigh(Sig)
    top = V[:, -20:]
    v_top = ((dev_ @ top) ** 2).sum(0) / (K - 1) / e[-20:]
    out = dict(M=M, n_emitters=n, K=K, T=T, alpha=st['alpha'], var_ratio_median=float(ratio.median()),
               var_ratio_q10_q90=[float(ratio.quantile(0.1)), float(ratio.quantile(0.9))],
               var_ratio_gaussian_only_median=float(ratioG.median()),
               chi2_per_dof=chi2 / M, top20_eig_var_ratio=[float(v_top.min()), float(v_top.median()), float(v_top.max())])
    # (b) block bootstrap on one long movie (Tb = 20 T frames, blocks of 500)
    Tb, blk = 20 * T, 500
    acc_g, state = [], None
    for t0 in range(0, Tb, blk):
        tr, state = sofi.simulate_traces(sc['blink'], n, blk, generator=g, device=DEV, state=state, return_state=True)
        mu = sc['bg'] + (tr * eps) @ Hn.T
        ph = torch.poisson(mu, generator=g)
        if sc['emccd']:
            ph = torch.where(ph > 0, torch._standard_gamma(ph.clamp(min=1e-6)), torch.zeros_like(ph))
        F = (ph + sc['rn'] * torch.randn(ph.shape, generator=g, device=DEV)).double()
        mb = F.mean(0)
        acc_g.append(((F[:, I] * F[:, J]).mean(0) - mb[I] * mb[J]) * blk / (blk - 1))
    Gb = torch.stack(acc_g)                                   # [nblocks, M]
    nb = len(Gb)
    gb = torch.Generator(device='cpu').manual_seed(5)
    reps = torch.stack([Gb[torch.randint(nb, (nb,), generator=gb).to(DEV)].mean(0) for _ in range(400)])
    v_bs = reps.var(0)
    SigB = Sig * T / Tb
    rb = v_bs / torch.diag(SigB)
    out.update(bootstrap_blocks=nb, bootstrap_var_ratio_median=float(rb.median()),
               bootstrap_var_ratio_q10_q90=[float(rb.quantile(0.1)), float(rb.quantile(0.9))])
    print(name, json.dumps(out, indent=None))
    return out


res = {nm: run(nm, sc, a.P, a.T, a.K) for nm, sc in SCEN.items()}
save_json(res, 'validate_noise.json')
