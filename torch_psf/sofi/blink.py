"""Blinking models for fluctuation (SOFI-type) data: continuous-time Markov chains integrated over the frame time.

States: 0 = off, 1 = on (emitting), optional 2 = long-lived dark state (entered from on, returns to off), optional
absorbing bleached state. Rates are per frame. The per-frame signal of an emitter is s_t in [0, 1], the fraction of
frame t it spends in the on state (so blinking faster than the frame time lowers Var[s], as in real cameras).

Parameterization used throughout (BlinkModel):
  duty     stationary on fraction of the off <-> on chain (without the dark state),
  t_on     mean on time (frames; < 1 means faster than the frame time),
  k_dark   on -> dark rate (per frame, 0 = two-state), t_dark mean dark time (frames),
  k_bleach on -> bleached rate (per frame, 0 = no bleaching).
"""
import math
from dataclasses import dataclass, asdict
import torch


@dataclass
class BlinkModel:
    duty: float = 0.3
    t_on: float = 2.0
    k_dark: float = 0.0
    t_dark: float = 100.0
    k_bleach: float = 0.0

    def rates(self):
        k_off = 1.0 / self.t_on
        k_on = k_off * self.duty / (1.0 - self.duty)
        return k_on, k_off

    def to_dict(self):
        return asdict(self)


def simulate_traces(model, n, T, substeps=None, generator=None, device='cuda', state=None, return_state=False):
    """Per-frame on fractions s [T, n] (float32) of n independent emitters, started in the stationary distribution of
    the off/on chain (or from `state` [n] int8 to continue a previous call). Euler sub-steps of the continuous-time
    chain: substeps per frame (default: enough that every rate * dt <= 0.2)."""
    k_on, k_off = model.rates()
    rmax = max(k_on, k_off, model.k_dark, 1.0 / model.t_dark if model.k_dark > 0 else 0.0, model.k_bleach, 1.0)
    S = int(substeps or max(4, math.ceil(rmax / 0.2)))
    dt = 1.0 / S
    p = lambda k: 1.0 - math.exp(-k * dt)
    p_on, p_off, p_dark, p_ret, p_bl = p(k_on), p(k_off), p(model.k_dark), \
        (p(1.0 / model.t_dark) if model.k_dark > 0 else 0.0), p(model.k_bleach)
    r = lambda: torch.rand(n, generator=generator, device=device)
    if state is None:
        st = (r() < model.duty).to(torch.int8)
    else:
        st = state.clone()
    out = torch.empty(T, n, device=device)
    for t in range(T):
        acc = torch.zeros(n, device=device)
        for _ in range(S):
            u = r()
            on = st == 1
            acc += on.float()
            # on -> off / dark / bleached (competing, small dt); off -> on; dark -> off
            nxt = st.clone()
            c1 = p_off
            c2 = c1 + p_dark
            c3 = c2 + p_bl
            nxt = torch.where(on & (u < c1), torch.zeros_like(st), nxt)
            nxt = torch.where(on & (u >= c1) & (u < c2), torch.full_like(st, 2), nxt)
            nxt = torch.where(on & (u >= c2) & (u < c3), torch.full_like(st, 3), nxt)
            nxt = torch.where((st == 0) & (u < p_on), torch.ones_like(st), nxt)
            nxt = torch.where((st == 2) & (u < p_ret), torch.zeros_like(st), nxt)
            st = nxt
        out[t] = acc * dt
    return (out, st) if return_state else out


@torch.no_grad()
def trace_stats(model, eps=1.0, n=4000, T=4000, max_lag=None, generator=None, device='cuda', seed=0):
    """Statistics of a = eps * s needed by the cumulant noise model (estimated from simulated traces, no bleaching):
    mean, var, r(tau) (autocorrelation, tau = 0..max_lag), alpha = sum_{tau=-inf..inf} r(tau)^2 (Gaussian long-run
    factor), beta = sum_tau r(tau) (long-run factor of the mean), k3 = E[(a - mu)^3] (zero lag), lrv4 = long-run
    variance of (a - mu)^2 (sum over all lags), c4 = lrv4 - 2 var^2 alpha (the non-Gaussian excess used in the
    covariance of second-order cumulant estimates)."""
    m = BlinkModel(**{**model.to_dict(), 'k_bleach': 0.0})
    g = generator or torch.Generator(device=device).manual_seed(seed)
    s = simulate_traces(m, n, T + 200, generator=g, device=device)[200:] * eps      # burn-in for the dark state
    mu = s.mean()
    x = s - mu
    var = (x ** 2).mean()
    L = int(max_lag or min(T // 4, max(50, int(20 * (m.t_on + (m.t_dark if m.k_dark > 0 else 0))))))
    X = torch.fft.rfft(x, n=2 * T, dim=0)
    ac = torch.fft.irfft(X * X.conj(), dim=0)[:L + 1].mean(1) / torch.arange(T, T - L - 1, -1, device=device)
    r = ac / ac[0]
    alpha = float(1 + 2 * (r[1:] ** 2).sum())
    beta = float(1 + 2 * r[1:].sum())
    y = x ** 2 - var
    Y = torch.fft.rfft(y, n=2 * T, dim=0)
    acy = torch.fft.irfft(Y * Y.conj(), dim=0)[:L + 1].mean(1) / torch.arange(T, T - L - 1, -1, device=device)
    lrv4 = float(acy[0] + 2 * acy[1:].sum())
    k3 = float((x ** 3).mean())
    return dict(mean=float(mu), var=float(var), r=r.cpu(), alpha=alpha, beta=beta, k3=k3, lrv4=lrv4,
                c4=lrv4 - 2 * float(var) ** 2 * alpha)
