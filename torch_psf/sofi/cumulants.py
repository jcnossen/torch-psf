"""Streaming second-order cumulants of a movie, per block of frames.

Separations: D(R) = {d = (dy, dx) : 0 < |d| <= R, dy > 0 or (dy == 0 and dx > 0)} (one of each +-d pair; for zero
lag G(r, d) = G(r + d, -d)). For R = 5 that is 40 separations.

Per block, for every pixel r = (y, x) and d in D:
    G[d, y, x] = Cov(F_t(y, x), F_t(y + dy, x + dx))        (zero lag)
and optionally the lag-1 covariances for d in {0} u D u -D:
    L1[d, y, x] = Cov(F_t(y, x), F_{t+1}(y + dy, x + dx)).
Entries whose partner pixel lies outside the frame are NaN (`mask` gives the valid ones). The covariance of a block
is computed from the block's own mean (bleaching -> blocks of a few hundred frames). Results of several blocks are
averaged by `result()` (weighted by the number of frames) and are also kept per block for a bootstrap.
"""
import torch


def separations(R, half=True, include_zero=False):
    """List of (dy, dx) with 0 < dy^2 + dx^2 <= R^2 (one of each +-pair when half=True)."""
    out = []
    Ri = int(R)
    for dy in range(-Ri, Ri + 1):
        for dx in range(-Ri, Ri + 1):
            r2 = dy * dy + dx * dx
            if r2 > R * R or (r2 == 0 and not include_zero):
                continue
            if half and r2 > 0 and not (dy > 0 or (dy == 0 and dx > 0)):
                continue
            out.append((dy, dx))
    return out


def valid_mask(H, W, D, device='cuda'):
    """[Nd, H, W] bool: partner pixel (y + dy, x + dx) inside the frame."""
    yy = torch.arange(H, device=device)[:, None]
    xx = torch.arange(W, device=device)[None, :]
    return torch.stack([((yy + dy >= 0) & (yy + dy < H) & (xx + dx >= 0) & (xx + dx < W)) for dy, dx in D])


def _shift(F, dy, dx, R):
    """F padded by R on every side: view of the partner pixels (y + dy, x + dx) for all (y, x) of the frame."""
    t, Hp, Wp = F.shape
    H, W = Hp - 2 * R, Wp - 2 * R
    return F[:, R + dy:R + dy + H, R + dx:R + dx + W]


class CumulantAccumulator:
    def __init__(self, H, W, R=5, block=500, lag1=False, device='cuda', dtype=torch.float64):
        self.H, self.W, self.R, self.block = H, W, int(R), block
        self.dev, self.dt = torch.device(device), dtype
        self.D = separations(R)
        self.D1 = separations(R, half=False, include_zero=True) if lag1 else None
        self.mask = valid_mask(H, W, self.D, self.dev)
        self.blocks = []          # list of dict(n, mean, G, L1)
        self._reset()
        self._prev = None         # last frame of the previous chunk (lag-1 within a block)

    def _reset(self):
        H, W, dev, dt = self.H, self.W, self.dev, self.dt
        self.n = 0
        self.s1 = torch.zeros(H, W, device=dev, dtype=dt)
        self.s2 = torch.zeros(len(self.D), H, W, device=dev, dtype=dt)
        if self.D1 is not None:
            self.n1 = 0
            self.a1 = torch.zeros(H, W, device=dev, dtype=dt)      # sum of F_t over the lag pairs (first frame)
            self.b1 = torch.zeros(H, W, device=dev, dtype=dt)      # sum of F_{t+1}
            self.l1 = torch.zeros(len(self.D1), H, W, device=dev, dtype=dt)

    def _accumulate(self, F):
        R = self.R
        Fp = torch.nn.functional.pad(F, (R, R, R, R))
        self.n += F.shape[0]
        self.s1 += F.sum(0)
        for k, (dy, dx) in enumerate(self.D):
            self.s2[k] += (F * _shift(Fp, dy, dx, R)).sum(0)
        if self.D1 is not None:
            Fa = F if self._prev is None else torch.cat([self._prev[None], F])
            if Fa.shape[0] > 1:
                A, B = Fa[:-1], Fa[1:]
                Bp = torch.nn.functional.pad(B, (R, R, R, R))
                self.n1 += A.shape[0]
                self.a1 += A.sum(0)
                self.b1 += B.sum(0)    # partner sums are taken from the shifted b1 in _close
                for k, (dy, dx) in enumerate(self.D1):
                    self.l1[k] += (A * _shift(Bp, dy, dx, R)).sum(0)
            self._prev = F[-1].clone()

    def _close(self):
        if self.n < 2:
            self._reset()
            return
        R = self.R
        m = self.s1 / self.n
        mp = torch.nn.functional.pad(m[None], (R, R, R, R))[0]
        G = torch.stack([self.s2[k] / self.n - m * mp[R + dy:R + dy + self.H, R + dx:R + dx + self.W]
                         for k, (dy, dx) in enumerate(self.D)]) * (self.n / (self.n - 1))
        G = torch.where(self.mask, G, torch.full_like(G, float('nan')))
        blk = dict(n=self.n, mean=m, G=G)
        if self.D1 is not None and self.n1 > 1:
            ma, mb = self.a1 / self.n1, self.b1 / self.n1
            mbp = torch.nn.functional.pad(mb[None], (R, R, R, R))[0]
            L = torch.stack([self.l1[k] / self.n1 - ma * mbp[R + dy:R + dy + self.H, R + dx:R + dx + self.W]
                             for k, (dy, dx) in enumerate(self.D1)])
            m1 = valid_mask(self.H, self.W, self.D1, self.dev)
            blk['L1'] = torch.where(m1, L, torch.full_like(L, float('nan')))
        self.blocks.append(blk)
        self._prev = None
        self._reset()

    def update(self, F):
        """Add frames F [t, H, W] (photons; any float dtype). Blocks close every `block` frames."""
        F = F.to(self.dt)
        i = 0
        while i < F.shape[0]:
            take = min(F.shape[0] - i, self.block - self.n)
            self._accumulate(F[i:i + take])
            i += take
            if self.n >= self.block:
                self._close()
        return self

    def finish(self):
        if self.n >= 2:
            self._close()
        return self

    def result(self, blocks=None):
        """Frame-weighted average over blocks (all, or the given indices): dict(n, mean, G[, L1])."""
        bl = self.blocks if blocks is None else [self.blocks[i] for i in blocks]
        w = torch.tensor([b['n'] for b in bl], device=self.dev, dtype=self.dt)
        w = w / w.sum()
        out = dict(n=sum(b['n'] for b in bl), D=self.D, mask=self.mask)
        for k in ('mean', 'G', 'L1'):
            if k in bl[0]:
                out[k] = sum(wi * b[k] for wi, b in zip(w, bl))
        return out

    def bootstrap(self, nboot=200, generator=None, key='G'):
        """Block bootstrap replicates of result()[key]: [nboot, ...] (blocks resampled with replacement)."""
        nb = len(self.blocks)
        g = generator or torch.Generator(device='cpu').manual_seed(0)
        stack = torch.stack([b[key] for b in self.blocks])
        w = torch.tensor([b['n'] for b in self.blocks], device=self.dev, dtype=self.dt)
        out = []
        for _ in range(nboot):
            idx = torch.randint(nb, (nb,), generator=g).to(self.dev)
            ww = w[idx] / w[idx].sum()
            out.append((ww.view(-1, *([1] * (stack.dim() - 1))) * stack[idx]).sum(0))
        return torch.stack(out)
