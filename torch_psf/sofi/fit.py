"""Low-level fit (CumulantFit) of the PSF (Zernikes) to second-order cross-cumulants of a dense blinking movie.

    minimize_{theta, rho >= 0}  sum_{d, i} w_d(i) (G_hat_d(i) - [sum_p rho_p (*) K_{d,p}(theta)](i))^2

theta = Noll coefficients (+ a common z shift dz of the sample: the defocus gauge), rho = non-negative density on the
sub-pixel grid (model.py). Alternating: FISTA (projected, non-negative) steps on rho, then a Levenberg-Marquardt step
on theta with rho fixed (Jacobian by forward-mode autodiff through the vectorial PSF), accepted only if the loss
decreases (method='lm', default); method='adam' uses Adam steps with the gradient at the current rho instead.
theta starts at the given PSF's coefficients (e.g. 0 plus a known astigmatic bias); rho starts from a non-negative
least-squares fit at that theta.

Weights: w = 1 / Var(G_hat) per entry, e.g. from CumulantAccumulator.bootstrap (smoothed over pixels), or None
(uniform per separation).

This is the real-space variable projection of theory note section 6, step 3 (the Fourier closed-form elimination of
rho is not needed: the model is a small convolution and rho >= 0 is kept throughout).
"""
import torch
from torch_psf.sofi.model import GridPSF, kernels, model_G, model_G_adjoint


def smooth_variance(V, k=5):
    """Box-smooth per-entry variances [Nd, H, W] (NaN = invalid) over pixels, per separation."""
    m = torch.isfinite(V)
    Vz = torch.where(m, V, torch.zeros_like(V))
    ker = torch.ones(1, 1, k, k, device=V.device, dtype=V.dtype)
    num = torch.nn.functional.conv2d(Vz[:, None], ker, padding=k // 2)[:, 0]
    den = torch.nn.functional.conv2d(m[:, None].to(V.dtype), ker, padding=k // 2)[:, 0]
    return torch.where(m, num / den.clamp(min=1), torch.full_like(V, float('nan')))


class CumulantFit:
    def __init__(self, psf, Q=13, os=2, z_layers=(0.0,), focus=None, fit_dz=True, fit=None, device=None):
        """psf: VectorialPSF with the optics and the starting coefficients. fit: indices into psf.zcoef to estimate
        (default all)."""
        self.gp = GridPSF(psf, Q, os, z_layers, focus)
        self.Q, self.os = Q, os
        self.fit_dz = fit_dz
        self.dev = self.gp.dev
        nz = len(psf.zcoef)
        self.fit_idx = torch.arange(nz, device=self.dev) if fit is None else torch.as_tensor(fit, device=self.dev)
        self.history = []

    def _K(self, zc, dz, D):
        return kernels(self.gp.h(zc, dz).double(), D)

    def _lipschitz(self, K, w, shape, iters=30):
        x = torch.rand(shape, device=self.dev, dtype=torch.float64)
        for _ in range(iters):
            y = model_G_adjoint(w * model_G(x, K), K)
            lam = y.norm() / x.norm()
            x = y / y.norm()
        return 2 * float(lam) * 1.05

    def _fista(self, rho, K, Gh, w, L, n):
        """n projected FISTA steps on rho >= 0 for fixed K."""
        x, y, t = rho, rho, 1.0
        for _ in range(n):
            r = model_G(y, K) - Gh
            grad = 2 * model_G_adjoint(w * r, K)
            xn = (y - grad / L).clamp(min=0)
            tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
            y = xn + (t - 1) / tn * (xn - x)
            x, t = xn, tn
        return x

    def fit(self, G, mask, var=None, n_outer=None, n_inner=None, n_init=300, lr=3e-3, lr_end=3e-4, D=None,
            verbose=False, method='lm', tol=1e-7, log=None, stop=None):
        if method == 'lm':
            return self.fit_lm(G, mask, var, n_outer or 40, n_inner or 60, n_init, D, verbose, tol, log, stop)
        n_outer, n_inner = n_outer or 400, n_inner or 10
        """G [Nd, H, W] (NaN where invalid), mask [Nd, H, W] bool, var [Nd, H, W] entry variances or None.
        Returns dict(zcoef [nz] (all coefficients, fitted ones updated), dz (um), rho, loss)."""
        dev = self.dev
        Nd, H, W = G.shape
        Gh = torch.where(mask, G, torch.zeros_like(G)).double()
        if var is None:
            w = mask.double()
            sc = torch.stack([Gh[k][mask[k]].var() for k in range(Nd)])
            w = w / sc[:, None, None]
        else:
            w = torch.where(mask & torch.isfinite(var), 1.0 / var.clamp(min=1e-30), torch.zeros_like(var)).double()
        w = w / w[mask].mean()
        zc0 = self.gp.psf.zcoef.detach().clone()
        theta = torch.zeros(len(self.fit_idx) + 1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (lr_end / lr) ** (i / max(1, n_outer - 1)))

        def unpack():
            zc = zc0.clone()
            zc = zc.index_put((self.fit_idx,), zc0[self.fit_idx] + theta[:-1])
            dz = theta[-1] if self.fit_dz else theta[-1] * 0
            return zc, dz

        Np = self.gp.Np
        shape = (Np, H + self.Q - 1, W + self.Q - 1)
        with torch.no_grad():
            K = self._K(*unpack(), D)
            L = self._lipschitz(K, w, shape)
            rho = self._fista(torch.zeros(shape, device=dev, dtype=torch.float64), K, Gh, w, L, n_init)
        for it in range(n_outer):
            with torch.no_grad():
                K = self._K(*unpack(), D)
                if it % 50 == 0:
                    L = self._lipschitz(K, w, shape)
                rho = self._fista(rho, K, Gh, w, L, n_inner)
            opt.zero_grad()
            zc, dz = unpack()
            r = model_G(rho, self._K(zc, dz, D)) - Gh
            loss = (w * r * r).sum()
            loss.backward()
            opt.step()
            sched.step()
            if it % 20 == 0 or it == n_outer - 1:
                self.history.append(dict(it=it, loss=float(loss.detach()), theta=theta.detach().cpu().numpy().copy()))
                if verbose:
                    print(f'  it {it:4d} loss {float(loss):.6e}')
        zc, dz = unpack()
        return dict(zcoef=zc.detach(), dz=float(dz), rho=rho, loss=float(loss), w=w)

    def _weights(self, G, mask, var):
        Nd = G.shape[0]
        Gh = torch.where(mask, G, torch.zeros_like(G)).double()
        if var is None:
            sc = torch.stack([Gh[k][mask[k]].var() for k in range(Nd)])
            w = mask.double() / sc[:, None, None]
        else:
            w = torch.where(mask & torch.isfinite(var), 1.0 / var.clamp(min=1e-30), torch.zeros_like(var)).double()
        return Gh, w / w[mask].mean()

    def fit_lm(self, G, mask, var=None, n_outer=40, n_inner=60, n_init=300, D=None, verbose=False, tol=1e-7,
               log=None, stop=None):
        from torch.func import jvp
        dev = self.dev
        Nd, H, W = G.shape
        Gh, w = self._weights(G, mask, var)
        sw = torch.sqrt(w)
        zc0 = self.gp.psf.zcoef.detach().clone()
        nt = len(self.fit_idx) + (1 if self.fit_dz else 0)
        theta = torch.zeros(nt, device=dev, dtype=torch.float32)

        def unpack(th):
            zc = zc0.index_put((self.fit_idx,), zc0[self.fit_idx] + th[:len(self.fit_idx)])
            dz = th[-1] if self.fit_dz else torch.zeros((), device=dev)
            return zc, dz

        def resid(th, rho):
            zc, dz = unpack(th)
            return sw * (model_G(rho, self._K(zc, dz, D)) - Gh)

        Np = self.gp.Np
        shape = (Np, H + self.Q - 1, W + self.Q - 1)
        K = self._K(*unpack(theta), D)
        L = self._lipschitz(K, w, shape)
        rho = self._fista(torch.zeros(shape, device=dev, dtype=torch.float64), K, Gh, w, L, n_init)
        lam = 1e-3
        loss = float((resid(theta, rho) ** 2).sum())
        for it in range(n_outer):
            # theta: one LM step with rho fixed
            r0 = resid(theta, rho)
            cols = []
            for k in range(nt):
                e = torch.zeros(nt, device=dev)
                e[k] = 1
                cols.append(jvp(lambda t: resid(t, rho), (theta,), (e,))[1].reshape(-1))
            Jm = torch.stack(cols, 1)                                  # [Ndata, nt]
            JtJ, Jtr = Jm.T @ Jm, Jm.T @ r0.reshape(-1)
            f0 = float((r0 ** 2).sum())
            for _ in range(8):
                A = JtJ + lam * torch.diag(torch.diag(JtJ))
                step = -torch.linalg.solve(A, Jtr)
                th_new = theta + step.float()
                f1 = float((resid(th_new, rho) ** 2).sum())
                if f1 < f0:
                    theta, lam = th_new, max(lam / 3, 1e-6)
                    break
                lam *= 4
            # rho: FISTA steps at the new theta
            K = self._K(*unpack(theta), D)
            if it % 10 == 0:
                L = self._lipschitz(K, w, shape)
            rho = self._fista(rho, K, Gh, w, L, n_inner)
            new = float((resid(theta, rho) ** 2).sum())
            self.history.append(dict(it=it, loss=new, theta=theta.cpu().numpy().copy(), lam=lam))
            if verbose:
                print(f'  it {it:3d} loss {new:.6e} lam {lam:.1e} |step| {float(step.norm()):.2e}')
            if log is not None:
                log(it, new, 0.0)
            if stop is not None and stop():
                break
            if abs(loss - new) < tol * loss and float(step.norm()) < 1e-5:
                break
            loss = new
        zc, dz = unpack(theta)
        return dict(zcoef=zc.detach(), dz=float(dz), rho=rho, loss=loss, w=w)
