"""Tensor convention shared by dipole-aware channel models and splines."""
import torch


IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def M_to_m6(M):
    """[..., 3, 3] -> [..., 6] (xx, yy, zz, xy, xz, yz) of the symmetric part."""
    S = (M + M.transpose(-1, -2)) / 2
    return torch.stack([S[..., i, j] for i, j in IJ], -1)


def m6_to_M(m):
    """[..., 6] -> symmetric [..., 3, 3]."""
    M = m.new_zeros(*m.shape[:-1], 3, 3)
    for k, (i, j) in enumerate(IJ):
        M[..., i, j] = m[..., k]
        M[..., j, i] = m[..., k]
    return M


def unit_tensors(device):
    return m6_to_M(torch.eye(6, device=device))


def _real_inner(x, yr, yi, axes):
    """Re(x . conj(y)), summed over the requested field components."""
    product = x.real * yr + x.imag * yi
    return product.sum(axes) if axes else product


def contract_fields(x, M=None, *, y=None, basis=False):
    """Real dipole bilinear form for fields [N, 3, (pol,), n, n]. An omitted y
    gives intensity. Return its six symmetric coefficients when basis=True,
    otherwise contract with M (isotropic if omitted)."""
    same_fields = y is None
    y = x if same_fields else y
    axes = tuple(range(1, x.ndim - 2))  # dipoles and any polarization axis
    if basis:
        def coefficient(i, j):
            return _real_inner(x[:, i], y[:, j].real, y[:, j].imag, (1,) if x.ndim == 5 else ())

        terms = []
        for i, j in IJ:
            term = coefficient(i, j)
            if i != j:
                term = 2 * term if same_fields else term + coefficient(j, i)
            terms.append(term)
        return torch.stack(terms, 1)
    if M is None:
        return _real_inner(x, y.real, y.imag, axes) / 3
    eq = 'ij,bj...->bi...' if M.ndim == 2 else 'bij,bj...->bi...'
    M = M.to(x.real.dtype)
    yr = torch.einsum(eq, M, y.real)
    yi = torch.einsum(eq, M, y.imag)
    return _real_inner(x, yr, yi, axes)


def default_weights(M=None, dipw=None):
    if M is None:
        w = torch.ones(3) if dipw is None else torch.as_tensor(dipw)
        M = torch.diag(w) / 3
    M = torch.as_tensor(M)
    if M.shape != (3, 3):
        raise ValueError('a spline needs one default dipole tensor per spline')
    return M_to_m6(M).detach()


def combine_splines(weights, tables):
    return torch.einsum('...k,...kzyx->...zyx', weights, tables)
