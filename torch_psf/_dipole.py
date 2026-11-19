"""Tensor convention shared by dipole-aware channel models and spline tables."""
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
        raise ValueError('a spline needs one default dipole tensor per table')
    return M_to_m6(M).detach()


def combine_tables(weights, tables):
    return torch.einsum('...k,...kzyx->...zyx', weights, tables)


class DipoleSpline:
    """Optional six knot volumes on the spline's own grid, in xx, yy, zz, xy, xz, yz order."""
    dipole_knots = None
    component_weights = None

    @property
    def knots(self):
        if self._knots is not None:
            return self._knots
        return self._default_knots()

    @knots.setter
    def knots(self, value):
        self._knots = value
        self.component_weights = None

    def _default_knots(self):
        return combine_tables(self.component_weights, self.dipole_knots)

    def _set_dipole_knots(self, tables, weights=None):
        if tables is not None:
            tables = torch.as_tensor(tables, device=self.dev, dtype=self.dtype)
            if tables.shape != (6, self.nz, self.nv, self.nu):
                raise ValueError('expected six dipole tables on the knot grid')
        self.dipole_knots = tables
        if weights is not None:
            self.component_weights = torch.as_tensor(weights, device=self.dev, dtype=self.dtype)
            if tables is None or self.component_weights.shape != (6,):
                raise ValueError('expected six default weights and six dipole tables')
            self._knots = None

    def dipole_basis(self, pos, grad=False):
        """[N, 6, P, P], off-diagonal images include both symmetric terms."""
        tables = self.dipole_knots
        if tables is None:
            raise ValueError('build the spline with dipole_components=True to render a dipole tensor')
        images = [self.render(pos, grad, knots=table) for table in tables]
        if grad:
            return tuple(torch.stack(parts, 1) for parts in zip(*images))
        return torch.stack(images, 1)

    def _dipole_render(self, pos, grad, dipM, knots):
        if knots is not None:
            raise ValueError('provide either knots or dipM')
        M = torch.as_tensor(dipM, device=self.dev, dtype=self.dtype)
        if M.shape not in ((3, 3), (len(pos), 3, 3)):
            raise ValueError('dipM must have shape [3,3] or [N,3,3]')
        weights = M_to_m6(M).reshape(-1, 6, 1, 1)
        images = self.dipole_basis(pos, grad)
        if grad:
            return tuple((v * weights).sum(1) for v in images)
        return (images * weights).sum(1)
