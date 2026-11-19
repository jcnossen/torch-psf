"""Tensor convention shared by dipole-aware channel models and spline tables."""
import torch


IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _m6(M):
    """Equals orientation.M_to_m6. Replace with that function when the branches are merged."""
    sym = (M + M.transpose(-1, -2)) / 2
    return torch.stack([sym[..., i, j] for i, j in IJ], -1)


def unit_tensors(device):
    units = torch.zeros(6, 3, 3, device=device)
    for k, (i, j) in enumerate(IJ):
        units[k, i, j] = units[k, j, i] = 1
    return units


def default_weights(M=None, dipw=None):
    if M is None:
        w = torch.ones(3) if dipw is None else torch.as_tensor(dipw)
        M = torch.diag(w) / 3
    M = torch.as_tensor(M)
    if M.shape != (3, 3):
        raise ValueError('a spline needs one default dipole tensor per table')
    return _m6(M).detach()


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
        weights = _m6(M).reshape(-1, 6, 1, 1)
        images = self.dipole_basis(pos, grad)
        if grad:
            return tuple((v * weights).sum(1) for v in images)
        return (images * weights).sum(1)
