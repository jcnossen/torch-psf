"""Vectorial PSF of an emitter inside a thin film on the coverslip, with another medium above the film (e.g. a dye in a
spin-coated PMMA film with air above; Hulleman et al. 2021).

Geometry: coverslip / immersion (n_imm, index-matched) | film (n_med, thickness film_um) | top medium (n_top). The
emitter is at height z_e above the coverslip inside the film (0 <= z_e <= film_um). Each plane-wave component of the
dipole field leaves the emitter downwards (straight to the coverslip) and upwards (reflected at the film / top
interface, r_10), with the multiple reflections in the film (1 / (1 - r_12 r_10 exp(2 i kz1 d))), per polarisation:

    E_pol = t_12 [(mu . p_down) exp(i kz1 z_e) + r_10 (mu . p_up) exp(i kz1 (2 d - z_e))] / (1 - r_12 r_10 exp(2 i kz1 d))

(s: mu . s for both). With n_top = n_med this is VectorialPSF; with film_um -> 0 it is VectorialPSF with n_med = n_top
(both checked in tests/test_orientation.py). Supercritical angles of the top medium (NA rho > n_top) are totally
reflected at the film top (|r_10| = 1), those of the film are evanescent in it.

The normalisation (energy) is that of VectorialPSF with the film index as the medium: an isotropic emitter at a
coverslip / film interface integrates to 1.
"""
import math
import torch
from torch_psf.vectorial import VectorialPSF


def _cos(n_a, s_a, n_b):
    """cos of the angle in medium n_b of a plane wave with sin s_a in medium n_a (complex; imag >= 0: evanescent)."""
    sb = (n_a * s_a / n_b).to(torch.complex64)
    c = torch.sqrt(1 - sb ** 2)
    return torch.where(c.imag < 0, -c, c)


class LayeredVectorialPSF(VectorialPSF):
    """VectorialPSF of an emitter in a film (index n_med, thickness film_um) on the coverslip under a top medium
    n_top (see the module docstring). film_um and n_top are parameters (film_um can be fitted:
    model.film_um.requires_grad_(True); it is used clamped to >= 0, and the emitter height to [0, film_um])."""
    type_name = 'vectorial_layered'

    def __init__(self, roisize, n_top=1.0, film_um=0.05, **kw):
        super().__init__(roisize, **kw)
        self.n_top = torch.nn.Parameter(torch.tensor(float(n_top), device=self.dev), requires_grad=False)
        self.film_um = torch.nn.Parameter(torch.tensor(float(film_um), device=self.dev), requires_grad=False)

    def _pol_parts(self):
        """Down / up pupil fields per polarisation: (Ps_down, Pp_down, Pp_up) [6, N, N] (Ps_up = Ps_down), the film
        kz, r_s, r_p at the film top and at the film / coverslip interface."""
        NA = self.NA
        k0 = 2 * math.pi / self.wavelength_um
        s_imm = NA * self.rho / self.n_imm
        c_imm = torch.sqrt((1 - s_imm ** 2).clamp(min=1e-6))
        n1 = self.n_med.to(torch.complex64)
        c1 = _cos(self.n_imm, s_imm, self.n_med)
        c1 = torch.where(c1.abs() < 1e-6, torch.full_like(c1, 1e-6j), c1)
        s1 = (NA * self.rho / self.n_med).to(torch.complex64)
        n2 = self.n_imm
        ts = 2 * n1 * c1 / (n1 * c1 + n2 * c_imm)
        tp = 2 * n1 * c1 / (n2 * c1 + n1 * c_imm)
        # reflection at the film / coverslip interface seen from the film (a wave going down, reflected up)
        r12s = (n1 * c1 - n2 * c_imm) / (n1 * c1 + n2 * c_imm)
        r12p = (n2 * c1 - n1 * c_imm) / (n2 * c1 + n1 * c_imm)
        # reflection at the film / top interface seen from the film
        n0 = self.n_top.to(torch.complex64)
        c0 = _cos(self.n_imm, s_imm, self.n_top)
        r10s = (n1 * c1 - n0 * c0) / (n1 * c1 + n0 * c0)
        r10p = (n0 * c1 - n1 * c0) / (n0 * c1 + n1 * c0)
        cphi, sphi = torch.cos(self.phi), torch.sin(self.phi)
        apod = self.mask / torch.sqrt(c_imm) * (n2 / n1) * c_imm / c1
        r2 = self.rho ** 2
        apod = apod * torch.exp(self.apod[0] * r2 + self.apod[1] * r2 ** 2 + self.apod[2] * r2 ** 3)
        apod = apod * self.pupil_amp
        # s part: ts (mu . s) (-sin phi, cos phi); p part: tp (mu . p) (cos phi, sin phi)
        # p_down = (c cos phi, c sin phi, -s); p_up = (-c cos phi, -c sin phi, -s)
        z = torch.zeros_like(cphi)
        Ps = torch.stack([ts * sphi * sphi, -ts * sphi * cphi, -ts * cphi * sphi, ts * cphi * cphi, z * ts, z * ts]) * apod
        pd = [c1 * cphi, c1 * sphi, -s1]
        pu = [-c1 * cphi, -c1 * sphi, -s1]
        mk = lambda pv: torch.stack([tp * pv[0] * cphi, tp * pv[0] * sphi, tp * pv[1] * cphi, tp * pv[1] * sphi,
                                     tp * pv[2] * cphi, tp * pv[2] * sphi]) * apod
        return Ps, mk(pd), mk(pu), k0 * self.n_med * c1, (r12s, r12p, r10s, r10p)

    def _emitter_pupil(self, P, ph, kz_med, ze):
        Ps, Ppd, Ppu, kz1, (r12s, r12p, r10s, r10p) = self._pol_parts()
        d = self.film_um.clamp_min(0)
        ze = torch.minimum(ze.clamp_min(0), d) if torch.is_tensor(ze) else ze
        ed = torch.exp(1j * kz1 * ze)                         # [B, N, N]
        eu = torch.exp(1j * kz1 * (2 * d - ze))
        e2 = torch.exp(2j * kz1 * d)
        Ds, Dp = 1 - r12s * r10s * e2, 1 - r12p * r10p * e2   # [N, N]
        S = Ps[None] * ((ed + r10s * eu) / Ds)[:, None]
        Pp = (Ppd[None] * ed[:, None] + (r10p * Ppu)[None] * eu[:, None]) / Dp
        out = (S + Pp) * ph[:, None]
        dd = self.diat
        if dd is not None and (torch.is_tensor(dd) or dd != 0):
            dd = dd if torch.is_tensor(dd) else torch.tensor(float(dd), device=self.dev)
            wx, wy = torch.sqrt(1 + dd), torch.sqrt(1 - dd)
            out = out * torch.stack([wx, wy, wx, wy, wx, wy]).to(out.real.dtype)[:, None, None]
        if bool((self.saf != 0).any()):
            raise NotImplementedError('LayeredVectorialPSF: saf terms are not supported')
        return out

    def resized(self, roisize, npupil=None, oversample=None, device=None):
        v = super().resized(roisize, npupil, oversample, device)
        new = LayeredVectorialPSF.__new__(LayeredVectorialPSF)
        new.__dict__.update(v.__dict__)
        with torch.no_grad():
            new.n_top = torch.nn.Parameter(self.n_top.detach().clone().to(v.dev), requires_grad=False)
            new.film_um = torch.nn.Parameter(self.film_um.detach().clone().to(v.dev),
                                             requires_grad=self.film_um.requires_grad)
        return new

    def _file_dict(self, source=None):
        d, arrays = super()._file_dict(source)
        d.update(n_top=float(self.n_top), film_um=float(self.film_um))
        return d, arrays

    @classmethod
    def _from_file(cls, d, arrays, roisize=None, device=None, **kw):
        v = VectorialPSF._from_file.__func__(cls, d, arrays, roisize=roisize, device=device, **kw)
        with torch.no_grad():
            v.n_top.fill_(float(d.get('n_top', 1.0)))
            v.film_um.fill_(float(d.get('film_um', 0.05)))
        return v

    def __repr__(self):
        return super().__repr__()[:-1] + f', n_top={float(self.n_top):.3f}, film_um={float(self.film_um):.4f})'
