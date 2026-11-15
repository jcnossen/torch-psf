"""Zernike polynomials (Noll index, Noll normalization: rms 1 over the unit disk)."""
import math
import torch


def noll_to_nm(j):
    """Noll index j (1-based) -> radial order n and azimuthal frequency m (m > 0: cosine term, m < 0: sine term)."""
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def zernike(j, rho, phi):
    """Noll-normalized Zernike polynomial Z_j(rho, phi) (rms 1 over the unit disk). rho, phi: tensors."""
    n, m = noll_to_nm(j)
    am = abs(m)
    R = torch.zeros_like(rho)
    for s in range((n - am) // 2 + 1):
        c = (-1) ** s * math.factorial(n - s) / (
            math.factorial(s) * math.factorial((n + am) // 2 - s) * math.factorial((n - am) // 2 - s))
        R = R + c * rho ** (n - 2 * s)
    if m == 0:
        return math.sqrt(n + 1) * R
    f = torch.cos(am * phi) if m > 0 else torch.sin(am * phi)
    return math.sqrt(2 * (n + 1)) * R * f


# names of the low orders (Noll), for user interfaces and reports
NOLL_NAMES = {1: 'piston', 2: 'tilt x', 3: 'tilt y', 4: 'defocus', 5: 'astigmatism 45', 6: 'astigmatism 0',
              7: 'coma y', 8: 'coma x', 9: 'trefoil y', 10: 'trefoil x', 11: 'spherical',
              12: '2nd astigmatism 0', 13: '2nd astigmatism 45', 14: 'quadrafoil 0', 15: 'quadrafoil 45',
              16: '2nd coma x', 17: '2nd coma y', 18: '2nd trefoil x', 19: '2nd trefoil y', 20: 'pentafoil x',
              21: 'pentafoil y', 22: '2nd spherical'}
