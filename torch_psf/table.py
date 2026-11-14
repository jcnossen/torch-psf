"""The localization table: a numpy structured array (one row per localization).
Positions in image px (x = column, y = row, pixel centres at integers), z in um, photons and bg (photons / px) in
photons; crlb_* are the Cramer-Rao standard deviations in the same units; loglik = -NLL of the fit (Poisson, constant
dropped); chi2 = mean squared normalized residual per pixel; at_limit = x or y at the model's offset limit."""
import numpy as np
import torch

LOC_DTYPE = np.dtype([('frame', 'i4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('photons', 'f4'), ('bg', 'f4'),
                      ('crlb_x', 'f4'), ('crlb_y', 'f4'), ('crlb_z', 'f4'), ('crlb_photons', 'f4'), ('crlb_bg', 'f4'),
                      ('loglik', 'f4'), ('chi2', 'f4'), ('at_limit', '?')])


def make_table(frame, xy, th, crlb, nll, chi2, at_limit):
    """Table from per-spot tensors: frame [n], image xy [n, 2], theta [n, 5], crlb [n, 5], nll [n], chi2 [n],
    at_limit [n]."""
    n = len(frame)
    L = np.zeros(n, LOC_DTYPE)
    c = lambda t: t.detach().cpu().numpy()
    L['frame'] = c(frame); L['x'] = c(xy[:, 0]); L['y'] = c(xy[:, 1]); L['z'] = c(th[:, 2])
    L['photons'] = c(th[:, 3]); L['bg'] = c(th[:, 4])
    for k, name in enumerate(('x', 'y', 'z', 'photons', 'bg')):
        L['crlb_' + name] = c(crlb[:, k])
    L['loglik'] = -c(nll); L['chi2'] = c(chi2); L['at_limit'] = c(at_limit)
    return L


def to_csv(table, path):
    """Write a localization table as CSV (one header line with the column names)."""
    np.savetxt(path, np.array(table.tolist(), dtype=np.float64).reshape(-1, len(table.dtype.names)), delimiter=',',
               header=','.join(table.dtype.names), comments='', fmt='%.6g')
