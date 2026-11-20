"""Non-square camera pixels (pixel_um = (x, y)): the model equals a square model on a finer grid, files keep the pair,
fits report both pixel sizes, and a bead fit recovers the aspect."""
import json

import numpy as np
import pytest
import torch

from torch_psf import Dataset, FourPiPSF, Optics, PSF, VectorialPSF
from torch_psf.estimation.beads import fit_beads, stage_focus
from torch_psf.optics import pixel_size
from torch_psf.spline import to_spline

PX, PY, FINE = 0.1, 0.15, 0.05          # x = 2 fine px, y = 3 fine px
POS = torch.tensor([[0.3, -0.4, 0.2], [-0.7, 0.55, -0.1]])


def vec(R, pixel, mode, os, seed=0):
    v = VectorialPSF(R, pixel_um=pixel, device='cpu', oversample=os, pixel_integration=mode, z_range=(-0.5, 0.5))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        v.zcoef.copy_(torch.randn(len(v.zidx), generator=g) * 0.03)
    return v


def fine_blocks(render_fine, R):
    """A fine square model rendered so that blocks of 2 x 3 fine pixels are the R x R non-square pixels: their sums."""
    Rf = 3 * R
    c, h, off = (Rf - 1) / 2, (R - 1) / 2, (Rf - 2 * R) // 2
    p = POS * torch.tensor([PX / FINE, PY / FINE, 1.0])
    p[:, 0] += off + 0.5 - c + 2 * h              # x blocks of two fine px are centred between fine pixel centres
    b = render_fine(p)[:, :, off:off + 2 * R]
    return b.reshape(len(POS), R, 3, R, 2).sum((2, 4))


def test_pixel_size_values():
    assert pixel_size(0.1) == 0.1 and pixel_size((0.1, 0.1)) == 0.1 and pixel_size([0.1, 0.2]) == (0.1, 0.2)
    with pytest.raises(ValueError):
        pixel_size((0.1, 0.2, 0.3))
    o = Optics(pixel_um=(0.117, 0.127))
    assert o.to_dict()['pixel_um'] == [0.117, 0.127] and Optics.from_dict(o.to_dict()) == o
    assert Optics.from_dict(dict(pixelsize_um=[0.1, 0.1])).pixel_um == 0.1
    assert isinstance(Optics().to_dict()['pixel_um'], float)


def test_point_sampling_equals_fine_square_grid():
    R = 11
    ns = vec(R, (PX, PY), 'midpoint', 1)
    Rs = 3 * (R - 1) + 1
    sq = vec(Rs, FINE, 'midpoint', 1)
    a = ns.render(POS)
    b = sq.render(POS * torch.tensor([PX / FINE, PY / FINE, 1.0]))
    c, h = (Rs - 1) // 2, (R - 1) // 2
    b = b[:, c + 3 * torch.arange(-h, h + 1)][:, :, c + 2 * torch.arange(-h, h + 1)] * (PX * PY / FINE ** 2)
    assert float((a - b).abs().max() / a.max()) < 1e-6


@pytest.mark.parametrize('mode,os,tol', [('exact', 3, 2e-5), ('gauss_legendre', 6, 1e-6)])
def test_pixel_integral_equals_fine_block_sums(mode, os, tol):
    R = 11
    a = vec(R, (PX, PY), mode, os).render(POS)
    b = fine_blocks(vec(3 * R, FINE, mode, 3).render, R)
    assert float((a - b).abs().max() / a.max()) < tol


def test_fourpi_equals_fine_block_sums():
    R = 11
    pos = POS.clone()
    mk = lambda R, p: FourPiPSF(R, pixel_um=p, device='cpu', z_range=(-0.4, 0.4), blur_px2=1e-12, oversample=3)
    a = mk(R, (PX, PY)).render(pos)
    f = mk(3 * R, FINE)
    b = torch.stack([fine_blocks(lambda p: f.render(p)[:, c], R) for c in range(a.shape[1])], 1)
    assert float((a - b).abs().max() / a.max()) < 2e-5


def test_derivatives_match_finite_differences():
    v = vec(13, (0.117, 0.127), 'exact', 3)
    img, *d = v.render(POS, grad=True)                             # image, d/dx, d/dy, d/dz
    e = 1e-3
    for k in range(3):
        dp = torch.zeros(3)
        dp[k] = e
        fd = (v.render(POS + dp) - v.render(POS - dp)) / (2 * e)
        assert float((d[k] - fd).abs().max() / fd.abs().max()) < 2e-2


def test_files_keep_the_pair(tmp_path):
    v = vec(13, (0.117, 0.127), 'exact', 3)
    p = tmp_path / 'ns.psf.zip'
    v.save(p)
    w = PSF.load(p, device='cpu')
    assert w.pixel_um == (0.117, 0.127)
    torch.testing.assert_close(w.render(POS), v.render(POS), rtol=0, atol=1e-7)
    s = vec(13, 0.1, 'exact', 3)
    s.save(tmp_path / 'sq.psf.zip')
    assert isinstance(PSF.load(tmp_path / 'sq.psf.zip', device='cpu').pixel_um, float)
    assert v.format_version() == (0, 6) and s.format_version() == (0, 3)      # older readers refuse the pair cleanly
    f4 = FourPiPSF(13, pixel_um=(0.117, 0.127), device='cpu', z_range=(-0.4, 0.4))
    f4.save(tmp_path / 'f4.psf.zip')
    g4 = PSF.load(tmp_path / 'f4.psf.zip', device='cpu')
    assert g4.pixel_um == (0.117, 0.127)
    torch.testing.assert_close(g4.render(POS), f4.render(POS), rtol=0, atol=1e-7)


def test_spline_of_a_nonsquare_model():
    v = vec(13, (0.117, 0.127), 'exact', 3)
    s = to_spline(v, (-0.4, 0.4), roisize=13)
    assert s.pixel_um == (0.117, 0.127)
    p = POS * torch.tensor([1.0, 1.0, 0.8])
    a, b = v.render(p), s.render(p)
    assert float((a - b).abs().max() / a.max()) < 2e-3


def test_fit_dataset_per_axis(tmp_path):
    v = vec(13, (0.117, 0.127), 'exact', 3)
    truth = torch.tensor([[0.2, -0.3, 0.1], [-0.4, 0.5, -0.2]]).repeat(10, 1)
    rois = torch.poisson(v.render(truth) * 3000 + 5, generator=torch.Generator().manual_seed(1))
    ds = v.fit(rois, iters=30, zstarts=3)
    assert ds.meta['pixel_nm'] == pytest.approx(117.0) and ds.meta['pixel_nm_y'] == pytest.approx(127.0)
    assert ds.pixel_nm_xy == pytest.approx((117.0, 127.0))
    z_um = ds.fields['pos'][:, 2] * 0.117                          # z is in px of the x pixel size
    assert float((z_um - truth[:, 2].double()).abs().median()) < 0.03
    nm = ds.pos_nm
    torch.testing.assert_close(nm[:, 1], ds.fields['pos'][:, 1] * 127.0)
    torch.testing.assert_close(nm[:, 2], ds.fields['pos'][:, 2] * 117.0)
    ds.save_csv(tmp_path / 'ns.csv')
    back = Dataset.load_csv(tmp_path / 'ns.csv', pixel_nm=(117.0, 127.0))
    assert back.meta['pixel_nm_y'] == 127.0
    np.testing.assert_allclose(back.fields['pos'].numpy(), ds.fields['pos'].numpy(), atol=1e-4)
    img = ds.render(pixel_nm=10.0, region=(-1, -1, 1, 1))
    assert img.shape == (26, 24)                                   # 2 px: 254 nm in y, 234 nm in x


def test_square_dataset_has_no_y_size():
    v = vec(13, 0.1, 'exact', 3)
    rois = torch.poisson(v.render(POS) * 2000 + 5, generator=torch.Generator().manual_seed(1))
    ds = v.fit(rois, iters=10, zstarts=3)
    assert 'pixel_nm_y' not in ds.meta and ds.pixel_nm_xy == (100.0, 100.0)


def test_bead_fit_recovers_the_aspect():
    true = vec(15, (0.112, 0.122), 'exact', 2, seed=3)                # aspect 1.0893
    nk, nb = 15, 4
    focus = stage_focus(nk, 0.08)
    g = torch.Generator().manual_seed(2)
    xy = (torch.rand(nb, 2, generator=g) - 0.5)
    pos = torch.cat([xy[:, None].expand(-1, nk, -1), torch.zeros(nb, nk, 1)], -1).reshape(-1, 3)
    with torch.no_grad():
        mu = true(pos, focus[None].expand(nb, -1).reshape(-1)) * 20000 + 10
    d = torch.poisson(mu, generator=g).reshape(nb, nk, 15, 15)
    start = vec(15, float(np.sqrt(0.112 * 0.122)), 'exact', 2, seed=3)
    with torch.no_grad():
        start.zcoef.zero_()
    r = fit_beads(d, focus, start, iters=250, fit_aspect=True)
    assert r['report']['pixel_aspect'] == pytest.approx(0.122 / 0.112, abs=0.006)
    px, py = r['psf'].pixel_um
    assert np.sqrt(px * py) == pytest.approx(np.sqrt(0.112 * 0.122), rel=1e-6)
    assert all(isinstance(v, float) for v in r['psf'].pixel_um)
