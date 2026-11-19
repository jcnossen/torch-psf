"""LayeredVectorialPSF: an emitter in a film on the coverslip under another medium."""
import torch
from torch_psf import VectorialPSF
from torch_psf.layered import LayeredVectorialPSF

KW = dict(pixel_um=0.065, NA=1.47, n_imm=1.518, wavelength_um=0.6, npupil=48, oversample=2, device='cpu')
POS = torch.tensor([[0.1, -0.2, 0.0]])
DIP = {'x': torch.diag(torch.tensor([1.0, 0, 0])), 'y': torch.diag(torch.tensor([0.0, 1, 0])),
       'z': torch.diag(torch.tensor([0.0, 0, 1]))}


def test_no_top_interface_is_the_plain_model():
    L = LayeredVectorialPSF(15, n_med=1.49, n_top=1.49, film_um=0.05, **KW)
    b = VectorialPSF(15, n_med=1.49, **KW)
    p = POS + torch.tensor([[0.0, 0.0, 0.02]])
    with torch.no_grad():
        for M in DIP.values():
            a, c = L(p, 0.1, dipM=M), b(p, 0.1, dipM=M)
            assert float((a - c).abs().max()) < 1e-5 * float(c.max())


def test_vanishing_film_is_an_emitter_in_the_top_medium():
    """film -> 0: the image of an x / y / z dipole has the shape of the same dipole in the top medium; the z dipole
    is weaker by (n_top / n_film)^4 relative to the in-plane ones (normal D continuous, tangential E continuous)."""
    L = LayeredVectorialPSF(15, n_med=1.49, n_top=1.0, film_um=1e-5, **KW)
    b = VectorialPSF(15, n_med=1.0, **KW)
    scale = {}
    with torch.no_grad():
        for k, M in DIP.items():
            a, c = L(POS, 0.0, dipM=M), b(POS, 0.0, dipM=M)
            assert float((a / a.sum() - c / c.sum()).abs().max()) < 1e-4 * float((c / c.sum()).max())
            scale[k] = float(a.sum() / c.sum())
    assert abs(scale['x'] / scale['z'] - 1.49 ** 4) < 0.01 * 1.49 ** 4


def test_resized_and_file_roundtrip(tmp_path):
    from torch_psf.psf import PSF
    L = LayeredVectorialPSF(13, n_med=1.49, n_top=1.0, film_um=0.04, **KW)
    L.set_zernike({5: 0.02})
    R = L.resized(15)
    assert isinstance(R, LayeredVectorialPSF) and abs(float(R.film_um) - 0.04) < 1e-7
    f = tmp_path / 'l.json'
    L.save(str(f))
    M = PSF.load(str(f), device='cpu')
    assert isinstance(M, LayeredVectorialPSF) and abs(float(M.n_top) - 1.0) < 1e-7
    with torch.no_grad():
        assert torch.allclose(M(POS, 0.0), L(POS, 0.0), atol=1e-7)
