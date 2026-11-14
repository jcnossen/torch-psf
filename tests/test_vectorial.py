"""VectorialPSF: reproduction of the pre-restructuring images, derivatives, normalization, optional terms."""
import os
import numpy as np
import torch
from conftest import legacy_reference
from torch_psf import VectorialPSF
from torch_psf.vectorial import photosel_tensor


def legacy():
    return {k: torch.tensor(v) for k, v in np.load(legacy_reference()).items()}


def model(R=13, **kw):
    r = legacy()
    v = VectorialPSF(R, pixel_um=0.108, wavelength_um=0.68, npupil=48, oversample=2, device='cpu', **kw)
    with torch.no_grad():
        v.zcoef.copy_(r['zcoef'])
    return v, r


def test_reproduces_legacy_images():
    v, r = model()
    with torch.no_grad():
        v.pupil_phase.copy_(r['pupil_phase']); v.pupil_amp.copy_(r['pupil_amp'])
        img = v(r['pos'], r['focus'])
    assert torch.allclose(img, r['vec_img'], rtol=0, atol=1e-7 * float(r['vec_img'].max()))


def test_render_gradient_matches_finite_differences():
    v, r = model(focus_um=0.4)
    pos = r['pos'][:3].clone()
    g, gx, gy, gz = [t.detach() for t in v.render(pos, grad=True)]
    for k, gk in enumerate((gx, gy, gz)):
        e = 1e-3
        p1, p2 = pos.clone(), pos.clone()
        p1[:, k] += e; p2[:, k] -= e
        with torch.no_grad():
            fd = (v.render(p1) - v.render(p2)) / (2 * e)
        assert float((fd - gk).abs().max() / gk.abs().max()) < 2e-3


def test_normalization_in_focus():
    v = VectorialPSF(41, npupil=64, oversample=3, device='cpu')
    with torch.no_grad():
        s = v(torch.zeros(1, 3), torch.zeros(1)).sum()
    assert 0.9 < float(s) <= 1.01      # emitter at the coverslip: ~1 (minus the tails outside the ROI)


def test_isotropic_dipole_tensor_is_the_default():
    v, r = model()
    with torch.no_grad():
        a = v(r['pos'], r['focus'])
        b = v(r['pos'], r['focus'], dipM=torch.eye(3) / 3)
        c = v(r['pos'], r['focus'], dipM=photosel_tensor(torch.tensor(0.0), (1.0, 0.0)))
    assert torch.allclose(a, b, atol=1e-6 * float(a.max()))
    assert torch.allclose(a, c, atol=1e-6 * float(a.max()))


def test_blur_broadens_and_keeps_energy():
    v, r = model(R=21)
    with torch.no_grad():
        a = v(r['pos'][:2], r['focus'][:2])
        b = v(r['pos'][:2], r['focus'][:2], blur=torch.tensor([0.3, 0.3, 0.0]))
    assert abs(float(a.sum() - b.sum())) < 0.01 * float(a.sum())
    assert float(b.max()) < float(a.max())


def test_resized_copy():
    v, r = model()
    with torch.no_grad():
        v.pupil_phase.copy_(r['pupil_phase'])
    v.blur = torch.tensor([0.1, 0.05, 0.0])
    w = v.resized(13)
    with torch.no_grad():
        assert torch.equal(v(r['pos'], r['focus']), w(r['pos'], r['focus']))
