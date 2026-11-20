"""Stage-0 output pins. Reference regeneration is a separate, explicit operation."""
import json
import os

import numpy as np
import pytest
import torch

import make_unify_pins as pins


REF = json.loads(pins.FILE.read_text(encoding='utf-8'))
# The references were recorded in float32 on one platform (Windows laptop). Other platforms (BLAS, threads,
# CUDA / CPU kernels) differ by up to ~1e-5 of the peak in gradients and moments, so the default tolerance is
# 5e-5 of the peak. TORCH_PSF_PINS_STRICT=1 uses 1e-6 (the recording platform; refactor stages are also checked
# there for bit identity against the previous commit).
TOL = 1e-6 if os.environ.get('TORCH_PSF_PINS_STRICT') == '1' else 5e-5


@pytest.fixture(scope='module')
def models():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cases = {name: (model, kw) for name, model, kw in pins.models()}
        assert set(cases) == set(REF['cases'])
        np.testing.assert_array_equal(pins.POS.numpy(), REF['positions'])
        yield cases
    finally:
        torch.set_num_threads(original)


def check_image(actual, reference):
    expected = np.asarray(reference['pixels'])
    assert actual.shape == expected.shape
    peak = np.abs(expected).max(axis=(-2, -1))
    np.testing.assert_array_less(np.abs(actual - expected).max(axis=(-2, -1)), TOL * peak + 1e-30)
    # Signed derivatives can have zero total or a vanishing moment through cancellation.
    # Scale the tolerance by the sum of absolute weighted contributions in that case.
    y, x = np.mgrid[:expected.shape[-2], :expected.shape[-1]]
    y, x = abs(y - y.mean()), abs(x - x.mean())
    moment_scale = np.einsum('...ij,kij->...k', abs(expected),
                             np.stack([np.ones_like(x), x, y, x*x, x*y, y*y]))
    np.testing.assert_array_less(np.abs(pins.moments(actual) - reference['moments']),
                                 TOL * moment_scale + 1e-30)


@pytest.mark.parametrize('name', REF['cases'])
def test_render_pin(name, models):
    expected = REF['cases'][name]
    model, kwargs = models[name]
    gradients, value = pins.outputs(model, kwargs)
    assert set(gradients) == set(expected) - {'forward'}
    check_image(value, expected.get('forward', expected['value']))
    for key, image in gradients.items():
        check_image(image, expected[key])


def test_gauss_legendre_with_blur():
    m = pins.vec('gauss_legendre')
    m.blur = torch.tensor([.14, .08, -.035])
    with pytest.raises(ValueError, match='blur'):
        m.render(pins.POS)


def test_orientation_pin(models):
    mu, jac, theta = pins.orientation()
    check_image(mu, REF['orientation']['mu'])
    check_image(jac, REF['orientation']['jac'])
    np.testing.assert_allclose(theta, REF['orientation']['theta'], rtol=TOL, atol=1e-8)


def test_sofi_grid_pin(models):
    value, jac = pins.sofi()
    check_image(value, REF['sofi']['value'])
    check_image(jac, REF['sofi']['jac'])


# The spline repr in fit_report.pdf changed at stage 3. On the original writer stack every
# other member, including psf.json, zstack.tif and preview.png, was verified byte for byte.
REPORT_REPR_HASHES = {
    'cubic': '17c25cdfc1cadb699ea50cba046d37c371fb32c3f7b59058296985198dff02fe',
    'integrated': '661e35644b556eb596e43ee4ff7c95f7686ded7cdf05bb67a2958fc375f68d6d',
    'integrated_angle_dipole': '751a71226188756967c20ba0bdb607f07349d68ddff16b5b621fbbe9e4203c51',
    'tiles': '2d91900b74c0cfc430c6cf2ad3b24dff89da01e59eabb33c88d8ff4278217fa0',
}


@pytest.mark.parametrize('name', pins.ZIP_CASES)
def test_psf_zip_bytes(name, models, tmp_path):
    got = pins.archive_pins(tmp_path / 'case.psf.zip', models[name][0])
    expected = REF['zip_sha256'][name]
    assert got['manifest'] == expected['manifest']
    if pins.writer_stack() == REF['writer_stack']:
        assert got['file'] == REPORT_REPR_HASHES.get(name, expected['file'])
    else:
        # On a different writer stack require whole-file reproducibility locally as well.
        assert got == pins.archive_pins(tmp_path / 'again.psf.zip', models[name][0])

    # The exact archive bytes pin the legacy writer. Exercise the reader against
    # the original rendered pixels and derivatives as well, including field tiles.
    if name in ('cubic', 'integrated', 'integrated_angle_dipole', 'phase_cubic',
                'phase_integrated', 'phase_dipole', 'tiles'):
        from torch_psf import PSF
        model, kw = models[name]
        loaded = PSF.load(tmp_path / 'case.psf.zip', device='cpu', angle=getattr(model, 'angle', None))
        gradients, value = pins.outputs(loaded, kw)
        expected = REF['cases'][name]
        check_image(value, expected.get('forward', expected['value']))
        for key, image in gradients.items():
            check_image(image, expected[key])
