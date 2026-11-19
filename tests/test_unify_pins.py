"""Stage-0 output pins. Reference regeneration is a separate, explicit operation."""
import json

import numpy as np
import pytest
import torch

import make_unify_pins as pins


REF = json.loads(pins.FILE.read_text(encoding='utf-8'))
TOL = 1e-6


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


@pytest.mark.parametrize('name', pins.ZIP_CASES)
def test_psf_zip_bytes(name, models, tmp_path):
    got = pins.archive_pins(tmp_path / 'case.psf.zip', models[name][0])
    expected = REF['zip_sha256'][name]
    assert got['manifest'] == expected['manifest']
    if pins.writer_stack() == REF['writer_stack']:
        assert got['file'] == expected['file']
    else:
        # On a different writer stack require whole-file reproducibility locally as well.
        assert got == pins.archive_pins(tmp_path / 'again.psf.zip', models[name][0])
