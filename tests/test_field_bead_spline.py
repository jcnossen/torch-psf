"""Pin dense field rendering to the same optical fixture checked on WebGPU."""
import json
from pathlib import Path

import numpy as np
import torch

from make_web_field_spline_reference import reference


def test_dense_field_spline_reference():
    pinned = json.loads((Path(__file__).parents[1] / 'web/test/field_spline_reference.json').read_text())
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        current = reference()
    finally:
        torch.set_num_threads(previous)
    for key in ('direct', 'frames'):
        np.testing.assert_allclose(current[key], pinned[key], rtol=5e-5, atol=2e-7)
