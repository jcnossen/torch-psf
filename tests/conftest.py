import os
import pytest
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='slow without a GPU')


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def legacy_reference():
    """Path of the legacy reference images (generated on first use from the git history; the test is skipped when
    that is not possible, e.g. in a shallow clone)."""
    import subprocess, sys
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'reference_e8e6fc5.npz')
    if not os.path.exists(f):
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(f)), 'make_legacy_reference.py')],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if r.returncode != 0 or not os.path.exists(f):
            pytest.skip('legacy reference not available (needs the git history with commit e8e6fc5): ' + r.stderr[-300:])
    return f
