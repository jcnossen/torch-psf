import os
import pytest
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='slow without a GPU')


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)

