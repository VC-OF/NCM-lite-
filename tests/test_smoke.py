import torch

def test_environment_smoke():
    assert torch.__version__ is not None
    x = torch.ones(2, 3)
    assert x.shape == (2, 3)
