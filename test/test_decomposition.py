import os

import numpy as np
import torch
from gMRItensor import compute_CP_decomposition
from gMRItensor import setup_backend


def test_backend():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()
    assert device.type == "cpu"
    os.environ["GMRITENSOR_USE_GPU"] = "TRUE"
    device = setup_backend()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def run_CP(use_gpu):
    os.environ["GMRITENSOR_USE_GPU"] = use_gpu
    device = setup_backend()

    tensor_1 = np.outer(np.array([0, 0, 1, 0]), np.array([0, 1, 0, 0])).astype(float)
    tensor = torch.from_numpy(tensor_1).to(device)
    decomp, errors = compute_CP_decomposition(tensor, 1, 1000, 1)
    print(decomp)
    weights, factors = decomp
    print(weights)

    assert torch.allclose(weights, torch.ones(1, dtype=weights.dtype, device=device))


def test_CP_cpu():
    run_CP("FALSE")


def test_CP_gpu():
    run_CP("TRUE")
