import os

import torch
from gMRItensor import setup_backend
from gMRItensor.replicability import half_half_split_replicability
from gMRItensor.replicability import repeated_CV_replicability


def test_half_half():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()

    tensor = torch.randn(99, 4, 10000).to(device)
    iterator = half_half_split_replicability(
        tensor,
        3,
        10,
        CP_max_iter=100,
        CP_init_repeats=2,
        CP_verbose_level=1,
    )
    for it in iterator:
        print(it[1])
        assert len(it[0]) == tensor.shape[0]


def test_CV_folds():
    os.environ["GMRITENSOR_USE_GPU"] = "FALSE"
    device = setup_backend()

    tensor = torch.randn(100, 4, 10000).to(device)
    iterator = repeated_CV_replicability(
        tensor,
        3,
        2,
        10,
        CP_max_iter=100,
        CP_init_repeats=2,
        CP_verbose_level=1,
    )
    for it in iterator:
        i, j, fms = it[1:]
        assert i != j


if __name__ == "__main__":
    print("--- Debugging Test ---")
    test_CV_folds()
    test_half_half()
    print("--- Test Completed Successfully ---")
