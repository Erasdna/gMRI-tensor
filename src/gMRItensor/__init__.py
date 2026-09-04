# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of my_package
# SPDX-License-Identifier:    MIT
import importlib.metadata

from .decomposition import compute_CP_decomposition
from .decomposition import compute_PARAFAC2_decomposition
from .decomposition import run_CP_decomposition_repeated
from .decomposition import run_PARAFAC2_decomposition_repeated
from .decomposition import setup_backend

__version__ = importlib.metadata.version(__package__)


__all__ = [
    "compute_CP_decomposition",
    "compute_PARAFAC2_decomposition",
    "run_CP_decomposition_repeated",
    "run_PARAFAC2_decomposition_repeated",
    "setup_backend",
]
