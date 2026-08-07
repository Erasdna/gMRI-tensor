"""Plotting functions for spatial mode visualization."""
from typing import Callable

import matplotlib
import numpy as np
import scienceplots  # noqa: F401

matplotlib.use("Agg")


def plot_spatial_mode(
    spatial_mode: np.ndarray,
    index_list: np.ndarray,
    csf_indices: np.ndarray,
    parenchyma_indices: np.ndarray,
    background: np.ndarray,
    slices: list,
    transform: Callable,
) -> None:
    """Plot spatial mode components mapped onto brain slices.

    Parameters
    ----------
    spatial_mode : np.ndarray
        Spatial mode matrix
    index_list : np.ndarray
        Index mapping for spatial locations
    csf_indices : np.ndarray
        Indices for CSF regions
    parenchyma_indices : np.ndarray
        Indices for parenchyma regions
    background : np.ndarray
        Background image
    slices : list
        List of slice indices to plot
    transform : Callable
        Transform function to apply to spatial data

    Raises
    ------
    NotImplementedError
        This function is not yet implemented
    """
    assert spatial_mode.shape[0] == index_list.shape[0]

    raise NotImplementedError
