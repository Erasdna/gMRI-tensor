from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def scale_mode(arr):
    return arr / np.linalg.norm(arr, ax=0)[None, :]


def plot_subject_mode(subject_mode: np.ndarray, subject_info: pd.DataFrame):
    raise NotImplementedError


def plot_time_mode(
    time_mode: np.ndarray, time_points: list
) -> tuple[plt.Figure, np.ndarray]:
    """Plot time modes across components.

    Args:
        time_mode: Array of shape (n_timepoints, n_components)
        time_points: List of time point values for x-axis

    Returns:
        Tuple of (figure, axes array)
    """
    n_components = time_mode.shape[1]
    
    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 4))
    
    # Handle single component case (axes won't be an array)
    if n_components == 1:
        axes = np.array([axes])
    
    for i, ax in enumerate(axes):
        ax.plot(time_points, time_mode[:, i])
        ax.set_xlabel("Time")
        ax.set_ylabel("Time Mode")
        ax.set_title(f"Component {i + 1}")
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, axes


def plot_spatial_mode(
    spatial_mode: np.ndarray,
    index_list,
    csf_indices,
    parenchyma_indices,
    background,
    slices: list,
    transform: Callable,
):
    raise NotImplementedError
