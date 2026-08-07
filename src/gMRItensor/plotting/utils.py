"""Shared utility functions for plotting."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def scale_mode(arr: np.ndarray) -> np.ndarray:
    """Scale mode by L2 norm along first axis.

    Parameters
    ----------
    arr : np.ndarray
        Mode array to scale

    Returns
    -------
    np.ndarray
        Scaled mode array
    """
    return arr / np.linalg.norm(arr, axis=0)[None, :]


def compute_figsize(
    n_components: int,
    n_columns: int,
    base_width_per_col: float = 3.0,
    base_height_per_row: float = 2.5,
) -> tuple[float, float]:
    """Compute appropriate figure size based on subplot grid dimensions.

    This function calculates a figure size that accounts for:
    - Number of subplot rows (components) and columns
    - Current matplotlib font size settings
    - Space needed for legends, labels, and annotations

    Parameters
    ----------
    n_components : int
        Number of components (rows in the subplot grid)
    n_columns : int
        Number of columns in the subplot grid
    base_width_per_col : float, optional
        Base width per column in inches, by default 3.0
    base_height_per_row : float, optional
        Base height per row in inches, by default 2.5

    Returns
    -------
    tuple[float, float]
        Figure size as (width, height) in inches
    """
    # Get current font size from matplotlib rcParams
    fontsize = plt.rcParams.get("font.size", 10)

    # Scale base dimensions by font size relative to default (10pt)
    font_scale = fontsize / 10.0

    # Calculate width: account for legend space in first column
    # First column needs extra space for legend
    width = base_width_per_col * font_scale * (n_columns)

    # Calculate height: scale by number of rows
    height = base_height_per_row * font_scale * n_components

    return (width, height)


def get_color_palette(n_colors: int) -> list:
    """Get color palette for plotting.

    Parameters
    ----------
    n_colors : int
        Number of colors needed

    Returns
    -------
    list
        List of colors from tab10 palette
    """
    return sns.color_palette("tab10", n_colors=n_colors)
