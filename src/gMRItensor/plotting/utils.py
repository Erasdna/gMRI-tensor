"""Shared utility functions for plotting."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use(["science", "no-latex"])


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
    width_ratios: list[float] | None = None,
) -> tuple[float, float]:
    """Compute appropriate figure size based on subplot grid dimensions.

    This function calculates a figure size that accounts for:
    - Number of subplot rows (components) and columns
    - Current matplotlib font size settings
    - Space needed for legends, labels, and annotations
    - Custom width ratios for columns (e.g., for images with different aspect ratios)

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
    width_ratios : list[float] | None, optional
        Relative width ratios for each column. If provided, these ratios
        are used to scale the width of each column. For example, [1, 0.5, 0.5, 0.1]
        means the first column is full width, second and third are half width,
        and fourth (colorbar) is 10% width.

    Returns
    -------
    tuple[float, float]
        Figure size as (width, height) in inches
    """
    # Get current font size from matplotlib rcParams
    fontsize = plt.rcParams.get("font.size", 10)

    # Scale base dimensions by font size relative to default (10pt)
    font_scale = fontsize / 10.0

    # Calculate width based on width_ratios if provided
    if width_ratios is not None:
        # Normalize ratios and scale by base width
        total_ratio = sum(width_ratios)
        width = base_width_per_col * font_scale * total_ratio
    else:
        # Default: uniform column widths
        width = base_width_per_col * font_scale * n_columns

    # Calculate height: scale by number of rows
    # Reduce height slightly to account for title space with tight layout
    height = base_height_per_row * font_scale * n_components * 1.1

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
