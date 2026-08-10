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
    page_width: float | None = None,
    width_to_height_ratio=1.618,
) -> tuple[float, float]:
    """Compute appropriate figure size based on subplot grid dimensions.

    This function calculates a figure size that accounts for:
    - Number of subplot rows (components) and columns
    - Current matplotlib font size settings
    - Space needed for legends, labels, and annotations
    - Custom width ratios for columns (e.g., for images with different aspect ratios)
    - Target page width for journal submissions

    Parameters
    ----------
    n_components : int
        Number of components (rows in the subplot grid)
    n_columns : int
        Number of columns in the subplot grid
    base_width_per_col : float, optional
        Base width per column in inches, by default 3.0
        Only used if page_width is None
    base_height_per_row : float, optional
        Base height per row in inches, by default 2.5
    width_ratios : list[float] | None, optional
        Relative width ratios for each column. If provided, these ratios
        are used to scale the width of each column. For example, [1, 0.5, 0.5, 0.1]
        means the first column is full width, second and third are half width,
        and fourth (colorbar) is 10% width.
    page_width : float | None, optional
        Target page width in inches (e.g., 3.5 for single column, 7.0 for double column).
        If provided, overrides base_width_per_col and computes optimal dimensions
        to fit this width. By default None.

    Returns
    -------
    tuple[float, float]
        Figure size as (width, height) in inches

    Notes
    -----
    Common journal page widths:
    - Single column: 3.5 inches
    - 1.5 column: 5.5 inches
    - Double column: 7.0 inches
    """
    # Get current font size from matplotlib rcParams
    fontsize = plt.rcParams.get("font.size", 10)

    # Scale base dimensions by font size relative to default (10pt)
    font_scale = fontsize / 10.0

    # Calculate width
    if page_width is not None:
        # Use target page width directly
        width = page_width
    elif width_ratios is not None:
        # Normalize ratios and scale by base width
        total_ratio = sum(width_ratios)
        width = base_width_per_col * font_scale * total_ratio
    else:
        # Default: uniform column widths
        width = base_width_per_col * font_scale * n_columns

    # Calculate height based on width and aspect ratio
    if width_ratios is not None:
        # Compute average aspect ratio from width_ratios
        avg_aspect = sum(width_ratios) / len(width_ratios)
    else:
        avg_aspect = 1.0

    # Height per row should maintain reasonable aspect ratio
    # Use golden ratio (1.618) as default aspect for each subplot
    height_per_row = (width / n_columns) * avg_aspect / width_to_height_ratio

    # Add extra space for titles, labels (10% per row)
    height = height_per_row * n_components * 1.1

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
