"""Shared utility functions for plotting."""
import matplotlib.pyplot as plt
import numpy as np

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
    page_width: float,
    width_ratios: list[float] | None = None,
    width_to_height_ratio: float = 1.618,
) -> tuple[float, float]:
    """Compute appropriate figure size based on page width and aspect ratio.

    This function calculates a figure size that accounts for:
    - Target page width for journal submissions
    - Number of subplot rows (components) and columns
    - Custom width ratios for columns (e.g., for images with different aspect ratios)
    - Desired width-to-height ratio for individual subplots

    Parameters
    ----------
    n_components : int
        Number of components (rows in the subplot grid)
    n_columns : int
        Number of columns in the subplot grid
    page_width : float
        Target page width in inches (e.g., 3.5 for single column, 7.0 for double column)
    width_ratios : list[float] | None, optional
        Relative width ratios for each column. If provided, these ratios
        are used to scale the width of each column. For example, [1, 0.5, 0.5, 0.1]
        means the first column is full width, second and third are half width,
        and fourth (colorbar) is 10% width. By default None.
    width_to_height_ratio : float, optional
        Desired width-to-height ratio for each subplot (width/height).
        Default is 1.618 (golden ratio). By default 1.618.

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
    # Use target page width directly
    width = page_width

    # Height per row should maintain the specified aspect ratio
    height_per_row = (width / n_columns) / width_to_height_ratio

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
    return [f"C{3*i}" for i in range(n_colors)]


def create_colorbar_with_offset(
    fig,
    ax,
    mappable,
    cax,
    label: str | None = None,
) -> float:
    """Create a colorbar with scientific notation and return required offset.

    This function creates a colorbar with scientific notation formatting and
    calculates the horizontal offset needed to prevent the exponent text from
    overlapping with other plot elements.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the colorbar
    ax : matplotlib.axes.Axes
        Axes associated with the colorbar
    mappable : matplotlib.cm.ScalarMappable
        Mappable object (e.g., from imshow or pcolormesh)
    cax : matplotlib.axes.Axes
        Axes for the colorbar
    label : str | None, optional
        Label for the colorbar. By default None.

    Returns
    -------
    float
        Width offset in axes coordinates needed to accommodate the exponent text.
        This value represents half the text width and can be used to adjust
        the position of the colorbar to prevent overlap.
    """
    formatter = plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        ax=ax,
        use_gridspec=True,
        format=formatter,
    )
    # Center-align the exponent text above the colorbar
    cbar.ax.yaxis.offsetText.set_visible(True)
    cbar.ax.yaxis.offsetText.set_horizontalalignment("center")
    cbar.set_label(label=label)

    # Force draw to get accurate text dimensions
    fig.canvas.draw()

    # Get the exponent text bounding box in display coordinates
    offset_text_bbox = cbar.ax.yaxis.offsetText.get_window_extent()

    # Transform to axes coordinates of the parent ax
    offset_text_bbox_ax = offset_text_bbox.transformed(ax.transAxes.inverted())

    # Calculate required left shift (half the text width in axes coordinates)
    text_width_ax = offset_text_bbox_ax.width

    return text_width_ax / 2.0
