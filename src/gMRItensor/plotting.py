from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from scipy.stats import linregress
from statannotations.Annotator import Annotator

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")


def scale_mode(arr):
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


def make_subject_boxplot(
    ax,
    df,
    x_column,
    y_column,
    legend=True,
    colors=["blue", "red"],
):
    # Get unique categories and create a mapping to integer positions
    categories = df[x_column].unique()
    n_categories = len(categories)

    # Create a temporary column with integer positions
    df_plot = df.copy()
    category_to_position = {cat: i for i, cat in enumerate(categories)}
    df_plot["_x_position"] = df_plot[x_column].map(category_to_position)

    boxplot = sns.boxplot(
        data=df_plot,
        x="_x_position",
        y=y_column,
        hue=x_column,
        ax=ax,
        palette=colors,
        width=0.25,
        legend=False,
        boxprops=dict(alpha=0.7),
        patch_artist=True,
        linewidth=1.5,
        saturation=1,
    )

    # Add statistical annotation for pairwise comparisons
    if n_categories >= 2:
        # Generate all pairwise comparisons using integer positions
        pairs = [
            (i, j) for i in range(n_categories) for j in range(i + 1, n_categories)
        ]

        annotator = Annotator(
            boxplot,
            pairs,
            data=df_plot,
            x="_x_position",
            y=y_column,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            hide_non_significant=True,
        )
        annotator.apply_and_annotate()

    # Add legend inside plot area if requested
    required_xlim = None
    if legend:
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.7)
            for i in range(n_categories)
        ]
        leg = ax.legend(
            handles,
            categories,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
        )

        # Draw the canvas to get accurate legend dimensions
        ax.figure.canvas.draw()

        # Get legend bounding box in display coordinates
        legend_bbox = leg.get_window_extent()

        # Transform to data coordinates
        legend_bbox_data = legend_bbox.transformed(ax.transData.inverted())

        # Calculate legend width in data coordinates
        legend_width = legend_bbox_data.x1 - legend_bbox_data.x0

        # Calculate required xlim:
        rightmost_tick = n_categories - 1
        # We compute the right limit of the plot:
        # Use 0.2 as a buffer + half of box width + legend width
        required_xlim = rightmost_tick + 0.2 + 0.125 + legend_width

    # Set tick positions and labels
    ax.set_xticks(range(n_categories))
    ax.set_xticklabels(categories)
    ax.set_xlabel(x_column)

    return ax.get_ylim(), required_xlim


def make_variable_correlation(
    ax,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    category: str,
    legend=True,
    colors=["blue", "red"],
):
    def fit_values(xs, ys, cat=""):
        fit = linregress(xs, ys)
        return fit

    line_plots = []
    legends = []
    pvalue_list = []

    sns.scatterplot(
        df,
        x=x_column,
        y=y_column,
        hue=category,
        palette=colors,
        legend=legend,
        ax=ax,
        alpha=0.7,
    )
    for k, cat in enumerate(df[category].unique()):
        cat_df = df.query(f"{category}=='{cat}'")
        xs = cat_df[x_column]
        ys = cat_df[y_column]

        fit = fit_values(xs, ys, cat)

        x_range = np.linspace(np.min(df[x_column]), np.max(df[x_column]))
        (line,) = ax.plot(
            x_range,
            fit.slope * x_range + fit.intercept,
            color=colors[k],
        )
        line_plots.append(line)
        pvalue_list.append(fit.pvalue)
        legends.append(rf"$R^2={fit.rvalue**2:.2f}$, $p={fit.pvalue:.1g} $")

    fit = fit_values(df[x_column], df[y_column], "all")

    if legend:
        leg = ax.legend(
            line_plots,
            legends,
        )
        for p_val, text in zip(pvalue_list, leg.get_texts()):
            try:
                if p_val < 0.05:
                    text.set_bbox(
                        dict(
                            facecolor="none",  # (0.5, 0.5, 0.5, 0.2),
                            edgecolor="black",
                            linewidth=0.5,
                            boxstyle="square,pad=0.2",
                        ),
                    )
            except ValueError:
                pass  # Skip if the text doesn't contain a parseable p-value
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_subject_mode(
    subject_mode: np.ndarray,
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    plotting_variables: list[str],
    figsize: tuple[float, float] | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot subject mode components against group variables and plotting variables.

    Parameters
    ----------
    subject_mode : np.ndarray
        Subject mode matrix with shape (n_subjects, n_components)
    subjects : list[str]
        List of subject identifiers
    subject_info : pd.DataFrame
        DataFrame containing subject metadata
    group_variable : str
        Column name in subject_info to use for grouping
    plotting_variables : list[str]
        List of column names in subject_info to correlate with components
    figsize : tuple[int, int] | None, optional
        Figure size (width, height). If None, automatically computed based on
        subplot grid dimensions and font size. By default None.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and axes array

    Raises
    ------
    ValueError
        If input validation fails
    """
    # Validate input dimensions
    if subject_mode.shape[0] != len(subjects):
        raise ValueError(
            f"""Number of subjects ({len(subjects)}) does not
            match subject_mode rows ({subject_mode.shape[0]},
        )""",
        )

    # Validate that required columns exist in subject_info
    required_columns = [group_variable] + plotting_variables
    missing_columns = [
        col for col in required_columns if col not in subject_info.columns
    ]
    if missing_columns:
        raise ValueError(
            f"The following columns are missing from subject_info: {missing_columns}",
        )

    # Validate that 'subjects' column exists in subject_info
    if "subjects" not in subject_info.columns:
        raise ValueError("subject_info must contain a 'subjects' column")

    scaled_subject_mode = scale_mode(subject_mode)
    subject_mode_df = pd.DataFrame(
        scaled_subject_mode,
        columns=[f"comp_{i}" for i in range(subject_mode.shape[1])],
    )
    subject_mode_df["subjects"] = subjects
    plotting_df = pd.merge(subject_mode_df, subject_info, how="inner", on="subjects")

    # Validate merge results
    if len(plotting_df) == 0:
        raise ValueError(
            "Merge resulted in empty DataFrame. Check that subject identifiers match "
            "between 'subjects' list and 'subjects' column in subject_info",
        )

    if len(plotting_df) < len(subjects):
        missing_count = len(subjects) - len(plotting_df)
        raise ValueError(
            f"{missing_count} subject(s) from the subjects list were not found in subject_info",
        )

    # Validate that group_variable has at least 2 unique values for comparison
    n_groups = plotting_df[group_variable].nunique()
    if n_groups < 2:
        raise ValueError(
            f"group_variable '{group_variable}' must have at least 2 unique values, "
            f"found {n_groups}",
        )

    # Compute figsize if not provided
    if figsize is None:
        n_components = subject_mode.shape[1]
        n_columns = 1 + len(plotting_variables)
        figsize = compute_figsize(n_components, n_columns)

    fig, axs = plt.subplots(
        subject_mode.shape[1],
        1 + len(plotting_variables),
        width_ratios=[1] + [1] * len(plotting_variables),
        figsize=figsize,
    )
    fig.tight_layout()

    # First pass: create all plots and collect required xlims
    ylims_list = []
    xlim_list = []

    for i, ax in enumerate(axs):
        ylims, required_xlim = make_subject_boxplot(
            ax[0],
            plotting_df,
            x_column=group_variable,
            y_column=f"comp_{i}",
            legend=True,  # Show legend on every row
        )
        ylims_list.append(ylims)
        if required_xlim is not None:
            xlim_list.append(required_xlim)

        ax[0].set_ylabel(rf"Component {i+1}")
        ax[0].set_xlabel("")

        if i == 0:
            ax[0].set_title(f"Subject mode v {group_variable}")

        for j, var in enumerate(plotting_variables):
            make_variable_correlation(
                ax[1 + j],
                plotting_df,
                x_column=var,
                y_column=f"comp_{i}",
                category=group_variable,
                legend=True,
            )
            if i == 0:
                ax[j + 1].set_title(f"Subject mode v {var}")

            if i == subject_mode.shape[1] - 1:
                ax[j + 1].set_xlabel(var)

    # Second pass: apply consistent xlim and ylim to all rows
    max_xlim = max(xlim_list) if xlim_list else 1.5
    print(xlim_list, max_xlim)
    for i, ax in enumerate(axs):
        ax[0].set_xlim(-0.25, max_xlim)
        for axx in ax:
            axx.set_ylim(*ylims_list[i])

    return fig, axs


def plot_subject_mode_correlation(
    subject_mode: np.ndarray,
    subjects: list,
    subject_info: pd.DataFrame,
    group_variable: str,
    figsize: tuple[float, float] | None = None,
):

    # Compute figsize if not provided
    n_components = subject_mode.shape[1]
    if figsize is None:
        figsize = compute_figsize(n_components, n_components)

    fig, axs = plt.subplots(
        subject_mode.shape[1],
        n_components,
        width_ratios=[1] * n_components,
        figsize=figsize,
    )
    fig.tight_layout()

    scaled_subject_mode = scale_mode(subject_mode)
    subject_mode_df = pd.DataFrame(
        scaled_subject_mode,
        columns=[f"comp_{i}" for i in range(subject_mode.shape[1])],
    )
    subject_mode_df["subjects"] = subjects
    plotting_df = pd.merge(subject_mode_df, subject_info, how="inner", on="subjects")

    ylims_list = []
    xlim_list = []
    for i, ax in enumerate(axs):
        ylims, required_xlim = make_subject_boxplot(
            ax[i],
            plotting_df,
            x_column=group_variable,
            y_column=f"comp_{i}",
            legend=True,  # Show legend on every row
        )
        ylims_list.append(ylims)
        if required_xlim is not None:
            xlim_list.append(required_xlim)

        if i == 0:
            ax[i].set_ylabel(rf"Component {i+1}")
        else:
            ax[i].set_ylabel("")
        ax[i].set_xlabel("")

        for j in range(n_components):
            if i != j:
                make_variable_correlation(
                    ax[j],
                    plotting_df,
                    x_column=f"comp_{j}",
                    y_column=f"comp_{i}",
                    category=group_variable,
                    legend=True,
                )
            if j == 0:
                ax[j].set_ylabel(rf"Component {i+1}")
            else:
                ax[j].set_ylabel("")

            if i == subject_mode.shape[1] - 1:
                ax[j].set_xlabel(f"Component {j+1}")
            else:
                ax[j].set_xlabel("")
    # Second pass: apply consistent xlim and ylim to all rows
    max_xlim = max(xlim_list) if xlim_list else 1.5
    print(xlim_list, max_xlim)
    for i, ax in enumerate(axs):
        ax[0].set_xlim(-0.25, max_xlim)
        for axx in ax:
            axx.set_ylim(*ylims_list[i])

    return fig, axs


def plot_spatial_mode(
    spatial_mode: np.ndarray,
    index_list,
    csf_indices,
    parenchyma_indices,
    background,
    slices: list,
    transform: Callable,
):
    assert spatial_mode.shape[0] == index_list.shape[0]

    raise NotImplementedError
