"""Plotting functions for subject mode analysis."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import scale_mode
from scipy.stats import linregress
from statannotations.Annotator import Annotator

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")


def make_subject_boxplot(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    legend: bool = True,
    colors: list | None = None,
) -> tuple[tuple[float, float], float | None]:
    """Boxplot `y_column` by `x_column`, with pairwise Mann-Whitney U tests.

    Falls back to a tab10-based palette when `colors` is None. Returns
    `(ylim, required_xlim)`, the latter being the x-axis upper limit needed
    to keep the legend from overlapping the data, or None without a legend.
    """
    # Integer x-positions keep the category spacing even.
    categories = df[x_column].unique()
    n_categories = len(categories)

    if colors is None:
        colors = get_color_palette(n_categories)

    df_plot = df.copy()
    category_to_position = {cat: i for i, cat in enumerate(categories)}
    df_plot["_x_position"] = df_plot[x_column].map(category_to_position)

    plot = sns.boxplot(
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

    if n_categories >= 2:
        pairs = [
            (i, j) for i in range(n_categories) for j in range(i + 1, n_categories)
        ]

        annotator = Annotator(
            plot,
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

        # Draw first, or the legend has no measurable extent yet.
        ax.figure.canvas.draw()
        legend_bbox = leg.get_window_extent()
        legend_bbox_data = legend_bbox.transformed(ax.transData.inverted())

        legend_width = legend_bbox_data.x1 - legend_bbox_data.x0

        # 0.2 buffer + half the box width + the legend.
        rightmost_tick = n_categories - 1
        required_xlim = rightmost_tick + 0.2 + 0.125 + legend_width

    ax.set_xticks(range(n_categories))
    ax.set_xticklabels(categories)
    ax.set_xlabel(x_column)

    if required_xlim is None:
        ax.set_xlim(-0.25, (n_categories - 1) + 0.25)
    return ax.get_ylim(), required_xlim


def make_variable_correlation(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    category: str,
    legend: bool = True,
    colors: list | None = None,
) -> None:
    """Scatter `y_column` against `x_column`, with a fit line per category.

    The legend carries each category's R2 and p-value.
    """

    if colors is None:
        n_categories = df[category].nunique()
        colors = get_color_palette(n_categories)

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
        alpha=1,
    )
    for k, cat in enumerate(df[category].unique()):
        cat_df = df.loc[df[category] == cat]
        xs = cat_df[x_column]
        ys = cat_df[y_column]

        fit = linregress(xs, ys)

        # Each group's own x-range: no extrapolation onto other groups' data.
        x_range = np.linspace(np.min(xs), np.max(xs))
        (line,) = ax.plot(
            x_range,
            fit.slope * x_range + fit.intercept,
            color=colors[k],
        )
        line_plots.append(line)
        pvalue_list.append(fit.pvalue)
        legends.append(rf"$R^2={fit.rvalue**2:.2f}$, $p={fit.pvalue:.1g} $")

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


def _prepare_plotting_dataframe(
    subject_mode: np.ndarray,
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    additional_variables: list[str] | None = None,
) -> pd.DataFrame:
    """Merge a scaled `(n_subjects, n_components)` subject mode with metadata.

    `additional_variables` names further `subject_info` columns to validate.
    Raises `ValueError` if the inputs do not line up.
    """
    if subject_mode.shape[0] != len(subjects):
        raise ValueError(
            f"""Number of subjects ({len(subjects)}) does not
            match subject_mode rows ({subject_mode.shape[0]},
        )""",
        )

    required_columns = [group_variable]
    if additional_variables is not None:
        required_columns.extend(additional_variables)

    missing_columns = [
        col for col in required_columns if col not in subject_info.columns
    ]
    if missing_columns:
        raise ValueError(
            f"The following columns are missing from subject_info: {missing_columns}",
        )

    if "subjects" not in subject_info.columns:
        raise ValueError("subject_info must contain a 'subjects' column")

    scaled_subject_mode = scale_mode(subject_mode)
    subject_mode_df = pd.DataFrame(
        scaled_subject_mode,
        columns=[f"comp_{i}" for i in range(subject_mode.shape[1])],
    )
    subject_mode_df["subjects"] = subjects
    plotting_df = pd.merge(subject_mode_df, subject_info, how="inner", on="subjects")

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

    n_groups = plotting_df[group_variable].nunique()
    if n_groups < 2:
        raise ValueError(
            f"group_variable '{group_variable}' must have at least 2 unique values, "
            f"found {n_groups}",
        )

    return plotting_df


def _finalize_boxplot_axes(
    axs: np.ndarray,
    ylims_list: list[tuple[float, float]],
    xlim_list: list[float],
    boxplot_columns: list[int],
    mirror_xlims: bool = False,
) -> None:
    """Apply per-row y-limits and a shared x-limit to the boxplot columns.

    Second-pass layout for `plot_subject_mode` and
    `plot_subject_mode_correlation`: every axes in row i takes that row's
    boxplot y-limits, and every boxplot axes takes an x-limit wide enough
    for the widest legend in the figure.

    `boxplot_columns[i]` is the column holding row i's boxplot.
    `mirror_xlims` additionally sets non-boxplot axes in column j to
    `ylims_list[j]`, mirroring each component's boxplot range onto the
    scatter plots that put that component on the x-axis.
    """
    max_xlim = max(xlim_list) if xlim_list else 1.5
    for i, ax_row in enumerate(axs):
        ax_row[boxplot_columns[i]].set_xlim(-0.25, max_xlim)
        for j, axx in enumerate(ax_row):
            axx.set_ylim(*ylims_list[i])
            if mirror_xlims and j != boxplot_columns[i]:
                axx.set_xlim(*ylims_list[j])


def plot_subject_mode(
    subject_mode: np.ndarray,
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    plotting_variables: list[str],
    page_width: float = 7.0,
    width_to_height_ratio: float = 1.618,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot subject mode components against group and continuous variables.

    One row per component: column 0 is a boxplot by `group_variable`,
    followed by one scatter column per entry in `plotting_variables`. See
    `compute_figsize` for the sizing arguments. Raises `ValueError` if the
    inputs do not line up.
    """
    plotting_df = _prepare_plotting_dataframe(
        subject_mode,
        subjects,
        subject_info,
        group_variable,
        plotting_variables,
    )

    n_components = subject_mode.shape[1]
    n_columns = 1 + len(plotting_variables)
    figsize = compute_figsize(
        n_components,
        n_columns,
        page_width=page_width,
        width_to_height_ratio=width_to_height_ratio,
    )

    fig, axs = plt.subplots(
        n_components,
        n_columns,
        width_ratios=[1] + [1] * len(plotting_variables),
        figsize=figsize,
        layout="compressed",
        squeeze=False,
    )

    # First pass: draw, collecting the limits each row needs.
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

            if i == n_components - 1:
                ax[j + 1].set_xlabel(var)

    _finalize_boxplot_axes(
        axs,
        ylims_list,
        xlim_list,
        boxplot_columns=[0] * n_components,
    )

    return fig, axs


def plot_subject_mode_correlation(
    subject_mode: np.ndarray,
    subjects: list[str],
    subject_info: pd.DataFrame,
    group_variable: str,
    page_width: float = 7.0,
    width_to_height_ratio: float = 1.0,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot a component-by-component matrix with group comparisons.

    Diagonal `(i, i)`: boxplot of component i by group. Off-diagonal
    `(i, j)`: component i against component j, with a regression line per
    group. See `compute_figsize` for the sizing arguments. Raises
    `ValueError` if the inputs do not line up.
    """
    plotting_df = _prepare_plotting_dataframe(
        subject_mode,
        subjects,
        subject_info,
        group_variable,
    )

    n_components = subject_mode.shape[1]
    figsize = compute_figsize(
        n_components,
        n_components,
        page_width=page_width,
        width_to_height_ratio=width_to_height_ratio,
    )
    fig, axs = plt.subplots(
        n_components,
        n_components,
        width_ratios=[1] * n_components,
        figsize=figsize,
        layout="compressed",
        squeeze=False,
    )

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

            if i == n_components - 1:
                ax[j].set_xlabel(f"Component {j+1}")
            else:
                ax[j].set_xlabel("")

    # Second pass: apply consistent xlim and ylim to all rows
    _finalize_boxplot_axes(
        axs,
        ylims_list,
        xlim_list,
        boxplot_columns=list(range(n_components)),
        mirror_xlims=True,
    )

    return fig, axs
