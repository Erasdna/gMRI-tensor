"""Plotting functions for subject mode analysis."""
from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from gMRItensor.group_statistics import require_columns
from gMRItensor.plotting.utils import compute_figsize
from gMRItensor.plotting.utils import get_color_palette
from gMRItensor.plotting.utils import scale_mode
from scipy.stats import linregress
from statannotations.Annotator import Annotator

plt.style.use(["science", "no-latex"])
matplotlib.use("Agg")


POOLED_LABEL = "All"
# Wrapped tick label: one line does not fit next to the groups' labels.
_POOLED_TICK = "All"
# Hollow: the pooled box is no group, so it takes no palette colour (the
# palette includes a grey).
POOLED_COLOR = "white"
# Fit line over every subject in the scatters, apart from the group colours.
POOLED_FIT_COLOR = "black"
# Light grey for boxes of non-group variables: colour means group, and this
# is far lighter than the palette's grey.
CATEGORY_COLOR = "0.85"


def _levels(values: pd.Series) -> list:
    """Sorted non-missing unique values: the category order (and so the
    colour order) of every categorical panel, as in `plot_evolving_mode`.
    Numeric codes sort numerically."""
    levels = list(values.dropna().unique())
    try:
        return sorted(levels)
    except TypeError:  # mixed types
        return sorted(levels, key=str)


def make_subject_boxplot(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    legend: bool = True,
    colors: list | None = None,
    pooled: bool = False,
    dot_hue: str | None = None,
) -> tuple[tuple[float, float], float | None]:
    """Boxplot `y_column` by `x_column`, with pairwise Mann-Whitney U tests.

    Categories are `_levels(df[x_column])`; rows missing either column are
    left out. `colors` are in that order (default `get_color_palette`).
    `pooled` appends a hollow box of every subject, labelled `POOLED_LABEL`:
    it shows which values are outliers for the whole dataset rather than
    within a group, and is never tested. `dot_hue` names another column
    (the group variable): the boxes are then light grey, and every subject is
    drawn as a dot in its `dot_hue` group's colour, so colour keeps meaning
    that group in every panel. Returns `(ylim, required_xlim)`,
    the latter being the x-axis upper limit needed to keep the legend from
    overlapping the data, or None without a legend.
    """
    dot_levels = _levels(df[dot_hue]) if dot_hue is not None else []
    df = df.dropna(subset=[x_column, y_column])
    categories = _levels(df[x_column])
    n_categories = len(categories)

    if dot_hue is not None:
        colors = [CATEGORY_COLOR] * n_categories
    elif colors is None:
        colors = get_color_palette(n_categories)
    labels = [str(category) for category in categories]
    position = {category: i for i, category in enumerate(categories)}
    # Integer x-positions keep the category spacing even.
    df_plot = df.assign(
        _x_position=df[x_column].map(position),
        _hue=df[x_column].map(dict(zip(categories, labels))),
    )
    hue_order = list(labels)
    palette = list(colors[:n_categories])
    if pooled:
        df_plot = pd.concat(
            [df_plot, df.assign(_x_position=n_categories, _hue=POOLED_LABEL)],
            ignore_index=True,
        )
        hue_order.append(POOLED_LABEL)
        palette.append(POOLED_COLOR)

    plot = sns.boxplot(
        data=df_plot,
        x="_x_position",
        y=y_column,
        hue="_hue",
        hue_order=hue_order,
        ax=ax,
        palette=palette,
        width=0.25,
        legend=False,
        boxprops=dict(alpha=0.7),
        patch_artist=True,
        linewidth=1.5,
        saturation=1,
    )
    if dot_hue is not None:
        sns.stripplot(
            data=df_plot,
            x="_x_position",
            y=y_column,
            hue=dot_hue,
            hue_order=dot_levels,
            palette=get_color_palette(len(dot_levels)),
            jitter=0.08,
            size=3,
            legend=False,
            ax=ax,
        )

    if n_categories >= 2:
        # Group positions only: the pooled box is never compared.
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

    n_boxes = len(hue_order)
    required_xlim = None
    if legend:
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.7)
            for i in range(n_categories)
        ]
        leg = ax.legend(
            handles,
            labels,
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
        rightmost_tick = n_boxes - 1
        required_xlim = rightmost_tick + 0.2 + 0.125 + legend_width

    ax.set_xticks(range(n_boxes))
    ax.set_xticklabels([*labels, _POOLED_TICK] if pooled else labels)
    ax.set_xlabel(x_column)

    if required_xlim is None:
        ax.set_xlim(-0.25, (n_boxes - 1) + 0.25)
    return ax.get_ylim(), required_xlim


def make_variable_correlation(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    category: str,
    legend: bool = True,
    colors: list | None = None,
    pooled: bool = False,
) -> None:
    """Scatter `y_column` against `x_column`, with a fit line per category.

    Rows missing any of the three columns are left out. Categories are
    `_levels(df[category])`, coloured in that order. `pooled` adds a black
    fit over every subject, labelled `POOLED_LABEL` in the legend. A fit
    needs at least 3 points and a non-constant x. The legend carries each
    fit's R2 and p-value.
    """
    df = df.dropna(subset=[x_column, y_column, category])
    categories = _levels(df[category])
    if colors is None:
        colors = get_color_palette(len(categories))

    line_plots = []
    legends = []
    pvalue_list = []

    sns.scatterplot(
        df,
        x=x_column,
        y=y_column,
        hue=category,
        hue_order=categories,
        palette=list(colors[: len(categories)]),
        legend=legend,
        ax=ax,
        alpha=1,
        s=15,
    )
    fit_sets = [
        (df.loc[df[category] == cat], colors[k], "") for k, cat in enumerate(categories)
    ]
    if pooled:
        fit_sets.append((df, POOLED_FIT_COLOR, f"{POOLED_LABEL}: "))
    for fit_df, color, prefix in fit_sets:
        xs = fit_df[x_column]
        ys = fit_df[y_column]
        if len(fit_df) < 3 or xs.nunique() < 2:
            continue

        fit = linregress(xs, ys)

        # The fitted subjects' own x-range: no extrapolation onto other
        # groups' data.
        x_range = np.linspace(np.min(xs), np.max(xs))
        (line,) = ax.plot(
            x_range,
            fit.slope * x_range + fit.intercept,
            color=color,
        )
        line_plots.append(line)
        pvalue_list.append(fit.pvalue)
        legends.append(rf"{prefix}$R^2$={fit.rvalue**2:.2f}, $p$={fit.pvalue:.1g}")

    if legend and line_plots:
        leg = ax.legend(
            line_plots,
            legends,
            fontsize="x-small",
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
    categorical: Sequence[str] = (),
    numerical: Sequence[str] = (),
) -> pd.DataFrame:
    """Merge a scaled `(n_subjects, n_components)` subject mode with metadata.

    `categorical` and `numerical` name further `subject_info` columns; the
    numerical ones are converted to numbers. Blank cells stay missing (each
    panel leaves them out), except in `group_variable`. Raises `ValueError`
    if the inputs do not line up.
    """
    if subject_mode.shape[0] != len(subjects):
        raise ValueError(
            f"""Number of subjects ({len(subjects)}) does not
            match subject_mode rows ({subject_mode.shape[0]},
        )""",
        )

    requested = {
        **{column: "numerical" for column in numerical},
        **{column: "categorical" for column in categorical},
        group_variable: "group_variable",
    }
    require_columns(list(subject_info.columns), requested, "subject_info")

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

    no_group = plotting_df.loc[plotting_df[group_variable].isna(), "subjects"]
    if not no_group.empty:
        raise ValueError(
            f"group_variable '{group_variable}' is missing for subject(s) "
            f"{list(no_group)}",
        )

    for column in numerical:
        values = pd.to_numeric(plotting_df[column], errors="coerce")
        bad = plotting_df[column].notna() & values.isna()
        if bad.any():
            first = plotting_df.loc[bad].iloc[0]
            raise ValueError(
                f"numerical variable '{column}' is not numeric: "
                f"{first[column]!r} for subject {first['subjects']}",
            )
        plotting_df[column] = values

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
    numerical_variables: Sequence[str] = (),
    categorical_variables: Sequence[str] = (),
    page_width: float = 7.0,
    width_to_height_ratio: float | None = None,
    min_panel_width: float | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot subject mode components against categorical and numerical variables.

    One row per component, one shared y-range per row. Boxplots come first:
    by `group_variable` (plus a pooled box of every subject), then by each
    of `categorical_variables`. Then one scatter column per entry in
    `numerical_variables`, coloured and fitted per `group_variable` group,
    plus a black fit over all subjects.
    A subject with a blank cell is left out of that variable's panels only.
    See `compute_figsize` for the sizing arguments; `width_to_height_ratio`
    defaults to 1.618 for up to two columns and 1.0 (square panels) beyond,
    where the panels get too flat for their labels. With `min_panel_width`
    (inches), a figure whose panels would be narrower is widened beyond
    `page_width` to that width per panel. Raises `ValueError` if the inputs
    do not line up.
    """
    categorical = list(
        dict.fromkeys(c for c in categorical_variables if c != group_variable),
    )
    numerical = list(numerical_variables)
    plotting_df = _prepare_plotting_dataframe(
        subject_mode,
        subjects,
        subject_info,
        group_variable,
        categorical,
        numerical,
    )

    boxplot_variables = [group_variable, *categorical]
    n_components = subject_mode.shape[1]
    n_columns = len(boxplot_variables) + len(numerical)
    if width_to_height_ratio is None:
        width_to_height_ratio = 1.618 if n_columns <= 2 else 1.0
    if min_panel_width is not None:
        page_width = max(page_width, n_columns * min_panel_width)
    figsize = compute_figsize(
        n_components,
        n_columns,
        page_width=page_width,
        width_to_height_ratio=width_to_height_ratio,
    )

    fig, axs = plt.subplots(
        n_components,
        n_columns,
        figsize=figsize,
        layout="compressed",
        sharey="row",
        squeeze=False,
    )

    for i, ax in enumerate(axs):
        y_column = f"comp_{i}"
        row_ylims = []
        for j, variable in enumerate(boxplot_variables):
            # The tick labels name the categories: no legend needed.
            # Colour means group: other variables get hollow boxes with
            # group-coloured subject dots.
            ylims, _ = make_subject_boxplot(
                ax[j],
                plotting_df,
                x_column=variable,
                y_column=y_column,
                legend=False,
                pooled=j == 0,
                dot_hue=None if j == 0 else group_variable,
            )
            row_ylims.append(ylims)
            ax[j].set_xlabel(variable if i == n_components - 1 else "")

        for j, variable in enumerate(numerical, start=len(boxplot_variables)):
            make_variable_correlation(
                ax[j],
                plotting_df,
                x_column=variable,
                y_column=y_column,
                category=group_variable,
                legend=True,
                pooled=True,
            )
            if i == n_components - 1:
                ax[j].set_xlabel(variable)

        # Boxplot ranges include the significance annotations.
        row_ylim = (min(lo for lo, _ in row_ylims), max(hi for _, hi in row_ylims))
        for axx in ax:
            axx.set_ylim(*row_ylim)
            axx.set_ylabel("")
        ax[0].set_ylabel(rf"Component {i+1}")

    for j, variable in enumerate([*boxplot_variables, *numerical]):
        axs[0, j].set_title(variable)

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
        sharey="row",
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
