"""`gmri` command line. Every setting is an argument; see `gmri <command> --help`.

    gmri preprocess    --manifest scans.csv --output-dir results --input-type T1map
    gmri decompose     run | plan | collect   (run --job N for one distributed job)
    gmri replicability run | plan | collect
    gmri plot          statistics | decomposition | group-maps | reproducibility

Each command calls one `gMRItensor.pipeline` stage with its
`gMRItensor.options` dataclass, so scripts can do the same.
"""
import argparse
import json
import sys
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gMRItensor import pipeline
from gMRItensor.decomposition import ConvergenceError
from gMRItensor.group_statistics import AGGREGATES
from gMRItensor.model_io import NEGLIGIBLE_SUBJECT_WEIGHT
from gMRItensor.options import DECOMPOSITION_PARTS
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import DistributedOptions
from gMRItensor.options import FitOptions
from gMRItensor.options import GroupMapPlotOptions
from gMRItensor.options import PreprocessingOptions
from gMRItensor.options import ReplicabilityOptions
from gMRItensor.options import STATISTICS
from gMRItensor.options import StatisticsPlotOptions
from gMRItensor.options import TensorOptions

_DISTRIBUTED_EPILOG = """\
Distributed restarts (e.g. a SLURM array):
  N=$(gmri {command} plan <arguments>)          # prints the number of jobs; saves plan.json
  gmri {command} run --output-dir DIR --job $SLURM_ARRAY_TASK_ID   # in each array job
  gmri {command} collect --output-dir DIR       # once every job has finished

`run` without --job fits everything in this process.
"""


class _Parser(argparse.ArgumentParser):
    """An `ArgumentParser` that takes no abbreviations: `--tol` must not
    pass for `--tolerance`, or `run --job` could not refuse it. Subparsers
    inherit the class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def _non_negative_modes(value: str) -> Any:
    """`auto`, `none` or comma-separated mode indices (0 subject, 1 time,
    2 label)."""
    if value in ("auto", "none"):
        return None if value == "none" else "auto"
    try:
        return tuple(int(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected auto, none or modes like 0,2; got {value!r}",
        ) from None


def _fit_option(value: str) -> tuple[str, Any]:
    """`KEY=VALUE`, VALUE read as JSON (true, 0.1, {...}) or else as text."""
    key, separator, text = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    try:
        return key, json.loads(text)
    except json.JSONDecodeError:
        return key, text


def _output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--page-width",
        default="double",
        help="figure width: single, onehalf, double or inches (default: double)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="file formats (default: pdf png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="raster resolution (default: 300)",
    )


def _page_width(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        return value


def _add_preprocess(commands: Any) -> None:
    parser = commands.add_parser(
        "preprocess",
        help="images -> per-ROI (and optionally per-voxel) signal tables",
        description=(
            "Read every scan in the manifest once and write <output-dir>/data/: "
            "roi_signal.parquet (median, mean and voxel counts per label and "
            "scan) and, with --store-voxels, voxels.parquet + voxels.coords.parquet. "
            "T1map/R1map give ΔR1 in 1/s; T1w gives the post/baseline ratio."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV: subject,time_point,baseline_path,post_injection_path,mask_path,"
        "segmentation_path (paths relative to the CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="results directory",
    )
    parser.add_argument(
        "--input-type",
        choices=("T1map", "R1map", "T1w"),
        required=True,
        help="what the images are: T1 maps, R1 maps or T1-weighted images",
    )
    parser.add_argument(
        "--time-unit",
        choices=("ms", "s"),
        help="unit of the maps: T1 in ms or s, R1 in 1/ms or 1/s; "
        "required for T1map and R1map",
    )
    parser.add_argument(
        "--store-voxels",
        action="store_true",
        help="also store every voxel's value, for voxel-wise decomposition "
        "(all scans must share one template)",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=5,
        help="parallel image readers (default: 5)",
    )


def _add_fit_arguments(
    parser: argparse.ArgumentParser,
    required: bool,
    replicability: bool,
) -> None:
    """Input, tensor and fit arguments shared by decompose and replicability
    (which spreads its fits with `--n-procs` instead of `--restart-procs`)."""
    parser.add_argument(
        "--input",
        type=Path,
        required=required,
        help="roi_signal.parquet (per-ROI fit) or voxels.parquet (per-voxel fit)",
    )
    parser.add_argument(
        "--method",
        choices=("cp", "parafac2", "cmf"),
        required=required,
        help="CP (shared time mode), PARAFAC2 (evolving per-subject time mode, "
        "coupled across subjects) or CMF (non-negative coupled matrix "
        "factorization: free per-subject time courses)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        required=required,
        help="ranks to fit",
    )
    parser.add_argument(
        "--statistic",
        choices=("median", "mean"),
        default="median",
        help="per-ROI statistic to decompose (roi_signal input) (default: median)",
    )
    tensor = parser.add_argument_group("tensor")
    tensor.add_argument(
        "--no-scale",
        action="store_true",
        help="do not divide each column by its std",
    )
    tensor.add_argument(
        "--center",
        action="store_true",
        help="subtract each column's mean (needs --non-negative-modes none)",
    )
    tensor.add_argument(
        "--min-timepoints",
        type=int,
        help="minimum valid time points per subject (default: all); CP imputes "
        "the missing ones, PARAFAC2/CMF fit shorter time courses",
    )
    tensor.add_argument(
        "--max-invalid-fraction",
        type=float,
        default=0.9,
        help="treat scans with a larger non-finite fraction as missing; CP "
        "imputes the other non-finite values, PARAFAC2/CMF drop their columns "
        "(default: 0.9)",
    )
    fit = parser.add_argument_group("fit")
    fit.add_argument(
        "--restarts",
        type=int,
        default=50,
        help="random initialisations per fit (default: 50)",
    )
    fit.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="iterations per restart (default: 2000)",
    )
    fit.add_argument(
        "--tolerance",
        type=float,
        help="convergence tolerance (default: the solver's own)",
    )
    if not replicability:
        fit.add_argument(
            "--restart-procs",
            type=int,
            default=1,
            help="parallel restart processes (default: 1)",
        )
    fit.add_argument(
        "--non-negative-modes",
        type=_non_negative_modes,
        default="auto",
        help="auto, none or modes like 0,2 (0 subject, 1 time, 2 label); CP: auto, "
        "none or 0,1,2; CMF auto = 1,2 (default: auto)",
    )
    fit.add_argument(
        "--solver",
        choices=("tensorly", "matcouply"),
        help="PARAFAC2 solver (default: matcouply); CMF always uses matcouply",
    )
    fit.add_argument(
        "--fit-option",
        type=_fit_option,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra run_*_decomposition_repeated option, VALUE as JSON (repeatable)",
    )
    distributed = parser.add_argument_group("distributed (plan)")
    distributed.add_argument(
        "--tasks-per-job",
        type=int,
        default=1,
        help="restarts per --job (default: 1)",
    )
    distributed.add_argument(
        "--store",
        default="restarts",
        help="restart result directory inside --output-dir (default: restarts)",
    )


def _add_replicability_arguments(
    parser: argparse.ArgumentParser,
    required: bool,
) -> None:
    group = parser.add_argument_group("replicability")
    group.add_argument(
        "--engine",
        choices=("halfhalf", "cv"),
        required=required,
        help="split halves or k-fold cross-validation",
    )
    group.add_argument(
        "--repeats",
        type=int,
        required=required,
        help="random splits / k-fold repeats",
    )
    group.add_argument("--splits", type=int, help="folds for cv (>= 3)")
    group.add_argument("--subject-info", type=Path, help="CSV with a subjects column")
    group.add_argument(
        "--stratify-by",
        help="subject-info column whose proportions each split keeps",
    )
    group.add_argument(
        "--n-procs",
        type=int,
        default=1,
        help="parallel fits (default: 1)",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the splits (default: 0)",
    )


def _add_fit_command(
    commands: Any,
    command: str,
    summary: str,
    replicability: bool,
) -> None:
    parser = commands.add_parser(
        command,
        help=summary,
        description=summary,
        epilog=_DISTRIBUTED_EPILOG.format(command=command),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser(
        "run",
        help="fit everything here, or one distributed job with --job",
        description=(
            "Without --job: fit every rank in this process (all fit arguments "
            "needed). With --job N: fit job N's restarts using the plan.json in "
            "--output-dir (only --output-dir and --job; every other setting "
            "comes from the plan)."
        ),
    )
    run.add_argument("--output-dir", type=Path, required=True, help="results directory")
    run.add_argument(
        "--job",
        type=int,
        metavar="N",
        help="0-based job index, e.g. $SLURM_ARRAY_TASK_ID",
    )
    _add_fit_arguments(run, required=False, replicability=replicability)
    if replicability:
        _add_replicability_arguments(run, required=False)
    # With --job every setting comes from plan.json, so these are refused.
    run.set_defaults(
        planned_flags=tuple(
            flag
            for action in run._actions
            if action.dest not in ("help", "output_dir", "job")
            for flag in action.option_strings
        ),
    )
    plan = actions.add_parser(
        "plan",
        help="save plan.json and print the number of --job jobs",
        description=(
            "Save every argument to <output-dir>/plan.json and print only the "
            "number of jobs."
        ),
    )
    plan.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="results directory",
    )
    _add_fit_arguments(plan, required=True, replicability=replicability)
    plan.add_argument(
        "--overwrite",
        action="store_true",
        help="remove the stored restarts of the planned ranks first, whatever "
        "their settings (by default the jobs reuse those stored with the same "
        "settings and refit the others)",
    )
    if replicability:
        _add_replicability_arguments(plan, required=True)
    collect = actions.add_parser(
        "collect",
        help="gather the jobs' restarts and write the outputs",
        description="Pick each fit's best restart from the store (settings from plan.json).",
    )
    collect.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="results directory with plan.json",
    )


def _add_subject_arguments(parser: argparse.ArgumentParser, group_help: str) -> None:
    """`--subject-info`, `--group-variable` and `--output-dir`, required."""
    parser.add_argument(
        "--subject-info",
        type=Path,
        required=True,
        help="CSV with a subjects column",
    )
    parser.add_argument("--group-variable", required=True, help=group_help)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="results directory",
    )


def _add_map_arguments(parser: argparse.ArgumentParser, segmentation_help: str) -> None:
    """`--segmentation`, `--background` and `--slices` of the brain maps."""
    parser.add_argument("--segmentation", type=Path, help=segmentation_help)
    parser.add_argument(
        "--background",
        type=Path,
        help="image to draw on (default: segmentation mask)",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs=3,
        metavar=("I", "J", "K"),
        help="voxel indices of the volume reoriented to RAS (nibabel's "
        "as_closest_canonical), not of the file as stored (default: centre of "
        "mass)",
    )


def _add_test_arguments(parser: argparse.ArgumentParser, alpha_help: str) -> None:
    """`--alpha` and `--min-group-n` of the group tests."""
    parser.add_argument("--alpha", type=float, default=0.05, help=alpha_help)
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=2,
        help="subjects per group to test (default: 2)",
    )


def _add_statistics_plot(kinds: Any) -> None:
    parser = kinds.add_parser(
        "statistics",
        help="group differences per ROI over time, one figure per ROI",
        description=(
            "Combine labels into regions, compute the statistics, print a summary "
            "of significant group differences and draw one figure per ROI and "
            "statistic. With neither --region nor --rois, every preset region "
            "present in the data is used."
        ),
    )
    parser.add_argument(
        "--roi-signal",
        type=Path,
        required=True,
        help="roi_signal.parquet from gmri preprocess",
    )
    _add_subject_arguments(parser, "subject-info column to compare, e.g. diagnosis")
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "a preset (e.g. cortical_grey_matter) or name=ids "
            "(e.g. mine=17,53,1001-1035); repeatable"
        ),
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        metavar="ROI",
        help="'all' labels, or label ids to plot separately",
    )
    parser.add_argument(
        "--statistics",
        nargs="+",
        choices=STATISTICS,
        default=["median"],
        help="statistics to compute, each in its own table (default: median)",
    )
    parser.add_argument(
        "--relaxivity",
        type=float,
        help="r1 in 1/(mM s) for concentrations (default: 3.2; not for T1w)",
    )
    _add_test_arguments(parser, "FDR-adjusted p threshold (default: 0.05)")
    parser.add_argument(
        "--layout",
        choices=("rows", "panels"),
        default="rows",
        help="figure layout (default: rows)",
    )
    parser.add_argument(
        "--sd-bands",
        action="store_true",
        help="draw the cohort mean +/- 1, 2 and 3 SD behind the subject curves, "
        "to spot outlying subjects (rows layout only)",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="also draw every label for every subject and session (parenchyma, "
        "CSF and their regions, under figures/roi/overview/) and list the "
        "flagged values in roi_analysis/overview_flags__<statistic>.csv",
    )
    parser.add_argument(
        "--overview-z",
        type=float,
        default=3.5,
        metavar="Z",
        help="flag a value when its modified z-score (x - median) / (1.4826 MAD) "
        "among all subjects for that label and session exceeds Z in absolute "
        "value (default: 3.5, Iglewicz & Hoaglin 1993)",
    )
    _output_options(parser)


def _add_decomposition_plot(kinds: Any) -> None:
    parser = kinds.add_parser(
        "decomposition",
        help="mode grid, subject, time and spatial modes and outliers of one model",
        description=(
            "Plot a saved model (<method>_rank_<r>/ from gmri decompose). Without "
            "part flags every part is drawn; spatial parts of a per-ROI model "
            "need --segmentation."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="model directory <method>_rank_<r>/ from gmri decompose (or its "
        "model.h5)",
    )
    _add_subject_arguments(parser, "subject-info column to colour by")
    parts = parser.add_argument_group("parts (default: all)")
    parts.add_argument(
        "--mode-grid",
        action="store_true",
        help="time, subject and spatial modes in one figure",
    )
    parts.add_argument(
        "--subject-mode",
        action="store_true",
        help="boxplots by --group-variable and --categorical, scatters against "
        "--numerical",
    )
    parts.add_argument(
        "--time",
        action="store_true",
        help="CP time mode, or PARAFAC2/CMF evolving mode",
    )
    parts.add_argument(
        "--spatial",
        action="store_true",
        help="spatial mode on brain slices",
    )
    parts.add_argument(
        "--outliers",
        action="store_true",
        help="leverage against residual per subject, with Bonferroni cut-offs; "
        "reloads the data the model was fitted to",
    )
    _add_map_arguments(
        parser,
        "label volume in the decomposition's label space (CSF ids +10000)",
    )
    parser.add_argument(
        "--categorical",
        nargs="+",
        default=[],
        metavar="COLUMN",
        help="subject-info columns to box-plot the subject mode by, besides "
        "--group-variable (values are categories, also numeric codes)",
    )
    parser.add_argument(
        "--numerical",
        nargs="+",
        default=[],
        metavar="COLUMN",
        help="numeric subject-info columns to scatter the subject mode against",
    )
    parser.add_argument(
        "--min-panel-width",
        type=float,
        metavar="INCHES",
        help="widen the subject-mode figure beyond --page-width so no panel is "
        "narrower (default: fit --page-width)",
    )
    parser.add_argument(
        "--subject-prefix",
        default="",
        metavar="PREFIX",
        help="prepended as PREFIX_ to the subject-mode figure name, to keep "
        "figures for different variable sets apart",
    )
    _add_test_arguments(parser, "evolving-mode significance (default: 0.05)")
    parser.add_argument(
        "--min-subject-coefficient",
        type=float,
        default=NEGLIGIBLE_SUBJECT_WEIGHT,
        metavar="C",
        help="PARAFAC2/CMF: drop a subject from a component's time curves, "
        "ribbon and tests when |subject mode| < C (default: %(default)g; 0 "
        "drops only loadings the fit set to zero; ignored for CP)",
    )
    parser.add_argument(
        "--outlier-alpha",
        type=float,
        default=0.01,
        help="family-wise p for the outlier cut-offs, Bonferroni-corrected over "
        "subjects (default: %(default)g)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="data the model was fitted to, for --outliers (default: the input "
        "recorded in the model's decompose.json); the other settings always "
        "come from that record",
    )
    _output_options(parser)


def _add_group_maps_plot(kinds: Any) -> None:
    parser = kinds.add_parser(
        "group-maps",
        help="each group's average map per time point, on brain slices",
        description=(
            "Aggregate each group's subjects per ROI or voxel at every time point "
            "and draw one figure per time point and region (parenchyma, CSF), one "
            "row per group. Per-ROI input needs --segmentation."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="roi_signal.parquet (per ROI) or voxels.parquet (per voxel)",
    )
    _add_subject_arguments(parser, "subject-info column that defines the groups")
    parser.add_argument(
        "--aggregate",
        choices=AGGREGATES,
        default="trimmed",
        help="across subjects: trimmed mean, mean or median; non-finite values "
        "are ignored (default: trimmed)",
    )
    parser.add_argument(
        "--trim",
        type=int,
        default=2,
        help="values dropped at each end for the trimmed mean; fewer than "
        "2*trim+1 values give the median (default: 2)",
    )
    parser.add_argument(
        "--statistic",
        choices=("median", "mean"),
        default="median",
        help="per-ROI statistic (roi_signal input) (default: median)",
    )
    _add_map_arguments(
        parser,
        "label volume in roi_signal's label space (CSF ids +10000)",
    )
    parser.add_argument(
        "--timepoints",
        type=int,
        nargs="+",
        default=[],
        help="time points to draw (default: all)",
    )
    _output_options(parser)


def _add_reproducibility_plot(kinds: Any) -> None:
    parser = kinds.add_parser(
        "reproducibility",
        help="redraw a restart-vs-restart FMS matrix from its CSV",
        description=(
            "Redraw the reproducibility.csv that decompose run/collect writes "
            "next to each rank's restarts: pairwise FMS between restarts, "
            "sorted by reconstruction error."
        ),
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="<store>/<method>_rank_<r>/reproducibility.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="STEM",
        help="figure path without extension (default: next to the CSV)",
    )
    _output_options(parser)


def _add_plot(commands: Any) -> None:
    parser = commands.add_parser(
        "plot",
        help="ROI statistics, decomposition, group-average or reproducibility figures",
    )
    kinds = parser.add_subparsers(dest="action", required=True)
    _add_statistics_plot(kinds)
    _add_decomposition_plot(kinds)
    _add_group_maps_plot(kinds)
    _add_reproducibility_plot(kinds)


def _slices(values: list[int] | None) -> tuple[int, int, int] | None:
    return (values[0], values[1], values[2]) if values else None


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="gmri",
        description=(
            "gMRI tensor pipeline. Each command reads only the files earlier "
            "commands wrote. See docs/cli.md."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_preprocess(commands)
    _add_fit_command(
        commands,
        "decompose",
        "Signal tables -> CP/PARAFAC2/CMF fit per rank",
        False,
    )
    _add_fit_command(
        commands,
        "replicability",
        "Signal tables -> factor match scores per rank",
        True,
    )
    _add_plot(commands)
    return parser


def _fit_settings(args: argparse.Namespace) -> dict[str, Any]:
    missing = [
        f"--{name}"
        for name in ("input", "method", "ranks")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            f"{' '.join(missing)} required (or pass --job N to run a planned job)",
        )
    return {
        "input": args.input,
        "output_dir": args.output_dir,
        "method": args.method,
        "ranks": tuple(args.ranks),
        "statistic": args.statistic,
        "tensor": TensorOptions(
            scale=not args.no_scale,
            center=args.center,
            min_timepoints=args.min_timepoints,
            max_invalid_fraction=args.max_invalid_fraction,
        ),
        "fit": FitOptions(
            restarts=args.restarts,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            restart_procs=getattr(args, "restart_procs", 1),
            non_negative_modes=args.non_negative_modes,
            solver=args.solver,
            extra=dict(args.fit_option),
        ),
        "distributed": DistributedOptions(
            store=args.store,
            tasks_per_job=args.tasks_per_job,
        ),
    }


def _decomposition_options(args: argparse.Namespace) -> DecompositionOptions:
    return DecompositionOptions(**_fit_settings(args))


def _replicability_options(args: argparse.Namespace) -> ReplicabilityOptions:
    if args.engine is None or args.repeats is None:
        raise ValueError(
            "--engine and --repeats required (or pass --job N to run a planned job)",
        )
    return ReplicabilityOptions(
        **_fit_settings(args),
        engine=args.engine,
        repeats=args.repeats,
        splits=args.splits,
        subject_info=args.subject_info,
        stratify_by=args.stratify_by,
        n_procs=args.n_procs,
        seed=args.seed,
    )


def _run_preprocess(args: argparse.Namespace) -> None:
    pipeline.run_preprocessing(
        PreprocessingOptions(
            manifest=args.manifest,
            output_dir=args.output_dir,
            input_type=args.input_type,
            time_unit=args.time_unit,
            store_voxels=args.store_voxels,
            n_procs=args.n_procs,
        ),
    )


def _given_flags(args: argparse.Namespace) -> list[str]:
    return [token.split("=", 1)[0] for token in args.argv if token.startswith("--")]


def _refuse_planned_flags(args: argparse.Namespace) -> None:
    """Raise if `run --job` was given any setting that plan.json owns."""
    refused = [flag for flag in _given_flags(args) if flag in args.planned_flags]
    if refused:
        raise ValueError(
            f"{refused[0]} cannot be used with --job: a planned job takes its "
            f"settings from {Path(args.output_dir) / 'plan.json'}; run `plan` "
            "again to change them",
        )


def _run_fit_command(args: argparse.Namespace) -> None:
    decompose = args.command == "decompose"
    if args.action == "collect":
        (
            pipeline.collect_decomposition
            if decompose
            else pipeline.collect_replicability
        )(
            args.output_dir,
        )
    elif args.action == "plan":
        if decompose:
            print(
                pipeline.plan_decomposition(
                    _decomposition_options(args),
                    overwrite=args.overwrite,
                ),
            )
        else:
            print(
                pipeline.plan_replicability_jobs(
                    _replicability_options(args),
                    overwrite=args.overwrite,
                ),
            )
    elif args.job is not None:
        _refuse_planned_flags(args)
        (
            pipeline.run_decomposition_job
            if decompose
            else pipeline.run_replicability_job
        )(
            args.output_dir,
            args.job,
        )
    elif "--tasks-per-job" in _given_flags(args):
        raise ValueError(
            "--tasks-per-job sets the job size of `plan`; `run` without --job "
            "fits everything in this process",
        )
    elif decompose:
        pipeline.run_decomposition(_decomposition_options(args))
    else:
        pipeline.run_replicability(_replicability_options(args))


def _plot_reproducibility(args: argparse.Namespace) -> None:
    written = pipeline.run_reproducibility_plot(
        args.csv,
        args.output,
        page_width=_page_width(args.page_width),
        formats=tuple(args.formats),
        dpi=args.dpi,
    )
    print(f"{len(written)} figure file(s): {', '.join(map(str, written))}.")


def _plot_statistics(args: argparse.Namespace) -> None:
    if "--overview-z" in _given_flags(args) and not args.overview:
        raise ValueError("--overview-z needs --overview")
    rois: Any = None
    if args.rois is not None:
        rois = "all" if args.rois == ["all"] else tuple(args.rois)
    options = StatisticsPlotOptions(
        roi_signal=args.roi_signal,
        subject_info=args.subject_info,
        group_variable=args.group_variable,
        output_dir=args.output_dir,
        regions=tuple(args.region),
        rois=rois,
        statistics=tuple(args.statistics),
        relaxivity=args.relaxivity,
        alpha=args.alpha,
        min_group_n=args.min_group_n,
        layout=args.layout,
        sd_bands=args.sd_bands,
        overview=args.overview,
        overview_z=args.overview_z,
        page_width=_page_width(args.page_width),
        formats=tuple(args.formats),
        dpi=args.dpi,
    )
    result = pipeline.run_statistics_plots(options)
    print(pipeline.format_significance_summary(result.summary, options.alpha))
    print(
        f"Tables: {options.output_dir / 'roi_analysis'}; {len(result.figures)} figure file(s).",
    )


def _plot_group_maps(args: argparse.Namespace) -> None:
    written = pipeline.run_group_map_plots(
        GroupMapPlotOptions(
            input=args.input,
            subject_info=args.subject_info,
            group_variable=args.group_variable,
            output_dir=args.output_dir,
            aggregate=args.aggregate,
            trim=args.trim,
            statistic=args.statistic,
            segmentation=args.segmentation,
            background=args.background,
            slices=_slices(args.slices),
            timepoints=tuple(args.timepoints),
            page_width=_page_width(args.page_width),
            formats=tuple(args.formats),
            dpi=args.dpi,
        ),
    )
    print(
        f"{len(written)} figure file(s) in {args.output_dir / 'figures' / 'group_maps'}.",
    )


def _plot_decomposition(args: argparse.Namespace) -> None:
    parts = tuple(part for part in DECOMPOSITION_PARTS if getattr(args, part))
    written = pipeline.run_decomposition_plots(
        DecompositionPlotOptions(
            model=args.model,
            subject_info=args.subject_info,
            group_variable=args.group_variable,
            output_dir=args.output_dir,
            parts=parts,
            segmentation=args.segmentation,
            background=args.background,
            slices=_slices(args.slices),
            categorical=tuple(args.categorical),
            numerical=tuple(args.numerical),
            min_panel_width=args.min_panel_width,
            subject_prefix=args.subject_prefix,
            alpha=args.alpha,
            min_group_n=args.min_group_n,
            min_subject_coefficient=args.min_subject_coefficient,
            outlier_alpha=args.outlier_alpha,
            input=args.input,
            page_width=_page_width(args.page_width),
            formats=tuple(args.formats),
            dpi=args.dpi,
        ),
    )
    print(
        f"{len(written)} figure file(s) in {args.output_dir / 'figures' / 'decomposition'}.",
    )


_PLOTS: dict[str, Callable[[argparse.Namespace], None]] = {
    "statistics": _plot_statistics,
    "decomposition": _plot_decomposition,
    "group-maps": _plot_group_maps,
    "reproducibility": _plot_reproducibility,
}


def _run_plot(args: argparse.Namespace) -> None:
    _PLOTS[args.action](args)


def main(argv: Sequence[str] | None = None) -> int:
    """Run `gmri ...`; 0 on success, 2 on a user error."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    args.argv = argv
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "preprocess": _run_preprocess,
        "decompose": _run_fit_command,
        "replicability": _run_fit_command,
        "plot": _run_plot,
    }
    try:
        handlers[args.command](args)
    except (FileNotFoundError, ValueError, ConvergenceError) as error:
        print(f"gmri {args.command}: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
