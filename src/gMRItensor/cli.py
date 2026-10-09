"""`gmri` command line. Every setting is an argument; see `gmri <command> --help`.

    gmri preprocess    --manifest scans.csv --output-dir results --input-type T1map
    gmri decompose     run | plan | collect   (run --job N for one distributed job)
    gmri replicability run | plan | collect
    gmri plot          statistics | decomposition

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
from gMRItensor.options import DECOMPOSITION_PARTS
from gMRItensor.options import DecompositionOptions
from gMRItensor.options import DecompositionPlotOptions
from gMRItensor.options import DistributedOptions
from gMRItensor.options import FitOptions
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


def _add_fit_arguments(parser: argparse.ArgumentParser, required: bool) -> None:
    """Input, tensor and fit arguments shared by decompose and replicability."""
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
        help="minimum valid time points per subject (default: all)",
    )
    tensor.add_argument(
        "--max-invalid-fraction",
        type=float,
        default=0.9,
        help="drop scans with a larger non-finite fraction (default: 0.9)",
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
    fit.add_argument(
        "--restart-procs",
        type=int,
        default=1,
        help="parallel restart processes (decompose) (default: 1)",
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
        help="PARAFAC2 solver (default: tensorly); CMF always uses matcouply",
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
            "--output-dir (only --output-dir and --job needed)."
        ),
    )
    run.add_argument("--output-dir", type=Path, required=True, help="results directory")
    run.add_argument(
        "--job",
        type=int,
        metavar="N",
        help="0-based job index, e.g. $SLURM_ARRAY_TASK_ID",
    )
    _add_fit_arguments(run, required=False)
    if replicability:
        _add_replicability_arguments(run, required=False)
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
    _add_fit_arguments(plan, required=True)
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


def _add_plot(commands: Any) -> None:
    parser = commands.add_parser("plot", help="ROI statistics or decomposition figures")
    kinds = parser.add_subparsers(dest="action", required=True)
    statistics = kinds.add_parser(
        "statistics",
        help="group differences per ROI over time, one figure per ROI",
        description=(
            "Combine labels into regions, compute the statistics, print a summary "
            "of significant group differences and draw one figure per ROI and "
            "statistic. With neither --region nor --rois, every preset region "
            "present in the data is used."
        ),
    )
    statistics.add_argument(
        "--roi-signal",
        type=Path,
        required=True,
        help="roi_signal.parquet from gmri preprocess",
    )
    statistics.add_argument(
        "--subject-info",
        type=Path,
        required=True,
        help="CSV with a subjects column",
    )
    statistics.add_argument(
        "--group-variable",
        required=True,
        help="subject-info column to compare, e.g. diagnosis",
    )
    statistics.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="results directory",
    )
    statistics.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "a preset (e.g. cortical_grey_matter) or name=ids "
            "(e.g. mine=17,53,1001-1035); repeatable"
        ),
    )
    statistics.add_argument(
        "--rois",
        nargs="+",
        metavar="ROI",
        help="'all' labels, or label ids to plot separately",
    )
    statistics.add_argument(
        "--statistics",
        nargs="+",
        choices=STATISTICS,
        default=["median"],
        help="statistics to compute, each in its own table (default: median)",
    )
    statistics.add_argument(
        "--relaxivity",
        type=float,
        help="r1 in 1/(mM s) for concentrations (default: 3.2; not for T1w)",
    )
    statistics.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR-adjusted p threshold (default: 0.05)",
    )
    statistics.add_argument(
        "--min-group-n",
        type=int,
        default=2,
        help="subjects per group to test (default: 2)",
    )
    statistics.add_argument(
        "--layout",
        choices=("rows", "panels"),
        default="rows",
        help="figure layout (default: rows)",
    )
    _output_options(statistics)

    decomposition = kinds.add_parser(
        "decomposition",
        help="mode grid, subject mode, time mode and spatial maps of one model",
        description=(
            "Plot a saved rank_<r>.h5. Without part flags every part is drawn; "
            "spatial parts of a per-ROI model need --segmentation."
        ),
    )
    decomposition.add_argument(
        "--model",
        type=Path,
        required=True,
        help="rank_<r>.h5 from gmri decompose",
    )
    decomposition.add_argument(
        "--subject-info",
        type=Path,
        required=True,
        help="CSV with a subjects column",
    )
    decomposition.add_argument(
        "--group-variable",
        required=True,
        help="subject-info column to colour by",
    )
    decomposition.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="results directory",
    )
    parts = decomposition.add_argument_group("parts (default: all)")
    parts.add_argument(
        "--mode-grid",
        action="store_true",
        help="time, subject and spatial modes in one figure",
    )
    parts.add_argument(
        "--subject-mode",
        action="store_true",
        help="group boxplots (and --covariates)",
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
    decomposition.add_argument(
        "--segmentation",
        type=Path,
        help="label volume in the decomposition's label space (CSF ids +10000)",
    )
    decomposition.add_argument(
        "--background",
        type=Path,
        help="image to draw on (default: segmentation mask)",
    )
    decomposition.add_argument(
        "--slices",
        type=int,
        nargs=3,
        metavar=("I", "J", "K"),
        help="slice indices (default: centre of mass)",
    )
    decomposition.add_argument(
        "--covariates",
        nargs="+",
        default=[],
        help="subject-info columns to scatter",
    )
    decomposition.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="evolving-mode significance (default: 0.05)",
    )
    decomposition.add_argument(
        "--min-group-n",
        type=int,
        default=2,
        help="subjects per group to test (default: 2)",
    )
    _output_options(decomposition)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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
            restart_procs=args.restart_procs,
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
            print(pipeline.plan_decomposition(_decomposition_options(args)))
        else:
            print(pipeline.plan_replicability_jobs(_replicability_options(args)))
    elif args.job is not None:
        (
            pipeline.run_decomposition_job
            if decompose
            else pipeline.run_replicability_job
        )(
            args.output_dir,
            args.job,
        )
    elif decompose:
        pipeline.run_decomposition(_decomposition_options(args))
    else:
        pipeline.run_replicability(_replicability_options(args))


def _run_plot(args: argparse.Namespace) -> None:
    if args.action == "statistics":
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
            page_width=_page_width(args.page_width),
            formats=tuple(args.formats),
            dpi=args.dpi,
        )
        result = pipeline.run_statistics_plots(options)
        print(pipeline.format_significance_summary(result.summary, options.alpha))
        print(
            f"Tables: {options.output_dir / 'roi_analysis'}; {len(result.figures)} figure file(s).",
        )
        return
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
            slices=(args.slices[0], args.slices[1], args.slices[2])
            if args.slices
            else None,
            covariates=tuple(args.covariates),
            alpha=args.alpha,
            min_group_n=args.min_group_n,
            page_width=_page_width(args.page_width),
            formats=tuple(args.formats),
            dpi=args.dpi,
        ),
    )
    print(
        f"{len(written)} figure file(s) in {args.output_dir / 'figures' / 'decomposition'}.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run `gmri ...`; 0 on success, 2 on a user error."""
    args = _parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "preprocess": _run_preprocess,
        "decompose": _run_fit_command,
        "replicability": _run_fit_command,
        "plot": _run_plot,
    }
    try:
        handlers[args.command](args)
    except (FileNotFoundError, ValueError) as error:
        print(f"gmri {args.command}: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
