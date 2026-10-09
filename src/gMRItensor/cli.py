"""`gmri` command line: one subcommand per stage, each with its own config.

    gmri preprocess    <config>
    gmri plot          <config>
    gmri decompose     run [--job N] | plan | collect  <config>
    gmri replicability run [--job N] | plan | collect  <config>

Config options are documented in docs/configuration.md.
"""
import argparse
import sys
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from gMRItensor import config as configs
from gMRItensor import pipeline

_CONFIG_DOCS = "Config options: docs/configuration.md (examples in examples/)."

_DISTRIBUTED_EPILOG = """\
Distributed restarts (e.g. a SLURM array), using the config's `distributed`
block:
  N=$(gmri {command} plan {name}.yaml)          # number of jobs
  sbatch --array=0-$((N-1)) ...  gmri {command} run {name}.yaml --job $SLURM_ARRAY_TASK_ID
  gmri {command} collect {name}.yaml            # once every job has finished

Without --job, `run` fits everything in this process.
"""


def _add_config(parser: argparse.ArgumentParser, stage: str) -> None:
    parser.add_argument(
        "config",
        help=f"path to the {stage} YAML config (paths in it are relative to it)",
    )


def _print_jobs(n_jobs: int) -> None:
    print(n_jobs)


def _staged(
    run: Callable[[Any], Any],
    run_job: Callable[[Any, int], Any],
    plan: Callable[[Any], int],
    collect: Callable[[Any], Any],
) -> dict[str, Callable[[Any, argparse.Namespace], Any]]:
    """`run`/`plan`/`collect` actions of a stage with distributed restarts."""
    return {
        "run": lambda config, args: (
            run(config) if args.job is None else run_job(config, args.job)
        ),
        "plan": lambda config, args: _print_jobs(plan(config)),
        "collect": lambda config, args: collect(config),
    }


def _add_staged_command(
    subparsers: Any,
    command: str,
    name: str,
    summary: str,
) -> None:
    parser = subparsers.add_parser(
        command,
        help=summary,
        description=f"{summary}. {_CONFIG_DOCS}",
        epilog=_DISTRIBUTED_EPILOG.format(command=command, name=name),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser(
        "run",
        help="fit every rank here, or one distributed job with --job",
        description=(
            "Without --job: fit every rank in this process and write the "
            "outputs. With --job N: fit job N's restarts into the restart "
            "store (finished restarts are skipped, so a re-queued job resumes)."
        ),
    )
    _add_config(run, name)
    run.add_argument(
        "--job",
        type=int,
        metavar="N",
        help="0-based job index, e.g. $SLURM_ARRAY_TASK_ID",
    )
    plan = actions.add_parser(
        "plan",
        help="print the number of --job jobs to submit",
        description="Print the number of `run --job` jobs, and nothing else.",
    )
    _add_config(plan, name)
    collect = actions.add_parser(
        "collect",
        help="gather the jobs' restarts and write the outputs",
        description=(
            "Pick each rank's best restart from the store and write the same "
            "outputs as `run` without --job. Fails if any restart is missing."
        ),
    )
    _add_config(collect, name)


# command -> (loader, {action: handler}); `None` action for single-step stages.
_COMMANDS: dict[
    str,
    tuple[Callable[[Any], Any], dict[Any, Callable[[Any, argparse.Namespace], Any]]],
] = {
    "preprocess": (
        configs.load_preprocessing_config,
        {None: lambda config, args: pipeline.run_preprocessing(config)},
    ),
    "plot": (
        configs.load_plotting_config,
        {None: lambda config, args: pipeline.run_plotting(config)},
    ),
    "decompose": (
        configs.load_decomposition_config,
        _staged(
            pipeline.run_decomposition,
            pipeline.run_decomposition_job,
            pipeline.plan_decomposition,
            pipeline.collect_decomposition,
        ),
    ),
    "replicability": (
        configs.load_replicability_config,
        _staged(
            pipeline.run_replicability,
            pipeline.run_replicability_job,
            pipeline.plan_replicability_jobs,
            pipeline.collect_replicability,
        ),
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmri",
        description=(
            "Run one gMRI-tensor stage from its YAML config. Each stage reads "
            f"only the files earlier stages wrote. {_CONFIG_DOCS}"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preprocess = commands.add_parser(
        "preprocess",
        help="images -> tracer and ROI statistics tables",
        description=(
            "Read every scan in the manifest once and write "
            f"<output_dir>/data/. {_CONFIG_DOCS}"
        ),
    )
    _add_config(preprocess, "preprocessing")
    plot = commands.add_parser(
        "plot",
        help="ROI statistics -> group tables and figures",
        description=(
            "Write group summary/significance tables, then draw the "
            f"configured figures and grids from them. {_CONFIG_DOCS}"
        ),
    )
    _add_config(plot, "plotting")
    _add_staged_command(
        commands,
        "decompose",
        "decomposition",
        "Tracer table -> CP/PARAFAC2 fit per rank",
    )
    _add_staged_command(
        commands,
        "replicability",
        "replicability",
        "Tracer table -> factor match scores per rank",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run `gmri ...`; 0 on success, 2 on a user error."""
    args = _parser().parse_args(argv)
    loader, actions = _COMMANDS[args.command]
    action = getattr(args, "action", None)
    try:
        actions[action](loader(args.config), args)
    except (FileNotFoundError, ValueError) as error:  # incl. ConfigError
        print(f"gmri {args.command}: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
