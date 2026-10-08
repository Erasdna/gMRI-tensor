"""`gmri` command line: one subcommand per stage, each with its own config."""
import argparse
import sys
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from gMRItensor import config as configs
from gMRItensor import pipeline

# subcommand -> (loader, stage runner, help)
_COMMANDS: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any], str]] = {
    "preprocess": (
        configs.load_preprocessing_config,
        pipeline.run_preprocessing,
        "images -> tracer and ROI statistics tables",
    ),
    "plot": (
        configs.load_plotting_config,
        pipeline.run_plotting,
        "ROI statistics -> group tables and figures",
    ),
    "decompose": (
        configs.load_decomposition_config,
        pipeline.run_decomposition,
        "tracer table -> CP/PARAFAC2 fits per rank",
    ),
    "replicability": (
        configs.load_replicability_config,
        pipeline.run_replicability,
        "tracer table -> factor match scores per rank",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmri",
        description="Run one gMRI-tensor stage from its YAML config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, help_text) in _COMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("config", help="path to the stage's YAML config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run `gmri <command> <config>`; 0 on success, 2 on a user error."""
    args = _parser().parse_args(argv)
    loader, runner, _ = _COMMANDS[args.command]
    try:
        runner(loader(args.config))
    except (FileNotFoundError, ValueError) as error:  # incl. ConfigError
        print(f"gmri {args.command}: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
