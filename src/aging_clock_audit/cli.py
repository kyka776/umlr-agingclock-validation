"""Command-line entry point for reproducible audits."""

from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import make_main_figure
from .simulation import run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aging-clock-audit",
        description="Run the independent UMLR aging-clock validation benchmark.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    benchmark = subcommands.add_parser("benchmark", help="run all frozen simulations")
    benchmark.add_argument("--replications", type=int, default=30)
    benchmark.add_argument("--seed", type=int, default=20260729)
    benchmark.add_argument("--lambda", dest="lambda_", type=float, default=0.02)
    benchmark.add_argument(
        "--tables", type=Path, default=Path("results/tables")
    )
    benchmark.add_argument(
        "--figure",
        type=Path,
        default=Path("results/figures/simulation-benchmark.png"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "benchmark":
        _, summary = run_benchmark(
            replications=arguments.replications,
            base_seed=arguments.seed,
            lambda_=arguments.lambda_,
            output_directory=arguments.tables,
        )
        figure = make_main_figure(summary, arguments.figure)
        print(f"wrote {arguments.tables / 'simulation_summary.csv'}")
        print(f"wrote {figure}")
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
