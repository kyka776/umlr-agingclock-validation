"""Run the aggregate-only OmniAge tutorial-data demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aging_clock_audit.plotting import make_real_data_figure
from aging_clock_audit.real_data import run_omniage_example


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/omniager_lung_inv.rds"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/figures/omniage-holdout.png"),
    )
    arguments = parser.parse_args()
    result = run_omniage_example(arguments.input)
    arguments.output.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(arguments.output / "omniage_holdout_metrics.csv", index=False)
    result.age_bins.to_csv(arguments.output / "omniage_holdout_age_bins.csv", index=False)
    (arguments.output / "omniage_holdout_provenance.json").write_text(
        json.dumps(result.provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    make_real_data_figure(result.metrics, result.age_bins, arguments.figure)
    print(f"wrote aggregate results to {arguments.output}")
    print(f"wrote {arguments.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
