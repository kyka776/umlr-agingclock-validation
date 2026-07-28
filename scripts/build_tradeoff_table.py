"""Derive the reviewer-facing UMLR trade-off table from benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).parents[1]
    source = pd.read_csv(root / "results" / "tables" / "simulation_summary.csv")
    indexed = source.set_index(["scenario", "method"])
    rows = []
    for scenario in source["scenario"].drop_duplicates():
        raw = indexed.loc[(scenario, "raw_lasso")]
        umlr = indexed.loc[(scenario, "umlr_oracle")]
        rows.append(
            {
                "scenario": scenario,
                "raw_absolute_residual_age_slope": abs(raw["residual_age_slope_mean"]),
                "umlr_absolute_residual_age_slope": abs(
                    umlr["residual_age_slope_mean"]
                ),
                "umlr_rmse_ratio_vs_raw": umlr["rmse_ratio_vs_raw"],
                "raw_association_absolute_error": raw[
                    "association_error_age_adjusted_abs_mean"
                ],
                "umlr_association_absolute_error": umlr[
                    "association_error_age_adjusted_abs_mean"
                ],
                "umlr_target_CI_inclusion_rate": umlr[
                    "latent_target_interval_inclusion_rate"
                ],
            }
        )
    output = pd.DataFrame(rows)
    path = root / "results" / "tables" / "tradeoff_table.csv"
    output.to_csv(path, index=False)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
