"""Build and execute the reader-facing audit notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    root = Path(__file__).parents[1]
    notebook_path = root / "notebooks" / "benchmark.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11+"}
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            """# UMLR aging-clock audit

## TL;DR

The authors' constrained fit consistently reduces residual-age slope in the
frozen simulations, but it is not a free correction: chronological-age RMSE
increases in most scenarios. The small open-data holdout does not support a
general performance recommendation. This notebook reads only committed
aggregate outputs; it contains no raw methylation or sample identifiers."""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
summary = pd.read_csv(TABLES / "simulation_summary.csv")
real = pd.read_csv(TABLES / "omniage_holdout_metrics.csv")
print(f"Simulation rows: {len(summary)}; real-data methods: {len(real)}")"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Context & Methods

Four methods are compared on identical train/test splits: raw Lasso, a common
residual correction, linear recalibration, and an independent numerical oracle
for the authors' UMLR constraints. The oracle is fitted on training data only.
It is an audit surface, not a replacement for the official R implementation."""
        ),
        nbformat.v4.new_code_cell(
            """tradeoff = (
    summary.loc[summary["method"].eq("umlr_oracle"), [
        "scenario", "residual_age_slope_mean", "rmse_ratio_vs_raw",
        "association_error_age_adjusted_abs_mean",
        "latent_target_interval_inclusion_rate",
    ]]
    .rename(columns={
        "residual_age_slope_mean": "residual_age_slope",
        "association_error_age_adjusted_abs_mean": "association_error",
        "latent_target_interval_inclusion_rate": "target_CI_inclusion",
    })
)
display(tradeoff.round(3))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Data

The simulation benchmark uses ten pre-specified scenarios and 30 replications
per scenario. The real-data demonstration uses the pinned OmniAge lung tutorial
dataset (56 samples, 2,974 methylation features), with a frozen group-by-age
stratified 38/18 split. Feature selection and scaling use training data only.
Only aggregate metrics and age-bin means are retained."""
        ),
        nbformat.v4.new_code_cell(
            """real_view = real[[
    "method", "mae", "rmse", "residual_age_slope",
    "outcome_beta_age_adjusted", "outcome_ci_low_age_adjusted",
    "outcome_ci_high_age_adjusted",
]]
display(real_view.round(3))"""
        ),
        nbformat.v4.new_markdown_cell("## Results"),
        nbformat.v4.new_code_cell(
            """display(Image(filename=str(FIGURES / "simulation-benchmark.png"), width=1100))
display(Image(filename=str(FIGURES / "omniage-holdout.png"), width=1100))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Takeaways

- UMLR reduced the absolute residual-age slope in every biased simulation.
- Its RMSE exceeded raw Lasso in seven of ten scenarios and rose by about 30%
  under regression-to-the-mean and 35% under age imbalance.
- Association recovery improved in most signal scenarios, but not in the null
  outcome or nonlinear stress test.
- On the small open holdout, all methods generalized poorly; UMLR had the
  highest MAE and the widest disease-state association interval.
- Use therefore requires an external calibration cohort, an explicit
  error-versus-bias trade-off, and validation for the intended downstream
  association. Do not silently refit on test ages."""
        ),
        nbformat.v4.new_markdown_cell(
            """## Key Assumptions

1. The two-region equations in the preprint are the controlling specification.
2. The independent SciPy oracle is for falsification, not runtime parity with
   the unlicensed official R source.
3. The open lung cohort is exploratory and small; no clinical claim is made.
4. Simulation conclusions are conditional on the frozen data-generating
   processes and penalty value.
5. The authors' controlled UK Biobank and Framingham analyses cannot be
   reproduced here without authorized data access."""
        ),
    ]
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbformat.write(notebook, notebook_path)
    print(f"wrote executed notebook {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
