"""Static research figures with explicit scales and visual QA-friendly layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle

from .simulation import METHODS, SCENARIOS

METHOD_LABELS = {
    "raw_lasso": "Raw Lasso",
    "residual_correction": "Residual correction",
    "linear_recalibration": "Linear recalibration",
    "umlr_oracle": "UMLR oracle",
}

SCENARIO_LABELS = {
    "unbiased_predictions": "Unbiased",
    "regression_to_mean": "RTM",
    "range_shift": "Range shift",
    "heteroscedastic_errors": "Heteroscedastic",
    "age_imbalance": "Age imbalance",
    "null_outcome": "Null outcome",
    "latent_signal": "Latent signal",
    "age_confounding": "Age confounding",
    "small_calibration_sample": "Small n",
    "nonlinear_bias": "Nonlinear",
}


def _matrix(summary: pd.DataFrame, value: str) -> np.ndarray:
    table = summary.pivot(index="method", columns="scenario", values=value)
    return table.reindex(index=METHODS, columns=SCENARIOS).to_numpy(dtype=float)


def _annotate(ax: plt.Axes, values: np.ndarray, *, decimals: int = 2) -> None:
    finite = np.abs(values[np.isfinite(values)])
    threshold = float(np.nanpercentile(finite, 70)) if finite.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if abs(value) >= threshold and threshold > 0 else "#20242A"
            ax.text(
                column,
                row,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.2,
                family="DejaVu Sans Mono",
            )


def _blossom(fig: plt.Figure) -> None:
    center_x, center_y = 0.972, 0.962
    radius = 0.007
    for angle in np.linspace(0, 2 * np.pi, 6)[:-1]:
        fig.add_artist(
            Circle(
                (center_x + 0.011 * np.cos(angle), center_y + 0.011 * np.sin(angle)),
                radius,
                transform=fig.transFigure,
                facecolor="#D89A2B",
                edgecolor="#28323C",
                linewidth=0.7,
                clip_on=False,
            )
        )
    fig.add_artist(
        Circle(
            (center_x, center_y),
            radius * 0.7,
            transform=fig.transFigure,
            facecolor="#FFFFFF",
            edgecolor="#28323C",
            linewidth=0.7,
            clip_on=False,
        )
    )


def make_main_figure(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Render the three-part simulation trade-off figure."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    blue = "#2E6F9E"
    orange = "#D36B36"
    neutral = "#F7F8FA"
    blue_map = LinearSegmentedColormap.from_list("blue_scale", [neutral, blue])
    diverging = LinearSegmentedColormap.from_list(
        "blue_orange", [blue, neutral, orange]
    )

    panels = [
        (
            np.abs(_matrix(summary, "residual_age_slope_mean")),
            "A. Absolute residual-age slope",
            "Ideal = 0; mean across pre-specified replications",
            blue_map,
            None,
        ),
        (
            _matrix(summary, "rmse_ratio_vs_raw"),
            "B. Chronological-age RMSE ratio vs raw Lasso",
            "Below 1 is lower RMSE; above 1 is the calibration trade-off",
            diverging,
            TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.8),
        ),
        (
            _matrix(summary, "association_error_age_adjusted_abs_mean"),
            "C. Absolute error in age-adjusted outcome association",
            "Compared with the latent biological-age-gap target",
            blue_map,
            None,
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.94, top=0.89, bottom=0.12, hspace=0.72)
    fig.suptitle(
        "UMLR aging-clock calibration benchmark",
        x=0.16,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#20242A",
    )
    fig.text(
        0.16,
        0.93,
        "Four methods · ten frozen scenarios · identical train/test splits per replication",
        ha="left",
        fontsize=10.5,
        color="#59636E",
    )
    _blossom(fig)

    for axis, (values, title, subtitle, color_map, norm) in zip(axes, panels, strict=True):
        image = axis.imshow(values, aspect="auto", cmap=color_map, norm=norm)
        if norm is None:
            image.set_clim(vmin=0.0, vmax=max(float(np.nanpercentile(values, 95)), 1e-9))
        _annotate(axis, values)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#20242A", pad=20)
        axis.text(
            0.0,
            1.04,
            subtitle,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#59636E",
        )
        axis.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[name] for name in METHODS])
        axis.set_xticks(
            np.arange(len(SCENARIOS)),
            [SCENARIO_LABELS[name] for name in SCENARIOS],
            rotation=25,
            ha="right",
        )
        axis.tick_params(axis="both", length=0, labelsize=8.5, pad=5)
        for spine in axis.spines.values():
            spine.set_edgecolor("#CBD1D8")
            spine.set_linewidth(0.8)
        color_bar = fig.colorbar(image, ax=axis, fraction=0.018, pad=0.018)
        color_bar.ax.tick_params(labelsize=7, colors="#59636E")
        color_bar.outline.set_edgecolor("#CBD1D8")

    fig.text(
        0.16,
        0.025,
        "Source: deterministic simulations in this repository. UMLR is an independent "
        "audit oracle, not a replacement package.",
        ha="left",
        fontsize=8.5,
        color="#59636E",
    )
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def make_real_data_figure(
    metrics: pd.DataFrame,
    age_bins: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Render aggregate-only holdout diagnostics; no sample-level points."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#777F88", "#2E6F9E", "#D89A2B", "#2C7A68"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.23, wspace=0.38)
    fig.suptitle(
        "Open-data holdout: OmniAge lung tutorial cohort",
        x=0.07,
        y=0.95,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#20242A",
    )
    fig.text(
        0.07,
        0.875,
        "56 samples · frozen stratified split · 18-sample untouched test set · "
        "aggregate outputs only",
        ha="left",
        fontsize=9.5,
        color="#59636E",
    )
    _blossom(fig)

    for color, method in zip(colors, METHODS, strict=True):
        selected = age_bins.loc[age_bins["method"] == method].sort_values("age_bin")
        axes[0].plot(
            selected["mean_chronological_age"],
            selected["mean_predicted_age"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=METHOD_LABELS[method],
        )
    limits = [
        float(
            min(
                age_bins["mean_chronological_age"].min(),
                age_bins["mean_predicted_age"].min(),
            )
        )
        - 2,
        float(
            max(
                age_bins["mean_chronological_age"].max(),
                age_bins["mean_predicted_age"].max(),
            )
        )
        + 2,
    ]
    axes[0].plot(limits, limits, linestyle="--", color="#AAB1B9", linewidth=1)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_title("A. Age-bin calibration", loc="left", fontweight="bold")
    axes[0].set_xlabel("Mean chronological age")
    axes[0].set_ylabel("Mean predicted age")

    ordered = metrics.set_index("method").reindex(METHODS)
    positions = np.arange(len(METHODS))
    axes[1].bar(positions, ordered["mae"], color=colors, width=0.68)
    for position, value in zip(positions, ordered["mae"], strict=True):
        axes[1].text(position, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("B. Holdout MAE", loc="left", fontweight="bold")
    axes[1].set_ylabel("Years · lower is better")
    axes[1].set_xticks(
        positions,
        [METHOD_LABELS[name] for name in METHODS],
        rotation=28,
        ha="right",
    )

    coefficients = ordered["outcome_beta_age_adjusted"].to_numpy(dtype=float)
    lower = ordered["outcome_ci_low_age_adjusted"].to_numpy(dtype=float)
    upper = ordered["outcome_ci_high_age_adjusted"].to_numpy(dtype=float)
    axes[2].axhline(0.0, color="#AAB1B9", linewidth=1, linestyle="--")
    for position, coefficient, low, high, color in zip(
        positions, coefficients, lower, upper, colors, strict=True
    ):
        axes[2].errorbar(
            position,
            coefficient,
            yerr=np.array([[coefficient - low], [high - coefficient]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5,
            elinewidth=2,
            capsize=4,
            zorder=3,
        )
    axes[2].set_title("C. Disease-state association", loc="left", fontweight="bold")
    axes[2].set_ylabel("Age-adjusted coefficient · 95% CI")
    axes[2].set_xticks(
        positions,
        [METHOD_LABELS[name] for name in METHODS],
        rotation=28,
        ha="right",
    )

    for axis in axes:
        axis.grid(axis="y", color="#E7E9EC", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#CBD1D8")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.text(
        0.07,
        0.025,
        "Exploratory only. Cohort, features and split were not selected to favor a method; "
        "no raw methylation or sample identifiers are included.",
        ha="left",
        fontsize=8,
        color="#59636E",
    )
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output
