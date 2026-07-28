"""Pre-specified simulation benchmark for the UMLR method."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import diagnostic_metrics, downstream_association
from .models import (
    ConstrainedLassoOracle,
    LinearRecalibration,
    ResidualCorrection,
    StandardLasso,
)

SCENARIOS = (
    "unbiased_predictions",
    "regression_to_mean",
    "range_shift",
    "heteroscedastic_errors",
    "age_imbalance",
    "null_outcome",
    "latent_signal",
    "age_confounding",
    "small_calibration_sample",
    "nonlinear_bias",
)

METHODS = ("raw_lasso", "residual_correction", "linear_recalibration", "umlr_oracle")


@dataclass(frozen=True)
class SimulatedSplit:
    train_predictors: np.ndarray
    train_age: np.ndarray
    train_latent_age: np.ndarray
    train_outcome: np.ndarray
    test_predictors: np.ndarray
    test_age: np.ndarray
    test_latent_age: np.ndarray
    test_outcome: np.ndarray


def _ages(
    rng: np.random.Generator,
    n: int,
    *,
    low: float,
    high: float,
    imbalanced: bool = False,
) -> np.ndarray:
    if imbalanced:
        return low + (high - low) * rng.beta(2.0, 6.0, size=n)
    return rng.uniform(low, high, size=n)


def _make_predictors(
    rng: np.random.Generator,
    latent_age: np.ndarray,
    chronological_age: np.ndarray,
    *,
    noise_scale: float,
    heteroscedastic: bool,
    nonlinear: bool,
) -> np.ndarray:
    centered = (latent_age - 55.0) / 15.0
    if nonlinear:
        signal = np.column_stack(
            [
                np.sin(centered * 1.8),
                centered**2,
                np.tanh(centered),
                np.cos(centered),
                np.sign(centered) * np.sqrt(np.abs(centered) + 1e-8),
                centered**3,
            ]
        )
    else:
        loadings = np.array([1.0, 0.8, -0.7, 0.55, -0.4, 0.25])
        signal = centered[:, None] * loadings[None, :]
    if heteroscedastic:
        row_scale = noise_scale * (0.35 + (chronological_age - 20.0) / 60.0)
    else:
        row_scale = np.repeat(noise_scale, chronological_age.size)
    return signal + rng.normal(size=signal.shape) * row_scale[:, None]


def generate_scenario(scenario: str, *, seed: int) -> SimulatedSplit:
    """Generate one training/test split without using test statistics in training."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    n_train = 40 if scenario == "small_calibration_sample" else 320
    n_test = 500
    if scenario == "range_shift":
        train_age = _ages(rng, n_train, low=25.0, high=65.0)
        test_age = _ages(rng, n_test, low=55.0, high=90.0)
    else:
        imbalanced = scenario == "age_imbalance"
        train_age = _ages(rng, n_train, low=25.0, high=85.0, imbalanced=imbalanced)
        test_age = _ages(rng, n_test, low=25.0, high=85.0, imbalanced=imbalanced)

    if scenario == "unbiased_predictions":
        train_gap = np.zeros(n_train)
        test_gap = np.zeros(n_test)
        feature_noise = 0.002
    else:
        train_gap = rng.normal(0.0, 5.0, size=n_train)
        test_gap = rng.normal(0.0, 5.0, size=n_test)
        feature_noise = 1.35 if scenario == "regression_to_mean" else 0.65

    train_latent = train_age + train_gap
    test_latent = test_age + test_gap

    def make_outcome(age: np.ndarray, gap: np.ndarray) -> np.ndarray:
        age_z = (age - 55.0) / 15.0
        gap_z = gap / 5.0
        noise = rng.normal(size=age.size)
        if scenario == "null_outcome":
            return noise
        if scenario == "latent_signal":
            return 0.9 * gap_z + 0.6 * noise
        if scenario == "age_confounding":
            return 1.1 * age_z + 0.45 * gap_z + 0.55 * noise
        return 0.45 * gap_z + 0.9 * noise

    train_outcome = make_outcome(train_age, train_gap)
    test_outcome = make_outcome(test_age, test_gap)
    heteroscedastic = scenario == "heteroscedastic_errors"
    nonlinear = scenario == "nonlinear_bias"
    train_predictors = _make_predictors(
        rng,
        train_latent,
        train_age,
        noise_scale=feature_noise,
        heteroscedastic=heteroscedastic,
        nonlinear=nonlinear,
    )
    test_predictors = _make_predictors(
        rng,
        test_latent,
        test_age,
        noise_scale=feature_noise,
        heteroscedastic=heteroscedastic,
        nonlinear=nonlinear,
    )

    # Standardization parameters are estimated on training predictors only.
    center = np.mean(train_predictors, axis=0)
    scale = np.std(train_predictors, axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    train_predictors = (train_predictors - center) / scale
    test_predictors = (test_predictors - center) / scale

    return SimulatedSplit(
        train_predictors=train_predictors,
        train_age=train_age,
        train_latent_age=train_latent,
        train_outcome=train_outcome,
        test_predictors=test_predictors,
        test_age=test_age,
        test_latent_age=test_latent,
        test_outcome=test_outcome,
    )


def _fit_and_predict(split: SimulatedSplit, *, lambda_: float) -> dict[str, np.ndarray]:
    raw_model = StandardLasso(lambda_=lambda_).fit(split.train_predictors, split.train_age)
    raw_train = raw_model.predict(split.train_predictors)
    raw_test = raw_model.predict(split.test_predictors)

    residual = ResidualCorrection().fit(split.train_age, raw_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residual_test = residual.transform(split.test_age, raw_test)

    recalibration = LinearRecalibration().fit(split.train_age, raw_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        recalibrated_test = recalibration.transform(raw_test)

    umlr = ConstrainedLassoOracle(lambda_=lambda_).fit(
        split.train_predictors, split.train_age
    )
    umlr_test = umlr.predict(split.test_predictors)
    return {
        "raw_lasso": raw_test,
        "residual_correction": residual_test,
        "linear_recalibration": recalibrated_test,
        "umlr_oracle": umlr_test,
    }


def run_replication(
    scenario: str,
    *,
    seed: int,
    replication: int,
    lambda_: float = 0.02,
) -> list[dict[str, float | int | str | bool]]:
    split = generate_scenario(scenario, seed=seed)
    predictions = _fit_and_predict(split, lambda_=lambda_)
    latent_gap = split.test_latent_age - split.test_age
    if scenario == "null_outcome":
        target_coefficient = 0.0
    else:
        target_coefficient = downstream_association(
            latent_gap,
            split.test_outcome,
            chronological_age=split.test_age,
        ).coefficient

    records: list[dict[str, float | int | str | bool]] = []
    for method, prediction in predictions.items():
        metrics = diagnostic_metrics(
            split.test_age,
            prediction,
            outcome=split.test_outcome,
            latent_biological_age=split.test_latent_age,
        )
        metrics.update(
            {
                "scenario": scenario,
                "method": method,
                "replication": replication,
                "seed": seed,
                "association_target": target_coefficient,
                "association_error_age_adjusted": (
                    metrics["outcome_beta_age_adjusted"] - target_coefficient
                ),
                "association_detected_age_adjusted": (
                    metrics["outcome_p_age_adjusted"] < 0.05
                ),
                "latent_target_in_ci": (
                    metrics["outcome_ci_low_age_adjusted"]
                    <= target_coefficient
                    <= metrics["outcome_ci_high_age_adjusted"]
                ),
            }
        )
        records.append(metrics)
    return records


def summarize_benchmark(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby(["scenario", "method"], sort=False)
    summary = grouped.agg(
        replications=("replication", "nunique"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        latent_rmse_mean=("latent_rmse", "mean"),
        calibration_in_the_large_mean=("calibration_in_the_large", "mean"),
        calibration_slope_mean=("calibration_slope", "mean"),
        residual_age_slope_mean=("residual_age_slope", "mean"),
        residual_age_correlation_mean=("residual_age_correlation", "mean"),
        outcome_beta_age_adjusted_mean=("outcome_beta_age_adjusted", "mean"),
        association_target_mean=("association_target", "mean"),
        association_error_age_adjusted_mean=("association_error_age_adjusted", "mean"),
        association_error_age_adjusted_abs_mean=(
            "association_error_age_adjusted",
            lambda values: float(np.mean(np.abs(values))),
        ),
        detection_rate_age_adjusted=("association_detected_age_adjusted", "mean"),
        latent_target_interval_inclusion_rate=("latent_target_in_ci", "mean"),
    ).reset_index()

    raw_rmse = (
        summary.loc[summary["method"] == "raw_lasso", ["scenario", "rmse_mean"]]
        .rename(columns={"rmse_mean": "raw_rmse_mean"})
        .set_index("scenario")
    )
    summary = summary.join(raw_rmse, on="scenario")
    summary["rmse_ratio_vs_raw"] = summary["rmse_mean"] / summary["raw_rmse_mean"]
    return summary


def run_benchmark(
    *,
    replications: int = 30,
    base_seed: int = 20260729,
    lambda_: float = 0.02,
    output_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replications < 1:
        raise ValueError("replications must be positive")
    records: list[dict[str, float | int | str | bool]] = []
    seed_sequence = np.random.SeedSequence(base_seed)
    child_sequences = seed_sequence.spawn(len(SCENARIOS) * replications)
    child_index = 0
    for scenario in SCENARIOS:
        for replication in range(replications):
            seed = int(child_sequences[child_index].generate_state(1)[0])
            child_index += 1
            records.extend(
                run_replication(
                    scenario,
                    seed=seed,
                    replication=replication,
                    lambda_=lambda_,
                )
            )
    records_frame = pd.DataFrame.from_records(records)
    summary_frame = summarize_benchmark(records_frame)
    if output_directory is not None:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        records_frame.to_csv(output / "simulation_replications.csv", index=False)
        summary_frame.to_csv(output / "simulation_summary.csv", index=False)
    return records_frame, summary_frame
