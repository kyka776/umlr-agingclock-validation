from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aging_clock_audit.simulation import (
    METHODS,
    SCENARIOS,
    generate_scenario,
    run_benchmark,
    run_replication,
)


def test_all_predefined_scenarios_generate_finite_splits() -> None:
    for index, scenario in enumerate(SCENARIOS):
        split = generate_scenario(scenario, seed=100 + index)
        assert split.train_predictors.shape[1] == 6
        assert split.test_predictors.shape[1] == 6
        assert np.all(np.isfinite(split.train_predictors))
        assert np.all(np.isfinite(split.test_predictors))


def test_replication_returns_every_method() -> None:
    records = run_replication(
        "regression_to_mean", seed=34, replication=0, lambda_=0.02
    )

    assert {record["method"] for record in records} == set(METHODS)
    assert all(record["scenario"] == "regression_to_mean" for record in records)


def test_small_benchmark_is_deterministic(tmp_path) -> None:
    first_records, first_summary = run_benchmark(
        replications=1, base_seed=88, output_directory=tmp_path / "first"
    )
    second_records, second_summary = run_benchmark(
        replications=1, base_seed=88, output_directory=tmp_path / "second"
    )

    pd.testing.assert_frame_equal(first_records, second_records)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert (tmp_path / "first" / "simulation_replications.csv").is_file()
    assert (tmp_path / "first" / "simulation_summary.csv").is_file()


def test_residual_correction_is_fitted_without_test_predictions() -> None:
    first = generate_scenario("range_shift", seed=9)
    second = generate_scenario("range_shift", seed=9)
    second.test_predictors[:] = 1_000.0

    # The generated training split remains byte-identical even when test values change.
    assert first.train_predictors == pytest.approx(second.train_predictors)
    assert first.train_age == pytest.approx(second.train_age)


def test_expected_simulation_properties_are_recovered() -> None:
    _, summary = run_benchmark(replications=8, base_seed=1337)
    indexed = summary.set_index(["scenario", "method"])

    raw_rtm = indexed.loc[("regression_to_mean", "raw_lasso")]
    umlr_rtm = indexed.loc[("regression_to_mean", "umlr_oracle")]
    raw_signal = indexed.loc[("latent_signal", "raw_lasso")]
    umlr_signal = indexed.loc[("latent_signal", "umlr_oracle")]

    assert raw_rtm["residual_age_slope_mean"] < -0.25
    assert abs(umlr_rtm["residual_age_slope_mean"]) < 0.05
    assert (
        umlr_signal["association_error_age_adjusted_abs_mean"]
        < raw_signal["association_error_age_adjusted_abs_mean"]
    )
    assert (
        indexed.loc[("null_outcome", "umlr_oracle"), "detection_rate_age_adjusted"]
        <= 0.125
    )
