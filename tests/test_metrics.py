from __future__ import annotations

import numpy as np
import pytest

from aging_clock_audit.metrics import diagnostic_metrics, downstream_association


def test_diagnostics_for_perfect_predictions() -> None:
    age = np.linspace(20.0, 80.0, 31)
    metrics = diagnostic_metrics(age, age)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["calibration_in_the_large"] == pytest.approx(0.0)
    assert metrics["calibration_slope"] == pytest.approx(1.0)
    assert metrics["residual_age_slope"] == pytest.approx(0.0)


def test_diagnostics_detect_regression_to_mean() -> None:
    age = np.linspace(20.0, 80.0, 31)
    prediction = np.mean(age) + 0.6 * (age - np.mean(age))
    metrics = diagnostic_metrics(age, prediction)

    assert metrics["calibration_slope"] == pytest.approx(0.6)
    assert metrics["residual_age_slope"] == pytest.approx(-0.4)
    assert metrics["residual_age_correlation"] == pytest.approx(-1.0)


def test_age_adjustment_recovers_outcome_coefficient() -> None:
    rng = np.random.default_rng(9)
    age = rng.uniform(40.0, 80.0, size=500)
    outcome = 0.08 * age + rng.normal(size=500)
    gap = 2.5 * outcome - 0.5 * age + rng.normal(scale=0.2, size=500)

    adjusted = downstream_association(gap, outcome, chronological_age=age)

    assert adjusted.coefficient == pytest.approx(2.5, abs=0.05)
    assert adjusted.ci_low < 2.5 < adjusted.ci_high
    assert adjusted.p_value < 1e-20


def test_metrics_reject_constant_age() -> None:
    with pytest.raises(ValueError, match="must not be constant"):
        diagnostic_metrics(np.repeat(50.0, 10), np.linspace(40.0, 60.0, 10))
