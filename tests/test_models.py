from __future__ import annotations

import pickle

import numpy as np
import pytest

from aging_clock_audit.models import (
    ConstrainedLassoOracle,
    LinearRecalibration,
    ResidualCorrection,
    StandardLasso,
)


def test_manual_exact_linear_example_recovers_coefficients() -> None:
    predictor = np.arange(1.0, 9.0).reshape(-1, 1)
    age = 2.0 + 3.0 * predictor[:, 0]

    model = ConstrainedLassoOracle(lambda_=0.0).fit(predictor, age)

    assert model.intercept_ == pytest.approx(2.0, abs=1e-7)
    assert model.coef_ == pytest.approx([3.0], abs=1e-7)
    assert model.predict(predictor) == pytest.approx(age, abs=1e-7)
    assert model.metadata_ is not None
    assert model.metadata_.constraint_error < 1e-8


def test_two_region_constraints_match_authors_equations() -> None:
    predictor = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    age = np.array([35.0, 39.0, 44.0, 50.0, 58.0, 67.0, 79.0])
    model = ConstrainedLassoOracle(lambda_=0.02).fit(predictor, age)
    prediction = model.predict(predictor)
    cutoff = float(np.mean(age))

    assert np.mean(prediction[age <= cutoff]) == pytest.approx(
        np.mean(age[age <= cutoff]), abs=1e-6
    )
    assert np.mean(prediction[age > cutoff]) == pytest.approx(
        np.mean(age[age > cutoff]), abs=1e-6
    )


def test_train_test_separation() -> None:
    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(80, 3))
    train_y = 55 + train_x @ np.array([4.0, -2.0, 1.0]) + rng.normal(size=80)
    model = ConstrainedLassoOracle(lambda_=0.01).fit(train_x, train_y)
    fingerprint_before = model.metadata_.training_fingerprint
    coefficients_before = model.coef_.copy()

    model.predict(rng.normal(size=(20, 3)))
    model.predict(rng.normal(loc=20, size=(50, 3)))

    assert model.metadata_.training_fingerprint == fingerprint_before
    assert model.coef_ == pytest.approx(coefficients_before)


def test_deterministic_output() -> None:
    rng = np.random.default_rng(12)
    predictors = rng.normal(size=(100, 4))
    age = 50 + predictors @ np.array([3.0, -2.0, 0.5, 1.0]) + rng.normal(size=100)

    first = ConstrainedLassoOracle(lambda_=0.03).fit(predictors, age)
    second = ConstrainedLassoOracle(lambda_=0.03).fit(predictors, age)

    assert first.coef_ == pytest.approx(second.coef_, abs=1e-9)
    assert first.intercept_ == pytest.approx(second.intercept_, abs=1e-9)
    assert first.predict(predictors) == pytest.approx(second.predict(predictors), abs=1e-9)


@pytest.mark.parametrize(
    ("predictors", "age"),
    [
        ([[1.0], [2.0], [np.nan]], [40.0, 50.0, 60.0]),
        ([[1.0], [2.0], [3.0]], [40.0, np.inf, 60.0]),
        ([[1.0], [2.0]], [40.0, 50.0]),
    ],
)
def test_invalid_or_missing_inputs_raise(predictors: object, age: object) -> None:
    with pytest.raises(ValueError):
        ConstrainedLassoOracle().fit(predictors, age)


def test_constant_age_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be constant"):
        ConstrainedLassoOracle().fit(np.arange(6.0).reshape(-1, 1), np.repeat(50.0, 6))


def test_constant_predictions_are_rejected_by_linear_recalibration() -> None:
    with pytest.raises(ValueError, match="must not be constant"):
        LinearRecalibration().fit(np.arange(40.0, 46.0), np.repeat(50.0, 6))


def test_extrapolation_warnings() -> None:
    age = np.arange(40.0, 60.0)
    prediction = 45.0 + 0.8 * (age - 45.0)
    residual = ResidualCorrection().fit(age, prediction)
    recalibration = LinearRecalibration().fit(age, prediction)

    with pytest.warns(RuntimeWarning, match="beyond"):
        residual.transform([30.0, 40.0, 70.0], [35.0, 45.0, 65.0])
    with pytest.warns(RuntimeWarning, match="beyond"):
        recalibration.transform([20.0, 50.0, 90.0])


def test_serialization_round_trip() -> None:
    predictor = np.arange(12.0).reshape(-1, 1)
    age = 40.0 + 2.0 * predictor[:, 0]
    model = ConstrainedLassoOracle(lambda_=0.0).fit(predictor, age)

    restored = pickle.loads(pickle.dumps(model))

    assert restored.metadata_ == model.metadata_
    assert restored.predict(predictor) == pytest.approx(model.predict(predictor))


def test_vectorized_and_scalar_predictions_agree() -> None:
    predictors = np.column_stack([np.arange(10.0), np.arange(10.0) ** 2])
    age = 40.0 + predictors @ np.array([2.0, 0.1])
    model = ConstrainedLassoOracle(lambda_=0.0).fit(predictors, age)
    vectorized = model.predict(predictors)
    scalar = np.array([model.predict(row)[0] for row in predictors])

    assert scalar == pytest.approx(vectorized)


def test_standard_lasso_and_oracle_agree_when_constraint_is_inactive() -> None:
    predictor = np.arange(1.0, 20.0).reshape(-1, 1)
    age = 5.0 + 2.0 * predictor[:, 0]
    raw = StandardLasso(lambda_=0.0).fit(predictor, age)
    constrained = ConstrainedLassoOracle(lambda_=0.0).fit(predictor, age)

    assert raw.predict(predictor) == pytest.approx(
        constrained.predict(predictor), abs=1e-6
    )
