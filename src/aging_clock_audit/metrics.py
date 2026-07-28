"""Diagnostics used by the simulation and real-data audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _as_finite_vector(values: object, *, name: str, min_size: int = 3) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    array = np.ravel(array)
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} observations")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _same_length(*arrays: np.ndarray) -> None:
    sizes = {array.size for array in arrays}
    if len(sizes) != 1:
        raise ValueError("all inputs must have the same number of observations")


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    centered_x = x - np.mean(x)
    denominator = float(centered_x @ centered_x)
    if denominator <= np.finfo(float).eps:
        raise ValueError("slope is undefined for a constant predictor")
    return float(centered_x @ (y - np.mean(y)) / denominator)


@dataclass(frozen=True)
class Association:
    coefficient: float
    standard_error: float
    p_value: float
    ci_low: float
    ci_high: float
    degrees_of_freedom: int


def downstream_association(
    gap: object,
    outcome: object,
    *,
    chronological_age: object | None = None,
    confidence: float = 0.95,
) -> Association:
    """Regress an age gap on an outcome, optionally adjusting for age."""

    gap_array = _as_finite_vector(gap, name="gap")
    outcome_array = _as_finite_vector(outcome, name="outcome")
    arrays = [gap_array, outcome_array]
    columns = [np.ones(gap_array.size), outcome_array]
    if chronological_age is not None:
        age_array = _as_finite_vector(chronological_age, name="chronological_age")
        arrays.append(age_array)
        columns.append(age_array)
    _same_length(*arrays)

    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("association design matrix is rank deficient")

    coefficients, _, _, _ = np.linalg.lstsq(design, gap_array, rcond=None)
    residuals = gap_array - design @ coefficients
    degrees_of_freedom = gap_array.size - design.shape[1]
    if degrees_of_freedom <= 0:
        raise ValueError("not enough observations for association inference")
    sigma2 = float(residuals @ residuals / degrees_of_freedom)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    coefficient = float(coefficients[1])
    if standard_error == 0.0:
        p_value = 0.0 if coefficient != 0.0 else 1.0
    else:
        statistic = coefficient / standard_error
        p_value = float(2 * stats.t.sf(abs(statistic), degrees_of_freedom))
    alpha = 1.0 - confidence
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    return Association(
        coefficient=coefficient,
        standard_error=standard_error,
        p_value=p_value,
        ci_low=coefficient - critical * standard_error,
        ci_high=coefficient + critical * standard_error,
        degrees_of_freedom=degrees_of_freedom,
    )


def diagnostic_metrics(
    chronological_age: object,
    predicted_age: object,
    *,
    outcome: object | None = None,
    latent_biological_age: object | None = None,
) -> dict[str, float]:
    """Compute the diagnostic metrics required by the initiative brief."""

    age = _as_finite_vector(chronological_age, name="chronological_age")
    prediction = _as_finite_vector(predicted_age, name="predicted_age")
    _same_length(age, prediction)
    if np.ptp(age) <= np.finfo(float).eps:
        raise ValueError("chronological_age must not be constant")

    error = prediction - age
    calibration_slope = _slope(age, prediction)
    if np.ptp(error) <= np.finfo(float).eps:
        residual_age_correlation = 0.0
    else:
        residual_age_correlation = float(np.corrcoef(error, age)[0, 1])
    metrics = {
        "n": float(age.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "calibration_in_the_large": float(np.mean(error)),
        "calibration_slope": calibration_slope,
        "residual_age_slope": calibration_slope - 1.0,
        "residual_age_correlation": residual_age_correlation,
    }

    if latent_biological_age is not None:
        latent = _as_finite_vector(latent_biological_age, name="latent_biological_age")
        _same_length(age, latent)
        latent_error = prediction - latent
        metrics["latent_mae"] = float(np.mean(np.abs(latent_error)))
        metrics["latent_rmse"] = float(np.sqrt(np.mean(latent_error**2)))

    if outcome is not None:
        outcome_array = _as_finite_vector(outcome, name="outcome")
        _same_length(age, outcome_array)
        unadjusted = downstream_association(error, outcome_array)
        adjusted = downstream_association(
            error, outcome_array, chronological_age=age
        )
        metrics.update(
            {
                "outcome_beta_unadjusted": unadjusted.coefficient,
                "outcome_p_unadjusted": unadjusted.p_value,
                "outcome_beta_age_adjusted": adjusted.coefficient,
                "outcome_se_age_adjusted": adjusted.standard_error,
                "outcome_p_age_adjusted": adjusted.p_value,
                "outcome_ci_low_age_adjusted": adjusted.ci_low,
                "outcome_ci_high_age_adjusted": adjusted.ci_high,
            }
        )

    return metrics
