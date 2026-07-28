"""Audit-only estimators and post-hoc baselines.

The constrained estimator is an independent numerical oracle for testing the
authors' equations.  It is deliberately named ``Oracle`` and is not presented as
a maintained replacement for the official R implementation.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def _finite_matrix(values: object, *, name: str, min_rows: int = 3) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] < min_rows or matrix.shape[1] < 1:
        raise ValueError(
            f"{name} must be a finite 2D matrix with at least {min_rows} rows"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _finite_vector(
    values: object,
    *,
    name: str,
    expected: int | None = None,
    min_size: int = 3,
) -> np.ndarray:
    vector = np.ravel(np.asarray(values, dtype=float))
    if vector.size < min_size or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain at least {min_size} finite values")
    if expected is not None and vector.size != expected:
        raise ValueError(f"{name} must have {expected} observations")
    return vector


def _prediction_matrix(values: object, *, name: str, n_features: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    elif matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1) if n_features == 1 else matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != n_features:
        raise ValueError(f"{name} must have {n_features} columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FitMetadata:
    n_fit: int
    n_features: int
    age_min: float
    age_max: float
    cutoff: float | None
    lambda_: float
    constraint_error: float
    optimizer_iterations: int
    training_fingerprint: str


@dataclass
class ConstrainedLassoOracle:
    """Solve the authors' two-anchor constrained Lasso objective."""

    lambda_: float = 0.01
    cutoff: float | None = None
    max_iterations: int = 4_000
    tolerance: float = 1e-10
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    metadata_: FitMetadata | None = None

    def fit(self, predictors: object, chronological_age: object) -> ConstrainedLassoOracle:
        matrix = _finite_matrix(predictors, name="predictors")
        age = _finite_vector(
            chronological_age, name="chronological_age", expected=matrix.shape[0]
        )
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        if np.ptp(age) <= np.finfo(float).eps:
            raise ValueError("chronological_age must not be constant")

        design = np.column_stack([np.ones(matrix.shape[0]), matrix])
        cutoff = float(np.mean(age) if self.cutoff is None else self.cutoff)
        lower = age <= cutoff
        upper = age > cutoff
        if not np.any(lower) or not np.any(upper):
            raise ValueError("cutoff must create two non-empty age regions")

        constraints = np.vstack([np.mean(design[lower], axis=0), np.mean(design[upper], axis=0)])
        targets = np.array([np.mean(age[lower]), np.mean(age[upper])])
        expanded_constraints = np.hstack([constraints, -constraints])
        linear_constraint = LinearConstraint(expanded_constraints, targets, targets)

        penalty_mask = np.ones(design.shape[1])
        penalty_mask[0] = 0.0
        expanded_penalty = np.concatenate([penalty_mask, penalty_mask])
        n = design.shape[0]

        def objective(split_coefficients: np.ndarray) -> float:
            coefficients = split_coefficients[: design.shape[1]] - split_coefficients[
                design.shape[1] :
            ]
            residual = design @ coefficients - age
            return float(
                0.5 * (residual @ residual) / n
                + self.lambda_ * (expanded_penalty @ split_coefficients)
            )

        def gradient(split_coefficients: np.ndarray) -> np.ndarray:
            coefficients = split_coefficients[: design.shape[1]] - split_coefficients[
                design.shape[1] :
            ]
            base = design.T @ (design @ coefficients - age) / n
            return np.concatenate([base, -base]) + self.lambda_ * expanded_penalty

        initial_coefficients, _, _, _ = np.linalg.lstsq(design, age, rcond=None)
        initial = np.concatenate(
            [np.maximum(initial_coefficients, 0.0), np.maximum(-initial_coefficients, 0.0)]
        )
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(0.0, np.inf),
            constraints=[linear_constraint],
            options={"ftol": self.tolerance, "maxiter": self.max_iterations, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"constrained optimization failed: {result.message}")

        coefficients = result.x[: design.shape[1]] - result.x[design.shape[1] :]
        constraint_error = float(np.max(np.abs(constraints @ coefficients - targets)))
        if constraint_error > 1e-6:
            raise RuntimeError(
                f"solution violates calibration constraints by {constraint_error:.3g}"
            )
        self.intercept_ = float(coefficients[0])
        self.coef_ = np.asarray(coefficients[1:])
        self.metadata_ = FitMetadata(
            n_fit=n,
            n_features=matrix.shape[1],
            age_min=float(np.min(age)),
            age_max=float(np.max(age)),
            cutoff=cutoff,
            lambda_=float(self.lambda_),
            constraint_error=constraint_error,
            optimizer_iterations=int(result.nit),
            training_fingerprint=_fingerprint(matrix, age),
        )
        return self

    def predict(self, predictors: object) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        matrix = _prediction_matrix(
            predictors, name="predictors", n_features=self.coef_.size
        )
        return self.intercept_ + matrix @ self.coef_


@dataclass
class StandardLasso:
    """Unconstrained Lasso baseline using the same numerical objective."""

    lambda_: float = 0.01
    max_iterations: int = 4_000
    tolerance: float = 1e-10
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    metadata_: FitMetadata | None = None

    def fit(self, predictors: object, chronological_age: object) -> StandardLasso:
        matrix = _finite_matrix(predictors, name="predictors")
        age = _finite_vector(
            chronological_age, name="chronological_age", expected=matrix.shape[0]
        )
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")

        design = np.column_stack([np.ones(matrix.shape[0]), matrix])
        penalty_mask = np.ones(design.shape[1])
        penalty_mask[0] = 0.0
        expanded_penalty = np.concatenate([penalty_mask, penalty_mask])
        n = design.shape[0]

        def objective(split_coefficients: np.ndarray) -> float:
            coefficients = split_coefficients[: design.shape[1]] - split_coefficients[
                design.shape[1] :
            ]
            residual = design @ coefficients - age
            return float(
                0.5 * (residual @ residual) / n
                + self.lambda_ * (expanded_penalty @ split_coefficients)
            )

        def gradient(split_coefficients: np.ndarray) -> np.ndarray:
            coefficients = split_coefficients[: design.shape[1]] - split_coefficients[
                design.shape[1] :
            ]
            base = design.T @ (design @ coefficients - age) / n
            return np.concatenate([base, -base]) + self.lambda_ * expanded_penalty

        initial_coefficients, _, _, _ = np.linalg.lstsq(design, age, rcond=None)
        initial = np.concatenate(
            [np.maximum(initial_coefficients, 0.0), np.maximum(-initial_coefficients, 0.0)]
        )
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            bounds=Bounds(0.0, np.inf),
            options={"ftol": self.tolerance, "maxiter": self.max_iterations},
        )
        if not result.success:
            raise RuntimeError(f"unconstrained optimization failed: {result.message}")
        coefficients = result.x[: design.shape[1]] - result.x[design.shape[1] :]
        self.intercept_ = float(coefficients[0])
        self.coef_ = np.asarray(coefficients[1:])
        self.metadata_ = FitMetadata(
            n_fit=n,
            n_features=matrix.shape[1],
            age_min=float(np.min(age)),
            age_max=float(np.max(age)),
            cutoff=None,
            lambda_=float(self.lambda_),
            constraint_error=0.0,
            optimizer_iterations=int(result.nit),
            training_fingerprint=_fingerprint(matrix, age),
        )
        return self

    def predict(self, predictors: object) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        matrix = _prediction_matrix(
            predictors, name="predictors", n_features=self.coef_.size
        )
        return self.intercept_ + matrix @ self.coef_


@dataclass
class ResidualCorrection:
    """Common post-hoc residual correction baseline, fit on training data only."""

    intercept_: float | None = None
    slope_: float | None = None
    age_min_: float | None = None
    age_max_: float | None = None
    training_fingerprint_: str | None = None

    def fit(self, chronological_age: object, predicted_age: object) -> ResidualCorrection:
        age = _finite_vector(
            chronological_age, name="chronological_age", min_size=1
        )
        prediction = _finite_vector(
            predicted_age, name="predicted_age", expected=age.size, min_size=1
        )
        design = np.column_stack([np.ones(age.size), age])
        if np.linalg.matrix_rank(design) < 2:
            raise ValueError("chronological_age must not be constant")
        coefficients, _, _, _ = np.linalg.lstsq(design, prediction - age, rcond=None)
        self.intercept_, self.slope_ = map(float, coefficients)
        self.age_min_, self.age_max_ = float(np.min(age)), float(np.max(age))
        self.training_fingerprint_ = _fingerprint(age, prediction)
        return self

    def transform(self, chronological_age: object, predicted_age: object) -> np.ndarray:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("fit must be called before transform")
        age = _finite_vector(chronological_age, name="chronological_age")
        prediction = _finite_vector(
            predicted_age, name="predicted_age", expected=age.size
        )
        assert self.age_min_ is not None and self.age_max_ is not None
        if np.min(age) < self.age_min_ or np.max(age) > self.age_max_:
            warnings.warn(
                "chronological ages extend beyond the correction fit range",
                RuntimeWarning,
                stacklevel=2,
            )
        return prediction - (self.intercept_ + self.slope_ * age)


@dataclass
class LinearRecalibration:
    """Map raw predictions back to chronological age using training data only."""

    intercept_: float | None = None
    slope_: float | None = None
    prediction_min_: float | None = None
    prediction_max_: float | None = None
    training_fingerprint_: str | None = None

    def fit(self, chronological_age: object, predicted_age: object) -> LinearRecalibration:
        age = _finite_vector(chronological_age, name="chronological_age")
        prediction = _finite_vector(
            predicted_age, name="predicted_age", expected=age.size
        )
        design = np.column_stack([np.ones(prediction.size), prediction])
        if np.linalg.matrix_rank(design) < 2:
            raise ValueError("predicted_age must not be constant")
        coefficients, _, _, _ = np.linalg.lstsq(design, age, rcond=None)
        self.intercept_, self.slope_ = map(float, coefficients)
        self.prediction_min_, self.prediction_max_ = (
            float(np.min(prediction)),
            float(np.max(prediction)),
        )
        self.training_fingerprint_ = _fingerprint(age, prediction)
        return self

    def transform(self, predicted_age: object) -> np.ndarray:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("fit must be called before transform")
        prediction = _finite_vector(predicted_age, name="predicted_age", min_size=1)
        assert self.prediction_min_ is not None and self.prediction_max_ is not None
        if (
            np.min(prediction) < self.prediction_min_
            or np.max(prediction) > self.prediction_max_
        ):
            warnings.warn(
                "raw predictions extend beyond the recalibration fit range",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.intercept_ + self.slope_ * prediction
