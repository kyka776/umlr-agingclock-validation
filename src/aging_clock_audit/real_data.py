"""Pinned, aggregate-only real-data demonstration.

Raw methylation values and sample-level predictions never leave ``data/raw``.
The public outputs contain only method-level metrics and age-bin means.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

from .metrics import diagnostic_metrics
from .models import (
    ConstrainedLassoOracle,
    LinearRecalibration,
    ResidualCorrection,
    StandardLasso,
)

OMNIAGE_LUNG_SHA256 = (
    "5a96d2f9f8f2807e220f9d699fc6f5364a80a49ccc946b0950357d13e36f2beb"
)


@dataclass(frozen=True)
class RealDataResult:
    metrics: pd.DataFrame
    age_bins: pd.DataFrame
    provenance: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratified_split(
    ages: np.ndarray, groups: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a frozen train/test split stratified by group and age tertile."""

    age_bins = np.digitize(ages, np.quantile(ages, [1 / 3, 2 / 3]), right=True)
    rng = np.random.default_rng(seed)
    test_indices: list[int] = []
    all_indices = np.arange(ages.size)
    for group in np.unique(groups):
        for age_bin in range(3):
            stratum = all_indices[(groups == group) & (age_bins == age_bin)]
            if stratum.size == 0:
                continue
            shuffled = rng.permutation(stratum)
            test_count = max(1, int(round(0.30 * stratum.size)))
            if test_count >= stratum.size:
                test_count = stratum.size - 1
            test_indices.extend(shuffled[:test_count].tolist())
    test = np.array(sorted(test_indices), dtype=int)
    train = np.setdiff1d(all_indices, test)
    if train.size < 10 or test.size < 10:
        raise RuntimeError("stratified split is unexpectedly small")
    return train, test


def _select_and_scale(
    matrix: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_raw = matrix[train_indices]
    variances = np.var(train_raw, axis=0, ddof=1)
    if not np.all(np.isfinite(variances)):
        raise ValueError("methylation matrix contains missing or non-finite values")
    selected = np.argsort(variances, kind="stable")[-feature_count:]
    center = np.mean(train_raw[:, selected], axis=0)
    scale = np.std(train_raw[:, selected], axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    return (
        (train_raw[:, selected] - center) / scale,
        (matrix[test_indices][:, selected] - center) / scale,
        selected,
    )


def run_omniage_example(
    path: str | Path,
    *,
    seed: int = 20260729,
    feature_count: int = 24,
    lambda_: float = 0.02,
) -> RealDataResult:
    """Run one frozen holdout analysis on the OmniAge lung tutorial dataset."""

    source = Path(path)
    actual_hash = _sha256(source)
    if actual_hash != OMNIAGE_LUNG_SHA256:
        raise ValueError(
            "OmniAge input checksum mismatch: "
            f"expected {OMNIAGE_LUNG_SHA256}, got {actual_hash}"
        )
    payload = rdata.read_rds(source)
    if not isinstance(payload, dict) or set(payload) != {"bmiq_m", "PhenoTypes"}:
        raise ValueError("unexpected OmniAge RDS structure")
    beta = np.asarray(payload["bmiq_m"], dtype=float).T
    phenotype = payload["PhenoTypes"].reset_index(drop=True)
    ages = phenotype["Age"].to_numpy(dtype=float)
    outcomes = phenotype["num"].to_numpy(dtype=float)
    groups = phenotype["Group"].astype(str).to_numpy()
    if beta.shape != (56, 2974) or ages.shape != (56,):
        raise ValueError(f"unexpected OmniAge data shape: {beta.shape}, {ages.shape}")

    train_indices, test_indices = _stratified_split(ages, groups, seed=seed)
    train_x, test_x, selected = _select_and_scale(
        beta,
        train_indices,
        test_indices,
        feature_count=feature_count,
    )
    train_age, test_age = ages[train_indices], ages[test_indices]
    test_outcome = outcomes[test_indices]

    raw = StandardLasso(lambda_=lambda_).fit(train_x, train_age)
    raw_train = raw.predict(train_x)
    raw_test = raw.predict(test_x)
    residual = ResidualCorrection().fit(train_age, raw_train)
    recalibration = LinearRecalibration().fit(train_age, raw_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residual_test = residual.transform(test_age, raw_test)
        recalibrated_test = recalibration.transform(raw_test)
    oracle = ConstrainedLassoOracle(lambda_=lambda_).fit(train_x, train_age)
    predictions = {
        "raw_lasso": raw_test,
        "residual_correction": residual_test,
        "linear_recalibration": recalibrated_test,
        "umlr_oracle": oracle.predict(test_x),
    }

    metric_rows: list[dict[str, float | int | str]] = []
    age_bin_rows: list[dict[str, float | int | str]] = []
    test_age_bins = np.digitize(
        test_age, np.quantile(train_age, [1 / 3, 2 / 3]), right=True
    )
    for method, prediction in predictions.items():
        row: dict[str, float | int | str] = {"method": method}
        row.update(diagnostic_metrics(test_age, prediction, outcome=test_outcome))
        metric_rows.append(row)
        for age_bin in range(3):
            mask = test_age_bins == age_bin
            age_bin_rows.append(
                {
                    "method": method,
                    "age_bin": age_bin + 1,
                    "n": int(np.sum(mask)),
                    "mean_chronological_age": float(np.mean(test_age[mask])),
                    "mean_predicted_age": float(np.mean(prediction[mask])),
                }
            )

    return RealDataResult(
        metrics=pd.DataFrame(metric_rows),
        age_bins=pd.DataFrame(age_bin_rows),
        provenance={
            "source_sha256": actual_hash,
            "source_shape_features_by_samples": [2974, 56],
            "train_n": int(train_indices.size),
            "test_n": int(test_indices.size),
            "selected_features": int(selected.size),
            "selection": "top training-only variance; stable tie breaking",
            "split_seed": seed,
            "split": "group x training-global-age-tertile stratified 70/30 holdout",
            "lambda": lambda_,
            "raw_or_sample_level_output_committed": False,
        },
    )
