from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aging_clock_audit.real_data import (
    _select_and_scale,
    _stratified_split,
    run_omniage_example,
)


def test_stratified_split_is_deterministic_and_disjoint() -> None:
    ages = np.linspace(40, 90, 60)
    groups = np.repeat(["a", "b", "c"], 20)
    first_train, first_test = _stratified_split(ages, groups, seed=17)
    second_train, second_test = _stratified_split(ages, groups, seed=17)
    np.testing.assert_array_equal(first_train, second_train)
    np.testing.assert_array_equal(first_test, second_test)
    assert np.intersect1d(first_train, first_test).size == 0
    assert np.union1d(first_train, first_test).size == ages.size


def test_feature_preprocessing_uses_training_statistics_only() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(24, 12))
    train = np.arange(16)
    test = np.arange(16, 24)
    train_scaled, test_scaled, selected = _select_and_scale(
        matrix, train, test, feature_count=5
    )
    changed = matrix.copy()
    changed[test] += 1_000
    changed_train, changed_test, changed_selected = _select_and_scale(
        changed, train, test, feature_count=5
    )
    np.testing.assert_allclose(train_scaled, changed_train)
    np.testing.assert_array_equal(selected, changed_selected)
    assert not np.allclose(test_scaled, changed_test)


def test_real_data_rejects_unpinned_input(tmp_path: Path) -> None:
    source = tmp_path / "unknown.rds"
    source.write_bytes(b"not the pinned data")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_omniage_example(source)
