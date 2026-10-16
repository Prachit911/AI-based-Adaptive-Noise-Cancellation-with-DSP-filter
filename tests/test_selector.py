"""Unit tests for the SupervisedFilterSelector and GBRT model."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from noise_suppression.constants import SAFETY_THRESHOLD_DB
from noise_suppression.features import N_FEATURES
from noise_suppression.filters import ACTIONS
from noise_suppression.selector import SupervisedFilterSelector


@pytest.fixture
def dummy_dataset():
    rng = np.random.default_rng(42)
    n_samples = 40
    X = rng.normal(0, 1.0, size=(n_samples, N_FEATURES)).astype(np.float32)

    # Let feature 0 strongly correlate with wiener filter benefit
    y_per_action = {}
    for act in ACTIONS:
        if act == "wiener":
            y_per_action[act] = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.1, size=n_samples)
        elif act == "keep_original":
            y_per_action[act] = np.zeros(n_samples, dtype=np.float32)
        else:
            y_per_action[act] = -0.5 + rng.normal(0, 0.2, size=n_samples)

    return X, y_per_action


def test_fit_and_predict(dummy_dataset):
    X, y_per_action = dummy_dataset
    selector = SupervisedFilterSelector(
        model_params={"n_estimators": 20, "max_depth": 3, "random_state": 42}
    )
    selector.fit(X, y_per_action)

    # Predict single sample
    gains = selector.predict_gains(X[0])
    assert len(gains) == len(ACTIONS)
    assert all(isinstance(v, float) for v in gains.values())

    # Predict batch
    batch_gains = selector.predict_gains_batch(X)
    assert len(batch_gains) == len(ACTIONS)
    assert batch_gains["wiener"].shape == (len(X),)

    # Select
    best_act = selector.select(X[0])
    assert best_act in ACTIONS

    # Batch select
    selections = selector.select_batch(X)
    assert len(selections) == len(X)


def test_safety_threshold(dummy_dataset):
    X, _ = dummy_dataset
    # Force all gains to negative
    y_per_action = {act: np.full(len(X), -2.0, dtype=np.float32) for act in ACTIONS}
    selector = SupervisedFilterSelector(
        safety_threshold=0.05,
        model_params={"n_estimators": 10, "max_depth": 2, "random_state": 42}
    )
    selector.fit(X, y_per_action)

    # Since all predicted gains are negative (< 0.05), safety fallback should trigger
    action = selector.select(X[0])
    assert action == "keep_original"


def test_save_load_roundtrip(tmp_path: Path, dummy_dataset):
    X, y_per_action = dummy_dataset
    selector = SupervisedFilterSelector(
        model_params={"n_estimators": 20, "max_depth": 3, "random_state": 42}
    )
    selector.fit(X, y_per_action)

    save_path = tmp_path / "test_selector.joblib"
    selector.save(save_path)
    assert save_path.exists()

    loaded = SupervisedFilterSelector.load(save_path)
    pred_orig = selector.select(X[5])
    pred_loaded = loaded.select(X[5])
    assert pred_orig == pred_loaded

    gains_orig = selector.predict_gains(X[5])
    gains_loaded = loaded.predict_gains(X[5])
    for act in ACTIONS:
        assert abs(gains_orig[act] - gains_loaded[act]) < 1e-4


def test_feature_importances(dummy_dataset):
    X, y_per_action = dummy_dataset
    selector = SupervisedFilterSelector(
        model_params={"n_estimators": 20, "max_depth": 3, "random_state": 42}
    )
    selector.fit(X, y_per_action)

    importances = selector.feature_importances()
    assert "wiener" in importances
    top_feature_name, top_weight = importances["wiener"][0]
    # Feature 0 was designed to correlate with wiener
    assert top_feature_name == selector.feature_names[0]
