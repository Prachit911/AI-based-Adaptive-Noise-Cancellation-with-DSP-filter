"""Supervised filter selector using Gradient Boosted Regression Trees.

Replaces the tabular Q-table with a continuous selector that:
  1. Predicts the SI-SDR improvement for each action from the feature vector.
  2. Picks the action with the highest predicted gain.
  3. Falls back to ``keep_original`` if the best predicted gain is below
     a safety threshold.

The selector trains one regressor per action (one-vs-rest style) on the
full-information sweep data, so it sees every action's true reward on
every training clip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .constants import FEATURE_VERSION, SAFETY_THRESHOLD_DB
from .features import FEATURE_NAMES, N_FEATURES
from .filters import ACTIONS


class SupervisedFilterSelector:
    """Predicts per-action SI-SDR improvement; picks the best action."""

    def __init__(
        self,
        safety_threshold: float = SAFETY_THRESHOLD_DB,
        model_params: dict[str, Any] | None = None,
    ):
        self.safety_threshold = safety_threshold
        self.feature_names = list(FEATURE_NAMES)
        self.feature_version = FEATURE_VERSION
        self.actions = list(ACTIONS)
        self.models: dict[str, GradientBoostingRegressor] = {}

        # Default GBRT hyper-parameters
        self.model_params = model_params or {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "min_samples_leaf": 10,
            "random_state": 42,
        }

    def fit(
        self,
        X: np.ndarray,
        y_per_action: dict[str, np.ndarray],
    ) -> "SupervisedFilterSelector":
        """Train one regressor per action.

        Parameters
        ----------
        X : (n_clips, n_features) feature matrix.
        y_per_action : mapping action_name → (n_clips,) SI-SDR improvements.
        """
        for action in self.actions:
            if action not in y_per_action:
                print(f"WARNING: no labels for action '{action}' -- skipping")
                continue
            y = y_per_action[action]
            if len(y) != X.shape[0]:
                raise ValueError(
                    f"X has {X.shape[0]} rows but y['{action}'] has {len(y)}"
                )
            print(f"  Training regressor for '{action}' "
                  f"(mean target: {np.mean(y):+.3f} dB) ...")
            model = GradientBoostingRegressor(**self.model_params)
            model.fit(X, y)
            self.models[action] = model
            print(f"    -> train R^2: {model.score(X, y):.4f}")

        return self

    def predict_gains(self, features: np.ndarray) -> dict[str, float]:
        """Predict SI-SDR improvement for each action."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return {
            action: float(model.predict(x)[0])
            for action, model in self.models.items()
        }

    def predict_gains_batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict gains for a batch of feature vectors."""
        X = np.asarray(X, dtype=np.float32)
        return {
            action: model.predict(X).astype(np.float32)
            for action, model in self.models.items()
        }

    def select(self, features: np.ndarray) -> str:
        """Select the best action for a single feature vector."""
        gains = self.predict_gains(features)
        if not gains:
            return "keep_original"
        best_action = max(gains, key=gains.get)
        if gains[best_action] < self.safety_threshold:
            return "keep_original"
        return best_action

    def select_batch(self, X: np.ndarray) -> list[str]:
        """Select the best action for each row in a feature matrix."""
        gains = self.predict_gains_batch(X)
        n = X.shape[0]
        actions = []
        for i in range(n):
            row_gains = {a: float(g[i]) for a, g in gains.items()}
            best = max(row_gains, key=row_gains.get)
            if row_gains[best] < self.safety_threshold:
                best = "keep_original"
            actions.append(best)
        return actions

    def feature_importances(self) -> dict[str, list[tuple[str, float]]]:
        """Return sorted feature importances per action."""
        result = {}
        for action, model in self.models.items():
            importances = model.feature_importances_
            pairs = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            result[action] = pairs
        return result

    def save(self, path: str | Path) -> None:
        """Save the selector to a joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 2,
            "algorithm": "supervised_gbrt_selector",
            "feature_version": self.feature_version,
            "feature_names": self.feature_names,
            "actions": self.actions,
            "safety_threshold": self.safety_threshold,
            "model_params": self.model_params,
            "models": self.models,
        }
        joblib.dump(payload, path)
        print(f"Selector saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedFilterSelector":
        """Load a selector from a joblib file."""
        payload = joblib.load(path)
        if payload.get("format_version") != 2:
            raise ValueError("Unsupported selector format version")
        selector = cls(
            safety_threshold=payload.get("safety_threshold", SAFETY_THRESHOLD_DB),
            model_params=payload.get("model_params"),
        )
        selector.feature_version = payload.get("feature_version", "unknown")
        selector.feature_names = payload.get("feature_names", list(FEATURE_NAMES))
        selector.actions = payload.get("actions", list(ACTIONS))
        selector.models = payload.get("models", {})
        return selector
