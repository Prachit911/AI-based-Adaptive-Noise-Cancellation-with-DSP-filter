"""Optional noise-type classifier trained on ESC-50 categories from audio features.

This classifier:
  1. Maps the acoustic feature vector to predicted ESC-50 noise classes (e.g., siren, engine, rain).
  2. Outputs class probabilities that can be appended to the feature vector as soft noise-type priors.
  3. Provides per-category diagnostics to assess which noise environments the feature extractor recognizes.

Usage:
    python -m noise_suppression.noise_classifier \
        --sweep data/sweep_results.csv \
        --output models/noise_classifier.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from .features import FEATURE_NAMES


class NoiseTypeClassifier:
    """Predicts ESC-50 noise category from continuous acoustic features."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y_categories: list[str] | np.ndarray) -> "NoiseTypeClassifier":
        """Fit classifier on features and string categories."""
        y_enc = self.label_encoder.fit_transform(y_categories)
        self.classes_ = [str(c) for c in self.label_encoder.classes_]
        self.model.fit(X, y_enc)
        return self

    def predict(self, features: np.ndarray) -> str:
        """Predict the most likely noise category for a single feature vector."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        idx = int(self.model.predict(x)[0])
        return self.classes_[idx]

    def predict_batch(self, X: np.ndarray) -> list[str]:
        """Predict categories for a batch of feature vectors."""
        X = np.asarray(X, dtype=np.float32)
        indices = self.model.predict(X)
        return [self.classes_[i] for i in indices]

    def predict_proba(self, features: np.ndarray) -> dict[str, float]:
        """Predict class probability distribution for a single clip."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        return {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes_)}

    def predict_proba_vector(self, features: np.ndarray) -> np.ndarray:
        """Return 1-D numpy array of probabilities aligned with self.classes_."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return self.model.predict_proba(x)[0].astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Persist classifier to joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "classes": self.classes_,
            "label_encoder": self.label_encoder,
            "model": self.model,
            "feature_names": list(FEATURE_NAMES),
        }
        joblib.dump(payload, path)
        print(f"Noise classifier saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "NoiseTypeClassifier":
        """Load classifier from joblib file."""
        payload = joblib.load(path)
        inst = cls()
        inst.classes_ = payload["classes"]
        inst.label_encoder = payload["label_encoder"]
        inst.model = payload["model"]
        return inst


def train_noise_classifier(
    sweep_csv: Path,
    output_path: Path,
) -> NoiseTypeClassifier:
    """Train and evaluate the noise classifier from sweep results."""
    df = pd.read_csv(sweep_csv)
    feat_cols = [f"feat_{name}" for name in FEATURE_NAMES]

    # Deduplicate mixtures (one feature vector per mixture)
    mix_df = df.drop_duplicates(subset=["mixture_id"]).copy()

    # Exclude clean or unknown categories if present
    mix_df = mix_df[mix_df["noise_category"].notna() & (mix_df["noise_category"] != "")].copy()

    # Split train/val
    has_train = "train" in mix_df["split"].values
    has_val = "val" in mix_df["split"].values

    if has_train:
        train_df = mix_df[mix_df["split"] == "train"]
        test_df = mix_df[mix_df["split"] == "val"] if has_val else train_df
    else:
        # If no split column or single split, split 80/20 randomly
        mask = np.random.default_rng(42).random(len(mix_df)) < 0.8
        train_df = mix_df[mask]
        test_df = mix_df[~mask]

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["noise_category"].tolist()
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df["noise_category"].tolist()

    print(f"Training Noise Classifier on {len(X_train)} samples across {len(set(y_train))} categories ...")
    clf = NoiseTypeClassifier()
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict_batch(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nNoise Classifier Accuracy on validation set: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    clf.save(output_path)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optional noise classifier on ESC-50 categories.")
    parser.add_argument("--sweep", type=Path, default=Path("data/sweep_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/noise_classifier.joblib"))
    args = parser.parse_args()

    train_noise_classifier(args.sweep, args.output)


if __name__ == "__main__":
    main()
