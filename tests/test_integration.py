"""End-to-end integration test of the full speech enhancement pipeline."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from noise_suppression.audio_utils import write_audio
from noise_suppression.constants import INTERNAL_SR
from noise_suppression.enhance import enhance_audio
from noise_suppression.features import FEATURE_NAMES, extract_features
from noise_suppression.filters import ACTIONS, apply_filter
from noise_suppression.selector import SupervisedFilterSelector
from noise_suppression.train_selector import load_sweep_dataset, evaluate_selector_on_split


def test_full_pipeline_end_to_end(tmp_path: Path):
    sr = INTERNAL_SR
    rng = np.random.default_rng(42)

    # 1. Create synthetic clean speech and noise clips
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
    clean_audio = 0.4 * np.sin(2 * np.pi * 350 * t) + 0.2 * np.sin(2 * np.pi * 700 * t)
    noise_audio = rng.normal(0, 0.1, size=len(t)).astype(np.float32)
    mixture_audio = clean_audio + noise_audio

    noisy_wav = tmp_path / "noisy_sample.wav"
    clean_wav = tmp_path / "clean_sample.wav"
    write_audio(noisy_wav, mixture_audio, sr)
    write_audio(clean_wav, clean_audio, sr)

    # 2. Simulate sweep data rows
    sweep_rows = []
    features = extract_features(mixture_audio, sr)

    for mix_idx, split_name in [(1, "train"), (2, "train"), (3, "val")]:
        mix_id = f"mix_{mix_idx:03d}"
        for act in ACTIONS:
            # Synthetic delta SI-SDR
            gain = 1.5 if act == "wiener" else (0.0 if act == "keep_original" else -0.5)
            row = {
                "mixture_id": mix_id,
                "split": split_name,
                "noise_category": "engine",
                "snr_db": 5.0,
                "action": act,
                "si_sdr_improvement_db": gain,
                "stoi_improvement": 0.05 if gain > 0 else -0.01,
                "si_sdr_output_db": 5.0 + gain,
            }
            for i, fname in enumerate(FEATURE_NAMES):
                row[f"feat_{fname}"] = float(features[i])
            sweep_rows.append(row)

    sweep_csv = tmp_path / "sweep_results.csv"
    pd.DataFrame(sweep_rows).to_csv(sweep_csv, index=False)
    assert sweep_csv.exists()

    # 3. Load sweep dataset and train selector
    train_meta, X_train, y_train, _ = load_sweep_dataset(sweep_csv, split="train")
    assert len(X_train) == 2
    selector = SupervisedFilterSelector(
        model_params={"n_estimators": 10, "max_depth": 2, "random_state": 42}
    )
    selector.fit(X_train, y_train)

    # 4. Evaluate on val
    val_meta, X_val, y_val, _ = load_sweep_dataset(sweep_csv, split="val")
    val_eval = evaluate_selector_on_split(selector, X_val, y_val, val_meta)
    assert val_eval["n_samples"] == 1
    assert "selector_mean_gain_db" in val_eval

    # 5. Save model
    model_path = tmp_path / "selector.joblib"
    selector.save(model_path)
    assert model_path.exists()

    # 6. Run enhance_audio CLI helper
    out_enhanced = tmp_path / "enhanced.wav"
    chosen_act, details = enhance_audio(
        input_path=noisy_wav,
        output_path=out_enhanced,
        method="selector",
        selector_path=model_path,
    )
    assert out_enhanced.exists()
    assert chosen_act in ACTIONS
    assert "Supervised Selector" in details
