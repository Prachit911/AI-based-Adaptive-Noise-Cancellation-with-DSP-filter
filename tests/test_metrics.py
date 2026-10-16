"""Unit tests for the speech enhancement metrics module."""
from __future__ import annotations

import numpy as np
import pytest

from noise_suppression.metrics import (
    compute_si_sdr,
    compute_stoi,
    compute_pesq,
    compute_all_metrics,
)


def test_compute_si_sdr_identity_and_noise():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean = np.sin(2 * np.pi * 440 * t)

    val_id = compute_si_sdr(clean, clean)
    assert val_id > 80.0

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, size=sr).astype(np.float32)
    val_noise = compute_si_sdr(clean, noise)
    assert val_noise < 0.0


def test_compute_all_metrics():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean = 0.5 * np.sin(2 * np.pi * 300 * t)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, size=sr).astype(np.float32)
    noisy = clean + noise
    enhanced = clean + 0.5 * noise  # simulated improvement

    res = compute_all_metrics(clean, noisy, enhanced, sr)
    expected_keys = [
        "si_sdr_input_db",
        "si_sdr_output_db",
        "si_sdr_improvement_db",
        "stoi_input",
        "stoi_output",
        "stoi_improvement",
        "pesq_input",
        "pesq_output",
        "pesq_improvement",
    ]
    for k in expected_keys:
        assert k in res

    # Enhanced has less noise, so SI-SDR improvement should be positive
    assert res["si_sdr_improvement_db"] > 0.0


def test_length_mismatch_handling():
    sr = 16000
    clean = np.ones(1000, dtype=np.float32)
    noisy = np.ones(1200, dtype=np.float32)
    enhanced = np.ones(900, dtype=np.float32)

    res = compute_all_metrics(clean, noisy, enhanced, sr)
    assert "si_sdr_improvement_db" in res
