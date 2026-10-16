"""Unit tests for the extended DSP filter bank."""
from __future__ import annotations

import numpy as np
import pytest

from noise_suppression.filters import ACTIONS, apply_filter


@pytest.fixture
def test_audio():
    sr = 16000
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
    # Speech-like harmonic tone + hum + noise
    sig = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 50 * t)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, size=len(t)).astype(np.float32)
    return sig + noise, sr


def test_all_actions_defined():
    assert len(ACTIONS) == 12
    assert "keep_original" in ACTIONS
    assert "spectral_subtraction" in ACTIONS
    assert "wiener" in ACTIONS
    assert "log_mmse" in ACTIONS
    assert "median_declicker" in ACTIONS
    assert "bandpass_then_wiener" in ACTIONS


@pytest.mark.parametrize("action", ACTIONS)
def test_filter_contract(action, test_audio):
    x, sr = test_audio
    y = apply_filter(action, x, sr)

    # 1. Exact same length
    assert len(y) == len(x), f"Action {action} failed length contract"

    # 2. Output dtype float32
    assert y.dtype == np.float32

    # 3. No NaN / Inf
    assert np.all(np.isfinite(y)), f"Action {action} produced NaN/Inf"

    # 4. RMS level preservation (within 2%)
    rms_in = np.sqrt(np.mean(x.astype(np.float64)**2))
    rms_out = np.sqrt(np.mean(y.astype(np.float64)**2))
    assert abs(rms_out - rms_in) / rms_in < 0.05, f"Action {action} failed level match"


def test_notch_50hz_suppression():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    pure_50hz = np.sin(2 * np.pi * 50 * t)
    notched = apply_filter("notch_50hz", pure_50hz, sr)

    # Power in 50Hz should drop dramatically
    p_orig = np.mean(pure_50hz**2)
    # Measure attenuation before level-match or directly via filter
    from noise_suppression.filters import notch_50hz
    raw_notched = notch_50hz(pure_50hz, sr)
    p_notched = np.mean(raw_notched**2)
    suppression_db = 10 * np.log10(p_orig / (p_notched + 1e-12))
    assert suppression_db > 20.0  # At least 20 dB suppression of 50 Hz


def test_median_declicker_transient_removal():
    sr = 16000
    x = np.zeros(sr, dtype=np.float32)
    # Add isolated clicks
    x[1000] = 1.0
    x[5000] = -1.0
    from noise_suppression.filters import median_declicker
    declicked = median_declicker(x, sr)
    assert abs(declicked[1000]) < 0.5
    assert abs(declicked[5000]) < 0.5
