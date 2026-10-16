"""Unit tests for features extraction module."""
from __future__ import annotations

import numpy as np
import pytest

from noise_suppression.features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
    extract_state,
)


def test_feature_vector_shape_and_dtype():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    feat = extract_features(audio, sr)
    assert feat.shape == (N_FEATURES,)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_features_determinism():
    sr = 16000
    rng = np.random.default_rng(123)
    audio = rng.normal(0, 0.1, size=sr).astype(np.float32)

    feat1 = extract_features(audio, sr)
    feat2 = extract_features(audio, sr)
    np.testing.assert_allclose(feat1, feat2, rtol=1e-5, atol=1e-5)


def test_short_and_silent_clips():
    sr = 16000
    # Very short clip
    short_audio = np.zeros(16, dtype=np.float32)
    feat_short = extract_features(short_audio, sr)
    assert feat_short.shape == (N_FEATURES,)
    assert np.all(np.isfinite(feat_short))

    # All digital silence
    silent_audio = np.zeros(sr, dtype=np.float32)
    feat_silent = extract_features(silent_audio, sr)
    assert feat_silent.shape == (N_FEATURES,)
    assert np.all(np.isfinite(feat_silent))


def test_features_discriminative():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 500 * t)
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 0.2, size=sr).astype(np.float32)

    feat_tone = extract_features(tone, sr)
    feat_noise = extract_features(white_noise, sr)

    flatness_idx = FEATURE_NAMES.index("spectral_flatness")
    # Pure tone has much lower spectral flatness than white noise
    assert feat_tone[flatness_idx] < feat_noise[flatness_idx]

    # Hum 50Hz detector
    hum_50hz = np.sin(2 * np.pi * 50 * t)
    feat_hum = extract_features(hum_50hz, sr)
    hum_idx = FEATURE_NAMES.index("hum_50hz_power_ratio")
    assert feat_hum[hum_idx] > 0.5


def test_legacy_extract_state():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    state = extract_state(audio, sr)
    assert isinstance(state, str)
    parts = state.split("_")
    assert len(parts) == 5
    for p in parts:
        assert p.isdigit()
