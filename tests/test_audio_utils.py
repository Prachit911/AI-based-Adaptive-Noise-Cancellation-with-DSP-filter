"""Unit tests for audio_utils module."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from noise_suppression.audio_utils import (
    read_audio,
    write_audio,
    resample_audio,
    trim_digital_silence,
    rms,
    normalize_level,
    level_match,
    mix_at_snr,
    mix_dual_noise,
    si_sdr,
)


def test_read_write_roundtrip(tmp_path: Path):
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)

    wav_path = tmp_path / "test_tone.wav"
    write_audio(wav_path, tone, sr)
    assert wav_path.exists()

    loaded, loaded_sr = read_audio(wav_path)
    assert loaded_sr == sr
    assert loaded.shape == tone.shape
    np.testing.assert_allclose(loaded, tone, atol=1e-4)


def test_resample_audio():
    sr_in = 44100
    sr_out = 16000
    t = np.linspace(0, 1.0, sr_in, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 300 * t)

    resampled = resample_audio(tone, sr_in, sr_out)
    assert abs(len(resampled) - sr_out) <= 2
    assert resampled.dtype == np.float32


def test_trim_digital_silence():
    sr = 16000
    # 0.5s digital zero, 1.0s tone, 0.5s digital zero
    zeros_prefix = np.zeros(int(0.5 * sr), dtype=np.float32)
    zeros_suffix = np.zeros(int(0.5 * sr), dtype=np.float32)
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)

    padded = np.concatenate([zeros_prefix, tone, zeros_suffix])
    trimmed = trim_digital_silence(padded, sr, threshold_db=-50.0, margin_ms=10.0)

    # Should have trimmed most of the 0.5s silence on both sides
    assert len(trimmed) < len(padded)
    margin_samples = int(sr * 0.010)
    assert abs(len(trimmed) - (len(tone) + 2 * margin_samples)) < sr * 0.05


def test_normalize_level_and_level_match():
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    x = 0.1 * np.sin(2 * np.pi * 440 * t)
    target = 0.05
    normed = normalize_level(x, target_rms=target)
    assert abs(rms(normed) - target) < 1e-4

    y = 0.8 * np.sin(2 * np.pi * 200 * t)
    matched = level_match(x, y)
    assert abs(rms(matched) - rms(x)) < 1e-4


def test_mix_at_snr():
    rng = np.random.default_rng(42)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean = 0.3 * np.sin(2 * np.pi * 400 * t)
    noise = rng.normal(0, 0.1, size=sr).astype(np.float32)

    snr_target = 10.0
    mixture = mix_at_snr(clean, noise, snr_target, rng)
    assert len(mixture) == len(clean)

    # Effective noise is mixture - clean
    extracted_noise = mixture - clean
    actual_snr = 20 * np.log10(rms(clean) / rms(extracted_noise))
    assert abs(actual_snr - snr_target) < 0.2


def test_si_sdr_known_cases():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean = np.sin(2 * np.pi * 440 * t)

    # Identity
    score_id = si_sdr(clean, clean)
    assert score_id > 80.0  # Perfect score

    # Scale invariant
    score_scaled = si_sdr(clean, 3.5 * clean)
    assert score_scaled > 80.0

    # Uncorrelated noise
    noise = np.cos(2 * np.pi * 880 * t)
    score_noisy = si_sdr(clean, clean + noise)
    assert score_noisy < score_id
