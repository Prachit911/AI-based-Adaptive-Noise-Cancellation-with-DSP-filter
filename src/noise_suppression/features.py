"""Noise-condition features computed from the noisy waveform only."""
from __future__ import annotations
import numpy as np
from scipy import signal


def extract_state(audio: np.ndarray, sample_rate: int) -> str:
    """Return a discrete state key suitable for a small tabular Q-learning agent.

    Features: spectral flatness, spectral flux, low-frequency energy share,
    high-frequency energy share, and frame-energy variability.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 32:
        return "1_0_0_0_0"
    nperseg = min(512, max(64, int(sample_rate * 0.032)), len(x))
    noverlap = min(int(nperseg * 0.75), nperseg - 1)
    freqs, _, z = signal.stft(x, fs=sample_rate, nperseg=nperseg,
                              noverlap=noverlap, window="hann", boundary=None)
    mag = np.abs(z) + 1e-10
    power = mag ** 2

    flat_per_frame = np.exp(np.mean(np.log(power), axis=0)) / (np.mean(power, axis=0) + 1e-10)
    flatness = float(np.median(flat_per_frame))

    normalized = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)
    if normalized.shape[1] > 1:
        flux = float(np.mean(np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0))))
    else:
        flux = 0.0

    total_power = float(np.sum(power) + 1e-10)
    low_share = float(np.sum(power[freqs < 250.0, :]) / total_power)
    high_share = float(np.sum(power[freqs >= 4000.0, :]) / total_power)

    frame_rms = np.sqrt(np.mean(power, axis=0) + 1e-10)
    variability = float(np.std(frame_rms) / (np.mean(frame_rms) + 1e-10))

    flat_bin = 0 if flatness < 0.08 else (1 if flatness < 0.25 else 2)
    flux_bin = 0 if flux < 0.035 else 1
    low_bin = 0 if low_share < 0.20 else 1
    high_bin = 0 if high_share < 0.05 else (1 if high_share < 0.15 else 2)
    variability_bin = 0 if variability < 0.40 else 1
    return f"{flat_bin}_{flux_bin}_{low_bin}_{high_bin}_{variability_bin}"
