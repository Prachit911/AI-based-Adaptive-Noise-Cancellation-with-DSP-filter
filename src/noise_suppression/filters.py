"""Baseline DSP filters used as selectable actions by the RL agent."""
from __future__ import annotations
import numpy as np
from scipy import signal

ACTIONS = ["keep_original", "spectral_subtraction", "wiener", "speech_bandpass", "notch_50hz"]


def _fit_length(y: np.ndarray, target_length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < target_length:
        y = np.pad(y, (0, target_length - y.size))
    return y[:target_length]


def _nperseg(x: np.ndarray, sample_rate: int) -> int:
    return min(512, max(16, int(sample_rate * 0.032)), len(x))


def _stft(x: np.ndarray, sample_rate: int):
    nperseg = _nperseg(x, sample_rate)
    noverlap = min(int(nperseg * 0.75), nperseg - 1)
    return signal.stft(x, fs=sample_rate, window="hann", nperseg=nperseg,
                       noverlap=noverlap, boundary="zeros", padded=True)


def _istft(z: np.ndarray, sample_rate: int, nperseg: int, target_length: int) -> np.ndarray:
    noverlap = min(int(nperseg * 0.75), nperseg - 1)
    _, y = signal.istft(z, fs=sample_rate, window="hann", nperseg=nperseg,
                        noverlap=noverlap, boundary=True)
    return _fit_length(y, target_length)


def _noise_frames(magnitude: np.ndarray) -> np.ndarray:
    # Estimate a noise profile from the quieter 20% of frames. This is a useful
    # baseline, not a perfect voice-activity detector.
    energies = np.sqrt(np.mean(magnitude ** 2, axis=0) + 1e-12)
    threshold = np.percentile(energies, 20)
    mask = energies <= threshold
    if not np.any(mask):
        mask[int(np.argmin(energies))] = True
    return mask


def spectral_subtraction(audio: np.ndarray, sample_rate: int, strength: float = 1.15) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 32:
        return x.copy()
    _, _, z = _stft(x, sample_rate)
    mag = np.abs(z)
    phase = np.exp(1j * np.angle(z))
    noise_mask = _noise_frames(mag)
    noise_profile = np.median(mag[:, noise_mask], axis=1, keepdims=True)
    enhanced_mag = np.maximum(mag - float(strength) * noise_profile, 0.08 * mag)
    y = _istft(enhanced_mag * phase, sample_rate, _nperseg(x, sample_rate), len(x))
    return y


def wiener_filter(audio: np.ndarray, sample_rate: int, strength: float = 1.0) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 32:
        return x.copy()
    _, _, z = _stft(x, sample_rate)
    power = np.abs(z) ** 2
    noise_mask = _noise_frames(np.sqrt(power))
    noise_power = np.median(power[:, noise_mask], axis=1, keepdims=True)
    gain = power / (power + float(strength) * noise_power + 1e-12)
    gain = np.clip(gain, 0.08, 1.0)
    nperseg = _nperseg(x, sample_rate)
    return _istft(z * gain, sample_rate, nperseg, len(x))


def speech_bandpass(audio: np.ndarray, sample_rate: int, low_hz: float = 80.0, high_hz: float = 7600.0) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    nyquist = sample_rate / 2.0
    low = max(20.0, float(low_hz))
    high = min(float(high_hz), nyquist * 0.95)
    if x.size < 16 or high <= low or nyquist <= low:
        return x.copy()
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    try:
        y = signal.sosfiltfilt(sos, x)
    except ValueError:  # Too-short clip for zero-phase filtering.
        y = signal.sosfilt(sos, x)
    return _fit_length(y, len(x))


def notch_50hz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Suppress 50 Hz electrical hum and a few harmonics where available."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 16 or sample_rate <= 120:
        return x.copy()
    y = x.astype(np.float64)
    for freq in (50.0, 100.0, 150.0, 200.0):
        if freq >= sample_rate / 2.0 - 5:
            continue
        b, a = signal.iirnotch(w0=freq, Q=30.0, fs=sample_rate)
        try:
            y = signal.filtfilt(b, a, y)
        except ValueError:
            y = signal.lfilter(b, a, y)
    return _fit_length(y, len(x))


def apply_filter(action: str, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if action == "keep_original":
        return np.asarray(audio, dtype=np.float32).copy()
    if action == "spectral_subtraction":
        return spectral_subtraction(audio, sample_rate)
    if action == "wiener":
        return wiener_filter(audio, sample_rate)
    if action == "speech_bandpass":
        return speech_bandpass(audio, sample_rate)
    if action == "notch_50hz":
        return notch_50hz(audio, sample_rate)
    raise ValueError(f"Unknown action '{action}'. Available actions: {', '.join(ACTIONS)}")
