"""Continuous audio feature extraction for the supervised selector.

Returns a fixed-order float32 vector (versioned) that captures spectral
shape, temporal dynamics, noise character, and band energies — everything
the selector needs to predict which filter will work best on a given clip.

The legacy discrete-state function ``extract_state`` is preserved for the
Q-table baseline agent.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import kurtosis as _scipy_kurtosis

from .constants import FEATURE_VERSION, INTERNAL_SR

# ─────────────────────────────────────────────────────────────────────
# Feature names — the canonical, ordered list.
# Every value of extract_features() is aligned to this list.
# ─────────────────────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    # 9 original features (continuous versions)
    "rms_dbfs",
    "spectral_flatness",
    "spectral_flux",
    "low_band_power_share",       # < 250 Hz
    "high_band_power_share",      # >= 4000 Hz
    "frame_energy_variability",
    "spectral_centroid_hz",
    "normalized_centroid",
    "spectral_entropy",
    # 13 MFCC means
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
    "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12",
    # Additional discriminative features
    "noise_stationarity",
    "impulsiveness_kurtosis",
    "impulsiveness_crest",
    "hum_50hz_power_ratio",
    "hum_60hz_power_ratio",
    "snr_estimate_db",
    "band_energy_0_250",          # log-scale absolute band energy
    "band_energy_250_1000",
    "band_energy_1000_4000",
    "band_energy_4000_8000",
    # 3 new cheap features
    "zero_crossing_rate",
    "harmonic_to_noise_ratio",
    "spectral_rolloff_hz",
]

N_FEATURES = len(FEATURE_NAMES)

# ─────────────────────────────────────────────────────────────────────
# STFT helper (shared with filters)
# ─────────────────────────────────────────────────────────────────────

def _nperseg(x: np.ndarray, sr: int) -> int:
    return min(512, max(64, int(sr * 0.032)), len(x))


def _stft_mag_phase(x: np.ndarray, sr: int):
    """Return (freqs, magnitude, power, n_frames, nperseg)."""
    nperseg = _nperseg(x, sr)
    noverlap = min(int(nperseg * 0.75), nperseg - 1)
    freqs, _, z = signal.stft(
        x, fs=sr, nperseg=nperseg, noverlap=noverlap,
        window="hann", boundary=None,
    )
    mag = np.abs(z) + 1e-10
    power = mag ** 2
    return freqs, mag, power, z.shape[1], nperseg


# ─────────────────────────────────────────────────────────────────────
# Individual feature extractors
# ─────────────────────────────────────────────────────────────────────

def _rms_dbfs(x: np.ndarray) -> float:
    rms_val = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
    return 20.0 * np.log10(rms_val + 1e-10)


def _spectral_flatness(power: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of power spectrum (per frame, median)."""
    flat_per_frame = np.exp(np.mean(np.log(power + 1e-10), axis=0)) / (
        np.mean(power, axis=0) + 1e-10
    )
    return float(np.median(flat_per_frame))


def _spectral_flux(mag: np.ndarray) -> float:
    """Mean L2 norm of frame-to-frame normalized magnitude difference."""
    normalized = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)
    if normalized.shape[1] > 1:
        diff = np.diff(normalized, axis=1)
        return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))
    return 0.0


def _band_power_share(
    power: np.ndarray, freqs: np.ndarray, low: float, high: float
) -> float:
    total = float(np.sum(power) + 1e-10)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask, :]) / total) if np.any(mask) else 0.0


def _frame_energy_variability(power: np.ndarray) -> float:
    frame_rms = np.sqrt(np.mean(power, axis=0) + 1e-10)
    return float(np.std(frame_rms) / (np.mean(frame_rms) + 1e-10))


def _spectral_centroid(mag: np.ndarray, freqs: np.ndarray, sr: int):
    """Return (centroid_hz, normalized_centroid)."""
    total_mag = np.sum(mag, axis=0) + 1e-10
    centroid_per_frame = np.sum(
        freqs[:, None] * mag, axis=0
    ) / total_mag
    centroid_hz = float(np.median(centroid_per_frame))
    nyquist = sr / 2.0
    normalized = centroid_hz / nyquist if nyquist > 0 else 0.0
    return centroid_hz, normalized


def _spectral_entropy(power: np.ndarray) -> float:
    """Entropy of the power spectrum distribution (per frame, median)."""
    # Normalize power to probability distribution per frame
    p = power / (np.sum(power, axis=0, keepdims=True) + 1e-10)
    entropy_per_frame = -np.sum(p * np.log2(p + 1e-10), axis=0)
    return float(np.median(entropy_per_frame))


def _mfcc_means(x: np.ndarray, sr: int) -> np.ndarray:
    """Return 13 MFCC means using librosa."""
    try:
        import librosa
        mfccs = librosa.feature.mfcc(y=x.astype(np.float32), sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1).astype(np.float32)
    except ImportError:
        # Fallback: return zeros if librosa unavailable
        return np.zeros(13, dtype=np.float32)


def _noise_stationarity(power: np.ndarray) -> float:
    """Variance of per-frame noise-floor estimates (min across bands).

    Low values → stationary noise; high values → non-stationary or speech-like.
    """
    # Simple noise floor: minimum power per frequency bin across frames
    frame_noise = np.min(power, axis=0)  # shape: (n_frames,)
    if frame_noise.size < 2:
        return 0.0
    return float(np.std(frame_noise) / (np.mean(frame_noise) + 1e-10))


def _impulsiveness(x: np.ndarray):
    """Return (excess kurtosis, crest factor)."""
    x64 = x.astype(np.float64)
    # Kurtosis
    try:
        kurt = float(_scipy_kurtosis(x64, fisher=True))
    except Exception:
        kurt = 0.0
    kurt = float(np.clip(kurt, -10.0, 100.0))

    # Crest factor: peak / RMS
    rms_val = np.sqrt(np.mean(x64 ** 2) + 1e-12)
    peak = float(np.max(np.abs(x64)))
    crest = peak / rms_val if rms_val > 1e-8 else 0.0
    crest = float(np.clip(crest, 0.0, 100.0))

    return kurt, crest


def _hum_power_ratio(
    signal_or_power: np.ndarray,
    sr_or_freqs: int | np.ndarray,
    center_hz: float,
    bw_hz_or_q: float = 10.0,
) -> float:
    """Ratio of power in a narrow band around *center_hz* to total power.

    If 1D time-domain audio `x` and integer `sr` are provided, uses a sharp
    notch filter to precisely isolate and measure the hum component.
    If 2D spectrogram `power` and 1D `freqs` are provided, computes the
    ratio from spectral bins around `center_hz`.
    """
    if isinstance(sr_or_freqs, (int, np.integer)):
        x = np.asarray(signal_or_power, dtype=np.float32).reshape(-1)
        sr = int(sr_or_freqs)
        if x.size < 16 or sr <= 2 * center_hz + 10:
            return 0.0
        try:
            b, a = signal.iirnotch(w0=center_hz, Q=bw_hz_or_q, fs=sr)
            notched = signal.filtfilt(b, a, x.astype(np.float64))
            p_total = float(np.mean(x.astype(np.float64) ** 2) + 1e-12)
            p_hum = float(np.mean((x.astype(np.float64) - notched) ** 2))
            return float(np.clip(p_hum / p_total, 0.0, 1.0))
        except Exception:
            return 0.0
    else:
        power = signal_or_power
        freqs = sr_or_freqs
        total = float(np.sum(power) + 1e-10)
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        bw = max(float(bw_hz_or_q), 1.5 * df)
        mask = (freqs >= center_hz - bw / 2) & (freqs <= center_hz + bw / 2)
        if not np.any(mask):
            closest_idx = np.argmin(np.abs(freqs - center_hz))
            mask = np.zeros_like(freqs, dtype=bool)
            mask[closest_idx] = True
        return float(np.sum(power[mask, :]) / total)


def _snr_estimate(x: np.ndarray, sr: int) -> float:
    """Simple VAD-based SNR estimate in dB.

    Uses frame energy to separate speech and noise frames.
    """
    frame_len = max(1, int(sr * 0.025))  # 25 ms frames
    hop = max(1, frame_len // 2)
    n_frames = max(1, (len(x) - frame_len) // hop)
    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + frame_len]
        energies[i] = np.mean(frame.astype(np.float64) ** 2)

    if n_frames < 4:
        return 0.0

    # Threshold: frames above 2× median energy are "speech"
    median_e = np.median(energies)
    speech_mask = energies > 2.0 * median_e
    noise_mask = ~speech_mask

    if not np.any(speech_mask) or not np.any(noise_mask):
        return 0.0

    speech_power = float(np.mean(energies[speech_mask]))
    noise_power = float(np.mean(energies[noise_mask]))
    ratio = speech_power / (noise_power + 1e-12)
    return float(np.clip(10.0 * np.log10(ratio + 1e-10), -30.0, 50.0))


def _band_energy_log(
    power: np.ndarray, freqs: np.ndarray, low: float, high: float
) -> float:
    """Absolute log-scale energy in a frequency band."""
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return -100.0
    return float(10.0 * np.log10(np.mean(power[mask, :]) + 1e-10))


def _zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of samples where the signal changes sign."""
    signs = np.sign(x)
    changes = np.abs(np.diff(signs))
    return float(np.mean(changes > 0))


def _harmonic_to_noise_ratio(x: np.ndarray, sr: int) -> float:
    """Estimate HNR using autocorrelation.

    Finds the peak of the autocorrelation in the pitch range (80-400 Hz)
    and computes HNR from the ratio.
    """
    x64 = x.astype(np.float64)
    if len(x64) < sr // 80:  # need at least one pitch period
        return 0.0

    # Autocorrelation via FFT (faster for long signals)
    n = len(x64)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    X = np.fft.rfft(x64, n=fft_size)
    acf = np.fft.irfft(X * np.conj(X))[:n]

    # Normalize
    acf_0 = acf[0]
    if acf_0 < 1e-12:
        return 0.0
    acf = acf / acf_0

    # Search for peak in pitch range (80-400 Hz)
    min_lag = max(1, sr // 400)
    max_lag = min(n - 1, sr // 80)
    if min_lag >= max_lag:
        return 0.0

    search = acf[min_lag:max_lag + 1]
    peak_val = float(np.max(search))
    peak_val = np.clip(peak_val, -1.0, 0.9999)  # avoid log(0) or inf

    # HNR in dB
    hnr = 10.0 * np.log10((peak_val + 1e-10) / (1.0 - peak_val + 1e-10))
    return float(np.clip(hnr, -20.0, 40.0))


def _spectral_rolloff(mag: np.ndarray, freqs: np.ndarray, percentile: float = 0.85) -> float:
    """Frequency below which *percentile* of the spectral energy lies."""
    total_energy = np.sum(mag, axis=0) + 1e-10
    cumulative = np.cumsum(mag, axis=0)
    threshold = percentile * total_energy
    # For each frame, find the frequency bin where cumulative exceeds threshold
    rolloff_per_frame = np.zeros(mag.shape[1])
    for f in range(mag.shape[1]):
        idx = np.searchsorted(cumulative[:, f], threshold[f])
        idx = min(idx, len(freqs) - 1)
        rolloff_per_frame[f] = freqs[idx]
    return float(np.median(rolloff_per_frame))


# ─────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────

def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return a 1-D float32 vector of length N_FEATURES.

    The vector is aligned to FEATURE_NAMES and tagged with FEATURE_VERSION.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)

    # Handle very short clips
    if x.size < 64:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # STFT-based features
    freqs, mag, power, n_frames, nperseg = _stft_mag_phase(x, sr)

    features = np.zeros(N_FEATURES, dtype=np.float32)
    idx = 0

    # 1. RMS dBFS
    features[idx] = _rms_dbfs(x); idx += 1
    # 2. Spectral flatness
    features[idx] = _spectral_flatness(power); idx += 1
    # 3. Spectral flux
    features[idx] = _spectral_flux(mag); idx += 1
    # 4. Low band power share (<250 Hz)
    features[idx] = _band_power_share(power, freqs, 0, 250); idx += 1
    # 5. High band power share (>=4000 Hz)
    features[idx] = _band_power_share(power, freqs, 4000, sr / 2); idx += 1
    # 6. Frame energy variability
    features[idx] = _frame_energy_variability(power); idx += 1
    # 7-8. Spectral centroid
    centroid_hz, norm_centroid = _spectral_centroid(mag, freqs, sr)
    features[idx] = centroid_hz; idx += 1
    features[idx] = norm_centroid; idx += 1
    # 9. Spectral entropy
    features[idx] = _spectral_entropy(power); idx += 1

    # 10-22. MFCCs (13 coefficients)
    mfcc = _mfcc_means(x, sr)
    features[idx:idx + 13] = mfcc; idx += 13

    # 23. Noise stationarity
    features[idx] = _noise_stationarity(power); idx += 1
    # 24-25. Impulsiveness
    kurt, crest = _impulsiveness(x)
    features[idx] = kurt; idx += 1
    features[idx] = crest; idx += 1
    # 26-27. Hum detectors
    features[idx] = _hum_power_ratio(x, sr, 50.0); idx += 1
    features[idx] = _hum_power_ratio(x, sr, 60.0); idx += 1
    # 28. SNR estimate
    features[idx] = _snr_estimate(x, sr); idx += 1
    # 29-32. Band energies (log scale)
    features[idx] = _band_energy_log(power, freqs, 0, 250); idx += 1
    features[idx] = _band_energy_log(power, freqs, 250, 1000); idx += 1
    features[idx] = _band_energy_log(power, freqs, 1000, 4000); idx += 1
    features[idx] = _band_energy_log(power, freqs, 4000, 8000); idx += 1
    # 33. Zero-crossing rate
    features[idx] = _zero_crossing_rate(x); idx += 1
    # 34. Harmonic-to-noise ratio
    features[idx] = _harmonic_to_noise_ratio(x, sr); idx += 1
    # 35. Spectral rolloff
    features[idx] = _spectral_rolloff(mag, freqs); idx += 1

    assert idx == N_FEATURES, f"Expected {N_FEATURES} features, got {idx}"

    # Replace any NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


# ─────────────────────────────────────────────────────────────────────
# Legacy discrete state for Q-table baseline
# ─────────────────────────────────────────────────────────────────────

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
