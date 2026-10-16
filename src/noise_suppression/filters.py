"""DSP filters used as selectable actions by the filter selector.

Filter bank (12 actions):
  1.  keep_original          — no processing (baseline)
  2.  spectral_subtraction   — over-subtraction with gain floor
  3.  wiener                 — Wiener filter with decision-directed gain
  4.  wiener_mild            — gentle Wiener (strength=0.5)
  5.  wiener_aggressive      — strong Wiener (strength=2.0)
  6.  speech_bandpass        — 80–7600 Hz 4th-order Butterworth
  7.  notch_50hz             — 50 Hz + harmonics notch
  8.  notch_60hz             — 60 Hz + harmonics notch
  9.  highpass_80hz          — 4th-order Butterworth high-pass at 80 Hz
  10. log_mmse               — Ephraim-Malah log-MMSE estimator
  11. median_declicker       — transient removal via median filter
  12. bandpass_then_wiener   — chain: speech bandpass → Wiener

Design contract for ``apply_filter``:
  - Input and output have **identical length**.
  - Output RMS is matched to input RMS (fair metrics).
  - No NaN or Inf in output.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import exp1  # for log-MMSE exponential integral

ACTIONS = [
    "keep_original",
    "spectral_subtraction",
    "wiener",
    "wiener_mild",
    "wiener_aggressive",
    "speech_bandpass",
    "notch_50hz",
    "notch_60hz",
    "highpass_80hz",
    "log_mmse",
    "median_declicker",
    "bandpass_then_wiener",
]


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────

def _fit_length(y: np.ndarray, target_length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < target_length:
        y = np.pad(y, (0, target_length - y.size))
    return y[:target_length]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def _level_match(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Scale *y* to have the same RMS as *x*."""
    rms_x = _rms(x)
    rms_y = _rms(y)
    if rms_y < 1e-10:
        return y
    return (y * (rms_x / rms_y)).astype(np.float32)


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


# ─────────────────────────────────────────────────────────────────────
# Minimum-statistics noise estimator (Martin, 2001 — simplified)
# ─────────────────────────────────────────────────────────────────────

def _minimum_statistics_noise(
    power: np.ndarray,
    alpha_s: float = 0.9,
    n_min_window: int = 12,
) -> np.ndarray:
    """Estimate noise power spectrum using minimum-statistics tracking.

    For each frequency bin, tracks a smoothed power estimate and keeps a
    running minimum over the last *n_min_window* frames.  Much more robust
    than the simple "quietest 20 %" heuristic.

    Parameters
    ----------
    power : (n_freq, n_frames) power spectrogram
    alpha_s : smoothing factor (0.8–0.95)
    n_min_window : number of frames for minimum search

    Returns
    -------
    noise_power : (n_freq, n_frames) estimated noise power per frame
    """
    n_freq, n_frames = power.shape
    noise_power = np.zeros_like(power)
    p_smooth = power[:, 0].copy()
    # Ring buffer of smoothed values for minimum search
    min_buf = np.full((n_freq, n_min_window), np.inf)
    min_buf[:, 0] = p_smooth

    noise_power[:, 0] = p_smooth

    for t in range(1, n_frames):
        p_smooth = alpha_s * p_smooth + (1.0 - alpha_s) * power[:, t]
        buf_idx = t % n_min_window
        min_buf[:, buf_idx] = p_smooth
        noise_power[:, t] = np.min(min_buf, axis=1)

    # Bias compensation (minimum statistics underestimate by ~factor of 1.5–2)
    noise_power *= 1.5

    return noise_power


def _noise_frames_legacy(magnitude: np.ndarray) -> np.ndarray:
    """Legacy noise estimator (quietest 20% of frames). Kept for fallback."""
    energies = np.sqrt(np.mean(magnitude ** 2, axis=0) + 1e-12)
    threshold = np.percentile(energies, 20)
    mask = energies <= threshold
    if not np.any(mask):
        mask[int(np.argmin(energies))] = True
    return mask


# ─────────────────────────────────────────────────────────────────────
# Filter implementations
# ─────────────────────────────────────────────────────────────────────

def spectral_subtraction(
    audio: np.ndarray,
    sample_rate: int,
    over_sub: float = 1.15,
    gain_floor: float = 0.08,
) -> np.ndarray:
    """Spectral subtraction with over-subtraction and gain floor."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 32:
        return x.copy()
    _, _, z = _stft(x, sample_rate)
    mag = np.abs(z)
    phase = np.exp(1j * np.angle(z))
    power = mag ** 2

    noise_power = _minimum_statistics_noise(power)
    noise_mag = np.sqrt(noise_power + 1e-10)

    enhanced_mag = np.maximum(mag - float(over_sub) * noise_mag, gain_floor * mag)
    nperseg = _nperseg(x, sample_rate)
    return _istft(enhanced_mag * phase, sample_rate, nperseg, len(x))


def wiener_filter(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Wiener filter with minimum-statistics noise estimation."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 32:
        return x.copy()
    _, _, z = _stft(x, sample_rate)
    power = np.abs(z) ** 2

    noise_power = _minimum_statistics_noise(power)
    gain = power / (power + float(strength) * noise_power + 1e-12)
    gain = np.clip(gain, 0.08, 1.0)

    nperseg = _nperseg(x, sample_rate)
    return _istft(z * gain, sample_rate, nperseg, len(x))


def speech_bandpass(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float = 80.0,
    high_hz: float = 7600.0,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    nyquist = sample_rate / 2.0
    low = max(20.0, float(low_hz))
    high = min(float(high_hz), nyquist * 0.95)
    if x.size < 16 or high <= low or nyquist <= low:
        return x.copy()
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    try:
        y = signal.sosfiltfilt(sos, x)
    except ValueError:
        y = signal.sosfilt(sos, x)
    return _fit_length(y, len(x))


def notch_50hz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Suppress 50 Hz electrical hum and a few harmonics."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 16 or sample_rate <= 120:
        return x.copy()
    y = x.astype(np.float64)
    for freq in (50.0, 100.0, 150.0, 200.0):
        if freq >= sample_rate / 2.0 - 5:
            continue
        b, a = signal.iirnotch(w0=freq, Q=10.0, fs=sample_rate)
        try:
            y = signal.filtfilt(b, a, y)
        except ValueError:
            y = signal.lfilter(b, a, y)
    return _fit_length(y, len(x))


def notch_60hz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Suppress 60 Hz electrical hum and harmonics (US/JP standard)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 16 or sample_rate <= 140:
        return x.copy()
    y = x.astype(np.float64)
    for freq in (60.0, 120.0, 180.0, 240.0):
        if freq >= sample_rate / 2.0 - 5:
            continue
        b, a = signal.iirnotch(w0=freq, Q=10.0, fs=sample_rate)
        try:
            y = signal.filtfilt(b, a, y)
        except ValueError:
            y = signal.lfilter(b, a, y)
    return _fit_length(y, len(x))


def highpass_80hz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """4th-order Butterworth high-pass at 80 Hz."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    nyquist = sample_rate / 2.0
    if x.size < 16 or nyquist <= 85:
        return x.copy()
    sos = signal.butter(4, 80.0, btype="highpass", fs=sample_rate, output="sos")
    try:
        y = signal.sosfiltfilt(sos, x)
    except ValueError:
        y = signal.sosfilt(sos, x)
    return _fit_length(y, len(x))


def log_mmse(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Ephraim-Malah log-MMSE speech estimator (1985).

    Uses minimum-statistics noise tracking and the decision-directed
    approach for a-priori SNR estimation.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 64:
        return x.copy()

    _, _, z = _stft(x, sample_rate)
    mag = np.abs(z)
    phase = np.exp(1j * np.angle(z))
    power = mag ** 2
    n_freq, n_frames = power.shape

    # Noise estimation via minimum statistics
    noise_power = _minimum_statistics_noise(power)

    # Decision-directed a-priori SNR estimation
    alpha_dd = 0.98  # decision-directed smoothing
    enhanced_power = np.zeros_like(power)
    prev_enhanced_mag = mag[:, 0].copy()

    for t in range(n_frames):
        # A-posteriori SNR
        gamma_t = power[:, t] / (noise_power[:, t] + 1e-10)
        gamma_t = np.clip(gamma_t, 1e-6, 1e6)

        # A-priori SNR (decision-directed)
        if t == 0:
            xi_t = np.maximum(gamma_t - 1.0, 0.01)
        else:
            xi_t = (alpha_dd * (prev_enhanced_mag ** 2) / (noise_power[:, t] + 1e-10)
                     + (1.0 - alpha_dd) * np.maximum(gamma_t - 1.0, 0.0))
        xi_t = np.maximum(xi_t, 1e-6)

        # Log-MMSE gain function
        nu_t = xi_t / (1.0 + xi_t) * gamma_t
        # Avoid overflow in exp1 for very large values
        nu_t_clipped = np.clip(nu_t, 1e-10, 500.0)

        gain = (xi_t / (1.0 + xi_t)) * np.exp(0.5 * exp1(nu_t_clipped))
        gain = np.clip(gain, 0.05, 5.0)  # safety clamp

        enhanced_mag = gain * mag[:, t]
        enhanced_power[:, t] = enhanced_mag ** 2
        prev_enhanced_mag = enhanced_mag

    enhanced = np.sqrt(np.maximum(enhanced_power, 0)) * phase
    nperseg = _nperseg(x, sample_rate)
    return _istft(enhanced, sample_rate, nperseg, len(x))


def median_declicker(
    audio: np.ndarray,
    sample_rate: int,
    kernel_ms: float = 5.0,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    """Remove transient clicks/claps via median filtering.

    1. Compute a local median envelope.
    2. Identify samples where |sample| > threshold_factor × local_median.
    3. Replace those samples with the local median value.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    kernel_samples = max(3, int(sample_rate * kernel_ms / 1000.0))
    # Ensure odd kernel
    if kernel_samples % 2 == 0:
        kernel_samples += 1

    if x.size < kernel_samples:
        return x.copy()

    # Local median of |x|
    abs_x = np.abs(x)
    local_median = signal.medfilt(abs_x, kernel_size=kernel_samples)

    # Detect transients
    threshold = threshold_factor * (local_median + 1e-10)
    transient_mask = abs_x > threshold

    # Replace transients with median-filtered signal
    median_signal = signal.medfilt(x, kernel_size=kernel_samples).astype(np.float32)
    y = x.copy()
    y[transient_mask] = median_signal[transient_mask]

    return _fit_length(y, len(x))


def bandpass_then_wiener(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Chain: speech bandpass → Wiener filter."""
    intermediate = speech_bandpass(audio, sample_rate)
    return wiener_filter(intermediate, sample_rate, strength=1.0)


# ─────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────

_DISPATCH = {
    "keep_original":        lambda audio, sr: np.asarray(audio, dtype=np.float32).copy(),
    "spectral_subtraction": spectral_subtraction,
    "wiener":               wiener_filter,
    "wiener_mild":          lambda audio, sr: wiener_filter(audio, sr, strength=0.5),
    "wiener_aggressive":    lambda audio, sr: wiener_filter(audio, sr, strength=2.0),
    "speech_bandpass":      speech_bandpass,
    "notch_50hz":           notch_50hz,
    "notch_60hz":           notch_60hz,
    "highpass_80hz":        highpass_80hz,
    "log_mmse":             log_mmse,
    "median_declicker":     median_declicker,
    "bandpass_then_wiener": bandpass_then_wiener,
}


def apply_filter(name: str, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply a named filter.

    Contract:
      - Returns the **same length** as input.
      - Returns approximately the **same RMS level** as input.
      - Contains no NaN or Inf.
    """
    if name not in _DISPATCH:
        raise ValueError(
            f"Unknown action '{name}'. Available: {', '.join(ACTIONS)}"
        )

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    y = _DISPATCH[name](x, sample_rate)

    # Enforce length
    y = _fit_length(y, len(x))

    # Enforce level match (skip for keep_original — it's already identical)
    if name != "keep_original":
        y = _level_match(y, x)

    # Sanitize
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y
