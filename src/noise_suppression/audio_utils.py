"""Audio loading, resampling, mixing, and metric utilities.

Extended from the original to include:
  - Digital silence trimming (for ESC-50 padding)
  - Fixed-level normalization
  - Level matching for fair metric comparison
  - Dual-noise mixing
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .constants import INTERNAL_SR


# ─────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────

def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return audio, int(sample_rate)


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    sf.write(str(path), x, int(sample_rate))


# ─────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────

def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    common = gcd(int(source_rate), int(target_rate))
    out = resample_poly(audio, target_rate // common, source_rate // common)
    return out.astype(np.float32)


def load_and_resample(path: str | Path, target_sr: int = INTERNAL_SR) -> np.ndarray:
    """Load an audio file and resample to *target_sr*."""
    audio, sr = read_audio(path)
    return resample_audio(audio, sr, target_sr)


# ─────────────────────────────────────────────────────────────────────
# Silence trimming  (for ESC-50 padded clips)
# ─────────────────────────────────────────────────────────────────────

def trim_digital_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -60.0,
    min_silence_ms: float = 50.0,
    margin_ms: float = 10.0,
) -> np.ndarray:
    """Trim leading/trailing near-digital-silence.

    This targets the exact-zero padding that ESC-50 clips often contain.
    A small margin is kept so transient onsets are not clipped.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 2:
        return x

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    frame_len = max(1, int(sr * min_silence_ms / 1000.0))
    margin_samples = max(0, int(sr * margin_ms / 1000.0))

    # Find first and last sample above threshold
    above = np.abs(x) > threshold_linear
    if not np.any(above):
        return x  # entire clip is silence — return as-is

    nonzero = np.nonzero(above)[0]
    start = max(0, nonzero[0] - margin_samples)
    end = min(len(x), nonzero[-1] + 1 + margin_samples)
    return x[start:end].copy()


# ─────────────────────────────────────────────────────────────────────
# Level normalization & matching
# ─────────────────────────────────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def normalize_level(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """Scale audio so its RMS equals *target_rms*.

    The default of 0.05 (~-26 dBFS) leaves comfortable headroom.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    current = rms(x)
    if current < 1e-8:
        return x  # near-silent — don't amplify noise floor
    return (x * (target_rms / current)).astype(np.float32)


def level_match(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Scale *estimate* to have the same RMS as *reference*.

    This prevents unfair metric scores from level differences.
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    ref_rms = np.sqrt(np.mean(ref * ref) + 1e-12)
    est_rms = np.sqrt(np.mean(est * est) + 1e-12)
    return (est * (ref_rms / est_rms)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Mixing
# ─────────────────────────────────────────────────────────────────────

def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix clean speech with a random segment of noise at a requested SNR."""
    clean = np.asarray(clean, dtype=np.float32).reshape(-1)
    noise = np.asarray(noise, dtype=np.float32).reshape(-1)
    if clean.size == 0 or noise.size == 0:
        raise ValueError("Clean speech and noise must both contain samples.")

    if noise.size < clean.size:
        repeats = int(np.ceil(clean.size / noise.size))
        noise = np.tile(noise, repeats)
    if noise.size > clean.size:
        max_start = noise.size - clean.size
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        noise = noise[start:start + clean.size]

    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if clean_rms < 1e-7:
        raise ValueError("Clean speech file has near-zero energy.")
    if noise_rms < 1e-7:
        raise ValueError("Noise file has near-zero energy.")

    desired_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (desired_noise_rms / noise_rms)
    return (clean + scaled_noise).astype(np.float32)


def mix_dual_noise(
    noise_a: np.ndarray,
    noise_b: np.ndarray,
    rng: np.random.Generator,
    ratio_db: float = 0.0,
) -> np.ndarray:
    """Mix two noise clips together at *ratio_db* relative level."""
    a = np.asarray(noise_a, dtype=np.float32).reshape(-1)
    b = np.asarray(noise_b, dtype=np.float32).reshape(-1)

    # Match lengths
    target_len = max(a.size, b.size)
    if a.size < target_len:
        a = np.tile(a, int(np.ceil(target_len / a.size)))[:target_len]
    if b.size < target_len:
        b = np.tile(b, int(np.ceil(target_len / b.size)))[:target_len]

    rms_a = rms(a)
    rms_b = rms(b)
    if rms_b > 1e-8:
        desired_b_rms = rms_a / (10.0 ** (ratio_db / 20.0))
        b = b * (desired_b_rms / rms_b)

    return (a + b).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Metrics (core SI-SDR — others are in metrics.py)
# ─────────────────────────────────────────────────────────────────────

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB; higher is generally better."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    if n == 0:
        return -100.0
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    denom = float(np.dot(ref, ref)) + 1e-12
    projection = (np.dot(est, ref) / denom) * ref
    noise = est - projection
    ratio = (np.dot(projection, projection) + 1e-12) / (np.dot(noise, noise) + 1e-12)
    value = 10.0 * np.log10(ratio)
    return float(np.clip(value, -100.0, 100.0))
