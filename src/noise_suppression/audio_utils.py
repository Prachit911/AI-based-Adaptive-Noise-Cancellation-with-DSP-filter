"""Audio loading, resampling, mixing, and metric utilities."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


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


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    common = gcd(int(source_rate), int(target_rate))
    out = resample_poly(audio, target_rate // common, source_rate // common)
    return out.astype(np.float32)


def rms(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
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
    # Preserve float headroom: do not peak-normalize the mixture, because that
    # would change the SNR and the reference/mixture relationship.
    return (clean + scaled_noise).astype(np.float32)


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
