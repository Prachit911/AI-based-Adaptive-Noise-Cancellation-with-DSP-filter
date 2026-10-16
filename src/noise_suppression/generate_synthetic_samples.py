"""Generate realistic synthetic clean speech and noise audio files for quick start, testing, and training.

This utility creates:
  1. High-quality synthetic clean speech clips with realistic formant synthesis,
     harmonic pitch contours, syllable envelopes, and speech-band resonance.
  2. Diverse environmental noise recordings:
     - 50 Hz and 60 Hz electrical hum with harmonics
     - Engine and motor low-frequency rumble
     - Stationary white noise
     - Impulsive clapping transients
     - Keyboard typing clatter
     - Wind turbulence
  3. Pre-populates data/clean, data/noise, data/eval_clean, and data/eval_noise.
  4. Generates a default noisy test sample at `noisy.wav`.

Usage:
    python -m noise_suppression.generate_synthetic_samples
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy import signal

from .audio_utils import write_audio, mix_at_snr
from .constants import INTERNAL_SR


def _synthesize_vocalic_syllable(
    duration_s: float,
    f0: float,
    formants: list[float],
    sr: int = INTERNAL_SR,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesize a voiced vowel sound via formant-filtered glottal pulse train."""
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)

    # Glottal pulse approximation: harmonics with 1/n rolloff + slight pitch vibrato
    pitch_contour = f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
    source = np.zeros(n_samples, dtype=np.float32)
    for harmonic in range(1, 25):
        if harmonic * f0 < sr / 2.0 - 100:
            source += (1.0 / harmonic) * np.sin(harmonic * phase)

    # Filter through 3 vocal tract formant resonances
    filtered = source.astype(np.float64)
    for f_center in formants:
        if f_center < sr / 2.0 - 50:
            bw = f_center * 0.12  # formant bandwidth
            b, a = signal.iirpeak(w0=f_center, Q=f_center / max(30.0, bw), fs=sr)
            filtered = signal.lfilter(b, a, filtered)

    # Smooth trapezoidal / Hanning envelope for syllable onset and offset
    env = signal.windows.tukey(n_samples, alpha=0.35)
    syllable = (filtered * env).astype(np.float32)

    # Normalize amplitude
    peak = np.max(np.abs(syllable)) + 1e-8
    return 0.6 * (syllable / peak)


def synthesize_speech_utterance(
    duration_s: float = 2.5,
    pitch_base: float = 140.0,
    sr: int = INTERNAL_SR,
    seed: int = 42,
) -> np.ndarray:
    """Generate a multi-syllable speech-like utterance with natural rhythm and pauses."""
    rng = np.random.default_rng(seed)

    # Vowel formant profiles [F1, F2, F3]
    vowels = [
        [730.0, 1090.0, 2440.0],  # /a/
        [270.0, 2290.0, 3010.0],  # /i/
        [530.0, 1840.0, 2480.0],  # /e/
        [300.0, 870.0, 2240.0],   # /u/
        [570.0, 840.0, 2410.0],   # /o/
    ]

    segments = []
    elapsed = 0.0

    while elapsed < duration_s:
        # Syllable length 0.2s - 0.4s
        syl_len = float(rng.uniform(0.18, 0.38))
        f_profile = rng.choice(vowels)
        f0 = pitch_base * float(rng.uniform(0.92, 1.08))
        syllable = _synthesize_vocalic_syllable(syl_len, f0, f_profile, sr, rng)
        segments.append(syllable)
        elapsed += syl_len

        # Short pause between syllables or words (0.04s - 0.12s)
        if elapsed < duration_s:
            pause_len = float(rng.uniform(0.04, 0.12))
            pause = np.zeros(int(pause_len * sr), dtype=np.float32)
            segments.append(pause)
            elapsed += pause_len

    speech = np.concatenate(segments)
    target_samples = int(duration_s * sr)
    if len(speech) > target_samples:
        speech = speech[:target_samples]
    elif len(speech) < target_samples:
        speech = np.pad(speech, (0, target_samples - len(speech)))

    # Apply 80 Hz highpass to eliminate sub-bass DC drift
    sos = signal.butter(2, 80.0, btype="highpass", fs=sr, output="sos")
    speech = signal.sosfiltfilt(sos, speech).astype(np.float32)

    peak = np.max(np.abs(speech)) + 1e-8
    return 0.7 * (speech / peak)


def synthesize_hum_noise(
    freq_hz: float = 50.0,
    duration_s: float = 3.0,
    sr: int = INTERNAL_SR,
    seed: int = 42,
) -> np.ndarray:
    """Generate electrical hum noise (50 Hz or 60 Hz with odd and even harmonics)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)

    hum = np.sin(2 * np.pi * freq_hz * t)
    # Harmonics: 2x, 3x, 4x, 5x
    hum += 0.35 * np.sin(2 * np.pi * (2 * freq_hz) * t + 0.5)
    hum += 0.20 * np.sin(2 * np.pi * (3 * freq_hz) * t + 1.2)
    hum += 0.08 * np.sin(2 * np.pi * (4 * freq_hz) * t + 2.0)

    # Faint background noise floor
    hum += 0.02 * rng.normal(0, 1.0, size=n).astype(np.float32)

    peak = np.max(np.abs(hum)) + 1e-8
    return 0.5 * (hum / peak)


def synthesize_white_noise(
    duration_s: float = 3.0,
    sr: int = INTERNAL_SR,
    seed: int = 42,
) -> np.ndarray:
    """Generate flat Gaussian white noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    noise = rng.normal(0, 0.25, size=n).astype(np.float32)
    peak = np.max(np.abs(noise)) + 1e-8
    return 0.5 * (noise / peak)


def synthesize_engine_rumble(
    duration_s: float = 3.0,
    sr: int = INTERNAL_SR,
    seed: int = 42,
) -> np.ndarray:
    """Generate low-frequency stationary engine/motor rumble."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    white = rng.normal(0, 1.0, size=n).astype(np.float32)

    # 4th-order lowpass filter below 220 Hz
    sos = signal.butter(4, 220.0, btype="lowpass", fs=sr, output="sos")
    rumble = signal.sosfilt(sos, white)

    # Add 45 Hz and 90 Hz mechanical rotational components
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    rumble += 0.3 * np.sin(2 * np.pi * 45 * t) + 0.15 * np.sin(2 * np.pi * 90 * t)

    peak = np.max(np.abs(rumble)) + 1e-8
    return 0.5 * (rumble.astype(np.float32) / peak)


def synthesize_impulsive_clicks(
    duration_s: float = 3.0,
    sr: int = INTERNAL_SR,
    density_per_sec: float = 6.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate impulsive transient clicks (clapping / keyboard clatter)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    x = np.zeros(n, dtype=np.float32)

    n_clicks = int(duration_s * density_per_sec)
    click_positions = rng.integers(100, n - 2000, size=n_clicks)

    decay_t = np.linspace(0, 0.015, int(0.015 * sr), endpoint=False, dtype=np.float32)
    decay_kernel = np.exp(-decay_t * 300.0) * np.sin(2 * np.pi * 2500 * decay_t)

    for pos in click_positions:
        amp = float(rng.uniform(0.5, 1.0))
        end = min(n, pos + len(decay_kernel))
        k_len = end - pos
        x[pos:end] += amp * decay_kernel[:k_len]

    # Add very faint background noise
    x += 0.01 * rng.normal(0, 0.1, size=n).astype(np.float32)

    peak = np.max(np.abs(x)) + 1e-8
    return 0.7 * (x / peak)


def synthesize_wind_noise(
    duration_s: float = 3.0,
    sr: int = INTERNAL_SR,
    seed: int = 42,
) -> np.ndarray:
    """Generate slowly fluctuating aerodynamic wind noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    white = rng.normal(0, 1.0, size=n).astype(np.float32)

    # Bandpass 50 - 500 Hz
    sos = signal.butter(3, [50.0, 500.0], btype="bandpass", fs=sr, output="sos")
    wind = signal.sosfilt(sos, white)

    # Slow gust modulation (0.3 - 0.7 Hz)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    gust = 0.5 * (1.0 + np.sin(2 * np.pi * 0.4 * t) * np.cos(2 * np.pi * 0.15 * t))
    wind = wind * gust

    peak = np.max(np.abs(wind)) + 1e-8
    return 0.5 * (wind.astype(np.float32) / peak)


def generate_all_samples(
    clean_dir: Path = Path("data/clean"),
    noise_dir: Path = Path("data/noise"),
    eval_clean_dir: Path = Path("data/eval_clean"),
    eval_noise_dir: Path = Path("data/eval_noise"),
    noisy_wav_path: Path = Path("noisy.wav"),
    seed: int = 42,
) -> None:
    """Generate full set of synthetic clean, noise, evaluation, and test audio files."""
    sr = INTERNAL_SR

    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    eval_clean_dir.mkdir(parents=True, exist_ok=True)
    eval_noise_dir.mkdir(parents=True, exist_ok=True)

    print("Synthesizing training clean speech files...")
    # 5 training clean clips with distinct speaker profiles
    clean_configs = [
        ("speaker1_male_clip1.wav", 130.0, 2.5, 101),
        ("speaker1_male_clip2.wav", 135.0, 2.8, 102),
        ("speaker2_female_clip1.wav", 220.0, 2.4, 201),
        ("speaker2_female_clip2.wav", 215.0, 2.7, 202),
        ("speaker3_child_clip1.wav", 290.0, 2.2, 301),
    ]
    sample_clean = None
    for name, pitch, dur, s in clean_configs:
        audio = synthesize_speech_utterance(duration_s=dur, pitch_base=pitch, sr=sr, seed=s)
        p = clean_dir / name
        write_audio(p, audio, sr)
        if sample_clean is None:
            sample_clean = audio

    print("Synthesizing evaluation clean speech files...")
    # 3 held-out evaluation speech clips (distinct speakers)
    eval_clean_configs = [
        ("eval_speaker4_deepmale.wav", 110.0, 2.5, 401),
        ("eval_speaker5_highfemale.wav", 240.0, 2.6, 501),
        ("eval_speaker6_medium.wav", 175.0, 2.4, 601),
    ]
    for name, pitch, dur, s in eval_clean_configs:
        audio = synthesize_speech_utterance(duration_s=dur, pitch_base=pitch, sr=sr, seed=s)
        write_audio(eval_clean_dir / name, audio, sr)

    print("Synthesizing training noise clips...")
    # Diverse noise types
    noise_generators = [
        ("hum_50hz.wav", lambda: synthesize_hum_noise(50.0, 3.0, sr, seed=11)),
        ("hum_60hz.wav", lambda: synthesize_hum_noise(60.0, 3.0, sr, seed=12)),
        ("engine.wav", lambda: synthesize_engine_rumble(3.0, sr, seed=13)),
        ("white_noise.wav", lambda: synthesize_white_noise(3.0, sr, seed=14)),
        ("clapping.wav", lambda: synthesize_impulsive_clicks(3.0, sr, density_per_sec=8.0, seed=15)),
        ("keyboard_typing.wav", lambda: synthesize_impulsive_clicks(3.0, sr, density_per_sec=12.0, seed=16)),
        ("wind.wav", lambda: synthesize_wind_noise(3.0, sr, seed=17)),
    ]
    sample_noise = None
    for name, gen_fn in noise_generators:
        audio = gen_fn()
        write_audio(noise_dir / name, audio, sr)
        if "hum_50hz" in name:
            sample_noise = audio

    print("Synthesizing evaluation noise clips...")
    eval_noise_generators = [
        ("eval_hum_50hz.wav", lambda: synthesize_hum_noise(50.0, 3.0, sr, seed=81)),
        ("eval_engine.wav", lambda: synthesize_engine_rumble(3.0, sr, seed=82)),
        ("eval_clapping.wav", lambda: synthesize_impulsive_clicks(3.0, sr, density_per_sec=7.0, seed=83)),
        ("eval_wind.wav", lambda: synthesize_wind_noise(3.0, sr, seed=84)),
    ]
    for name, gen_fn in eval_noise_generators:
        audio = gen_fn()
        write_audio(eval_noise_dir / name, audio, sr)

    # Create root noisy.wav for quick CLI testing
    if sample_clean is not None and sample_noise is not None:
        rng = np.random.default_rng(seed)
        # Mix clean speech + 50Hz hum at 5.0 dB SNR
        sample_noisy = mix_at_snr(sample_clean, sample_noise, snr_db=5.0, rng=rng)
        write_audio(noisy_wav_path, sample_noisy, sr)
        print(f"Created default test clip: {noisy_wav_path}")

    print("\n[OK] Synthetic audio dataset generation complete!")
    print(f"  Clean speech (train): {clean_dir} ({len(clean_configs)} clips)")
    print(f"  Noise (train):        {noise_dir} ({len(noise_generators)} clips)")
    print(f"  Clean speech (eval):  {eval_clean_dir} ({len(eval_clean_configs)} clips)")
    print(f"  Noise (eval):         {eval_noise_dir} ({len(eval_noise_generators)} clips)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic clean and noise audio files for the project.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--noise-dir", type=Path, default=Path("data/noise"))
    parser.add_argument("--eval-clean-dir", type=Path, default=Path("data/eval_clean"))
    parser.add_argument("--eval-noise-dir", type=Path, default=Path("data/eval_noise"))
    parser.add_argument("--noisy-wav", type=Path, default=Path("noisy.wav"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_all_samples(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        eval_clean_dir=args.eval_clean_dir,
        eval_noise_dir=args.eval_noise_dir,
        noisy_wav_path=args.noisy_wav,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
