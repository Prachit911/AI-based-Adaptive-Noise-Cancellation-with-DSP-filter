"""Evaluation metrics for speech enhancement: SI-SDR, STOI, PESQ.

PESQ is optional (requires C compiler to install). If unavailable,
the metric returns NaN and a warning is printed once.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from .audio_utils import si_sdr, level_match


# ─────────────────────────────────────────────────────────────────────
# Optional imports (graceful degradation)
# ─────────────────────────────────────────────────────────────────────

def _try_import_stoi():
    try:
        from pystoi import stoi as _stoi
        return _stoi
    except ImportError:
        warnings.warn("pystoi not installed — STOI metric will return NaN.", stacklevel=2)
        return None


def _try_import_pesq():
    try:
        from pesq import pesq as _pesq
        return _pesq
    except ImportError:
        warnings.warn("pesq not installed — PESQ metric will return NaN.", stacklevel=2)
        return None


@lru_cache(maxsize=1)
def _get_stoi_fn():
    return _try_import_stoi()


@lru_cache(maxsize=1)
def _get_pesq_fn():
    return _try_import_pesq()


# ─────────────────────────────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────────────────────────────

def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB."""
    return si_sdr(reference, estimate)


def compute_stoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    extended: bool = False,
) -> float:
    """Short-Time Objective Intelligibility.

    Returns a value in [0, 1]; higher = more intelligible.
    """
    stoi_fn = _get_stoi_fn()
    if stoi_fn is None:
        return float("nan")

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(ref.size, est.size)
    if n == 0:
        return float("nan")
    ref, est = ref[:n], est[:n]

    try:
        return float(stoi_fn(ref, est, sr, extended=extended))
    except Exception:
        return float("nan")


def compute_pesq(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    mode: str = "wb",
) -> float:
    """Perceptual Evaluation of Speech Quality.

    Returns a value in [1.0, 4.5] (wideband); higher = better.
    Requires sr = 16000 for wideband mode.
    """
    pesq_fn = _get_pesq_fn()
    if pesq_fn is None:
        return float("nan")

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(ref.size, est.size)
    if n == 0:
        return float("nan")
    ref, est = ref[:n], est[:n]

    try:
        return float(pesq_fn(sr, ref, est, mode))
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────
# Convenience: compute all metrics at once
# ─────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """Compute all evaluation metrics for one (clean, noisy, enhanced) triple.

    Returns a dict with keys:
      - si_sdr_input_db:        SI-SDR of noisy vs clean
      - si_sdr_output_db:       SI-SDR of enhanced vs clean
      - si_sdr_improvement_db:  output − input
      - stoi_input:             STOI of noisy
      - stoi_output:            STOI of enhanced
      - stoi_improvement:       output − input
      - pesq_input:             PESQ of noisy
      - pesq_output:            PESQ of enhanced
      - pesq_improvement:       output − input
    """
    # Ensure same length
    n = min(len(clean), len(noisy), len(enhanced))
    clean = np.asarray(clean, dtype=np.float32)[:n]
    noisy = np.asarray(noisy, dtype=np.float32)[:n]
    enhanced = np.asarray(enhanced, dtype=np.float32)[:n]

    # Level-match enhanced to noisy for fair comparison
    enhanced = level_match(noisy, enhanced)

    # SI-SDR
    si_sdr_in = compute_si_sdr(clean, noisy)
    si_sdr_out = compute_si_sdr(clean, enhanced)

    # STOI
    stoi_in = compute_stoi(clean, noisy, sr)
    stoi_out = compute_stoi(clean, enhanced, sr)

    # PESQ
    pesq_in = compute_pesq(clean, noisy, sr)
    pesq_out = compute_pesq(clean, enhanced, sr)

    return {
        "si_sdr_input_db": si_sdr_in,
        "si_sdr_output_db": si_sdr_out,
        "si_sdr_improvement_db": si_sdr_out - si_sdr_in,
        "stoi_input": stoi_in,
        "stoi_output": stoi_out,
        "stoi_improvement": stoi_out - stoi_in,
        "pesq_input": pesq_in,
        "pesq_output": pesq_out,
        "pesq_improvement": pesq_out - pesq_in,
    }
