"""Shared constants for the noise-suppression pipeline.

Every module imports from here so that sample rate, feature version,
SNR list, and column names stay in sync across the entire project.
"""
from __future__ import annotations

# ── Audio ────────────────────────────────────────────────────────────
INTERNAL_SR = 16_000          # single internal sample rate (Hz)
MAX_CLIP_SECONDS = 12         # cap long recordings during training

# ── Mixing ───────────────────────────────────────────────────────────
SNR_RANGE_DB = [-5, 0, 5, 10, 15, 20]
DUAL_NOISE_FRACTION = 0.10    # 10 % of mixtures use two noise clips

# ── Features ─────────────────────────────────────────────────────────
FEATURE_VERSION = "v2"

# ── Manifest columns ────────────────────────────────────────────────
MANIFEST_COLUMNS = [
    "mixture_id",
    "split",               # train / val / test
    "clean_file",
    "noise_file",
    "noise_file_2",        # second noise clip (empty for single-noise)
    "noise_category",
    "noise_category_2",
    "esc50_fold",
    "snr_db",
    "seed",
    "speaker_id",
]

# ── Evaluation ───────────────────────────────────────────────────────
RESULTS_COLUMNS = [
    "mixture_id", "split", "noise_category", "snr_db",
    "method", "si_sdr_improvement_db", "stoi", "pesq_wb",
]

# ── Selector ─────────────────────────────────────────────────────────
SAFETY_THRESHOLD_DB = 0.05   # if best predicted gain < this → keep_original
