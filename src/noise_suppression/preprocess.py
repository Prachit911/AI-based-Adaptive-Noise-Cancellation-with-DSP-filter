"""Preprocess raw audio and build mixture WAV files from a manifest.

Steps for each audio file:
  1. Load → mono.
  2. Resample to INTERNAL_SR (16 kHz).
  3. ESC-50 clips: trim leading/trailing digital silence.
  4. Apply fixed-level normalization.
  5. Save processed clip.

Steps for each manifest row:
  1. Load processed clean and noise.
  2. (Optional) mix two noise clips for dual-noise rows.
  3. Mix at the specified SNR using the row's deterministic seed.
  4. Save mixture and clean reference side-by-side.

Usage:
    python -m noise_suppression.preprocess \
        --manifest data/manifest.csv \
        --output-dir data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .audio_utils import (
    load_and_resample,
    mix_at_snr,
    mix_dual_noise,
    normalize_level,
    trim_digital_silence,
    write_audio,
)
from .constants import INTERNAL_SR


def _is_esc50_path(path: str) -> bool:
    """Heuristic: ESC-50 files live under an 'audio' directory."""
    return "ESC-50" in path or "/audio/" in path.replace("\\", "/")


def preprocess_single_file(
    src: Path,
    dst: Path,
    is_noise: bool = False,
) -> None:
    """Load, resample, trim (if noise), normalize, and save."""
    if dst.exists():
        return  # already processed

    audio = load_and_resample(src, INTERNAL_SR)

    # Trim digital silence from ESC-50 noise clips
    if is_noise:
        audio = trim_digital_silence(audio, INTERNAL_SR)

    # Normalize level
    audio = normalize_level(audio, target_rms=0.05)

    write_audio(dst, audio, INTERNAL_SR)


def build_mixtures(
    manifest: pd.DataFrame,
    processed_dir: Path,
    mixtures_dir: Path,
) -> None:
    """Create mixture and reference WAV files from a processed manifest."""
    clean_dir = processed_dir / "clean"
    noise_dir = processed_dir / "noise"
    refs_dir = mixtures_dir / "references"

    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc="Building mixtures"):
        mix_id = row["mixture_id"]
        split = row["split"]
        out_path = mixtures_dir / split / f"{mix_id}.wav"
        ref_path = refs_dir / split / f"{mix_id}_ref.wav"

        if out_path.exists() and ref_path.exists():
            continue  # already built

        # Load processed clean
        clean_name = Path(row["clean_file"]).stem + ".wav"
        clean = load_and_resample(clean_dir / clean_name, INTERNAL_SR)

        # Load processed noise
        noise_name = Path(row["noise_file"]).stem + ".wav"
        noise = load_and_resample(noise_dir / noise_name, INTERNAL_SR)

        # Dual noise mixing
        nf2 = row.get("noise_file_2")
        if pd.notna(nf2) and str(nf2).strip() and str(nf2).strip().lower() != "nan":
            noise_name_2 = Path(str(nf2)).stem + ".wav"
            noise_2 = load_and_resample(noise_dir / noise_name_2, INTERNAL_SR)
            rng2 = np.random.default_rng(int(row["seed"]) + 1)
            noise = mix_dual_noise(noise, noise_2, rng2, ratio_db=0.0)

        # Mix at specified SNR with deterministic seed
        rng = np.random.default_rng(int(row["seed"]))
        snr_db = float(row["snr_db"])
        mixture = mix_at_snr(clean, noise, snr_db, rng)

        # Save
        write_audio(out_path, mixture, INTERNAL_SR)
        write_audio(ref_path, clean, INTERNAL_SR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw audio and build mixtures from a manifest."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--internal-sr", type=int, default=INTERNAL_SR)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    processed_dir = args.output_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Preprocess all unique clean files ────────────────────
    clean_dir = processed_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    unique_clean = manifest["clean_file"].unique()
    print(f"Preprocessing {len(unique_clean)} unique clean files...")
    for src_str in tqdm(unique_clean, desc="Clean files"):
        src = Path(src_str)
        dst = clean_dir / (src.stem + ".wav")
        preprocess_single_file(src, dst, is_noise=False)

    # ── Step 2: Preprocess all unique noise files ────────────────────
    noise_dir = processed_dir / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    noise_cols = ["noise_file"]
    if "noise_file_2" in manifest.columns:
        noise_cols.append("noise_file_2")
    unique_noise = set()
    for col in noise_cols:
        for v in manifest[col].dropna():
            if str(v).strip():
                unique_noise.add(str(v))
    print(f"Preprocessing {len(unique_noise)} unique noise files...")
    for src_str in tqdm(sorted(unique_noise), desc="Noise files"):
        src = Path(src_str)
        dst = noise_dir / (src.stem + ".wav")
        preprocess_single_file(src, dst, is_noise=True)

    # ── Step 3: Build mixtures ───────────────────────────────────────
    mixtures_dir = processed_dir / "mixtures"
    print(f"Building {len(manifest)} mixtures...")
    build_mixtures(manifest, processed_dir, mixtures_dir)

    # Summary
    for split in ("train", "val", "test"):
        n = (manifest["split"] == split).sum()
        print(f"  {split}: {n} mixtures")
    print(f"\nDone. Processed files in: {processed_dir}")


if __name__ == "__main__":
    main()
