"""Build a reproducible manifest of speech-plus-noise mixtures.

Splits data with **zero leakage**:
  - ESC-50 uses its official 5-fold split (folds 1-3 train, 4 val, 5 test).
  - LibriSpeech is split by speaker ID so no voice appears in more than one split.

Usage:
    python -m noise_suppression.make_manifest \
        --librispeech-root path/to/LibriSpeech \
        --esc50-root path/to/ESC-50-master \
        --output data/manifest.csv \
        --check-leakage
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    DUAL_NOISE_FRACTION,
    INTERNAL_SR,
    MANIFEST_COLUMNS,
    SNR_RANGE_DB,
)

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

# ── ESC-50 categories to use as noise ────────────────────────────────
DEFAULT_NOISE_CATEGORIES = [
    "airplane", "car_horn", "clapping", "engine", "keyboard_typing",
    "rain", "vacuum_cleaner", "washing_machine", "wind",
]


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _list_audio(root: Path) -> list[Path]:
    """Recursively list audio files under *root*."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _speaker_id_from_path(path: Path) -> str:
    """Extract speaker ID from LibriSpeech path structure.

    Standard layout: ``<split>/<speaker>/<chapter>/<utterance.flac>``
    Fallback: use the immediate parent directory name.
    """
    parts = path.parts
    # Look for the pattern …/<speaker>/<chapter>/<file.flac>
    if len(parts) >= 3:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _deterministic_seed(mixture_id: str, global_seed: int) -> int:
    """Derive a per-mixture seed from the mixture ID + global seed."""
    h = hashlib.sha256(f"{global_seed}:{mixture_id}".encode()).hexdigest()
    return int(h[:8], 16)


def _split_speakers(
    speakers: list[str],
    rng: random.Random,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, str]:
    """Assign each speaker to exactly one split. Returns speaker→split."""
    shuffled = speakers.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    # Ensure test gets at least 1 speaker
    n_val = min(n_val, n - n_train - 1)
    if n_val < 1:
        n_val = 1
        n_train = n - 2
    mapping: dict[str, str] = {}
    for i, spk in enumerate(shuffled):
        if i < n_train:
            mapping[spk] = "train"
        elif i < n_train + n_val:
            mapping[spk] = "val"
        else:
            mapping[spk] = "test"
    return mapping


# ─────────────────────────────────────────────────────────────────────
# Manifest builder
# ─────────────────────────────────────────────────────────────────────

def build_manifest(
    librispeech_root: Path,
    esc50_root: Path,
    *,
    seed: int = 42,
    categories: list[str] | None = None,
    mixtures_per_clean: int = 3,
) -> pd.DataFrame:
    """Create a manifest DataFrame with zero-leakage splits."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    categories = categories or DEFAULT_NOISE_CATEGORIES

    # ── 1. Load ESC-50 metadata ──────────────────────────────────────
    esc_csv = esc50_root / "meta" / "esc50.csv"
    if not esc_csv.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {esc_csv}")

    esc_meta = pd.read_csv(esc_csv)
    esc_meta = esc_meta[esc_meta["category"].isin(categories)].copy()
    if esc_meta.empty:
        raise ValueError(f"No ESC-50 clips found for categories: {categories}")

    # Resolve audio paths and filter to existing files
    esc_meta["path"] = esc_meta["filename"].apply(
        lambda f: esc50_root / "audio" / f
    )
    esc_meta = esc_meta[esc_meta["path"].apply(lambda p: p.is_file())].copy()

    # Split by official folds: 1-3 train, 4 val, 5 test
    fold_to_split = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
    esc_meta["split"] = esc_meta["fold"].map(fold_to_split)

    # Build per-split noise pools
    noise_pool: dict[str, list[dict]] = defaultdict(list)
    for _, row in esc_meta.iterrows():
        noise_pool[row["split"]].append({
            "path": str(row["path"]),
            "category": row["category"],
            "fold": int(row["fold"]),
        })

    # ── 2. Load LibriSpeech files & split by speaker ─────────────────
    clean_files = _list_audio(librispeech_root)
    if len(clean_files) < 2:
        raise ValueError(
            f"Found {len(clean_files)} clean files in {librispeech_root}; "
            "need at least 2."
        )

    # Group by speaker
    files_by_speaker: dict[str, list[Path]] = defaultdict(list)
    for p in clean_files:
        spk = _speaker_id_from_path(p)
        files_by_speaker[spk].append(p)

    speaker_split = _split_speakers(list(files_by_speaker.keys()), rng)

    # Build per-split clean pools
    clean_pool: dict[str, list[dict]] = defaultdict(list)
    for spk, files in files_by_speaker.items():
        split = speaker_split[spk]
        for f in files:
            clean_pool[split].append({"path": str(f), "speaker_id": spk})

    # ── 3. Generate mixture rows ─────────────────────────────────────
    rows: list[dict] = []
    mixture_counter = 0

    for split in ("train", "val", "test"):
        cleans = clean_pool.get(split, [])
        noises = noise_pool.get(split, [])
        if not cleans or not noises:
            print(f"WARNING: split '{split}' has {len(cleans)} clean, "
                  f"{len(noises)} noise files — skipping")
            continue

        for clean_info in cleans:
            for _ in range(mixtures_per_clean):
                mixture_counter += 1
                mix_id = f"mix_{mixture_counter:06d}"
                snr_db = float(rng.choice(SNR_RANGE_DB))
                mseed = _deterministic_seed(mix_id, seed)

                # Pick primary noise
                noise_info = rng.choice(noises)

                # Decide if dual-noise
                is_dual = (rng.random() < DUAL_NOISE_FRACTION
                           and len(noises) >= 2)
                noise_info_2: dict | None = None
                if is_dual:
                    others = [n for n in noises
                              if n["path"] != noise_info["path"]]
                    if others:
                        noise_info_2 = rng.choice(others)
                    else:
                        is_dual = False

                rows.append({
                    "mixture_id": mix_id,
                    "split": split,
                    "clean_file": clean_info["path"],
                    "noise_file": noise_info["path"],
                    "noise_file_2": noise_info_2["path"] if is_dual else "",
                    "noise_category": noise_info["category"],
                    "noise_category_2": (noise_info_2["category"]
                                         if is_dual else ""),
                    "esc50_fold": noise_info["fold"],
                    "snr_db": snr_db,
                    "seed": mseed,
                    "speaker_id": clean_info["speaker_id"],
                })

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return df


# ─────────────────────────────────────────────────────────────────────
# Leakage check
# ─────────────────────────────────────────────────────────────────────

def check_leakage(df: pd.DataFrame) -> bool:
    """Assert zero leakage across splits. Returns True if clean."""
    ok = True

    # Speaker leakage
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 >= s2:
                continue
            spk1 = set(df.loc[df["split"] == s1, "speaker_id"])
            spk2 = set(df.loc[df["split"] == s2, "speaker_id"])
            overlap = spk1 & spk2
            if overlap:
                print(f"LEAKAGE: speakers in both {s1} & {s2}: {overlap}")
                ok = False

    # ESC-50 fold leakage
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 >= s2:
                continue
            f1 = set(df.loc[df["split"] == s1, "esc50_fold"])
            f2 = set(df.loc[df["split"] == s2, "esc50_fold"])
            overlap = f1 & f2
            if overlap:
                print(f"LEAKAGE: ESC-50 folds in both {s1} & {s2}: {overlap}")
                ok = False

    if ok:
        print("[OK] No speaker leakage across splits.")
        print("[OK] No ESC-50 fold leakage across splits.")

    # Summary
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n_spk = sub["speaker_id"].nunique()
        n_cat = sub["noise_category"].nunique()
        print(f"  {split:6s}: {len(sub):5d} mixtures, "
              f"{n_spk:3d} speakers, {n_cat:2d} noise categories")

    return ok


def build_manifest_from_folders(
    clean_dir: Path,
    noise_dir: Path,
    *,
    seed: int = 42,
    mixtures_per_clean: int = 3,
) -> pd.DataFrame:
    """Create a manifest DataFrame from clean and noise audio directories with zero-leakage splits."""
    rng = random.Random(seed)

    clean_files = _list_audio(clean_dir)
    noise_files = _list_audio(noise_dir)

    if not clean_files:
        raise FileNotFoundError(f"No clean audio found in {clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No noise audio found in {noise_dir}")

    # Group clean files by speaker (stem prefix or parent)
    files_by_speaker: dict[str, list[Path]] = defaultdict(list)
    for p in clean_files:
        spk = p.stem.split("_")[0] if "_" in p.stem else p.parent.name
        files_by_speaker[spk].append(p)

    speaker_split = _split_speakers(list(files_by_speaker.keys()), rng)

    clean_pool: dict[str, list[dict]] = defaultdict(list)
    for spk, files in files_by_speaker.items():
        split = speaker_split[spk]
        for f in files:
            clean_pool[split].append({"path": str(f), "speaker": spk})

    # Noise pool: assign noise files to splits (train/val/test) with distinct folds
    shuffled_noise = noise_files.copy()
    rng.shuffle(shuffled_noise)
    noise_pool: dict[str, list[dict]] = defaultdict(list)
    for i, nf in enumerate(shuffled_noise):
        cat = nf.stem.split("_")[0] if "_" in nf.stem else nf.stem
        # Map to folds: 1-3 for train, 4 for val, 5 for test
        if i % 5 == 3:
            s, fold = "val", 4
        elif i % 5 == 4:
            s, fold = "test", 5
        else:
            s, fold = "train", (i % 3) + 1
        noise_pool[s].append({"path": str(nf), "category": cat, "fold": fold})

    # Ensure each split has noise
    for s in ("train", "val", "test"):
        if not noise_pool[s]:
            fallback = shuffled_noise[0]
            fold = 4 if s == "val" else (5 if s == "test" else 1)
            noise_pool[s].append({
                "path": str(fallback),
                "category": fallback.stem,
                "fold": fold,
            })

    rows = []
    mix_counter = 0

    for split in ("train", "val", "test"):
        c_items = clean_pool[split]
        n_items = noise_pool[split]
        if not c_items or not n_items:
            continue

        for c_entry in c_items:
            clean_path = c_entry["path"]
            spk = c_entry["speaker"]

            for _ in range(mixtures_per_clean):
                mix_counter += 1
                mix_id = f"mix_{mix_counter:05d}"
                n1 = rng.choice(n_items)
                use_dual = (split == "train") and (rng.random() < DUAL_NOISE_FRACTION) and len(n_items) > 1
                if use_dual:
                    n2 = rng.choice([x for x in n_items if x["path"] != n1["path"]] or [n1])
                    n2_path = n2["path"]
                    n2_cat = n2["category"]
                else:
                    n2_path = ""
                    n2_cat = ""

                snr = float(rng.choice(SNR_RANGE_DB))
                p_seed = _deterministic_seed(mix_id, seed)

                rows.append({
                    "mixture_id": mix_id,
                    "split": split,
                    "clean_file": clean_path,
                    "noise_file": n1["path"],
                    "noise_file_2": n2_path,
                    "noise_category": n1["category"],
                    "noise_category_2": n2_cat,
                    "esc50_fold": n1["fold"],
                    "snr_db": snr,
                    "seed": p_seed,
                    "speaker_id": spk,
                })

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible manifest of speech+noise mixtures."
    )
    parser.add_argument("--librispeech-root", type=Path, default=None,
                        help="Path to LibriSpeech (or a split like dev-clean)")
    parser.add_argument("--esc50-root", type=Path, default=None,
                        help="Path to ESC-50-master")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"),
                        help="Path to clean speech directory (used if public datasets not specified)")
    parser.add_argument("--noise-dir", type=Path, default=Path("data/noise"),
                        help="Path to noise directory (used if public datasets not specified)")
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixtures-per-clean", type=int, default=3,
                        help="Number of random noise pairings per clean clip")
    parser.add_argument("--categories", nargs="+",
                        default=DEFAULT_NOISE_CATEGORIES)
    parser.add_argument("--check-leakage", action="store_true",
                        help="Run leakage assertions after building")
    args = parser.parse_args()

    if args.librispeech_root is not None and args.esc50_root is not None:
        print(f"LibriSpeech root: {args.librispeech_root}")
        print(f"ESC-50 root:      {args.esc50_root}")
        df = build_manifest(
            librispeech_root=args.librispeech_root.expanduser().resolve(),
            esc50_root=args.esc50_root.expanduser().resolve(),
            seed=args.seed,
            categories=args.categories,
            mixtures_per_clean=args.mixtures_per_clean,
        )
    else:
        print(f"Building manifest from folders: clean={args.clean_dir}, noise={args.noise_dir}")
        df = build_manifest_from_folders(
            clean_dir=args.clean_dir,
            noise_dir=args.noise_dir,
            seed=args.seed,
            mixtures_per_clean=args.mixtures_per_clean,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nManifest saved to: {args.output}  ({len(df)} rows)")

    if args.check_leakage:
        print("\n-- Leakage check --")
        clean = check_leakage(df)
        if not clean:
            raise SystemExit("LEAKAGE DETECTED — fix splits before proceeding.")


if __name__ == "__main__":
    main()
