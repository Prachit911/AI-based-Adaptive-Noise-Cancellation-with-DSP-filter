"""Robustly prepare LibriSpeech + ESC-50 for the noise-suppression project.

Run from the project root (C:\\ai_noise_suppression):
    python prepare_public_datasets_fixed.py --reset

The script auto-detects datasets in the project root or its src/ folder.
You can also pass --librispeech-root and --esc50-root explicitly.
--reset clears only generated audio inside project data/{clean,eval_clean,noise,eval_noise};
it never deletes/moves the original LibriSpeech or ESC-50 downloads.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
DEFAULT_CATEGORIES = [
    "airplane", "car_horn", "clapping", "engine", "keyboard_typing",
    "rain", "vacuum_cleaner", "washing_machine", "wind",
]
OUTPUT_NAMES = ("clean", "eval_clean", "noise", "eval_noise")


def list_audio(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def is_esc50_root(path: Path) -> bool:
    return (path / "audio").is_dir() and (path / "meta" / "esc50.csv").is_file()


def is_librispeech_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    # Avoid accidentally treating the entire project/src directory as the dataset.
    has_expected_layout = path.name.lower() in {"librispeech", "dev-clean-2", "dev-clean", "train-clean-100", "train-clean-360", "train-clean-500"}
    has_split_folder = any((path / name).is_dir() for name in ("dev-clean-2", "dev-clean", "train-clean-100", "train-clean-360", "train-clean-500"))
    return (has_expected_layout or has_split_folder) and any(path.rglob("*.flac"))


def detect_root(explicit: Path | None, project_root: Path, kind: str) -> Path:
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        valid = is_librispeech_root(resolved) if kind == "librispeech" else is_esc50_root(resolved)
        if not valid:
            expected = "a LibriSpeech folder containing .flac files" if kind == "librispeech" else "an ESC-50 folder containing audio/ and meta/esc50.csv"
            raise SystemExit(f"The supplied {kind} path is not {expected}: {resolved}")
        return resolved

    names = ["LibriSpeech", "dev-clean-2", "dev-clean", "train-clean-100"] if kind == "librispeech" else ["ESC-50-master", "ESC50", "ESC-50"]
    candidates: list[Path] = []
    for base in (project_root, project_root / "src"):
        candidates.extend(base / name for name in names)
        # The user may have placed the extracted split itself inside src/.
        if kind == "librispeech":
            candidates.extend(base / name for name in ("dev-clean-2", "dev-clean", "train-clean-100"))
        else:
            candidates.append(base)
    # Support when this script is run from a directory that is itself a dataset root.
    candidates.append(project_root)
    for candidate in candidates:
        candidate = candidate.resolve()
        if kind == "librispeech" and is_librispeech_root(candidate):
            return candidate
        if kind == "esc50" and is_esc50_root(candidate):
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise SystemExit(
        f"Could not auto-detect {kind} dataset. Searched: {searched}\n"
        f"Pass its location explicitly with --{kind}-root \"FULL\\PATH\"."
    )


def make_writable(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
    except (OSError, FileNotFoundError):
        pass


def clear_generated_audio(folder: Path) -> int:
    """Remove staged audio only; also remove accidental directories named *.flac/*.wav."""
    removed = 0
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return removed

    # Deeper entries first so nested category folders can be removed afterward.
    entries = sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in entries:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            make_writable(path)
            try:
                path.unlink()
                removed += 1
            except PermissionError as exc:
                raise SystemExit(
                    f"Cannot delete generated file (it may be open in another app): {path}\n"
                    "Close audio players/editors and rerun this script."
                ) from exc
        elif path.is_dir() and path.suffix.lower() in AUDIO_EXTENSIONS:
            try:
                shutil.rmtree(path)
                removed += 1
            except PermissionError as exc:
                raise SystemExit(f"Cannot remove accidental directory named like an audio file: {path}") from exc

    for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    folder.mkdir(parents=True, exist_ok=True)
    return removed


def copy_flat_with_unique_names(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for src in files:
        name = src.name
        if name.lower() in used:
            name = f"{src.parent.parent.name}_{src.parent.name}_{src.name}"
        target = destination / name
        if name.lower() in used:
            raise SystemExit(f"Filename collision while preparing output: {src}")
        # A previous interrupted/manual attempt may have left a directory at this filename.
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            make_writable(target)
            target.unlink()
        try:
            shutil.copy2(src, target)
        except PermissionError as exc:
            raise SystemExit(
                f"Permission denied while copying:\n  source: {src}\n  target: {target}\n"
                "Close apps using files in the project data folder. If this is a stale generated split, "
                "rerun this script with --reset."
            ) from exc
        used.add(name.lower())


def split_clean_files(files: list[Path], ratio: float, rng: random.Random):
    if len(files) < 2:
        raise SystemExit("Need at least two clean speech files to create training/evaluation splits.")
    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        # Standard LibriSpeech layout: <split>/<speaker>/<chapter>/<utterance.flac>
        if len(path.parents) >= 2:
            speaker = path.parent.parent.name
        else:
            speaker = path.parent.name
        by_speaker[speaker].append(path)

    if len(by_speaker) >= 2:
        speakers = list(by_speaker)
        rng.shuffle(speakers)
        target = round(len(files) * ratio)
        train_speakers: list[str] = []
        eval_speakers: list[str] = []
        train_count = 0
        for i, speaker in enumerate(speakers):
            count = len(by_speaker[speaker])
            remaining = len(speakers) - i
            if not train_speakers or (remaining > 1 and abs(target - (train_count + count)) <= abs(target - train_count)):
                train_speakers.append(speaker)
                train_count += count
            else:
                eval_speakers.append(speaker)
        if not eval_speakers:
            eval_speakers.append(train_speakers.pop())
        train = sorted(p for speaker in train_speakers for p in by_speaker[speaker])
        evaluation = sorted(p for speaker in eval_speakers for p in by_speaker[speaker])
        return train, evaluation, f"speaker-level split ({len(train_speakers)} train speakers, {len(eval_speakers)} eval speakers)"

    shuffled = files.copy()
    rng.shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, round(len(shuffled) * ratio)))
    return sorted(shuffled[:cut]), sorted(shuffled[cut:]), "utterance-level fallback (only one speaker/chapter group found)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech + ESC-50 into the project's train/evaluation folders.")
    parser.add_argument("--librispeech-root", type=Path, default=None, help="Dataset path; auto-detected in project root/src if omitted")
    parser.add_argument("--esc50-root", type=Path, default=None, help="Dataset path; auto-detected in project root/src if omitted")
    parser.add_argument("--data-root", type=Path, default=None, help="Output data folder (defaults to <project-root>/data)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--reset", action="store_true", help="Clear staged audio under data/clean, eval_clean, noise, eval_noise before preparing a fresh split")
    args = parser.parse_args()

    if not 0.5 <= args.train_ratio < 1.0:
        parser.error("--train-ratio must be >= 0.5 and < 1.0")

    project_root = Path(__file__).resolve().parent
    data_root = (args.data_root or (project_root / "data")).expanduser().resolve()
    libri_root = detect_root(args.librispeech_root, project_root, "librispeech")
    esc_root = detect_root(args.esc50_root, project_root, "esc50")

    clean_files = list_audio(libri_root)
    if len(clean_files) < 2:
        raise SystemExit(f"Found only {len(clean_files)} audio files under {libri_root}; expected at least 2.")
    with (esc_root / "meta" / "esc50.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"filename", "category"}.issubset(rows[0].keys()):
        raise SystemExit(f"Unexpected metadata format in {esc_root / 'meta' / 'esc50.csv'}; expected filename and category columns.")

    requested = set(args.categories)
    available = {row["category"] for row in rows}
    missing = sorted(requested - available)
    if missing:
        raise SystemExit(f"Unknown ESC-50 categories: {', '.join(missing)}")

    audio_by_category: dict[str, list[Path]] = defaultdict(list)
    for row in rows:
        if row["category"] in requested:
            candidate = esc_root / "audio" / row["filename"]
            if candidate.is_file():
                audio_by_category[row["category"]].append(candidate)
    missing_audio = sorted(category for category in requested if len(audio_by_category.get(category, [])) < 2)
    if missing_audio:
        raise SystemExit(f"Not enough audio clips for categories: {', '.join(missing_audio)}")

    output_dirs = {name: data_root / name for name in OUTPUT_NAMES}
    existing = {name: list_audio(path) for name, path in output_dirs.items()}
    nonempty = [f"{name} ({len(paths)} files)" for name, paths in existing.items() if paths]
    # Even if a previous copy stopped midway, require an explicit reset before replacing files.
    if nonempty and not args.reset:
        raise SystemExit(
            "Staged audio already exists: " + ", ".join(nonempty) + ".\n"
            "Run again with --reset to replace only staged audio under data/clean, data/eval_clean, data/noise, data/eval_noise."
        )

    if args.reset:
        removed = sum(clear_generated_audio(path) for path in output_dirs.values())
        # Ensure known output folders exist even when empty or newly created.
        for path in output_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        print(f"Reset staged output: removed {removed} audio item(s) from {data_root}")
    else:
        for path in output_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    print(f"LibriSpeech source: {libri_root}")
    print(f"ESC-50 source:      {esc_root}")
    print(f"Project data root:  {data_root}")

    rng = random.Random(args.seed)
    clean_train, clean_eval, split_note = split_clean_files(clean_files, args.train_ratio, rng)
    copy_flat_with_unique_names(clean_train, output_dirs["clean"])
    copy_flat_with_unique_names(clean_eval, output_dirs["eval_clean"])
    print("\nClean speech split:")
    print(f"  Total: {len(clean_files)} | train: {len(clean_train)} | eval: {len(clean_eval)}")
    print(f"  Method: {split_note}")

    print("\nESC-50 noise split:")
    train_counts: Counter[str] = Counter()
    eval_counts: Counter[str] = Counter()
    for category in sorted(requested):
        clips = sorted(audio_by_category[category])
        rng.shuffle(clips)
        n_train = max(1, min(len(clips) - 1, round(len(clips) * args.train_ratio)))
        train_clips, eval_clips = clips[:n_train], clips[n_train:]
        copy_flat_with_unique_names(train_clips, output_dirs["noise"] / category)
        copy_flat_with_unique_names(eval_clips, output_dirs["eval_noise"] / category)
        train_counts[category] = len(train_clips)
        eval_counts[category] = len(eval_clips)
        print(f"  {category:20s} train={len(train_clips):2d} | eval={len(eval_clips):2d}")

    print("\nDataset preparation complete!")
    print(f"  Clean train: {len(clean_train)} -> {output_dirs['clean']}")
    print(f"  Clean eval:  {len(clean_eval)} -> {output_dirs['eval_clean']}")
    print(f"  Noise train: {sum(train_counts.values())} -> {output_dirs['noise']}")
    print(f"  Noise eval:  {sum(eval_counts.values())} -> {output_dirs['eval_noise']}")
    print("Original LibriSpeech and ESC-50 files were not moved or changed.")
    print("Next command: $env:PYTHONPATH = \"$PWD\\src\"; python -m noise_suppression.train_agent --episodes 100")


if __name__ == "__main__":
    main()
