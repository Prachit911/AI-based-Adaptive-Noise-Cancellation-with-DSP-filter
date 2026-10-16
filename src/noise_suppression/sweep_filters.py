"""Run every filter on every training mixture (full-information feedback).

This is the core data-generation step: instead of sampling one action per
episode (like Q-learning), we run ALL actions on ALL clips and record the
SI-SDR improvement.  The result is a CSV that the supervised selector
trains on directly.

Usage:
    python -m noise_suppression.sweep_filters \
        --manifest data/manifest.csv \
        --processed-dir data/processed \
        --output data/sweep_results.csv \
        --splits train val \
        --n-jobs 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .audio_utils import load_and_resample, si_sdr, level_match
from .constants import INTERNAL_SR
from .features import FEATURE_NAMES, N_FEATURES, extract_features
from .filters import ACTIONS, apply_filter
from .metrics import compute_stoi


def _process_one_mixture(
    mix_id: str,
    split: str,
    noise_category: str,
    snr_db: float,
    mixtures_dir: Path,
    refs_dir: Path,
) -> list[dict]:
    """Run all filters on one mixture and return a list of result rows."""
    mix_path = mixtures_dir / split / f"{mix_id}.wav"
    ref_path = refs_dir / split / f"{mix_id}_ref.wav"

    if not mix_path.exists() or not ref_path.exists():
        return []

    mixture = load_and_resample(mix_path, INTERNAL_SR)
    clean = load_and_resample(ref_path, INTERNAL_SR)

    # Ensure same length
    n = min(len(mixture), len(clean))
    mixture = mixture[:n]
    clean = clean[:n]

    # Extract features once
    features = extract_features(mixture, INTERNAL_SR)

    # Baseline SI-SDR
    baseline_si_sdr = si_sdr(clean, mixture)

    # Compute STOI of the input
    stoi_baseline = compute_stoi(clean, mixture, INTERNAL_SR)

    rows = []
    for action in ACTIONS:
        enhanced = apply_filter(action, mixture, INTERNAL_SR)
        enhanced = enhanced[:n]  # ensure length match

        # Level-match for fair metric
        if action != "keep_original":
            enhanced = level_match(mixture, enhanced)

        si_sdr_out = si_sdr(clean, enhanced)
        si_sdr_imp = si_sdr_out - baseline_si_sdr

        stoi_out = compute_stoi(clean, enhanced, INTERNAL_SR)
        stoi_imp = stoi_out - stoi_baseline if not np.isnan(stoi_out) else float("nan")

        row = {
            "mixture_id": mix_id,
            "split": split,
            "noise_category": noise_category,
            "snr_db": snr_db,
            "action": action,
            "si_sdr_improvement_db": round(float(si_sdr_imp), 4),
            "stoi_improvement": round(float(stoi_imp), 6) if not np.isnan(stoi_imp) else float("nan"),
            "si_sdr_output_db": round(float(si_sdr_out), 4),
        }
        # Add features
        for i, fname in enumerate(FEATURE_NAMES):
            row[f"feat_{fname}"] = round(float(features[i]), 6)

        rows.append(row)

    return rows


def run_sweep(
    manifest: pd.DataFrame,
    processed_dir: Path,
    splits: list[str],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the full filter sweep across specified splits."""
    mixtures_dir = processed_dir / "mixtures"
    refs_dir = processed_dir / "mixtures" / "references"

    # Filter to requested splits
    df = manifest[manifest["split"].isin(splits)].copy()
    print(f"Sweeping {len(df)} mixtures across {len(ACTIONS)} actions "
          f"= {len(df) * len(ACTIONS)} evaluations")

    if n_jobs == 1:
        # Sequential (easier to debug)
        all_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Filter sweep"):
            result = _process_one_mixture(
                row["mixture_id"],
                row["split"],
                row["noise_category"],
                row["snr_db"],
                mixtures_dir,
                refs_dir,
            )
            all_rows.extend(result)
    else:
        # Parallel
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_process_one_mixture)(
                row["mixture_id"],
                row["split"],
                row["noise_category"],
                row["snr_db"],
                mixtures_dir,
                refs_dir,
            )
            for _, row in df.iterrows()
        )
        all_rows = [r for batch in results for r in batch]

    return pd.DataFrame(all_rows)


def compute_oracle(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the oracle (best filter per clip)."""
    idx = sweep_df.groupby("mixture_id")["si_sdr_improvement_db"].idxmax()
    oracle = sweep_df.loc[idx].copy()
    oracle["method"] = "oracle"
    return oracle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all filters on all training mixtures."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/sweep_results.csv"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel workers (1=sequential)")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    results_df = run_sweep(manifest, args.processed_dir, args.splits, args.n_jobs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nSweep results saved to: {args.output} ({len(results_df)} rows)")

    # Print summary per action
    print("\n-- Per-action mean SI-SDR improvement (dB) --")
    summary = results_df.groupby("action")["si_sdr_improvement_db"].mean()
    for action in ACTIONS:
        if action in summary.index:
            print(f"  {action:28s}: {summary[action]:+7.3f} dB")

    # Oracle
    oracle_df = compute_oracle(results_df)
    oracle_mean = oracle_df["si_sdr_improvement_db"].mean()
    print(f"\n  {'oracle':28s}: {oracle_mean:+7.3f} dB")
    best_fixed = summary.max()
    print(f"  {'best fixed filter':28s}: {best_fixed:+7.3f} dB")
    if oracle_mean > best_fixed:
        gap = oracle_mean - best_fixed
        print(f"  Oracle advantage over best fixed: {gap:+.3f} dB")


if __name__ == "__main__":
    main()
