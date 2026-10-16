"""Full evaluation of speech noise suppression across methods, SNR buckets, and noise categories.

Evaluates:
  1. keep_original (unprocessed baseline)
  2. Each individual fixed DSP filter in ACTIONS
  3. Best fixed filter (highest mean Δ SI-SDR across test set)
  4. Q-learning agent (tabular baseline)
  5. Supervised filter selector (GBRT selector)
  6. Oracle (clip-level upper bound: best filter per clip)

Metrics computed per clip:
  - SI-SDR improvement (dB)
  - STOI improvement
  - PESQ-WB improvement (if pesq is installed)
  - 95% bootstrap confidence intervals for SI-SDR gain

Outputs:
  - outputs/evaluation_results.csv (per-clip metrics for all methods)
  - outputs/evaluation_summary.csv (aggregated results with CIs)
  - outputs/evaluation_report.md (formatted Markdown report with tables)

Usage:
    python -m noise_suppression.evaluate \
        --manifest data/manifest.csv \
        --processed-dir data/processed \
        --selector models/selector.joblib \
        --q-table models/q_table.json \
        --output-dir outputs/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .agent import QLearningFilterSelector
from .audio_utils import load_and_resample, level_match, mix_at_snr, read_audio, resample_audio
from .constants import INTERNAL_SR
from .features import extract_features, extract_state
from .filters import ACTIONS, apply_filter
from .metrics import compute_all_metrics
from .selector import SupervisedFilterSelector


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return low, high


def evaluate_manifest_split(
    manifest_path: Path,
    processed_dir: Path,
    selector_path: Path | None = None,
    q_table_path: Path | None = None,
    split: str = "test",
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Evaluate all methods on the given manifest split."""
    manifest = pd.read_csv(manifest_path)
    split_df = manifest[manifest["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No samples found for split '{split}' in {manifest_path}")

    if max_samples and max_samples > 0:
        split_df = split_df.head(max_samples)

    # Load models
    selector = None
    if selector_path and Path(selector_path).exists():
        print(f"Loaded supervised selector from {selector_path}")
        selector = SupervisedFilterSelector.load(selector_path)

    agent = None
    if q_table_path and Path(q_table_path).exists():
        print(f"Loaded Q-learning agent from {q_table_path}")
        agent = QLearningFilterSelector.load(q_table_path)

    mixtures_dir = processed_dir / "mixtures" / split
    refs_dir = processed_dir / "mixtures" / "references" / split

    records = []
    print(f"Evaluating {len(split_df)} test mixtures ...")

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Evaluating {split}"):
        mix_id = str(row["mixture_id"])
        mix_path = mixtures_dir / f"{mix_id}.wav"
        ref_path = refs_dir / f"{mix_id}_ref.wav"

        if not mix_path.exists() or not ref_path.exists():
            continue

        noisy = load_and_resample(mix_path, INTERNAL_SR)
        clean = load_and_resample(ref_path, INTERNAL_SR)

        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]

        features = extract_features(noisy, INTERNAL_SR)
        discrete_state = extract_state(noisy, INTERNAL_SR)

        # Store results for each action to determine oracle later
        action_metrics: dict[str, dict[str, float]] = {}
        for action in ACTIONS:
            enhanced = apply_filter(action, noisy, INTERNAL_SR)[:n]
            m = compute_all_metrics(clean, noisy, enhanced, INTERNAL_SR)
            action_metrics[action] = m
            records.append({
                "mixture_id": mix_id,
                "split": split,
                "noise_category": row.get("noise_category", "unknown"),
                "snr_db": row.get("snr_db", 0),
                "method": action,
                "chosen_action": action,
                "si_sdr_improvement_db": m["si_sdr_improvement_db"],
                "si_sdr_output_db": m["si_sdr_output_db"],
                "stoi_output": m["stoi_output"],
                "stoi_improvement": m["stoi_improvement"],
                "pesq_output": m["pesq_output"],
                "pesq_improvement": m["pesq_improvement"],
            })

        # Oracle (best action for this specific clip based on SI-SDR improvement)
        best_act = max(action_metrics, key=lambda a: action_metrics[a]["si_sdr_improvement_db"])
        best_m = action_metrics[best_act]
        records.append({
            "mixture_id": mix_id,
            "split": split,
            "noise_category": row.get("noise_category", "unknown"),
            "snr_db": row.get("snr_db", 0),
            "method": "oracle",
            "chosen_action": best_act,
            "si_sdr_improvement_db": best_m["si_sdr_improvement_db"],
            "si_sdr_output_db": best_m["si_sdr_output_db"],
            "stoi_output": best_m["stoi_output"],
            "stoi_improvement": best_m["stoi_improvement"],
            "pesq_output": best_m["pesq_output"],
            "pesq_improvement": best_m["pesq_improvement"],
        })

        # Q-learning agent
        if agent is not None:
            q_act = agent.select(discrete_state)
            q_m = action_metrics[q_act]
            records.append({
                "mixture_id": mix_id,
                "split": split,
                "noise_category": row.get("noise_category", "unknown"),
                "snr_db": row.get("snr_db", 0),
                "method": "q_learning_agent",
                "chosen_action": q_act,
                "si_sdr_improvement_db": q_m["si_sdr_improvement_db"],
                "si_sdr_output_db": q_m["si_sdr_output_db"],
                "stoi_output": q_m["stoi_output"],
                "stoi_improvement": q_m["stoi_improvement"],
                "pesq_output": q_m["pesq_output"],
                "pesq_improvement": q_m["pesq_improvement"],
            })

        # Supervised selector
        if selector is not None:
            sel_act = selector.select(features)
            sel_m = action_metrics.get(sel_act, action_metrics["keep_original"])
            records.append({
                "mixture_id": mix_id,
                "split": split,
                "noise_category": row.get("noise_category", "unknown"),
                "snr_db": row.get("snr_db", 0),
                "method": "supervised_selector",
                "chosen_action": sel_act,
                "si_sdr_improvement_db": sel_m["si_sdr_improvement_db"],
                "si_sdr_output_db": sel_m["si_sdr_output_db"],
                "stoi_output": sel_m["stoi_output"],
                "stoi_improvement": sel_m["stoi_improvement"],
                "pesq_output": sel_m["pesq_output"],
                "pesq_improvement": sel_m["pesq_improvement"],
            })

    return pd.DataFrame(records)


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-method statistics and compute 95% bootstrap CIs."""
    methods = sorted(df["method"].unique(), key=lambda m: (
        0 if m == "keep_original" else
        999 if m == "oracle" else
        998 if m == "supervised_selector" else
        997 if m == "q_learning_agent" else 1,
        m
    ))

    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        sisdr = sub["si_sdr_improvement_db"].to_numpy()
        ci_low, ci_high = bootstrap_ci(sisdr)

        stoi = sub["stoi_improvement"].to_numpy()
        stoi_valid = stoi[np.isfinite(stoi)]

        pesq = sub["pesq_improvement"].to_numpy()
        pesq_valid = pesq[np.isfinite(pesq)]

        rows.append({
            "method": m,
            "mean_delta_si_sdr_db": float(np.mean(sisdr)),
            "std_delta_si_sdr_db": float(np.std(sisdr)),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "mean_delta_stoi": float(np.mean(stoi_valid)) if len(stoi_valid) > 0 else float("nan"),
            "mean_delta_pesq": float(np.mean(pesq_valid)) if len(pesq_valid) > 0 else float("nan"),
            "n_samples": len(sub),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df


def generate_markdown_report(
    summary_df: pd.DataFrame,
    df: pd.DataFrame,
    output_path: Path,
) -> str:
    """Generate comprehensive markdown report."""
    # Find best fixed filter
    fixed_mask = ~summary_df["method"].isin(["keep_original", "oracle", "supervised_selector", "q_learning_agent"])
    fixed_df = summary_df[fixed_mask]
    best_fixed_row = fixed_df.loc[fixed_df["mean_delta_si_sdr_db"].idxmax()] if not fixed_df.empty else None

    # Selector row
    sel_rows = summary_df[summary_df["method"] == "supervised_selector"]
    sel_row = sel_rows.iloc[0] if not sel_rows.empty else None

    # Oracle row
    oracle_rows = summary_df[summary_df["method"] == "oracle"]
    oracle_row = oracle_rows.iloc[0] if not oracle_rows.empty else None

    md = []
    md.append("# Speech Noise Suppression — Evaluation Report\n")
    md.append(f"**Total test mixtures evaluated:** {len(df['mixture_id'].unique())}\n")

    # Overall Summary Table
    md.append("## Overall Method Comparison\n")
    md.append("| Method | Δ SI-SDR (dB) | 95% Bootstrap CI | Δ STOI | Δ PESQ | N |")
    md.append("|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        ci_str = f"[{r['ci_95_low']:+.2f}, {r['ci_95_high']:+.2f}]" if not np.isnan(r['ci_95_low']) else "N/A"
        stoi_str = f"{r['mean_delta_stoi']:+.4f}" if not np.isnan(r['mean_delta_stoi']) else "N/A"
        pesq_str = f"{r['mean_delta_pesq']:+.3f}" if not np.isnan(r['mean_delta_pesq']) else "N/A"
        md.append(f"| **{r['method']}** | {r['mean_delta_si_sdr_db']:+7.3f} | {ci_str} | {stoi_str} | {pesq_str} | {int(r['n_samples'])} |")
    md.append("")

    # Key Findings / Head-to-Head
    if sel_row is not None and best_fixed_row is not None and oracle_row is not None:
        sel_gain = sel_row["mean_delta_si_sdr_db"]
        fixed_gain = best_fixed_row["mean_delta_si_sdr_db"]
        oracle_gain = oracle_row["mean_delta_si_sdr_db"]
        advantage = sel_gain - fixed_gain
        gap_fraction = (advantage / (oracle_gain - fixed_gain + 1e-9)) * 100 if oracle_gain > fixed_gain else 0.0

        md.append("## Key Insights & Head-to-Head Analysis\n")
        md.append(f"- **Supervised Selector vs Best Fixed Filter (`{best_fixed_row['method']}`):**")
        md.append(f"  - Selector: `{sel_gain:+7.3f} dB` vs Best Fixed: `{fixed_gain:+7.3f} dB`")
        md.append(f"  - **Advantage:** `{advantage:+7.3f} dB`")
        md.append(f"- **Oracle Ceiling:** `{oracle_gain:+7.3f} dB`")
        md.append(f"- **Oracle Gap Recovered:** `{gap_fraction:.1f}%`\n")

    # Breakdown by SNR
    md.append("## Breakdown by Input SNR Bucket\n")
    snr_piv = df.pivot_table(index="snr_db", columns="method", values="si_sdr_improvement_db", aggfunc="mean")
    cols_to_show = [c for c in ["keep_original", "wiener", "spectral_subtraction", "q_learning_agent", "supervised_selector", "oracle"] if c in snr_piv.columns]
    md.append("| SNR (dB) | " + " | ".join(cols_to_show) + " |")
    md.append("|" + "---|" * (len(cols_to_show) + 1))
    for snr, row in snr_piv.iterrows():
        vals = " | ".join(f"{row[c]:+.2f}" if c in row and not np.isnan(row[c]) else "N/A" for c in cols_to_show)
        md.append(f"| **{snr:+.0f} dB** | {vals} |")
    md.append("")

    # Breakdown by Noise Category
    md.append("## Breakdown by Noise Category\n")
    cat_piv = df.pivot_table(index="noise_category", columns="method", values="si_sdr_improvement_db", aggfunc="mean")
    md.append("| Category | " + " | ".join(cols_to_show) + " |")
    md.append("|" + "---|" * (len(cols_to_show) + 1))
    for cat, row in cat_piv.iterrows():
        vals = " | ".join(f"{row[c]:+.2f}" if c in row and not np.isnan(row[c]) else "N/A" for c in cols_to_show)
        md.append(f"| **{cat}** | {vals} |")
    md.append("")

    report_str = "\n".join(md)
    output_path.write_text(report_str, encoding="utf-8")
    return report_str


def evaluate_directory_pairs(
    clean_dir: Path,
    noise_dir: Path,
    selector_path: Path | None = None,
    q_table_path: Path | None = None,
    n_samples: int = 30,
    seed: int = 7,
) -> pd.DataFrame:
    """Evaluate methods on synthetic mixtures created on-the-fly from clean and noise folders."""
    import random
    from .audio_utils import read_audio, resample_audio, mix_at_snr

    audio_exts = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    clean_files = sorted(p for p in clean_dir.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts)
    noise_files = sorted(p for p in noise_dir.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts)

    if not clean_files:
        raise FileNotFoundError(f"No clean audio found in {clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No noise audio found in {noise_dir}")

    selector = None
    if selector_path and Path(selector_path).exists():
        print(f"Loaded supervised selector from {selector_path}")
        selector = SupervisedFilterSelector.load(selector_path)

    agent = None
    if q_table_path and Path(q_table_path).exists():
        print(f"Loaded Q-learning agent from {q_table_path}")
        agent = QLearningFilterSelector.load(q_table_path)

    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)
    records = []

    print(f"Evaluating {n_samples} mixtures from {clean_dir} and {noise_dir} ...")
    for idx in tqdm(range(n_samples), desc="Evaluating folders"):
        clean_path = py_rng.choice(clean_files)
        noise_path = py_rng.choice(noise_files)

        clean, sr = read_audio(clean_path)
        noise, noise_sr = read_audio(noise_path)
        clean = resample_audio(clean, sr, INTERNAL_SR)
        noise = resample_audio(noise, noise_sr, INTERNAL_SR)

        if clean.size > INTERNAL_SR * 12:
            start = int(rng.integers(0, clean.size - INTERNAL_SR * 12 + 1))
            clean = clean[start:start + INTERNAL_SR * 12]

        snr_db = float(py_rng.choice([-5.0, 0.0, 5.0, 10.0, 15.0]))
        noisy = mix_at_snr(clean, noise, snr_db, rng)

        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]

        features = extract_features(noisy, INTERNAL_SR)
        discrete_state = extract_state(noisy, INTERNAL_SR)
        mix_id = f"sample_{idx + 1:04d}"
        noise_cat = noise_path.stem

        action_metrics: dict[str, dict[str, float]] = {}
        for action in ACTIONS:
            enhanced = apply_filter(action, noisy, INTERNAL_SR)[:n]
            m = compute_all_metrics(clean, noisy, enhanced, INTERNAL_SR)
            action_metrics[action] = m
            records.append({
                "mixture_id": mix_id,
                "split": "eval",
                "noise_category": noise_cat,
                "snr_db": snr_db,
                "method": action,
                "chosen_action": action,
                "si_sdr_improvement_db": m["si_sdr_improvement_db"],
                "si_sdr_output_db": m["si_sdr_output_db"],
                "stoi_output": m["stoi_output"],
                "stoi_improvement": m["stoi_improvement"],
                "pesq_output": m["pesq_output"],
                "pesq_improvement": m["pesq_improvement"],
            })

        # Oracle
        best_act = max(action_metrics, key=lambda a: action_metrics[a]["si_sdr_improvement_db"])
        best_m = action_metrics[best_act]
        records.append({
            "mixture_id": mix_id,
            "split": "eval",
            "noise_category": noise_cat,
            "snr_db": snr_db,
            "method": "oracle",
            "chosen_action": best_act,
            "si_sdr_improvement_db": best_m["si_sdr_improvement_db"],
            "si_sdr_output_db": best_m["si_sdr_output_db"],
            "stoi_output": best_m["stoi_output"],
            "stoi_improvement": best_m["stoi_improvement"],
            "pesq_output": best_m["pesq_output"],
            "pesq_improvement": best_m["pesq_improvement"],
        })

        # Q-learning agent
        if agent is not None:
            q_act = agent.select(discrete_state)
            q_m = action_metrics[q_act]
            records.append({
                "mixture_id": mix_id,
                "split": "eval",
                "noise_category": noise_cat,
                "snr_db": snr_db,
                "method": "q_learning_agent",
                "chosen_action": q_act,
                "si_sdr_improvement_db": q_m["si_sdr_improvement_db"],
                "si_sdr_output_db": q_m["si_sdr_output_db"],
                "stoi_output": q_m["stoi_output"],
                "stoi_improvement": q_m["stoi_improvement"],
                "pesq_output": q_m["pesq_output"],
                "pesq_improvement": q_m["pesq_improvement"],
            })

        # Supervised selector
        if selector is not None:
            sel_act = selector.select(features)
            sel_m = action_metrics.get(sel_act, action_metrics["keep_original"])
            records.append({
                "mixture_id": mix_id,
                "split": "eval",
                "noise_category": noise_cat,
                "snr_db": snr_db,
                "method": "supervised_selector",
                "chosen_action": sel_act,
                "si_sdr_improvement_db": sel_m["si_sdr_improvement_db"],
                "si_sdr_output_db": sel_m["si_sdr_output_db"],
                "stoi_output": sel_m["stoi_output"],
                "stoi_improvement": sel_m["stoi_improvement"],
                "pesq_output": sel_m["pesq_output"],
                "pesq_improvement": sel_m["pesq_improvement"],
            })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate noise suppression methods on held-out data.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/eval_clean"), help="Held-out clean speech folder")
    parser.add_argument("--noise-dir", type=Path, default=Path("data/eval_noise"), help="Held-out noise folder")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest CSV path")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--selector", type=Path, default=Path("models/selector.joblib"))
    parser.add_argument("--q-table", type=Path, default=Path("models/q_table.json"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=None, help="Output evaluation CSV")
    parser.add_argument("--n", "--max-samples", dest="max_samples", type=int, default=30,
                        help="Number of mixtures to evaluate")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine evaluation mode: manifest vs folders
    use_manifest = False
    if args.manifest is not None and args.manifest.exists():
        use_manifest = True
    elif args.manifest is None and Path("data/manifest.csv").exists():
        # Only use manifest if explicitly present or if eval_clean is not populated
        if not (args.clean_dir.exists() and any(args.clean_dir.glob("*.wav"))):
            use_manifest = True
            args.manifest = Path("data/manifest.csv")

    if use_manifest and args.manifest is not None:
        print(f"Running evaluation from manifest: {args.manifest} (split: {args.split})")
        results_df = evaluate_manifest_split(
            manifest_path=args.manifest,
            processed_dir=args.processed_dir,
            selector_path=args.selector,
            q_table_path=args.q_table,
            split=args.split,
            max_samples=args.max_samples,
        )
    else:
        # Fall back to data/clean and data/noise if eval folders don't have audio
        clean_dir = args.clean_dir
        noise_dir = args.noise_dir
        if not (clean_dir.exists() and any(clean_dir.glob("*"))):
            clean_dir = Path("data/clean")
        if not (noise_dir.exists() and any(noise_dir.glob("*"))):
            noise_dir = Path("data/noise")

        results_df = evaluate_directory_pairs(
            clean_dir=clean_dir,
            noise_dir=noise_dir,
            selector_path=args.selector,
            q_table_path=args.q_table,
            n_samples=args.max_samples or 30,
            seed=args.seed,
        )

    results_csv = args.output or (args.output_dir / "evaluation.csv")
    results_df.to_csv(results_csv, index=False)
    # Also write evaluation_results.csv for full backwards compatibility
    results_df.to_csv(args.output_dir / "evaluation_results.csv", index=False)
    print(f"\nPer-sample evaluation results saved to: {results_csv}")

    summary_df = compute_summary_table(results_df)
    summary_csv = args.output_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary table saved to: {summary_csv}")

    report_md = args.output_dir / "evaluation_report.md"
    report_text = generate_markdown_report(summary_df, results_df, report_md)
    print(f"Full markdown evaluation report saved to: {report_md}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Mean Delta SI-SDR dB)")
    print("=" * 60)
    for _, r in summary_df.iterrows():
        print(f"  {r['method']:26s}: {r['mean_delta_si_sdr_db']:+7.3f} dB  "
              f"[{r['ci_95_low']:+.2f}, {r['ci_95_high']:+.2f}]")


if __name__ == "__main__":
    main()
